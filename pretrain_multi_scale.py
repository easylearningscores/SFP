import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
from tqdm import tqdm
from dataloader_rec import create_dataloaders, ClimateReconstructionDataset
from generative_world_model import Generative_World_Model

# Setup logging
backbone = 'beamvq_reconstruction_v1'
logging.basicConfig(filename=f'./logs/{backbone}_training_log.log', 
                    level=logging.INFO, 
                    format='%(asctime)s %(message)s')

# Set a specific seed
seed = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

# =========================================================================== dist train ========================================================================================================================
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
num_gpus = torch.cuda.device_count()

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

# ============================================================= data load ===================================================
DATA_DIR = "/jizhicfs/easyluwu/scaling_law/ft_local/low_res"

# 创建数据加载器
loaders = create_dataloaders(
    data_path=DATA_DIR,
    train_years=range(1980, 2019),
    test_years=range(2019, 2022),
    batch_size=3,
    variables=range(69) ) # 使用全部变量

train_loader = loaders['train']
test_loader = loaders['test']

# 分布式采样器
train_sampler = data.distributed.DistributedSampler(train_loader.dataset)
train_loader = data.DataLoader(train_loader.dataset,
                             batch_size=train_loader.batch_size,
                             sampler=train_sampler,
                             num_workers=4,
                             pin_memory=True)

test_sampler = data.distributed.DistributedSampler(test_loader.dataset)
test_loader = data.DataLoader(test_loader.dataset,
                            batch_size=test_loader.batch_size,
                            sampler=test_sampler,
                            num_workers=4,
                            pin_memory=True)

# 检查数据形状
for inputs, targets in iter(train_loader):
    print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
    break

# ================================================ model load ===========================================================
model = Generative_World_Model(
    in_channel=69,
    res_layers=2,
    embedding_nums=1024, 
    embedding_dim=256,
    top_k=10).to(device)

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

# ============================== criterion and optimizer ======================================================
criterion = nn.MSELoss()  
vq_weight = 0.1  

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)

# ===========================train val and test ======================================
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_recon_loss = 0.0
    train_vq_loss = 0.0
    
    for inputs, targets in tqdm(train_loader, desc="Training", disable=local_rank != 0):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        pred, top_k_features, vq_loss = model(inputs)
        
        recon_loss = criterion(pred, targets)
        
        loss = recon_loss + vq_weight * vq_loss
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        train_recon_loss += recon_loss.item() * inputs.size(0)
        train_vq_loss += vq_loss.item() * inputs.size(0)
    
    return {
        'total_loss': train_loss / len(train_loader.dataset),
        'recon_loss': train_recon_loss / len(train_loader.dataset),
        'vq_loss': train_vq_loss / len(train_loader.dataset)
    }

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_recon_loss = 0.0
    val_vq_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation", disable=local_rank != 0):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            pred, top_k_features, vq_loss = model(inputs)
            
            recon_loss = criterion(pred, targets)
            loss = recon_loss + vq_weight * vq_loss
            
            val_loss += loss.item() * inputs.size(0)
            val_recon_loss += recon_loss.item() * inputs.size(0)
            val_vq_loss += vq_loss.item() * inputs.size(0)
    
    return {
        'total_loss': val_loss / len(val_loader.dataset),
        'recon_loss': val_recon_loss / len(val_loader.dataset),
        'vq_loss': val_vq_loss / len(val_loader.dataset)
    }

def test(model, test_loader, criterion, device):
    path = './results'
    os.makedirs(path, exist_ok=True)
    
    model.eval()
    test_loss = 0.0
    test_recon_loss = 0.0
    test_vq_loss = 0.0
    
    all_inputs = []
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing", disable=local_rank != 0):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            pred, top_k_features, vq_loss = model(inputs)
            
            all_inputs.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(pred.cpu().numpy())
            
            recon_loss = criterion(pred, targets)
            loss = recon_loss + vq_weight * vq_loss
            
            test_loss += loss.item() * inputs.size(0)
            test_recon_loss += recon_loss.item() * inputs.size(0)
            test_vq_loss += vq_loss.item() * inputs.size(0)
    
    
    if local_rank == 0:
        all_inputs = np.concatenate(all_inputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
        
        np.save(f'{path}/{backbone}_inputs.npy', all_inputs)
        np.save(f'{path}/{backbone}_targets.npy', all_targets)
        np.save(f'{path}/{backbone}_outputs.npy', all_outputs)
    
    return {
        'total_loss': test_loss / len(test_loader.dataset),
        'recon_loss': test_recon_loss / len(test_loader.dataset),
        'vq_loss': test_vq_loss / len(test_loader.dataset)
    }

# ================================================ main training loop =====================================================
num_epochs = 1000
best_val_loss = float('inf')
best_model_path = f'./checkpoints/{backbone}_best_model.pth'
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

if local_rank == 0 and os.path.exists(best_model_path):
    try:
        logging.info('Loading best model from checkpoint.')
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint)
    except Exception as e:
        logging.error(f'Error loading model checkpoint: {e}')

for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch) 
    
    train_metrics = train(model, train_loader, criterion, optimizer, device)
    val_metrics = validate(model, test_loader, criterion, device)
    
    scheduler.step()
    
    if local_rank == 0:
        logging.info(f'Epoch {epoch + 1}/{num_epochs}')
        logging.info(f'Train Total Loss: {train_metrics["total_loss"] * num_gpus:.7f}')
        logging.info(f'Train Recon Loss: {train_metrics["recon_loss"] * num_gpus:.7f}')
        logging.info(f'Train VQ Loss: {train_metrics["vq_loss"] * num_gpus:.7f}')
        
        logging.info(f'Val Total Loss: {val_metrics["total_loss"] * num_gpus:.7f}')
        logging.info(f'Val Recon Loss: {val_metrics["recon_loss"] * num_gpus:.7f}')
        logging.info(f'Val VQ Loss: {val_metrics["vq_loss"] * num_gpus:.7f}')
        
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'New best model saved at epoch {epoch + 1}')

if local_rank == 0:
    try:
        model.load_state_dict(torch.load(best_model_path))
        test_metrics = test(model, test_loader, criterion, device)
        
        logging.info("Testing completed with best model:")
        logging.info(f'Test Total Loss: {test_metrics["total_loss"] * num_gpus:.7f}')
        logging.info(f'Test Recon Loss: {test_metrics["recon_loss"] * num_gpus:.7f}')
        logging.info(f'Test VQ Loss: {test_metrics["vq_loss"] * num_gpus:.7f}')
    except Exception as e:
        logging.error(f'Error during testing: {e}')

dist.destroy_process_group()