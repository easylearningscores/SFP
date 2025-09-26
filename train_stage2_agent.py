import os
import yaml
import argparse
import logging
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from models.gwm import GenerativeWorldModel
from models.agent import Agent
from data.dataloader import SFPDataset
from planning.planner import BeamSearchPlanner
from utils.rewards import get_reward_function

def setup_distributed():
    """Initialize the DDP environment"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def setup_logging(log_path, rank):
    """Set up logging, only outputting to a file on the main process (rank 0)"""
    if rank == 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(log_path),
                                logging.StreamHandler()
                            ])
    else: # Other processes only log errors
        logging.basicConfig(level=logging.ERROR)

def reduce_tensor(tensor, world_size):
    """Aggregate tensor values from all GPUs"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def main(config):
    """Main training function"""
    # --- DDP Setup ---
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    setup_logging(config['log_path'], local_rank)
    
    if local_rank == 0:
        logging.info("--- Starting SFP Framework Stage 2 (DDP Mode): Agent Training ---")
        logging.info(f"Using {world_size} GPUs.")

    # --- 1. Data Loading ---
    train_dataset = SFPDataset(config['data']['train_path'], mode='stage2')
    val_dataset = SFPDataset(config['data']['val_path'], mode='stage2')
    
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], sampler=val_sampler, num_workers=4, pin_memory=True)
    if local_rank == 0:
        logging.info("Distributed data loaders created.")

    s_t_sample, _ = next(iter(train_loader))
    shape_in = s_t_sample.shape[1:]

    # --- 2. Initialize Models and Planner ---
    # a. Load and freeze GWM
    gwm_config = config['models']['gwm']
    gwm = GenerativeWorldModel(...).to(device) # Parameters omitted
    # Load GWM weights only on rank 0 and then broadcast to other processes to ensure consistency
    if local_rank == 0:
        gwm.load_state_dict(torch.load(config['models']['gwm']['checkpoint_path'], map_location='cpu'))
    
    # Broadcast model weights from rank 0 to all other processes
    for param in gwm.parameters():
        dist.broadcast(param.data, src=0)
    
    gwm.eval()
    for param in gwm.parameters():
        param.requires_grad = False
    if local_rank == 0:
        logging.info(f"GWM loaded from {config['models']['gwm']['checkpoint_path']}, broadcast, and frozen.")

    # b. Initialize Agent and wrap with DDP
    agent_config = config['models']['agent']
    agent = Agent(...).to(device) # Parameters omitted
    agent = DDP(agent, device_ids=[local_rank], find_unused_parameters=False)
    if local_rank == 0:
        logging.info(f"Agent initialized (backbone: {agent_config['backbone_name']}) and wrapped with DDP.")

    # c. Initialize Planner
    planner_config = config['planning']
    reward_fn = get_reward_function(planner_config['reward_function'])
    planner = BeamSearchPlanner(gwm, reward_fn, beam_width=planner_config['beam_width'])
    if local_rank == 0:
        logging.info(f"Planner initialized (reward function: {planner_config['reward_function']}).")

    # --- 3. Initialize Optimizer and Loss Function ---
    optimizer = optim.Adam(agent.parameters(), lr=config['training']['lr'])
    policy_criterion = torch.nn.MSELoss()
    
    # --- 4. Training Loop ---
    best_val_loss = torch.tensor(float('inf')).to(device)
    for epoch in range(config['training']['epochs']):
        agent.train()
        train_sampler.set_epoch(epoch) # Ensure different shuffling each epoch
        train_loss_epoch = torch.tensor(0.0).to(device)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=(local_rank != 0))
        for s_t, y_gt in progress_bar:
            s_t, y_gt = s_t.to(device), y_gt.to(device)
            
            # 1. Agent generates an action (note the call to agent.module)
            action = agent.module.get_action(s_t)
            
            # 2. Planner finds the pseudo-label
            pseudo_label = planner.plan(s_t, action, y_gt)
            
            # 3. Update the agent
            optimizer.zero_grad()
            predicted_state = agent(s_t) # Can be called directly after being wrapped by DDP
            policy_loss = policy_criterion(predicted_state, pseudo_label.detach())
            policy_loss.backward()
            optimizer.step()
            
            train_loss_epoch += policy_loss
        
        # Aggregate training loss from all GPUs
        avg_train_loss_tensor = reduce_tensor(train_loss_epoch, world_size) / len(train_loader)
        
        if local_rank == 0:
            logging.info(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss_tensor.item():.6f}")

        # --- 5. Validation ---
        agent.eval()
        val_loss_epoch = torch.tensor(0.0).to(device)
        with torch.no_grad():
            for s_t, y_gt in val_loader:
                s_t, y_gt = s_t.to(device), y_gt.to(device)
                predicted_state = agent.module(s_t) # Use agent.module to get the original model during validation
                val_loss = policy_criterion(predicted_state, y_gt)
                val_loss_epoch += val_loss
        
        avg_val_loss_tensor = reduce_tensor(val_loss_epoch, world_size) / len(val_loader)
        
        if local_rank == 0:
            logging.info(f"Epoch {epoch+1} - Validation Loss (MSE): {avg_val_loss_tensor.item():.6f}")

        # --- 6. Save the Best Model (only on rank 0) ---
        if avg_val_loss_tensor < best_val_loss:
            best_val_loss = avg_val_loss_tensor
            if local_rank == 0:
                torch.save(agent.module.state_dict(), config['models']['agent']['save_path'])
                logging.info(f"New best model found, saved to {config['models']['agent']['save_path']}")
    
    # Wait for all processes to finish
    dist.destroy_process_group()
    if local_rank == 0:
        logging.info("--- Stage 2 Training Complete ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SFP Stage 2 DDP Training")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)