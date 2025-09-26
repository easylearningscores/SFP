import os
import torch
import netCDF4 as nc
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List

class ClimateReconstructionDataset(Dataset):
    """
    Climate Data Reconstruction Dataset (On-demand Loading Version)
    Features:
    1. Does not preload all data into memory; reads from disk on demand.
    2. Supports multi-variable selection and time step control.
    3. Specifically designed for reconstruction tasks (input = target).
    """
    
    def __init__(self, 
                 data_path: str,
                 years: List[int],
                 variables: List[int] = range(69),
                 time_steps: List[int] = None,
                 lead_time: int = 0):
        """
        Args:
            data_path: Path to the data directory.
            years: List of years to use.
            variables: List of variable indices to use (defaults to all 69 variables).
            time_steps: Optional, specifies which time steps to use (e.g., range(12, 357, 3)).
            lead_time: Time step interval (usually 0 for reconstruction tasks).
        """
        self.data_path = data_path
        self.years = years
        self.variables = variables
        self.lead_time = lead_time
        
        # Build a list of valid indices (year, time_idx)
        self.indices = []
        for year in years:
            file_path = os.path.join(data_path, f"{year}_norm.nc")
            with nc.Dataset(file_path) as ds:
                max_time = ds.variables['atmosphere_variables'].shape[0] - 1
            
            # If time_steps is not specified, use all valid time steps
            if time_steps is None:
                valid_steps = range(max_time + 1)
            else:
                valid_steps = [t for t in time_steps if t <= max_time]
            
            self.indices.extend([(year, t) for t in valid_steps])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        year, time_idx = self.indices[idx]
        file_path = os.path.join(self.data_path, f"{year}_norm.nc")
        
        with nc.Dataset(file_path) as ds:
            # Load the same data as input and target
            data = ds.variables['atmosphere_variables'][
                time_idx,        # Single time step
                self.variables,  # Selected variables
                :, :             # All latitudes and longitudes
            ]
        
        # Convert to tensor and handle NaNs
        data = torch.tensor(data, dtype=torch.float32)
        data = torch.nan_to_num(data, nan=0.0)
        
        return data, data.clone()  # Input and target are the same

def create_dataloaders(data_path: str,
                       train_years: List[int] = range(1980, 2019),
                       test_years: List[int] = range(2019, 2022),
                       batch_size: int = 32,
                       **dataset_kwargs) -> Dict[str, DataLoader]:
    """
    Create data loaders (on-demand loading version).
    
    Returns:
        A dictionary {'train': train_loader, 'test': test_loader}.
    """
    train_set = ClimateReconstructionDataset(
        data_path=data_path,
        years=train_years,
        **dataset_kwargs
    )
    
    test_set = ClimateReconstructionDataset(
        data_path=data_path,
        years=test_years,
        **dataset_kwargs
    )
    
    return {
        'train': DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        ),
        'test': DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    }

# Example usage
if __name__ == '__main__':
    DATA_DIR = "/jizhicfs/easyluwu/scaling_law/ft_local/low_res"
    
    # Create data loaders
    loaders = create_dataloaders(
        data_path=DATA_DIR,
        train_years=range(1980, 2019),
        test_years=range(2019, 2022),
        batch_size=2,
        variables=range(69)  # Use all variables
    )
    
    # Test data loading
    print(f"Number of training samples: {len(loaders['train'].dataset)}")
    print(f"Number of test samples: {len(loaders['test'].dataset)}")
    
    # Check one batch
    inputs, targets = next(iter(loaders['train']))
    print(f"\nInput shape: {inputs.shape}")  # Should be [B, 69, 180, 360]
    print(f"Target shape: {targets.shape}")  # Should be the same as input
    print(f"Data range: {inputs.min().item():.3f} ~ {inputs.max().item():.3f}")