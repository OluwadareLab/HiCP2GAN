"""
Dataset class for Plug&Play GAN training.
Loads paired 40x40 tiles and 224x224 global crops.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import os


class PnPGANDataset(Dataset):
    """
    Dataset that loads paired 40x40 tiles and 224x224 global crops.
    
    The dataset assumes:
    - 40x40 data: hicarn_10kb40kb_c40_s40_b201_nonpool_{split}.npz
    - 224x224 data: hicarn_10kb40kb_c224_s224_b201_nonpool_{split}.npz
    
    For each sample, we need to:
    1. Get a 224x224 global crop (LR and HR)
    2. Get corresponding 40x40 tiles that make up that global crop
    """
    
    def __init__(self, data_dir_40, data_dir_224, split='train', tile_size=40, global_size=224):
        """
        Args:
            data_dir_40: Directory containing 40x40 .npz files
            data_dir_224: Directory containing 224x224 .npz files
            split: 'train' or 'valid'
            tile_size: Size of local tiles (40)
            global_size: Size of global crops (224)
        """
        self.tile_size = tile_size
        self.global_size = global_size
        self.split = split
        
        # Load 40x40 data
        file_40 = os.path.join(data_dir_40, f'hicarn_10kb40kb_c40_s40_b201_nonpool_{split}.npz')
        data_40 = np.load(file_40, allow_pickle=True)
        # Data is stored as [N, 1, H, W] - squeeze to remove channel dimension if needed
        data_40_tensor = torch.tensor(data_40['data'], dtype=torch.float32)
        target_40_tensor = torch.tensor(data_40['target'], dtype=torch.float32)
        # Remove extra channel dimension if present (should be [N, H, W])
        if len(data_40_tensor.shape) == 4 and data_40_tensor.shape[1] == 1:
            self.data_40 = data_40_tensor.squeeze(1)  # [N, H, W]
            self.target_40 = target_40_tensor.squeeze(1)  # [N, H, W]
        else:
            self.data_40 = data_40_tensor
            self.target_40 = target_40_tensor
        self.inds_40 = data_40['inds']
        
        # Load 224x224 data
        file_224 = os.path.join(data_dir_224, f'hicarn_10kb40kb_c224_s224_b201_nonpool_{split}.npz')
        data_224 = np.load(file_224, allow_pickle=True)
        data_224_tensor = torch.tensor(data_224['data'], dtype=torch.float32)
        target_224_tensor = torch.tensor(data_224['target'], dtype=torch.float32)
        # Remove extra channel dimension if present (should be [N, H, W])
        if len(data_224_tensor.shape) == 4 and data_224_tensor.shape[1] == 1:
            self.data_224 = data_224_tensor.squeeze(1)  # [N, H, W]
            self.target_224 = target_224_tensor.squeeze(1)  # [N, H, W]
        else:
            self.data_224 = data_224_tensor
            self.target_224 = target_224_tensor
        self.inds_224 = data_224['inds']
        
        # Create mapping from 224x224 indices to 40x40 tiles
        # Each 224x224 crop corresponds to approximately (224/40)^2 = 31.36 tiles
        # We'll need to find tiles that overlap with each 224x224 region
        self._create_tile_mapping()
        
        print(f"Loaded {len(self.data_224)} global crops and {len(self.data_40)} tiles for {split} set")
    
    def _create_tile_mapping(self):
        """
        Create mapping from global crop index to corresponding tile indices.
        For simplicity, we'll sample tiles that could belong to the same region.
        """
        # For now, we'll use a simple approach: 
        # For each 224x224 crop, we'll randomly sample tiles from the same chromosome
        # In practice, you'd want to match based on genomic coordinates
        self.tile_mapping = []
        
        for i in range(len(self.inds_224)):
            # Get chromosome from 224x224 index
            chr_num_224 = self.inds_224[i][0]
            
            # Find tiles from same chromosome
            tile_indices = []
            for j in range(len(self.inds_40)):
                if self.inds_40[j][0] == chr_num_224:
                    tile_indices.append(j)
            
            # Sample up to 36 tiles (6x6 grid) that could form a 224x224 region
            # In practice, you'd want to ensure they're spatially adjacent
            if len(tile_indices) >= 36:
                selected = np.random.choice(tile_indices, 36, replace=False)
            else:
                selected = tile_indices[:36] if len(tile_indices) >= 36 else tile_indices
            
            self.tile_mapping.append(selected.tolist() if isinstance(selected, np.ndarray) else selected)
    
    def __len__(self):
        return len(self.data_224)
    
    def __getitem__(self, idx):
        """
        Returns:
            lr_global: 224x224 LR global crop [1, 224, 224]
            hr_global: 224x224 HR global crop [1, 224, 224]
            lr_tiles: List of 40x40 LR tiles [N, 1, 40, 40]
            hr_tiles: List of 40x40 HR tiles [N, 1, 40, 40]
            tile_indices: Indices of tiles used
        """
        # Get global crops
        lr_global = self.data_224[idx].unsqueeze(0)  # [1, 224, 224]
        hr_global = self.target_224[idx].unsqueeze(0)  # [1, 224, 224]
        
        # Get corresponding tiles
        tile_indices = self.tile_mapping[idx]
        num_tiles = len(tile_indices)
        
        if num_tiles > 0:
            lr_tiles = self.data_40[tile_indices].unsqueeze(1)  # [N, 1, 40, 40]
            hr_tiles = self.target_40[tile_indices].unsqueeze(1)  # [N, 1, 40, 40]
        else:
            # Fallback: create dummy tiles from global crop
            # Extract 40x40 patches from the global crop
            lr_tiles = []
            hr_tiles = []
            for i in range(0, self.global_size, self.tile_size):
                for j in range(0, self.global_size, self.tile_size):
                    if i + self.tile_size <= self.global_size and j + self.tile_size <= self.global_size:
                        lr_tiles.append(lr_global[:, i:i+self.tile_size, j:j+self.tile_size])
                        hr_tiles.append(hr_global[:, i:i+self.tile_size, j:j+self.tile_size])
            lr_tiles = torch.stack(lr_tiles)  # [N, 1, 40, 40]
            hr_tiles = torch.stack(hr_tiles)  # [N, 1, 40, 40]
        
        return {
            'lr_global': lr_global,
            'hr_global': hr_global,
            'lr_tiles': lr_tiles,
            'hr_tiles': hr_tiles,
            'tile_indices': tile_indices
        }
