#!/usr/bin/env python3
"""
Script to find optimal batch size for your GPU.
Run this to determine the maximum batch size that fits in your GPU memory.
"""
import torch
import sys
import os

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '../../..'))
sys.path.insert(0, script_dir)
sys.path.insert(0, root_dir)

from Models.HiCARN_1 import Generator
from dataset import PnPGANDataset
from discriminators import LocalDiscriminator, GlobalDiscriminator
from torch.utils.data import DataLoader

def find_max_batch_size(data_dir_40, data_dir_224, hicfoundation_path, device='cuda:0', start_batch=8):
    """
    Find the maximum batch size that fits in GPU memory.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy dataset
    try:
        dataset = PnPGANDataset(data_dir_40, data_dir_224, split='train')
        sample = dataset[0]
        print(f"Dataset loaded successfully")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create models
    model_G = Generator(num_channels=64).to(device)
    model_D_local = LocalDiscriminator(in_channels=1).to(device)
    model_D_global = GlobalDiscriminator(hicfoundation_path, freeze_encoder=True).to(device)
    
    batch_size = start_batch
    max_batch = None
    
    print(f"\nSearching for optimal batch size starting from {batch_size}...")
    
    while True:
        try:
            print(f"\nTrying batch_size = {batch_size}...")
            torch.cuda.empty_cache()
            
            # Create data loader with current batch size
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=0, pin_memory=False)
            batch = next(iter(loader))
            
            # Move to device
            lr_global = batch['lr_global'].to(device)
            hr_global = batch['hr_global'].to(device)
            lr_tiles = batch['lr_tiles'].to(device)
            hr_tiles = batch['hr_tiles'].to(device)
            
            # Forward pass for generator
            B, N, C, H, W = lr_tiles.shape
            lr_tiles_flat = lr_tiles.view(B * N, C, H, W)
            sr_tiles_flat = model_G(lr_tiles_flat)
            
            # Forward pass for discriminators
            pred_local = model_D_local(sr_tiles_flat[:min(100, len(sr_tiles_flat))])
            pred_global = model_D_global(hr_global, return_features=False)
            
            # Backward pass (dummy)
            loss = sr_tiles_flat.mean()
            loss.backward()
            
            # Clean up
            del lr_global, hr_global, lr_tiles, hr_tiles, sr_tiles_flat
            del pred_local, pred_global, loss
            torch.cuda.empty_cache()
            
            print(f"✓ batch_size = {batch_size} fits in memory!")
            max_batch = batch_size
            batch_size += 4  # Increase by 4
            
            if batch_size > 128:  # Safety limit
                print(f"\nReached safety limit. Maximum tested: {max_batch}")
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"✗ batch_size = {batch_size} exceeds GPU memory")
                torch.cuda.empty_cache()
                break
            else:
                raise e
    
    print(f"\n{'='*50}")
    print(f"Optimal batch size: {max_batch}")
    print(f"Recommended batch size: {max_batch - 2} (for safety margin)")
    print(f"{'='*50}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Find optimal batch size')
    parser.add_argument('--data_dir_40', type=str, required=True)
    parser.add_argument('--data_dir_224', type=str, required=True)
    parser.add_argument('--hicfoundation_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--start_batch', type=int, default=8)
    
    args = parser.parse_args()
    
    find_max_batch_size(args.data_dir_40, args.data_dir_224, args.hicfoundation_path,
                       args.device, args.start_batch)
