"""
Utility functions for Plug&Play GAN training.
"""
import torch
import torch.nn.functional as F
import numpy as np
from math import log10


def stitch_tiles(tiles, grid_size=(6, 6), tile_size=40):
    """
    Stitch 40x40 tiles into a larger image.
    
    Args:
        tiles: Tensor of shape [N, 1, 40, 40] where N = grid_size[0] * grid_size[1]
        grid_size: (rows, cols) of the grid
        tile_size: Size of each tile
    
    Returns:
        stitched: Tensor of shape [1, H, W] where H = grid_size[0] * tile_size
    """
    if tiles.shape[0] < grid_size[0] * grid_size[1]:
        # Pad with zeros if needed
        num_needed = grid_size[0] * grid_size[1]
        padding = torch.zeros(num_needed - tiles.shape[0], *tiles.shape[1:], 
                             device=tiles.device, dtype=tiles.dtype)
        tiles = torch.cat([tiles, padding], dim=0)
    
    # Reshape to grid
    tiles = tiles[:grid_size[0] * grid_size[1]]  # Ensure correct number
    tiles = tiles.view(grid_size[0], grid_size[1], 1, tile_size, tile_size)
    
    # Concatenate along rows
    rows = []
    for i in range(grid_size[0]):
        row = torch.cat([tiles[i, j] for j in range(grid_size[1])], dim=2)  # Concatenate horizontally
        rows.append(row)
    
    # Concatenate rows vertically
    stitched = torch.cat(rows, dim=1)  # [1, H, W]
    
    return stitched.squeeze(0)  # [H, W]


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR between two images.
    
    Args:
        img1, img2: Tensors of same shape
        max_val: Maximum value in images
    
    Returns:
        psnr: PSNR in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * log10(max_val) - 10 * log10(mse)
    return psnr


def calculate_ssim_batch(img1, img2, window_size=11):
    """
    Calculate SSIM for a batch of images.
    Uses the SSIM function from Utils.
    """
    import sys
    import os
    # Add root directory to path - need to go up 3 levels from pnp_gan/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '../../..'))
    sys.path.insert(0, root_dir)
    from Utils.SSIM import ssim
    return ssim(img1, img2, window_size=window_size)


def save_sample_images(lr_tiles, hr_tiles, sr_tiles, lr_global, hr_global, sr_global, 
                       epoch, save_dir, num_samples=4):
    """
    Save sample images for visualization.
    """
    import os
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save tile samples
    fig, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
    
    for i in range(min(num_samples, lr_tiles.shape[0])):
        # LR tile
        axes[0, i].imshow(lr_tiles[i, 0].cpu().numpy(), cmap='Reds')
        axes[0, i].set_title(f'LR Tile {i}')
        axes[0, i].axis('off')
        
        # SR tile
        axes[1, i].imshow(sr_tiles[i, 0].cpu().numpy(), cmap='Reds')
        axes[1, i].set_title(f'SR Tile {i}')
        axes[1, i].axis('off')
        
        # HR tile
        axes[2, i].imshow(hr_tiles[i, 0].cpu().numpy(), cmap='Reds')
        axes[2, i].set_title(f'HR Tile {i}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_tiles.png'))
    plt.close()
    
    # Save global samples
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(lr_global[0, 0].cpu().numpy(), cmap='Reds')
    axes[0].set_title('LR Global')
    axes[0].axis('off')
    
    axes[1].imshow(sr_global[0, 0].cpu().numpy(), cmap='Reds')
    axes[1].set_title('SR Global')
    axes[1].axis('off')
    
    axes[2].imshow(hr_global[0, 0].cpu().numpy(), cmap='Reds')
    axes[2].set_title('HR Global')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_global.png'))
    plt.close()
