"""
Main training script for Plug&Play GAN.
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add paths - need to go up 3 levels from pnp_gan/ to reach root
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '../../..'))
# Add local directory first to avoid conflicts with parent modules
sys.path.insert(0, script_dir)
sys.path.insert(0, root_dir)
from Models.HiCARN_1 import Generator
# SSIM will be imported from utils
from math import log10

# Import from local pnp_gan module (local directory is first in path)
from dataset import PnPGANDataset
from discriminators import LocalDiscriminator, GlobalDiscriminator
# Import utils from local directory using importlib to avoid conflicts
import importlib.util
pnp_utils_path = os.path.join(script_dir, 'utils.py')
spec = importlib.util.spec_from_file_location("pnp_gan_utils", pnp_utils_path)
pnp_gan_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pnp_gan_utils)
stitch_tiles = pnp_gan_utils.stitch_tiles
calculate_psnr = pnp_gan_utils.calculate_psnr
calculate_ssim_batch = pnp_gan_utils.calculate_ssim_batch
save_sample_images = pnp_gan_utils.save_sample_images
import torch.nn.functional as F


def adjust_learning_rate(epoch, initial_lr=0.0003, decay_epoch=30, decay_rate=0.1):
    """Adjust learning rate"""
    lr = initial_lr * (decay_rate ** (epoch // decay_epoch))
    return lr


def train_epoch(model_G, model_D_local, model_D_global, 
                train_loader, optimizer_G, optimizer_D_local, optimizer_D_global,
                device, epoch, num_epochs, 
                lambda_rec=1.0, lambda_adv_local=0.01, lambda_adv_global=0.01, 
                lambda_fm=0.1, warmup_epochs=5, use_amp=False,
                scaler_G=None, scaler_D_local=None, scaler_D_global=None,
                gradient_accumulation_steps=1):
    """
    Train for one epoch.
    """
    model_G.train()
    model_D_local.train()
    model_D_global.train()
    
    # Warmup: disable adversarial losses for first few epochs
    if epoch <= warmup_epochs:
        lambda_adv_local = 0.0
        lambda_adv_global = 0.0
    
    # Loss functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    criterion_huber = nn.HuberLoss(delta=1.0)
    
    # Labels for real/fake
    real_label = 1.0
    fake_label = 0.0
    
    total_g_loss = 0.0
    total_d_local_loss = 0.0
    total_d_global_loss = 0.0
    total_rec_loss = 0.0
    total_adv_local_loss = 0.0
    total_adv_global_loss = 0.0
    total_fm_loss = 0.0
    num_samples = 0
    
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
    
    # Clear cache at start of epoch
    torch.cuda.empty_cache()
    
    for batch_idx, batch in enumerate(train_bar):
        # Clear cache at start of each batch to prevent OOM
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        lr_global = batch['lr_global'].to(device)  # [B, 1, 224, 224]
        hr_global = batch['hr_global'].to(device)  # [B, 1, 224, 224]
        lr_tiles = batch['lr_tiles'].to(device)  # [B, N, 1, 40, 40]
        hr_tiles = batch['hr_tiles'].to(device)  # [B, N, 1, 40, 40]
        
        batch_size = lr_global.shape[0]
        num_samples += batch_size
        
        # ========== Train Generator on tiles ==========
        # Generate SR tiles
        B, N, C, H, W = lr_tiles.shape
        lr_tiles_flat = lr_tiles.view(B * N, C, H, W)  # [B*N, 1, 40, 40]
        
        # Clear cache before generator forward pass
        torch.cuda.empty_cache()
        sr_tiles_flat = model_G(lr_tiles_flat)  # [B*N, 1, 40, 40]
        sr_tiles = sr_tiles_flat.view(B, N, C, H, W)  # [B, N, 1, 40, 40]
        
        # Stitch tiles to create global SR
        sr_global_list = []
        for b in range(batch_size):
            # Stitch tiles for this sample
            # Assuming we have a 6x6 grid (36 tiles = 240x240, crop to 224x224)
            grid_size = (6, 6)
            if N >= 36:
                tiles_to_stitch = sr_tiles[b, :36].view(36, C, H, W)
            else:
                # Pad if needed
                tiles_to_stitch = sr_tiles[b]
                padding = torch.zeros(36 - N, C, H, W, device=tiles_to_stitch.device)
                tiles_to_stitch = torch.cat([tiles_to_stitch, padding], dim=0)
            
            stitched = stitch_tiles(tiles_to_stitch, grid_size=grid_size, tile_size=40)
            # Crop to 224x224 if needed
            if stitched.shape[0] > 224:
                stitched = stitched[:224, :224]
            elif stitched.shape[0] < 224:
                # Pad if needed
                pad_h = 224 - stitched.shape[0]
                pad_w = 224 - stitched.shape[1]
                stitched = F.pad(stitched, (0, pad_w, 0, pad_h))
            
            sr_global_list.append(stitched)
        
        sr_global = torch.stack(sr_global_list).unsqueeze(1)  # [B, 1, 224, 224]
        
        # ========== Update D_local ==========
        # Only zero grad at start or after accumulation steps
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer_D_local.zero_grad()
        
        # Real tiles
        hr_tiles_flat = hr_tiles.view(B * N, C, H, W)
        pred_real_local = model_D_local(hr_tiles_flat)
        loss_D_real_local = criterion_gan(pred_real_local, 
                                         torch.ones_like(pred_real_local) * real_label)
        
        # Fake tiles
        pred_fake_local = model_D_local(sr_tiles_flat.detach())
        loss_D_fake_local = criterion_gan(pred_fake_local,
                                          torch.ones_like(pred_fake_local) * fake_label)
        
        loss_D_local = (loss_D_real_local + loss_D_fake_local) * 0.5
        loss_D_local = loss_D_local / gradient_accumulation_steps
        loss_D_local.backward()
        
        # Only step optimizer every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping to prevent explosion (more aggressive)
            torch.nn.utils.clip_grad_norm_(model_D_local.parameters(), max_norm=0.5)
            optimizer_D_local.step()
            optimizer_D_local.zero_grad()
            torch.cuda.empty_cache()
        
        # ========== Update D_global ==========
        # Only zero grad at start or after accumulation steps
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer_D_global.zero_grad()
        
        # Real global
        pred_real_global = model_D_global(hr_global, return_features=False)
        loss_D_real_global = criterion_gan(pred_real_global,
                                         torch.ones_like(pred_real_global) * real_label)
        
        # Fake global
        pred_fake_global = model_D_global(sr_global.detach(), return_features=False)
        loss_D_fake_global = criterion_gan(pred_fake_global,
                                           torch.ones_like(pred_fake_global) * fake_label)
        
        loss_D_global = (loss_D_real_global + loss_D_fake_global) * 0.5
        loss_D_global = loss_D_global / gradient_accumulation_steps
        loss_D_global.backward()
        
        # Only step optimizer every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping to prevent explosion (more aggressive)
            torch.nn.utils.clip_grad_norm_(model_D_global.parameters(), max_norm=0.5)
            optimizer_D_global.step()
            optimizer_D_global.zero_grad()
            torch.cuda.empty_cache()
        
        # ========== Update G ==========
        # Only zero grad at start or after accumulation steps
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer_G.zero_grad()
        
        # Reconstruction loss on tiles
        hr_tiles_flat = hr_tiles.view(B * N, C, H, W)
        loss_rec = criterion_huber(sr_tiles_flat, hr_tiles_flat)
        
        # Local adversarial loss
        pred_fake_local_G = model_D_local(sr_tiles_flat)
        loss_adv_local = criterion_gan(pred_fake_local_G,
                                      torch.ones_like(pred_fake_local_G) * real_label)
        
        # Global adversarial loss
        pred_fake_global_G, features_fake = model_D_global(sr_global, return_features=True)
        loss_adv_global = criterion_gan(pred_fake_global_G,
                                       torch.ones_like(pred_fake_global_G) * real_label)
        
        # Feature matching loss
        _, features_real = model_D_global(hr_global, return_features=True)
        loss_fm = criterion_l1(features_fake, features_real)
        
        # Total generator loss
        # Clip adversarial losses to prevent them from dominating
        loss_adv_local_clipped = torch.clamp(loss_adv_local, max=10.0)
        loss_adv_global_clipped = torch.clamp(loss_adv_global, max=10.0)
        
        loss_G = (lambda_rec * loss_rec + 
                 lambda_adv_local * loss_adv_local_clipped +
                 lambda_adv_global * loss_adv_global_clipped +
                 lambda_fm * loss_fm)
        loss_G = loss_G / gradient_accumulation_steps
        
        # Check for NaN before backward
        if torch.isnan(loss_G):
            print(f"WARNING: NaN detected in generator loss at batch {batch_idx}. Skipping batch.")
            num_samples -= batch_size  # Don't count skipped batch
            # Clear cache and continue
            torch.cuda.empty_cache()
            continue
        
        loss_G.backward()
        
        # Only step optimizer every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping to prevent explosion (more aggressive)
            torch.nn.utils.clip_grad_norm_(model_G.parameters(), max_norm=0.5)
            optimizer_G.step()
            optimizer_G.zero_grad()
            
            # Clear cache after every optimizer step to prevent OOM
            torch.cuda.empty_cache()
        
        # Clear intermediate variables to free memory immediately
        del pred_real_local, pred_fake_local, pred_real_global, pred_fake_global
        del pred_fake_local_G, pred_fake_global_G, features_fake, features_real
        del sr_global, stitched, tiles_to_stitch, sr_global_list
        
        # Accumulate losses (multiply by gradient_accumulation_steps to get correct average)
        total_g_loss += loss_G.item() * batch_size * gradient_accumulation_steps
        total_d_local_loss += loss_D_local.item() * batch_size * gradient_accumulation_steps
        total_d_global_loss += loss_D_global.item() * batch_size * gradient_accumulation_steps
        total_rec_loss += loss_rec.item() * batch_size * gradient_accumulation_steps
        # Use clipped values for accumulation to reflect actual training
        total_adv_local_loss += loss_adv_local_clipped.item() * batch_size * gradient_accumulation_steps
        total_adv_global_loss += loss_adv_global_clipped.item() * batch_size * gradient_accumulation_steps
        total_fm_loss += loss_fm.item() * batch_size * gradient_accumulation_steps
        
        # Update progress bar
        if num_samples > 0:
            train_bar.set_description(
                f'Epoch {epoch}/{num_epochs} | '
                f'G: {total_g_loss/num_samples:.4f} | '
                f'D_l: {total_d_local_loss/num_samples:.4f} | '
                f'D_g: {total_d_global_loss/num_samples:.4f}'
            )
    
    # Prevent division by zero
    if num_samples == 0:
        print("WARNING: No valid samples processed in this epoch!")
        return {
            'g_loss': float('inf'),
            'd_local_loss': float('inf'),
            'd_global_loss': float('inf'),
            'rec_loss': float('inf'),
            'adv_local_loss': float('inf'),
            'adv_global_loss': float('inf'),
            'fm_loss': float('inf')
        }
    
    return {
        'g_loss': total_g_loss / num_samples,
        'd_local_loss': total_d_local_loss / num_samples,
        'd_global_loss': total_d_global_loss / num_samples,
        'rec_loss': total_rec_loss / num_samples,
        'adv_local_loss': total_adv_local_loss / num_samples,
        'adv_global_loss': total_adv_global_loss / num_samples,
        'fm_loss': total_fm_loss / num_samples
    }


def validate(model_G, model_D_local, model_D_global, val_loader, device):
    """
    Validate the model.
    """
    model_G.eval()
    model_D_local.eval()
    model_D_global.eval()
    
    criterion_l1 = nn.L1Loss()
    criterion_huber = nn.HuberLoss(delta=1.0)
    
    total_ssim_tiles = 0.0
    total_psnr_tiles = 0.0
    total_ssim_global = 0.0
    total_psnr_global = 0.0
    total_l1_tiles = 0.0
    total_l1_global = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            lr_global = batch['lr_global'].to(device)
            hr_global = batch['hr_global'].to(device)
            lr_tiles = batch['lr_tiles'].to(device)
            hr_tiles = batch['hr_tiles'].to(device)
            
            batch_size = lr_global.shape[0]
            num_samples += batch_size
            
            # Generate SR tiles
            B, N, C, H, W = lr_tiles.shape
            lr_tiles_flat = lr_tiles.view(B * N, C, H, W)
            sr_tiles_flat = model_G(lr_tiles_flat)
            sr_tiles = sr_tiles_flat.view(B, N, C, H, W)
            
            # Stitch to global
            sr_global_list = []
            for b in range(batch_size):
                grid_size = (6, 6)
                if N >= 36:
                    tiles_to_stitch = sr_tiles[b, :36].view(36, C, H, W)
                else:
                    tiles_to_stitch = sr_tiles[b]
                    padding = torch.zeros(36 - N, C, H, W, device=tiles_to_stitch.device)
                    tiles_to_stitch = torch.cat([tiles_to_stitch, padding], dim=0)
                
                stitched = stitch_tiles(tiles_to_stitch, grid_size=grid_size, tile_size=40)
                if stitched.shape[0] > 224:
                    stitched = stitched[:224, :224]
                elif stitched.shape[0] < 224:
                    pad_h = 224 - stitched.shape[0]
                    pad_w = 224 - stitched.shape[1]
                    stitched = F.pad(stitched, (0, pad_w, 0, pad_h))
                
                sr_global_list.append(stitched)
            
            sr_global = torch.stack(sr_global_list).unsqueeze(1)  # [B, 1, 224, 224]
            
            # Check for NaN in generated images before computing metrics
            if torch.isnan(sr_tiles_flat).any() or torch.isnan(sr_global).any():
                print(f"WARNING: NaN detected in generated images. Skipping batch.")
                num_samples -= batch_size  # Don't count skipped batch
                continue
            
            # Helper function to check for NaN/Inf (works with both tensor and float)
            def is_nan_or_inf(val):
                if isinstance(val, torch.Tensor):
                    return torch.isnan(val).any() or torch.isinf(val).any()
                else:
                    import math
                    return math.isnan(val) or math.isinf(val)
            
            # Metrics on tiles
            hr_tiles_flat = hr_tiles.view(B * N, C, H, W)
            ssim_tiles = calculate_ssim_batch(sr_tiles_flat, hr_tiles_flat)
            psnr_tiles = calculate_psnr(sr_tiles_flat, hr_tiles_flat)
            l1_tiles = criterion_l1(sr_tiles_flat, hr_tiles_flat)
            
            # Check for NaN in metrics (handle both tensor and float)
            if is_nan_or_inf(ssim_tiles) or is_nan_or_inf(psnr_tiles) or is_nan_or_inf(l1_tiles):
                print(f"WARNING: NaN/Inf detected in tile metrics. Skipping batch.")
                num_samples -= batch_size  # Don't count skipped batch
                continue
            
            # Metrics on global
            ssim_global = calculate_ssim_batch(sr_global, hr_global)
            psnr_global = calculate_psnr(sr_global, hr_global)
            l1_global = criterion_l1(sr_global, hr_global)
            
            # Check for NaN in global metrics (handle both tensor and float)
            if is_nan_or_inf(ssim_global) or is_nan_or_inf(psnr_global) or is_nan_or_inf(l1_global):
                print(f"WARNING: NaN/Inf detected in global metrics. Skipping batch.")
                num_samples -= batch_size  # Don't count skipped batch
                continue
            
            total_ssim_tiles += ssim_tiles.item() * batch_size
            total_psnr_tiles += psnr_tiles * batch_size
            total_ssim_global += ssim_global.item() * batch_size
            total_psnr_global += psnr_global * batch_size
            total_l1_tiles += l1_tiles.item() * batch_size
            total_l1_global += l1_global.item() * batch_size
    
    # Prevent division by zero
    if num_samples == 0:
        print("WARNING: No valid samples processed in validation!")
        return {
            'ssim_tiles': float('nan'),
            'psnr_tiles': float('nan'),
            'ssim_global': float('nan'),
            'psnr_global': float('nan'),
            'l1_tiles': float('nan'),
            'l1_global': float('nan')
        }
    
    return {
        'ssim_tiles': total_ssim_tiles / num_samples,
        'psnr_tiles': total_psnr_tiles / num_samples,
        'ssim_global': total_ssim_global / num_samples,
        'psnr_global': total_psnr_global / num_samples,
        'l1_tiles': total_l1_tiles / num_samples,
        'l1_global': total_l1_global / num_samples
    }


def main():
    parser = argparse.ArgumentParser(description='Train Plug&Play GAN')
    parser.add_argument('--data_dir_40', type=str, required=True,
                       help='Directory containing 40x40 data')
    parser.add_argument('--data_dir_224', type=str, required=True,
                       help='Directory containing 224x224 data')
    parser.add_argument('--hicfoundation_path', type=str, required=True,
                       help='Path to HiCFoundation checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/pnp_gan',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/pnp_gan',
                       help='Directory to save logs and images')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8, increase if you have more GPU memory)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision for faster training and less memory')
    parser.add_argument('--pin_memory', action='store_true', default=False,
                       help='Pin memory for faster GPU transfer (disable in Docker if shared memory issues)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers (0 to disable multiprocessing, recommended for Docker)')
    parser.add_argument('--lr_G', type=float, default=0.0003,
                       help='Learning rate for generator')
    parser.add_argument('--lr_D', type=float, default=0.0001,
                       help='Learning rate for discriminators')
    parser.add_argument('--lambda_rec', type=float, default=1.0,
                       help='Weight for reconstruction loss')
    parser.add_argument('--lambda_adv_local', type=float, default=0.01,
                       help='Weight for local adversarial loss')
    parser.add_argument('--lambda_adv_global', type=float, default=0.01,
                       help='Weight for global adversarial loss')
    parser.add_argument('--lambda_fm', type=float, default=0.1,
                       help='Weight for feature matching loss')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Number of warmup epochs (no adversarial loss)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, 'images'), exist_ok=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Datasets
    train_dataset = PnPGANDataset(args.data_dir_40, args.data_dir_224, split='train')
    val_dataset = PnPGANDataset(args.data_dir_40, args.data_dir_224, split='valid')
    
    # Data loaders with optimized settings
    # Note: In Docker, use num_workers=0 to avoid shared memory issues
    # For maximum GPU efficiency, increase batch_size instead
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory if args.num_workers == 0 else False,  # Only pin memory if no multiprocessing
        persistent_workers=False  # Disable to avoid shared memory issues
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory if args.num_workers == 0 else False,
        persistent_workers=False
    )
    
    # Mixed precision training scalers
    if args.use_amp:
        scaler_G = torch.cuda.amp.GradScaler()
        scaler_D_local = torch.cuda.amp.GradScaler()
        scaler_D_global = torch.cuda.amp.GradScaler()
        print("Mixed precision training enabled")
    else:
        scaler_G = None
        scaler_D_local = None
        scaler_D_global = None
    
    # Models
    model_G = Generator(num_channels=64).to(device)
    model_D_local = LocalDiscriminator(in_channels=1).to(device)
    model_D_global = GlobalDiscriminator(args.hicfoundation_path, freeze_encoder=True).to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(model_G.parameters(), lr=args.lr_G)
    optimizer_D_local = optim.Adam(model_D_local.parameters(), lr=args.lr_D)
    optimizer_D_global = optim.Adam(model_D_global.gan_head.parameters(), lr=args.lr_D)
    
    # Training history
    history = {
        'train': {'g_loss': [], 'd_local_loss': [], 'd_global_loss': [], 
                 'rec_loss': [], 'adv_local_loss': [], 'adv_global_loss': [], 'fm_loss': []},
        'val': {'ssim_tiles': [], 'psnr_tiles': [], 'ssim_global': [], 
               'psnr_global': [], 'l1_tiles': [], 'l1_global': []}
    }
    
    best_ssim_global = 0.0
    start_epoch = 1
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        model_G.load_state_dict(checkpoint['generator'])
        model_D_local.load_state_dict(checkpoint['discriminator_local'])
        model_D_global.gan_head.load_state_dict(checkpoint['discriminator_global'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D_local.load_state_dict(checkpoint['optimizer_D_local'])
        optimizer_D_global.load_state_dict(checkpoint['optimizer_D_global'])
        start_epoch = checkpoint['epoch'] + 1
        best_ssim_global = checkpoint.get('best_ssim_global', 0.0)
        history = checkpoint.get('history', history)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs + 1):
        # Adjust learning rate
        lr_G = adjust_learning_rate(epoch, args.lr_G)
        lr_D = adjust_learning_rate(epoch, args.lr_D)
        for param_group in optimizer_G.param_groups:
            param_group['lr'] = lr_G
        for param_group in optimizer_D_local.param_groups:
            param_group['lr'] = lr_D
        for param_group in optimizer_D_global.param_groups:
            param_group['lr'] = lr_D
        
        # Train
        train_metrics = train_epoch(
            model_G, model_D_local, model_D_global,
            train_loader, optimizer_G, optimizer_D_local, optimizer_D_global,
            device, epoch, args.num_epochs,
            args.lambda_rec, args.lambda_adv_local, args.lambda_adv_global,
            args.lambda_fm, args.warmup_epochs,
            args.use_amp, 
            scaler_G if args.use_amp else None,
            scaler_D_local if args.use_amp else None,
            scaler_D_global if args.use_amp else None
        )
        
        # Validate
        val_metrics = validate(model_G, model_D_local, model_D_global, val_loader, device)
        
        # Check for NaN in validation metrics - early stopping
        has_nan = False
        for key, value in val_metrics.items():
            if isinstance(value, float) and (value != value or value == float('inf') or value == float('-inf')):
                print(f"ERROR: NaN or Inf detected in validation metric {key}: {value}")
                has_nan = True
            elif isinstance(value, torch.Tensor) and (torch.isnan(value).any() or torch.isinf(value).any()):
                print(f"ERROR: NaN or Inf detected in validation metric {key}: {value}")
                has_nan = True
        
        if has_nan:
            print("ERROR: NaN/Inf detected in validation metrics. Training stopped.")
            print("Recommendation: Resume from epoch 5 checkpoint with lower learning rates.")
            break
        
        # Check for NaN in training metrics
        for key, value in train_metrics.items():
            if isinstance(value, float) and (value != value or value == float('inf') or value == float('-inf')):
                print(f"ERROR: NaN or Inf detected in training metric {key}: {value}")
                has_nan = True
            elif isinstance(value, torch.Tensor) and (torch.isnan(value).any() or torch.isinf(value).any()):
                print(f"ERROR: NaN or Inf detected in training metric {key}: {value}")
                has_nan = True
        
        if has_nan:
            print("ERROR: NaN/Inf detected in training metrics. Training stopped.")
            break
        
        # Update history
        for key in train_metrics:
            history['train'][key].append(train_metrics[key])
        for key in val_metrics:
            history['val'][key].append(val_metrics[key])
        
        # Print metrics
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"Train - G Loss: {train_metrics['g_loss']:.4f}, "
              f"Rec: {train_metrics['rec_loss']:.4f}, "
              f"Adv Local: {train_metrics['adv_local_loss']:.4f}, "
              f"Adv Global: {train_metrics['adv_global_loss']:.4f}, "
              f"FM: {train_metrics['fm_loss']:.4f}")
        print(f"Val - SSIM Global: {val_metrics['ssim_global']:.4f}, "
              f"PSNR Global: {val_metrics['psnr_global']:.4f}, "
              f"SSIM Tiles: {val_metrics['ssim_tiles']:.4f}, "
              f"PSNR Tiles: {val_metrics['psnr_tiles']:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'generator': model_G.state_dict(),
            'discriminator_local': model_D_local.state_dict(),
            'discriminator_global': model_D_global.gan_head.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D_local': optimizer_D_local.state_dict(),
            'optimizer_D_global': optimizer_D_global.state_dict(),
            'best_ssim_global': best_ssim_global,
            'history': history
        }
        
        # Save best model
        if val_metrics['ssim_global'] > best_ssim_global:
            best_ssim_global = val_metrics['ssim_global']
            checkpoint['best_ssim_global'] = best_ssim_global
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"Saved best model with SSIM Global: {best_ssim_global:.4f}")
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest_model.pth'))
        
        # Save sample images every 10 epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                lr_global = sample_batch['lr_global'][:1].to(device)
                hr_global = sample_batch['hr_global'][:1].to(device)
                lr_tiles = sample_batch['lr_tiles'][:1].to(device)
                hr_tiles = sample_batch['hr_tiles'][:1].to(device)
                
                B, N, C, H, W = lr_tiles.shape
                lr_tiles_flat = lr_tiles.view(B * N, C, H, W)
                sr_tiles_flat = model_G(lr_tiles_flat)
                sr_tiles = sr_tiles_flat.view(B, N, C, H, W)
                
                # Stitch global
                grid_size = (6, 6)
                if N >= 36:
                    tiles_to_stitch = sr_tiles[0, :36].view(36, C, H, W)
                else:
                    tiles_to_stitch = sr_tiles[0]
                    padding = torch.zeros(36 - N, C, H, W, device=tiles_to_stitch.device)
                    tiles_to_stitch = torch.cat([tiles_to_stitch, padding], dim=0)
                
                stitched = stitch_tiles(tiles_to_stitch, grid_size=grid_size, tile_size=40)
                if stitched.shape[0] > 224:
                    stitched = stitched[:224, :224]
                elif stitched.shape[0] < 224:
                    pad_h = 224 - stitched.shape[0]
                    pad_w = 224 - stitched.shape[1]
                    stitched = F.pad(stitched, (0, pad_w, 0, pad_h))
                
                sr_global = stitched.unsqueeze(0).unsqueeze(0)
                
                save_sample_images(
                    lr_tiles[0], hr_tiles[0], sr_tiles[0],
                    lr_global[0], hr_global[0], sr_global[0],
                    epoch, os.path.join(args.log_dir, 'images')
                )
        
        # Save history to text file
        with open(os.path.join(args.log_dir, 'training_history.txt'), 'w') as f:
            f.write("Epoch\tTrain_G_Loss\tTrain_Rec_Loss\tTrain_Adv_Local\tTrain_Adv_Global\tTrain_FM\t")
            f.write("Val_SSIM_Global\tVal_PSNR_Global\tVal_SSIM_Tiles\tVal_PSNR_Tiles\tVal_L1_Global\tVal_L1_Tiles\n")
            for i in range(len(history['train']['g_loss'])):
                f.write(f"{i+1}\t")
                f.write(f"{history['train']['g_loss'][i]:.6f}\t")
                f.write(f"{history['train']['rec_loss'][i]:.6f}\t")
                f.write(f"{history['train']['adv_local_loss'][i]:.6f}\t")
                f.write(f"{history['train']['adv_global_loss'][i]:.6f}\t")
                f.write(f"{history['train']['fm_loss'][i]:.6f}\t")
                f.write(f"{history['val']['ssim_global'][i]:.6f}\t")
                f.write(f"{history['val']['psnr_global'][i]:.6f}\t")
                f.write(f"{history['val']['ssim_tiles'][i]:.6f}\t")
                f.write(f"{history['val']['psnr_tiles'][i]:.6f}\t")
                f.write(f"{history['val']['l1_global'][i]:.6f}\t")
                f.write(f"{history['val']['l1_tiles'][i]:.6f}\n")
    
    print("Training completed!")


if __name__ == '__main__':
    main()
