# Plug&Play GAN Training for Hi-C Data Enhancement

This module implements a multi-scale Plug&Play GAN for enhancing Hi-C data resolution, following the plan outlined in `PnP_GAN_Plan.md`.

## Overview

The training uses:
- **Generator (G)**: HiCARN-1 generator that operates on 40×40 tiles (LR → SR)
- **D_local**: PatchGAN discriminator for 40×40 tiles
- **D_global**: HiCFoundation-based discriminator for 224×224 global crops

## Data Requirements

You need two sets of data:
1. **40×40 tiles**: Located in `Data/R16_down/GM12878/data_40_40/`
   - `hicarn_10kb40kb_c40_s40_b201_nonpool_train.npz`
   - `hicarn_10kb40kb_c40_s40_b201_nonpool_valid.npz`

2. **224×224 global crops**: Located in `Data/R16_down/GM12878/data_224/`
   - `hicarn_10kb40kb_c224_s224_b201_nonpool_train.npz`
   - `hicarn_10kb40kb_c224_s224_b201_nonpool_valid.npz`

## Installation

Make sure you have all dependencies installed. The code uses:
- PyTorch
- NumPy
- Matplotlib
- tqdm
- HiCARN models (from parent directory)
- HiCFoundation models

## Usage

### Training

```bash
cd /storage/solowofi/HiCARN/FoundationGANWorks/HiCFoundation/pnp_gan

python train.py \
    --data_dir_40 /storage/solowofi/HiCARN/Data/R16_down/GM12878/data_40_40 \
    --data_dir_224 /storage/solowofi/HiCARN/Data/R16_down/GM12878/data_224 \
    --hicfoundation_path /storage/solowofi/HiCARN/FoundationGANWorks/HiCFoundation/hicfoundation_model/hicfoundation_resolution.pth.tar \
    --checkpoint_dir checkpoints/pnp_gan \
    --log_dir logs/pnp_gan \
    --num_epochs 100 \
    --batch_size 4 \
    --lr_G 0.0003 \
    --lr_D 0.0001 \
    --lambda_rec 1.0 \
    --lambda_adv_local 0.01 \
    --lambda_adv_global 0.01 \
    --lambda_fm 0.1 \
    --warmup_epochs 5 \
    --device cuda:0
```

### Arguments

- `--data_dir_40`: Directory containing 40×40 .npz files
- `--data_dir_224`: Directory containing 224×224 .npz files
- `--hicfoundation_path`: Path to HiCFoundation checkpoint file
- `--checkpoint_dir`: Directory to save model checkpoints
- `--log_dir`: Directory to save logs and images
- `--num_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 4, adjust based on GPU memory)
- `--lr_G`: Learning rate for generator (default: 0.0003)
- `--lr_D`: Learning rate for discriminators (default: 0.0001)
- `--lambda_rec`: Weight for reconstruction loss (default: 1.0)
- `--lambda_adv_local`: Weight for local adversarial loss (default: 0.01)
- `--lambda_adv_global`: Weight for global adversarial loss (default: 0.01)
- `--lambda_fm`: Weight for feature matching loss (default: 0.1)
- `--warmup_epochs`: Number of epochs with no adversarial loss (default: 5)
- `--device`: Device to use (default: cuda:0)
- `--resume`: Path to checkpoint to resume training from (optional)

### Plotting Results

After training, generate plots:

```bash
python plot_results.py \
    --log_file logs/pnp_gan/training_history.txt \
    --save_dir logs/pnp_gan/plots
```

This will generate:
- `ssim_curves.png`: SSIM curves for global and tiles
- `psnr_curves.png`: PSNR curves for global and tiles
- `generator_losses.png`: All generator loss components
- `combined_metrics.png`: Combined view of all metrics

## Training Process

1. **Warmup Phase** (first `warmup_epochs` epochs):
   - Only reconstruction loss and feature matching are active
   - Adversarial losses are disabled (λ = 0)

2. **Full Training**:
   - Generator is trained with:
     - Reconstruction loss (L1/Huber on tiles)
     - Local adversarial loss (from D_local)
     - Global adversarial loss (from D_global)
     - Feature matching loss (HiCFoundation features)
   - Discriminators are updated to distinguish real vs fake

3. **Validation**:
   - Metrics computed at both tile (40×40) and global (224×224) scales
   - SSIM and PSNR are primary metrics
   - Best model saved based on global SSIM

## Output Files

- `checkpoints/pnp_gan/best_model.pth`: Best model checkpoint
- `checkpoints/pnp_gan/latest_model.pth`: Latest checkpoint
- `logs/pnp_gan/training_history.txt`: Training metrics per epoch
- `logs/pnp_gan/images/epoch_*.png`: Sample images every 10 epochs

## Notes

- The dataset class pairs 40×40 tiles with 224×224 global crops
- Tiles are stitched together to form global crops for the discriminator
- HiCFoundation encoder is frozen; only the GAN head is trained
- Adjust batch size based on available GPU memory
- Learning rates decay every 30 epochs

## Troubleshooting

1. **Out of memory**: Reduce batch size or use gradient accumulation
2. **Import errors**: Make sure parent directories are in Python path
3. **Data loading issues**: Verify .npz files exist and have correct keys
4. **HiCFoundation loading**: Check that the checkpoint path is correct
