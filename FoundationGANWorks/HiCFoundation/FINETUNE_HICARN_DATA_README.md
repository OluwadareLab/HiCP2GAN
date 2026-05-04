# Fine-tuning HiCFoundation with HiCARN Data Format

This guide explains how to fine-tune **HiCFoundation** using your existing data format (from `HiCARN/Data/R16_down/GM12878/data`). 

**Note**: This guide is exclusively for HiCFoundation fine-tuning. The data format reference (HiCARN) is only mentioned to indicate the source data format - we are NOT training or fine-tuning DiCARN or any other model.

## Overview

HiCARN uses `.npz` files with format:
- `data`: (N, 1, 40, 40) - Low-resolution input
- `target`: (N, 1, 40, 40) - High-resolution target
- `inds`: (N, 3) - Indices (optional)

HiCFoundation expects `.pkl` files with dictionaries containing:
- `input`: (H, W) - Input Hi-C matrix
- `2d_target`: (H, W) - Target Hi-C matrix

## Step 1: Convert Data Format

Convert your HiCARN `.npz` files to HiCFoundation `.pkl` format:

```bash
cd FoundationGANWorks/HiCFoundation

python convert_hicarn_to_hicfoundation.py \
    --input_dir /path/to/HiCARN/Data/R16_down/GM12878/data \
    --output_dir hicfoundation_data \
    --train_file hicarn_10kb40kb_c40_s40_b201_nonpool_train.npz \
    --val_file hicarn_10kb40kb_c40_s40_b201_nonpool_valid.npz \
    --target_size 48 48
```

**Important Notes:**
- Your HiCARN data is 40×40, but HiCFoundation's patch_size=16 requires dimensions divisible by 16
- We pad 40×40 → 48×48 (8 pixels padding total, 4 pixels per side when centered)
- Alternatively, you can use `--target_size 224 224` for standard HiCFoundation input size (more padding)

The conversion script will:
1. Create `hicfoundation_data/train/` and `hicfoundation_data/val/` directories
2. Convert each sample to individual `.pkl` files
3. Create `train.txt` and `val.txt` config files

## Step 2: Prepare Pre-trained Model

You need a pre-trained HiCFoundation model checkpoint. Download or use your pre-trained model:
- Update `PRETRAIN_MODEL` path in the fine-tuning scripts

## Step 3: Run HiCFoundation Fine-tuning (Two Scripts in Parallel)

We provide two scripts for HiCFoundation fine-tuning with different strategies. Both scripts fine-tune HiCFoundation exclusively - one freezes the encoder (decoder-only), the other trains the full model:

### Option A: HiCFoundation Decoder-Only Fine-tuning (GPU 0)

This script fine-tunes **HiCFoundation** with the encoder frozen - only trains the decoder (8 transformer blocks, 128 attention heads):

```bash
# Edit the script to set your paths, then run:
bash finetune_task3_decoder_only.sh
```

Or manually:
```bash
export CUDA_VISIBLE_DEVICES=0

python finetune.py \
    --model vit_large_patch16 \
    --pretrain /path/to/pretrained/checkpoint.pth \
    --data_path hicfoundation_data \
    --train_config hicfoundation_data/train.txt \
    --valid_config hicfoundation_data/val.txt \
    --output finetune_task3_decoder_only \
    --batch_size 128 \
    --accum_iter 4 \
    --epochs 50 \
    --warmup_epochs 5 \
    --blr 1.5e-3 \
    --min_lr 0.0 \
    --weight_decay 0.05 \
    --layer_decay 0.75 \
    --input_row_size 48 \
    --input_col_size 48 \
    --patch_size 16 \
    --num_workers 8 \
    --print_freq 100 \
    --save_freq 5 \
    --tensorboard 1 \
    --finetune 1 \
    --loss_type 1 \
    --device cuda
```

### Option B: HiCFoundation Full Model Fine-tuning (GPU 1)

This script fine-tunes **HiCFoundation** with both encoder and decoder trainable (32 transformer blocks total, 512 attention heads):

```bash
# Edit the script to set your paths, then run:
bash finetune_task3_full_model.sh
```

Or manually:
```bash
export CUDA_VISIBLE_DEVICES=1

python finetune.py \
    --model vit_large_patch16 \
    --pretrain /path/to/pretrained/checkpoint.pth \
    --data_path hicfoundation_data \
    --train_config hicfoundation_data/train.txt \
    --valid_config hicfoundation_data/val.txt \
    --output finetune_task3_full_model \
    --batch_size 128 \
    --accum_iter 4 \
    --epochs 50 \
    --warmup_epochs 5 \
    --blr 1.5e-3 \
    --min_lr 0.0 \
    --weight_decay 0.05 \
    --layer_decay 0.75 \
    --input_row_size 48 \
    --input_col_size 48 \
    --patch_size 16 \
    --num_workers 8 \
    --print_freq 100 \
    --save_freq 5 \
    --tensorboard 1 \
    --finetune 2 \
    --loss_type 1 \
    --device cuda
```

### Run Both HiCFoundation Fine-tuning Scripts in Parallel

To run both HiCFoundation fine-tuning strategies simultaneously on different GPUs:

```bash
# Terminal 1 - Decoder-only fine-tuning
bash finetune_task3_decoder_only.sh

# Terminal 2 - Full model fine-tuning (in parallel)
bash finetune_task3_full_model.sh
```

Or use a job scheduler/system:

```bash
# Using nohup to run in background
nohup bash finetune_task3_decoder_only.sh > decoder_only.log 2>&1 &
nohup bash finetune_task3_full_model.sh > full_model.log 2>&1 &
```

## Hyperparameters Used

All hyperparameters follow HiCFoundation's default fine-tuning settings (documented in `HiCFoundation_Model_Report.md`):

### Model Configuration
- **Model**: `vit_large_patch16`
- **Input Size**: 48×48 (padded from 40×40)
- **Patch Size**: 16
- **Decoder Embedding Dim**: 512
- **Decoder Depth**: 8 blocks
- **Decoder Attention Heads**: 16

### Training Configuration
- **Batch Size**: 128 per GPU
- **Gradient Accumulation**: 4 iterations
- **Effective Batch Size**: 128 × 4 = 512
- **Epochs**: 50
- **Warmup Epochs**: 5
- **Base Learning Rate**: 1.5e-3
- **Min Learning Rate**: 0.0
- **Learning Rate Schedule**: Cosine decay with warmup
- **Weight Decay**: 0.05
- **Layer-wise LR Decay**: 0.75 (for full model fine-tuning)

### Loss Function
- **Loss Type**: 1 (MSE Loss)
- The model uses task=0 which outputs multiple components, but for resolution enhancement we focus on the 2D output loss

## Output

The fine-tuning scripts will create:
- `finetune_task3_decoder_only/` or `finetune_task3_full_model/`
  - `model/` - Checkpoint files
    - `checkpoint-*.pth.tar` - Periodic checkpoints
    - `model_best.pth.tar` - Best validation loss checkpoint
  - `log/` - Training/validation logs
  - `tensorboard/` - TensorBoard logs (if enabled)

## Monitoring Training

To monitor training with TensorBoard:

```bash
tensorboard --logdir finetune_task3_decoder_only/tensorboard
tensorboard --logdir finetune_task3_full_model/tensorboard
```

## Notes

1. **Data Padding**: 40×40 → 48×48 padding is minimal. If you prefer standard 224×224 input size, change `--target_size 224 224` in conversion and update input sizes in fine-tuning scripts.

2. **Task Configuration**: The scripts use the default fine-tuning task (task=0), which outputs multiple components. For resolution enhancement, the 2D output (`pred_2d`) is what we use, matching Task 3 functionality.

3. **Memory Requirements**: Full model fine-tuning requires more GPU memory than decoder-only. Adjust `--batch_size` and `--accum_iter` if needed.

4. **Pre-trained Model**: Make sure you have a pre-trained HiCFoundation checkpoint. Update the `PRETRAIN_MODEL` path in the scripts.

5. **Data Format**: The conversion script handles the normalization - HiCARN data (0-1 range) will be converted to HiCFoundation's expected format (log10 + RGB conversion happens in the dataset loader).

## Troubleshooting

1. **Dimension Mismatch**: Ensure input sizes are divisible by patch_size (16). 48×48 works, 40×40 doesn't.

2. **CUDA Out of Memory**: Reduce `--batch_size` or increase `--accum_iter` to maintain effective batch size.

3. **Data Loading Errors**: Verify `.pkl` files are created correctly and config files (`train.txt`, `val.txt`) point to correct directories.

4. **Model Loading Errors**: Check that pre-trained checkpoint path is correct and checkpoint format matches expected structure.

