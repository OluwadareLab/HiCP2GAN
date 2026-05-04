#!/bin/bash
# Quick start script for Plug&Play GAN training

# Set paths
DATA_DIR_40="/storage/solowofi/HiCARN/Data/R16_down/GM12878/data_40_40"
DATA_DIR_224="/storage/solowofi/HiCARN/Data/R16_down/GM12878/data_224"
HICFOUNDATION_PATH="/storage/solowofi/HiCARN/FoundationGANWorks/HiCFoundation/hicfoundation_model/hicfoundation_resolution.pth.tar"
CHECKPOINT_DIR="checkpoints/pnp_gan"
LOG_DIR="logs/pnp_gan"

# Create directories
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

# Run training
python train.py \
    --data_dir_40 $DATA_DIR_40 \
    --data_dir_224 $DATA_DIR_224 \
    --hicfoundation_path $HICFOUNDATION_PATH \
    --checkpoint_dir $CHECKPOINT_DIR \
    --log_dir $LOG_DIR \
    --num_epochs 100 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --use_amp \
    --num_workers 0 \
    --lr_G 0.0001 \
    --lr_D 0.00005 \
    --lambda_rec 1.0 \
    --lambda_adv_local 0.005 \
    --lambda_adv_global 0.005 \
    --lambda_fm 0.1 \
    --warmup_epochs 5 \
    --device cuda:0

# Plot results after training (commented out - run manually after training completes)
# python plot_results.py \
#     --log_file $LOG_DIR/training_history.txt \
#     --save_dir $LOG_DIR
