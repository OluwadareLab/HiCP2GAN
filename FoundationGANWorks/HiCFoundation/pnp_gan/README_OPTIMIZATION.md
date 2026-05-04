# GPU Memory Optimization Guide

## For 24GB GPU Memory (Docker Container)

### Recommended Settings

Since Docker containers have limited shared memory, use these settings for maximum GPU efficiency:

```bash
python train.py \
    --data_dir_40 /path/to/data_40_40 \
    --data_dir_224 /path/to/data_224 \
    --hicfoundation_path /path/to/model.pth.tar \
    --batch_size 32 \          # Increase this to max out GPU memory
    --use_amp \                 # Mixed precision (saves ~50% memory, 2x speedup)
    --num_workers 0 \           # Required in Docker to avoid shared memory issues
    --device cuda:0
```

### Key Points:

1. **Batch Size**: Start with 16-32 and increase until you get OOM errors. With mixed precision, you can typically use 2x larger batches.

2. **Mixed Precision (AMP)**: **Enabled by default** - This gives you:
   - ~50% memory reduction
   - ~2x training speedup
   - Minimal accuracy loss

3. **num_workers=0**: Required in Docker to avoid shared memory issues. The slight overhead is minimal compared to GPU computation time.

4. **Finding Maximum Batch Size**:
   ```bash
   python find_optimal_batch_size.py \
       --data_dir_40 /path/to/data_40_40 \
       --data_dir_224 /path/to/data_224 \
       --hicfoundation_path /path/to/model.pth.tar \
       --start_batch 16
   ```

### Memory Usage Estimates:

- **Batch size 16**: ~8-10GB (with mixed precision)
- **Batch size 32**: ~14-16GB (with mixed precision)
- **Batch size 48**: ~20-22GB (with mixed precision) - Try this for 24GB GPU

### Alternative: Increase Docker Shared Memory

If you want to use `num_workers > 0`, you can increase Docker shared memory:

```bash
docker run --shm-size=8g ...  # Increase shared memory to 8GB
```

Or mount `/dev/shm`:
```bash
docker run -v /dev/shm:/dev/shm --shm-size=8g ...
```

However, `num_workers=0` is usually fine since data loading is fast and GPU computation is the bottleneck.

### Monitoring GPU Usage

Monitor GPU memory during training:
```bash
watch -n 1 nvidia-smi
```

Adjust batch size to use 90-95% of GPU memory for maximum efficiency.
