# HiCFoundation Model Documentation

## Table of Contents
1. [Model Overview](#model-overview)
2. [Model Architecture](#model-architecture)
3. [Hyperparameters](#hyperparameters)
4. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
5. [Training Details](#training-details)
6. [Inference Details](#inference-details)

---

## Model Overview

HiCFoundation is a Masked Autoencoder (MAE) based on Vision Transformer (ViT) architecture, specifically designed for Hi-C contact map analysis. The model follows a self-supervised pre-training approach followed by task-specific fine-tuning.

**Key Features:**
- Vision Transformer backbone with encoder-decoder architecture
- Masked autoencoding for self-supervised pre-training
- Positional embeddings for spatial understanding
- Count embeddings for Hi-C experiment normalization
- Multiple fine-tuning tasks support

---

## Model Architecture

### Overall Structure

HiCFoundation consists of two main components:
1. **Encoder**: Processes visible (unmasked) patches
2. **Decoder**: Reconstructs the full image from encoded patches and mask tokens

### Encoder Architecture

The encoder is based on Vision Transformer (ViT) with the following structure:

**Default Configuration (vit_large_patch16):**
- **Patch Size**: 16×16 pixels
- **Input Channels**: 3 (RGB representation of Hi-C data)
- **Embedding Dimension**: 1024
- **Depth (Number of Transformer Blocks)**: 24
- **Number of Attention Heads**: 16
- **MLP Ratio**: 4.0
- **Normalization**: LayerNorm (eps=1e-6)

**Components:**
1. **Patch Embedding Layer** (`PatchEmbed`)
   - Converts input image patches into embeddings
   - Input: `(N, 3, H, W)` → Output: `(N, num_patches, embed_dim)`
   - Uses 2D convolution to extract patches

2. **Positional Embeddings**
   - Fixed sine-cosine positional embeddings (2D rectangular)
   - Supports rectangular inputs (different height and width)
   - Size: `(1, num_patches + 1, embed_dim)` for encoder
   - Includes position for CLS token

3. **CLS Token**
   - Learnable classification token
   - Initialized with normal distribution (std=0.02)

4. **Count Embedding**
   - Converts total Hi-C count to positional embedding
   - Uses log10 normalization: `log10(total_count)`
   - Encoded using sine-cosine embedding (similar to positional embeddings)
   - Adds +1 offset to distinguish from other embeddings
   - Dimension: matches `embed_dim`

5. **Transformer Blocks** (24 layers)
   - Standard ViT Block from timm library
   - Includes Multi-Head Self-Attention and MLP
   - QKV bias: True
   - QK scale: None (uses default sqrt scaling)

6. **Final Normalization**
   - LayerNorm applied after all transformer blocks

### Decoder Architecture

The decoder reconstructs the full image from encoded patches:

**Default Configuration:**
- **Decoder Embedding Dimension**: 512
- **Decoder Depth**: 8 transformer blocks
- **Decoder Attention Heads**: 16
- **MLP Ratio**: 4.0

**Components:**
1. **Decoder Embedding**
   - Linear projection: `embed_dim → decoder_embed_dim`
   - Projects encoder outputs to decoder dimension

2. **Mask Token**
   - Learnable mask token for masked patches
   - Dimension: `decoder_embed_dim`
   - Initialized with normal distribution (std=0.02)

3. **Decoder Positional Embeddings**
   - Fixed sine-cosine embeddings for decoder
   - Size: `(1, num_patches, decoder_embed_dim)`
   - No CLS token in decoder

4. **Decoder Transformer Blocks** (8 layers)
   - Same structure as encoder blocks
   - Processes full sequence (visible + mask tokens)

5. **Decoder Prediction Head**
   - **Patch Reconstruction**: Linear layer `decoder_embed_dim → patch_size² × in_chans`
   - Reconstructs pixel values for each patch
   - For default config: `512 → 16² × 3 = 768`

6. **Count Prediction Head**
   - Linear layer: `decoder_embed_dim → 1`
   - Predicts log10 of submatrix total count

### Fine-tuning Model Head

For fine-tuning, the model uses `Finetune_Model_Head` which:
- Reuses the pre-trained encoder (vit_backbone)
- Adds task-specific decoder heads
- Supports multiple tasks:
  - Task 0: Fine-tuning (2D + 1D outputs + embedding)
  - Task 1: Reproducibility analysis
  - Task 2: Loop calling
  - Task 3: Resolution enhancement
  - Task 4: Epigenomic assay prediction (multi-track output)
  - Task 5: scHi-C enhancement
  - Task 6: Embedding analysis
  - Task 7: Reconstruction visualization (pre-training only)

**Fine-tuning Decoder Configuration:**
- **Decoder Embedding Dimension**: 512
- **Decoder Depth**: 8
- **Decoder Attention Heads**: 16
- **MLP Ratio**: 4.0

---

## Hyperparameters

### Pre-training Hyperparameters

#### Model Architecture
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `model` | `vit_large_patch16` | Model architecture variant |
| `patch_size` | 16 | Size of image patches |
| `embed_dim` | 1024 | Encoder embedding dimension |
| `depth` | 24 | Number of encoder transformer blocks |
| `num_heads` | 16 | Number of attention heads in encoder |
| `decoder_embed_dim` | 512 | Decoder embedding dimension |
| `decoder_depth` | 8 | Number of decoder transformer blocks |
| `decoder_num_heads` | 16 | Number of attention heads in decoder |
| `mlp_ratio` | 4.0 | MLP expansion ratio |
| `in_chans` | 3 | Input channels (RGB) |

#### Input/Output Dimensions
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `input_row_size` | 224 | Height of input submatrix |
| `input_col_size` | 224 | Width of input submatrix (for pretrain) |
| `img_size` | (224, 224) | Input image dimensions |

#### Training Configuration
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `batch_size` | 128 | Batch size per GPU |
| `accum_iter` | 4 | Gradient accumulation iterations |
| `epochs` | 100 | Total training epochs |
| `warmup_epochs` | 10 | Warmup epochs for learning rate |
| `start_epoch` | 0 | Starting epoch (for resume) |
| `mask_ratio` | 0.75 | Ratio of patches to mask (75%) |

#### Optimizer Settings
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `blr` | 1.5e-3 | Base learning rate |
| `lr` | Calculated | Learning rate = `blr × eff_batch_size / 256` |
| `min_lr` | 0.0 | Minimum learning rate for decay |
| `weight_decay` | 0.05 | Weight decay coefficient |
| `optimizer` | AdamW | Optimizer type |
| `betas` | (0.9, 0.95) | AdamW beta parameters |

**Effective Batch Size Calculation:**
```
eff_batch_size = batch_size × accum_iter × world_size
lr = blr × eff_batch_size / 256
```

#### Loss Function
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `loss_alpha` | 1.0 | Weight for SSIM and count losses |
| Loss Components:
  - **SSIM Loss**: `1 - SSIM(pred_image, target_image)`
  - **Count Loss**: `MSE(log10(count_pred), log10(matrix_count))`
  - **Contrastive Loss**: Patch-wise contrastive loss (normalized patches)

**Total Loss:**
```
loss = loss_alpha × (ssim_loss + count_loss) + contrastive_loss
```

#### Data Preprocessing
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `sparsity_ratio` | 0.05 | Minimum sparsity threshold (5% non-zero) |
| `num_workers` | 8 | Data loading workers per GPU |
| `pin_mem` | False | Pin memory (optional) |

#### Learning Rate Schedule
- **Warmup Phase**: Linear warmup from 0 to `lr` over `warmup_epochs`
- **Cosine Decay**: After warmup, cosine decay to `min_lr`
- Formula:
  ```python
  if epoch < warmup_epochs:
      lr = lr * epoch / warmup_epochs
  else:
      lr = min_lr + (lr - min_lr) * 0.5 * (1 + cos(π * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
  ```

#### Other Settings
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `seed` | 888 | Random seed |
| `save_freq` | 1 | Frequency of checkpoint saving |
| `print_freq` | 1 | Frequency of logging |
| `tensorboard` | 0 | Enable TensorBoard (0/1) |

### Fine-tuning Hyperparameters

#### Model Configuration
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `model` | `vit_large_patch16` | Backbone model |
| `patch_size` | 16 | Patch size |
| `input_row_size` | 224 | Input height |
| `input_col_size` | 4000 | Input width (can vary by task) |
| `decoder_embed_dim` | 512 | Decoder embedding dimension |
| `decoder_depth` | 8 | Decoder depth |
| `decoder_num_heads` | 16 | Decoder attention heads |
| `mlp_ratio` | 4.0 | MLP ratio |
| `finetune` | 0 | Fine-tuning mode (1: decoder only, 2: full model) |

#### Training Configuration
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `batch_size` | 128 | Batch size per GPU |
| `accum_iter` | 4 | Gradient accumulation iterations |
| `epochs` | 50 | Total training epochs |
| `warmup_epochs` | 5 | Warmup epochs |
| `start_epoch` | 0 | Starting epoch |

#### Optimizer Settings
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `blr` | 1.5e-3 | Base learning rate |
| `lr` | Calculated | Learning rate = `blr × eff_batch_size / 256` |
| `min_lr` | 0.0 | Minimum learning rate |
| `weight_decay` | 0.05 | Weight decay |
| `layer_decay` | 0.75 | Layer-wise learning rate decay factor |
| `optimizer` | AdamW | Optimizer |
| `betas` | (0.9, 0.95) | AdamW betas |

**Layer-wise Learning Rate Decay:**
- Different layers have different learning rates
- Scale: `layer_decay^(num_layers - layer_id)`
- Later layers (higher IDs) get lower learning rates

#### Loss Function
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `loss_type` | 0 | Loss type (1: MSE, 2: Cosine) |

**Fine-tuning Loss Components:**
- **Embedding Loss**: Cosine/MSE between predicted and target embeddings
- **2D Output Loss**: Cosine/MSE for 2D matrix prediction
- **1D Output Loss**: Cosine/MSE for 1D vector prediction

**Total Loss:**
```
loss = embedding_loss + output_2d_loss + output_1d_loss
```
(Each component can be 0 if corresponding target is None)

#### Other Settings
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `seed` | 888 | Random seed |
| `save_freq` | 1 | Checkpoint save frequency |
| `print_freq` | 1 | Logging frequency |
| `num_workers` | 8 | Data loading workers |
| `tensorboard` | 0 | TensorBoard logging (0/1) |

### Inference Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `batch_size` | 128 | Inference batch size |
| `stride` | 20 | Sliding window stride |
| `bound` | 200 | Off-diagonal bound for scanning |
| `task` | 0 | Inference task ID |
| `max_cutoff` | None | Maximum value clipping |
| `fill_diagonal_zero` | False | Fill diagonal with zeros |
| `num_workers` | 8 | Data loading workers |
| `print_freq` | 1 | Logging frequency |

---

## Data Preprocessing Pipeline

### Pre-training Data Preprocessing

**Input Format:**
- `.pkl` files containing dictionaries with keys:
  - `'input'`: Hi-C matrix (numpy array or scipy.sparse.coo_matrix)
  - `'input_count'`: Total count of Hi-C experiment (optional, float)
  - `'diag'`: Diagonal starting index (optional, int or None)

**Preprocessing Steps:**

1. **Data Loading**
   - Load `.pkl` files from specified directories
   - Validate input matrix exists
   - Check input size meets window requirements

2. **Sparsity Filtering**
   - Calculate sparsity: `non_zero_count / total_elements`
   - Skip samples with sparsity < `sparsity_ratio` (default: 0.05)
   - Re-sample if sparsity is too low

3. **Submatrix Sampling**
   - Extract random submatrix of size `(window_height, window_width)`
   - Default size: 224×224
   - Supports diagonal-aware sampling:
     - If `diag` is provided, samples near diagonal regions
     - Ensures diagonal boundaries align with patch boundaries (multiple of `patch_size`)

4. **Mask Generation**
   - Create binary mask: 1 for valid (non-zero) pixels, 0 for zero pixels
   - Shape: `(1, H, W)`

5. **Log10 Transformation**
   - Apply log10: `log10(input + 1)`
   - Normalizes Hi-C contact values
   - Calculates max_value for normalization

6. **RGB Conversion**
   - Convert 2D Hi-C matrix to 3-channel RGB image:
     - **Red channel**: All ones (constant)
     - **Green channel**: `(max_value - data_log) / max_value`
     - **Blue channel**: `(max_value - data_log) / max_value` (same as green)
   - Result shape: `(H, W, 3)` (channel-last format)

7. **ImageNet Normalization**
   - Convert to tensor and apply ImageNet normalization:
     - Mean: `[0.485, 0.456, 0.406]`
     - Std: `[0.229, 0.224, 0.225]`
   - Final shape: `(3, H, W)` (channel-first)

8. **Count Processing**
   - If `input_count` exists, use it
   - Otherwise, use placeholder: 1,000,000,000
   - Convert to tensor format

9. **Diagonal Position Processing**
   - Convert diagonal position from pixel units to patch units
   - Formula: `patch_diag = pixel_diag / patch_size`

**Output:**
- `input_matrix`: `(3, 224, 224)` - Normalized RGB image
- `mask_matrix`: `(1, 224, 224)` - Binary mask
- `hic_count`: Scalar - Total Hi-C count
- `return_diag`: Scalar - Diagonal position in patch units
- `matrix_count`: Scalar - Sum of submatrix values

### Fine-tuning Data Preprocessing

**Input Format:**
- `.pkl` files with dictionaries containing:
  - `'input'`: Input Hi-C matrix (required)
  - `'input_count'`: Total count (optional)
  - `'2d_target'`: 2D target matrix (optional)
  - `'embed_target'`: Embedding target vector (optional)
  - `'1d_target'`: 1D target vector (optional)

**Preprocessing Steps:**

1. **Data Loading**
   - Load `.pkl` files
   - Validate input and at least one target exists
   - Check input size matches expected dimensions exactly

2. **Input Matrix Processing**
   - Convert sparse to dense if needed
   - Replace NaN with 0
   - Convert to float32
   - Apply log10: `log10(input + 1)`
   - Calculate max_value

3. **RGB Conversion**
   - Same as pre-training:
     - Red: all ones
     - Green/Blue: `(max_value - data_log) / max_value`

4. **ImageNet Normalization**
   - Same normalization as pre-training

5. **Target Processing**
   - **2D Target**: Convert sparse to dense, NaN to 0, float32
   - **1D Target**: NaN to 0, float32
   - **Embed Target**: NaN to 0, float32

6. **Count Processing**
   - Use `input_count` if available, else None

**Output:**
- `input_matrix`: `(3, H, W)` - Normalized RGB image
- `total_count`: Scalar or None - Hi-C count
- `target_matrix`: `(H, W)` or None - 2D target
- `embed_target`: `(D,)` or None - Embedding target
- `target_vector`: `(L,)` or None - 1D target

**Note:** Input size must exactly match `(input_row_size, input_col_size)` - no random cropping.

### Inference Data Preprocessing

**Input Format:**
- `.hic`, `.cool`, `.pkl`, `.txt`, `.pairs`, or `.npy` files
- For `.pkl`: Dictionary with chromosome keys → sparse matrices

**Preprocessing Steps:**

1. **Data Loading**
   - Load Hi-C data file
   - Extract chromosome-specific matrices
   - Convert to sparse COO format if needed

2. **Symmetrization**
   - For square matrices, create symmetric version:
     - Combine upper and lower triangular parts
     - Divide diagonal by 2 to avoid double counting

3. **Padding (if needed)**
   - Pad matrices to at least `(window_height, window_width)`
   - For task 5 (scHi-C), pad around center

4. **Sliding Window Generation**
   - Generate window positions based on:
     - `stride`: Step size for sliding window
     - `bound`: Maximum off-diagonal distance
     - `locus_embedding`: If True, use locus-centered windows
   - Create list of `(chromosome, row_start, col_start, row_end, col_end, middle_point)`

5. **Window Extraction**
   - For each window position:
     - Extract submatrix from sparse data
     - Convert to dense array
     - Zero-pad to exact window size if needed
     - Apply diagonal zero-filling if `fill_diagonal_zero=True`

6. **Value Clipping**
   - If `max_cutoff` specified, clip values: `min(input, max_cutoff)`

7. **Log10 Transformation**
   - Apply: `log10(input + 1)`

8. **RGB Conversion**
   - Same as pre-training/fine-tuning

9. **ImageNet Normalization**
   - Same normalization

**Output:**
- `input_matrix`: `(3, H, W)` - Normalized RGB image
- `total_count`: Scalar - Total reads in input
- `[chrom, row_start, col_start]`: Location information

**Sliding Window Modes:**

1. **Standard Mode** (task ≠ 6):
   - Extract windows with stride
   - Skip windows where `|row_start - col_start| > bound`

2. **Locus Embedding Mode**:
   - Extract windows centered on diagonal
   - Column position determined by row position

3. **Embedding Mode** (task = 6):
   - Extract overlapping windows
   - Center point-based extraction

---

## Training Details

### Pre-training Training Loop

1. **Forward Pass:**
   - Input: `(N, 3, H, W)` image, mask, total_count, diagonal position
   - Encoder processes visible patches (25% with 75% masking)
   - Decoder reconstructs full image
   - Outputs: SSIM loss, contrastive loss, count prediction, reconstructed image, mask

2. **Loss Calculation:**
   - SSIM Loss: `1 - SSIM(pred_image, target_image)`
   - Count Loss: `MSE(log10(count_pred), log10(matrix_count))`
   - Contrastive Loss: Patch-wise normalized contrastive loss
   - Total: `loss_alpha × (ssim_loss + count_loss) + contrastive_loss`

3. **Backward Pass:**
   - Mixed precision training (autocast)
   - Gradient accumulation over `accum_iter` steps
   - Gradient clipping (optional)
   - Optimizer step

4. **Learning Rate Schedule:**
   - Linear warmup for `warmup_epochs`
   - Cosine decay to `min_lr` after warmup

5. **Masking Strategy:**
   - Random masking: 75% of patches masked
   - Symmetric masking: If diagonal provided, mask symmetrically around diagonal
   - Mask tokens added in decoder

### Fine-tuning Training Loop

1. **Forward Pass:**
   - Input: `(N, 3, H, W)` image, total_count
   - Encoder extracts features
   - Task-specific decoder head produces outputs
   - Outputs depend on task (embedding, 2D, 1D)

2. **Loss Calculation:**
   - Multiple loss components (only active if target provided):
     - Embedding loss
     - 2D output loss (flattened)
     - 1D output loss
   - Total: Sum of active losses

3. **Backward Pass:**
   - Standard backward pass (no masking)
   - Gradient accumulation
   - Layer-wise learning rate decay applied

4. **Learning Rate Schedule:**
   - Same as pre-training (warmup + cosine decay)

5. **Model Freezing:**
   - Option 1: Freeze encoder, train decoder only
   - Option 2: Train full model

### Validation

- Same forward pass as training
- No gradient computation
- Metrics logged (loss components)
- Visualizations saved to TensorBoard (optional)

---

## Inference Details

### Inference Process

1. **Data Loading:**
   - Load Hi-C data file
   - Generate sliding window positions
   - Extract windows sequentially or in batches

2. **Model Forward:**
   - Process each window through model
   - Extract task-specific outputs

3. **Output Aggregation:**
   - Aggregate predictions across windows
   - Handle overlaps (average or max)
   - Reconstruct full chromosome matrices

4. **Output Format:**
   - Depends on task:
     - Task 1: Embedding vectors
     - Task 2/3/5: Enhanced Hi-C matrices
     - Task 4: Multi-track predictions
     - Task 6: Embedding maps

### Position Embedding Interpolation

When input size differs from pre-training:
- Interpolate positional embeddings using bicubic interpolation
- Supports rectangular inputs
- Applied to both encoder and decoder embeddings

---

## Key Implementation Details

### Positional Embeddings

1. **2D Sine-Cosine Embeddings:**
   - Half dimensions for height, half for width
   - Formula: `sin/cos(ω × pos)` where `ω = 1 / 10000^(2i/d)`
   - Fixed (not learnable)

2. **Count Embeddings:**
   - Convert log10(count) to sine-cosine embedding
   - Adds +1 offset for distinction
   - Same dimension as positional embeddings

### Masking Mechanism

1. **Random Masking:**
   - Generate random noise for each patch
   - Sort patches by noise value
   - Keep top 25% (lowest noise), mask 75%

2. **Symmetric Masking:**
   - If diagonal provided, apply symmetric noise
   - Ensures symmetric regions have similar masking patterns

### Loss Functions

1. **SSIM Loss:**
   - Structural Similarity Index
   - Uses Gaussian window (size=11, sigma=1.5)
   - Data range: 1.0 (after normalization)
   - K constants: (0.01, 0.03)

2. **Contrastive Loss:**
   - Normalize patches (L2 normalization)
   - Compute similarity matrix
   - Cross-entropy loss for matching patches

3. **Count Loss:**
   - MSE between predicted and actual log10 counts

### Initialization

- **Patch Embedding**: Xavier uniform initialization
- **CLS Token**: Normal distribution (std=0.02)
- **Mask Token**: Normal distribution (std=0.02)
- **Linear Layers**: Xavier uniform initialization
- **LayerNorm**: Bias=0, Weight=1.0
- **Positional Embeddings**: Fixed sine-cosine (not learnable)

---

## Model Variants

Currently supported:
- **vit_large_patch16**: Default configuration (described above)

Vision Transformer variants (for backbone):
- **vit_base_patch16**: embed_dim=768, depth=12, heads=12
- **vit_large_patch16**: embed_dim=1024, depth=24, heads=16
- **vit_huge_patch14**: embed_dim=1280, depth=32, heads=16, patch_size=14

---

## References

The implementation references:
- **timm**: PyTorch Image Models (Vision Transformer implementation)
- **DeiT**: Data-Efficient Image Transformers
- **MAE**: Masked Autoencoders (Facebook Research)
- **AdPE**: Adaptive Positional Embeddings

---

## File Structure

Key model files:
- `model/models_hicfoundation.py`: Main pre-training model
- `model/Vision_Transformer_count.py`: Vision Transformer backbone with count embedding
- `model/Finetune_Model_Head.py`: Fine-tuning head
- `model/pos_embed.py`: Positional embedding utilities
- `model/SSIM.py`: SSIM loss implementation
- `model/lr_sched.py`: Learning rate scheduling
- `model/lr_decay.py`: Layer-wise learning rate decay
- `model/model_utils.py`: Model utilities (save/load)
- `model/NativeScaler.py`: Mixed precision training scaler

Data processing:
- `data_processing/pretrain_dataset.py`: Pre-training dataset
- `data_processing/finetune_dataset.py`: Fine-tuning dataset
- `data_processing/inference_dataset.py`: Inference dataset
- `data_processing/collate_fn.py`: Batch collation

Training:
- `pretrain/main_worker.py`: Pre-training main
- `pretrain/train_epoch.py`: Pre-training epoch loop
- `pretrain/val_epoch.py`: Pre-training validation
- `finetune/main_worker.py`: Fine-tuning main
- `finetune/train_epoch.py`: Fine-tuning epoch loop
- `finetune/val_epoch.py`: Fine-tuning validation
- `finetune/loss.py`: Fine-tuning loss functions

