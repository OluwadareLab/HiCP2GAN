# Plug&Play GAN Architecture for Hi-C Resolution Enhancement

## Model Architecture Diagram

```mermaid
graph TB
    subgraph "Input Data Pipeline"
        LR40[LR Tiles<br/>40×40×1]
        HR40[HR Tiles<br/>40×40×1]
        LR224[LR Global<br/>224×224×1]
        HR224[HR Global<br/>224×224×1]
    end
    
    subgraph "Generator Network (G)"
        G[HiCARN-1 Generator<br/>Cascading Residual Network<br/>40×40 → 40×40]
        SR40[SR Tiles<br/>40×40×1]
    end
    
    subgraph "Local Discriminator (D_local)"
        DL[PatchGAN Discriminator<br/>40×40 → Real/Fake]
        DL_real[Real Score]
        DL_fake[Fake Score]
    end
    
    subgraph "Global Discriminator (D_global)"
        HF[HiCFoundation Encoder<br/>Frozen Backbone<br/>ViT-Large]
        GH[GAN Head<br/>Trainable]
        DG[224×224 → Real/Fake]
        DG_real[Real Score]
        DG_fake[Fake Score]
        FM[Feature Matching<br/>Intermediate Features]
    end
    
    subgraph "Loss Computation"
        L_rec[Reconstruction Loss<br/>L1/Huber on Tiles]
        L_adv_local[Adversarial Loss Local<br/>BCE on D_local]
        L_adv_global[Adversarial Loss Global<br/>BCE on D_global]
        L_fm[Feature Matching Loss<br/>L1 on HiCFoundation Features]
        L_total[Total Generator Loss<br/>λ_rec·L_rec + λ_adv_l·L_adv_l<br/>+ λ_adv_g·L_adv_g + λ_fm·L_fm]
    end
    
    subgraph "Stitching Module"
        STITCH[Tile Stitching<br/>36 tiles → 224×224]
        SR224[SR Global<br/>224×224×1]
    end
    
    %% Data flow
    LR40 --> G
    G --> SR40
    SR40 --> STITCH
    STITCH --> SR224
    
    %% Discriminator paths
    HR40 --> DL
    SR40 -.detach.-> DL
    DL --> DL_real
    DL --> DL_fake
    
    HR224 --> HF
    SR224 -.detach.-> HF
    HF --> GH
    GH --> DG
    DG --> DG_real
    DG --> DG_fake
    HF --> FM
    
    %% Loss computation
    SR40 --> L_rec
    HR40 --> L_rec
    DL_fake --> L_adv_local
    DG_fake --> L_adv_global
    FM --> L_fm
    
    L_rec --> L_total
    L_adv_local --> L_total
    L_adv_global --> L_total
    L_fm --> L_total
    
    %% Styling
    classDef generator fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef discriminator fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef loss fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef data fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    
    class G,SR40 generator
    class DL,DG,HF,GH discriminator
    class L_rec,L_adv_local,L_adv_global,L_fm,L_total loss
    class LR40,HR40,LR224,HR224,SR224 data
```

## Detailed Architecture Description

### 1. Generator (G): HiCARN-1
- **Architecture**: Cascading Residual Network
- **Input**: Low-resolution tiles (40×40×1)
- **Output**: Super-resolution tiles (40×40×1)
- **Components**:
  - Entry 3×3 convolution
  - 5 Cascading Blocks (each with 3 Residual Blocks)
  - Body 1×1 convolutions for feature fusion
  - Exit 3×3 convolution

### 2. Local Discriminator (D_local)
- **Type**: PatchGAN discriminator
- **Input**: 40×40×1 tiles
- **Output**: Real/Fake probability map
- **Architecture**: 
  - 4 convolutional blocks with BatchNorm and LeakyReLU
  - Final 3×3 convolution for binary classification

### 3. Global Discriminator (D_global)
- **Backbone**: HiCFoundation ViT-Large encoder (Frozen)
  - Vision Transformer with 24 layers
  - Embedding dimension: 1024
  - Patch size: 16×16
  - Input: 224×224×3 (RGB formatted)
- **Head**: Lightweight GAN head (Trainable)
  - Linear layers: 1024 → 256 → 1
  - Output: Real/Fake logits

### 4. Loss Functions

#### Generator Loss:
```
L_G = λ_rec · L_rec + λ_adv_local · L_adv_local + λ_adv_global · L_adv_global + λ_fm · L_fm
```

Where:
- **L_rec**: Huber loss between SR and HR tiles
- **L_adv_local**: BCE loss encouraging D_local(SR_tiles) → real
- **L_adv_global**: BCE loss encouraging D_global(SR_global) → real
- **L_fm**: L1 loss between HiCFoundation features of SR_global and HR_global

#### Discriminator Losses:
- **D_local**: BCE loss distinguishing real vs fake 40×40 tiles
- **D_global**: BCE loss distinguishing real vs fake 224×224 crops

### 5. Training Strategy

1. **Warmup Phase** (epochs 1-5):
   - Only reconstruction loss (λ_adv_local = λ_adv_global = 0)
   - Feature matching enabled

2. **Full Training**:
   - All losses active
   - Gradient clipping (max_norm = 0.5)
   - Adversarial loss clipping (max = 10.0)

### 6. Data Flow

```
LR Tiles (40×40) → Generator → SR Tiles (40×40)
                                    ↓
                              Stitch (6×6 grid)
                                    ↓
                            SR Global (224×224)
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
            D_local (40×40)              D_global (224×224)
                    ↓                               ↓
            Adversarial Loss              Adversarial Loss + Feature Matching
```

## Key Features

- **Multi-scale supervision**: Both local (40×40) and global (224×224) discriminators
- **Pre-trained foundation model**: Leverages HiCFoundation's learned representations
- **Feature matching**: Ensures global structure consistency using intermediate features
- **Stable training**: Gradient clipping, loss clipping, and warmup phase
