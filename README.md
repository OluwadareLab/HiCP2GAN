# HiCP2GAN: A plug-and-play foundation model–based GAN for Hi-C enhancement

**HiCP2GAN** is a plug-and-play adversarial framework for Hi-C super-resolution: a **trainable generator** from `Models/` is paired with a **HiCFoundation** vision transformer used as the **discriminator** (`HiCP2GAN_train.py`). This repository ships **standalone data preprocessing** (`data/`), **training**, and **40×40 inference** (`predict_40x40.py`).

Bioinformatics Lab, University of Colorado Colorado Springs

### Developer

**Samuel Olowofila**  

Department of Computer Science  

University of Colorado Colorado Springs, Colorado Springs, CO  

Email: [solowofi@uccs.edu](mailto:solowofi@uccs.edu)

### Contact

**Oluwatosin Oluwadare, PhD**  

Department of Computer Science  

University of North Texas, Denton TX  

Email: [oluwatosin.oluwadare@unt.edu](mailto:oluwatosin.oluwadare@unt.edu)

---

## Overview

- **Environment:** install dependencies locally (**Build instructions**) or use the published Docker image **`oluwadarelab/hicp2gan:latest`** (see **Docker image** under Build instructions).
- **Preprocessing:** three independent scripts (`Read_Data.py`, `Downsample.py`, `Generate.py`) under `data/`, plus an optional multi-cell driver in `scripts/`.
- **Training:** `HiCP2GAN_train.py` optimizes the generator and foundation-based discriminator on paired LR/HR patch `.npz` files.
- **Inference:** `predict_40x40.py` loads a saved **generator** checkpoint and writes chromosome-level predictions under `predict/`.
- **Weights:** pretrained checkpoints are **not** on GitHub; download the bundle from [Zenodo](https://zenodo.org/uploads/20030290) into `checkpoints/` (see **Pretrained checkpoints** below).

---

## Pretrained checkpoints (`checkpoints/`)

The GitHub repository **does not include** trained weights (they are large and would exceed normal hosting limits). After cloning, download the curated checkpoint bundle from **Zenodo** and unpack it so a `checkpoints/` directory sits at the **root of this repository** (next to `README.md`, `Models/`, etc.):

**Download:** [Zenodo — HiCP2GAN checkpoints](https://zenodo.org/uploads/20030290)

Total size is typically on the order of **~7 GB** (includes HiCFoundation `.pth.tar` files). Pass paths under **`checkpoints/`** to **`--foundation_ckpt`** (training) and **`-ckpt`** (inference); resolution-specific bundles use **`checkpoints/R16/`** and **`checkpoints/R64/`** with subfolders for conventional GANs, HiCP2GAN runs, standalone generators where present, **`pretrained/`**, and optional experiment-specific directories (e.g. Capricorn, misc).

---

## Build instructions

1. **Clone** this repository and enter the directory (use your published Git URL):
   ```bash
   git clone <repository-url> && cd HiCP2GAN
   ```

2. **Create a virtual environment** (recommended) and install dependencies:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set `PYTHONPATH`** so `Models`, `Utils`, and `data` resolve when you run scripts from any working directory:
   ```bash
   export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
   ```

4. **Optional:** define where data live (overrides defaults in `data/Arg_Parser.py`):
   ```bash
   export HICP2GAN_DATA_ROOT="/path/to/your/data/tree"
   export HICARN_DATA_ROOT="$HICP2GAN_DATA_ROOT"   # legacy alias, optional
   ```

### Docker image (reproducible environment)

HiCP2GAN is also distributed as a **Docker** image on Docker Hub. For a containerized setup (CUDA + dependencies pre-installed), follow the steps below instead of—or after—cloning the repository on the host.

1. **Clone** this repository and `cd` into it (same as step 1 above), so your working tree is available to mount into the container.

2. **Pull** the HiCP2GAN image from Docker Hub:
   ```bash
   docker pull oluwadarelab/hicp2gan:latest
   ```

3. **Verify** the image is present:
   ```bash
   docker image ls | grep hicp2gan
   ```

4. **Run** a container with the current directory mounted (adjust GPU flags if you do not use NVIDIA GPUs). The example below mounts `${PWD}` at the same path inside the container and sets the working directory so relative paths match the host:
   ```bash
   docker run --rm -it --gpus all --name hicp2gan -v "${PWD}:${PWD}" -w "${PWD}" oluwadarelab/hicp2gan:latest
   ```

   Inside the shell, set `PYTHONPATH` and data locations if they are not already configured in the image:
   ```bash
   export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
   export HICP2GAN_DATA_ROOT="/path/to/your/data/tree"   # mount extra volumes if data live outside this clone
   ```

   Then run preprocessing, training, or inference as documented below (e.g. `python data/Read_Data.py …`, `python HiCP2GAN_train.py …`, `python predict_40x40.py …`). **Weights are not in Git:** unpack the [Zenodo checkpoint bundle](https://zenodo.org/uploads/20030290) into `./checkpoints/` on the host before training or inference; with `-v "${PWD}:${PWD}"`, that folder appears at the same path inside the container.

**Notes:** `--gpus all` requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). For CPU-only hosts, omit `--gpus all` (training will be slow). If your Hi-C data live outside the clone, add another `-v /host/data:/host/data` and point `HICP2GAN_DATA_ROOT` at that path.

---

## Dependencies

Recommended stack (see `requirements.txt` for minimum versions):

- Python ≥ 3.9
- PyTorch ≥ 2.0, torchvision ≥ 0.15
- NumPy, SciPy, pandas, tqdm
- Matplotlib, scikit-learn, timm *(training / HiCFoundation discriminator)*

GPU and a suitable CUDA driver are strongly recommended for training.

---

## Data sources (Rao et al.)

Intrachromosomal Hi-C matrices from **Rao et al., 2014** are available under GEO [**GSE63525**](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525). Examples used in related lab workflows include primary intrachromosomal releases for [**GM12878**](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FGM12878%5Fprimary%5Fintrachromosomal%5Fcontact%5Fmatrices%2Etar%2Egz), [**K562**](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE63525&format=file&file=GSE63525%5FK562%5Fintrachromosomal%5Fcontact%5Fmatrices%2Etar%2Egz), [**HMEC**](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FHMEC%5Fintrachromosomal%5Fcontact%5Fmatrices.tar.gz), and [**NHEK**](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FNHEK%5Fintrachromosomal%5Fcontact%5Fmatrices.tar.gz).

Place extracted **RAWobserved** / normalization files under the raw tree expected by `Read_Data.py` (see **Data layout** below).

---

## Data layout and environment

All preprocessing scripts read **`root_dir`** from `data/Arg_Parser.py`. Default: `Data/R16_down`. Override with `HICP2GAN_DATA_ROOT` or `HICARN_DATA_ROOT` (see Build instructions).

Under `root_dir` (per cell line):

- **Raw (Rao-style):** `{root_dir}/raw/{CELL}/…/10kb_resolution_intrachromosomal/chr*/…`
- **Intermediate matrices:** `{root_dir}/{CELL}/mat/chrN_10kb.npz`, `chrN_40kb.npz`, …
- **Training patches:** `{root_dir}/{CELL}/data_40_40/hicarn_10kb40kb_c40_s40_b201_nonpool_{train,valid,…}.npz`

Chromosome splits (`train`, `valid`, cell-line-specific tests) live in `data/Arg_Parser.py` (`set_dict`).

---

## Data preprocessing

Preprocessing follows the same **three-stage** outline as the original HiCARN workflow: one Python file per stage under `data/`.

### 1. Processing raw data → `mat/*.npz`

Produces `{root_dir}/{CELL}/mat/chr*_10kb.npz`:

```bash
python data/Read_Data.py -c GM12878 -hr 10kb -q MAPQGE30 -n KRnorm
```

| Argument | Description |
|----------|-------------|
| `-c` | **Required.** Cell line folder name under `{root_dir}/raw/`. |
| `-hr` | Resolution: `5kb`, `10kb`, … (default `10kb`). |
| `-q` | `MAPQGE30` or `MAPQG0` (default `MAPQGE30`). |
| `-n` | `KRnorm`, `SQRTVCnorm`, or `VCnorm` (default `KRnorm`). |

### 2. Downsampling high-res → low-res

Writes `chr*_40kb.npz` beside the high-res `.npz` in `mat/`:

```bash
python data/Downsample.py -c GM12878 -hr 10kb -lr 40kb -r 16
```

| Argument | Description |
|----------|-------------|
| `-hr` | Source resolution (e.g. `10kb`). |
| `-lr` | Target low resolution label in filenames (e.g. `40kb`). |
| `-r` | Downsampling factor (e.g. `16` for 10kb→40kb; use `64` if that matches your design). |
| `-c` | Cell line. |

### 3. Creating train / validation / test patch datasets

Edit **`set_dict`** in `data/Arg_Parser.py` if you need different chromosome splits. Example for **train**:

```bash
python data/Generate.py -c GM12878 -hr 10kb -lr 40kb -lrc 100 \
  -s train -chunk 40 -stride 40 -bound 201 -scale 1 -type max
```

Repeat with `-s valid`, `GM12878_test`, `NHEK_test`, `HMEC_test`, `K562_test`, etc.

| Argument | Description |
|----------|-------------|
| `-s` | Split name (must exist in `set_dict`). |
| `-chunk`, `-stride` | Submatrix size and step (often both `40`). |
| `-bound` | Genomic distance cap in bins (e.g. `201`). |
| `-lrc` | LR clamp / scale anchor (e.g. `100`). |
| `-scale`, `-type` | Pooling: typically `-scale 1` and `-type max`. |

Outputs: `{root_dir}/{CELL}/data_40_40/hicarn_{hr}{lr}_c40_s40_b201_nonpool_{split}.npz`.

### Optional: batch driver (several cell lines)

```bash
export HICP2GAN_DATA_ROOT="/path/to/Data/R64_down"   # example
bash scripts/generate_data_all_cell_lines.sh
```

The script exports `PYTHONPATH`, sets `HICP2GAN_DATA_ROOT`, and runs the three stages in order for each configured cell line.

---

## Training (HiCP2GAN)

**Entrypoint:** `HiCP2GAN_train.py` (legacy symlink name: `HiCFoundGAN_PnP.py`).

- **Generator:** any `Models.*` class you pass (default `Models.HiCARN_1.Generator`).
- **Discriminator:** HiCFoundation ViT trunk + head — supply **`--foundation_ckpt`** (after downloading weights from [Zenodo](https://zenodo.org/uploads/20030290), e.g. **`checkpoints/pretrained/hicfoundation_resolution.pth.tar`**).

Example (adjust paths to your `data_40_40` layout):

```bash
python HiCP2GAN_train.py \
  --train_npz "$HICP2GAN_DATA_ROOT/GM12878/data_40_40/hicarn_10kb40kb_c40_s40_b201_nonpool_train.npz" \
  --valid_npz "$HICP2GAN_DATA_ROOT/GM12878/data_40_40/hicarn_10kb40kb_c40_s40_b201_nonpool_valid.npz" \
  --foundation_ckpt checkpoints/pretrained/hicfoundation_resolution.pth.tar \
  --foundation_ctor_module FoundationGANWorks.HiCFoundation.model.Vision_Transformer_count \
  --foundation_ctor_class vit_large_patch16 \
  --num_transformer_layers 11 \
  --gen_module Models.HiCARN_1 \
  --gen_class Generator \
  --out_dir checkpoints/my_run \
  --epochs 100 \
  --batch_size 64
```

See **`python HiCP2GAN_train.py --help`** for all flags (`--gen_kwargs`, loss weights, `out_dir`, etc.). Validation logs PSNR / SSIM / GenomeDISCO-style metrics using `Utils/`.

---

## Inference / testing (`predict_40x40.py`)

Loads a **generator** checkpoint and runs patch-wise prediction on a test `.npz` from preprocessing:

```bash
python predict_40x40.py \
  -c GM12878 \
  -lr 40kb \
  -f hicarn_10kb40kb_c40_s40_b201_nonpool_GM12878_test.npz \
  -m HiCARN_1 \
  -ckpt checkpoints/R64/conventional_gan/HiCARN_2_R64/03_10_18_23_finalg_10kb40kb_c40_s40_b201_nonpool_HiCARN_2_R64.pytorch \
  --cuda 0
```

| Argument | Description |
|----------|-------------|
| `-c` | Cell line (used for paths under `root_dir`). |
| `-f` | NPZ **filename** (resolved under `…/data_40_40/`, then `…/data/`, then legacy `data/`). |
| `-m` | One of **`HiCARN_1`**, **`HiCARN_2`**, **`DeepHiC`** (extend in script for other `Models/`). |
| `-ckpt` | Path to generator weights (`.pytorch` / compatible). |
| `--cuda` | GPU id, or CPU behavior per script logic. |

Outputs: `{root_dir}/predict/{cell_line}/predict_chr{N}_{low_res}.npz`.

**Note:** The original HiCARN predictor warns that reconstruction can require **very large host RAM**; plan resources accordingly.

---

## Accessing predicted `.npz` files

Each per-chromosome file stores arrays compressed by NumPy. The enhanced map is typically under the key **`hicarn`**; **`compact`** holds indices for sparse reconstruction (see `predict_40x40.py` / `Utils.io.spreadM` usage).

Example:

```python
import numpy as np
d = np.load("path/to/predict_chr20_40kb.npz", allow_pickle=True)
hr = d["hicarn"]
```

---

## Repository map

| Path | Purpose |
|------|--------|
| `data/` | `Read_Data.py`, `Downsample.py`, `Generate.py`, `Arg_Parser.py` |
| `scripts/generate_data_all_cell_lines.sh` | Optional multi-cell preprocessing driver |
| `Models/` | Generator architectures (`HiCARN_1`, `HiCARN_2`, `DeepHiC`, `DiCARN`, …) |
| `Utils/` | I/O, SSIM, GenomeDISCO, checkpoints, etc. |
| `FoundationGANWorks/` | HiCFoundation **code** for the discriminator (weights: separate download) |
| `HiCP2GAN_train.py` | Main training CLI |
| `predict_40x40.py` | 40×40 patch inference |
| `checkpoints/` | *(Not in Git.)* Download from [Zenodo](https://zenodo.org/uploads/20030290) |

---

## Citation

If you use this code, please cite the HiCP2GAN manuscript and the original HiCARN / HiCFoundation references appropriate to your study.
