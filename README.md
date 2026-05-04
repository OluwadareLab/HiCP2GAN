# HiCP2GAN: A Plug and Play Foundation Model-based GAN for Hi-C Enhancement

Plug-and-play Hi-C enhancement: a **trainable generator** from `Models/` plus a **HiCFoundation** vision backbone used as the discriminator (`HiCP2GAN_train.py`). This repository includes **standalone data preprocessing** (HiCARN-style), **training**, and **40×40 inference**.

## Setup

```bash
cd /path/to/HiCP2GAN
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Point Python at the repo root when running scripts (or install as a package):

```bash
export PYTHONPATH="/path/to/HiCP2GAN${PYTHONPATH:+:$PYTHONPATH}"
```

## Data layout and environment

All preprocessing scripts read **`root_dir`** from `data/Arg_Parser.py`. By default `root_dir` is `Data/R16_down`. Override with:

```bash
export HICP2GAN_DATA_ROOT="/path/to/your/data/tree"
# optional legacy alias:
export HICARN_DATA_ROOT="$HICP2GAN_DATA_ROOT"
```

Expected layout under `root_dir` (per cell line):

- **Raw (Rao-style):** `{root_dir}/raw/{CELL}/…/10kb_resolution_intrachromosomal/chr*/…`
- **Intermediate matrices:** `{root_dir}/{CELL}/mat/chrN_10kb.npz`, `chrN_40kb.npz`, …
- **Training patches:** `{root_dir}/{CELL}/data_40_40/hicarn_10kb40kb_c40_s40_b201_nonpool_{train,valid,…}.npz`

Chromosome splits for `train` / `valid` / cell-line-specific test keys are defined in `data/Arg_Parser.py` (`set_dict`).

---

## 1. Data preprocessing

Preprocessing is split into **three standalone scripts** under `data/` (same logical stages as the original HiCARN workflow). Run them from anywhere as long as `PYTHONPATH` includes the repository root.

### Stage A — raw contacts to dense `.npz` (per chromosome)

```bash
python data/Read_Data.py -c GM12878 -hr 10kb -q MAPQGE30 -n KRnorm
```

Writes `{root_dir}/GM12878/mat/chr*_10kb.npz`.

### Stage B — downsample high-res to low-res

```bash
python data/Downsample.py -c GM12878 -hr 10kb -lr 40kb -r 16
```

Adjust **`-r`** to match your target resolution (e.g. 16× for 10kb→40kb, or 64× if that matches your experiment). Writes `chr*_40kb.npz` next to the 10kb files in `mat/`.

### Stage C — build patch datasets (`train` / `valid` / test splits)

```bash
python data/Generate.py -c GM12878 -hr 10kb -lr 40kb -lrc 100 \
  -s train -chunk 40 -stride 40 -bound 201 -scale 1 -type max
```

Repeat **`-s`** for `valid`, `GM12878_test`, `NHEK_test`, etc. (see `set_dict` in `data/Arg_Parser.py`). Outputs land in:

`{root_dir}/{CELL}/data_40_40/hicarn_{hr}{lr}_c40_s40_b201_nonpool_{split}.npz`.

### Optional: batch driver for several cell lines

```bash
export HICP2GAN_DATA_ROOT="/path/to/Data/R64_down"   # example; edit script defaults if needed
bash scripts/generate_data_all_cell_lines.sh
```

The shell script sets `PYTHONPATH`, exports `HICP2GAN_DATA_ROOT`, and calls `data/Read_Data.py`, `data/Downsample.py`, and `data/Generate.py` in sequence for each configured cell line.

---

## 2. Training (HiCP2GAN)

**Entrypoint:** `HiCP2GAN_train.py` (same code as the legacy name `HiCFoundGAN_PnP.py` if present as a symlink).

Training loads:

- **Generator:** any module under `Models/` exposing the chosen class (default `Models.HiCARN_1.Generator`).
- **Discriminator:** HiCFoundation ViT trunk + head; requires **`--foundation_ckpt`** (your pretrained weights, not shipped in this repo).

Minimal example (paths must match where your `.npz` files were written, often `data_40_40`):

```bash
python HiCP2GAN_train.py \
  --train_npz "$HICP2GAN_DATA_ROOT/GM12878/data_40_40/hicarn_10kb40kb_c40_s40_b201_nonpool_train.npz" \
  --valid_npz "$HICP2GAN_DATA_ROOT/GM12878/data_40_40/hicarn_10kb40kb_c40_s40_b201_nonpool_valid.npz" \
  --foundation_ckpt /path/to/hicfoundation_weights.pth.tar \
  --foundation_ctor_module FoundationGANWorks.HiCFoundation.model.Vision_Transformer_count \
  --foundation_ctor_class vit_large_patch16 \
  --num_transformer_layers 11 \
  --gen_module Models.HiCARN_1 \
  --gen_class Generator \
  --out_dir checkpoints/my_run \
  --epochs 100 \
  --batch_size 64
```

Useful flags (non-exhaustive; see **`python HiCP2GAN_train.py --help`**):

| Flag | Role |
|------|------|
| `--gen_module` / `--gen_class` | Plug-in generator (e.g. `Models.HiCARN_2`, `Models.DeepHiC`, …) |
| `--gen_kwargs` | JSON dict for the generator constructor |
| `--foundation_ctor_module` / `--foundation_ctor_class` | ViT factory (defaults can be set in code) |
| `--num_transformer_layers` | How many transformer blocks feed the discriminator trunk |
| `--adv_weight`, `--l1_weight`, `--feat_weight` | Loss balance |
| `--out_dir` | Checkpoints, curves, and per-epoch validation metrics |

Validation reports PSNR / SSIM / GenomeDISCO-style metrics via `Utils/`.

---

## 3. Inference / testing (`predict_40x40.py`)

**Entrypoint:** `predict_40x40.py` — loads a **trained generator checkpoint** (not the full GAN bundle unless you adapt the loader) and runs patch-wise forward on a `.npz` produced by `data/Generate.py`.

```bash
python predict_40x40.py \
  -c GM12878 \
  -lr 40kb \
  -f hicarn_10kb40kb_c40_s40_b201_nonpool_GM12878_test.npz \
  -m HiCARN_1 \
  -ckpt /path/to/generator_weights.pytorch \
  --cuda 0
```

The script resolves the input file under `{root_dir}/{cell_line}/data_40_40/`, then `…/data/`, then legacy `{root_dir}/data/`. Writes per-chromosome predictions under:

`{root_dir}/predict/{cell_line}/predict_chr{N}_{low_res}.npz`.

**Supported `-m` values in this script:** `HiCARN_1`, `HiCARN_2`, `DeepHiC` (see the `if model == …` branches in `predict_40x40.py`). Extending to other `Models/` definitions follows the same pattern.

**Resource note:** The original HiCARN predictor warns that full runs can require **very large RAM**; adjust batch size / parallelism only after inspecting `predict_40x40.py` if you hit memory limits.

---

## Repository map

| Path | Purpose |
|------|--------|
| `data/` | `Read_Data.py`, `Downsample.py`, `Generate.py`, `Arg_Parser.py` |
| `scripts/generate_data_all_cell_lines.sh` | Optional multi-cell preprocessing driver |
| `Models/` | Generator architectures (`HiCARN_1`, `HiCARN_2`, `DeepHiC`, `DiCARN`, …) |
| `Utils/` | I/O, SSIM, GenomeDISCO helpers for training / metrics |
| `FoundationGANWorks/` | HiCFoundation **code** for the discriminator (download weights separately) |
| `HiCP2GAN_train.py` | Main training CLI |
| `predict_40x40.py` | 40×40 patch inference |

---

## Citation

If you use this code, please cite the HiCP2GAN / HiCARN / HiCFoundation works referenced in your paper.
