"""
HiCP2GAN training: plug-and-play Hi-C generator + HiCFoundation ViT discriminator.

Run: ``python HiCP2GAN_train.py --help``. The same entrypoint is available as
``HiCFoundGAN_PnP.py`` (symlink) for compatibility with older HiCARN workflows.
"""
import os, io, tarfile, json, math, argparse, importlib, sys
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader
from Utils.SSIM import ssim
from Utils.GenomeDISCO import compute_reproducibility
import scipy.sparse as sps


# --- live_plotter.py (inline) ---
import matplotlib
matplotlib.use("Agg")  # safe for headless servers/SSH
import matplotlib.pyplot as plt
from collections import defaultdict
import os

class LivePlotter:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.steps = []
        self.hist = defaultdict(list)
        self._gstep = 0

    def log_batch(self, metrics: dict, step: int = None):
        # metrics: dict like {"D": float, "G": float, "L1": float, "ADV": float, "FM": float}
        self._gstep = (self._gstep + 1) if step is None else step
        self.steps.append(self._gstep)
        for k, v in metrics.items():
            self.hist[k].append(float(v))

    def log_epoch(self, epoch: int, val_psnr: float):
        self.hist["epoch"].append(epoch)
        self.hist["val_psnr"].append(float(val_psnr))
        self.render_curves()

    def render_curves(self):
        # Loss panel
        plt.figure(figsize=(8,5))
        if "D" in self.hist: plt.plot(self.steps, self.hist["D"], label="D")
        if "G" in self.hist: plt.plot(self.steps, self.hist["G"], label="G")
        if "L1" in self.hist: plt.plot(self.steps, self.hist["L1"], label="L1")
        if "ADV" in self.hist: plt.plot(self.steps, self.hist["ADV"], label="ADV")
        if "FM" in self.hist: plt.plot(self.steps, self.hist["FM"], label="FM")
        plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Losses")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "loss_curves.png"))
        plt.close()

        # PSNR panel (per-epoch)
        if "val_psnr" in self.hist:
            plt.figure(figsize=(8,5))
            xs = self.hist.get("epoch", list(range(1, len(self.hist["val_psnr"])+1)))
            plt.plot(xs, self.hist["val_psnr"], marker="o")
            plt.xlabel("Epoch"); plt.ylabel("PSNR (dB)"); plt.title("Validation PSNR")
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, "val_psnr.png"))
            plt.close()


# ----------------------------
# Config
# ----------------------------
@dataclass
class CFG:
    # Data: NPZ files with keys 'data' (LR), 'target' (HR), and optional 'inds'
    train_npz: str = "Data/R16_down/GM12878/data/hicarn_10kb40kb_c40_s40_b201_nonpool_train.npz"
    valid_npz: str = "Data/R16_down/GM12878/data/hicarn_10kb40kb_c40_s40_b201_nonpool_valid.npz"

    # Training
    epochs: int = 100
    batch_size: int = 64
    num_workers: int = 8
    amp: bool = True
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)
    grad_clip_g: float = 1.0
    grad_clip_d: float = 1.0
    log_every: int = 100

    # Loss weights
    adv_weight: float = 1.0
    l1_weight: float = 10.0
    feat_weight: float = 10.0

    # Generator plug & play
    gen_module: str = "Models.HiCARN_1"  # python module path
    gen_class: str = "Generator"          # class name
    gen_kwargs: Optional[Dict] = None    # JSON-encoded dict via CLI

    # HiCFoundation discriminator trunk
    foundation_ctor_module: Optional[str] = None  # e.g., "foundation.model"
    foundation_ctor_class: Optional[str] = None   # e.g., "HiCFoundationBackbone"
    foundation_ckpt: str = "checkpoints/hicfoundation_pretrain.pth.tar"
    freeze_foundation: bool = False
    num_frozen_layers: int = 0  # Always 0 for depth ablation (all layers finetuned)
    num_transformer_layers: int = 24  # Number of transformer blocks to use (1-24)
    d_hidden: int = 256

    # Runtime
    out_dir: str = "out_depth_ablation_corrected"
    seed: int = 1337
    # device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ----------------------------
# Utilities
# ----------------------------
def seed_everything(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= eps: return 99.0
    return 10.0 * math.log10(1.0 / mse)  # assumes [0,1]

def normalize_per_patch(x: torch.Tensor) -> torch.Tensor:
    # x: (1, H, W)
    xmin = x.amin(dim=(-2,-1), keepdim=True)
    xmax = x.amax(dim=(-2,-1), keepdim=True)
    return (x - xmin) / (xmax - xmin + 1e-8)

def torch_load_any(path: str, map_location="cpu"):
    """
    Loads either a raw PyTorch checkpoint (.pth/.pt/.pth.tar)
    or the first plausible weights file inside a real .tar.
    """
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            prefer = [m for m in members if m.name.endswith((".pth", ".pt", ".pth.tar", ".ckpt"))]
            member = prefer[0] if prefer else max(members, key=lambda m: m.size)
            byts = tf.extractfile(member).read()
            buf = io.BytesIO(byts)
            return torch.load(buf, map_location=map_location)
    return torch.load(path, map_location=map_location)

def strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = OrderedDict()
    for k, v in sd.items():
        out[k[7:]] = v if k.startswith("module.") else v
    return out


# ----------------------------
# Dataset (HiCARN-style NPZ)
# ----------------------------
class HiCARNPairNPZ(Dataset):
    """
    Expects .npz with keys:
      - 'data'   : (N, 1, 40, 40) low-res inputs
      - 'target' : (N, 1, 40, 40) high-res targets
      - 'inds'   : optional
    """
    def __init__(self, npz_path: str, normalize=False):
        obj = np.load(npz_path)
        self.lr = obj["data"].astype(np.float32)
        self.hr = obj["target"].astype(np.float32)
        self.normalize = normalize
        assert self.lr.shape == self.hr.shape, f"LR/HR mismatch: {self.lr.shape} vs {self.hr.shape}"
        self.inds = obj["inds"] if "inds" in obj else None

    def __len__(self):
        return self.lr.shape[0]

    def __getitem__(self, i):
        lr = torch.from_numpy(self.lr[i])  # (1,40,40)
        hr = torch.from_numpy(self.hr[i])  # (1,40,40)
        if self.normalize:
            lr = normalize_per_patch(lr)
            hr = normalize_per_patch(hr)
        return lr, hr


# ----------------------------
# Generator builder (plug & play)
# ----------------------------
def build_generator(cfg: CFG) -> nn.Module:
    gen_kwargs = cfg.gen_kwargs or {}
    mod = importlib.import_module(cfg.gen_module)
    
    # Wrap generators that output 28x28 to handle 40x40 input/output
    # These generators (HiCSR, HiCNN, HiCPlus) expect 28x28 input and output 28x28
    # We wrap them to resize 40x40 -> 28x28 (area interpolation), process, then upsample 28x28 -> 40x40
    if cfg.gen_module in ["Models.HiCSR", "Models.HiCNN", "Models.HiCPlus"]:
        Gen40x40Class = getattr(mod, "Generator40x40")
        # For HiCSR, we need to pass num_res_blocks (defaults to 15 if not provided)
        if cfg.gen_module == "Models.HiCSR":
            num_res_blocks = gen_kwargs.get("num_res_blocks", 15)
            G = Gen40x40Class(num_res_blocks=num_res_blocks)
        else:
            G = Gen40x40Class()
        print(f"[generator] Wrapped {cfg.gen_module}.{cfg.gen_class} with Generator40x40 to handle 40x40 input/output")
    else:
        GenClass = getattr(mod, cfg.gen_class)
        G = GenClass(**gen_kwargs)
    
    return G


# ----------------------------
# HiCFoundation trunk loader
# ----------------------------
def build_foundation_trunk(cfg: CFG) -> nn.Module:
    """
    You can:
      - Provide ctor via cfg.foundation_ctor_module & cfg.foundation_ctor_class
      - Or fall back to a tiny conv trunk placeholder if ctor not provided
    """
    if cfg.foundation_ctor_module and cfg.foundation_ctor_class:
        # Add HiCFoundation directory to path for relative imports
        hicfoundation_path = os.path.join(os.path.dirname(__file__), "FoundationGANWorks", "HiCFoundation")
        if os.path.exists(hicfoundation_path) and hicfoundation_path not in sys.path:
            sys.path.insert(0, hicfoundation_path)
        
        # Adjust module path if it starts with FoundationGANWorks
        module_path = cfg.foundation_ctor_module
        if module_path.startswith("FoundationGANWorks.HiCFoundation."):
            # Remove the FoundationGANWorks.HiCFoundation prefix since we added it to path
            module_path = module_path.replace("FoundationGANWorks.HiCFoundation.", "")
        
        mod = importlib.import_module(module_path)
        Trunk = getattr(mod, cfg.foundation_ctor_class)
        
        # Special handling for HiCFoundation models
        if "Vision_Transformer" in cfg.foundation_ctor_module or "vit_large" in cfg.foundation_ctor_class:
            # Use factory function with proper args for 40x40 input
            # Input is 1 channel, but HiCFoundation expects 3 channels (RGB)
            # Note: vit_large_patch16 already has in_chans=3 hardcoded, so don't pass it
            # We'll create a wrapper to handle this
            if hasattr(Trunk, '__call__'):
                # It's a function (like vit_large_patch16)
                full_model = Trunk(img_size=(40, 40))
            else:
                # It's a class
                full_model = Trunk(img_size=(40, 40), in_chans=3)
            
            # Extract encoder backbone (patch_embed + blocks + norm)
            # MODIFIED: Only use first num_transformer_layers blocks
            class EncoderBackbone(nn.Module):
                def __init__(self, full_model, num_layers: int = 24):
                    super().__init__()
                    self.patch_embed = full_model.patch_embed
                    self.pos_embed = full_model.pos_embed
                    # Use original blocks directly when using all layers (matches frozen0 setup)
                    # Only create new ModuleList when truncating
                    if num_layers == 24:
                        self.blocks = full_model.blocks  # Direct reference (same as HiCFoundGAN_train.py)
                    else:
                        self.blocks = nn.ModuleList(list(full_model.blocks[:num_layers]))
                    self.norm = full_model.norm
                    # Add channel adapter: 1 channel -> 3 channels
                    self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1)
                    self.num_layers = num_layers
                    print(f"[foundation] Using {num_layers}/{len(full_model.blocks)} transformer blocks")
                
                def forward(self, x):
                    # x: (B, 1, 40, 40)
                    # Convert to 3 channels
                    x = self.channel_adapter(x)  # (B, 3, 40, 40)
                    # Patch embed
                    x = self.patch_embed(x)  # (B, N, D)
                    # Add positional embedding (skip cls token position)
                    x = x + self.pos_embed[:, 1:, :]
                    # Add cls token
                    cls_token = self.pos_embed[:, :1, :]
                    x = torch.cat([cls_token.expand(x.shape[0], -1, -1), x], dim=1)
                    # Transformer blocks (only first num_layers)
                    for blk in self.blocks:
                        x = blk(x)
                    x = self.norm(x)
                    # Return features (B, N+1, D) - can be pooled later
                    return x
            
            trunk = EncoderBackbone(full_model, num_layers=cfg.num_transformer_layers)
        else:
            trunk = Trunk()  # adapt ctor args if needed
    else:
        # Placeholder trunk (replace with real HiCFoundation backbone)
        class TinyConvTrunk(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(1, 64, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64,128,3, padding=1), nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(128,256,3, padding=1), nn.LeakyReLU(0.2, inplace=True),
                )
            def forward(self, x):  # (B,1,40,40) -> (B,256,40,40)
                return self.net(x)
        trunk = TinyConvTrunk()

    # Load checkpoint (flexible for .pth/.pt/.pth.tar)
    if cfg.foundation_ckpt and os.path.isfile(cfg.foundation_ckpt):
        obj = torch_load_any(cfg.foundation_ckpt, map_location="cpu")
        # Common patterns: obj["state_dict"], obj["model"], or obj itself
        if isinstance(obj, dict):
            state_dict = obj.get("state_dict", obj.get("model", obj))
        else:
            state_dict = obj
        state_dict = strip_module_prefix(state_dict)
        missing, unexpected = trunk.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[foundation] missing keys: {len(missing)} (ok if heads differ)")
        if unexpected:
            print(f"[foundation] unexpected keys: {len(unexpected)} (ok if extra heads present)")
    else:
        print("[foundation] WARNING: checkpoint not found or not provided; using random init.")

    return trunk


# ----------------------------
# Foundation Discriminator wrapper
# ----------------------------
class FoundationDiscriminator(nn.Module):
    """
    Wraps the HiCFoundation trunk and adds a real/fake head.
    Accepts (B,1,40,40). If trunk returns (B,C,h,w), we GAP it; if (B,C), we keep it.
    Supports selective freezing of transformer blocks.
    """
    def __init__(self, trunk: nn.Module, hidden: int = 256, freeze_trunk: bool = False, num_frozen_layers: int = 0):
        super().__init__()
        self.trunk = trunk
        self.num_frozen_layers = num_frozen_layers
        
        # Handle freezing - for depth ablation, we always finetune all layers (num_frozen_layers=0)
        if num_frozen_layers == 0:
            # All layers are trainable
            for p in self.trunk.parameters():
                p.requires_grad = True
            print(f"[foundation] All transformer blocks are trainable (num_frozen_layers=0)")
        elif freeze_trunk and num_frozen_layers == -1:
            # Legacy: freeze everything
            for p in self.trunk.parameters():
                p.requires_grad = False
        elif num_frozen_layers > 0:
            # Selective freezing: freeze first N transformer blocks
            self._freeze_selective_layers(num_frozen_layers)

        # Probe trunk output channels to size the head
        with torch.no_grad():
            # make the probe on whatever device the trunk lives on
            trunk_device = next(self.trunk.parameters()).device
            probe = torch.zeros(1, 1, 40, 40, device=trunk_device)
            f = self.trunk(probe)
            if f.ndim == 4:
                # 4D: (B, C, H, W) - spatial features
                c = f.shape[1]
                self.prep = nn.AdaptiveAvgPool2d((1,1))
                in_dim = c
            elif f.ndim == 3:
                # 3D: (B, N, D) - sequence of tokens (e.g., transformer output)
                # Use mean pooling over sequence dimension, or use cls token
                # For HiCFoundation, first token is cls token, so we can use it directly
                in_dim = f.shape[2]  # embedding dimension
                self.prep = lambda x: x[:, 0, :] if x.ndim == 3 else x  # Use cls token (first token)
            elif f.ndim == 2:
                # 2D: (B, D) - already flattened features
                self.prep = nn.Identity()
                in_dim = f.shape[1]
            else:
                raise ValueError(f"Unexpected trunk output shape: {tuple(f.shape)}")

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),  # logits
        )

    def _freeze_selective_layers(self, num_frozen: int):
        """
        Freeze the first N transformer blocks in the encoder.
        num_frozen: number of blocks to freeze (0 = all unfrozen, 24 = all frozen)
        """
        total_blocks = 0
        
        # Freeze patch embedding
        if hasattr(self.trunk, 'patch_embed'):
            for p in self.trunk.patch_embed.parameters():
                p.requires_grad = False
        
        # Freeze positional embedding (usually fixed anyway)
        if hasattr(self.trunk, 'pos_embed'):
            self.trunk.pos_embed.requires_grad = False
        
        # Freeze first N transformer blocks
        if hasattr(self.trunk, 'blocks') and isinstance(self.trunk.blocks, nn.ModuleList):
            total_blocks = len(self.trunk.blocks)
            for i, block in enumerate(self.trunk.blocks):
                if i < num_frozen:
                    for p in block.parameters():
                        p.requires_grad = False
                else:
                    for p in block.parameters():
                        p.requires_grad = True
            
            # Freeze norm if all blocks are frozen
            if num_frozen >= total_blocks and hasattr(self.trunk, 'norm'):
                for p in self.trunk.norm.parameters():
                    p.requires_grad = False
            elif hasattr(self.trunk, 'norm'):
                for p in self.trunk.norm.parameters():
                    p.requires_grad = True
        
        if total_blocks > 0:
            print(f"[foundation] Frozen {num_frozen}/{total_blocks} transformer blocks")
        else:
            print(f"[foundation] Warning: Could not find transformer blocks in trunk. Freezing all parameters.")
            for p in self.trunk.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor, return_features: bool = False):
        f = self.trunk(x)
        if f.ndim == 4:
            # 4D: (B, C, H, W) - spatial features
            z = self.prep(f).flatten(1)
        elif f.ndim == 3:
            # 3D: (B, N, D) - sequence of tokens
            z = self.prep(f)  # Extract cls token or pool
        else:
            # 2D: (B, D) - already flattened
            z = f
        logit = self.head(z)
        if return_features:
            return logit, z
        return logit


# ----------------------------
# Losses
# ----------------------------
class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, pred_logits: torch.Tensor, is_real: bool):
        target = torch.ones_like(pred_logits) if is_real else torch.zeros_like(pred_logits)
        return self.bce(pred_logits, target)

def feature_matching_loss(fake_feat: torch.Tensor, real_feat: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(fake_feat, real_feat)

def output_adapter(x: torch.Tensor, gen_module: str) -> torch.Tensor:
    """
    Consistent output adapter for all generators.
    Uses generator module name to determine output format (not heuristic detection).
    
    - HiCSR outputs tanh (range [-1, 1]): converts to [0, 1]
    - All others: clamps to [0, 1]
    """
    # Module-based detection (reliable, not heuristic)
    if gen_module == "Models.HiCSR":
        # Convert tanh output to [0, 1]
        x = (x + 1.0) / 2.0
    
    # Clamp to [0, 1] for all generators
    x = torch.clamp(x, 0.0, 1.0)
    
    return x


# ----------------------------
# Training
# ----------------------------
def train(cfg: CFG):
    os.makedirs(cfg.out_dir, exist_ok=True)
    seed_everything(cfg.seed)
    device = cfg.device
    print(f"[device] {device} | AMP={cfg.amp}")
    print(f"[config] num_transformer_layers={cfg.num_transformer_layers}, num_frozen_layers={cfg.num_frozen_layers} (all finetuned)")

    # Data
    train_ds = HiCARNPairNPZ(cfg.train_npz, normalize=False)
    val_ds   = HiCARNPairNPZ(cfg.valid_npz, normalize=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    # Models
    G = build_generator(cfg).to(device)
    trunk = build_foundation_trunk(cfg).to(device)
    D = FoundationDiscriminator(
        trunk, 
        hidden=cfg.d_hidden, 
        freeze_trunk=False,  # Always False for depth ablation
        num_frozen_layers=0  # Always 0 for depth ablation (all layers finetuned)
    ).to(device)

    # Optimizers & scalers
    opt_g = torch.optim.Adam(filter(lambda p: p.requires_grad, G.parameters()), lr=cfg.lr_g, betas=cfg.betas)
    opt_d = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=cfg.lr_d, betas=cfg.betas)
    scaler_g = amp.GradScaler(enabled=cfg.amp)
    scaler_d = amp.GradScaler(enabled=cfg.amp)

    gan_loss = GANLoss()
    l1_loss = nn.L1Loss()

    best_val_psnr = 0.0
    best_val_ssim = 0.0
    best_val_genomedisco = 0.0

    plotter = LivePlotter(cfg.out_dir)
    
    # Track validation metrics per epoch for saving to file
    val_metrics_epochs = []  # List of (epoch, psnr, ssim, genomedisco)

    for epoch in range(1, cfg.epochs + 1):
        G.train(); D.train()
        meters = {"loss_d":0.0,"loss_g":0.0,"l1":0.0,"adv":0.0,"feat":0.0,"n":0}

        for it, (lr_patch, hr_patch) in enumerate(train_loader, 1):
            lr_patch = lr_patch.to(device, non_blocking=True)
            hr_patch = hr_patch.to(device, non_blocking=True)
            bsz = lr_patch.size(0)
            meters["n"] += bsz

            # ---- Update D ----
            opt_d.zero_grad(set_to_none=True)
            with amp.autocast(enabled=cfg.amp):
                with torch.no_grad():
                    fake_hr = output_adapter(G(lr_patch), cfg.gen_module)
                d_real = D(hr_patch)
                d_fake = D(fake_hr.detach())
                loss_d = gan_loss(d_real, True) + gan_loss(d_fake, False)
            scaler_d.scale(loss_d).backward()
            if cfg.grad_clip_d > 0:
                scaler_d.unscale_(opt_d)
                nn.utils.clip_grad_norm_(D.parameters(), cfg.grad_clip_d)
            scaler_d.step(opt_d)
            scaler_d.update()

            # ---- Update G ----
            opt_g.zero_grad(set_to_none=True)
            with amp.autocast(enabled=cfg.amp):
                fake_hr = output_adapter(G(lr_patch), cfg.gen_module)
                d_fake_for_g, fake_feat = D(fake_hr, return_features=True)
                _, real_feat = D(hr_patch, return_features=True)

                loss_adv  = gan_loss(d_fake_for_g, True) * cfg.adv_weight
                loss_l1   = l1_loss(fake_hr, hr_patch) * cfg.l1_weight
                loss_feat = feature_matching_loss(fake_feat, real_feat) * cfg.feat_weight
                loss_g = loss_adv + loss_l1 + loss_feat

            scaler_g.scale(loss_g).backward()
            if cfg.grad_clip_g > 0:
                scaler_g.unscale_(opt_g)
                nn.utils.clip_grad_norm_(G.parameters(), cfg.grad_clip_g)
            scaler_g.step(opt_g)
            scaler_g.update()

            meters["loss_d"] += loss_d.item() * bsz
            meters["loss_g"] += loss_g.item() * bsz
            meters["l1"]     += loss_l1.item() * bsz
            meters["adv"]    += loss_adv.item() * bsz
            meters["feat"]   += loss_feat.item() * bsz

            if it % cfg.log_every == 0:
                denom = max(1, meters["n"])
                print(f"[epoch {epoch}] it {it}/{len(train_loader)}  "
                      f"D {meters['loss_d']/denom:.4f} | G {meters['loss_g']/denom:.4f} "
                      f"(L1 {meters['l1']/denom:.4f}, ADV {meters['adv']/denom:.4f}, FM {meters['feat']/denom:.4f})")
                
                # Plotter logging
                plotter.log_batch({
                    "D":   meters["loss_d"]/denom,
                    "G":   meters["loss_g"]/denom,
                    "L1":  meters["l1"]/denom,
                    "ADV": meters["adv"]/denom,
                    "FM":  meters["feat"]/denom,
                })
                plotter.render_curves()  # update PNGs

        # ---- Validation ----
        G.eval()
        with torch.no_grad():
            psnrs = []
            ssims = []
            genomediscos = []
            
            for lr_patch, hr_patch in val_loader:
                lr_patch = lr_patch.to(device, non_blocking=True)
                hr_patch = hr_patch.to(device, non_blocking=True)
                sr = output_adapter(G(lr_patch), cfg.gen_module)
                # Clamp HR to [0,1] for consistent metric computation
                hr_patch_clamped = torch.clamp(hr_patch, 0.0, 1.0)
                
                # Compute PSNR (use clamped HR)
                psnrs.append(psnr(sr, hr_patch_clamped))
                
                # Compute SSIM (use clamped HR)
                batch_ssim = ssim(sr, hr_patch_clamped)
                ssims.append(batch_ssim.item())
                
                # Compute GenomeDISCO (per sample, use clamped HR)
                for i in range(sr.size(0)):
                    sr_np = sr[i, 0].cpu().detach().numpy()  # (H, W)
                    hr_np = hr_patch_clamped[i, 0].cpu().detach().numpy()  # (H, W)
                    
                    # Convert to sparse matrices for GenomeDISCO
                    sr_sparse = sps.csr_matrix(sr_np)
                    hr_sparse = sps.csr_matrix(hr_np)
                    
                    try:
                        gd_score = compute_reproducibility(sr_sparse, hr_sparse, transition=True)
                        # Only append finite values (filters NaN, inf, and other invalid results)
                        if np.isfinite(gd_score):
                            genomediscos.append(gd_score)
                    except Exception as e:
                        # If GenomeDISCO computation fails, skip this sample
                        print(f"[warning] GenomeDISCO computation failed for sample {i}: {e}")
                        continue
            
            val_psnr = float(np.mean(psnrs)) if psnrs else 0.0
            val_ssim = float(np.mean(ssims)) if ssims else 0.0
            # Use nanmean to ignore failed samples (NaN values)
            val_genomedisco = float(np.nanmean(genomediscos)) if genomediscos else 0.0
        
        # Save validation metrics for this epoch
        val_metrics_epochs.append((epoch, val_psnr, val_ssim, val_genomedisco))
        
        # Compute is_best BEFORE updating best metrics (critical fix!)
        is_best = val_psnr > best_val_psnr
        
        # Update best metrics
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
        if val_genomedisco > best_val_genomedisco:
            best_val_genomedisco = val_genomedisco
        
        print(f"[epoch {epoch}] val PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | GenomeDISCO: {val_genomedisco:.4f}")
        
        # Log epoch metrics to plotter
        plotter.log_epoch(epoch, val_psnr)

        # ---- Save checkpoint ----
        ckpt = {
            "cfg": cfg.__dict__,
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "best_val_psnr": best_val_psnr,
            "best_val_ssim": best_val_ssim,
            "best_val_genomedisco": best_val_genomedisco
        }
        save_path = os.path.join(cfg.out_dir, f"ckpt_epoch{epoch}.pt")
        torch.save(ckpt, save_path)
        if is_best:
            torch.save(ckpt, os.path.join(cfg.out_dir, "ckpt_best.pt"))
            print(f"[save] best checkpoint updated (PSNR={best_val_psnr:.2f} dB, SSIM={best_val_ssim:.4f}, GenomeDISCO={best_val_genomedisco:.4f})")

    # Save validation metrics per epoch to text files
    metrics_file = os.path.join(cfg.out_dir, "val_metrics_epochs.txt")
    with open(metrics_file, 'w') as f:
        f.write("epoch,psnr,ssim,genomedisco\n")
        for epoch, psnr_val, ssim_val, gd_val in val_metrics_epochs:
            f.write(f"{epoch},{psnr_val:.6f},{ssim_val:.6f},{gd_val:.6f}\n")
    print(f"[save] Validation metrics per epoch saved to {metrics_file}")
    
    # Also save individual metric files for compatibility
    psnr_file = os.path.join(cfg.out_dir, "val_psnr_epochs.txt")
    with open(psnr_file, 'w') as f:
        f.write("epoch,psnr\n")
        for epoch, psnr_val, _, _ in val_metrics_epochs:
            f.write(f"{epoch},{psnr_val:.6f}\n")
    
    ssim_file = os.path.join(cfg.out_dir, "val_ssim_epochs.txt")
    with open(ssim_file, 'w') as f:
        f.write("epoch,ssim\n")
        for epoch, _, ssim_val, _ in val_metrics_epochs:
            f.write(f"{epoch},{ssim_val:.6f}\n")
    
    gd_file = os.path.join(cfg.out_dir, "val_genomedisco_epochs.txt")
    with open(gd_file, 'w') as f:
        f.write("epoch,genomedisco\n")
        for epoch, _, _, gd_val in val_metrics_epochs:
            f.write(f"{epoch},{gd_val:.6f}\n")

    print("Training complete.")


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="HiCFoundationGAN training with variable transformer depth")
    p.add_argument("--train_npz", type=str)
    p.add_argument("--valid_npz", type=str)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--num_workers", type=int)
    p.add_argument("--amp", type=int)  # 1/0

    # Loss weights
    p.add_argument("--adv_weight", type=float)
    p.add_argument("--l1_weight", type=float)
    p.add_argument("--feat_weight", type=float)

    # Opt
    p.add_argument("--lr_g", type=float)
    p.add_argument("--lr_d", type=float)

    # Generator
    p.add_argument("--gen_module", type=str)
    p.add_argument("--gen_class", type=str)
    p.add_argument("--gen_kwargs", type=str, help="JSON string for generator kwargs, e.g. '{\"num_channels\":64}'")

    # Foundation trunk
    p.add_argument("--foundation_ctor_module", type=str, help="Module path of HiCFoundation backbone class")
    p.add_argument("--foundation_ctor_class", type=str, help="Class name of HiCFoundation backbone")
    p.add_argument("--foundation_ckpt", type=str)
    p.add_argument("--freeze_foundation", type=int)  # 1/0 (ignored, always False for depth ablation)
    p.add_argument("--num_frozen_layers", type=int, default=0, help="Number of transformer blocks to freeze (always 0 for depth ablation)")
    p.add_argument("--num_transformer_layers", type=int, default=24, help="Number of transformer blocks to use (1-24)")
    p.add_argument("--d_hidden", type=int)

    # Misc
    p.add_argument("--out_dir", type=str)
    p.add_argument("--seed", type=int)

    args = p.parse_args()
    cfg = CFG()
    for k, v in vars(args).items():
        if v is None:
            continue
        if k in {"amp", "freeze_foundation"}:
            setattr(cfg, k, bool(v))
        elif k == "gen_kwargs":
            setattr(cfg, k, json.loads(v))
        else:
            setattr(cfg, k, v)
    
    # Ensure num_frozen_layers is always 0 for depth ablation
    cfg.num_frozen_layers = 0
    cfg.freeze_foundation = False
    
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    train(cfg)

