"""
Discriminators for Plug&Play GAN.
- D_local: PatchGAN for 40x40 tiles
- D_global: HiCFoundation-based discriminator for 224x224 crops
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path to import HiCFoundation modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from inference.load_model import load_model, format_input


class LocalDiscriminator(nn.Module):
    """
    PatchGAN discriminator for 40x40 tiles.
    """
    def __init__(self, in_channels=1, num_filters=64):
        super(LocalDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, num_filters, normalize=False),
            *discriminator_block(num_filters, num_filters * 2),
            *discriminator_block(num_filters * 2, num_filters * 4),
            *discriminator_block(num_filters * 4, num_filters * 8, stride=1),
            nn.Conv2d(num_filters * 8, 1, 3, padding=1)
        )
    
    def forward(self, img):
        return self.model(img)


class GlobalDiscriminator(nn.Module):
    """
    Global discriminator using HiCFoundation encoder + GAN head.
    """
    def __init__(self, hicfoundation_path, freeze_encoder=True, freeze_layers=None):
        """
        Args:
            hicfoundation_path: Path to HiCFoundation checkpoint
            freeze_encoder: Whether to freeze the encoder
            freeze_layers: Number of layers to freeze (None = freeze all)
        """
        super(GlobalDiscriminator, self).__init__()
        
        # Load HiCFoundation model
        # The load_model returns a Finetune_Model_Head which contains vit_backbone
        self.model = load_model(hicfoundation_path, input_row_size=224, input_col_size=224, task=3)
        
        # Access the backbone (encoder) from the model
        # The model structure is: model.vit_backbone contains the ViT encoder
        self.backbone = self.model.vit_backbone
        
        # Freeze encoder if requested
        if freeze_encoder:
            if freeze_layers is None:
                # Freeze entire encoder backbone
                for param in self.backbone.parameters():
                    param.requires_grad = False
            else:
                # Freeze first N layers
                for i, block in enumerate(self.backbone.blocks):
                    if i < freeze_layers:
                        for param in block.parameters():
                            param.requires_grad = False
        
        # GAN head: takes encoder features and outputs real/fake
        # The encoder outputs features of shape [B, N_patches+1, embed_dim]
        # We'll use the CLS token or average pool
        embed_dim = self.backbone.embed_dim
        self.gan_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
        
        self.format_input = format_input
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: Input tensor [B, 1, 224, 224]
            return_features: Whether to return intermediate features for feature matching
        
        Returns:
            logits: [B, 1] real/fake logits
            features: (optional) intermediate features for feature matching
        """
        # Format input for HiCFoundation (convert to RGB, normalize)
        # Create device-aware format_input to handle GPU tensors
        device = x.device
        x_squeezed = x.squeeze(1)  # [B, 224, 224]
        
        # Handle NaN values
        x_squeezed = torch.nan_to_num(x_squeezed)
        max_value = torch.max(x_squeezed).item()  # Get scalar value
        
        # Log transform
        x_log = torch.log10(x_squeezed + 1)  # [B, 224, 224]
        max_value_log = float(torch.log10(torch.tensor(max_value + 1, device=device)).item())
        
        # Debug: verify x_log shape
        if len(x_log.shape) != 3:
            raise ValueError(f"x_log should have 3 dimensions [B, H, W], got {x_log.shape}. x_squeezed shape was {x_squeezed.shape}")
        
        # Vectorized RGB conversion (more efficient and avoids shape issues)
        # Create RGB channels: [B, 224, 224] -> [B, 3, 224, 224]
        data_red = torch.ones_like(x_log)  # [B, 224, 224]
        data_log1 = (max_value_log - x_log) / max_value_log  # [B, 224, 224]
        
        # Ensure we have correct shapes before stacking
        assert len(data_red.shape) == 3, f"data_red shape: {data_red.shape}"
        assert len(data_log1.shape) == 3, f"data_log1 shape: {data_log1.shape}"
        
        # Stack along channel dimension: [B, 3, 224, 224]
        x_rgb = torch.stack([data_red, data_log1, data_log1], dim=1)
        
        # Verify final shape
        if len(x_rgb.shape) != 4:
            raise ValueError(f"x_rgb has wrong shape: {x_rgb.shape}, expected [B, 3, 224, 224]")
        if x_rgb.shape[1] != 3:
            raise ValueError(f"x_rgb has wrong channels: {x_rgb.shape[1]}, expected 3. Full shape: {x_rgb.shape}")
        
        # Normalize with ImageNet stats
        # Ensure mean and std are 4D tensors: [1, 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=x_rgb.dtype)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=x_rgb.dtype)
        mean = mean.view(1, 3, 1, 1)  # [1, 3, 1, 1]
        std = std.view(1, 3, 1, 1)    # [1, 3, 1, 1]
        
        # Debug: check shapes before normalization
        if len(mean.shape) != 4 or mean.shape != (1, 3, 1, 1):
            raise ValueError(f"mean has wrong shape: {mean.shape}, expected (1, 3, 1, 1)")
        if len(std.shape) != 4 or std.shape != (1, 3, 1, 1):
            raise ValueError(f"std has wrong shape: {std.shape}, expected (1, 3, 1, 1)")
        
        x_formatted = (x_rgb - mean) / std  # [B, 3, 224, 224]
        
        # Ensure correct shape: [B, 3, 224, 224]
        if len(x_formatted.shape) != 4:
            raise ValueError(f"Expected x_formatted to have 4 dimensions, got {len(x_formatted.shape)}: {x_formatted.shape}. x_rgb shape was {x_rgb.shape}, mean shape was {mean.shape}, std shape was {std.shape}")
        if x_formatted.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {x_formatted.shape[1]}. Full shape: {x_formatted.shape}")
        
        # Get encoder features from backbone using forward_features
        # The backbone has forward_features method that handles patch embedding, CLS token, etc.
        total_count = torch.ones(x_formatted.shape[0], device=x_formatted.device) * 1e9
        x = self.backbone.forward_features(x_formatted, total_count)  # [B, N_patches+1, embed_dim]
        
        # Extract CLS token (first token) for classification
        cls_features = x[:, 0]  # [B, embed_dim]
        
        # GAN head
        logits = self.gan_head(cls_features)  # [B, 1]
        
        if return_features:
            # Return intermediate features for feature matching
            # Use the full sequence (excluding CLS token) or average pool
            # For feature matching, we'll use the patch tokens (excluding CLS)
            patch_features = x[:, 1:]  # [B, N_patches, embed_dim]
            # Average pool over patches
            features = patch_features.mean(dim=1)  # [B, embed_dim]
            return logits, features
        else:
            return logits
