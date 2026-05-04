"""
PyTorch reimplementation of EnHiC (Hu & Ma, 2021).
Original: TensorFlow 2.x with custom layers for rank-1 matrix decomposition.
Key ideas: multi-scale processing, rank-1 decomposition/reconstruction, symmetry enforcement.
Adapted to work with 40x40 Hi-C contact matrix patches.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Downpixel(nn.Module):
    """Space-to-depth with average pooling (matches TF space_to_depth).
    Uses asymmetric 'SAME' padding to preserve spatial dims before pixel_unshuffle."""

    def __init__(self, r):
        super().__init__()
        self.r = r
        self.register_buffer(
            'avg_kernel', torch.ones(1, 1, r, r) / (r * r)
        )
        total_pad = r - 1
        self.pad_before = total_pad // 2
        self.pad_after = total_pad - self.pad_before

    def forward(self, x):
        x = F.pad(x, [self.pad_before, self.pad_after, self.pad_before, self.pad_after])
        x = F.conv2d(x, self.avg_kernel.expand(x.size(1), -1, -1, -1), groups=x.size(1))
        return F.pixel_unshuffle(x, self.r)


class Subpixel(nn.Module):
    """Conv2d followed by depth-to-space (PixelShuffle)."""

    def __init__(self, in_channels, out_channels, kernel_size, r, padding='same'):
        super().__init__()
        pad = kernel_size // 2 if padding == 'same' else 0
        self.conv = nn.Conv2d(in_channels, out_channels * r * r, kernel_size,
                              padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * r * r)
        self.shuffle = nn.PixelShuffle(r)
        nn.init.normal_(self.conv.weight, mean=0.01, std=0.1)

    def forward(self, x):
        return self.shuffle(F.relu(self.bn(self.conv(x))))


class Reconstruct_R1M(nn.Module):
    """Rank-1 matrix reconstruction: v * v^T with learnable channel weights."""

    def __init__(self, channels):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        # x: (B, C, H, 1) — column vector per channel
        v = x + 1e-6
        vt = v.transpose(2, 3)          # (B, C, 1, H)
        return v * vt * self.w           # (B, C, H, H)


class Weight_R1M(nn.Module):
    """Channel-wise positive weighting."""

    def __init__(self, channels):
        super().__init__()
        self.w = nn.Parameter(torch.empty(1, channels, 1, 1).uniform_(0, 4))

    def forward(self, x):
        return x * F.relu(self.w)


class Symmetry_R1M(nn.Module):
    """Enforce matrix symmetry: keep upper triangle + transpose."""

    def __init__(self, size):
        super().__init__()
        ones = torch.ones(size, size)
        diag = torch.diag(torch.ones(size)) * 0.5
        mask = torch.triu(ones) - diag
        self.register_buffer('mask', mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        up = x * self.mask
        return up + up.transpose(2, 3)


class Sum_R1M(nn.Module):
    """Sum across channels to produce single-channel output."""

    def forward(self, x):
        return x.sum(dim=1, keepdim=True)


class Normal(nn.Module):
    """Row/column normalization with learnable weights."""

    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, 1, dim, 1))

    def forward(self, x):
        row_norm = torch.sqrt((x * x).sum(dim=2, keepdim=True) + 1e-8)
        col_norm = torch.sqrt((x * x).sum(dim=3, keepdim=True) + 1e-8)
        div = x / (row_norm * col_norm + 1e-8)
        w = F.relu(self.w)
        wt = w.transpose(2, 3)
        return div * (w * wt)


class DownsampleDecomposition(nn.Module):
    """Downpixel → 1×W conv (full-width) → WeightR1M → ReconstructR1M."""

    def __init__(self, spatial_size, channels_decompose, in_channels, downsample_ratio):
        super().__init__()
        self.down = Downpixel(downsample_ratio)
        dp_channels = in_channels * downsample_ratio * downsample_ratio
        self.conv = nn.Conv2d(dp_channels, channels_decompose,
                              kernel_size=(1, spatial_size), bias=False)
        nn.init.normal_(self.conv.weight, mean=0.01, std=0.1)
        self.weight = Weight_R1M(channels_decompose)
        self.reconstruct = Reconstruct_R1M(channels_decompose)

    def forward(self, x):
        x = self.down(x)
        x = F.relu(self.conv(x))       # (B, C, H, 1)
        x = self.weight(x)
        return self.reconstruct(x)     # (B, C, H, H)


class Rank1ChannelConv(nn.Module):
    """1×1 conv → WeightR1M → SymmetryR1M."""

    def __init__(self, in_channels, out_channels, spatial_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.weight = Weight_R1M(out_channels)
        self.symmetry = Symmetry_R1M(spatial_size)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.weight(x)
        return self.symmetry(x)


class UpsampleConv(nn.Module):
    """Subpixel upsample → SymmetryR1M."""

    def __init__(self, in_channels, out_channels, r, output_size):
        super().__init__()
        self.subpixel = Subpixel(in_channels, out_channels, kernel_size=3, r=r)
        self.symmetry = Symmetry_R1M(output_size)

    def forward(self, x):
        return self.symmetry(self.subpixel(x))


class Rank1Estimation(nn.Module):
    """SumR1M → Normal."""

    def __init__(self, dim):
        super().__init__()
        self.sum = Sum_R1M()
        self.normal = Normal(dim)

    def forward(self, x):
        return self.normal(self.sum(x))


class Generator(nn.Module):
    """
    EnHiC generator for 40×40 Hi-C patches.
    Multi-scale (x2, x4) rank-1 decomposition with upsampling reconstruction.
    """

    def __init__(self, len_high_size=40, scale=4):
        super().__init__()
        len_x2 = len_high_size // (scale // 2)  # 20
        len_x4 = len_high_size // scale          # 10

        # x4 branch
        self.dsd_x4 = DownsampleDecomposition(len_x4, 384, 1, 4)
        self.r1c_x4 = Rank1ChannelConv(384, 128, len_x4)
        self.r1e_x4 = Rank1Estimation(len_x4)
        self.usc_x4 = UpsampleConv(128, 128, r=2, output_size=len_x2)

        # x2 branch
        self.dsd_x2 = DownsampleDecomposition(len_x2, 768, 1, 2)
        self.r1c_x2 = Rank1ChannelConv(768, 128, len_x2)
        self.r1e_x2 = Rank1Estimation(len_x2)

        # merge & upsample to full resolution
        self.cc = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        nn.init.uniform_(self.cc[0].weight, 0, 1.0 / 128)

        self.usc_x2 = UpsampleConv(128, 64, r=2, output_size=len_high_size)

        self.sum_high = nn.Conv2d(64, 1, 1, bias=False)
        nn.init.uniform_(self.sum_high.weight, 0, 1)
        self.out_high = Normal(len_high_size)

    def forward(self, x):
        # x4 path
        rech_x4 = self.dsd_x4(x)
        sym_x4 = self.r1c_x4(rech_x4)
        sym_x4 = self.usc_x4(sym_x4)

        # x2 path
        rech_x2 = self.dsd_x2(x)
        sym_x2 = self.r1c_x2(rech_x2)

        # merge
        concat = torch.cat([sym_x4, sym_x2], dim=1)
        concat = self.cc(concat)
        sym = self.usc_x2(concat)

        high_out = F.relu(self.sum_high(sym))
        high_out = self.out_high(high_out)
        return high_out


class DownConv(nn.Module):
    """Conv → LeakyReLU → MaxPool for discriminator."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
        )
        nn.init.normal_(self.block[0].weight, mean=0.01, std=0.1)

    def forward(self, x):
        return self.block(x)


class R1DecomposeReconstruct(nn.Module):
    """Rank-1 decompose + reconstruct (for discriminator)."""

    def __init__(self, spatial_size, channels_decompose, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels_decompose,
                              kernel_size=(1, spatial_size), bias=False)
        nn.init.normal_(self.conv.weight, mean=0.01, std=0.1)
        self.reconstruct = Reconstruct_R1M(channels_decompose)

    def forward(self, x):
        x = F.relu(self.conv(x))  # (B, C, H, 1)
        return self.reconstruct(x)


class Discriminator(nn.Module):
    """
    EnHiC multi-scale PatchGAN discriminator for 40×40 patches.
    Uses rank-1 decomposition at multiple scales.
    """

    def __init__(self, len_high_size=40, scale=4):
        super().__init__()
        len_x1 = len_high_size          # 40
        len_x2 = len_high_size // (scale // 2)  # 20
        len_x4 = len_high_size // scale  # 10
        len_x8 = len_high_size // (scale * 2)  # 5

        # x1 branch
        self.r1dr_x1 = R1DecomposeReconstruct(len_x1, 512, 1)
        self.dc_x1 = DownConv(512, 80)

        # x2 branch
        self.dp_x2 = Downpixel(2)
        self.r1dr_x2 = R1DecomposeReconstruct(len_x2, 512, 4)
        self.r1c_x2 = Rank1ChannelConv(512, 40, len_x2)

        self.dc_x2 = DownConv(120, 120)

        # x4 branch
        self.dp_x4 = Downpixel(4)
        self.r1dr_x4 = R1DecomposeReconstruct(len_x4, 256, 16)
        self.r1c_x4 = Rank1ChannelConv(256, 20, len_x4)

        self.dc_x4 = DownConv(140, 60)

        # x8 branch
        self.dp_x8 = Downpixel(8)
        self.r1dr_x8 = R1DecomposeReconstruct(len_x8, 128, 64)
        self.r1c_x8 = Rank1ChannelConv(128, 10, len_x8)

        self.dc_x8 = DownConv(70, 80)

        # final classification
        self.head = nn.Sequential(
            nn.Conv2d(80, 80, 1),
            nn.Flatten(),
            nn.Linear(80 * 2 * 2, 1),
        )

    def forward(self, x):
        # x1 path
        r1_x1 = self.r1dr_x1(x)
        dc_x1 = self.dc_x1(r1_x1)       # -> 20x20

        # x2 path
        dp_x2 = self.dp_x2(x)
        r1_x2 = self.r1dr_x2(dp_x2)
        r1c_x2 = self.r1c_x2(r1_x2)     # -> 20x20
        cat_x2 = torch.cat([r1c_x2, dc_x1], dim=1)
        dc_x2 = self.dc_x2(cat_x2)       # -> 10x10

        # x4 path
        dp_x4 = self.dp_x4(x)
        r1_x4 = self.r1dr_x4(dp_x4)
        r1c_x4 = self.r1c_x4(r1_x4)     # -> 10x10
        cat_x4 = torch.cat([r1c_x4, dc_x2], dim=1)
        dc_x4 = self.dc_x4(cat_x4)       # -> 5x5

        # x8 path
        dp_x8 = self.dp_x8(x)
        r1_x8 = self.r1dr_x8(dp_x8)
        r1c_x8 = self.r1c_x8(r1_x8)     # -> 5x5
        cat_x8 = torch.cat([r1c_x8, dc_x4], dim=1)
        dc_x8 = self.dc_x8(cat_x8)       # -> 2x2

        return self.head(dc_x8)
