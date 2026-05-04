"""
PyTorch reimplementation of hicGAN (Liu & Wang, 2019).
Original: TensorFlow 1.x with TensorLayer.
Architecture: SRResNet-style generator with 5 residual blocks + PatchGAN discriminator.
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, num_res_blocks=5, num_channels=64):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(1, num_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )
        self.mid = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(num_channels),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(num_channels, 128, 3, 1, 1),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(256, 1, 1, 1, 0),
            nn.Tanh(),
        )

    def forward(self, x):
        entry = self.entry(x)
        res = self.res_blocks(entry)
        mid = self.mid(res) + entry
        return self.tail(mid)


class Discriminator(nn.Module):
    """Conv discriminator matching the original hicGAN architecture."""

    def __init__(self):
        super().__init__()

        def conv_bn_lrelu(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            conv_bn_lrelu(64, 64, 2),    # -> H/2
            conv_bn_lrelu(64, 64, 1),
            conv_bn_lrelu(64, 64, 2),    # -> H/4
            conv_bn_lrelu(64, 64, 1),
            conv_bn_lrelu(64, 64, 2),    # -> H/8
            conv_bn_lrelu(64, 128, 1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat)
