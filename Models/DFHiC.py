# Code converted from TensorFlow/TensorLayer to PyTorch
# Original: TensorFlow/TensorLayer implementation
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    DFHiC Generator - PyTorch implementation.
    
    Architecture:
    - Multiple dilated convolution layers with residual connections
    - Uses dilation_rate=2 for dilated convolutions
    - Final residual connection adds output to input
    """
    def __init__(self):
        super(Generator, self).__init__()
        
        # Weight initialization (stddev=0.02)
        w_init = lambda m: self._init_weights(m, std=0.02)
        
        # First block: 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2, dilation=2)
        
        # Second block: 64 channels
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        
        # Third block: 128 channels
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        
        # Fourth block: 256 channels
        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2)
        
        # Final output: 1 channel
        self.conv_out = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
        # Apply weight initialization
        self.apply(w_init)
    
    @staticmethod
    def _init_weights(m, std=0.02):
        """Initialize weights with normal distribution (std=0.02)."""
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass through DFHiC generator.
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
        
        Returns:
            Output tensor of shape (B, 1, H, W) with residual connection
        """
        # Store input for residual connection
        x_input = x
        
        # First block: 32 channels with residual
        n_0 = self.relu(self.conv1(x))
        n_1 = self.relu(self.conv2(n_0))
        n_2 = self.relu(self.conv3(n_1))
        n_3 = n_0 + n_2  # Residual connection
        
        # Second block: 64 channels with residual
        n_4 = self.relu(self.conv4(n_3))
        n_5 = self.relu(self.conv5(n_4))
        n_6 = n_4 + n_5  # Residual connection
        
        # Third block: 128 channels with residual
        n_7 = self.relu(self.conv6(n_6))
        n_8 = self.relu(self.conv7(n_7))
        n_9 = n_7 + n_8  # Residual connection
        
        # Fourth block: 256 channels
        n_10 = self.relu(self.conv8(n_9))
        n_11 = self.relu(self.conv9(n_10))
        
        # Final output
        n = self.conv_out(n_11)
        
        # Add residual connection (output + input)
        out = n + x_input
        
        return out


class Discriminator(nn.Module):
    """
    Placeholder for DFHiC Discriminator (if needed in future).
    Currently not used in standalone validation.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        # Discriminator not implemented as it's not used in standalone training
        pass
    
    def forward(self, x):
        raise NotImplementedError("Discriminator not implemented for DFHiC")
