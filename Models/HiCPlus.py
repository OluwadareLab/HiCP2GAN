# Code was taken from https://github.com/wangjuan001/hicplus
import torch.nn as nn
import torch.nn.functional as F

conv2d1_filters_numbers = 8
conv2d1_filters_size = 9
conv2d2_filters_numbers = 8
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 5

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, conv2d1_filters_numbers, conv2d1_filters_size)
        self.conv2 = nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers, conv2d2_filters_size)
        self.conv3 = nn.Conv2d(conv2d2_filters_numbers, 1, conv2d3_filters_size)

    def forward(self, x):
        #print("start forwardingf")
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        return x


class Generator40x40(nn.Module):
    """
    Wrapper for HiCPlus Generator to handle 40x40 input/output.
    The original generator expects 28x28 input and outputs 28x28.
    This wrapper resizes 40x40 to 28x28 (preserving information via area interpolation),
    runs the generator, then upsamples back to 40x40.
    """
    def __init__(self):
        super(Generator40x40, self).__init__()
        self.generator = Generator()
    
    def forward(self, x):
        # x shape: (B, 1, 40, 40)
        # Resize to 28x28 using area interpolation (preserves averages, better than crop)
        x_28 = F.interpolate(x, size=(28, 28), mode='area')  # (B, 1, 28, 28)
        
        # Pass through original generator
        out_28x28 = self.generator(x_28)  # (B, 1, 28, 28)
        
        # Upsample back to 40x40 using nearest interpolation (no trainable parameters)
        out_40x40 = F.interpolate(out_28x28, size=(40, 40), mode='nearest')  # (B, 1, 40, 40)
        
        return out_40x40
