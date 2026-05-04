# Code was taken from http://dna.cs.miami.edu/HiCNN2/
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_ReLU_Block(nn.Module):
  def __init__(self):
    super(Conv_ReLU_Block, self).__init__()
    self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    return self.relu(self.conv(x))


class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.net1_conv1 = nn.Conv2d(1, 64, 13)
    self.net1_conv2 = nn.Conv2d(64, 64, 1)
    self.net1_conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
    self.net1_conv4R = nn.Conv2d(128, 128, 3, padding=1, bias=False)
    self.net1_conv5 = nn.Conv2d(128 * 25, 1000, 1, padding=0, bias=True)
    self.net1_conv6 = nn.Conv2d(1000, 64, 1, padding=0, bias=True)
    self.net1_conv7 = nn.Conv2d(64, 1, 3, padding=1, bias=False)

    self.net2_conv1 = nn.Conv2d(1, 8, 13)
    self.net2_conv2 = nn.Conv2d(8, 1, 1)
    self.residual_layer_vdsr = self.make_layer(Conv_ReLU_Block, 18)
    self.input_vdsr = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    self.output_vdsr = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

    self.net3_conv1 = nn.Conv2d(1, 8, 9)
    self.net3_conv2 = nn.Conv2d(8, 8, 1)
    self.net3_conv3 = nn.Conv2d(8, 1, 5)

    self.relu = nn.ReLU(inplace=True)

    self.weights = nn.Parameter((torch.ones(1, 3) / 3), requires_grad=True)
    # He initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

  def make_layer(self, block, num_of_layer):
    layers = []
    for _ in range(num_of_layer):
      layers.append(block())
    return nn.Sequential(*layers)

  def forward(self, input):
    # ConvNet1
    x = self.relu(self.net1_conv1(input))
    x = self.relu(self.net1_conv2(x))
    residual = x
    x2 = self.net1_conv3(x)
    output1 = x2
    outtmp = []
    for i in range(25):
      output1 = self.net1_conv4R(self.relu(self.net1_conv4R(self.relu(output1))))
      output1 = torch.add(output1, x2)
      outtmp.append(output1)
    output1 = torch.cat(outtmp, 1)
    output1 = self.net1_conv5(output1)
    output1 = self.net1_conv6(output1)
    output1 = torch.add(output1, residual)
    output1 = self.net1_conv7(output1)

    # ConvNet2
    x_vdsr = self.relu(self.net2_conv1(input))
    x_vdsr = self.relu(self.net2_conv2(x_vdsr))
    residual2 = x_vdsr
    output2 = self.relu(self.input_vdsr(x_vdsr))
    output2 = self.residual_layer_vdsr(output2)
    output2 = self.output_vdsr(output2)
    output2 = torch.add(output2, residual2)

    # ConvNet3
    output3 = self.net3_conv1(input)
    output3 = F.relu(output3)
    output3 = self.net3_conv2(output3)
    output3 = F.relu(output3)
    output3 = self.net3_conv3(output3)
    output3 = F.relu(output3)

    # w1*output1 + w2*output2 + w3*output3
    w = self.weights / (self.weights.sum(dim=1, keepdim=True) + 1e-8)
    output = output1 * w[0,0] + output2 * w[0,1] + output3 * w[0,2]


    return output


class Generator40x40(nn.Module):
  """
  Wrapper for HiCNN Generator to handle 40x40 input/output.
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
