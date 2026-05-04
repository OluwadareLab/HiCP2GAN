import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1) * 0.1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out

class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Basic_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Residual_Block(nn.Module):
    def __init__(self, num_channels):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=2, dilation=2)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out

class Cascading_Block(nn.Module):
    def __init__(self, channels):
        super(Cascading_Block, self).__init__()
        self.r1 = Residual_Block(channels)
        self.r2 = Residual_Block(channels)
        self.r3 = Residual_Block(channels)
        self.c1 = Basic_Block(channels * 2, channels)
        self.c2 = Basic_Block(channels * 3, channels)
        self.c3 = Basic_Block(channels * 4, channels)

    def forward(self, x):
        c0 = o0 = x
        b1 = self.r1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        b2 = self.r2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        b3 = self.r3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        return o3

class Generator(nn.Module):
    def __init__(self, num_channels):
        super(Generator, self).__init__()
        self.entry = nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1)
        self.cb1 = Cascading_Block(num_channels)
        self.att1 = SelfAttention(num_channels)
        self.cb2 = Cascading_Block(num_channels)
        self.att2 = SelfAttention(num_channels)
        self.cb3 = Cascading_Block(num_channels)
        self.cb4 = Cascading_Block(num_channels)
        self.cb5 = Cascading_Block(num_channels)
        self.cv1 = nn.Conv2d(num_channels * 2, num_channels, kernel_size=1)
        self.cv2 = nn.Conv2d(num_channels * 3, num_channels, kernel_size=1)
        self.cv3 = nn.Conv2d(num_channels * 4, num_channels, kernel_size=1)
        self.cv4 = nn.Conv2d(num_channels * 5, num_channels, kernel_size=1)
        self.cv5 = nn.Conv2d(num_channels * 6, num_channels, kernel_size=1)
         # 3x3 exit convolution layer
        # self.exit1 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
        self.exit1 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
        self.exit = nn.Conv2d(num_channels, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.cb1(o0)
        a1 = self.att1(b1)
        c1 = torch.cat([c0, a1], dim=1)
        o1 = self.cv1(c1)

        b2 = self.cb2(o1)
        a2 = self.att2(b2)
        c2 = torch.cat([c1, a2], dim=1)
        o2 = self.cv2(c2)

        b3 = self.cb3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.cv3(c3)

        b4 = self.cb4(o3)
        c4 = torch.cat([c3, b4], dim=1)
        o4 = self.cv4(c4)

        b5 = self.cb5(o4)
        c5 = torch.cat([c4, b5], dim=1)
        o5 = self.cv5(c5)

        o6 = self.exit1(o5)
        o7 = self.exit1(o6)
        out = self.exit(o7)
        return out

