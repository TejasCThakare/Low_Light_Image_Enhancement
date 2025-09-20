import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------
# convolutional block
# -----------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, activation=nn.ReLU(True)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = activation

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# -----------------------------------
# Decomposition Network (R + I)
# -----------------------------------
class DecompositionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.Conv2d(64, 4, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layers(x)
        R, I = out[:, :3, :, :], out[:, 3:, :, :]
        return R, I

# -----------------------------------
# RelightNet with spatial attention
# -----------------------------------
class RelightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(1, 16)
        self.conv2 = ConvBlock(16, 32)
        self.conv3 = ConvBlock(32, 64)

        self.attn1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 64, 1),
            nn.Sigmoid()
        )

        self.upsample = nn.Sequential(
            ConvBlock(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBlock(32, 16),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, I):
        x1 = self.conv1(I)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        attn = self.attn1(x3)
        x3 = x3 * attn  # apply spatial attention

        out = self.upsample(x3)
        out = F.interpolate(out, size=I.shape[2:], mode='bilinear', align_corners=False)
        return out

# -----------------------------------
# Fusion with guided attention
# -----------------------------------
class GuidedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(6, 16),  # R (3) + I_enh (3) = 6
            ConvBlock(16, 16),
            nn.Conv2d(16, 3, 3, padding=1)
        )

    def forward(self, R, I_enh):
        x = torch.cat([R, I_enh.expand_as(R)], dim=1)
        return self.conv(x)

# -----------------------------------
# Multi-scale residual block for refinement
# -----------------------------------
class MultiScaleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.branch2 = nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2)
        self.branch3 = nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=3)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.fuse(out)

# -----------------------------------
# Refiner using multi-scale context
# -----------------------------------
class RefinerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            ConvBlock(3, 32),
            MultiScaleResBlock(32, 32),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encode(x)
        return (x + 1) / 2  # [0, 1] output

# -----------------------------------
# Full Retinex++ Hybrid Model
# -----------------------------------
class RetinexEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom_net = DecompositionNet()
        self.relight_net = RelightNet()
        self.fusion = GuidedFusion()
        self.refiner = RefinerNet()

    def forward(self, x):
        R, I = self.decom_net(x)
        I_enh = self.relight_net(I)
        fusion_out = self.fusion(R, I_enh)
        final_out = self.refiner(fusion_out)
        return final_out, R, I, I_enh
