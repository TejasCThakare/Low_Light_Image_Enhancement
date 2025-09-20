import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------
# 🔹 KinD++: Residual Block with InstanceNorm
# -------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

# -------------------------------------------------------
# 🔹 KinD++: DecompositionNet with Encoder-Decoder + Skip
# -------------------------------------------------------
class DecompositionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(64)
        )
        self.middle = ResBlock(64)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.middle(feat)
        out = self.decoder(feat)
        R, I = out[:, :3, :, :], out[:, 3:, :, :]
        return R, I

# -------------------------------------------------------
# 🔹 Zero-DCE++: Curve Estimation Network
# -------------------------------------------------------
class CurveEstimationNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

def enhance_with_curve(I, curve, steps=8):
    x = I
    for _ in range(steps):
        x = x + curve * (x * (1 - x))
    return x

# -------------------------------------------------------
# 🔹 KinD++: RelightNet with Residual Blocks + Curve
# -------------------------------------------------------
class RelightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.res1 = ResBlock(1)
        self.res2 = ResBlock(1)
        self.curve_est = CurveEstimationNet()

    def forward(self, I):
        x = self.res1(I)
        x = self.res2(x)
        curve = self.curve_est(x)
        I_enh = enhance_with_curve(I, curve)
        return I_enh, curve

# -------------------------------------------------------
# 🔹 KinD++: Gated Fusion Network
# -------------------------------------------------------
class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(6, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, R, I_enh):
        I3 = I_enh.expand_as(R)
        x = torch.cat([R, I3], dim=1)
        gate = self.gate(x)
        fused = self.fusion(x)
        return gate * fused + (1 - gate) * R

# -------------------------------------------------------
# 🔹 MIRNet: Residual Refinement Block (RRB)
# -------------------------------------------------------
class RRB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        identity = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x + identity

# -------------------------------------------------------
# 🔹 MIRNet: Squeeze-and-Excitation Attention
# -------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w

# -------------------------------------------------------
# 🔹 MIRNet: Multi-Scale Fusion Attention (MFFA)
# -------------------------------------------------------
class MFFA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Conv2d(channels, channels, 5, padding=2),
            nn.Conv2d(channels, channels, 7, padding=3)
        ])
        self.fuse = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x):
        out = [branch(x) for branch in self.branches]
        out = torch.cat(out, dim=1)
        return self.fuse(out)

# -------------------------------------------------------
# 🔹 Full MIRNet-style Refiner
# -------------------------------------------------------
class RefinerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Conv2d(3, 64, 3, padding=1)
        self.block = nn.Sequential(
            RRB(64),
            MFFA(64),
            SEBlock(64),
            RRB(64)
        )
        self.output = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.block(x)
        x = self.output(x)
        return (x + 1) / 2  # Normalize to [0, 1]

# -------------------------------------------------------
# 🔹 Full Retinex++ L3-Pro Model (KinD++ + Zero-DCE++ + MIRNet)
# -------------------------------------------------------
class RetinexEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = DecompositionNet()
        self.relight = RelightNet()
        self.fusion = FusionNet()
        self.refiner = RefinerNet()

    def forward(self, x):
        R, I = self.decom(x)
        I_enh, curve = self.relight(I)
        fused = self.fusion(R, I_enh)
        final = self.refiner(fused)
        return final, R, I, I_enh, curve
