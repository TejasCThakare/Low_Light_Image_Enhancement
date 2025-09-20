import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from torchvision import models
from pytorch_msssim import ms_ssim
from kornia.color import rgb_to_lab
# from kornia.losses import ssim
from kornia.losses import SSIMLoss


# ----------------------------
# 🔹 L1 Pixel Loss
# ----------------------------
def l1_loss(pred, target):
    return F.l1_loss(pred, target)

# ----------------------------
# 🔹 LAB Color Loss
# ----------------------------
def lab_color_loss(pred, target):
    lab_pred = rgb_to_lab(pred)
    lab_target = rgb_to_lab(target)
    return F.l1_loss(lab_pred, lab_target)

# ----------------------------
# 🔹 LPIPS Perceptual Loss
# ----------------------------
class LPIPSLoss(nn.Module):
    def __init__(self, net='alex'):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net)

    def forward(self, pred, target):
        return self.lpips(pred, target).mean()

# ----------------------------
# 🔹 VGG Feature Loss (Perceptual)
# ----------------------------
class VGGLoss(nn.Module):
    def __init__(self, layer='relu3_3'):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        x = (x - 0.5) * 2
        y = (y - 0.5) * 2
        return F.l1_loss(self.vgg(x), self.vgg(y))

# ----------------------------
# 🔹 MS-SSIM Loss (Perceptual + Structural)
# ----------------------------
def msssim_loss(pred, target):
    if pred.shape[-1] < 160 or pred.shape[-2] < 160:
        return 1.0 - ssim(pred, target, window_size=11, max_val=1.0)
    else:
        return 1.0 - ms_ssim(pred, target, data_range=1.0, size_average=True)

# ----------------------------
# 🔹 Histogram Matching Loss
# ----------------------------
def histogram_loss(pred, target, bins=64):
    hist_pred = torch.histc(pred, bins=bins, min=0.0, max=1.0)
    hist_target = torch.histc(target, bins=bins, min=0.0, max=1.0)
    hist_pred /= hist_pred.sum()
    hist_target /= hist_target.sum()
    return F.l1_loss(hist_pred, hist_target)

# ----------------------------
# 🔹 Tone Contrast Loss
# ----------------------------
def tone_contrast_loss(pred, target):
    contrast_pred = torch.std(pred, dim=[2, 3])
    contrast_target = torch.std(target, dim=[2, 3])
    return F.l1_loss(contrast_pred, contrast_target)

# ----------------------------
# 🔹 Reflectance Edge Loss (KinD++)
# ----------------------------
def edge_loss(R, target):
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32, device=R.device)
    sobel_x = sobel_x.expand(3, 1, 3, 3)
    grad_R = F.conv2d(R, sobel_x, padding=1, groups=3)
    grad_target = F.conv2d(target, sobel_x, padding=1, groups=3)
    return F.l1_loss(grad_R, grad_target)

# ----------------------------
# 🔹 Illumination Smoothness Loss (KinD++)
# ----------------------------
def illumination_smoothness_loss(I, R):
    grad_Ix = torch.abs(I[:, :, :, :-1] - I[:, :, :, 1:])
    grad_Iy = torch.abs(I[:, :, :-1, :] - I[:, :, 1:, :])
    grad_Rx = torch.mean(torch.abs(R[:, :, :, :-1] - R[:, :, :, 1:]), dim=1, keepdim=True)
    grad_Ry = torch.mean(torch.abs(R[:, :, :-1, :] - R[:, :, 1:, :]), dim=1, keepdim=True)
    weight_x = torch.exp(-10 * grad_Rx)
    weight_y = torch.exp(-10 * grad_Ry)
    return torch.mean(grad_Ix * weight_x) + torch.mean(grad_Iy * weight_y)

# ----------------------------
# 🔹 Dual Reflectance Supervision Loss (KinD++)
# ----------------------------
def reflectance_consistency_loss(R_low, R_high):
    return F.l1_loss(R_low, R_high)

# ----------------------------
# 🔹 Total Variation Loss (Zero-DCE++)
# ----------------------------
def total_variation_loss(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

# ----------------------------
# 🔹 Curve Regularization Loss (Zero-DCE++)
# ----------------------------
def curve_regularization(curve):
    return torch.mean(torch.abs(curve))

# ----------------------------
# 🔹 SSIM-based Reflectance Loss (MIRNet/KinD++)

# ----------------------------
def ssim_on_reflectance(R_pred, R_gt):
    ssim_loss = SSIMLoss(window_size=11)
    return ssim_loss(R_pred, R_gt)
# def ssim_on_reflectance(R_pred, R_gt):
#     return 1.0 - ssim(R_pred, R_gt, window_size=11, max_val=1.0)

