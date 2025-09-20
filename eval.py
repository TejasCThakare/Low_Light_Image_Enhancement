import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from retinex_model import RetinexEnhancer
from losses import LPIPSLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor_to_np(t):
    return (t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype('uint8')

def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size: img = img.resize((size, size))
    return transforms.ToTensor()(img)

def evaluate_model(model, val_data_dir, patch_size=512, save_dir=None,
                   save_intermediates=False, calc_lpips=False):
    model.eval()
    lpips_loss = LPIPSLoss().to(device) if calc_lpips else None

    low_dir = os.path.join(val_data_dir, "low")
    high_dir = os.path.join(val_data_dir, "high")
    image_names = sorted(os.listdir(low_dir))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if save_intermediates:
            os.makedirs(os.path.join(save_dir, "R"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "I"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "I_enh"), exist_ok=True)

    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0

    for name in tqdm(image_names, desc="🔍 Evaluating"):
        low_img = load_image(os.path.join(low_dir, name), patch_size).unsqueeze(0).to(device)
        high_img = load_image(os.path.join(high_dir, name), patch_size)

        with torch.no_grad():
            pred, R, I, I_enh, _ = model(low_img)
            pred = pred[0].clamp(0, 1)

        if save_dir:
            save_image(pred, os.path.join(save_dir, name))
            if save_intermediates:
                save_image(R[0], os.path.join(save_dir, "R", name))
                save_image(I[0].expand(3, -1, -1), os.path.join(save_dir, "I", name))
                save_image(I_enh[0].expand(3, -1, -1), os.path.join(save_dir, "I_enh", name))

        pred_np = tensor_to_np(pred)
        gt_np = tensor_to_np(high_img)

        total_psnr += psnr(gt_np, pred_np, data_range=255)
        total_ssim += ssim(gt_np, pred_np, data_range=255, channel_axis=-1)

        if calc_lpips:
            total_lpips += lpips_loss(pred.unsqueeze(0), high_img.unsqueeze(0).to(device)).item()

    avg_psnr = total_psnr / len(image_names)
    avg_ssim = total_ssim / len(image_names)
    avg_lpips = total_lpips / len(image_names) if calc_lpips else None

    return avg_psnr, avg_ssim, avg_lpips
