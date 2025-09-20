import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from retinex_model import RetinexEnhancer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor_to_np(t):
    return (t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype('uint8')

def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size: img = img.resize((size, size))
    return transforms.ToTensor()(img)

def evaluate_model(model, val_data_dir, patch_size=512, save_dir=None, save_intermediates=False):
    model.eval()
    low_dir = os.path.join(val_data_dir, "low")
    high_dir = os.path.join(val_data_dir, "high")
    image_names = sorted(os.listdir(low_dir))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if save_intermediates:
            os.makedirs(os.path.join(save_dir, "R"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "I"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "I_enh"), exist_ok=True)

    total_psnr, total_ssim = 0.0, 0.0

    for name in tqdm(image_names, desc="🔍 Evaluating"):
        low_img = load_image(os.path.join(low_dir, name), patch_size).unsqueeze(0).to(device)
        high_img = load_image(os.path.join(high_dir, name), patch_size)

        with torch.no_grad():
            pred, R, I, I_enh = model(low_img)
            pred = pred[0].clamp(0, 1)

        if save_dir:
            save_image(pred, os.path.join(save_dir, name))
            if save_intermediates:
                save_image(R[0], os.path.join(save_dir, "R", name))
                save_image(I[0].expand(3, -1, -1), os.path.join(save_dir, "I", name))
                save_image(I_enh[0].expand(3, -1, -1), os.path.join(save_dir, "I_enh", name))

        total_psnr += psnr(tensor_to_np(high_img), tensor_to_np(pred), data_range=255)
        total_ssim += ssim(tensor_to_np(high_img), tensor_to_np(pred), data_range=255, channel_axis=-1)

    avg_psnr = total_psnr / len(image_names)
    avg_ssim = total_ssim / len(image_names)
    return avg_psnr, avg_ssim


#  Optional CLI usage from main.py (if --mode eval is used)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Validation directory with low/high folders")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained model .pth")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="results_eval")
    parser.add_argument("--save_intermediates", action="store_true")
    args = parser.parse_args()

    model = RetinexEnhancer().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    psnr_val, ssim_val = evaluate_model(
        model,
        val_data_dir=args.data_dir,
        patch_size=args.patch_size,
        save_dir=args.save_dir,
        save_intermediates=args.save_intermediates
    )

    print(f"\n Evaluation Complete")
    print(f" PSNR: {psnr_val:.2f}")
    print(f" SSIM: {ssim_val:.4f}")
