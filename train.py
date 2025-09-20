import os
import torch
import torch.optim as optim
from tqdm import tqdm
from dataloader import get_loader
from retinex_model import RetinexEnhancer
from losses import (
    l1_loss, lab_color_loss, LPIPSLoss, VGGLoss,
    msssim_loss, edge_loss, histogram_loss, tone_contrast_loss,
    illumination_smoothness_loss, reflectance_consistency_loss,
    total_variation_loss, curve_regularization, ssim_on_reflectance
)
from utils import save_intermediates
from eval import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(data_dir, epochs=100, batch_size=4, patch_size=192, lr=1e-4,
          save_dir='checkpoints', fine_tune=False, resume_ckpt=None):

    os.makedirs(save_dir, exist_ok=True)

    loader = get_loader(root_dir=data_dir, split='train', batch_size=batch_size,
                        patch_size=patch_size, shuffle=True, augment=True)

    # ✅ Improved domain detection
    data_dir_lower = data_dir.lower()
    if "real" in data_dir_lower:
        domain_type = "Real"
    elif "synthetic" in data_dir_lower:
        domain_type = "Synthetic"
    else:
        domain_type = "LOL-v1 (Real)"

    print(f"📦 Training on [{domain_type}] domain")

    if fine_tune and lr == 1e-4:
        lr = 5e-5
        print(f"🛠 Fine-tuning mode enabled → using lr={lr}, shorter epochs recommended")

    model = RetinexEnhancer().to(device)
    if resume_ckpt:
        print(f"📥 Loading pretrained weights: {resume_ckpt}")
        model.load_state_dict(torch.load(resume_ckpt, map_location=device))

    lpips_loss = LPIPSLoss().to(device)
    vgg_loss = VGGLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_psnr = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for i, (low, high, names) in enumerate(pbar):
            low = low.to(device)
            high = high.to(device)

            output, R, I, I_enh, curve = model(low)

            # Compute losses
            loss_l1 = l1_loss(output, high)
            loss_lab = lab_color_loss(output, high)
            loss_lpips = lpips_loss(output, high)
            loss_vgg = vgg_loss(output, high)
            loss_msssim = msssim_loss(output, high)
            loss_edge = edge_loss(R, high)
            loss_hist = histogram_loss(output, high)
            loss_tone = tone_contrast_loss(output, high)
            loss_smooth_I = illumination_smoothness_loss(I, R)
            loss_tv_I = total_variation_loss(I_enh)
            loss_curve_reg = curve_regularization(curve)
            loss_reflect_ssim = ssim_on_reflectance(R, high)
            loss_reflect_dual = reflectance_consistency_loss(R, R.detach())

            total_loss = (
                1.0 * loss_l1 +
                0.5 * loss_lab +
                0.4 * loss_lpips +
                0.4 * loss_vgg +
                0.4 * loss_msssim +
                0.3 * loss_edge +
                0.3 * loss_hist +
                0.3 * loss_tone +
                0.2 * loss_smooth_I +
                0.2 * loss_tv_I +
                0.1 * loss_curve_reg +
                0.3 * loss_reflect_ssim +
                0.1 * loss_reflect_dual
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            pbar.set_postfix({
                "L1": loss_l1.item(),
                "LPIPS": loss_lpips.item(),
                "SSIM": loss_msssim.item(),
                "TV": loss_tv_I.item(),
                "SmoothI": loss_smooth_I.item(),
                "Total": total_loss.item()
            })

            if i == 0 and epoch % 5 == 0:
                save_intermediates(R[0], I[0], I_enh[0], f"ep{epoch+1}_{names[0]}", out_dir="visuals")

        # Save model
        torch.save(model.state_dict(), os.path.join(save_dir, f"retinex_epoch{epoch+1}.pth"))

        # ✅ Fix: Unpack 3 returned values from evaluation
        val_psnr, val_ssim, _ = evaluate_model(model, val_data_dir=os.path.join(data_dir, "test"), patch_size=patch_size)
        print(f"🔍 Epoch {epoch+1} PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"✅ Best model saved (Epoch {best_epoch}, PSNR: {best_psnr:.2f})")

    print(f"\n🏁 Training complete! Best PSNR: {best_psnr:.2f} at epoch {best_epoch}")
