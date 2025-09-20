import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from retinex_model import RetinexEnhancer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def enhance_image(input_path, model_path, output_path="results/enhanced.jpg",
                  patch_size=None, save_intermediates=False, apply_gamma=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = Image.open(input_path).convert("RGB")
    if patch_size:
        img = img.resize((patch_size, patch_size))

    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    model = RetinexEnhancer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        pred, R, I, I_enh, _ = model(tensor)
        pred = pred[0].clamp(0, 1)

        if apply_gamma:
            pred = pred.pow(1 / 2.2)

    save_image(pred, output_path)
    print(f"✅ Saved enhanced image: {output_path}")

    if save_intermediates:
        base = os.path.splitext(os.path.basename(output_path))[0]
        base_dir = os.path.dirname(output_path)
        save_image(R[0], os.path.join(base_dir, f"{base}_R.jpg"))
        save_image(I[0].expand(3, -1, -1), os.path.join(base_dir, f"{base}_I.jpg"))
        save_image(I_enh[0].expand(3, -1, -1), os.path.join(base_dir, f"{base}_Ienh.jpg"))
        print("🧪 Saved intermediate outputs: R, I, I_enh")
