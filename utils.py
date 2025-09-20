import os
from torchvision.utils import save_image

def save_intermediates(R, I, I_enh, name, out_dir="visuals"):
    os.makedirs(out_dir, exist_ok=True)
    save_image(R, os.path.join(out_dir, f"{name}_R.jpg"))
    save_image(I.expand_as(R), os.path.join(out_dir, f"{name}_I.jpg"))
    save_image(I_enh.expand_as(R), os.path.join(out_dir, f"{name}_Ienh.jpg"))
