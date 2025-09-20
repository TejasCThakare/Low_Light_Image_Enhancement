import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class LOLPlusDataset(Dataset):
    def __init__(self, root_dir, split='train', patch_size=None, augment=False):
        self.low_dir = os.path.join(root_dir, split, 'low')
        self.high_dir = os.path.join(root_dir, split, 'high')

        self.image_names = [
            name for name in sorted(os.listdir(self.low_dir))
            if os.path.exists(os.path.join(self.high_dir, name))
        ]

        self.patch_size = patch_size
        self.augment = augment
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        low_img = Image.open(os.path.join(self.low_dir, name)).convert('RGB')
        high_img = Image.open(os.path.join(self.high_dir, name)).convert('RGB')

        if self.patch_size:
            low_img, high_img = self.random_crop(low_img, high_img)

        if self.augment:
            if random.random() > 0.5:
                low_img = T.functional.hflip(low_img)
                high_img = T.functional.hflip(high_img)
            if random.random() > 0.5:
                low_img = T.functional.vflip(low_img)
                high_img = T.functional.vflip(high_img)

        low_tensor = self.to_tensor(low_img)
        high_tensor = self.to_tensor(high_img)

        return low_tensor, high_tensor, name

    def random_crop(self, low, high):
        w, h = low.size
        ps = self.patch_size
        left = random.randint(0, w - ps)
        top = random.randint(0, h - ps)
        box = (left, top, left + ps, top + ps)
        return low.crop(box), high.crop(box)

def get_loader(root_dir, batch_size=4, split='train', patch_size=192, shuffle=True, augment=True):
    dataset = LOLPlusDataset(root_dir, split, patch_size, augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
