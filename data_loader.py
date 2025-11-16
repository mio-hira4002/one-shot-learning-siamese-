import os
import random
from PIL import Image
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SiameseDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
        ])
        all_dirs = os.listdir(root_dir)
        self.classes = []
        for d in all_dirs:
            if d.startswith("."):
                continue
            full = os.path.join(root_dir, d)
            if not os.path.isdir(full):
                continue
            imgs = glob(os.path.join(full, "*"))
            if len(imgs) > 0:
                self.classes.append(d)
        print("[DEBUG] 有効クラス =", self.classes)
        self.image_paths = {
            cls: glob(os.path.join(root_dir, cls, "*"))
            for cls in self.classes
        }
        for cls in self.classes:
            print(f"[DEBUG] {cls} の画像枚数 =", len(self.image_paths[cls]))
        self.all_images = []
        for cls, paths in self.image_paths.items():
            for p in paths:
                self.all_images.append((p, cls))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):

        # anchor
        img1_path, cls1 = self.all_images[idx]
        img1 = self.transform(Image.open(img1_path).convert("L"))

        # 50%: same class / 50%: different class
        if random.random() < 0.5:
            img2_path = random.choice(self.image_paths[cls1])
            label = 1
        else:
            negative_cls = random.choice([c for c in self.classes if c != cls1])
            img2_path = random.choice(self.image_paths[negative_cls])
            label = 0

        img2 = self.transform(Image.open(img2_path).convert("L"))
        return img1, img2, torch.tensor(label, dtype=torch.float32)


def get_train_validation_loader(data_dir, batch_size, *args, **kwargs):
    dataset = SiameseDataset(data_dir)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_ds, valid_ds = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader


def get_test_loader(data_dir):
    dataset = SiameseDataset(data_dir)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return test_loader
