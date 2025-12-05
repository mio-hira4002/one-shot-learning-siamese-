import os
import random
from PIL import Image
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

def get_image_files(folder):
    files = glob(os.path.join(folder, "*"))
    return [f for f in files if f.lower().endswith(VALID_EXT)]

class SiameseDataset(Dataset):
    def __init__(self, root_dir, mode="train", transform=None, train_transform=None, use_otsu=True):
        self.use_otsu = use_otsu
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize((105, 105)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])

        #validation/test時の標準transform
        # if transform is None:
        #     self.transform = transforms.Compose([
        #         transforms.Resize((105, 105)),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5,), (0.5,)) #平均値、標準偏差はともに0.5で正規化
        #     ])
        # else:
        #     self.transform = transform

        # #train時のデータ拡張
        # if train_transform is None:
        #     self.train_transform = transforms.Compose([
        #         transforms.Resize((105, 105)),
        #         transforms.RandomHorizontalFlip(),#ランダムに反転
        #         transforms.ColorJitter(),#明るさ
        #         transforms.RandomRotation(10), #回転
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5,), (0.5,))
        #     ])
        # else:
        #     self.train_transform = train_transform

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
            cls: get_image_files(os.path.join(root_dir, cls))
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
    

    def preprocess(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"画像が読み込めません: {path}")
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# グレースケール
        _, th_otsu = cv2.threshold(img_g, 0, 255, cv2.THRESH_OTSU)# Otsu 二値化
        # ヒストグラム平坦化
        eq = cv2.equalizeHist(img_g)
        pil_img = Image.fromarray(th_otsu)#PILへ変換
        return pil_img

    def __getitem__(self, idx):

        # anchor
        img1_path, cls1 = self.all_images[idx]
        img1_pil = self.preprocess(img1_path)
        img1 = self.transform(img1_pil)
        if self.use_otsu:
            img1_pil = self.preprocess(img1_path) 
            img1 = self.transform(img1_pil)
        else:
            img1 = self.transform(Image.open(img1_path).convert("L"))

        # 50%: same class / 50%: different class
        if random.random() < 0.5:
            img2_path = random.choice(self.image_paths[cls1])
            label = 1
        else:
            negative_cls = random.choice([c for c in self.classes if c != cls1])
            img2_path = random.choice(self.image_paths[negative_cls])
            label = 0

        img2_pil = self.preprocess(img2_path)
        img2 = self.transform(img2_pil)
        if self.use_otsu:
            img2_pil = self.preprocess(img2_path)
            img2 = self.transform(img2_pil)
        else:
            img2 = self.transform(Image.open(img2_path).convert("L"))
        return img1, img2, torch.tensor(label, dtype=torch.float32)


        # img1_path, cls1 = self.all_images[idx]
        # img1_raw = Image.open(img1_path).convert("L")

        # if self.mode == "train":
        #     # print(f"[DEBUG] train_transform applied to anchor: {os.path.basename(img1_path)}")
        #     img1 = self.train_transform(img1_raw)
        # else:
        #     img1 = self.transform(img1_raw)

        # if random.random() < 0.5:# same class
        #     img2_path = random.choice(self.image_paths[cls1])
        #     label = 1
        # else:# different class
        #     negative_cls = random.choice([c for c in self.classes if c != cls1])
        #     img2_path = random.choice(self.image_paths[negative_cls])
        #     label = 0
            
        # img2_raw = Image.open(img2_path).convert("L")
        # # train 時は augmentation、それ以外は標準 transform
        # if self.mode == "train":
        #     # print(f"[DEBUG] train_transform applied to pair: {os.path.basename(img2_path)}")
        #     img2 = self.train_transform(img2_raw)
        # else:
        #     img2 = self.transform(img2_raw)
        # return img1, img2, torch.tensor(label, dtype=torch.float32)


def train_validation_dataset(data_dir, batch_size, use_otsu=False, *args, **kwargs):
    dataset = SiameseDataset(data_dir, use_otsu=use_otsu)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_ds, valid_ds = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    return train_dataloader, valid_dataloader


def test_dataset(data_dir):
    dataset = SiameseDataset(data_dir)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return test_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data_dir = "../data2" 
    dataset = SiameseDataset(data_dir)
    num_samples = 6  
    fig, axes = plt.subplots(num_samples, 2, figsize=(6, num_samples * 3))

    for i in range(num_samples):
        img1, img2, label = dataset[i]
        img1_np = img1.squeeze().numpy()
        img2_np = img2.squeeze().numpy()
        axes[i, 0].imshow(img1_np, cmap="gray")
        axes[i, 0].set_title(f"Anchor {i}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(img2_np, cmap="gray")
        axes[i, 1].set_title(f"Pair {i} (label={label.item():.0f})")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()
