import os
import torch
from torchvision import transforms
import torch.optim as optim
import numpy as np
from data_loader import train_validation_dataset
import matplotlib.pyplot as plt

from model import SiameseNet
from config import config
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.losses = []
        self.accs = []
        self.val_losses = []
        self.val_accs = []
    #学習
    def train(self):
        train_loader, valid_loader = train_validation_dataset(
            self.config.data_dir,
            self.config.batch_size,
            use_otsu=True
        )

        #前処理できているかチェック
        import matplotlib.pyplot as plt
        sample_img1, sample_img2, sample_label = next(iter(train_loader))
        img1_np = sample_img1[0].squeeze().numpy()
        img2_np = sample_img2[0].squeeze().numpy()
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img1_np, cmap="gray")
        plt.title("")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img2_np, cmap="gray")
        plt.title("")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


        model = SiameseNet().to(self.device)
        #重みの減衰を追加 weight_decay=1e-5
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=1e-5)
        #学習率を徐々に下げる
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        #シグモイド＋BCE をまとめた2値分類用損失関数
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(self.config.epochs):
            model.train() 
            running_loss = 0.0
            running_acc = 0.0
            for img1, img2, labels in train_loader:   # ← SiameseDataset が返す3つを受け取る
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad() #勾配の初期化
                output = model(img1, img2) #予測値
                loss = criterion(output.squeeze(), labels)
                loss.backward() #誤差逆伝播
                optimizer.step()
                running_loss += loss.item() #エポックごとの平均 loss を後で計算できるように、バッチごとの loss を足し込んでいる
                preds = (torch.sigmoid(output).squeeze() > 0.5).float()
                acc = (preds == labels).float().mean()
                running_acc += acc.item()

            running_loss /= len(train_loader)
            running_acc /= len(train_loader)
            self.losses.append(running_loss)
            self.accs.append(running_acc)

            #validation loop
            model.eval()
            val_running_loss = 0.0
            val_running_acc = 0.0
            with torch.no_grad():
                for img1, img2, labels in valid_loader:
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                    labels = labels.to(self.device)
                    val_output = model(img1, img2)
                    val_loss = criterion(val_output.squeeze(), labels)
                    val_running_loss += val_loss.item()
                    preds = (torch.sigmoid(val_output).squeeze() > 0.5).float()
                    acc = (preds == labels).float().mean()
                    val_running_acc += acc.item()

            val_running_loss /= len(valid_loader)
            val_running_acc /= len(valid_loader)

            self.val_losses.append(val_running_loss)
            self.val_accs.append(val_running_acc)

            print(f"Epoch [{epoch+1}/{self.config.epochs}] "
                f"Train Loss: {running_loss:.4f}  Train Acc: {running_acc:.4f} | "
                f"Val Loss: {val_running_loss:.4f}  Val Acc: {val_running_acc:.4f}")

            scheduler.step()

        plt.style.use("ggplot")
        plt.plot(self.losses, label="train loss")
        plt.plot(self.val_losses, label="validation loss")
        plt.legend()
        plt.show()

    #テスト
    def test(self):
        print("\n========== TEST ==========")
        
        # モデル準備
        model = SiameseNet().to(self.device)
        checkpoint_path = os.path.join(self.config.logs_dir, "best_model.pth")

        if not os.path.exists(checkpoint_path):
            print("No checkpoint found! Run training first.")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded model from {checkpoint_path} (epoch {checkpoint['epoch']}, acc {checkpoint['best_valid_acc']:.4f})")

        # テストデータ読み込み
        test_loader = test_dataset(self.config.data_dir)

        correct = 0
        total = 0

        model.eval()
        with torch.no_grad():
            for x1, x2, y in test_loader:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)

                out = model(x1, x2)
                pred = (torch.sigmoid(out) > 0.5).long()

                correct += (pred.squeeze() == y.long()).sum().item()
                total += y.size(0)

        test_acc = correct / total
        print(f"[Test Accuracy] {test_acc:.4f}")
                # ----- 可視化用リスト -----
        scores = []
        labels_list = []

        model.eval()
        scores = []
        labels_list = []

        #testグラフの生成に必要なコード
        with torch.no_grad():
            for x1, x2, y in test_loader:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)

                out = model(x1, x2)
                prob = torch.sigmoid(out).squeeze()

                scores.append(prob.cpu().item())
                labels_list.append(y.cpu().item())


        # --------- グラフ描画 ---------
        plt.style.use("ggplot")
        plt.figure(figsize=(8, 4))

        plt.hist([s for s, t in zip(scores, labels_list) if t == 1],
                 bins=30, alpha=0.6, label="Positive (Same Class)")
        plt.hist([s for s, t in zip(scores, labels_list) if t == 0],
                 bins=30, alpha=0.6, label="Negative (Different Class)")

        plt.title("Test Score Distribution (Sigmoid Output)")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()


        

if __name__ == "__main__":
    trainer = Trainer(config())
    trainer.train()
    trainer.test()
    #if __name__ == "__main__":について整理すると pythonのファイルは全て自動で 用意されている"__name__" という変数がある。
    # 実際に"python 3 trainer.py"で直接そのファイルを実行した場合のみ、指定されたtrainer.py内の__name__変数が__main__として設定される。