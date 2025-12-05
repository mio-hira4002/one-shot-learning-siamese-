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

        accs = np.array(self.accs)
        val_accs = np.array(self.val_accs)

        #平均値 ± 標準偏差
        # print("Accuracy (mean ± std):")
        # print(f"Train : {accs.mean():.4f} ± {accs.std():.4f}")
        # print(f"Valid : {val_accs.mean():.4f} ± {val_accs.std():.4f}\n")

        #改善度合い（差分の平均と標準偏差）
        # diff = val_accs - accs
        # print("Improvement (Valid - Train):")
        # print(f"Mean : {diff.mean():.4f}")
        # print(f"Std  : {diff.std():.4f}")


            # for batch_idx, (x1, x2, y) in enumerate(tqdm(train_loader, colour="green")):
            #     x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
            #     out = model(x1, x2)
            #     loss = criterion(out, y.unsqueeze(1))
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            #     train_loss += loss.item()
            #     current_lr = optimizer.param_groups[0]["lr"]
            #     print("[Train] epoch={}  batch={}/{}  loss={:.4f}  lr={:.6f}".format(
            #         epoch, batch_idx, len(train_loader), loss.item(), current_lr)
            #     )

                
            # train_loss /= len(train_loader)
            # # Evaluation mode
            # model.eval() 
            # valid_correct = 0
            # with torch.no_grad():
            #     for batch_idx, (x1, x2, y) in enumerate(tqdm(valid_loader)):
            #         x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
            #         out = model(x1, x2)
            #         pred = (torch.sigmoid(out) > 0.5).long()
            #         valid_correct += (pred.squeeze() == y.long()).sum().item()
            # valid_acc = valid_correct / len(valid_loader.dataset)
            # print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}  Valid Acc: {valid_acc:.4f}")
            # # checkpoint
            # if valid_acc > best_valid_acc: 
            #     best_valid_acc = valid_acc
            #     os.makedirs(self.config.logs_dir, exist_ok=True)
            #     checkpoint = {
            #         'epoch': epoch,
            #         'model_state': model.state_dict(),
            #         'optimizer_state': optimizer.state_dict(),
            #         'best_valid_acc': best_valid_acc
            #     }
            #     torch.save(checkpoint, os.path.join(self.config.logs_dir, "best_model.pth"))
            #     print(f" → Best model saved to {self.config.logs_dir}/best_model.pth")
            # scheduler.step()

    #テスト
    # def test(self):
    #     print("\n========== TEST ==========")
    #     model = SiameseNet().to(self.device)
    #     checkpoint = torch.load(os.path.join(self.config.logs_dir, "best_model.pth"))
    #     model.load_state_dict(checkpoint['model_state'])
    #     print(f"Loaded model from {self.config.logs_dir}/best_model.pth (epoch {checkpoint['epoch']}, acc {checkpoint['best_valid_acc']:.4f})")
    #     test_loader = get_test_loader(self.config.data_dir)
    #     correct = 0
    #     with torch.no_grad():
    #         for batch_idx, (x1, x2, y) in enumerate(tqdm(test_loader, colour="green")):
    #             x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
    #             out = model(x1, x2)
    #             score = torch.sigmoid(out)
    #             pred = (torch.sigmoid(out) > 0.5).long()
    #             correct += (pred.squeeze() == y.long()).sum().item()
    #             print(f"[Test] batch={batch_idx}/{len(test_loader)}  score={score.item():.4f}  "
    #                   f"pred={pred.item()}  label={y.item()}")
    #     test_acc = correct / len(test_loader.dataset)
    #     print(f"[Test Acc] {test_acc:.4f}")

if __name__ == "__main__":
    trainer = Trainer(config())
    trainer.train()
    # trainer.test()
