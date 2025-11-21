import os
from glob import glob
import torch
import torch.optim as optim
from tqdm import tqdm
from data_loader import get_train_validation_loader, get_test_loader
from model import SiameseNet
from config import config
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer(object):

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    #学習
    def train(self):
        train_loader, valid_loader = get_train_validation_loader(
            self.config.data_dir,
            self.config.batch_size
        )

        model = SiameseNet().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        criterion = torch.nn.BCEWithLogitsLoss()
        best_valid_acc = 0

        for epoch in range(self.config.epochs):
            # Training mode
            model.train() 
            train_loss = 0.0
            print(f"\n========== Epoch {epoch} ==========")
            for batch_idx, (x1, x2, y) in enumerate(tqdm(train_loader, colour="green")):
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                out = model(x1, x2)
                loss = criterion(out, y.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                current_lr = optimizer.param_groups[0]["lr"]
                print("[Train] epoch={}  batch={}/{}  loss={:.4f}  lr={:.6f}".format(
                    epoch, batch_idx, len(train_loader), loss.item(), current_lr
                ))
            train_loss /= len(train_loader)
            # Evaluation mode
            model.eval() 
            valid_correct = 0
            with torch.no_grad():
                for batch_idx, (x1, x2, y) in enumerate(tqdm(valid_loader)):
                    x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                    out = model(x1, x2)
                    pred = (torch.sigmoid(out) > 0.5).long()
                    valid_correct += (pred.squeeze() == y.long()).sum().item()
            valid_acc = valid_correct / len(valid_loader.dataset)
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}  Valid Acc: {valid_acc:.4f}")
            # checkpoint
            if valid_acc > best_valid_acc: 
                best_valid_acc = valid_acc
                os.makedirs(self.config.logs_dir, exist_ok=True)
                checkpoint = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_valid_acc': best_valid_acc
                }
                torch.save(checkpoint, os.path.join(self.config.logs_dir, "best_model.pth"))
                print(f" → Best model saved to {self.config.logs_dir}/best_model.pth")
            scheduler.step()

    #テスト
    def test(self):
        print("\n========== TEST ==========")
        model = SiameseNet().to(self.device)
        checkpoint = torch.load(os.path.join(self.config.logs_dir, "best_model.pth"))
        model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded model from {self.config.logs_dir}/best_model.pth (epoch {checkpoint['epoch']}, acc {checkpoint['best_valid_acc']:.4f})")
        test_loader = get_test_loader(self.config.data_dir)
        correct = 0
        with torch.no_grad():
            for batch_idx, (x1, x2, y) in enumerate(tqdm(test_loader, colour="green")):
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                out = model(x1, x2)
                score = torch.sigmoid(out)
                pred = (torch.sigmoid(out) > 0.5).long()
                correct += (pred.squeeze() == y.long()).sum().item()
                print(f"[Test] batch={batch_idx}/{len(test_loader)}  score={score.item():.4f}  "
                      f"pred={pred.item()}  label={y.item()}")
        test_acc = correct / len(test_loader.dataset)
        print(f"[Test Acc] {test_acc:.4f}")

if __name__ == "__main__":
    trainer = Trainer(config())
    trainer.train()
    trainer.test()
