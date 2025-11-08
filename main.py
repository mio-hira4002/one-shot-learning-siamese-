from fire import Fire
from model import SiameseNet
from PIL import Image
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# ユーティリティ関数
def print_status(string):
    line = '*' * 20
    print(line + " " + string + " " + line)


# 前処理関数
def preprocess(path, device):
    img = Image.open(path).convert("L")
    img = ImageOps.exif_transpose(img)  
    img = ImageOps.pad(img, (105, 105), color=0, centering=(0.5, 0.5))  
    img = np.array(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    return img


# 特徴マップの差分を可視化
def visualize_difference_feature_maps(model, base_img, compare_img, show=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    first_conv = model.conv[0]
    with torch.no_grad():
        fmap_base = first_conv(base_img.to(device)).squeeze(0).cpu()
        fmap_compare = first_conv(compare_img.to(device)).squeeze(0).cpu()

    diff_maps = torch.abs(fmap_base - fmap_compare)
    diff_maps = (diff_maps - diff_maps.min()) / (diff_maps.max() - diff_maps.min())
    diff_maps = diff_maps.numpy()

    cols, rows = 8, diff_maps.shape[0] // 8
    plt.figure(figsize=(16, 12))
    for i in range(diff_maps.shape[0]):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(diff_maps[i], cmap='inferno')
        plt.axis('off')

    plt.suptitle("Difference Feature Maps (|Conv1(base) - Conv1(compare)|)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()


#単一画像比較
def compare_images(img1_path, img2_path):
    print_status("Start One-Shot Comparison")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNet().to(device)

    # モデルの重み(学習済みパラメータはコメントアウト部分)
    # try:
    #     ckpt = torch.load(model_path, map_location=device)
    #     model.load_state_dict(ckpt['model_state'])
    #     print(f"Loaded model from {model_path}")
    # except Exception as e:
    #     print(f"[!] Warning: could not load model ({e}). Using untrained weights.")
    print("[*] Using randomly initialized model weights.")
    model.eval()

    base_img = preprocess(img1_path, device)
    compare_img = preprocess(img2_path, device)
    with torch.no_grad():
        out = model(base_img, compare_img)
        score = torch.sigmoid(out).item()

        print(f"類似度スコア（0〜1）: {score:.4f}")
        print("→ 同一個体の可能性あり" if score > 0.5 else "→ 同一個体の可能性なし")



#フォルダ内の画像を比較
def compare_folder(base_img_path, folder_path):
    print_status("Start Folder Comparison")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNet().to(device)
    print("[*] Using randomly initialized model weights.")
    model.eval()

    base_img = preprocess(base_img_path, device)
    print("\n[DEBUG] --- 畳み込み層ごとの出力 ---")
    x = base_img.clone()
    for i, layer in enumerate(model.conv):
        x = layer(x)
        print(f"[Layer {i:02d}] {layer.__class__.__name__} → 出力shape: {x.shape}")
    print(f"Flatten後: {x.view(x.size(0), -1).shape}")
    print("[DEBUG] -------------------------------\n")

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        compare_path = os.path.join(folder_path, filename)
        if os.path.abspath(compare_path) == os.path.abspath(base_img_path):
            continue  

        img = preprocess(compare_path, device)

        print(f"[DEBUG] {os.path.basename(compare_path)} のテンソル shape = {img.shape}")
        B, C, H, W = img.shape
        print(f"┣ バッチサイズ: {B}")
        print(f"┣ チャネル数  : {C}")
        print(f"┣ 高さ（H）  : {H}")
        print(f"┗ 幅（W）    : {W}")
        
        with torch.no_grad():
            emb_base = model.conv(base_img).view(1, -1)
            emb_img = model.conv(img).view(1, -1)

            diff = torch.abs(emb_base - emb_img)
            l1_distance = torch.sum(diff).item()
            out = model(base_img, img)
            score = torch.sigmoid(out).item()

            print("---------------------------------------------------")
            print(f"[DEBUG] emb_base.shape: {emb_base.shape}")
            print(f"[DEBUG] emb_img.shape : {emb_img.shape}")
            print(f"[DEBUG] |f(x1)-f(x2)| の最初の5成分: {diff[0, :6].cpu().numpy()}")
            print(f"[DEBUG] L1距離の総和: {l1_distance:.4f}")
            print(f"[DEBUG] モデルの raw 出力 (logit): {out.item():.4f}")
            print(f"[DEBUG] シグモイド後の類似度スコア: {score:.4f}")
            print(f"[DEBUG] raw out.shape = {out.shape}")
            print(f"[DEBUG] raw out tensor = {out.cpu().numpy()}")
            result = "同一個体の可能性が高い" if score > 0.5 else "別個体の可能性が高い"
            print(f"{filename} → 類似度スコア: {score:.4f} → {result}")


            #特徴可視化マップ
            # print("\n[DEBUG] 特徴マップを可視化します...")
            # show_feature_maps(fmap_img, "Compared Image - Conv1 Feature Maps") #比較先

            #差分特徴マップの可視化
            # print("\n[DEBUG] 差分特徴マップを可視化します...")
            # visualize_difference_feature_maps(model, base_img, img)

if __name__ == '__main__':
    Fire({
        "compare": compare_images,
        "batch": compare_folder
    })