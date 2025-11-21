from fire import Fire
from model import SiameseNet
from PIL import Image
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import seaborn as sns
from config import config

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
    cfg = config()
    model_path = os.path.join(cfg.logs_dir, "best_model.pth")

    # 学習済みモデルの読み込み
    try:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"✓ Loaded trained model from {model_path}")
        print(f"  Epoch: {ckpt['epoch']}, Valid Acc: {ckpt['best_valid_acc']:.4f}")
    except Exception as e:
        print(f"[!] Warning: could not load model ({e}). Using untrained weights.")
    
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
    cfg = config()
    model_path = os.path.join(cfg.logs_dir, "best_model.pth")

    # 学習済みモデルの読み込み
    try:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"best_modelを読み込んでいます...{model_path}")
        print(f"エポック: {ckpt['epoch']}, 正解率: {ckpt['best_valid_acc']:.4f}")
    except Exception as e:
        print(f"[!] Warning: could not load model ({e}). Using untrained weights.")
    
    model.eval()

    try:
        base_img = preprocess(base_img_path, device)
    except FileNotFoundError as e:
        print(f"[!] Error: base image not found: {base_img_path}")
        return
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

        try:
            img = preprocess(compare_path, device)
        except FileNotFoundError:
            print(f"[!] Warning: compare image not found (skipping): {compare_path}")
            continue

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
            print(f"[DEBUG] L1距離の総和: {l1_distance:.4f}")
            vals = diff.view(-1).cpu().numpy()
            top5_idx = np.argsort(-vals)[:5]
            top5 = [(int(i), float(vals[i])) for i in top5_idx]
            print(f"[DEBUG] top5 diff indices and values: {top5}")
            print(f"[DEBUG] diff mean/std: {vals.mean():.4f}/{vals.std():.4f}")
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


# data3を使った推論
def infer_data3():
    """
    data3フォルダの画像を使用して、全ペアの類似度を計算し、
    類似度マトリックスとして可視化する
    """
    print_status("Start ：Data3 推論と可視化")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNet().to(device)
    cfg = config()
    model_path = os.path.join(cfg.logs_dir, "best_model.pth")
    
    # 学習済みモデルの読み込み
    try:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"best_modelを読み込んでいます...{model_path}")
        print(f" エポック: {ckpt['epoch']}, 正解率: {ckpt['best_valid_acc']:.4f}\n")
    except Exception as e:
        print(f"[!] Error: could not load model ({e}). Please train the model first.")
        return
    
    model.eval()
    
    # data3フォルダのパス
    data3_dir = os.path.join(cfg.BASE_DIR, "..", "data3")
    image_files = sorted([f for f in os.listdir(data3_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if len(image_files) == 0:
        print(f"[!] No images found in {data3_dir}")
        return
    
    print(f"Found {len(image_files)} images in data3:")
    for img_file in image_files:
        print(f"  - {img_file}")
    print()
    
    # 画像を前処理してリストに格納
    images = []
    for img_file in image_files:
        img_path = os.path.join(data3_dir, img_file)
        img = preprocess(img_path, device)
        images.append(img)
    
    # 類似度マトリックスを計算
    n = len(images)
    similarity_matrix = np.zeros((n, n))
    
    print("Computing similarity matrix...")
    with torch.no_grad():
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # 自分自身は完全一致
                else:
                    out = model(images[i], images[j])
                    score = torch.sigmoid(out).item()
                    similarity_matrix[i, j] = score
                    print(f"  {image_files[i]} vs {image_files[j]}: {score:.4f}")
    
    print("\n" + "="*60)
    print("類似度行列")
    print("="*60)
    print(f"{'':20}", end="")
    for name in image_files:
        print(f"{name[:15]:>15}", end="")
    print()
    
    for i, name_i in enumerate(image_files):
        print(f"{name_i[:20]:20}", end="")
        for j in range(n):
            print(f"{similarity_matrix[i, j]:15.4f}", end="")
        print()
    print("="*60 + "\n")
    
    # ヒートマップで可視化
    plt.figure(figsize=(12, 10))
    
    # 類似度マトリックスのヒートマップ
    ax = sns.heatmap(similarity_matrix, 
                     annot=True, 
                     fmt='.3f',
                     cmap='Greens',
                     xticklabels=[f.replace('のコピー', '') for f in image_files],
                     yticklabels=[f.replace('のコピー', '') for f in image_files],
                     cbar_kws={'label': 'Similarity Score'},
                     vmin=0, vmax=1)
    
    plt.title('similarity matrix - Data3 \n(Trained Siamese Network)', 
              fontsize=16, pad=20)
    plt.xlabel('Compare Image', fontsize=12)
    plt.ylabel('Base Image', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 結果を保存
    output_dir = os.path.join(cfg.BASE_DIR, "..", "data_output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "similarity_matrix_data3.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    
    plt.show()
    
    # 推論結果のサマリー
    print("\n" + "="*60)
    print("推論結果")
    print("="*60)
    
    threshold = 0.5
    print(f"判定閾値: {threshold}")
    print(f"\n同一個体と判定されたペア:")
    same_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i, j] > threshold:
                same_pairs.append((image_files[i], image_files[j], similarity_matrix[i, j]))
                print(f"  ✓ {image_files[i]} ⟷ {image_files[j]}: {similarity_matrix[i, j]:.4f}")
    
    if not same_pairs:
        print("  (なし)")
    
    print(f"\n別個体と判定されたペア:")
    diff_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i, j] <= threshold:
                diff_pairs.append((image_files[i], image_files[j], similarity_matrix[i, j]))
                print(f"  ✗ {image_files[i]} ⟷ {image_files[j]}: {similarity_matrix[i, j]:.4f}")
    
    if not diff_pairs:
        print("  (なし)")
    
    print("="*60)

if __name__ == '__main__':
    Fire({
        "compare": compare_images,
        "batch": compare_folder,
        "infer": infer_data3
    })