# 🐾 Siamese One-Shot Learning（画像識別モデル）

手書き動物の画像識別モデルの構築を行いました。 
Siamese Networkっていう手法を使って、2枚の画像が同じ個体かどうか判定します。

---

## 📍特徴

-  犬やキリンの画像を2枚比較して、同じ個体か判定
-  類似度スコア（0〜1）を計算
-  特徴マップの可視化で、どこを見て判断してるかチェック

---

##  📁ファイル構成

```
siamese network/
├── main.py          # 推論用（画像比較・可視化）
├── trainer.py       # 学習用（モデルを訓練）
├── model.py         # Siamese Networkの定義
├── data_loader.py   # データ読み込み
├── config.py        # 設定ファイル
└── requirements.txt # 必要なライブラリ

data2/               # 学習データ（dog 30枚、giraffe 30枚...）
data3/               # 推論テスト用（dog 3枚、giraffe 3枚...）
```

---

## ⚙️ セットアップ

### 1. 必要なライブラリをインストール

```bash
cd "siamese network"
pip install -r requirements.txt
```

### 2. 学習データの準備

- 似た種類の画像３０枚、６クラス分用意する
- それぞれ以下のようなフォルダ内にクラスごとのファイルを作成し、画像を保管してください
`data2/`に以下の構造でデータを配置：

```
data2/
├── dog/
│   ├── dog1.jpg
│   ├── dog2.jpg
│   └── ...
└── giraffe/
    ├── giraffe 1.jpeg
    ├── giraffe 2.jpeg
    └── ...
```
data3/
├── dog/
│   ├── dog1.jpg
│   ├── dog2.jpg
│   └── ...
└── giraffe/
    ├── giraffe 1.jpeg
    ├── giraffe 2.jpeg
    └── ...


---

## 実行の手順

### 学習する 🐶

```bash
python trainer.py
```

- `data2/`のデータで学習
- 学習済みモデルは`logs/best_model.pth`に保存される
- エポック数やバッチサイズは`config.py`で変更可能

### 推論する（2枚の画像を比較）🐈

```bash
python main.py compare --img1_path=../data3/dog 1.jpg --img2_path=../data3/dog 2.jpg
```

### フォルダ内の画像を一括比較　🐕

```bash
python main.py batch --base_img_path=../data3/dog 1.jpg --folder_path=../data3
```

### data3で類似度マトリックスを可視化 📊

```bash
python main.py infer_data3
```

- `data3/`内の全画像について、ペアごとの類似度を計算
- ヒートマップで可視化して保存

---

## 📊 出力例

### 類似度スコア

```
類似度スコア（0〜1）: 0.8234
→ 同一個体の可能性あり
```

### 類似度マトリックス

`data_output/similarity_matrix_data3.png`に保存されます。

---

## 🔧 設定変更

`config.py`で以下を変更できます：

```python
data_dir = "../data2"     # 学習データのパス
batch_size = 16           # バッチサイズ
epochs = 10               # エポック数
lr = 0.0005               # 学習率
```
---


## 💡 Tips

- 🐕 データが少なくても学習できる（One-Shot Learning）
- 📸 画像は自動でリサイズ＆グレースケール化される
- 🎯 閾値（0.5）より大きければ「同じ」と判定

---

🐕 Happy Coding! 🐾
