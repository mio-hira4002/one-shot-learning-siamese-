# 🐾 Siamese One-Shot Learning（犬鼻紋識別モデル）

本プロジェクトは **Siamese Neural Network（双子ネットワーク）** を応用し、  
犬の鼻紋画像を用いた「個体識別」および「特徴差分の可視化」を目的としています。   

---

## 🧠 プロジェクト概要

### 🎯 目的

- 犬の鼻紋画像から **同一個体かどうかを判定** するAIモデルを構築。  
- Siamese Network により、画像ペアの特徴量差を学習し、類似度スコアを算出。  
- 畳み込み層（Conv層）の **特徴マップ差分を可視化** して識別根拠を明示。

### 🐾 特徴

- 🐕 画像比較の自動処理（個体識別スコアの算出）  
- 🦴 畳み込み層の特徴マップを可視化  
- 🐩 未学習モデルを利用した構造確認・実験に対応  
- 🐾 Fire コマンドによる柔軟な CLI 操作（`compare` / `batch`）

---

## 📂 ディレクトリ構成

```plaintext
siamese-one-shot-pytorch/
│
├── main.py               # メインスクリプト（比較・可視化）
├── model.py              # Siamese Network 定義
├── requirements.txt      # 依存ライブラリ一覧
│
└── README.md             # 本ファイル
data/                 # 犬の鼻紋画像データ
├── dog1.jpg          # 基準画像（比較元）
├── dog2.jpg          # 比較対象
└── ...               # その他画像
