# DViN — Thesis Fork
[![Python](https://img.shields.io/badge/python-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-%237732a8)

本 repo 為論文研究用途，基於 [DViN](https://github.com/XxFChen/DViN)（Dynamic Visual Routing Network for Weakly Supervised Referring Expression Comprehension）進行架構延伸與改進。

修改紀錄與研究方向詳見 [RESEARCH_LOG.md](RESEARCH_LOG.md)。

![DViN](assets/fig2.jpg)

---

## Project Structure

```txt
├── README.md                    <- 專案說明文件（本文件）
├── RESEARCH_LOG.md              <- 修改紀錄與研究方向筆記
│
├── config/                      <- 各資料集的訓練設定檔
│   ├── refcoco.yaml
│   ├── refcoco+.yaml
│   ├── refcocog.yaml
│   └── referit.yaml
│
├── data/
│   ├── anns/                    <- 標注 JSON（cat_name.json 為 prompt template 用途）
│   └── images/                  <- COCO train2014 與 ReferIT 圖片
│
├── datasets/
│   └── dataloader.py            <- 資料集載入，支援 RefCOCO / RefCOCO+ / RefCOCOg / ReferIT
│
├── models/
│   ├── language_encoder.py      <- 文字描述編碼器（LSTM + GloVe）
│   ├── network_blocks.py        <- 共用網路模組（DCN、SPP、Attention 等）
│   ├── clip_encoder.py          <- CLIP 特徵提取
│   ├── sam_encoder.py           <- EfficientSAM 特徵提取
│   ├── visual_encoder.py        <- YOLOv3 視覺骨幹，含 prompt template encoder
│   ├── Experts_model.md         <- Expert 模型下載說明
│   │
│   └── DViR/                    <- 核心模型實作
│       ├── __init__.py
│       ├── head.py              <- Anchor-Prompt Contrastive Loss
│       ├── net.py               <- 原始 DViN 模型（Hard Top-2 routing）
│       └── net_v2.py            <- 改進版本（CrossAttentionRouter + Soft All-4 fusion）
│
├── utils/
│   ├── DCN/                     <- Deformable Convolution Networks（需編譯）
│   ├── config.py                <- 設定檔管理
│   ├── distributed.py           <- 分散式訓練工具
│   ├── ckpt.py                  <- Checkpoint 存取
│   ├── logging.py               <- 訓練 log 工具
│   └── utils.py                 <- 通用工具（EMA、IoU、box 轉換等）
│
├── EfficientSAM/                <- EfficientSAM 子模組
├── apex/                        <- NVIDIA Apex（混合精度訓練）
│
├── train.py                     <- 訓練腳本
├── test.py                      <- 評估腳本
├── demo.py                      <- 單張圖片推論示範
└── requirements.txt             <- Python 依賴套件
```

---

## Installation

### 建立環境

```bash
conda create -n DViN python=3.9 -y
conda activate DViN
```

### 安裝 PyTorch（建議版本）

```bash
# PyTorch 1.11.0 with CUDA 11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

### 安裝 Apex（混合精度訓練）

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
cd ..
```

### 編譯 DCN layer

```bash
cd utils/DCN
./make.sh
cd ../..
```

### 安裝其餘依賴

```bash
pip install -r requirements.txt
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
```

### 下載預訓練權重

Expert 模型（CLIP、DINOv2、EfficientSAM、ConvNeXt）請參考 [models/Experts_model.md](models/Experts_model.md)。

YOLOv3 權重：[Google Drive](https://drive.google.com/file/d/1nxVTx8Zv52VSO-ccHVFe2ggG0HbGnw9g/view?usp=sharing)（建議放在專案根目錄）

---

## Data Preparation

依照 [SimREC 資料準備說明](https://github.com/luogen1996/SimREC/blob/main/DATA_PRE_README.md) 下載圖片與標注，或直接使用 `data/anns/` 中已備妥的標注檔案。

```txt
data/
├── anns/
│   ├── refcoco.json
│   ├── refcoco+.json
│   ├── refcocog.json
│   ├── refclef.json
│   └── cat_name.json
└── images/
    ├── train2014/
    └── refclef/
```

> **注意**：YOLOv3 使用 COCO train2014 訓練，已排除 RefCOCO / RefCOCO+ / RefCOCOg 的 val/test 圖片。

---

## Training

```bash
python train.py --config ./config/[DATASET_NAME].yaml
```

使用 `net_v2.py`（改進版 router）訓練時，請在 config 中指定對應的模型設定。

---

## Evaluation

```bash
python test.py --config ./config/[DATASET_NAME].yaml --eval-weights [PATH_TO_CHECKPOINT]
```

---

## Model Zoo

### Weakly Supervised REC

| Method | RefCOCO val | testA | testB | RefCOCO+ val | testA | testB | RefCOCOg val-g |
|--------|-------------|-------|-------|--------------|-------|-------|----------------|
| DViN   | 67.67       | 70.90 | 59.39 | 52.54        | 57.52 | 45.31 | 55.04          |

### Weakly Supervised RES

| Method | RefCOCO val | testA | testB | RefCOCO+ val | testA | testB | RefCOCOg val-g |
|--------|-------------|-------|-------|--------------|-------|-------|----------------|
| DViN   | 61.43       | 63.81 | 56.97 | 46.79        | 51.87 | 39.85 | 46.49          |

### Pseudo Labels（弱監督訓練其他模型）

| Method       | RefCOCO val | testA | testB | RefCOCO+ val | testA | testB | RefCOCOg val-g |
|--------------|-------------|-------|-------|--------------|-------|-------|----------------|
| DViN_SimREC  | 67.29       | 73.09 | 60.65 | 51.54        | 59.06 | 39.59 | 51.73          |
| DViN_TransVG | 64.99       | 68.87 | 64.48 | 50.72        | 57.36 | 38.64 | 50.47          |

---

## Visualization

| Description | Result |
|-------------|--------|
| "Kid on right in back blondish hair" | ![vs0](assets/vs_0.jpg) |
| "Top broccoli" | ![vs1](assets/vs_1.jpg) |
| "Yellow and blue vehicle close to the camera" | ![vs2](assets/vs_2.jpg) |
| "Second from the right" | ![vs4](assets/vs_4.jpg) |

藍色框為 ground truth。
