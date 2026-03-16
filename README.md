# DViN
[![Python](https://img.shields.io/badge/python-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-%237732a8)


This repo is the official implementation of the paper "DViN: Dynamic Visual Routing Network for Weakly Supervised Referring Expression Comprehension"
![DViN](assets/fig2.jpg)

## Project structure

The directory structure of the project looks like this:

```txt
├── README.md            <- The top-level README for developers using this project.
│
├── config               <- configuration 
│
├── data
│   ├── anns            <- note: cat_name.json is for prompt template usage
│
├── datasets               <- dataloader file
│
│
├── models  <- Source code for use in this project.
│   │
│   ├── language_encoder.py             <- encoder for images' text descriptions 
│   ├── network_blocks.py               <- files included essential model blocks 
│   ├── clip_encoder.py                  <- encoder for extracting CLIP model embeddings 
│   ├── sam_encoder.py                  <- encoder for extracting SAM model embeddings 
│   ├── visual_encoder.py               <- visual backbone ,also includes prompt template encoder
│   │
│   │
│   ├── DViN           <- most important files for DViN model implementations
│   │   ├── __init__.py
│   │   ├── head.py   <- for anchor-prompt contrastive loss
|   |   ├── net.py    <- main code for DViN model
│   │
│   │
├── utils  <- hepler functions
├── requirements.txt     <- The requirements file for reproducing the analysis environment
│── train.py   <- script for training the model
│── test.py <- script for testing from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

## Installation 
Instructions on how to clone and set up your repository:

### Clone this repo :

- Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/XxFChen/DViN.git
cd DViN
```

### Create a conda virtual environment and activate it:
```bash
conda create -n DViN python=3.9 -y
conda activate DViN
```
### Install the required dependencies:
- Install Pytorch following the [offical installation instructions](https://pytorch.org/get-started/locally/) 

(We run all our experiments on pytorch 1.11.0 with CUDA 11.3)

- Install apex following the [official installation guide](https://github.com/NVIDIA/apex#quick-start) for more details.

(or use the following commands we copied from their offical repo)
```bash
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
#### Compile the DCN layer:
```bash
cd utils/DCN
./make.sh
```
#### Install remaining dependencies
```bash
pip install -r requirements.txt
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
```
### Data Preparation
- Download images and Generate annotations according to [SimREC](https://github.com/luogen1996/SimREC/blob/main/DATA_PRE_README.md) 

(We also prepared the annotations inside the data/anns folder for saving your time)

- Download the pretrained weights of YoloV3 from [Google Drive](https://drive.google.com/file/d/1nxVTx8Zv52VSO-ccHVFe2ggG0HbGnw9g/view?usp=sharing) 

(We recommend to put it in the main path of DViN otherwise, please modify the path in config files)

- The data directory should look like this:

```txt
├── data
│   ├── anns            
│       ├── refcoco.json            
│       ├── refcoco+.json              
│       ├── refcocog.json                 
│       ├── refclef.json
│       ├── cat_name.json       
│   ├── images 
│       ├── train2014
│           ├── COCO_train2014_000000515716.jpg              
│           ├── ...
│       ├── refclef
│           ├── 99.jpg              
│           ├── ...

... the remaining directories    
```
- NOTE: our YoloV3 is trained on COCO’s training images, excluding those in RefCOCO, RefCOCO+, and RefCOCOg’s validation+testing

## Training 

```bash
python train.py --config ./configs/[DATASET_NAME].yaml
```
## Evaluation

```bash 
python test.py --config ./config/[DATASET_NAME].yaml --eval-weights [PATH_TO_CHECKPOINT_FILE]
```

## Model Zoo

### Weakly REC 
| Method | RefCOCO | | | RefCOCO+ | | | RefCOCOg |
| ------ | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
|        | val     | testA   | testB   | val     | testA   | testB   | val-g   |
| DViN    | 67.67   | 70.90  | 59.39   | 52.54   | 57.52   | 45.31   | 55.04   |

### Weakly RES
| Method | RefCOCO | | | RefCOCO+ | | | RefCOCOg |
| ------ | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
|        | val     | testA   | testB   | val     | testA   | testB   | val-g   |
| DViN    | 61.43   | 63.81   | 56.97   | 46.79   | 51.87   | 39.85  | 46.49   |

### Pesudo Labels to training other models ( Weakly Supervsied Training Schema)

| Method            | RefCOCO |        |        | RefCOCO+ |        |        | RefCOCOg |
| ----------------- | ------- | ------ | ------ | -------- | ------ | ------ | -------- |
|                   | val     | testA  | testB  | val      | testA  | testB  | val-g    |
| DViN_SimREC    | 67.29   | 73.09  | 60.65  | 51.54    | 59.06  | 39.59  | 51.73    |
| DViN_TransVG   | 64.99   | 68.87  | 64.48  | 50.72 |57.36| 38.64  | 50.47     |


## Visualization Prediction Results (Blue box is ground truth)

Image Description :  "Kid on right in back blondish hair"

![vs](assets/vs_0.jpg)

Image Description :  "Top broccoli"

![vs](assets/vs_1.jpg)

Image Description :  "Yellow and blue vehicle close to the camera"

![vs](assets/vs_2.jpg)

Image Description :  "Second from the right"

![vs](assets/vs_4.jpg)

---

## Research Extensions（本研究延伸）

本研究基於 DViN 進行架構改進與延伸，主要目標為提升 Referring Expression Comprehension 的精確度，並探索整合即時推論能力。

### net_v2.py 架構改進

在 `models/DViR/net_v2.py` 中對原始 routing 機制進行了以下改進：

| 項目 | 原始 net.py | 改進後 net_v2.py |
|------|------------|-----------------|
| Router 類型 | 簡單 Linear Layer (1024→4) | CrossAttentionRouter（Scaled Dot-Product Attention） |
| 專家選擇策略 | Hard Top-2 selection | Soft All-4 加權融合 |
| Expert 投影層 | 無 | `expert_pool_proj`（對齊語意空間） |
| YOLO 混合權重 | 固定二選一 | 動態加權 `(1 - max_router_prob)` |
| 對比損失來源 | 僅 Top-1 expert | 全部融合後的 expert features |
| Load Balancing | 內聯計算 | 獨立方法 `load_balancing_loss()` |

**改進優點**：
- CrossAttentionRouter 讓 YOLO features 作為 query、expert features 作為 key，routing 更具語意感知能力
- Soft All-4 fusion 確保所有 expert 在訓練時都有梯度流，load balancing 更穩定
- 動態 YOLO 混合權重使模型能根據情境自適應地決定 YOLO 特徵與 expert 特徵的比重

### 未來研究方向：YOLOE 整合

計畫整合 [YOLOE](https://github.com/THU-MIG/yoloe)（Real-Time Seeing Anything，ICCV 2025）以實現即時推論能力。

**YOLOE 核心特性：**
- 305.8 FPS on T4 GPU，支援即時開放詞彙偵測
- **RepRTA**：文字提示嵌入精煉模組，inference 時可 re-parameterize 進主網路，無額外推論開銷
- **SAVPE**：空間感知視覺提示編碼器
- **Prompt-free mode**：內建 1200+ 類別大詞彙，無需外部語言模型

**整合策略選項：**

| 方案 | 說明 | 優點 | 挑戰 |
|------|------|------|------|
| **方案 A**：取代 YOLOv3 骨幹 | 以 YOLOE backbone 替換 `visual_encoder.py` 的 YOLOv3 | 獲得開放詞彙能力，推論更快 | 需重新對齊多尺度特徵輸出 (13×13 @ 1024ch) |
| **方案 B**：作為第 5 個 Expert | 在現有 4 expert 基礎上，新增 YOLOE 作為第 5 位 expert | 架構改動最小，`CrossAttentionRouter` 可直接延伸至 5 expert | 需實作 YOLOE feature extractor 的 wrapper |
| **方案 C**：兩階段設計 | YOLOE 生成 region proposals → DViN 做精細 grounding | 各司其職，解耦性強 | 兩階段訓練複雜度較高 |

目前傾向方案 B 作為第一步驗證，因為與現有架構整合成本最低。

### 模型優化方向

1. **Language Encoder 升級**：LSTM+GloVe → DistilBERT / BERT-tiny，獲得更豐富的上下文語意表示，同時維持推論速度
2. **Router Value Projection**：CrossAttentionRouter 目前只使用 Q+K attention，加入 V projection 可讓 routing 攜帶更多語意資訊
3. **多尺度 Expert Fusion**：目前 expert fusion 在單一 13×13 解析度，考慮採用 FPN 式多尺度融合以改善對小物體的定位能力
4. **端到端 RES 輸出**：EfficientSAM 已作為 expert 整合，可延伸直接輸出 segmentation mask，實現 bbox 與 mask 的統一預測

