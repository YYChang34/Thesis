# Research Log — DViN Thesis Fork

> **給新 session 的說明**：這個檔案記錄了本論文研究的所有修改、討論與規劃。開新 session 時請先讀這個檔案來了解目前進度與背景。

---

## 專案背景

- **基礎論文**：DViN — Dynamic Visual Routing Network for Weakly Supervised Referring Expression Comprehension
- **原始 Repo**：[XxFChen/DViN](https://github.com/XxFChen/DViN)
- **研究目標**：在 DViN 架構基礎上進行延伸，探索更強的 expert routing 機制與即時推論整合

### 任務說明

**Referring Expression Comprehension (REC)**：給定一張圖片與一段文字描述（如「右後方金髮小孩」），模型預測文字所指物體的邊界框。

**弱監督設定**：不使用 box-level annotation 監督，改用 anchor-based contrastive learning。

---

## 原始架構摘要

### 核心模型（`models/DViR/net.py`）

```
輸入: Image (B, 3, H, W) + Text query
  │
  ├── YOLOv3 Visual Encoder → 多尺度特徵 (13×13, 26×26, 52×52 @ 1024ch)
  ├── Language Encoder (LSTM + GloVe) → text feature (B, 512)
  │
  ├── 4 個 Frozen Expert Encoders（均輸出 13×13 @ 1024ch）：
  │     ├── CLIP (openai/clip-vit-base-patch32)
  │     ├── DINOv2
  │     ├── EfficientSAM (ViT-T)
  │     └── ConvNeXt-tiny
  │
  ├── Router: Linear(1024→4) → Softmax → Hard Top-2 selection
  ├── Expert Fusion: 加權合併 Top-2 expert features
  ├── 與 YOLO features 融合
  │
  ├── WeakREChead: Anchor-Prompt Contrastive Loss
  └── 輸出: bbox prediction (inference) / total loss (training)
```

### 訓練 Loss

```
total_loss = loss_anchor_contrastive + recon_loss + λ * sparse_loss + loss_expert_yolo_contrastive
```

---

## 修改紀錄

### [2026-03-16] net_v2.py — CrossAttentionRouter + Soft All-4 Fusion

**檔案**：`models/DViR/net_v2.py`

**改動內容**：

| 項目 | 原始 net.py | 改進後 net_v2.py |
|------|------------|-----------------|
| Router 類型 | `Linear(1024→4)` | `CrossAttentionRouter`（Scaled Dot-Product Attention） |
| 專家選擇策略 | Hard Top-2 selection | Soft All-4 加權融合 |
| Expert 投影層 | 無 | `expert_pool_proj`（特徵對齊） |
| YOLO 混合權重 | 固定二選一 | 動態加權 `(1 - max_router_prob)` |
| 對比損失來源 | 僅 Top-1 expert | 全部融合後的 expert features |
| Load Balancing | 內聯計算 | 獨立方法 `load_balancing_loss()` |

**新增的 CrossAttentionRouter**：

```python
class CrossAttentionRouter(nn.Module):
    # YOLO features 作為 query，expert pooled features 作為 key
    # Scaled dot-product attention → softmax → router weights
    # 相比 Linear router，routing 更具語意感知能力
```

**改進效果**：
- Soft All-4 fusion 確保所有 expert 在訓練時都有梯度流，load balancing 更穩定
- 動態 YOLO 混合權重使模型能根據 router confidence 自適應調整特徵比重
- 對比損失使用融合後的 expert features（而非只有 Top-1），提供更完整的優化信號

---

## 未來研究方向

### YOLOE 整合計畫

**目標**：整合 [YOLOE](https://github.com/THU-MIG/yoloe)（ICCV 2025，Real-Time Seeing Anything）以實現即時推論。

**YOLOE 核心特性**：
- 305.8 FPS on T4 GPU
- **RepRTA**：文字提示嵌入精煉，re-parameterize 後無額外推論開銷
- **SAVPE**：空間感知視覺提示編碼器
- **Prompt-free mode**：內建 1200+ 類別大詞彙

**整合策略選項**：

| 方案 | 說明 | 難度 | 狀態 |
|------|------|------|------|
| **方案 A**：取代 YOLOv3 | YOLOE backbone 替換 `visual_encoder.py` 的 YOLOv3 | 中 | 評估中 |
| **方案 B**：第 5 個 Expert | YOLOE encoder 加入現有 expert 池，router 改為 5 expert | 低 | **初步傾向** |
| **方案 C**：兩階段設計 | YOLOE 出 proposals → DViN 精細 grounding | 高 | 評估中 |

**方案 B 實作要點**（若採用）：
1. 實作 `yoloe_encoder.py`，wrapper YOLOE feature extractor，輸出 13×13 @ 1024ch
2. `net_v2.py` 中將 `num_experts=4` 改為 `num_experts=5`
3. `CrossAttentionRouter` 架構不需改動（已參數化）
4. 更新 `expert_pool_proj` 對應維度

### 其他模型優化方向

| 方向 | 說明 | 優先度 |
|------|------|--------|
| Language Encoder 升級 | LSTM+GloVe → DistilBERT / BERT-tiny | 中 |
| Router Value Projection | CrossAttentionRouter 加入 V projection，攜帶更多語意資訊 | 中 |
| 多尺度 Expert Fusion | FPN 式多尺度融合，改善小物體定位 | 低 |
| 端到端 RES 輸出 | EfficientSAM expert 直接輸出 segmentation mask | 低 |

---

## 目前實驗結果

### Baseline（原始 DViN，net.py）

| Method | RefCOCO val | testA | testB | RefCOCO+ val | testA | testB | RefCOCOg val-g |
|--------|-------------|-------|-------|--------------|-------|-------|----------------|
| DViN   | 67.67       | 70.90 | 59.39 | 52.54        | 57.52 | 45.31 | 55.04          |

### net_v2.py 實驗結果

> 尚未完成訓練，待補充。

---

## 待辦事項

- [ ] 在各資料集上跑 net_v2.py baseline 實驗，與 net.py 比較
- [ ] 評估 YOLOE 整合方案（方案 A vs 方案 B）
- [ ] 若採用方案 B，實作 `yoloe_encoder.py`
- [ ] 考慮 Language Encoder 升級實驗（DistilBERT）

---

## 環境資訊

- Python 3.9
- PyTorch 1.11.0 + CUDA 11.3
- NVIDIA Apex（混合精度訓練）
- 主要 Expert 模型：CLIP ViT-B/32、DINOv2、EfficientSAM ViT-T、ConvNeXt-tiny
