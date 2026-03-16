# Research Log — DViN Thesis Fork (v2)

> **給新 session 的說明**：這個檔案記錄了本論文研究的所有修改、討論與規劃。開新 session 時請先讀這個檔案來了解目前進度與背景。

---

## 專案背景

- **基礎論文**：DViN — Dynamic Visual Routing Network for Weakly Supervised Referring Expression Comprehension
- **原始 Repo**：[XxFChen/DViN](https://github.com/XxFChen/DViN)
- **研究目標**：在 DViN 架構基礎上進行延伸，以方案 A 為主線——升級視覺骨幹（YOLOE）與語言編碼器（DistilBERT），並強化 CrossAttentionRouter 的學習機制

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

---

### [2026-03-16] 方案 A — YOLOE + DistilBERT 整合（主線）

**目標**：升級視覺骨幹與語言編碼器，在不改變架構邏輯的前提下全面提升特徵品質。

#### 視覺端：YOLOE 取代 YOLOv3

- YOLOE 多尺度特徵語意更豐富，對開放詞彙泛化能力更強
- CrossAttentionRouter 的 query 來自視覺特徵，視覺品質提升後 routing 決策更精準
- **工程注意**：YOLOE 的 neck 輸出 channel 數與 stride 與 YOLOv3 不完全相同，`visual_encoder.py` 需要對接，須讀 YOLOE neck 架構確認輸出維度

#### 語言端：DistilBERT 取代 LSTM+GloVe

- GloVe 是靜態詞向量，DistilBERT 具備上下文感知能力，同一詞在不同 query 中有不同表示
- 對 REC 任務中細粒度 query 理解（如「右邊的人」vs「左邊的人」）有直接改善
- **工程注意**：DistilBERT 輸出 768-dim，原 LSTM 為 512-dim，CrossAttentionRouter 的 key/query projection 維度需同步調整

#### 兩端互相增益

視覺特徵與文字特徵同時提升後，CrossAttentionRouter 做 cross-modal attention 時語意空間對齊更好，routing 的 load balancing 理論上也更穩定。

---

### [2026-03-16] net_v3.py — 方案 A 完整實作

**檔案**：`models/DViR/net_v3.py`

**新增檔案**：
- `models/distilbert_encoder.py` — DistilBERT 語言編碼器
- `models/yoloe_encoder.py` — YOLOE 視覺編碼器 wrapper（anchor-free → anchor-based 適配）
- `config/refcoco_v3.yaml` — net_v3 專用設定檔

**修改檔案**：
- `models/language_encoder.py` — 加入 DistilBERT 支援
- `datasets/dataloader.py` — 支援 DistilBERT tokenizer（條件分支）
- `train.py` / `test.py` — ModelLoader 支援 NET_VERSION、DistilBERT batch unpacking

**改動內容**：

| 項目 | net_v2.py | net_v3.py |
|------|-----------|-----------|
| 視覺編碼器 | YOLOv3 | YOLOE（可透過 VIS_ENC 切換） |
| 語言編碼器 | LSTM+GloVe (512-dim) | DistilBERT (768-dim)（可透過 LANG_ENC 切換） |
| HIDDEN_SIZE | 512 | 768 |
| Router | CrossAttentionRouter (Q/K) | CrossAttentionRouterV2 (Q/K/V projection) |
| Loss 項目 | 4 項 | 5 項（+Cross-Modal Alignment Loss） |
| Ablation 控制 | 無 | USE_V_PROJ, USE_ALIGN_LOSS 開關 |

**Config 新增參數**：
- `NET_VERSION`: 'net' / 'net_v2' / 'net_v3'
- `USE_V_PROJ`: True/False — 控制 Router Value Projection
- `USE_ALIGN_LOSS`: True/False — 控制 Cross-Modal Alignment Loss
- `LAMBDA_ALIGN`: 0.1 — alignment loss 權重
- `BERT_FREEZE_LAYERS`: 4 — 凍結 DistilBERT 前幾層
- `YOLOE_VARIANT`: 'yoloe-v2-l.pt' — YOLOE 模型變體

---

### [2026-03-16] 方案 A 優化項目

以下優化按投報率排序，前兩項建議優先實作。

#### 優先實作

**1. Router Value Projection**

在 CrossAttentionRouter 加入 V projection，Router 不只決定「選誰」，還能學習「怎麼混合」：

```python
class CrossAttentionRouter(nn.Module):
    def __init__(self, ...):
        ...
        self.v_proj = nn.Linear(expert_dim, expert_dim)  # 新增

    def forward(self, query, keys, values):
        v = self.v_proj(values)   # 新增 V projection
        attn = softmax(query @ keys.T / sqrt(d))
        return attn @ v           # 融合後攜帶更多語意
```

改動不超過 10 行，但對 routing 品質的影響可能比換 backbone 更直接。

**2. Cross-Modal Feature Alignment Loss**

在現有 `loss_expert_yolo_contrastive` 基礎上，額外加一個 text-visual alignment loss，直接監督融合後的 expert features 與 DistilBERT text features 的距離：

```python
loss_text_visual_align = contrastive_loss(
    fused_expert_features,   # 融合後的 expert features
    text_features            # DistilBERT 輸出
)

total_loss = loss_anchor_contrastive + recon_loss \
           + λ1 * sparse_loss \
           + λ2 * loss_expert_yolo_contrastive \
           + λ3 * loss_text_visual_align   # 新增
```

給 Router 更明確的學習信號，不只靠 downstream contrastive loss 間接優化。

#### 次要優化

**3. Dynamic Temperature（Router query complexity）**

讓 Router 根據 text feature 的 attention 分布集中程度，動態調整 soft/hard 程度：

- query 越複雜 → 傾向 soft all-4 fusion
- query 越簡單 → 允許更 hard 的 top-1 selection
- 實作：將 temperature 參數改為可學習，或由 text feature 預測

**4. 多尺度 Expert Fusion**

讓部分 expert（如 EfficientSAM）保留 26×26 輸出，以 FPN 式方式融合：

- 對 RefCOCO testB（小物體比例較高）提升預期明顯
- 改動有獨立的 ablation 意義

#### 錦上添花

**5. Curriculum Learning**：先用簡單 query 訓練，逐漸引入複雜 query，只需改 dataloader sampling 策略，不改模型架構。

**6. Expert Gradient Scaling**：訓練初期對弱 expert（EfficientSAM、ConvNeXt-tiny）的 routing weight 加 warm-up 係數，強迫 Router 探索所有 expert 貢獻。

#### 優先度總覽

| 優先度 | 優化項目 | 預期影響 | 實作難度 |
|--------|----------|----------|----------|
| ★★★ | Router Value Projection | routing 品質直接提升 | 低 |
| ★★★ | Cross-Modal Alignment Loss | 給 Router 更強的學習信號 | 低 |
| ★★☆ | Dynamic Temperature | 複雜 query 處理能力 | 中 |
| ★★☆ | 多尺度 Expert Fusion | 小物體定位改善 | 中 |
| ★☆☆ | Curriculum Learning | 訓練穩定性 | 低 |
| ★☆☆ | Expert Gradient Scaling | load balancing 穩定 | 低 |

---

## Ablation 設計

方案 A 天然支持乾淨的消融實驗，每組都有獨立的 ablation 意義：

| 實驗組 | 視覺端 | 語言端 | Router | 備註 |
|--------|--------|--------|--------|------|
| Baseline | YOLOv3 | LSTM+GloVe | Linear | 原始 net.py |
| +CrossAttn | YOLOv3 | LSTM+GloVe | CrossAttentionRouter | net_v2.py |
| +視覺升級 | YOLOE | LSTM+GloVe | CrossAttentionRouter | 量化視覺端貢獻 |
| +語言升級 | YOLOv3 | DistilBERT | CrossAttentionRouter | 量化語言端貢獻 |
| 完整方案 A | YOLOE | DistilBERT | CrossAttentionRouter | 主線 |
| +V Projection | YOLOE | DistilBERT | CrossAttentionRouter+V | 優化項 1 |
| +Align Loss | YOLOE | DistilBERT | CrossAttentionRouter+V | 優化項 2 |

---

## 訓練成本規劃

**資料集**：RefCOCO 系列全部（約 26K 圖）
**設定**：30 epoch、batch size 16
**平台**：vast.ai

| GPU | 單次訓練時間 | 單次費用 | 完整研究週期估計 |
|-----|------------|---------|----------------|
| A100 PCIe 40GB | ~1.0 hr | $1.2~1.6 | $15~$25 |
| A100 SXM4 40GB | ~0.75 hr | $1.4~1.7 | $15~$25 |
| A100 SXM4 80GB | ~0.68 hr | $1.5~1.9 | $15~$30 |

> 完整研究週期 = 單次費用 × 10~15（含 ablation、超參數搜尋、三資料集 evaluation）
>
> **建議**：方案 A 用 A100 PCIe 40GB 即可，顯存足夠，成本最低。

---

## 論文定位

**貢獻點**：在 weakly supervised REC 框架下，系統性驗證更強的視覺與語言表示對 expert routing 機制的影響，並提出 Value Projection 與 Cross-Modal Alignment Loss 強化 Router 的學習信號。

**定性評估**：每個改動都有清楚的動機和可量化的貢獻，ablation 結構完整，審稿人容易接受。

---

## 目前實驗結果

### Baseline（原始 DViN，net.py）

| Method | RefCOCO val | testA | testB | RefCOCO+ val | testA | testB | RefCOCOg val-g |
|--------|-------------|-------|-------|--------------|-------|-------|----------------|
| DViN   | 67.67       | 70.90 | 59.39 | 52.54        | 57.52 | 45.31 | 55.04          |

### net_v2.py 實驗結果

> 尚未完成訓練，待補充。

### 方案 A 實驗結果

> 尚未開始，待補充。

---

## 待辦事項

- [ ] 在各資料集上跑 net_v2.py baseline 實驗，與 net.py 比較
- [x] 讀 YOLOE neck 架構，確認多尺度輸出維度，規劃 `visual_encoder.py` 改動
- [x] 實作 DistilBERT 整合，調整 CrossAttentionRouter key/query projection 維度
- [x] 實作 Router Value Projection（優先）
- [x] 實作 Cross-Modal Feature Alignment Loss（優先）
- [ ] 跑完整 ablation 表格（7 組實驗）
- [ ] 評估 Dynamic Temperature 與多尺度 Expert Fusion 效益
- [ ] 安裝 ultralytics 套件並驗證 YOLOE 模型可載入
- [ ] 在 vast.ai 上做完整 forward pass 測試（驗證顯存使用量）

---

## 未來研究方向（暫緩）

### 方案 C：兩階段設計

YOLOE 出 proposals → DViN 精細 grounding。

**暫緩原因**：
- 顯存需求高（K=10 時約 18~28GB，需 A100 SXM4 80GB）
- NMS 不可微，端到端訓練需額外設計（soft NMS 或 Straight-Through Estimator）
- 弱監督設定下 Stage 1 recall 無法直接量化，debug 成本高
- 完整研究週期費用估計 $80~$180，風險較高
- 適合作為方案 A 完成後的 future work 或下一篇論文方向

### 其他暫緩項目

| 方向 | 說明 | 暫緩原因 |
|------|------|---------|
| 端到端 RES 輸出 | EfficientSAM 直接輸出 segmentation mask | 超出目前論文範疇 |
| 多 GPU 訓練 | DDP 加速 | 成本增加，單 GPU 已足夠 |

---

## 環境資訊

- Python 3.9
- PyTorch 1.11.0 + CUDA 11.3
- NVIDIA Apex（混合精度訓練）
- 主要 Expert 模型：CLIP ViT-B/32、DINOv2、EfficientSAM ViT-T、ConvNeXt-tiny
- 訓練平台：vast.ai（A100 PCIe 40GB）
