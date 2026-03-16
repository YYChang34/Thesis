# 論文架構模板：Chapter 4 - Methodology / 方法論

> 根據實驗室學長姐論文格式整理
> 資料集：RefCOCO, RefCOCO+, RefCOCOg, ReferIT（Weakly Supervised 設定）
> MoE 應用於：視覺專家路由 + 動態特徵融合

---

## Chapter 4: Methodology / 方法論

> 📏 **【篇幅參考】** 根據 6 篇學長姐論文分析，Chapter 4 總長度約 **5-6 頁**

---

### 4.1 Motivation

> 📏 **【建議篇幅】** 2 段落，約 0.3-0.5 頁
>
> | 項目 | 建議 |
> |------|------|
> | 段落數 | 2 段 |
> | 頁數 | 0.3-0.5 頁 |
> | 圖表 | ❌ 無 |
> | 公式 | ❌ 無 |
>
> **核心內容（僅包含）：**
> - 領域重要性（1-2 句）
> - 現有方法不足（1-2 句）
> - 本研究動機（1-2 句）
>
> **不應包含：**
> - ❌ 詳細的技術比較（放 Related Work 或 4.3）
> - ❌ 具體的方法設計（放 4.4）
> - ❌ 數學公式

**需要撰寫的內容：**
- 為什麼 Referring Expression Comprehension（REC）重要？（視覺語言理解在 AI 應用中的核心地位）
- 現有強監督方法的不足：需要大量昂貴的 bounding box 標注，難以大規模擴展
- 弱監督設定的挑戰與機會：僅用影像層級的類別標籤進行訓練
- 為什麼需要多個視覺專家？單一 backbone 無法同時覆蓋語義理解（CLIP）、細粒度特徵（DINOv2）、邊界感知（EfficientSAM）與局部紋理（ConvNeXt）的全面需求

> ✍️ **【寫作風格與語氣】**
> - **語氣**：從宏觀角度切入，強調研究領域的重要性與現實應用價值
> - **開頭句式參考**：
>   - "It is common for... to..." (從普遍現象引入)
>   - "XXX serves as fundamental components of..." (強調重要性)
>   - "The World Health Organization reports that..." (引用權威數據)
>   - "Global data volume is predicted to grow..." (用統計數據開頭)
> - **段落結構**：現實問題 → 現有方法限制 → 本研究的必要性
> - **範例開頭**：
>   - "Facial expressions serve as fundamental components of human non-verbal interaction, transmitting emotional states, intentions, and social cues."
>   - "It is common for real-world datasets to be long-tailed. For example, in medical imaging..."

---

### 4.2 Problem Statement

> 📏 **【建議篇幅】** 2-3 段落，約 0.3-0.5 頁
>
> | 項目 | 建議 |
> |------|------|
> | 段落數 | 2-3 段 |
> | 頁數 | 0.3-0.5 頁 |
> | 圖表 | ❌ 無 |
> | 公式 | ✅ 1-2 個（輸入輸出定義）|
>
> **核心內容（僅包含）：**
> - 任務定義（1 句）
> - 輸入表示 $\mathbf{X}_m \in \mathbb{R}^{T \times d}$
> - 輸出定義（sentiment score / class）
> - 目標陳述（1 句）
>
> **不應包含：**
> - ❌ 損失函數公式（放 4.4.5）
> - ❌ 詳細的輸出頭設計（放 4.4）
> - ❌ Aligned/Unaligned 詳細說明（放 5.1 Datasets）

**需要撰寫的內容：**
- 正式定義 Weakly Supervised Referring Expression Comprehension 任務
- 輸入：影像 $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ 與自然語言查詢 $\mathbf{Q} = \{w_1, w_2, \ldots, w_T\}$
- 輸出：Bounding Box 預測 $\hat{b} = (x, y, w, h)$，定位語言描述所指涉的目標物件
- 弱監督設定說明：訓練時不使用 box annotation，僅以影像層級標籤或文字描述監督
- 數學符號定義（任務目標、模型學習函數）

> ✍️ **【寫作風格與語氣】**
> - **語氣**：正式、精確、學術性，使用數學符號定義問題
> - **開頭句式參考**：
>   - "The task of XXX is to..." (直接定義任務)
>   - "The objective of XXX is to identify and categorize..."
>   - "XXX addresses the challenge of..."
> - **必須包含**：
>   - 輸入定義：`The input is a sequence of... X = {x₁, x₂, ..., xₙ}`
>   - 輸出定義：`The output is... Y = {y₁, y₂, ..., yₘ}`
>   - 目標陳述：`The goal is to accurately predict...`
> - **範例**：
>   - "The task of Sign Language Translation (SLT) is to translate a sign language video clip into a corresponding text sentence. The input of SLT is a sequence of T video frames V = {v₁, v₂, ..., vₜ}..."

---

### 4.3 Research Challenges

> 📏 **【建議篇幅】** 3-4 段落，約 0.5 頁
>
> | 項目 | 建議 |
> |------|------|
> | 段落數 | 3-4 段（每個挑戰 1 段）|
> | 頁數 | 0.4-0.6 頁 |
> | 圖表 | ❌ 無 |
> | 公式 | ❌ 無 |
>
> **核心內容（僅包含）：**
> - 開頭總述（1 句說明有幾個挑戰）
> - 每個挑戰：問題描述 → 為何困難（2-3 句/挑戰）
>
> **不應包含：**
> - ❌ 解決方案（放 4.4）
> - ❌ 詳細的技術分析

**需要撰寫的內容：**
- 挑戰 1：視覺–語言語義鴻溝（Visual-Linguistic Semantic Gap）：影像特徵與文字描述處於不同的特徵空間，跨模態對齊困難
- 挑戰 2：弱監督學習（Weak Supervision）：訓練時缺乏精確的 bounding box 標注，模型難以直接學習定位目標
- 挑戰 3：視覺專家異質性（Expert Heterogeneity）：CLIP、DINOv2、EfficientSAM、ConvNeXt 四個 backbone 輸出特徵空間差異大，難以有效融合
- 挑戰 4：動態路由不穩定性（Router Instability）：CrossAttentionRouter 的 soft routing 權重可能在訓練初期不穩定，導致 load imbalance 或 expert collapse

> ✍️ **【寫作風格與語氣】**
> - **語氣**：分析性、條列式，清楚說明技術困難點
> - **開頭句式參考**：
>   - "There are X challenges in XXX task, as listed below." (直接列舉)
>   - "Addressing XXX presents two/three main challenges." (總結式開頭)
>   - "The extension of XXX to YYY faces two main challenges."
> - **結構格式**：
>   - 使用編號列表：1. Challenge A: ..., 2. Challenge B: ...
>   - 或使用「The first challenge... The second challenge...」
> - **每個挑戰的寫法**：問題描述 → 具體困難 → 為何難以解決
> - **範例**：
>   - "There are three challenges in Facial Expression Recognition, which are listed as follows. 1. Inter-class similarity: Certain facial expressions share overlapping facial characteristics, making it difficult to distinguish between them..."

---

### 4.4 Proposed System Architecture - 根據code

> 📏 **【建議篇幅】** 4.4 整體約 3.5-4.5 頁（含所有 subsection）
>
> | 項目 | 建議 |
> |------|------|
> | 4.4 總述 | 0.3-0.5 頁 |
> | 4.4.1-4.4.5 | 各 0.5-1.0 頁 |
> | 圖表 | ✅ 1-2 個架構圖 |
> | 公式 | ✅ 多個（各模組的數學定義）|
>
> **核心內容（僅包含）：**
> - 架構總覽（搭配 Figure）
> - 各模組簡介（1-2 句/模組）
>
> **Subsection 篇幅分配：**
> | Subsection | 頁數 | 重點 |
> |------------|------|------|
> | 4.4.1 Data Preprocessing | 0.3-0.5 | 特徵來源、維度 |
> | 4.4.2 Feature Encoding | 0.5-0.7 | 編碼方式 |
> | 4.4.3 Multi-modal Fusion | 0.5-0.7 | 融合策略 |
> | 4.4.4 MoE Module | 1.0-1.5 | ⭐ 核心貢獻，需詳細 |
> | 4.4.5 Loss Function | 0.3-0.5 | 損失公式 |

**需要撰寫的內容：**
- 整體架構圖 (Figure)
- 系統流程概述：Image + Language Query → Visual Encoder (YOLOv3) / Language Encoder (LSTM+GloVe) / 4 Expert Encoders (frozen) → CrossAttentionRouter → Soft Expert Fusion → Anchor-Prompt Contrastive Learning → BBox Prediction

> ✍️ **【寫作風格與語氣】**
> - **語氣**：描述性、流程導向，搭配圖片說明
> - **開頭句式參考**：
>   - "As shown in Figure X, our system comprises three parts: A, B, and C."
>   - "The architecture of the proposed model is illustrated in Figure X, which is divided into three modules..."
>   - "Figure X shows the architecture of our proposed model."
>   - "Our system is designed for an end-to-end pipeline for..."
> - **結構**：先總述整體架構 → 簡要說明各模組功能 → 各子章節詳細展開
> - **範例**：
>   - "The proposed system architecture is illustrated in Figure 6, which consists of five components: Data Augmentation, Feature Extraction Backbone, Adaptive Cross-Attention Mechanism, Cross-Modal Alignment, and Transformer Classifier."

> 📊 **【Figure 需求 - 必須】Overall System Architecture**
> - 類型：系統架構圖
> - 內容：展示整個模型從輸入到輸出的完整流程
> - 包含：Image + Language Query → YOLOv3 Visual Encoder / LSTM+GloVe Language Encoder / 4 Frozen Expert Encoders (CLIP, DINOv2, EfficientSAM, ConvNeXt) → CrossAttentionRouter → Soft Expert Fusion → Anchor-Prompt Contrastive Learning → BBox Prediction
> - 參考：學長姐論文中的 "System Architecture" / "Overall Architecture" 圖
> - 工具：PowerPoint 繪製

#### 4.4.1 Data Preprocessing

> 📏 **【建議篇幅】** 2 段落，約 0.3-0.5 頁 | 公式：0-1 個 | 表格：可選 1 個（特徵維度表）

**需要撰寫的內容：**
- 說明 RefCOCO / RefCOCO+ / RefCOCOg / ReferIT 資料集格式（COCO 影像 + referring expression 文字標注 + bounding box）
- 資料集分割說明（train / val / testA / testB）及各分割的特性
- 影像預處理流程：resize 至 416×416、正規化（mean/std）
- 語言查詢預處理：tokenize、padding、GloVe word embedding lookup
- Weakly Supervised 設定說明：訓練時僅使用文字描述監督，不使用 box annotation

> ✍️ **【寫作風格與語氣】**
> - **語氣**：技術性、說明性
> - **開頭句式參考**：
>   - "The Data Preprocessing stage is to convert raw... into a format suitable for..."
>   - "This stage encompasses two main steps: ... and ..."
> - **範例**：
>   - "The Data Preprocessing stage is to convert raw sign language videos into a format suitable for the subsequent translation model. This stage encompasses two main steps: visual feature extraction from video frames and the generation of sign embeddings."

#### 4.4.2 Feature Extraction / Feature Encoding

> 📏 **【建議篇幅】** 2-3 段落，約 0.5-0.7 頁 | 公式：1-2 個 | 圖表：可選

**需要撰寫的內容：**
- **Visual Encoder（YOLOv3）**：以 DarkNet-53 backbone 提取多尺度視覺特徵，輸出 13×13、26×26、52×52 三個尺度 @ 1024ch；主要使用 13×13 feature map 做後續融合
- **Language Encoder（LSTM + GloVe）**：將 query tokens 以 300-d GloVe embeddings 初始化，經 Bi-LSTM 編碼後取最後隱層狀態，輸出 (B, 512) 文字特徵
- **Expert Encoders（4 個，均為 frozen）**：
  - CLIP ViT-B/32：強語義對齊能力，輸出影像–語言對齊特徵
  - DINOv2（ViT-S/14 或 ViT-B/14）：細粒度視覺特徵，self-supervised 訓練
  - EfficientSAM（ViT-T backbone）：邊界感知特徵，適合目標定位
  - ConvNeXt-tiny：局部紋理與結構特徵
- 所有 expert 輸出經 projection layer 統一至 13×13 @ 1024ch，與 YOLOv3 特徵尺寸對齊

> ✍️ **【寫作風格與語氣】**
> - **語氣**：技術性、解釋性，說明選擇的理由
> - **開頭句式參考**：
>   - "We select XXX as our model's backbone, chosen for its strong capabilities in..."
>   - "The foundation of our... is a feature extraction process centered on..."
> - **範例**：
>   - "We select PERT as our model's backbone, chosen for its strong capabilities in generating context-aware embeddings from Chinese text."

#### 4.4.3 Multi-modal Fusion

> 📏 **【建議篇幅】** 2-3 段落，約 0.5-0.7 頁 | 公式：1-2 個 | 圖表：可選

**需要撰寫的內容：**
- 說明 Visual-Language Cross-Attention 對齊機制：以 YOLOv3 pooled feature 為 query，以各 expert feature 為 key/value，計算注意力權重
- 說明為何選擇 attention-based fusion 而非 early/late fusion：attention 可動態依據語言查詢內容調整各 expert 的貢獻比重
- 跨模態特徵投影：確保 language feature（512-d）與 visual feature（1024-d）維度對齊後再進行 attention 運算
- 融合後的特徵表示：weighted expert feature 作為 4.4.4 MoE module 的輸入

> ✍️ **【寫作風格與語氣】**
> - **語氣**：技術性、比較性
> - **可說明**：為何選擇 attention-based fusion、相較於 early/late fusion 的優勢（動態、query-aware）

#### 4.4.4 Visual Expert Routing and Dynamic Fusion

> 📏 **【建議篇幅】** ⭐ 核心貢獻章節，約 1.0-1.5 頁 | 公式：3-5 個 | 圖表：✅ 必須 1 個架構圖
>
> 此為論文核心創新，需要比其他 subsection 更詳細的說明。

**需要撰寫的內容：**
- MoE 架構說明：應用於視覺專家路由與動態特徵融合
- Expert Network 設計：4 個視覺 backbone，每個專家擅長不同面向的視覺理解
  - CLIP Expert：語義對齊能力，強調影像–語言跨模態特徵
  - DINOv2 Expert：細粒度視覺表徵，self-supervised 自監督特徵
  - EfficientSAM Expert：邊界與分割感知特徵，強調目標輪廓定位
  - ConvNeXt Expert：局部紋理與結構特徵，互補前三者的全局偏向
- CrossAttentionRouter（Soft Routing Mechanism）：
  - 以 YOLOv3 pooled feature 作為 query：$\mathbf{q} = \mathbf{W}_q \cdot \mathbf{f}_\text{yolo}$
  - 以各 expert pooled feature 作為 key：$\mathbf{k}_i = \mathbf{W}_k \cdot \mathbf{f}_i^{\text{exp}}$
  - 計算 soft routing weight：$\mathbf{w} = \text{softmax}\left(\mathbf{q}\mathbf{K}^\top / \sqrt{d}\right) \in \mathbb{R}^4$
  - 全 4 個 expert 均參與（Soft All-4，非 Hard Top-K 選擇）
- 動態融合策略：
  - Expert 加權融合：$\mathbf{F}_\text{expert} = \sum_{i=1}^{4} w_i \cdot \mathbf{F}_i$
  - YOLOv3 與 expert 混合：$\mathbf{F}_\text{final} = (1 - w_{\max}) \cdot \mathbf{f}_\text{yolo} + w_{\max} \cdot \mathbf{F}_\text{expert}$
  - $w_{\max}$ 動態調整 YOLO 與 expert 的貢獻比例
- 與 Anchor-Prompt Contrastive Learning 的連接方式

> ✍️ **【寫作風格與語氣】**
> - **語氣**：技術性、創新性，強調這是你的核心貢獻
> - **開頭句式參考**：
>   - "To address the challenge of..., we propose..."
>   - "We design a Visual Expert Routing architecture where..."
>   - "The proposed CrossAttentionRouter dynamically assigns..."
> - **必須包含**：數學公式說明 CrossAttentionRouter 的 routing 機制與動態融合策略

> 📊 **【Figure 需求 - 必須】Visual Expert Routing and Dynamic Fusion**
> - 類型：模組架構圖（這是你的核心貢獻，需要清楚呈現）
> - 內容：展示 MoE 用於視覺專家路由與動態融合的完整架構
> - 包含：
>   - 4 個 Expert Encoder 輸入特徵（CLIP, DINOv2, EfficientSAM, ConvNeXt）
>   - CrossAttentionRouter（以 YOLO feature 為 query，計算 soft weight w₁, w₂, w₃, w₄）
>   - Soft All-4 加權組合過程
>   - YOLO 與 expert 動態混合（1 - w_max / w_max 權重）
>   - 融合後的 F_final 輸出至 Anchor-Prompt Contrastive Learning
> - 工具：PowerPoint 繪製

#### 4.4.5 Loss Function

> 📏 **【建議篇幅】** 1-2 段落，約 0.3-0.5 頁 | 公式：2-4 個（各損失 + 總損失）

**需要撰寫的內容：**
- **Anchor-Prompt Contrastive Loss（$\mathcal{L}_\text{anchor}$）**：核心弱監督損失，拉近 anchor region 特徵與語言 prompt 特徵的距離，推遠負樣本
- **Reconstruction Loss（$\mathcal{L}_\text{recon}$）**：輔助監督，確保 expert 特徵不偏離原始影像語義
- **Sparse Loss（$\mathcal{L}_\text{sparse}$）**：鼓勵 router 產生稀疏分配，避免所有 expert 被平均使用
- **Expert-YOLO Contrastive Loss（$\mathcal{L}_\text{contrast}$）**：拉近 YOLO 特徵與 expert fusion 特徵，確保兩者語義一致
- 總損失函數公式：$\mathcal{L}_\text{total} = \mathcal{L}_\text{anchor} + \mathcal{L}_\text{recon} + \lambda \cdot \mathcal{L}_\text{sparse} + \mathcal{L}_\text{contrast}$

> ✍️ **【寫作風格與語氣】**
> - **語氣**：精確、數學化
> - **必須包含**：各損失函數的數學公式及其監督信號來源說明
> - **格式**：先個別說明各損失，最後給出總損失公式

---

## Chapter 4 寫作風格總結

### 語氣特徵
| 章節 | 語氣特徵 | 常用動詞 |
|------|----------|----------|
| 4.1 Motivation | 宏觀、重要性強調 | serves as, is common, reports that |
| 4.2 Problem Statement | 正式、精確、數學化 | is defined as, the task of, the goal is |
| 4.3 Research Challenges | 分析性、條列式 | faces X challenges, presents difficulties |
| 4.4 Proposed Architecture | 描述性、流程導向 | is illustrated in, comprises, consists of |

### 常用轉折與連接詞
- **承接**：Furthermore, Moreover, Additionally, In addition
- **對比**：However, In contrast, On the other hand, While
- **因果**：Therefore, Thus, Consequently, As a result
- **舉例**：For example, For instance, Specifically
- **總結**：In summary, Overall, In conclusion

---

## 附錄：學長姐 Chapter 4 結構參考

| 學長姐 | 結構特點 |
|--------|----------|
| 張世何 | 4.1~4.3 (標準開頭) + 4.4 架構 + 4.5~4.7 各模組詳述 |
| 徐偉哲 | 4.1 含子章節 (Motivation, System Model 等) + 4.2~4.3 |
| 林冠斌 | 4.1~4.4 (標準) + 4.5~4.7 各模組 + 4.7.x 細部子章節 |
| 江子青 | 4.1~4.3 (標準) + 4.4 含多個子章節 (4.4.1~4.4.5) |
| 王俊顏 | 4.1~4.4 (標準) + 4.4.x 各模組 + 4.5 Discussion |
| 許顥蓉 | 4.1~4.3 (標準) + 4.4 含 6 個子章節 (4.4.1~4.4.6) |

---

## 建議撰寫順序

> 📊 **根據學長姐論文分析結果調整**
>
> 分析發現兩種寫作模式：
> - **任務驅動型**（江子青、王俊顏、許顥蓉）：4.1-4.2 與 4.4 低耦合，可先寫
> - **技術驅動型**（徐偉哲、林冠斌）：4.2 預告技術需求，需與 4.4 平行寫
>
> 你的論文屬於「任務驅動型」— 動機明確（Weakly Supervised REC 的重要性與弱監督挑戰），因此可以先寫 4.1-4.2。

### 推薦撰寫順序

**Phase 1：動機與問題定義（可立即開始）**
1. **4.1 Motivation** - 撰寫 Weakly Supervised REC 的重要性、強監督方法的不足、多視覺專家的動機
2. **4.2 Problem Statement** - 定義輸入輸出、任務目標
3. **4.2 末尾加過渡句** - 埋下架構伏筆，例如：
   > "Given the heterogeneous nature of multimodal data and the varying contribution of each modality, this task calls for models capable of learning modality-specific representations and dynamically routing information across modalities."

**Phase 2：架構設計（需要 code 配合）**
4. **4.4 Proposed Architecture** - 根據實作定義架構和各模組
5. **4.3 Research Challenges** - 根據 4.4 回頭補充具體挑戰（可選調整）
6. **4.4.5 Loss Function** - 確認損失函數設計

### 學長姐寫作模式參考

| 論文 | 4.1-4.2 與 4.4 耦合度 | 寫作建議 |
|------|----------------------|----------|
| 許顥蓉 | 🟢 低（任務驅動） | 4.1-4.2 可完全獨立先寫 |
| 王俊顏 | 🟢 低（任務驅動） | 4.1-4.2 可完全獨立先寫 |
| 江子青 | 🟢 低（任務驅動） | 4.1-4.2 可完全獨立先寫 |
| 張世何 | 🟡 中 | 4.1 可先寫，4.2 建議與 4.4 平行 |
| 林冠斌 | 🔴 高（技術驅動） | 4.2 需與 4.4 平行寫 |
| 徐偉哲 | 🔴 高（技術驅動） | 4.2 需與 4.4 平行寫 |

---

## Chapter 4 圖片清單

| 編號 | 章節 | 圖片名稱 | 類型 | 優先度 | 工具 |
|------|------|----------|------|--------|------|
| Fig.1 | 4.4 | Overall System Architecture | 系統架構圖 | ⭐ 必須 | PowerPoint |
| Fig.2 | 4.4.4 | Visual Expert Routing and Dynamic Fusion (CrossAttentionRouter + 4 Experts) | 模組架構圖 | ⭐ 必須 | PowerPoint |

> 註：CrossAttentionRouter 的 cross-attention 機制可與 Chapter 3 相關內容對應，4.4.4 著重展示 MoE routing 架構

### 圖片繪製建議

**PowerPoint 繪製技巧（架構圖）：**
- 使用一致的色彩方案（建議：藍色系 for Text, 綠色系 for Audio, 橙色系 for Video）
- 模組用圓角矩形表示
- 箭頭表示資料流向
- 加入簡短標註說明

---

*此模板根據實驗室 6 篇學長姐論文格式整理，適用於「Weakly Supervised Referring Expression Comprehension 結合 Visual Expert Routing（DViN + CrossAttentionRouter）」研究主題。*
