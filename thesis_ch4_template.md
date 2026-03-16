# 論文架構模板：Chapter 4 - Methodology / 方法論

> 根據實驗室學長姐論文格式整理
> 資料集：CMU-MOSI, CMU-MOSEI（使用 MMSA 套件提取的特徵）
> MoE 應用於：模態表徵學習 + 融合路由

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
- 為什麼多模態情緒分析很重要？
- 現有方法的不足之處
- 為什麼要引入 MoE 架構？MoE 能解決什麼問題？

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
- 正式定義問題
- 輸入：多模態特徵 (Text, Audio, Video)
- 輸出：情感類別（正、負面） / 情感極性分數
- 數學符號定義（如有必要）

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
- 挑戰 1：多模態異質性 (Modality Heterogeneity)
- 挑戰 2：跨模態對齊 (Cross-modal Alignment)
- 挑戰 3：模態融合策略 (Fusion Strategy)
- 挑戰 4：（根據你的研究補充其他挑戰）

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
- 系統流程概述

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
> - 包含：三個模態輸入 → Feature Encoding → MoE-based Representation Learning → Fusion Routing → Classifier → Output
> - 參考：學長姐論文中的 "System Architecture" / "Overall Architecture" 圖
> - 工具：PowerPoint 繪製

#### 4.4.1 Data Preprocessing

> 📏 **【建議篇幅】** 2 段落，約 0.3-0.5 頁 | 公式：0-1 個 | 表格：可選 1 個（特徵維度表）

**需要撰寫的內容：**
- 說明使用 MMSA 套件提供的預提取特徵
- 特徵格式說明（pkl 檔案）
- Aligned vs Unaligned 特徵的差異與選擇原因
- 各模態特徵維度說明

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
- 說明 MMSA 套件中各模態的特徵提取方式
  - Text: 通常為 BERT/GloVe embeddings
  - Audio: 通常為 COVAREP features
  - Video: 通常為 Facet/OpenFace features
- 並且說明你對輸入特徵的進一步編碼處理

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
- 融合策略說明 (Early / Late / Hybrid Fusion)
- Cross-modal Attention（有使用）
- 融合後的特徵表示

> ✍️ **【寫作風格與語氣】**
> - **語氣**：技術性、比較性
> - **可說明**：為何選擇此融合策略、與其他策略的比較優勢

#### 4.4.4 Mixture of Experts for Modality Representation and Fusion

> 📏 **【建議篇幅】** ⭐ 核心貢獻章節，約 1.0-1.5 頁 | 公式：3-5 個 | 圖表：✅ 必須 1 個架構圖
>
> 此為論文核心創新，需要比其他 subsection 更詳細的說明。

**需要撰寫的內容：**
- MoE 架構說明：應用於模態表徵學習與融合路由
- Expert Network 設計：每個模態對應一個專家
  - Text Expert：學習文本模態的專屬表徵
  - Audio Expert：學習音訊模態的專屬表徵
  - Video Expert：學習視覺模態的專屬表徵
- Soft Routing Mechanism：
  - Router / Gating Network 設計
  - 如何根據輸入特徵動態計算各專家的權重分配
  - Soft Routing 公式說明
- 融合策略：各專家表徵的加權組合形成最終融合特徵
- 與下游分類器的連接方式

> ✍️ **【寫作風格與語氣】**
> - **語氣**：技術性、創新性，強調這是你的核心貢獻
> - **開頭句式參考**：
>   - "To address the challenge of..., we propose..."
>   - "We design a Mixture of Experts architecture where..."
> - **必須包含**：數學公式說明 routing 機制

> 📊 **【Figure 需求 - 必須】Mixture of Experts for Modality Representation and Fusion**
> - 類型：模組架構圖（這是你的核心貢獻，需要清楚呈現）
> - 內容：展示 MoE 用於模態表徵學習與融合路由的完整架構
> - 包含：
>   - 各模態輸入特徵
>   - Router/Gating Network（計算權重 w₁, w₂, w₃）
>   - 三個 Expert Networks（Text Expert, Audio Expert, Video Expert）
>   - Soft Routing 加權組合過程
>   - 融合後的表徵輸出
> - 工具：PowerPoint 繪製

#### 4.4.5 Loss Function

> 📏 **【建議篇幅】** 1-2 段落，約 0.3-0.5 頁 | 公式：2-4 個（各損失 + 總損失）

**需要撰寫的內容：**
- 主要損失函數 (e.g., Cross-Entropy Loss, MSE Loss)
- 輔助損失（如有，e.g., Load Balancing Loss for MoE）
- 總損失函數公式

> ✍️ **【寫作風格與語氣】**
> - **語氣**：精確、數學化
> - **必須包含**：損失函數的數學公式
> - **格式**：`The total loss is defined as: L_total = L_main + λ · L_aux`

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
> 你的論文屬於「任務驅動型」— 動機明確（多模態情緒分析的重要性），因此可以先寫 4.1-4.2。

### 推薦撰寫順序

**Phase 1：動機與問題定義（可立即開始）**
1. **4.1 Motivation** - 撰寫多模態情緒分析的重要性、現有方法不足
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
| Fig.2 | 4.4.4 | MoE for Modality Representation and Fusion | 模組架構圖 | ⭐ 必須 | PowerPoint |

> 註：Cross-modal Attention Mechanism 已於 Chapter 3 介紹，故不重複繪製

### 圖片繪製建議

**PowerPoint 繪製技巧（架構圖）：**
- 使用一致的色彩方案（建議：藍色系 for Text, 綠色系 for Audio, 橙色系 for Video）
- 模組用圓角矩形表示
- 箭頭表示資料流向
- 加入簡短標註說明

---

*此模板根據實驗室 6 篇學長姐論文格式整理，適用於「多模態情緒分析結合 MoE 架構」研究主題。*
