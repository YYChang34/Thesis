# 論文架構模板：Chapter 5 - Experiments / 實驗

> 根據實驗室學長姐論文格式整理
> 資料集：CMU-MOSI, CMU-MOSEI（使用 MMSA 套件提取的特徵）
> MoE 應用於：模態表徵學習 + 融合路由

---

## Chapter 5: Experiments / 實驗 - 根據code

> 📏 **【篇幅參考】** 根據 6 篇學長姐論文分析，Chapter 5 總長度約 **8-12 頁**
>
> | Section | 最小 | 建議 | 最大 |
> |---------|------|------|------|
> | 5.1 Datasets | 0.5 頁 | 1.0 頁 | 2.0 頁 |
> | 5.2 Metrics | 0.3 頁 | 0.8 頁 | 1.5 頁 |
> | 5.3 Setup | 0.4 頁 | 0.8 頁 | 1.2 頁 |
> | 5.4 Results | 2.0 頁 | 4.0 頁 | 6.0 頁 |
> | 5.5 Ablation | 0.3 頁 | 0.8 頁 | 1.5 頁 |

---

### 5.1 Datasets

> 📏 **【建議篇幅】** 約 0.8-1.2 頁
>
> | 項目 | 建議 |
> |------|------|
> | 頁數 | 0.8-1.2 頁 |
> | 表格 | ✅ 1 個（資料集統計表）|
> | 圖表 | ❌ 通常無 |
>
> **核心內容（僅包含）：**
> - 資料集名稱、來源、引用
> - 樣本數量、類別分布
> - Train/Valid/Test 切分
>
> **不應包含：**
> - ❌ 特徵提取細節（放 4.4.1 或 5.1.3）
> - ❌ 資料增強方法（放 4.4.1）

**需要撰寫的內容：**

> ✍️ **【寫作風格與語氣】**
> - **語氣**：說明性、客觀描述
> - **開頭句式參考**：
>   - "In this research, our experiments are conducted on two widely adopted XXX datasets: A and B."
>   - "The performance of our proposed model is evaluated on X datasets to assess its effectiveness across various..."
>   - "To evaluate our model's performance on both public benchmarks and..., we utilize two distinct datasets..."
> - **範例**：
>   - "In this research, our experiments are conducted on two widely adopted FER datasets: AffectNet and RAF-DB. The detailed description of each dataset is provided as follows."

#### 5.1.1 CMU-MOSI
- 資料集簡介
- 資料統計 (樣本數、類別分布)
- Train / Valid / Test 切分方式
- 使用的特徵檔案：
  - `MOSI_aligned.pkl`
  - `MOSI_unaligned.pkl`

> ✍️ **【寫作風格與語氣】**
> - **格式**：可使用 bullet points 或段落描述
> - **必須包含**：資料集來源、樣本數量、標籤類型、切分方式
> - **範例格式**：
>   - "• CMU-MOSI [ref] is a widely recognized dataset for multimodal sentiment analysis, containing X video clips..."

#### 5.1.2 CMU-MOSEI
- 資料集簡介
- 資料統計 (樣本數、類別分布)
- Train / Valid / Test 切分方式
- 使用的特徵檔案：
  - `MOSEI_aligned.pkl`
  - `MOSEI_unaligned.pkl`

#### 5.1.3 Feature Description
- 說明 MMSA 套件提取的特徵內容
- Aligned 特徵：已對齊至相同時間步長
- Unaligned 特徵：保持各模態原始時間步長
- 各模態特徵維度（Text dim, Audio dim, Video dim）

---

### 5.2 Evaluation Metrics

> 📏 **【建議篇幅】** 約 0.6-1.0 頁
>
> | 項目 | 建議 |
> |------|------|
> | 頁數 | 0.6-1.0 頁 |
> | 公式 | ✅ 4-5 個（每個指標定義）|
> | 表格 | ❌ 通常無 |
>
> **核心內容（僅包含）：**
> - 每個指標的名稱和數學定義
> - 為何選擇這些指標（1 句）
>
> **不應包含：**
> - ❌ 指標的詳細歷史或比較

**需要撰寫的內容：**
- Accuracy (Acc-2, Acc-7)
- F1 Score (Weighted F1)
- Mean Absolute Error (MAE)
- 各指標的定義與計算方式

> ✍️ **【寫作風格與語氣】**
> - **語氣**：精確、定義性
> - **開頭句式參考**：
>   - "To evaluate/assess the performance of our model, we employ X commonly used metrics, including..."
>   - "We employ four commonly used metrics to evaluate the performance of our model, including Accuracy, Precision, Recall, and F1-Score."
>   - "To measure the performance of our XXX model, we adopt the XXX and YYY scores, which are standard and commonly used metrics in the field of..."
> - **必須包含**：每個指標的數學定義公式
> - **範例**：
>   - "To assess the classification efficacy of our model, we employ the Top-1 Accuracy as our evaluation metric. This metric quantifies the proportion of samples for which the model's highest-probability prediction correctly matches the ground-truth label, which is formally defined in Equation X."

---

### 5.3 Experimental Setup

> 📏 **【建議篇幅】** 約 0.7-1.0 頁（含 5.3.1 + 5.3.2）
>
> | 項目 | 建議 |
> |------|------|
> | 頁數 | 0.7-1.0 頁 |
> | 表格 | ✅ 1-2 個（環境表 + 超參數表）|
> | 公式 | ❌ 通常無 |
>
> **核心內容（僅包含）：**
> - 硬體/軟體環境
> - 超參數設定
> - 訓練細節（epochs, batch size 等）
>
> **不應包含：**
> - ❌ 模型架構細節（放 4.4）
> - ❌ 損失函數（放 4.4.5）

**需要撰寫的內容：**

#### 5.3.1 Experimental Environment

> 📏 **【建議篇幅】** 0.3 頁 | 表格：✅ 1 個
- 作業系統：Ubuntu 24.04 LTS
- CPU：AMD Ryzen 9 9900X
- RAM：64GB DRAM
- GPU：NVIDIA RTX 4070 Ti Super
- 深度學習框架：PyTorch 2.10.0
- CUDA 版本：13.0

> ✍️ **【寫作風格與語氣】**
> - **語氣**：簡潔、列表式
> - **格式**：通常使用表格呈現，或簡單列點

#### 5.3.2 Implementation Details (透過Optuna搜索)

> 📏 **【建議篇幅】** 0.4-0.5 頁 | 表格：✅ 1 個（超參數表）

**需要撰寫的內容：**
- Batch Size
- Learning Rate
- Optimizer (e.g., Adam, AdamW)
- Training Epochs
- 其他超參數設定

> ✍️ **【寫作風格與語氣】**
> - **格式**：可使用表格或段落說明
> - **必須包含**：所有重要的超參數設定

---

### 5.4 Experimental Results and Analysis

> 📏 **【建議篇幅】** ⭐ 最重要章節，約 3.5-5.0 頁
>
> | 項目 | 建議 |
> |------|------|
> | 頁數 | 3.5-5.0 頁 |
> | 表格 | ✅ 2-3 個（主結果表）|
> | 圖表 | 可選（視覺化比較）|
>
> **核心內容（僅包含）：**
> - 與 baseline 的數值比較
> - 結果分析（為何優/劣）
> - 各資料集的詳細結果
>
> **Subsection 篇幅分配：**
> | Subsection | 頁數 |
> |------------|------|
> | 5.4.1 Baseline Comparison | 1.0-1.5 頁 |
> | 5.4.2 Results on MOSI | 1.0-1.5 頁 |
> | 5.4.3 Results on MOSEI | 1.0-1.5 頁 |

**需要撰寫的內容：**

> ✍️ **【寫作風格與語氣】**
> - **語氣**：分析性、比較性
> - **開頭句式參考**：
>   - "In this section, we present and analyze the experimental results of our work in comparison with previous works [refs]."
>   - "This section is dedicated to the evaluation of our method, comparing its performance with other existing models, including [refs]."
>   - "We evaluate our work on the X, Y, and Z datasets, comparing its performance against several state-of-the-art methods."
> - **範例**：
>   - "In this section, we present a comprehensive evaluation of our SLT model's performance on the PHOENIX14T and CSL-Daily datasets. To ensure a fair comparison with established benchmarks, our evaluation focuses exclusively on specialized architectures trained on the official benchmark data."

#### 5.4.1 Comparison with Baseline Methods
- Baseline 方法列表
- 主要比較結果表格
- 與 SOTA 方法的比較分析

> ✍️ **【寫作風格與語氣】**
> - **必須包含**：結果表格 + 文字分析
> - **分析寫法**：指出自己方法的優勢、與最佳方法的差距、可能原因

#### 5.4.2 Results on CMU-MOSI
- CMU-MOSI 詳細結果
- Aligned vs Unaligned 結果比較（如適用）
- 結果分析與討論

> ✍️ **【寫作風格與語氣】**
> - **開頭**："First, we evaluate the performance of the model on the CMU-MOSI dataset. The results of all models are presented in Table X."
> - **分析**：指出表現最好的指標、與 baseline 的比較

#### 5.4.3 Results on CMU-MOSEI
- CMU-MOSEI 詳細結果
- Aligned vs Unaligned 結果比較（如適用）
- 結果分析與討論

---

### 5.5 Ablation Studies

> 📏 **【建議篇幅】** 約 0.8-1.5 頁
>
> | 項目 | 建議 |
> |------|------|
> | 頁數 | 0.8-1.5 頁 |
> | 表格 | ✅ 1-2 個（消融實驗表）|
> | 圖表 | ✅ 1-2 個（視覺化分析）|
>
> **核心內容（僅包含）：**
> - 模組有效性驗證（w/o 實驗）
> - 模態貢獻度分析
> - MoE 機制分析
>
> **Subsection 篇幅分配：**
> | Subsection | 頁數 |
> |------------|------|
> | 5.5.1 Effect of Modules | 0.3-0.5 頁 |
> | 5.5.2 Effect of Modalities | 0.3-0.5 頁 |
> | 5.5.3 MoE Analysis | 0.3-0.5 頁 |

**需要撰寫的內容：**

> ✍️ **【寫作風格與語氣】**
> - **語氣**：驗證性、分析性
> - **開頭句式參考**：
>   - "In this section, we conduct an ablation study to verify the effectiveness of the core components in our proposed method."
>   - "To understand and validate the contribution of each key component in our proposed architecture, we conducted a series of additive ablation studies on the XXX dataset."
>   - "In this section, we explore the impacts of each component in our proposed method through ablation studies conducted on the XXX dataset."
> - **範例**：
>   - "To understand and validate the contribution of each key component in our proposed architecture, we conducted a series of additive ablation studies on the PHOENIX14T dataset. We began with a baseline model and progressively integrated our proposed modules, observing the incremental impact on translation performance."

#### 5.5.1 Effect of Proposed Modules
- 你所設計的各個模組功能是否使用的比較
- 建議格式：逐一移除/加入模組，觀察性能變化
- 例如：
  - w/o Module A
  - w/o Module B
  - w/o Module C
  - Full Model (所有模組)

> ✍️ **【寫作風格與語氣】**
> - **表格格式**：使用 ✓ 和 - 表示模組是否啟用
> - **分析寫法**："The results demonstrate that..." "We observe that removing XXX leads to a X% drop in..."

#### 5.5.2 Effect of Different Modalities
- 單模態 vs 多模態的比較
- 各模態的貢獻度分析
- 例如：T (Text only), A (Audio only), V (Video only), T+A, T+V, A+V, T+A+V

> 📊 **【Figure 需求 - 建議】Modality Contribution Comparison**
> - 類型：長條圖 (Bar Chart)
> - 內容：不同模態組合的性能比較
> - X軸：模態組合 (T, A, V, T+A, T+V, A+V, T+A+V)
> - Y軸：Accuracy 或 F1 Score
> - 工具：Python matplotlib 繪製 或 Excel 製圖

#### 5.5.3 Analysis of MoE-based Representation Learning and Fusion Routing
- 專家設計：每個模態對應一個專家 (Text Expert, Audio Expert, Video Expert)
- 表徵學習分析：
  - 各專家學到的模態表徵特性
  - 專家表徵的可視化分析（如 t-SNE）
- Soft Routing 機制分析：
  - Router 權重分佈視覺化
  - 不同樣本的路由權重分析
  - 各專家對融合表徵的貢獻度
- 可選分析：
  - Soft Routing vs Hard Routing 比較
  - 不同情緒類別下的路由模式差異
  - 融合表徵相較於單一專家表徵的優勢

> 📊 **【Figure 需求 - 建議】Soft Routing Weight Distribution**
> - 類型：視覺化分析圖
> - 內容選項（擇一或多）：
>   - (a) 堆疊長條圖：不同情緒類別下各專家的平均權重分配
>   - (b) 熱力圖 (Heatmap)：樣本 vs 專家權重
>   - (c) 箱型圖 (Box Plot)：各專家權重的分佈情況
> - 目的：展示模型學到的 routing 模式是否有意義
> - 工具：Python matplotlib/seaborn 繪製

---

## Chapter 5 寫作風格總結

### 語氣特徵
| 章節 | 語氣特徵 | 常用動詞 |
|------|----------|----------|
| 5.1 Datasets | 說明性、客觀 | is conducted on, is evaluated on |
| 5.2 Evaluation Metrics | 定義性、精確 | we employ, is defined as, quantifies |
| 5.3 Experimental Setup | 簡潔、列表式 | is conducted on, is set to, we use |
| 5.4 Results | 比較性、分析性 | we present, comparing with, outperforms |
| 5.5 Ablation Studies | 驗證性、實驗性 | we conduct, to verify, to validate |

### 常用轉折與連接詞
- **承接**：Furthermore, Moreover, Additionally, In addition
- **對比**：However, In contrast, On the other hand, While
- **因果**：Therefore, Thus, Consequently, As a result
- **舉例**：For example, For instance, Specifically
- **總結**：In summary, Overall, In conclusion

---

## 附錄：學長姐 Chapter 5 結構參考

| 學長姐 | 結構特點 |
|--------|----------|
| 張世何 | 5.1 Environment + 5.2 Dataset + 5.3 Metrics + 5.4 Training + 5.5 Results + 5.6 Ablation |
| 徐偉哲 | 5.1 Datasets + 5.2 Config + 5.3 Metrics + 5.4 Performance |
| 林冠斌 | 5.1~5.4 (標準) + 5.5 Results (含子章節) + 5.6 Ablation |
| 江子青 | 5.1~5.4 (標準) + 5.5 Ablation |
| 王俊顏 | 5.1 Datasets (含子章節) + 5.2 Setup (含子章節) + 5.3~5.5 |
| 許顥蓉 | 5.1~5.3 (標準) + 5.4 Results (按資料集分子章節) + 5.5 Ablation |

---

## 建議撰寫順序

> 📊 **根據學長姐論文分析結果**
>
> Chapter 5 的撰寫順序相對固定，主要分為「實驗前可寫」與「實驗後才寫」兩階段。

### 推薦撰寫順序

**Phase 1：實驗設定（可與 Chapter 4 平行撰寫）**
1. **5.1 Datasets** - CMU-MOSI, CMU-MOSEI 介紹（資料集資訊固定）
2. **5.2 Evaluation Metrics** - 評估指標定義（Acc-2, Acc-7, F1, MAE, Corr）
3. **5.3.1 Experimental Environment** - 硬體環境（已確定）
4. **5.3.2 Implementation Details** - 超參數設定（Optuna 搜索完成後填入）

**Phase 2：結果分析（跑完實驗後）**
5. **5.4.1 Comparison with Baseline** - 與 SOTA 方法比較
6. **5.4.2 Results on CMU-MOSI** - MOSI 結果分析
7. **5.4.3 Results on CMU-MOSEI** - MOSEI 結果分析

**Phase 3：消融實驗（最後撰寫）**
8. **5.5.1 Effect of Proposed Modules** - 模組有效性驗證
9. **5.5.2 Effect of Different Modalities** - 模態貢獻度分析
10. **5.5.3 Analysis of MoE** - MoE 機制深入分析

### 學長姐 Chapter 5 撰寫特點

| 論文 | 特點 | 可參考之處 |
|------|------|-----------|
| 張世何 | 5.4 Training Details 獨立成節 | 訓練細節較多時可參考 |
| 徐偉哲 | 結構精簡，5.4 直接呈現結果 | 實驗較少時可參考 |
| 林冠斌 | 5.5 Results 含多個子章節 | 多資料集時可參考 |
| 許顥蓉 | 5.4 按資料集分子章節 | 強調各資料集差異時可參考 |

### 撰寫時機建議

| 章節 | 何時可以寫 | 依賴條件 |
|------|-----------|----------|
| 5.1 Datasets | ✅ 立即可寫 | 無 |
| 5.2 Metrics | ✅ 立即可寫 | 無 |
| 5.3.1 Environment | ✅ 立即可寫 | 無 |
| 5.3.2 Implementation | ⏳ Optuna 完成後 | 超參數確定 |
| 5.4 Results | ⏳ 實驗完成後 | 有實驗數據 |
| 5.5 Ablation | ⏳ 主實驗完成後 | 有消融實驗數據 |

---

## 資料檔案清單

| 資料集 | 格式 | 檔案名稱 |
|--------|------|----------|
| CMU-MOSI | Aligned | `MOSI_aligned.pkl` |
| CMU-MOSI | Unaligned | `MOSI_unaligned.pkl` |
| CMU-MOSEI | Aligned | `MOSEI_aligned.pkl` |
| CMU-MOSEI | Unaligned | `MOSEI_unaligned.pkl` |

> 特徵來源：MMSA 套件 (https://github.com/thuiar/MMSA)

---

## Chapter 5 圖片清單

| 編號 | 章節 | 圖片名稱 | 類型 | 優先度 | 工具 |
|------|------|----------|------|--------|------|
| Fig.3 | 5.5.2 | Modality Contribution Comparison | 長條圖 | 建議 | Python/Excel |
| Fig.4 | 5.5.3 | Soft Routing Weight Distribution | 視覺化分析圖 | 建議 | Python |

### 圖片繪製建議

**Python 繪製技巧（實驗結果圖）：**
- 使用 matplotlib + seaborn
- 設定適當的 figure size (建議 8x6 或 10x6)
- 使用清晰的標籤和圖例
- 輸出高解析度 PNG 或 PDF

---

*此模板根據實驗室 6 篇學長姐論文格式整理，適用於「多模態情緒分析結合 MoE 架構」研究主題。*
