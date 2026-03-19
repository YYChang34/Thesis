# Research Log — DViN Thesis Fork (v3)

> **給新 session 的說明**：這個檔案延續 `RESEARCH_LOG_v2.md`，記錄 v3 階段的修改。開新 session 時請先讀 v2 再讀本檔案。

---

## 修改紀錄

### [2026-03-19] Gradient Accumulation 支援 + 雙 GTX-5090 訓練設定

**動機**：準備使用兩台 GTX-5090 進行訓練，加入 gradient accumulation 以在 VRAM 受限下模擬更大的有效 batch size，提升訓練穩定性。

**修改檔案**：

| 檔案 | 修改內容 |
|------|---------|
| `train.py` | 訓練迴圈加入 gradient accumulation 邏輯 |
| `config/refcoco_v3.yaml` | 新增 `GRAD_ACCUM_STEPS`，GPU 改為 `[0, 1]`，開啟 AMP |
| `config/refcoco.yaml` | 新增 `GRAD_ACCUM_STEPS: 1`（向後相容） |

**`train.py` 改動細節**：

1. **Loss 縮放**：每個 micro-batch 的 loss 除以 `accum_steps`，確保累積後的梯度量級與不使用 accumulation 時一致
2. **梯度累積**：每個 micro-batch 都做 backward，但只在每 `accum_steps` 個 batch（或 epoch 最後一個 batch）才執行：
   - Gradient clipping（`nn.utils.clip_grad_norm_`）
   - Optimizer step
   - Scheduler step
   - EMA update
   - `optimizer.zero_grad()`
3. **Scheduler 調整**：`get_lr_scheduler()` 的 `n_iter_per_epoch` 改為 `ceil(len(train_loader) / accum_steps)`，確保 warmup 和 cosine annealing 的 step 數正確
4. **AMP 修正**：AMP 路徑中，在 `scalar.step()` 之前先呼叫 `scalar.unscale_()` 再做 gradient clipping，修正原始程式碼中 clip 在 scale 之後的順序問題
5. **移除效能瓶頸**：移除每個 batch 的 `gc.collect()` + `torch.cuda.empty_cache()`（這會嚴重拖慢訓練速度）

**Config 新增參數**：

```yaml
GRAD_ACCUM_STEPS: 2   # Gradient accumulation steps
```

**有效 Batch Size 計算**：

```
有效 batch size = BATCH_SIZE × GRAD_ACCUM_STEPS
               = 16 × 2 = 32

每 GPU 實際 micro-batch = BATCH_SIZE / NUM_GPUs = 16 / 2 = 8
```

**向後相容**：所有 `GRAD_ACCUM_STEPS` 讀取均使用 `getattr(__C, 'GRAD_ACCUM_STEPS', 1)`，舊 config 不含此參數時預設為 1，行為等同原始程式碼。

---

## 訓練環境

| 項目 | 規格 |
|------|------|
| GPU | 2 × GTX-5090 |
| 訓練模式 | DDP + Gradient Accumulation |
| AMP | 開啟（建議用於 5090） |
| Effective Batch Size | 32（可透過 `GRAD_ACCUM_STEPS` 調整） |
