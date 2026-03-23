# AML / 詐騙偵測專案（BitoPro）

從 BitoPro API 抓取交易資料，進行特徵工程後，使用 XGBoost 訓練人頭戶偵測模型。

主流程：**feature_engineering.py → model_xgboost.py**

---

## 檔案說明

- `feature_engineering.py`：從 BitoPro API 分頁抓取資料 → 清洗 → 特徵工程 → 輸出特徵 CSV
- `model_xgboost.py`：讀取特徵 CSV，進行 ablation study（三種特徵版本）、Optuna 調參、訓練與推論

---

## 使用方式

**Step 1：產生特徵資料集**

```bash
python feature_engineering.py
```

輸出：
- `train_feature.csv`：訓練集（含 `status` 標籤）
- `test_feature.csv`：測試集
- `feature_full.csv`：全量用戶特徵表（含 IsolationForest 異常分數）

**Step 2：訓練與推論**

```bash
python model_xgboost.py
```

預設讀取 `train_feature_v2.csv` / `test_feature_v2.csv`，可在 `main()` 修改路徑與參數。

---

## 特徵工程說明（feature_engineering.py）

資料來源（透過 BitoPro API 抓取）：

| 資料表 | 說明 |
|---|---|
| `user_info` | 用戶基本資料與 KYC 時間 |
| `twd_transfer` | 台幣出入金紀錄 |
| `crypto_transfer` | 虛擬貨幣轉帳紀錄 |
| `usdt_twd_trading` | USDT/TWD 撮合交易紀錄 |
| `usdt_swap` | 一鍵買賣（Swap）紀錄 |
| `train_label` / `predict_label` | 訓練/預測標籤 |

特徵類型：

- **KYC 時間差**：各 KYC 階段完成時間差（人頭戶通常極短）
- **基礎聚合特徵**：各渠道的交易筆數、金額統計、夜間/週末比例、IP 多樣性、時間熵
- **鏈別風險特徵**：TRC20 / BSC 使用比例、地址重用率、流向不平衡
- **網路特徵**：內轉網路的 in/out degree、資金流向不對稱
- **錢包風險**：高風險共用地址比例、地址重用率
- **IP 特徵**：跨渠道共用 IP 比例、IP 跳躍率
- **資金流動**：快進快出旗標（台幣入金 → 24 小時內虛幣提領）
- **行為序列**：可疑操作序列比例（twd_in → 1 小時內 → crypto_out）
- **金額異常**：整數金額比例、變異係數、偏態
- **交叉特徵**：深夜 × 金額、KYC 速度 × 交易密度、TRC20 × 夜間等複合訊號
- **IsolationForest 異常分數**：用 train+test 合併後 fit，確保分布一致

---

## 模型說明（model_xgboost.py）

### Ablation Study（三種特徵版本）

| Mode | 說明 |
|---|---|
| `full` | 全部可用數值欄位（分數上限，可能含 leakage） |
| `no_leak` | 移除高風險可疑欄位（建議主要參考版本） |
| `safe` | 移除高風險欄位 + 人口學欄位（最接近真實部署情境） |

若 `full` 與 `safe` 分數差距 > 0.05，建議提交 `safe` 版本，避免線上線下落差。

### 切分策略

優先使用 **time-based split**（前 80% 訓練、後 20% 驗證），模擬「用歷史預測未來」的場景；找不到合適時間欄位時退回 random stratified split。

### 超參數調優

預設使用 **Optuna**（直接優化驗證集 F1），未安裝時自動退回手動參數。

```bash
pip install optuna  # 選用
pip install shap    # 選用，用於 SHAP summary plot
```

---

## 輸出結構

```
output_xgb_v2/
├── compare_modes.csv          # 三種 mode 的指標比較表
├── full/
│   ├── feature_importance.csv # 特徵重要性與排名
│   ├── best_params.csv        # 最佳模型參數
│   ├── metrics.csv            # F1 / AUC / Precision / Recall / PR-AUC
│   ├── threshold_analysis.csv # 各 threshold 的 precision/recall/f1
│   ├── valid_detail.csv       # 驗證集預測明細
│   └── submission.csv         # 測試集預測結果
├── no_leak/                   # 同上
└── safe/                      # 同上
```

---

## 環境需求

```bash
pip install pandas numpy requests scikit-learn xgboost matplotlib seaborn
pip install optuna shap  # 選用
```
