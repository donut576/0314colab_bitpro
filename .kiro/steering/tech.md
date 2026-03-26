# Tech Stack

## Language
- Python 3.x

## Core Libraries
- `pandas` >= 2.0.0 — 資料處理與特徵工程
- `numpy` >= 1.24.0 — 數值運算
- `requests` >= 2.28.0 — BitoPro API 分頁抓取
- `scikit-learn` >= 1.3.0 — IsolationForest、train_test_split、評估指標
- `xgboost` >= 2.0.0 — 主要分類模型（使用 `hist` tree method）
- `matplotlib` >= 3.7.0 — 視覺化
- `seaborn` >= 0.12.0 — 視覺化

## Optional Libraries
- `optuna` >= 3.0.0 — 超參數自動調優（優化驗證集 F1）
- `shap` >= 0.43.0 — 特徵重要性解釋（SHAP summary plot）

## Common Commands

```bash
# 安裝依賴
pip install pandas numpy requests scikit-learn xgboost matplotlib seaborn
pip install optuna shap  # 選用

# Step 1：產生特徵資料集
python feature_engineering.py
# 輸出：train_feature.csv、test_feature.csv、feature_full.csv

# Step 2：訓練與推論
python model_xgboost.py
# 輸出：output_xgb_v2/ 下的各 mode 結果
```

## Key Conventions
- 金額欄位縮放：`ori_samount * 1e-8` = 實際金額（台幣或虛幣）
- 缺值填補：一律用 `0`（不用 `-1`，避免樹模型誤學）
- `inf` 值：訓練前統一替換為 `nan` 再填 `0`
- 安全除法：使用 `safe_divide(a, b)` 避免除以零（分母加 `1e-9`）
- 時間切分：優先 time-based split（前 80% 訓練），找不到時間欄位才退回 random stratified split
- Optuna 不可用時自動退回手動參數（`_default_xgb_params`）
