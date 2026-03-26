# Product

AML / 詐騙偵測專案（BitoPro）

從 BitoPro API 抓取交易資料，進行特徵工程後，使用 XGBoost 訓練人頭戶偵測模型。

## 目標

識別 BitoPro 交易所中的人頭戶（mule accounts）與黑名單用戶。

## 核心流程

`feature_engineering.py` → `model_xgboost.py`

1. 從 BitoPro API 分頁抓取多張資料表
2. 清洗資料並進行特徵工程（KYC 時間差、行為聚合、鏈別風險、網路特徵等）
3. 輸出特徵 CSV，供 XGBoost 模型訓練與推論
4. 執行 ablation study（full / no_leak / safe 三種特徵版本），評估 data leakage 影響
5. 輸出預測結果與模型評估報告

## 資料來源

透過 BitoPro API 抓取：`user_info`、`twd_transfer`、`crypto_transfer`、`usdt_twd_trading`、`usdt_swap`、`train_label`、`predict_label`
