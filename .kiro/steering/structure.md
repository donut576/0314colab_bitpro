# Project Structure

## Overview

Flat structure — all source files live at the root level.

```
/
├── feature_engineering.py   # Step 1：API 抓取 → 清洗 → 特徵工程 → 輸出 CSV
├── model_xgboost.py         # Step 2：讀取特徵 CSV → ablation → 調參 → 訓練 → 推論
├── requirements.txt         # Python 依賴
├── README.md
│
├── train_feature.csv        # 訓練集特徵（feature_engineering.py 輸出）
├── test_feature.csv         # 測試集特徵（feature_engineering.py 輸出）
├── feature_full.csv         # 全量用戶特徵（含 IsolationForest 分數）
│
└── output_xgb_v2/           # model_xgboost.py 輸出目錄
    ├── compare_modes.csv    # 三種 mode 指標比較
    ├── full/                # 全特徵版本結果
    ├── no_leak/             # 移除高風險欄位版本結果
    └── safe/                # 最保守版本結果
        ├── feature_importance.csv
        ├── best_params.csv
        ├── metrics.csv
        ├── threshold_analysis.csv
        ├── valid_detail.csv
        └── submission.csv
```

## feature_engineering.py 內部結構

| 區塊 | 說明 |
|------|------|
| 通用函式 | `fetch_table_paginated`、`safe_divide`、`add_time_cols`、`calc_gap_features` 等 |
| 資料清洗 | `prepare_user_info`、`prepare_twd_transfer`、`prepare_crypto_transfer`、`prepare_trade`、`prepare_swap` |
| 基礎聚合特徵 | `build_twd_features`、`build_crypto_features`、`build_trade_features`、`build_swap_features` |
| 進階特徵 | `build_network_features`、IP 特徵、快進快出旗標、行為序列、交叉特徵、IsolationForest |

## model_xgboost.py 內部結構

| 區塊 | 說明 |
|------|------|
| 全域設定 | `TARGET_COL`、`ID_COL`、leakage 關鍵字清單、時間欄位優先順序 |
| Threshold 搜尋 | `find_best_threshold` — 掃描 [0.01, 0.95] 找最佳 F1 切點 |
| 超參數調優 | `tune_xgb_with_optuna` / `_default_xgb_params` |
| 資料準備 | `prepare_xy(mode)` — full / no_leak / safe 三種欄位過濾策略 |
| 切分 | `split_data` — time-based 優先，fallback random stratified |
| 實驗 | `run_experiment` — 單一 mode 完整流程 |
| 主程式 | `main` — 跑三種 mode，輸出 compare_modes.csv |

## Ablation Study Modes

| Mode | 說明 | 用途 |
|------|------|------|
| `full` | 全部數值欄位 | 分數上限參考，可能含 leakage |
| `no_leak` | 移除高風險欄位 | 主要參考版本 |
| `safe` | 移除高風險 + 人口學欄位 | 最接近真實部署情境 |

若 `full` 與 `safe` F1 差距 > 0.05，提交 `safe` 版本。
