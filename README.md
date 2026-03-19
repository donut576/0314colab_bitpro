# 0314colab_bitpro

AML / 詐騙偵測專案：從 BitoPro API 抓資料 → 特徵工程 → 用三個模型訓練/推論產出預測。

主流程：**colab_1.py → model（LightGBM / XGBoost / RandomForest）**

---

## Structure

- `colab_1.py`：抓 API + 清洗 + 特徵工程 → 輸出 `train_df.csv` / `test_df.csv`（以及 `feature_df.csv`）
- `model_lightgbm.py`：讀 `train_df.csv` 訓練、對 `test_df.csv` 推論
- `model_xgboost.py`：同上（XGBoost）
- `model_random_forest.py`：同上（RandomForest）
- `run_all_models.py`：一鍵跑三個模型（如有整合輸出也在此）

---

## Usage（操作順序）

1. 產生資料集  
   - 執行：`python colab_1.py`  
   - 產物：`train_df.csv`, `test_df.csv`（+ `feature_df.csv`）

2. 訓練/推論（擇一或全跑）  
   - 單跑：  
     - `python model_lightgbm.py`  
     - `python model_xgboost.py`  
     - `python model_random_forest.py`  
   - 全跑：`python run_all_models.py`

---

## Outputs

- 主要中間檔：`train_df.csv`, `test_df.csv`, `feature_df.csv`
- 模型輸出：依各 `model_*.py` / `run_all_models.py` 設定（通常為 submission / prediction CSV）


### Per-model CSV outputs (auto-created folders)

三個模型（LightGBM / XGBoost / RandomForest）在訓練完成後，會自動建立輸出資料夾：

- `output/LightGBM/`
- `output/XGBoost/`
- `output/RandomForest/`

並在各自資料夾輸出以下 CSV（檔名固定）：

1. `feature_importance.csv`  
   - 包含 `feature importance + rank`（特徵重要性與排名），通常為由高到低排序。

2. `best_params.csv`  
   - 紀錄該模型的最佳參數（Best Parameters）。

3. `metrics.csv`  
   - 模型評估指標（Validation metrics），包含：
     - `F1`
     - `AUC`
     - `Precision`
     - `Recall`
     - `Best Threshold`

> 註：各模型也可能仍會輸出 submission / prediction 類 CSV（依各 `model_*.py` 實作為準）。