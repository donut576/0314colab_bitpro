# Bito Guard AML Fraud Detection

Bito Guard 是一個 AML / 詐騙風險偵測專案，包含資料特徵工程、模型訓練、FastAPI 後端服務，以及 React Dashboard。

Live Demo: [https://aml-frontend-mu.vercel.app](https://aml-frontend-mu.vercel.app)

主要流程：

```text
feature_engineering.py -> run_all_models.py -> API service -> aml-frontend
```

## 專案結構

```text
.
├── app/                    # FastAPI 進階服務層
│   ├── main.py             # API 入口
│   ├── routers/            # predict、explain、alerts、cases、monitoring 等路由
│   ├── services/           # 模型、告警、案件、監控、特徵倉儲等服務邏輯
│   ├── models/             # Pydantic schemas
│   └── migrations/         # PostgreSQL schema
├── aml-frontend/           # React + Vite + Recharts Dashboard
│   ├── src/
│   ├── server.py           # 輕量 Dashboard API，讀取 output_results/
│   ├── vite.config.js
│   └── vercel.json         # Vercel 靜態站部署設定
├── tests/                  # API 與 property-based tests
├── deploy/                 # AWS / EC2 部署腳本
├── sagemaker/              # SageMaker 環境設定
├── feature_engineering.py  # Step 1：產生訓練與測試特徵
├── run_all_models.py       # Step 2：訓練 XGBoost / LightGBM / Random Forest
├── model_xgboost.py        # 單模型實驗腳本
├── model_LightGBM.py       # 單模型實驗腳本
├── model_Rf.py             # 單模型實驗腳本
├── model_ensemble.py       # 實驗用 ensemble 腳本
├── model_stack.py          # 實驗用 stacking 腳本
├── explain_fraud.py        # 離線解釋結果產生腳本
├── render.yaml             # Render Blueprint
├── Dockerfile              # 後端 API container
└── requirements.txt        # Python dependencies
```

## 輸出資料

以下檔案與資料夾是程式執行後產生，預設不進 Git：

| 路徑 | 來源 | 用途 |
|---|---|---|
| `train_feature.csv` | `feature_engineering.py` | 訓練集特徵 |
| `test_feature.csv` | `feature_engineering.py` | 測試集特徵 |
| `feature_full.csv` | `feature_engineering.py` | 全量用戶特徵 |
| `output_results/` | `run_all_models.py` | Dashboard 主要讀取的模型結果 |
| `output_xgb_v2/` | `model_xgboost.py` | XGBoost 單模型輸出 |

## 本機開發

### 1. 安裝 Python 依賴

```bash
pip install -r requirements.txt
```

### 2. 產生特徵

```bash
python feature_engineering.py
```

### 3. 訓練模型

```bash
python run_all_models.py
```

`run_all_models.py` 會輸出到 `output_results/{xgb,lgb,rf}/{full,no_leak,safe}/`，包含 metrics、feature importance、threshold analysis、test scores 與 SHAP JSON。

### 4. 啟動 Dashboard API

```bash
cd aml-frontend
python -m uvicorn server:app --port 8000 --reload
```

### 5. 啟動前端

```bash
cd aml-frontend
npm install
npm run dev
```

開啟 `http://localhost:3000`。

## 進階 FastAPI 服務

```bash
uvicorn app.main:app --reload
```

啟動後可開啟：

| 頁面 | URL |
|---|---|
| Health check | `http://localhost:8000/health` |
| Swagger docs | `http://localhost:8000/docs` |
| Prometheus metrics | `http://localhost:8000/metrics` |

主要環境變數：

| 變數 | 預設值 | 說明 |
|---|---|---|
| `MODEL_S3_URI` | `s3://aml-models/model_registry/latest` | 模型 artifact 路徑 |
| `DATABASE_URL` | `postgresql://...@localhost:5432/aml` | Audit / case / feature store 資料庫 |
| `REDIS_URL` | `redis://localhost:6379/0` | Feature store / stream 相關服務 |
| `DEFAULT_MODE` | `safe` | 預設特徵版本 |
| `PSI_WARNING_THRESHOLD` | `0.1` | Drift 警告門檻 |
| `PSI_CRITICAL_THRESHOLD` | `0.2` | Drift 嚴重門檻 |

## 模型說明

### 支援模型

| 模型 | 腳本 | 說明 |
|---|---|---|
| XGBoost | `model_xgboost.py` / `run_all_models.py` | 主要模型 |
| LightGBM | `model_LightGBM.py` / `run_all_models.py` | 對照模型 |
| Random Forest | `model_Rf.py` / `run_all_models.py` | 對照模型 |

### Ablation modes

| Mode | 說明 |
|---|---|
| `full` | 全部可用數值欄位，分數上限參考，可能含 leakage |
| `no_leak` | 移除高風險可疑欄位 |
| `safe` | 移除高風險欄位與人口學欄位，較接近部署情境 |

若 `full` 與 `safe` 分數差距明顯，建議以上線情境優先參考 `safe`。

## 前端部署到 Vercel

若後端 API 已部署在 Render 或其他服務，可以只部署 `aml-frontend`。

1. 將 repo push 到 GitHub。
2. 到 Vercel 選 **Add New -> Project**。
3. 選取此 repo。
4. **Root Directory** 選 `aml-frontend`。
5. 到 **Settings -> Environment Variables** 新增：

| 變數 | 說明 |
|---|---|
| `VITE_API_BASE_URL` | 後端 API 公開網址，例如 `https://aml-api.onrender.com` |

不要在 `VITE_API_BASE_URL` 最後加 `/`。

Vercel 會讀取 `aml-frontend/vercel.json`：

| 設定 | 值 |
|---|---|
| Install Command | `npm ci` |
| Build Command | `npm run build` |
| Output Directory | `dist` |

設定環境變數後，請重新部署一次，讓前端 build 時讀到 API URL。

## 部署到 Render

本專案提供 `render.yaml`，可建立：

| 服務 | 說明 |
|---|---|
| `aml-api` | FastAPI Web Service |
| `aml-frontend` | Vite Static Site |
| `aml-redis` | Redis |
| `aml-postgres` | PostgreSQL |

步驟：

1. 將 repo push 到 GitHub。
2. 到 Render 選 **New + -> Blueprint**。
3. 選取此 repo，Render 會讀取 `render.yaml`。
4. 在 Render Console 補上必要環境變數。

必要環境變數：

| 服務 | 變數 | 說明 |
|---|---|---|
| `aml-api` | `MODEL_S3_URI` | 模型 artifact 路徑 |
| `aml-frontend` | `VITE_API_BASE_URL` | `aml-api` 的公開 URL |

`DATABASE_URL` 與 `REDIS_URL` 會由 Blueprint 自動綁定。

## 測試

```bash
pytest
```

測試快取如 `.hypothesis/`、`.pytest_cache/` 不需要提交到 Git。
