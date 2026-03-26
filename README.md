# AML / 詐騙偵測專案（BitoPro）

從 BitoPro API 抓取交易資料，進行特徵工程後，使用 XGBoost 訓練人頭戶偵測模型；並提供 FastAPI 服務層供線上即時推論。

主流程：**feature_engineering.py → model_xgboost.py**

---

## 專案結構

```
/
├── feature_engineering.py   # Step 1：API 抓取 → 清洗 → 特徵工程 → 輸出 CSV
├── model_xgboost.py         # Step 2：讀取特徵 CSV → ablation → 調參 → 訓練 → 推論
├── model_Rf.py              # 實驗用：Random Forest 對照模型
├── model_LightGBM.py        # 實驗用：LightGBM 對照模型
├── requirements.txt
│
├── train_feature.csv        # 訓練集特徵（feature_engineering.py 輸出）
├── test_feature.csv         # 測試集特徵（feature_engineering.py 輸出）
├── feature_full.csv         # 全量用戶特徵（含 IsolationForest 分數）
│
├── output_xgb_v2/           # model_xgboost.py 輸出目錄
│   ├── compare_modes.csv
│   ├── full/
│   ├── no_leak/
│   └── safe/
│       ├── feature_importance.csv
│       ├── best_params.csv
│       ├── metrics.csv
│       ├── threshold_analysis.csv
│       ├── valid_detail.csv
│       └── submission.csv
│
└── app/                     # FastAPI 服務層（線上推論）
    ├── main.py
    ├── config.py
    ├── models/              # Pydantic request/response schemas
    └── routers/             # predict / explain / drift / audit / model
        services/            # 業務邏輯：ModelLoader、XGBPredictor、SHAPExplainer 等
```

---

## 使用方式

### ML Pipeline

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

輸出至 `output_xgb_v2/`，包含三種 ablation mode 的評估結果與預測檔。

### FastAPI 服務

```bash
uvicorn app.main:app --reload
```

啟動後可至 `http://localhost:8000/docs` 查看 Swagger UI。

主要設定透過環境變數或 `.env` 檔控制（參見 `app/config.py`）：

| 變數 | 預設值 | 說明 |
|---|---|---|
| `MODEL_S3_URI` | `s3://aml-models/model_registry/latest` | 模型 artifact 路徑 |
| `DATABASE_URL` | `postgresql://...@localhost:5432/aml` | Audit log 資料庫 |
| `DEFAULT_MODE` | `safe` | 預設特徵版本 |
| `PSI_WARNING_THRESHOLD` | `0.1` | Drift 警告門檻 |
| `PSI_CRITICAL_THRESHOLD` | `0.2` | Drift 嚴重門檻 |

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

---

## 環境需求

```bash
pip install pandas numpy requests scikit-learn xgboost matplotlib seaborn
pip install optuna shap      # 選用
pip install fastapi uvicorn pydantic-settings  # FastAPI 服務層
```

或直接：

```bash
pip install -r requirements.txt
```
