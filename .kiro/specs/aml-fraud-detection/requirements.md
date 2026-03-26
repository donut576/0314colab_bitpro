# AML Fraud Detection System — Requirements

## Overview

Build a production-grade AML (Anti-Money Laundering) fraud detection system for BitoPro exchange. The system identifies mule accounts and blacklisted users from transaction data, serves predictions via API, and provides explainable, auditable outputs for compliance officers.

---

## Stakeholders

- **Compliance Officers** — primary end users; need explainable predictions, audit trails, and exportable reports
- **Data / ML Engineers** — maintain feature pipeline and model retraining
- **Platform / Backend Engineers** — integrate model serving into existing infrastructure

---

## Requirements

### REQ-1: Feature Engineering Pipeline

**User Story:**
As a data engineer, I want an automated pipeline that fetches raw transaction data from the BitoPro API and outputs clean feature CSVs, so that the model always trains on up-to-date data.

**Acceptance Criteria:**
- [ ] Paginated fetch from all 7 API tables: `user_info`, `twd_transfer`, `crypto_transfer`, `usdt_twd_trading`, `usdt_swap`, `train_label`, `predict_label`
- [ ] Outputs `train_feature.csv`, `test_feature.csv`, `feature_full.csv`
- [ ] Feature set covers: KYC time deltas, behavioral aggregates, chain-level risk (TRC20/BSC), network graph features, IP features, fast-in-fast-out flags, cross features, IsolationForest anomaly score
- [ ] Pipeline is idempotent — re-running produces consistent output given the same source data
- [ ] Handles API pagination, timeouts, and empty responses gracefully

---

### REQ-2: XGBoost Model Training with Ablation Study

**User Story:**
As an ML engineer, I want to train an XGBoost model across three feature modes (full / no_leak / safe) and compare their metrics, so that I can detect data leakage and choose the safest model for deployment.

**Acceptance Criteria:**
- [ ] Supports three modes: `full`, `no_leak`, `safe`
- [ ] Time-based split (front 80% train, back 20% validate) with fallback to random stratified split
- [ ] Optuna hyperparameter tuning (optimizing validation F1); graceful fallback to manual params if Optuna unavailable
- [ ] Outputs per mode: `metrics.csv`, `feature_importance.csv`, `threshold_analysis.csv`, `valid_detail.csv`, `submission.csv`, `best_params.csv`
- [ ] Outputs `compare_modes.csv` summarizing all three modes
- [ ] If `full` vs `safe` F1 gap > 0.05, system recommends submitting `safe` version

---

### REQ-3: Model Serving API

**User Story:**
As a backend engineer, I want a REST API that accepts user transaction data and returns fraud risk scores in real time, so that the system can flag suspicious accounts during live operations.

**Acceptance Criteria:**
- [ ] FastAPI service exposing at minimum:
  - `POST /predict` — accepts CSV upload or JSON payload of user features; returns `user_id`, `risk_score`, `predicted_label`, `threshold_used`
  - `GET /health` — liveness check
  - `GET /model/info` — returns current model version, mode, training date, feature count
- [ ] Supports all three modes (`full`, `no_leak`, `safe`) via request parameter; defaults to `safe`
- [ ] Returns predictions within 2 seconds for batches up to 1,000 users
- [ ] Model artifact loaded at startup from **AWS S3** (bucket path configurable via env var `MODEL_S3_URI`)
- [ ] Supports model versioning via S3 object versioning or a dedicated `model_registry/` prefix
- [ ] API returns HTTP 422 with descriptive error if input schema is invalid

---

### REQ-4: Explainability (SHAP)

**User Story:**
As a compliance officer, I want to see which features drove a specific user's fraud risk score, so that I can justify a flagging decision in an audit or regulatory review.

**Acceptance Criteria:**
- [ ] Per-user SHAP values available via `GET /explain/{user_id}` or included in `/predict` response (optional flag)
- [ ] Response includes top-N contributing features (default N=10) with feature name, value, and SHAP contribution direction (positive = increases risk)
- [ ] SHAP summary plot (global) available as a downloadable PNG via `GET /explain/summary`
- [ ] Explainability output is human-readable — feature names must be descriptive, not raw column codes
- [ ] Falls back gracefully if `shap` library is not installed (returns 501 with clear message)

---

### REQ-5: Data Drift Detection

**User Story:**
As an ML engineer, I want the system to detect when incoming inference data has drifted significantly from the training distribution, so that I can trigger retraining before model performance degrades silently.

**Acceptance Criteria:**
- [ ] Drift detection runs on each inference batch (or on a scheduled basis)
- [ ] Monitors key features: at minimum the top-20 features by SHAP importance
- [ ] Uses **Population Stability Index (PSI)** to flag drift per feature
- [ ] `GET /drift/report` returns per-feature drift scores and an overall drift status (`ok` / `warning` / `critical`)
- [ ] Drift status `critical` triggers a warning in the API response and logs an alert
- [ ] Training distribution statistics (mean, std, percentiles) are saved alongside the model artifact at training time

---

### REQ-6: Compliance Dashboard (React Frontend)

**User Story:**
As a compliance officer, I want a multi-tab dashboard that shows model performance, feature importance, SHAP explanations, inference results, and fraud reports, so that I can monitor the system and act on flagged accounts without needing to run code.

**Acceptance Criteria:**
- [ ] Tabs: Model Metrics, Feature Importance, SHAP Explainability, Inference, Fraud Report
- [ ] Model Metrics tab: displays F1, AUC, Precision, Recall, PR-AUC per mode; supports mode switching (`full` / `no_leak` / `safe`)
- [ ] Feature Importance tab: bar chart of top-20 features; sortable table
- [ ] SHAP tab: global summary plot + per-user drill-down (enter `user_id` → see top contributing features)
- [ ] Inference tab: upload CSV → call `/predict` → display flagged users with risk scores
- [ ] Fraud Report tab: ranked list of flagged users; exportable as CSV
- [ ] All API calls have fallback mock data when backend is unavailable
- [ ] Real-time mode switching reflected across all tabs without page reload

---

### REQ-7: Audit Trail & Compliance Export

**User Story:**
As a compliance officer, I want every prediction and model decision to be logged with a timestamp and model version, so that I can produce a complete audit trail for regulators.

**Acceptance Criteria:**
- [ ] Every `/predict` call is logged to a **PostgreSQL table** (`prediction_logs`): timestamp, model version, mode, input hash, output (user_id + risk_score + label)
- [ ] Logs are append-only and stored in a structured format (e.g., JSONL or database table)
- [ ] `GET /audit/log` returns paginated prediction history filterable by date range and user_id
- [ ] Fraud Report CSV export includes: user_id, risk_score, predicted_label, threshold, model_version, prediction_timestamp
- [ ] Audit logs are retained for a minimum of 90 days

---

## Non-Functional Requirements

| Category | Requirement |
|---|---|
| Performance | `/predict` p95 latency < 2s for batch of 1,000 users |
| Reliability | API uptime > 99.5%; graceful degradation if model artifact missing |
| Security | No PII in logs; API authentication required in production |
| Maintainability | Model artifact versioned; rollback to previous version supported |
| Observability | Structured logging; drift alerts surfaced in dashboard |

---

## Out of Scope

- Real-time streaming inference (batch inference only)
- Multi-model ensemble beyond XGBoost
- Automated retraining pipeline (drift detection triggers alert only, not auto-retrain)
- User authentication / RBAC for the dashboard (noted as future work)
