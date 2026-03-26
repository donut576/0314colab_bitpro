# Implementation Plan: AML Fraud Detection System

## Overview

Extend the existing two-script pipeline (`feature_engineering.py` → `model_xgboost.py`) with a FastAPI serving layer, SHAP explainability, PSI drift detection, PostgreSQL audit logging, and a React compliance dashboard. New code lives under `app/` (FastAPI service), `frontend/` (React SPA), and `tests/` (all tests).

## Tasks

- [x] 1. Set up FastAPI application skeleton and configuration
  - Create `app/` directory structure: `main.py`, `config.py`, `routers/`, `services/`, `models/`
  - Implement `app/config.py` loading `MODEL_S3_URI`, `DATABASE_URL`, and other env vars via `pydantic-settings`
  - Wire FastAPI lifespan (startup/shutdown) in `app/main.py` with placeholder service initialization
  - _Requirements: REQ-3_

- [~] 2. Implement Pydantic schemas
  - [~] 2.1 Create `app/models/prediction.py` with `PredictRequest`, `PredictionResult`, `SHAPContribution`
    - Include `mode` defaulting to `"safe"`, `include_shap: bool = False`, `users: list[dict]`
    - _Requirements: REQ-3.2, REQ-3.6_
  - [~] 2.2 Create `app/models/explain.py`, `app/models/drift.py`, `app/models/audit.py`
    - `DriftReport`, `FeatureDriftResult`, `AuditQueryFilters`, `PredictionLogRecord` as per design
    - _Requirements: REQ-4, REQ-5, REQ-7_

- [~] 3. Implement ModelLoader service
  - [~] 3.1 Create `app/services/model_loader.py` implementing `ModelLoader` class
    - `load_from_s3(s3_uri)` downloads `model.ubj`, `metadata.json`, `training_stats.json`, `feature_importance.csv` via `boto3`
    - `get_model()`, `get_metadata()`, `get_training_stats()` return cached objects
    - Raise descriptive error (surfaced as 503) if S3 download fails at startup
    - _Requirements: REQ-3.4, REQ-3.5_
  - [ ]* 3.2 Write property test for ModelLoader metadata round-trip
    - **Property 7: Default mode is safe**
    - **Validates: Requirements REQ-3.2**

- [~] 4. Implement XGBPredictor service
  - [~] 4.1 Create `app/services/predictor.py` implementing `XGBPredictor` class
    - `predict_batch(features: pd.DataFrame)` aligns columns to training feature list, fills inf→nan→0, applies threshold, returns `list[PredictionResult]`
    - `predict_single(user_id, features)` delegates to `predict_batch`
    - Raise 400 if batch size > 1,000
    - _Requirements: REQ-3.1, REQ-3.3, REQ-3.6_
  - [ ]* 4.2 Write property test for invalid input returning 422/400
    - **Property 8: Invalid input returns 422**
    - **Validates: Requirements REQ-3.6**

- [~] 5. Implement SHAPExplainer service
  - [~] 5.1 Create `app/services/shap_explainer.py` implementing `SHAPExplainer` class
    - Initialize `shap.TreeExplainer` from loaded model at startup; catch `ImportError` and set `shap_available = False`
    - `explain_user(user_id, features, top_n=10)` returns `list[SHAPContribution]` sorted by `abs(shap_value)` descending
    - `get_global_summary_png(sample_df)` renders SHAP summary plot to bytes and returns PNG
    - Return HTTP 501 when `shap_available = False`
    - _Requirements: REQ-4.1, REQ-4.2, REQ-4.3, REQ-4.5_
  - [ ]* 5.2 Write property test for SHAP explanation completeness
    - **Property 9: SHAP explanation completeness**
    - **Validates: Requirements REQ-4.1, REQ-4.2**

- [~] 6. Implement DriftDetector service
  - [~] 6.1 Create `app/services/drift_detector.py` implementing `DriftDetector` class
    - `compute_psi(feature, current_values)` uses pre-computed bins from `training_stats.json`; returns float
    - PSI thresholds: `< 0.1` → `ok`, `0.1–0.2` → `warning`, `≥ 0.2` → `critical`
    - `compute_batch_drift(batch_df)` runs PSI on top-20 SHAP features; returns `DriftReport` with `overall_status` = worst feature status
    - _Requirements: REQ-5.1, REQ-5.2, REQ-5.3, REQ-5.4_
  - [ ]* 6.2 Write property test for PSI self-similarity
    - **Property 10: PSI self-similarity is zero**
    - **Validates: Requirements REQ-5.3**
  - [ ]* 6.3 Write property test for drift status reflecting worst feature
    - **Property 11: Drift status reflects worst feature**
    - **Validates: Requirements REQ-5.4, REQ-5.5**

- [~] 7. Checkpoint — core services complete
  - Ensure all tests pass, ask the user if questions arise.

- [~] 8. Implement AuditLogger service
  - [~] 8.1 Create `app/services/audit_logger.py` implementing `AuditLogger` class
    - `log_prediction(record: PredictionLogRecord)` async write via `asyncpg`; falls back to in-memory queue if DB unavailable
    - `query_logs(filters: AuditQueryFilters)` returns paginated records with SQL WHERE clauses for `user_id`, `date_from`, `date_to`
    - `export_csv(filters)` returns CSV bytes with columns: `user_id`, `risk_score`, `predicted_label`, `threshold_used`, `model_version`, `prediction_timestamp`
    - `retained_until` set to `created_at + timedelta(days=90)` on insert
    - _Requirements: REQ-7.1, REQ-7.2, REQ-7.3, REQ-7.4, REQ-7.5_
  - [~] 8.2 Create `prediction_logs` table migration SQL in `app/migrations/001_create_prediction_logs.sql`
    - Include all columns, indexes, and CHECK constraints from design
    - _Requirements: REQ-7.1_
  - [ ]* 8.3 Write property test for audit log round-trip and append-only invariant
    - **Property 12: Audit log round-trip and append-only invariant**
    - **Validates: Requirements REQ-7.1, REQ-7.2**
  - [ ]* 8.4 Write property test for audit log filter correctness
    - **Property 13: Audit log filter correctness**
    - **Validates: Requirements REQ-7.3**
  - [ ]* 8.5 Write property test for CSV column completeness
    - **Property 14: Audit CSV export column completeness**
    - **Validates: Requirements REQ-7.4**
  - [ ]* 8.6 Write property test for 90-day retention invariant
    - **Property 15: 90-day retention invariant**
    - **Validates: Requirements REQ-7.5**

- [~] 9. Implement FastAPI routers and wire services
  - [~] 9.1 Create `app/routers/predict.py` — `POST /predict`
    - Validate request via `PredictRequest`; call `XGBPredictor.predict_batch`; call `DriftDetector.compute_batch_drift`; call `AuditLogger.log_prediction` (async, non-blocking); optionally call `SHAPExplainer.explain_user` if `include_shap=True`
    - Include `drift_warning` field in response when `overall_status='critical'`
    - _Requirements: REQ-3.1, REQ-5.5, REQ-7.1_
  - [~] 9.2 Create `app/routers/explain.py` — `GET /explain/{user_id}` and `GET /explain/summary`
    - `/explain/{user_id}` fetches most recent log from `AuditLogger`, re-runs SHAP, returns `SHAPContribution` list
    - `/explain/summary` returns PNG bytes with `Content-Type: image/png`
    - Return 501 if SHAP unavailable; 404 if user not in logs
    - _Requirements: REQ-4.1, REQ-4.3, REQ-4.5_
  - [~] 9.3 Create `app/routers/drift.py` — `GET /drift/report`
    - Returns latest `DriftReport` (cached from last `/predict` call)
    - _Requirements: REQ-5.4_
  - [~] 9.4 Create `app/routers/audit.py` — `GET /audit/log`
    - Accepts query params `user_id`, `date_from`, `date_to`, `page`, `page_size`; delegates to `AuditLogger.query_logs`
    - `?export=csv` triggers `AuditLogger.export_csv` and returns file download
    - _Requirements: REQ-7.3, REQ-7.4_
  - [~] 9.5 Create `app/routers/model.py` — `GET /health` and `GET /model/info`
    - `/health` checks model loaded and DB connected; returns 200 or 503
    - `/model/info` returns metadata + top features from `ModelLoader`
    - _Requirements: REQ-3.1, REQ-3.5_

- [ ] 10. Write unit tests for API endpoints
  - [ ]* 10.1 Write unit tests in `tests/unit/test_api_endpoints.py`
    - Use FastAPI `TestClient`; mock `ModelLoader`, `XGBPredictor`, `SHAPExplainer`, `DriftDetector`, `AuditLogger`
    - Cover: `POST /predict` 200, 400 (batch > 1000), 422 (bad schema), 503 (model not loaded); `GET /health`; `GET /model/info`; `GET /explain/{user_id}` 200/404/501; `GET /drift/report`; `GET /audit/log` with filters; CSV export
    - _Requirements: REQ-3, REQ-4, REQ-5, REQ-7_

- [~] 11. Checkpoint — API layer complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Write property-based tests for pipeline (Hypothesis)
  - [ ]* 12.1 Create `tests/property/test_pbt_pipeline.py`
    - **Property 1: Pagination completeness** — mock API returning N pages, assert total rows = N × batch_size; `@given(st.integers(1,20), st.integers(100,1000))`
    - **Validates: Requirements REQ-1.1**
  - [ ]* 12.2 Add property test for feature set coverage to `tests/property/test_pbt_pipeline.py`
    - **Property 2: Feature set coverage** — generate random transaction DataFrames, run pipeline, assert columns from each category present
    - **Validates: Requirements REQ-1.3**
  - [ ]* 12.3 Add property test for pipeline idempotence to `tests/property/test_pbt_pipeline.py`
    - **Property 3: Pipeline idempotence** — run pipeline twice on same input, assert identical output
    - **Validates: Requirements REQ-1.4**
  - [ ]* 12.4 Add property test for leakage column exclusion to `tests/property/test_pbt_pipeline.py`
    - **Property 4: Leakage column exclusion** — generate DataFrames with leakage-keyword columns, assert exclusion per mode
    - **Validates: Requirements REQ-2.1**
  - [ ]* 12.5 Add property test for time-based split ordering to `tests/property/test_pbt_pipeline.py`
    - **Property 5: Time-based split temporal ordering** — generate DataFrames with random timestamps, assert train max ≤ valid min
    - **Validates: Requirements REQ-2.2**
  - [ ]* 12.6 Add property test for leakage recommendation threshold to `tests/property/test_pbt_pipeline.py`
    - **Property 6: Leakage recommendation threshold** — generate (full_f1, safe_f1) pairs, assert recommendation logic
    - **Validates: Requirements REQ-2.6**

- [ ] 13. Write property-based tests for API (Hypothesis)
  - [ ]* 13.1 Create `tests/property/test_pbt_api.py`
    - **Property 7: Default mode is safe** — generate valid predict requests without `mode`, assert response mode='safe'
    - **Validates: Requirements REQ-3.2**
  - [ ]* 13.2 Add property test for invalid input to `tests/property/test_pbt_api.py`
    - **Property 8: Invalid input returns 422** — generate malformed request bodies, assert 422
    - **Validates: Requirements REQ-3.6**
  - [ ]* 13.3 Add property test for SHAP explanation completeness to `tests/property/test_pbt_api.py`
    - **Property 9: SHAP explanation completeness** — generate random predictions, call explain, assert all fields present and list length = top_n
    - **Validates: Requirements REQ-4.1, REQ-4.2**

- [ ] 14. Write property-based tests for drift detection (Hypothesis)
  - [ ]* 14.1 Create `tests/property/test_pbt_drift.py`
    - **Property 10: PSI self-similarity is zero** — generate random distributions, compute PSI against self, assert < 0.05
    - **Validates: Requirements REQ-5.3**
  - [ ]* 14.2 Add property test for drift status worst-feature rule to `tests/property/test_pbt_drift.py`
    - **Property 11: Drift status reflects worst feature** — generate random per-feature PSI values, assert overall = max severity
    - **Validates: Requirements REQ-5.4, REQ-5.5**

- [ ] 15. Write property-based tests for audit (Hypothesis)
  - [ ]* 15.1 Create `tests/property/test_pbt_audit.py`
    - **Property 12: Audit log round-trip and append-only invariant** — generate random batches, predict, query, assert records present and count non-decreasing
    - **Validates: Requirements REQ-7.1, REQ-7.2**
  - [ ]* 15.2 Add property test for audit filter correctness to `tests/property/test_pbt_audit.py`
    - **Property 13: Audit log filter correctness** — generate random filter params, assert all returned records satisfy filters
    - **Validates: Requirements REQ-7.3**
  - [ ]* 15.3 Add property test for CSV column completeness to `tests/property/test_pbt_audit.py`
    - **Property 14: Audit CSV export column completeness** — generate random audit records, export CSV, assert all 6 required columns present
    - **Validates: Requirements REQ-7.4**
  - [ ]* 15.4 Add property test for 90-day retention invariant to `tests/property/test_pbt_audit.py`
    - **Property 15: 90-day retention invariant** — generate random `created_at` values, assert `retained_until = created_at + 90d`
    - **Validates: Requirements REQ-7.5**

- [~] 16. Implement React compliance dashboard
  - [~] 16.1 Scaffold React SPA in `frontend/` with 5-tab layout (Model Metrics, Feature Importance, SHAP, Inference, Fraud Report)
    - Set up `apiClient` with mock fallback: on network error or non-2xx, return mock data instead of throwing
    - _Requirements: REQ-6.7_
  - [~] 16.2 Implement Model Metrics tab
    - Fetch `GET /model/info`; display F1, AUC, Precision, Recall, PR-AUC; mode switcher (`full`/`no_leak`/`safe`) updates all tabs without page reload
    - _Requirements: REQ-6.2_
  - [~] 16.3 Implement Feature Importance tab
    - Bar chart of top-20 features from `/model/info`; sortable table
    - _Requirements: REQ-6.3_
  - [~] 16.4 Implement SHAP tab
    - Display global summary PNG from `GET /explain/summary`; per-user drill-down via `GET /explain/{user_id}`
    - _Requirements: REQ-6.4_
  - [~] 16.5 Implement Inference tab
    - CSV upload → `POST /predict` → display flagged users with risk scores
    - _Requirements: REQ-6.5_
  - [~] 16.6 Implement Fraud Report tab
    - Fetch `GET /audit/log`; render only `predicted_label=1` users sorted by `risk_score` descending; CSV export button
    - _Requirements: REQ-6.6, REQ-7.4_

- [ ] 17. Write dashboard tests (Jest/RTL)
  - [ ]* 17.1 Write unit tests in `tests/unit/test_dashboard_components.test.tsx`
    - Test each tab renders with mock data; test mode switcher propagates across tabs; test CSV export triggers download
    - _Requirements: REQ-6.2, REQ-6.3, REQ-6.4, REQ-6.5, REQ-6.6_
  - [ ]* 17.2 Create `tests/property/test_pbt_dashboard.test.tsx` for dashboard property tests
    - **Property 16: Fraud report ranking and filtering** — generate random prediction lists, assert only `predicted_label=1` and sorted desc by `risk_score`
    - **Property 17: Dashboard mock fallback** — simulate network errors, assert component renders with mock data
    - **Validates: Requirements REQ-6.6, REQ-6.7**

- [~] 18. Final checkpoint — full system integration
  - Ensure all Python tests pass (`pytest tests/ -x`), ask the user if questions arise.
  - Ensure all React tests pass (`jest --run`), ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis with `@settings(max_examples=100)`; each test tagged `# Feature: aml-fraud-detection, Property N: ...`
- React tests use Jest + React Testing Library
- `app/` is the new FastAPI service directory; existing `feature_engineering.py` and `model_xgboost.py` at root are not modified
