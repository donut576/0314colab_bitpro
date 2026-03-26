# AML Advanced Features — Requirements

## Introduction

This document specifies the advanced evolution of the BitoPro AML fraud detection system. The existing system provides batch XGBoost inference, SHAP explainability, PSI drift detection, and a compliance dashboard. This roadmap extends it into a production-grade, real-time, graph-aware, behaviorally-intelligent AML platform suitable for a crypto exchange operating at scale.

The eight feature areas below address the full detection lifecycle: streaming ingestion, graph-based money flow analysis, identity clustering, temporal behavioral profiling, adaptive risk thresholds, analyst case management, AI-assisted investigation, and cross-platform alerting.

---

## Glossary

- **Streaming_Pipeline**: The real-time event ingestion and scoring subsystem built on Kafka or AWS Kinesis.
- **Graph_Engine**: The component responsible for constructing and querying the transaction graph and running GNN inference.
- **Identity_Clusterer**: The component that links accounts sharing behavioral or network signals into identity clusters.
- **Sequence_Model**: The temporal model (LSTM or Transformer) that scores behavioral sequences per user.
- **Threshold_Controller**: The component that dynamically adjusts fraud score thresholds based on business rules and model calibration.
- **Case_Manager**: The backend service managing analyst investigation workflows, case states, and audit trails.
- **AI_Copilot**: The LLM-powered assistant that generates natural-language explanations and investigation suggestions for analysts.
- **Alert_Router**: The component that dispatches risk alerts to external channels (LINE, email, webhook).
- **Feature_Store**: The centralized store for pre-computed and real-time features shared across models.
- **Risk_Score**: A float in [0, 1] representing the probability that a user or transaction is fraudulent.
- **Risk_Level**: A categorical label derived from Risk_Score: HIGH (≥ threshold_high), MEDIUM (≥ threshold_medium), LOW (below threshold_medium).
- **Case**: A structured investigation record created when a user is flagged, containing evidence, analyst notes, and resolution status.
- **GNN**: Graph Neural Network — a model that learns fraud patterns from the transaction graph topology.
- **SHAP**: SHapley Additive exPlanations — the existing feature attribution method.
- **PSI**: Population Stability Index — the existing drift detection metric.
- **Mule_Account**: A user account used to receive and forward illicit funds, the primary detection target.
- **Identity_Cluster**: A set of accounts linked by shared signals (IP, device, behavioral fingerprint) suspected to be controlled by the same entity.

---

## Requirements

### REQ-A1: Real-Time Streaming Fraud Detection

**User Story:**
As a risk operations engineer, I want transaction events to be scored for fraud within seconds of occurring, so that the system can block or flag suspicious activity before funds leave the platform.

**Problem Solved:**
The existing system is batch-only. Fraud can complete within minutes; batch scoring hours later is too late for intervention.

**Data Needed:**
- Real-time transaction events from BitoPro (twd_transfer, crypto_transfer, usdt_swap) published to a message broker.
- Pre-computed user feature snapshots from the Feature_Store.
- Model artifact loaded in the Streaming_Pipeline worker.

**Priority / Difficulty:** P0 / High

#### Acceptance Criteria

1. WHEN a transaction event is published to the message broker, THE Streaming_Pipeline SHALL consume and score the event within 5 seconds end-to-end (p95 latency).
2. THE Streaming_Pipeline SHALL support at least 500 events per second sustained throughput without message loss.
3. WHEN the Streaming_Pipeline scores a transaction, THE Streaming_Pipeline SHALL enrich the event with pre-computed user features from the Feature_Store before inference.
4. WHEN a scored event has a Risk_Score above the configured threshold, THE Streaming_Pipeline SHALL publish a risk alert to the Alert_Router within 1 second of scoring.
5. IF the message broker is temporarily unavailable, THEN THE Streaming_Pipeline SHALL buffer events locally and resume processing upon reconnection without data loss.
6. IF the Feature_Store is unavailable during enrichment, THEN THE Streaming_Pipeline SHALL fall back to a reduced feature set and annotate the prediction with a `feature_degraded: true` flag.
7. THE Streaming_Pipeline SHALL expose a `GET /stream/health` endpoint returning broker connectivity status, consumer lag, and events-per-second throughput.
8. WHILE the Streaming_Pipeline is running, THE Streaming_Pipeline SHALL emit per-event prediction records to the audit log with the same schema as batch predictions.
9. THE Streaming_Pipeline SHALL support both Apache Kafka and AWS Kinesis as interchangeable broker backends, selectable via environment variable `STREAM_BROKER_TYPE`.

---

### REQ-A2: Graph-Based Fraud Detection (Money Flow Network)

**User Story:**
As an ML engineer, I want to model the transaction network as a graph and detect fraud rings using graph neural networks, so that the system can catch coordinated mule networks that individual-account models miss.

**Problem Solved:**
XGBoost operates on per-user features and cannot detect ring structures where individually low-risk accounts collectively form a fraud network.

**Data Needed:**
- Transaction edges: (sender_user_id, receiver_user_id, amount, timestamp, channel).
- Wallet address edges: (user_id, wallet_address) for crypto transfers.
- Node features: existing per-user feature vector from Feature_Store.
- Ground-truth labels for supervised GNN training.

**Priority / Difficulty:** P1 / Very High

#### Acceptance Criteria

1. THE Graph_Engine SHALL construct a directed transaction graph where nodes are user accounts and edges are fund transfers, updated incrementally with each new transaction batch.
2. WHEN the transaction graph is updated, THE Graph_Engine SHALL recompute node embeddings using a GNN within 10 minutes for graphs up to 1 million nodes.
3. THE Graph_Engine SHALL support at minimum a 2-layer GraphSAGE or GAT architecture with configurable hidden dimensions.
4. WHEN a user is queried, THE Graph_Engine SHALL return the user's graph embedding, their 1-hop and 2-hop neighbor count, and their betweenness centrality score.
5. THE Graph_Engine SHALL expose a `POST /graph/score` endpoint that accepts a list of user_ids and returns graph-based risk scores in addition to the existing XGBoost score.
6. WHEN graph-based and XGBoost scores are both available, THE Graph_Engine SHALL produce an ensemble score as a weighted combination, with weights configurable via environment variables.
7. THE Graph_Engine SHALL expose a `GET /graph/subgraph/{user_id}` endpoint returning the ego network (up to 2 hops) as a JSON node-link structure suitable for frontend visualization.
8. IF a user's 1-hop neighborhood contains more than a configurable fraction of HIGH Risk_Level accounts, THEN THE Graph_Engine SHALL elevate the user's Risk_Level by one tier regardless of individual score.
9. THE Graph_Engine SHALL store graph snapshots versioned by date, retaining at minimum the last 30 daily snapshots for temporal analysis.
10. WHERE graph visualization is enabled in the dashboard, THE Graph_Engine SHALL provide node color encoding by Risk_Level and edge weight encoding by transfer amount.

---

### REQ-A3: Multi-Account Linking and Identity Clustering

**User Story:**
As a compliance officer, I want the system to automatically group accounts that appear to be controlled by the same entity, so that I can investigate the full scope of a fraud operation rather than isolated accounts.

**Problem Solved:**
Fraudsters operate multiple accounts. Investigating accounts in isolation misses the coordinated pattern and underestimates the total exposure.

**Data Needed:**
- Shared IP addresses across accounts.
- Shared device fingerprints (if available).
- Shared wallet addresses across crypto transfers.
- KYC submission timing patterns.
- Behavioral feature similarity vectors from Feature_Store.

**Priority / Difficulty:** P1 / High

#### Acceptance Criteria

1. THE Identity_Clusterer SHALL link accounts sharing the same IP address within a configurable time window (default: 30 days) into candidate Identity_Clusters.
2. THE Identity_Clusterer SHALL link accounts sharing the same withdrawal wallet address into candidate Identity_Clusters.
3. WHEN two candidate clusters share at least one common account, THE Identity_Clusterer SHALL merge them into a single Identity_Cluster.
4. THE Identity_Clusterer SHALL assign a cluster_risk_score to each Identity_Cluster, computed as the maximum Risk_Score among all member accounts.
5. WHEN a new account is scored, THE Identity_Clusterer SHALL check whether the account belongs to an existing Identity_Cluster and include the cluster_risk_score in the prediction response.
6. THE Identity_Clusterer SHALL expose a `GET /clusters/{cluster_id}` endpoint returning all member account IDs, shared signals, cluster_risk_score, and cluster creation timestamp.
7. THE Identity_Clusterer SHALL expose a `GET /clusters/account/{user_id}` endpoint returning the Identity_Cluster membership for a given account, or a 404 if the account is not clustered.
8. IF an Identity_Cluster's cluster_risk_score exceeds the HIGH threshold, THEN THE Identity_Clusterer SHALL automatically create a Case in the Case_Manager for analyst review.
9. THE Identity_Clusterer SHALL recompute cluster memberships on a configurable schedule (default: every 6 hours) and emit a diff report of new, merged, and dissolved clusters.
10. THE Identity_Clusterer SHALL expose a `GET /clusters/stats` endpoint returning total cluster count, average cluster size, and count of HIGH-risk clusters.

---

### REQ-A4: Behavioral Profiling Over Time (Sequence Models)

**User Story:**
As an ML engineer, I want to model each user's transaction behavior as a time series and detect anomalous behavioral shifts, so that the system can catch accounts that gradually escalate suspicious activity to evade static rule-based detection.

**Problem Solved:**
Static feature aggregates miss temporal patterns. A mule account may look normal for weeks before activating. Sequence models capture the trajectory, not just the snapshot.

**Data Needed:**
- Per-user ordered transaction sequences: (timestamp, channel, amount, counterparty_type, direction).
- Session-level behavioral events: login time, KYC step timing, withdrawal request timing.
- Historical feature snapshots from Feature_Store (daily or weekly rollups).

**Priority / Difficulty:** P1 / High

#### Acceptance Criteria

1. THE Sequence_Model SHALL encode each user's transaction history as an ordered sequence of event vectors, with a configurable lookback window (default: 90 days).
2. THE Sequence_Model SHALL support at minimum an LSTM or Transformer architecture with configurable sequence length and embedding dimensions.
3. WHEN a user's behavioral sequence is scored, THE Sequence_Model SHALL return a sequence_anomaly_score in [0, 1] representing deviation from the user's own historical baseline.
4. WHEN a user's sequence_anomaly_score exceeds a configurable threshold, THE Sequence_Model SHALL flag the user for review and include the top-3 anomalous events with their timestamps in the response.
5. THE Sequence_Model SHALL be retrained on a configurable schedule (default: weekly) using the latest transaction sequences.
6. THE Sequence_Model SHALL expose a `POST /sequence/score` endpoint accepting a user_id and returning the sequence_anomaly_score, top anomalous events, and the model version used.
7. WHEN both XGBoost and Sequence_Model scores are available, THE Streaming_Pipeline SHALL include both scores in the prediction response with their respective model versions.
8. THE Sequence_Model SHALL store per-user behavioral baselines in the Feature_Store, updated incrementally after each scoring run.
9. IF a user has fewer than a configurable minimum number of transactions (default: 5), THEN THE Sequence_Model SHALL return a `insufficient_history: true` flag and skip sequence scoring for that user.
10. THE Sequence_Model SHALL expose a `GET /sequence/profile/{user_id}` endpoint returning the user's behavioral baseline statistics and the last 10 scored events.

---

### REQ-A5: Adaptive Threshold System

**User Story:**
As a risk operations manager, I want the fraud detection thresholds to adapt dynamically based on business context, model calibration, and operational capacity, so that the system balances fraud catch rate against analyst workload and false positive cost.

**Problem Solved:**
A fixed threshold ignores business reality. During high-volume periods, a static threshold floods analysts with alerts. During low-volume periods, it may be too conservative. Thresholds must reflect both model confidence and operational constraints.

**Data Needed:**
- Current model calibration curve (precision-recall at each threshold).
- Historical alert volume and analyst resolution rate from Case_Manager.
- Business-defined cost parameters: cost_of_false_positive, cost_of_false_negative.
- Current queue depth from Case_Manager.

**Priority / Difficulty:** P1 / Medium

#### Acceptance Criteria

1. THE Threshold_Controller SHALL maintain separate threshold values for HIGH and MEDIUM Risk_Level classifications, both configurable via API without service restart.
2. WHEN the Case_Manager queue depth exceeds a configurable maximum (default: 500 open cases), THE Threshold_Controller SHALL automatically raise the HIGH threshold to reduce new alert volume, up to a configurable ceiling.
3. WHEN the Case_Manager queue depth falls below a configurable minimum (default: 50 open cases), THE Threshold_Controller SHALL lower the HIGH threshold toward the model-optimal value, down to a configurable floor.
4. THE Threshold_Controller SHALL expose a `GET /thresholds/current` endpoint returning the current HIGH and MEDIUM thresholds, their last update timestamp, and the reason for the last change.
5. THE Threshold_Controller SHALL expose a `POST /thresholds/override` endpoint allowing authorized operators to manually set thresholds with a mandatory reason string and expiry timestamp.
6. WHEN a manual threshold override expires, THE Threshold_Controller SHALL revert to the adaptive algorithm and log the reversion event.
7. THE Threshold_Controller SHALL compute and expose a `GET /thresholds/simulation` endpoint that accepts a proposed threshold value and returns the estimated alert volume, estimated recall, and estimated precision based on the last 7 days of predictions.
8. THE Threshold_Controller SHALL log every threshold change (automatic or manual) to the audit log with the old value, new value, reason, and operator identity.
9. WHERE cost parameters are configured, THE Threshold_Controller SHALL compute the expected cost at each candidate threshold and select the threshold that minimizes total expected cost.
10. THE Threshold_Controller SHALL expose a `GET /thresholds/history` endpoint returning the last 30 threshold change events with timestamps and reasons.

---

### REQ-A6: Alerting and Case Management System

**User Story:**
As a compliance analyst, I want a structured case management system where flagged accounts are automatically converted into investigation cases, so that I can track, prioritize, and resolve fraud investigations with a complete audit trail.

**Problem Solved:**
The existing system flags accounts but provides no workflow for analysts to act on them. Without case management, alerts are lost, duplicated, or resolved inconsistently.

**Data Needed:**
- Prediction outputs (user_id, risk_score, risk_level, model_version, shap_values).
- Graph subgraph for the flagged user.
- Identity_Cluster membership.
- Historical prediction records for the user.

**Priority / Difficulty:** P0 / Medium

#### Acceptance Criteria

1. WHEN a prediction result has Risk_Level equal to HIGH, THE Case_Manager SHALL automatically create a Case containing the user_id, risk_score, risk_level, triggering model version, SHAP top features, and creation timestamp.
2. THE Case_Manager SHALL assign each Case a unique case_id and set its initial status to `open`.
3. THE Case_Manager SHALL support the following Case status transitions: `open` → `in_review` → `resolved` or `open` → `escalated` → `resolved`.
4. WHEN an analyst updates a Case status, THE Case_Manager SHALL record the analyst_id, timestamp, new status, and an optional note in the Case audit trail.
5. THE Case_Manager SHALL expose a `GET /cases` endpoint returning paginated cases filterable by status, risk_level, date range, and assigned analyst.
6. THE Case_Manager SHALL expose a `GET /cases/{case_id}` endpoint returning the full Case detail including audit trail, linked Identity_Cluster, graph subgraph summary, and all historical predictions for the user.
7. THE Case_Manager SHALL expose a `POST /cases/{case_id}/assign` endpoint allowing a supervisor to assign a case to a specific analyst.
8. THE Case_Manager SHALL expose a `POST /cases/{case_id}/resolve` endpoint requiring a resolution_type (`confirmed_fraud`, `false_positive`, `insufficient_evidence`) and a mandatory resolution_note.
9. WHEN a Case is resolved as `confirmed_fraud`, THE Case_Manager SHALL tag the user_id in a confirmed fraud registry and notify the Alert_Router.
10. THE Case_Manager SHALL expose a `GET /cases/stats` endpoint returning open case count, average resolution time, false positive rate, and confirmed fraud rate for a configurable time window.
11. IF a user already has an open Case, THEN THE Case_Manager SHALL append the new prediction to the existing Case rather than creating a duplicate.
12. THE Case_Manager SHALL support bulk case export as CSV including all fields required for regulatory reporting.

---

### REQ-A7: AI Copilot for Analysts (LLM Integration)

**User Story:**
As a compliance analyst, I want an AI assistant that can explain a flagged account's risk factors in plain language and suggest investigation steps, so that I can make faster, better-informed decisions especially on complex cases.

**Problem Solved:**
SHAP values and graph metrics are powerful but require ML expertise to interpret. Junior analysts spend significant time understanding why an account was flagged. An LLM copilot translates technical signals into actionable investigation narratives.

**Data Needed:**
- Case detail: risk_score, SHAP top features, sequence_anomaly_score, graph neighborhood summary, Identity_Cluster membership.
- Historical similar cases and their resolutions from Case_Manager.
- Regulatory context: AML typology descriptions (configurable knowledge base).

**Priority / Difficulty:** P2 / High

#### Acceptance Criteria

1. THE AI_Copilot SHALL expose a `POST /copilot/explain/{case_id}` endpoint that returns a natural-language explanation of why the account was flagged, referencing the top SHAP features, graph signals, and sequence anomalies in plain language.
2. THE AI_Copilot SHALL expose a `POST /copilot/suggest/{case_id}` endpoint that returns a prioritized list of investigation steps tailored to the specific risk signals present in the case.
3. WHEN generating an explanation or suggestion, THE AI_Copilot SHALL complete the response within 10 seconds under normal load.
4. THE AI_Copilot SHALL support streaming responses via Server-Sent Events so that the analyst UI can display text progressively.
5. THE AI_Copilot SHALL never include raw PII (full name, national ID, phone number) in generated responses; all user references SHALL use user_id only.
6. THE AI_Copilot SHALL expose a `POST /copilot/compare/{case_id}` endpoint that retrieves the top-3 most similar historical resolved cases and summarizes their resolution rationale.
7. WHEN the LLM provider is unavailable, THE AI_Copilot SHALL return a structured fallback response containing the raw SHAP top features and a template-based explanation without LLM generation.
8. THE AI_Copilot SHALL log every request and response (excluding PII) to the audit log with the case_id, model version, and latency.
9. WHERE a configurable AML typology knowledge base is provided, THE AI_Copilot SHALL reference relevant typology patterns in its explanations.
10. THE AI_Copilot SHALL support configurable LLM backends selectable via environment variable `COPILOT_LLM_PROVIDER` (e.g., OpenAI, Anthropic, local Ollama).

---

### REQ-A8: Cross-Platform Risk Intelligence and Alerting

**User Story:**
As a risk operations manager, I want high-risk alerts to be automatically dispatched to LINE, email, and internal dashboards in real time, so that the team is notified immediately without needing to monitor the dashboard continuously.

**Problem Solved:**
The existing system surfaces alerts only in the dashboard. Analysts miss time-sensitive alerts when not actively monitoring. Cross-platform notifications enable immediate response.

**Data Needed:**
- Alert payload: user_id, risk_score, risk_level, case_id, top risk signals, timestamp.
- Routing configuration: which channels receive which risk levels.
- Rate limiting state: per-channel alert counts within rolling windows.

**Priority / Difficulty:** P1 / Medium

#### Acceptance Criteria

1. THE Alert_Router SHALL support at minimum three notification channels: LINE Notify (or LINE Bot), email (SMTP), and internal webhook.
2. WHEN a HIGH Risk_Level alert is generated, THE Alert_Router SHALL dispatch notifications to all configured channels within 30 seconds.
3. WHEN a MEDIUM Risk_Level alert is generated, THE Alert_Router SHALL dispatch notifications only to channels configured to receive MEDIUM alerts.
4. THE Alert_Router SHALL enforce per-channel rate limiting: no more than a configurable maximum number of alerts per hour per channel (default: 60), to prevent alert fatigue.
5. WHEN the per-channel rate limit is reached, THE Alert_Router SHALL queue excess alerts and dispatch them in the next available window, logging the delay.
6. THE Alert_Router SHALL include in every notification: case_id, user_id, risk_score, risk_level, top-3 risk signals, and a deep link to the Case_Manager UI.
7. THE Alert_Router SHALL expose a `GET /alerts/history` endpoint returning the last 1,000 dispatched alerts with channel, status (delivered/failed/queued), and timestamp.
8. IF a notification dispatch fails after 3 retry attempts, THEN THE Alert_Router SHALL mark the alert as `failed`, log the error, and surface the failure in the dashboard.
9. THE Alert_Router SHALL support alert suppression rules: if the same user_id has triggered an alert within a configurable cooldown window (default: 1 hour), suppress duplicate alerts.
10. THE Alert_Router SHALL expose a `POST /alerts/test` endpoint that sends a test notification to all configured channels to verify connectivity.

---

### REQ-A9: Feature Store and Scalable Data Infrastructure

**User Story:**
As a data engineer, I want a centralized feature store that serves pre-computed features to all models in real time, so that feature computation is consistent, reusable, and not duplicated across the streaming pipeline, batch training, and graph engine.

**Problem Solved:**
Without a feature store, each model recomputes features independently, leading to training-serving skew, duplicated computation, and inconsistent feature definitions across models.

**Data Needed:**
- All features currently produced by feature_engineering.py.
- Real-time event features computed by the Streaming_Pipeline.
- Graph embeddings from the Graph_Engine.
- Behavioral baselines from the Sequence_Model.

**Priority / Difficulty:** P1 / High

#### Acceptance Criteria

1. THE Feature_Store SHALL serve pre-computed user feature vectors with a read latency of less than 20 milliseconds at p99 for single-user lookups.
2. THE Feature_Store SHALL support both point-in-time correct feature retrieval (for training) and latest-value retrieval (for inference).
3. WHEN a feature computation job completes, THE Feature_Store SHALL update the stored feature vector for affected users within 60 seconds.
4. THE Feature_Store SHALL version feature definitions, ensuring that a model trained on feature schema v2 can always retrieve v2 features even after v3 is deployed.
5. THE Feature_Store SHALL expose a `GET /features/{user_id}` endpoint returning the latest feature vector for a user, with feature schema version and last-updated timestamp.
6. THE Feature_Store SHALL expose a `POST /features/batch` endpoint accepting a list of user_ids and returning their feature vectors in a single response.
7. IF a user_id has no stored features, THEN THE Feature_Store SHALL return a zero-vector with a `cold_start: true` flag rather than a 404 error.
8. THE Feature_Store SHALL support feature group namespacing (e.g., `behavioral`, `graph`, `kyc`, `sequence`) so that models can request only the feature groups they need.
9. THE Feature_Store SHALL log all feature reads with user_id, feature schema version, and requesting service for lineage tracking.
10. THE Feature_Store SHALL expose a `GET /features/stats` endpoint returning total user count, feature schema version, last update timestamp, and storage utilization.

---

### REQ-A10: ML and System Monitoring

**User Story:**
As an ML engineer, I want comprehensive monitoring of model performance, data quality, and system health across all components, so that I can detect degradation early and maintain SLA compliance.

**Problem Solved:**
The existing system has PSI drift detection for the XGBoost model only. With multiple models (GNN, Sequence_Model, XGBoost ensemble), monitoring must be unified and cover model performance, not just feature drift.

**Data Needed:**
- Prediction logs from all models.
- Ground-truth labels from confirmed fraud cases (delayed feedback).
- System metrics: latency, throughput, error rates per component.
- Feature drift metrics from Feature_Store.

**Priority / Difficulty:** P1 / Medium

#### Acceptance Criteria

1. THE Monitoring_System SHALL track and expose the following per-model metrics updated at least every 15 minutes: prediction count, average risk_score, HIGH/MEDIUM/LOW distribution, and p50/p95/p99 inference latency.
2. WHEN confirmed fraud labels become available from resolved Cases, THE Monitoring_System SHALL compute rolling precision, recall, and F1 for each model over the last 7 and 30 days.
3. THE Monitoring_System SHALL extend the existing PSI drift detection to cover all models in the ensemble, not only XGBoost.
4. WHEN any model's rolling F1 drops more than 0.05 below its baseline F1 (recorded at deployment), THE Monitoring_System SHALL emit a `model_degradation` alert to the Alert_Router.
5. THE Monitoring_System SHALL expose a `GET /monitoring/dashboard` endpoint returning a unified health summary for all components: Streaming_Pipeline, Graph_Engine, Sequence_Model, Feature_Store, Case_Manager, and Alert_Router.
6. THE Monitoring_System SHALL track and alert on system-level SLA breaches: Streaming_Pipeline p95 latency > 5s, Feature_Store p99 read latency > 20ms, Case_Manager API p95 latency > 500ms.
7. WHEN a SLA breach is detected, THE Monitoring_System SHALL emit an alert within 60 seconds of the breach occurring.
8. THE Monitoring_System SHALL retain monitoring metrics for at least 90 days for trend analysis and regulatory review.
9. THE Monitoring_System SHALL expose a `GET /monitoring/model/{model_name}/calibration` endpoint returning the current calibration curve (precision-recall at 20 threshold points) for the specified model.
10. THE Monitoring_System SHALL support integration with external observability platforms (Prometheus metrics endpoint, Grafana-compatible JSON) via configurable exporters.

---

### REQ-A11: Analyst Investigation Dashboard (Frontend Upgrade)

**User Story:**
As a compliance analyst, I want an upgraded investigation dashboard that integrates case management, graph visualization, behavioral timelines, and the AI copilot in a single interface, so that I can conduct end-to-end fraud investigations without switching between tools.

**Problem Solved:**
The existing dashboard is read-only and model-centric. Analysts need an action-oriented interface that supports the full investigation workflow.

**Data Needed:**
- All Case_Manager endpoints.
- Graph_Engine subgraph endpoint.
- Sequence_Model profile endpoint.
- AI_Copilot endpoints.
- Alert_Router history endpoint.

**Priority / Difficulty:** P1 / Medium

#### Acceptance Criteria

1. THE Dashboard SHALL add a Case Management tab displaying a filterable, sortable table of all Cases with columns: case_id, user_id, risk_score, risk_level, status, assigned analyst, and creation timestamp.
2. WHEN an analyst opens a Case detail view, THE Dashboard SHALL display: risk score timeline, SHAP waterfall chart, graph neighborhood visualization (up to 2 hops), Identity_Cluster membership, and the AI_Copilot explanation panel.
3. THE Dashboard SHALL render the graph neighborhood visualization as an interactive force-directed graph with node color encoding by Risk_Level and edge weight encoding by transfer amount.
4. THE Dashboard SHALL display a behavioral timeline chart showing the user's transaction activity over the configured lookback window, with anomalous events highlighted.
5. THE Dashboard SHALL embed the AI_Copilot panel in the Case detail view, allowing analysts to request explanations and investigation suggestions without leaving the page.
6. THE Dashboard SHALL allow analysts to update Case status, add notes, and assign cases directly from the Case detail view, with changes reflected in real time.
7. THE Dashboard SHALL add a real-time alert feed panel showing the last 20 alerts dispatched by the Alert_Router, auto-refreshing every 30 seconds.
8. THE Dashboard SHALL add a System Health tab displaying the unified monitoring dashboard from `GET /monitoring/dashboard`, with visual indicators for SLA compliance.
9. WHEN any API call fails, THE Dashboard SHALL display a non-blocking error toast and fall back to the last successfully loaded data rather than showing a blank state.
10. THE Dashboard SHALL support dark mode and be responsive for screen widths from 1280px to 2560px.

---

## Non-Functional Requirements

| Category | Requirement |
|---|---|
| Streaming Latency | End-to-end event scoring p95 < 5 seconds |
| Batch Inference | `/predict` p95 < 2 seconds for 1,000 users (existing SLA maintained) |
| Feature Store Read | p99 single-user lookup < 20 milliseconds |
| Graph Recompute | Node embedding refresh < 10 minutes for 1M-node graph |
| Alert Dispatch | HIGH-risk alert delivered to all channels < 30 seconds |
| API Availability | All services uptime > 99.5% |
| Case Manager API | p95 latency < 500 milliseconds |
| Data Retention | Prediction logs, audit trails, monitoring metrics: minimum 90 days |
| Security | No PII in logs or LLM prompts; all admin endpoints require authentication |
| Scalability | Streaming_Pipeline horizontally scalable to 10× baseline throughput via consumer group scaling |
| Observability | All services emit structured logs and Prometheus-compatible metrics |
| Portability | Streaming broker backend switchable between Kafka and Kinesis via env var |

---

## Out of Scope

- Automated model retraining triggered by drift alerts (drift detection triggers alert only).
- KYC document verification or OCR.
- Direct integration with external law enforcement databases.
- Mobile application for analysts.
- Multi-tenant / multi-exchange support (BitoPro only in this phase).
