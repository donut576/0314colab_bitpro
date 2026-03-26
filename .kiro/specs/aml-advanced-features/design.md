# AML Advanced Features — Design

## Overview

This document describes the technical design for the advanced evolution of the BitoPro AML fraud detection platform. The existing system provides batch XGBoost inference, SHAP explainability, PSI drift detection, and a compliance dashboard. This design extends it into a production-grade, real-time, graph-aware, behaviorally-intelligent AML platform.

### Design Goals

- Score transactions within 5 seconds end-to-end via a streaming pipeline (Kafka/Kinesis)
- Detect coordinated fraud rings using graph neural networks (GraphSAGE/GAT)
- Link multi-account fraud operations via identity clustering
- Capture temporal behavioral shifts using LSTM/Transformer sequence models
- Adapt detection thresholds dynamically to operational capacity
- Provide structured case management with full audit trails
- Accelerate analyst investigations via an LLM-powered AI copilot
- Dispatch risk alerts to LINE, email, and webhooks in real time
- Centralize feature computation in a shared feature store
- Unify monitoring across all models and system components
- Upgrade the analyst dashboard to support end-to-end investigation workflows

### Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Streaming broker | Kafka (primary) / Kinesis (secondary) | Switchable via `STREAM_BROKER_TYPE` env var; Kafka for on-prem, Kinesis for AWS |
| Graph framework | PyTorch Geometric (PyG) | Best-in-class GNN support; GraphSAGE and GAT available out of the box |
| Graph storage | NetworkX (in-memory) + PostgreSQL snapshots | NetworkX for traversal; Postgres for versioned daily snapshots |
| Sequence model | PyTorch LSTM / Transformer | Flexible architecture; configurable via env vars |
| Feature store backend | Redis (hot) + PostgreSQL (cold) | Redis for <20ms p99 reads; Postgres for point-in-time correct training retrieval |
| Case management DB | PostgreSQL `cases` table | ACID guarantees; consistent with existing audit log infrastructure |
| LLM integration | OpenAI / Anthropic / Ollama (configurable) | Switchable via `COPILOT_LLM_PROVIDER`; Ollama for air-gapped deployments |
| Alert channels | LINE Notify, SMTP, HTTP webhook | Covers all required channels; rate-limited per channel |
| Monitoring metrics | Prometheus + Grafana-compatible JSON | Standard observability stack; integrates with existing infra |
| Property-based testing | Hypothesis (Python) | Consistent with existing test suite |

---

## Architecture

```mermaid
graph TD
    subgraph Ingest["Ingestion Layer"]
        A[BitoPro Transaction Events] -->|publish| B[Kafka / Kinesis]
    end

    subgraph Stream["Streaming Pipeline"]
        B -->|consume| C[StreamConsumer]
        C -->|enrich| D[Feature_Store]
        D -->|features| E[XGBPredictor]
        E -->|score| F[EnsembleScorer]
        F -->|risk alert| G[Alert_Router]
        F -->|prediction record| H[AuditLogger]
    end

    subgraph Graph["Graph Engine"]
        I[Transaction Graph] -->|GNN inference| J[GraphSAGE/GAT]
        J -->|embeddings| F
        J -->|subgraph API| K[GET /graph/subgraph]
    end

    subgraph Sequence["Sequence Model"]
        L[User Event Sequences] -->|LSTM/Transformer| M[SequenceScorer]
        M -->|anomaly score| F
        M -->|baseline| D
    end

    subgraph Identity["Identity Clusterer"]
        N[Shared IP/Wallet/Device] -->|union-find| O[ClusterRegistry]
        O -->|cluster risk| F
        O -->|auto-create case| P[Case_Manager]
    end

    subgraph Cases["Case Management"]
        P[Case_Manager] -->|cases DB| Q[(PostgreSQL cases)]
        P -->|analyst workflow| R[Dashboard]
    end

    subgraph Copilot["AI Copilot"]
        S[POST /copilot/explain] -->|LLM prompt| T[LLM Provider]
        T -->|SSE stream| R
    end

    subgraph Alerts["Alert Router"]
        G -->|LINE Notify| U[LINE]
        G -->|SMTP| V[Email]
        G -->|HTTP| W[Webhook]
    end

    subgraph Monitor["Monitoring"]
        X[Monitoring_System] -->|Prometheus| Y[/metrics]
        X -->|unified health| Z[GET /monitoring/dashboard]
    end

    subgraph FeatureStore["Feature Store"]
        D -->|Redis hot| AA[Redis]
        D -->|Postgres cold| AB[(PostgreSQL features)]
    end
```

### Data Flow — Streaming Inference

```
Transaction Event → Kafka/Kinesis
  → StreamConsumer (consume, deserialize)
  → Feature_Store.get(user_id)          # Redis <20ms
  → XGBPredictor.predict_single()
  → GraphEngine.get_score(user_id)      # cached embedding
  → SequenceScorer.score(user_id)       # cached baseline
  → EnsembleScorer.combine()            # weighted average
  → if risk_score > threshold → Alert_Router.dispatch()
  → AuditLogger.log_prediction()        # async, non-blocking
```

---

## Components and Interfaces

### 1. StreamConsumer

Consumes events from Kafka or Kinesis. Broker backend selected via `STREAM_BROKER_TYPE` env var. Implements local buffering for broker unavailability.

```python
class StreamConsumer:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_health(self) -> StreamHealth: ...  # broker status, lag, events/sec
```

### 2. GraphEngine

Builds and queries the transaction graph. Runs GNN inference to produce node embeddings and risk scores.

```python
class GraphEngine:
    def update_graph(self, transactions: list[TransactionEdge]) -> None: ...
    def recompute_embeddings(self) -> None: ...  # GraphSAGE/GAT forward pass
    def get_score(self, user_id: str) -> GraphScore: ...
    def get_subgraph(self, user_id: str, hops: int = 2) -> SubgraphResult: ...
    def get_ensemble_score(self, user_id: str, xgb_score: float) -> float: ...
```

### 3. IdentityClusterer

Links accounts via shared signals using a union-find algorithm. Recomputes on schedule.

```python
class IdentityClusterer:
    def recompute_clusters(self) -> ClusterDiff: ...
    def get_cluster(self, cluster_id: str) -> IdentityCluster: ...
    def get_cluster_for_account(self, user_id: str) -> IdentityCluster | None: ...
    def get_stats(self) -> ClusterStats: ...
```

### 4. SequenceScorer

Encodes user transaction histories as sequences and scores behavioral anomalies.

```python
class SequenceScorer:
    def score(self, user_id: str) -> SequenceScore: ...
    def get_profile(self, user_id: str) -> BehavioralProfile: ...
    def retrain(self) -> None: ...
```

### 5. ThresholdController

Manages adaptive thresholds based on queue depth and cost parameters.

```python
class ThresholdController:
    def get_current(self) -> ThresholdState: ...
    def set_override(self, value: float, reason: str, expiry: datetime) -> None: ...
    def simulate(self, proposed: float) -> ThresholdSimulation: ...
    def get_history(self, limit: int = 30) -> list[ThresholdChangeEvent]: ...
    def tick(self) -> None: ...  # called periodically; applies adaptive logic
```

### 6. CaseManager

Manages investigation cases with full audit trails and status transitions.

```python
class CaseManager:
    def create_case(self, prediction: PredictionResult) -> Case: ...
    def get_case(self, case_id: str) -> Case: ...
    def list_cases(self, filters: CaseFilters) -> PaginatedCases: ...
    def update_status(self, case_id: str, status: CaseStatus, analyst_id: str, note: str) -> None: ...
    def assign(self, case_id: str, analyst_id: str) -> None: ...
    def resolve(self, case_id: str, resolution: CaseResolution) -> None: ...
    def get_stats(self, window_days: int) -> CaseStats: ...
    def export_csv(self, filters: CaseFilters) -> bytes: ...
```

### 7. AICopilot

Generates natural-language explanations and investigation suggestions via LLM.

```python
class AICopilot:
    def explain(self, case_id: str) -> AsyncIterator[str]: ...  # SSE stream
    def suggest(self, case_id: str) -> AsyncIterator[str]: ...  # SSE stream
    def compare(self, case_id: str) -> SimilarCasesResult: ...
```

### 8. AlertRouter

Dispatches risk alerts to configured channels with rate limiting and retry logic.

```python
class AlertRouter:
    def dispatch(self, alert: RiskAlert) -> None: ...
    def get_history(self, limit: int = 1000) -> list[AlertRecord]: ...
    def send_test(self) -> list[ChannelTestResult]: ...
```

### 9. FeatureStore

Centralized feature serving with Redis hot layer and PostgreSQL cold storage.

```python
class FeatureStore:
    def get(self, user_id: str, groups: list[str] | None = None) -> FeatureVector: ...
    def get_batch(self, user_ids: list[str]) -> list[FeatureVector]: ...
    def put(self, user_id: str, features: FeatureVector) -> None: ...
    def get_stats(self) -> FeatureStoreStats: ...
```

### 10. MonitoringSystem

Tracks per-model metrics, SLA compliance, and emits alerts on degradation.

```python
class MonitoringSystem:
    def get_dashboard(self) -> UnifiedHealthSummary: ...
    def get_model_calibration(self, model_name: str) -> CalibrationCurve: ...
    def tick(self) -> None: ...  # updates metrics, checks SLA, emits alerts
```

---

## Data Models

### PostgreSQL: `cases` Table

```sql
CREATE TABLE cases (
    case_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             VARCHAR(64) NOT NULL,
    risk_score          DOUBLE PRECISION NOT NULL,
    risk_level          VARCHAR(16) NOT NULL CHECK (risk_level IN ('HIGH', 'MEDIUM', 'LOW')),
    status              VARCHAR(32) NOT NULL DEFAULT 'open'
                            CHECK (status IN ('open', 'in_review', 'escalated', 'resolved')),
    assigned_analyst    VARCHAR(64),
    model_version       VARCHAR(64) NOT NULL,
    shap_top_features   JSONB,
    cluster_id          UUID,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE case_audit_trail (
    id              BIGSERIAL PRIMARY KEY,
    case_id         UUID NOT NULL REFERENCES cases(case_id),
    analyst_id      VARCHAR(64) NOT NULL,
    action          VARCHAR(64) NOT NULL,
    old_status      VARCHAR(32),
    new_status      VARCHAR(32),
    note            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cases_user_id    ON cases (user_id);
CREATE INDEX idx_cases_status     ON cases (status);
CREATE INDEX idx_cases_created_at ON cases (created_at DESC);
```

### PostgreSQL: `identity_clusters` Table

```sql
CREATE TABLE identity_clusters (
    cluster_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    member_user_ids     TEXT[] NOT NULL,
    shared_signals      JSONB NOT NULL,   -- {ips: [...], wallets: [...], devices: [...]}
    cluster_risk_score  DOUBLE PRECISION NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_clusters_member_ids ON identity_clusters USING GIN (member_user_ids);
```

### PostgreSQL: `threshold_history` Table

```sql
CREATE TABLE threshold_history (
    id              BIGSERIAL PRIMARY KEY,
    threshold_type  VARCHAR(16) NOT NULL CHECK (threshold_type IN ('HIGH', 'MEDIUM')),
    old_value       DOUBLE PRECISION NOT NULL,
    new_value       DOUBLE PRECISION NOT NULL,
    reason          TEXT NOT NULL,
    operator        VARCHAR(64),          -- NULL for automatic changes
    expiry          TIMESTAMPTZ,          -- NULL for permanent changes
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### Redis: Feature Store Key Schema

```
features:{user_id}:{schema_version}  →  JSON blob (FeatureVector)
TTL: 24 hours (refreshed on write)
```

### Pydantic Schemas (Key Types)

```python
class RiskLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class CaseStatus(str, Enum):
    OPEN = "open"
    IN_REVIEW = "in_review"
    ESCALATED = "escalated"
    RESOLVED = "resolved"

class FeatureVector(BaseModel):
    user_id: str
    schema_version: str
    features: dict[str, float]
    groups: list[str]           # e.g. ["behavioral", "graph", "kyc", "sequence"]
    last_updated: datetime
    cold_start: bool = False

class GraphScore(BaseModel):
    user_id: str
    graph_risk_score: float     # [0, 1]
    embedding: list[float]
    hop1_count: int
    hop2_count: int
    betweenness_centrality: float
    elevated: bool              # True if neighborhood HIGH fraction exceeded threshold

class SequenceScore(BaseModel):
    user_id: str
    sequence_anomaly_score: float   # [0, 1]
    top_anomalous_events: list[AnomalousEvent]
    model_version: str
    insufficient_history: bool = False

class Case(BaseModel):
    case_id: str
    user_id: str
    risk_score: float
    risk_level: RiskLevel
    status: CaseStatus
    assigned_analyst: str | None
    model_version: str
    shap_top_features: list[SHAPContribution]
    cluster_id: str | None
    audit_trail: list[CaseAuditEntry]
    created_at: datetime
    updated_at: datetime

class RiskAlert(BaseModel):
    case_id: str
    user_id: str
    risk_score: float
    risk_level: RiskLevel
    top_signals: list[str]      # top-3 risk signal descriptions
    timestamp: datetime
    deep_link: str              # URL to Case_Manager UI
```

### New API Endpoints Summary

| Method | Path | Component | Description |
|---|---|---|---|
| GET | /stream/health | StreamConsumer | Broker status, lag, events/sec |
| POST | /graph/score | GraphEngine | Graph-based risk scores for user list |
| GET | /graph/subgraph/{user_id} | GraphEngine | Ego network up to 2 hops |
| GET | /clusters/{cluster_id} | IdentityClusterer | Cluster detail |
| GET | /clusters/account/{user_id} | IdentityClusterer | Cluster membership for account |
| GET | /clusters/stats | IdentityClusterer | Aggregate cluster statistics |
| POST | /sequence/score | SequenceScorer | Behavioral anomaly score |
| GET | /sequence/profile/{user_id} | SequenceScorer | Behavioral baseline + last 10 events |
| GET | /thresholds/current | ThresholdController | Current thresholds + last change reason |
| POST | /thresholds/override | ThresholdController | Manual threshold override |
| GET | /thresholds/simulation | ThresholdController | Simulate proposed threshold |
| GET | /thresholds/history | ThresholdController | Last 30 threshold change events |
| GET | /cases | CaseManager | Paginated, filterable case list |
| GET | /cases/{case_id} | CaseManager | Full case detail |
| POST | /cases/{case_id}/assign | CaseManager | Assign case to analyst |
| POST | /cases/{case_id}/resolve | CaseManager | Resolve case with type + note |
| GET | /cases/stats | CaseManager | Aggregate case statistics |
| POST | /copilot/explain/{case_id} | AICopilot | LLM explanation (SSE) |
| POST | /copilot/suggest/{case_id} | AICopilot | Investigation suggestions (SSE) |
| POST | /copilot/compare/{case_id} | AICopilot | Similar historical cases |
| GET | /alerts/history | AlertRouter | Last 1,000 dispatched alerts |
| POST | /alerts/test | AlertRouter | Test notification to all channels |
| GET | /features/{user_id} | FeatureStore | Latest feature vector |
| POST | /features/batch | FeatureStore | Batch feature retrieval |
| GET | /features/stats | FeatureStore | Feature store statistics |
| GET | /monitoring/dashboard | MonitoringSystem | Unified health summary |
| GET | /monitoring/model/{model_name}/calibration | MonitoringSystem | Model calibration curve |

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Streaming enrichment completeness

*For any* transaction event published to the broker, the scored event must contain a feature vector sourced from the Feature_Store; if the Feature_Store is unavailable, the prediction response must include `feature_degraded: true` and must not include features from a different user.

**Validates: Requirements REQ-A1.3, REQ-A1.6**

---

### Property 2: Streaming audit parity

*For any* transaction event scored by the Streaming_Pipeline, a corresponding prediction record must appear in the audit log with the same `user_id`, `risk_score`, and `model_version` as the scored event.

**Validates: Requirements REQ-A1.8**

---

### Property 3: Graph neighborhood elevation

*For any* user whose 1-hop neighborhood contains a fraction of HIGH-risk accounts exceeding the configured threshold, the user's Risk_Level must be elevated by at least one tier compared to their individual score alone.

**Validates: Requirements REQ-A2.8**

---

### Property 4: Ensemble score bounds

*For any* user with both a graph score and an XGBoost score available, the ensemble score must be a weighted combination of the two and must remain in [0, 1].

**Validates: Requirements REQ-A2.6**

---

### Property 5: Cluster merge transitivity

*For any* set of candidate clusters where clusters A and B share a member and clusters B and C share a member, the resulting merged cluster must contain all members of A, B, and C.

**Validates: Requirements REQ-A3.3**

---

### Property 6: Cluster risk score is maximum member score

*For any* Identity_Cluster, the `cluster_risk_score` must equal the maximum `risk_score` among all member accounts. Adding a new member with a higher score must update the cluster risk score accordingly.

**Validates: Requirements REQ-A3.4**

---

### Property 7: Sequence score range and insufficient history flag

*For any* user with fewer than the configured minimum transaction count, the sequence scorer must return `insufficient_history: true` and must not return a `sequence_anomaly_score`. For any user with sufficient history, the returned score must be in [0, 1].

**Validates: Requirements REQ-A4.9, REQ-A4.3**

---

### Property 8: Adaptive threshold bounds

*For any* queue depth above the configured maximum, the HIGH threshold must be greater than or equal to its current value (never lowered when queue is full). For any queue depth below the configured minimum, the HIGH threshold must be less than or equal to its current value (never raised when queue is empty). In both cases the threshold must remain within [floor, ceiling].

**Validates: Requirements REQ-A5.2, REQ-A5.3**

---

### Property 9: Threshold override expiry reversion

*For any* manual threshold override with a non-null expiry timestamp, once the current time exceeds the expiry, the ThresholdController must revert to the adaptive algorithm value and log a reversion event.

**Validates: Requirements REQ-A5.6**

---

### Property 10: Case deduplication

*For any* user who already has an open Case, submitting a new HIGH-risk prediction for that user must append the prediction to the existing Case rather than creating a new Case. The total number of open Cases for that user must remain 1.

**Validates: Requirements REQ-A6.11**

---

### Property 11: Case status transition validity

*For any* Case, only the transitions `open → in_review`, `open → escalated`, `in_review → resolved`, and `escalated → resolved` are valid. Any attempt to apply an invalid transition must be rejected.

**Validates: Requirements REQ-A6.3**

---

### Property 12: Alert rate limiting

*For any* channel with a configured hourly rate limit N, the number of alerts dispatched to that channel within any rolling 60-minute window must not exceed N. Excess alerts must be queued, not dropped.

**Validates: Requirements REQ-A8.4, REQ-A8.5**

---

### Property 13: Alert suppression within cooldown

*For any* user_id that has triggered an alert within the configured cooldown window, a subsequent alert for the same user_id must be suppressed (not dispatched) until the cooldown expires.

**Validates: Requirements REQ-A8.9**

---

### Property 14: Feature store cold start

*For any* user_id with no stored features, `GET /features/{user_id}` must return a zero-vector with `cold_start: true` and must not return HTTP 404.

**Validates: Requirements REQ-A9.7**

---

### Property 15: Feature store schema versioning round-trip

*For any* feature vector written under schema version V, retrieving that vector with schema version V must return an equivalent feature vector regardless of whether a newer schema version has been deployed.

**Validates: Requirements REQ-A9.4**

---

### Property 16: Monitoring F1 degradation detection

*For any* model whose rolling F1 drops more than 0.05 below its baseline F1, the MonitoringSystem must emit a `model_degradation` alert. For any model whose rolling F1 is within 0.05 of baseline, no degradation alert must be emitted.

**Validates: Requirements REQ-A10.4**

---

### Property 17: Copilot PII exclusion

*For any* LLM-generated explanation or suggestion, the response text must not contain any string matching patterns for full names, national IDs, or phone numbers; all user references must use `user_id` only.

**Validates: Requirements REQ-A7.5**

---

## Error Handling

| Scenario | HTTP Status | Behavior |
|---|---|---|
| Broker unavailable (streaming) | — | Buffer locally; resume on reconnect; annotate predictions with `feature_degraded` if Feature_Store also down |
| Feature_Store unavailable | 200 + flag | Fall back to reduced feature set; set `feature_degraded: true` in prediction |
| Graph embeddings stale (>10 min) | 200 + warning | Return last known embedding with `embedding_stale: true` flag |
| Sequence model: insufficient history | 200 + flag | Return `insufficient_history: true`; skip sequence score from ensemble |
| LLM provider unavailable | 200 (fallback) | Return structured fallback with raw SHAP features and template explanation |
| Case not found | 404 | Standard 404 with `case_id` in error body |
| Invalid case status transition | 422 | Return 422 with allowed transitions listed |
| Alert dispatch failure (3 retries) | — | Mark alert `failed`; log error; surface in dashboard |
| Threshold override expired | — | Revert to adaptive; log reversion event |
| Redis unavailable (Feature_Store) | — | Fall through to PostgreSQL cold storage; log degraded mode |
| Cluster recompute in progress | 200 + flag | Return last known cluster membership with `cluster_stale: true` |

---

## Testing Strategy

### Dual Testing Approach

Both unit tests and property-based tests are required and complementary:
- Unit tests: specific examples, integration points, edge cases, error conditions
- Property tests: universal correctness across all inputs via Hypothesis

### Unit Tests

Focus on:
- StreamConsumer health endpoint returns correct broker status fields
- GraphEngine subgraph returns valid node-link JSON structure
- IdentityClusterer merge logic on known overlapping clusters
- ThresholdController adaptive logic at boundary queue depths
- CaseManager status transition rejection for invalid transitions
- AICopilot fallback response when LLM provider returns 500
- AlertRouter rate limit enforcement with known alert sequences
- FeatureStore cold-start returns zero-vector with flag

### Property-Based Tests

Use **Hypothesis** (Python) for all property tests. Each test runs a minimum of **100 iterations**.

Tag format: `# Feature: aml-advanced-features, Property {N}: {property_text}`

| Property | Test Description | Hypothesis Strategy |
|---|---|---|
| P1: Streaming enrichment completeness | Generate random events + feature store states (available/unavailable), assert enrichment correctness | `st.booleans()` for availability, `st.builds(TransactionEvent)` |
| P2: Streaming audit parity | Generate random scored events, assert audit log contains matching record | `st.lists(st.builds(TransactionEvent))` |
| P3: Graph neighborhood elevation | Generate random neighborhoods with varying HIGH fractions, assert elevation logic | `st.floats(0,1)` for fractions, `st.integers` for counts |
| P4: Ensemble score bounds | Generate random (graph_score, xgb_score, weight) triples, assert result in [0,1] | `st.floats(0,1)` for scores and weights |
| P5: Cluster merge transitivity | Generate random overlapping cluster sets, assert union-find produces correct merged clusters | `st.lists(st.sets(st.text()))` |
| P6: Cluster risk score is max | Generate random member risk scores, assert cluster_risk_score = max | `st.lists(st.floats(0,1), min_size=1)` |
| P7: Sequence score range and flag | Generate users with varying transaction counts, assert score range and flag correctness | `st.integers(0, 100)` for tx count |
| P8: Adaptive threshold bounds | Generate random queue depths and threshold states, assert bounds invariant | `st.integers(0, 1000)` for queue depth |
| P9: Threshold override expiry | Generate overrides with past/future expiry timestamps, assert reversion behavior | `st.datetimes()` |
| P10: Case deduplication | Generate sequences of HIGH predictions for same user, assert single open case | `st.lists(st.builds(PredictionResult))` |
| P11: Case status transition validity | Generate random status transition attempts, assert only valid ones succeed | `st.sampled_from(CaseStatus)` |
| P12: Alert rate limiting | Generate alert sequences with timestamps, assert per-channel count ≤ limit in any 60-min window | `st.lists(st.datetimes())` |
| P13: Alert suppression | Generate alert sequences for same user_id within cooldown, assert suppression | `st.datetimes()`, `st.text()` for user_id |
| P14: Feature store cold start | Generate unknown user_ids, assert zero-vector + cold_start flag | `st.text()` for user_ids |
| P15: Feature store schema versioning | Generate feature vectors at version V, deploy V+1, assert V retrieval unchanged | `st.builds(FeatureVector)` |
| P16: Monitoring F1 degradation | Generate (baseline_f1, rolling_f1) pairs, assert alert emitted iff drop > 0.05 | `st.floats(0,1)` |
| P17: Copilot PII exclusion | Generate case details with PII-like strings, assert LLM response contains none | `st.builds(Case)` with PII fields |

### Property Test Configuration

```python
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(...)
def test_property_N_description(...):
    # Feature: aml-advanced-features, Property N: <property text>
    ...
```

### Test File Layout

```
tests/
├── unit/
│   ├── test_stream_consumer.py
│   ├── test_graph_engine.py
│   ├── test_identity_clusterer.py
│   ├── test_threshold_controller.py
│   ├── test_case_manager.py
│   ├── test_ai_copilot.py
│   ├── test_alert_router.py
│   ├── test_feature_store.py
│   └── test_monitoring_system.py
└── property/
    ├── test_pbt_streaming.py       # P1, P2
    ├── test_pbt_graph.py           # P3, P4
    ├── test_pbt_clustering.py      # P5, P6
    ├── test_pbt_sequence.py        # P7
    ├── test_pbt_thresholds.py      # P8, P9
    ├── test_pbt_cases.py           # P10, P11
    ├── test_pbt_alerts.py          # P12, P13
    ├── test_pbt_feature_store.py   # P14, P15
    └── test_pbt_monitoring.py      # P16, P17
```
