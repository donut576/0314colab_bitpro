CREATE TABLE IF NOT EXISTS cases (
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

CREATE TABLE IF NOT EXISTS case_audit_trail (
    id              BIGSERIAL PRIMARY KEY,
    case_id         UUID NOT NULL REFERENCES cases(case_id),
    analyst_id      VARCHAR(64) NOT NULL,
    action          VARCHAR(64) NOT NULL,
    old_status      VARCHAR(32),
    new_status      VARCHAR(32),
    note            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cases_user_id      ON cases (user_id);
CREATE INDEX IF NOT EXISTS idx_cases_status       ON cases (status);
CREATE INDEX IF NOT EXISTS idx_cases_created_at   ON cases (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_case_audit_case_id ON case_audit_trail (case_id);
