CREATE TABLE IF NOT EXISTS threshold_history (
    id              BIGSERIAL PRIMARY KEY,
    threshold_type  VARCHAR(16) NOT NULL CHECK (threshold_type IN ('HIGH', 'MEDIUM')),
    old_value       DOUBLE PRECISION NOT NULL,
    new_value       DOUBLE PRECISION NOT NULL,
    reason          TEXT NOT NULL,
    operator        VARCHAR(64),
    expiry          TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_threshold_history_created_at ON threshold_history (created_at DESC);
