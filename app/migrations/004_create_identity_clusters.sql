CREATE TABLE IF NOT EXISTS identity_clusters (
    cluster_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    member_user_ids     TEXT[] NOT NULL,
    shared_signals      JSONB NOT NULL DEFAULT '{}',
    cluster_risk_score  DOUBLE PRECISION NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_clusters_member_ids ON identity_clusters USING GIN (member_user_ids);
CREATE INDEX IF NOT EXISTS idx_clusters_risk_score ON identity_clusters (cluster_risk_score DESC);
