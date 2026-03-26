CREATE TABLE IF NOT EXISTS feature_vectors (
    user_id         VARCHAR(64) NOT NULL,
    schema_version  VARCHAR(16) NOT NULL,
    features        JSONB NOT NULL DEFAULT '{}',
    groups          TEXT[] NOT NULL DEFAULT '{}',
    last_updated    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, schema_version)
);
CREATE INDEX IF NOT EXISTS idx_feature_vectors_user_id ON feature_vectors (user_id);
CREATE INDEX IF NOT EXISTS idx_feature_vectors_schema_version ON feature_vectors (schema_version);
