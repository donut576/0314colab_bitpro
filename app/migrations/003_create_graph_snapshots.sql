CREATE TABLE IF NOT EXISTS graph_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    snapshot_date   DATE NOT NULL,
    node_count      INT NOT NULL,
    edge_count      INT NOT NULL,
    snapshot_data   JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_graph_snapshots_date ON graph_snapshots (snapshot_date DESC);
