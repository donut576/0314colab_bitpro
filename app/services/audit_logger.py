"""AuditLogger — async PostgreSQL audit trail for predictions."""

from __future__ import annotations


class AuditLogger:
    """Placeholder — implemented in Task 8."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self.db_connected: bool = False

    async def log_prediction(self, record) -> None:
        raise NotImplementedError("AuditLogger.log_prediction not yet implemented")

    async def query_logs(self, filters) -> list:
        raise NotImplementedError("AuditLogger.query_logs not yet implemented")

    async def export_csv(self, filters) -> bytes:
        raise NotImplementedError("AuditLogger.export_csv not yet implemented")
