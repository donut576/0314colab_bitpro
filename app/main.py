"""FastAPI application entry point with lifespan startup/shutdown."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.config import get_settings
from app.routers import audit, drift, explain, model, predict
from app.services.audit_logger import AuditLogger
from app.services.drift_detector import DriftDetector
from app.services.model_loader import ModelLoader
from app.services.predictor import XGBPredictor
from app.services.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state — populated during lifespan startup
# ---------------------------------------------------------------------------

class AppState:
    model_loader: ModelLoader
    predictor: XGBPredictor
    shap_explainer: SHAPExplainer
    drift_detector: DriftDetector
    audit_logger: AuditLogger


state = AppState()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise services on startup; clean up on shutdown."""
    settings = get_settings()

    logger.info("Starting AML Fraud Detection API…")

    # 1. Model loader — downloads artifact from S3
    state.model_loader = ModelLoader()
    try:
        state.model_loader.load_from_s3(settings.model_s3_uri)
        logger.info("Model loaded from %s", settings.model_s3_uri)
    except NotImplementedError:
        logger.warning("ModelLoader not yet implemented — running without model")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load model from S3: %s", exc)

    # 2. Predictor
    state.predictor = XGBPredictor(state.model_loader)

    # 3. SHAP explainer
    state.shap_explainer = SHAPExplainer(state.model_loader)

    # 4. Drift detector
    state.drift_detector = DriftDetector(state.model_loader)

    # 5. Audit logger
    state.audit_logger = AuditLogger(settings.database_url)

    logger.info("All services initialised — API ready")

    yield  # application runs here

    # Shutdown
    logger.info("Shutting down AML Fraud Detection API…")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="AML Fraud Detection API",
        description="Real-time fraud risk scoring, SHAP explainability, drift detection, and audit trail.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(predict.router, tags=["Prediction"])
    app.include_router(explain.router, tags=["Explainability"])
    app.include_router(drift.router, tags=["Drift"])
    app.include_router(audit.router, tags=["Audit"])
    app.include_router(model.router, tags=["Model"])

    return app


app = create_app()
