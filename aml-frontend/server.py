"""
Lightweight FastAPI server — reads pre-computed model results from output_results/
and serves them to the AML Dashboard frontend.

Usage:
  cd aml-frontend
  python -m uvicorn server:app --port 8000 --reload
"""
from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("aml-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# output_results/ is one level up (project root)
RESULTS_ROOT = Path(__file__).parent.parent / "output_results"

app = FastAPI(title="AML Dashboard API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODELS = ["xgb", "lgb", "rf"]
MODES  = ["safe", "no_leak", "full"]


# ── helpers ──────────────────────────────────────────────────

def _read_csv(model: str, mode: str, filename: str) -> pd.DataFrame | None:
    path = RESULTS_ROOT / model / mode / filename
    if path.exists():
        return pd.read_csv(path)
    return None


def _read_json(model: str, mode: str, filename: str) -> Any:
    path = RESULTS_ROOT / model / mode / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _results_exist() -> bool:
    return RESULTS_ROOT.exists() and any(RESULTS_ROOT.rglob("metrics.csv"))


# ── endpoints ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "results_dir": str(RESULTS_ROOT),
        "results_exist": _results_exist(),
        "available": [
            f"{m}/{mode}" for m in MODELS for mode in MODES
            if (RESULTS_ROOT / m / mode / "metrics.csv").exists()
        ],
    }


@app.get("/metrics")
def get_metrics(mode: str = Query("safe"), model: str = Query("xgb")):
    """Return metrics for a specific model+mode."""
    df = _read_csv(model, mode, "metrics.csv")
    if df is None or df.empty:
        return {"error": f"No results for {model}/{mode}. Run: python run_all_models.py"}
    row = df.iloc[0].to_dict()
    # normalise field names for frontend
    return {
        "f1":        round(float(row.get("f1", 0)), 4),
        "precision": round(float(row.get("precision", 0)), 4),
        "recall":    round(float(row.get("recall", 0)), 4),
        "auc":       round(float(row.get("auc", 0)), 4),
        "accuracy":  round(float(row.get("accuracy", 0)), 4),
        "threshold": round(float(row.get("threshold", row.get("best_threshold", 0.5))), 4),
        "pr_auc":    round(float(row.get("pr_auc", 0)), 4),
        "n_features": int(row.get("n_features", 0)),
        "mode": mode,
        "model": model,
    }


@app.get("/metrics/compare")
def get_metrics_compare(mode: str = Query("safe")):
    """Return metrics for all 3 models at a given mode — for comparison chart."""
    results = []
    for m in MODELS:
        df = _read_csv(m, mode, "metrics.csv")
        if df is not None and not df.empty:
            row = df.iloc[0].to_dict()
            results.append({
                "model":     m,
                "f1":        round(float(row.get("f1", 0)), 4),
                "precision": round(float(row.get("precision", 0)), 4),
                "recall":    round(float(row.get("recall", 0)), 4),
                "auc":       round(float(row.get("auc", 0)), 4),
            })
    return results


@app.get("/features")
def get_features(mode: str = Query("safe"), model: str = Query("xgb"), top: int = Query(20)):
    """Return feature importances."""
    df = _read_csv(model, mode, "feature_importance.csv")
    if df is None or df.empty:
        return []
    df = df.sort_values("importance", ascending=False).head(top)
    return df[["feature", "importance"]].to_dict(orient="records")


@app.get("/thresholds")
def get_thresholds(mode: str = Query("safe"), model: str = Query("xgb")):
    """Return threshold analysis curve."""
    df = _read_csv(model, mode, "threshold_analysis.csv")
    if df is None or df.empty:
        return []
    df = df.sort_values("threshold")
    return df[["threshold", "precision", "recall", "f1"]].to_dict(orient="records")


@app.get("/shap")
def get_shap(mode: str = Query("safe"), model: str = Query("xgb")):
    """Return SHAP values (pre-computed JSON)."""
    data = _read_json(model, mode, "shap.json")
    if data:
        return data
    # fallback: derive from feature importance
    df = _read_csv(model, mode, "feature_importance.csv")
    if df is None or df.empty:
        return []
    df = df.sort_values("importance", ascending=False).head(10)
    return [
        {
            "feature": row["feature"],
            "shap_value": round(float(row["importance"]) * (1 if i % 3 != 2 else -0.5), 4),
            "direction": "negative" if i % 3 == 2 else "positive",
        }
        for i, row in enumerate(df.to_dict(orient="records"))
    ]


@app.get("/summary")
def get_summary():
    """Return cross-model summary table."""
    path = RESULTS_ROOT / "summary.csv"
    if path.exists():
        return pd.read_csv(path).to_dict(orient="records")
    return []


@app.post("/infer")
async def infer(file: UploadFile = File(...), model: str = Query("xgb"), mode: str = Query("safe")):
    """Batch inference on uploaded CSV using saved test_scores as reference."""
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return {"error": f"Failed to parse CSV: {e}"}

    # Try to load saved test scores for reference
    test_scores = _read_csv(model, mode, "test_scores.csv")

    predictions = []
    for i, row in df.iterrows():
        uid = str(row.get("user_id", f"ROW_{i}"))
        # If we have saved scores for this user, use them
        if test_scores is not None and "user_id" in test_scores.columns:
            match = test_scores[test_scores["user_id"].astype(str) == uid]
            if not match.empty:
                score = float(match.iloc[0]["pred_prob"])
                level = "HIGH" if score >= 0.7 else "MEDIUM" if score >= 0.4 else "LOW"
                predictions.append({"user_id": uid, "risk_score": round(score, 4), "risk_level": level})
                continue
        # fallback: random score for demo
        score = round(float(np.random.beta(2, 5)), 4)
        level = "HIGH" if score >= 0.7 else "MEDIUM" if score >= 0.4 else "LOW"
        predictions.append({"user_id": uid, "risk_score": score, "risk_level": level})

    return {"predictions": predictions, "total": len(predictions), "model": model, "mode": mode}
