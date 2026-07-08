# =============================================================================
# CREDIT CARD FRAUD DETECTION — FastAPI Prediction Service
# Endpoint: POST /predict  |  GET /health  |  GET /model-info
# Author: Chahat Thakur | github.com/chahatthakur24
# =============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
import numpy as np
import pandas as pd
import joblib
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# App setup
# =============================================================================
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="""
    ## Fraud Detection ML Service

    Predicts whether a credit card transaction is **fraudulent or legitimate**
    using an ensemble of trained ML models (Logistic Regression, Random Forest, XGBoost).

    ### Key Endpoints
    - `POST /predict` — Score a single transaction
    - `POST /predict/batch` — Score multiple transactions at once
    - `GET /model-info` — Current model metrics and thresholds
    - `GET /health` — Service health check
    """,
    version="1.0.0",
    contact={"name": "Chahat Thakur", "url": "https://github.com/chahatthakur24"}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# =============================================================================
# Load models & scalers at startup
# =============================================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def load_artifact(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        logger.warning(f"Artifact not found: {path}")
        return None
    return joblib.load(path)

models = {}
scalers = {}
best_threshold = 0.5

@app.on_event("startup")
async def load_models():
    global models, scalers, best_threshold
    logger.info("Loading models...")

    models["logistic"] = load_artifact("logistic_regression.pkl")
    models["rf"]       = load_artifact("random_forest.pkl")
    models["xgboost"]  = load_artifact("xgboost.pkl")
    scalers["robust"]  = load_artifact("robust_scaler.pkl")
    scalers["standard"]= load_artifact("standard_scaler.pkl")
    thresh = load_artifact("best_threshold.pkl")
    if thresh is not None:
        best_threshold = thresh

    loaded = [k for k, v in models.items() if v is not None]
    logger.info(f"Models loaded: {loaded}")
    logger.info(f"Best threshold: {best_threshold:.2f}")


# =============================================================================
# Schemas
# =============================================================================
class TransactionInput(BaseModel):
    """All 28 PCA features + Time + Amount for one transaction."""
    Time   : float = Field(..., description="Seconds elapsed since first transaction in dataset")
    Amount : float = Field(..., ge=0, description="Transaction amount in USD")
    V1  : float; V2  : float; V3  : float; V4  : float
    V5  : float; V6  : float; V7  : float; V8  : float
    V9  : float; V10 : float; V11 : float; V12 : float
    V13 : float; V14 : float; V15 : float; V16 : float
    V17 : float; V18 : float; V19 : float; V20 : float
    V21 : float; V22 : float; V23 : float; V24 : float
    V25 : float; V26 : float; V27 : float; V28 : float

    model: Optional[Literal["logistic", "rf", "xgboost", "ensemble"]] = "ensemble"

    @validator("Amount")
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Amount must be non-negative")
        return v

    class Config:
        schema_extra = {
            "example": {
                "Time": 406, "Amount": 149.62,
                "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
                "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
                "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
                "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
                "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
                "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
                "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
                "model": "ensemble"
            }
        }


class BatchInput(BaseModel):
    transactions: list[TransactionInput]


class PredictionResult(BaseModel):
    transaction_id    : Optional[int]
    fraud_probability : float
    is_fraud          : bool
    risk_tier         : Literal["Low", "Medium", "High"]
    risk_score        : int
    confidence        : Literal["Low", "Medium", "High"]
    model_used        : str
    processing_ms     : float
    explanation       : dict


# =============================================================================
# Core prediction logic
# =============================================================================
def engineer_features(raw: dict) -> np.ndarray:
    """Apply same feature engineering as training pipeline."""
    hour        = (raw["Time"] % 86400) // 3600
    is_night    = 1 if hour < 5 or hour > 22 else 0
    log_amount  = np.log1p(raw["Amount"])
    high_amount = 1 if raw["Amount"] > 1000 else 0
    amt_bucket  = min(int(raw["Amount"] // 200), 5)

    v14 = raw["V14"]; v17 = raw["V17"]; v12 = raw["V12"]
    v14_x_v17  = v14 * v17
    v14_x_v12  = v14 * v12
    v14_squared= v14 ** 2

    # Rule-based risk score (as feature)
    rs = 0
    if raw["Amount"] > 1000: rs += 30
    elif raw["Amount"] > 500: rs += 15
    if is_night: rs += 20
    if v14 < -5: rs += 25
    if v17 < -5: rs += 15
    if v12 < -5: rs += 10
    risk_score_val = min(rs, 100)
    risk_tier_val  = 0 if risk_score_val <= 30 else (1 if risk_score_val <= 60 else 2)

    v_features = [raw[f"V{i}"] for i in range(1, 29)]

    feature_vector = (
        v_features +
        [hour, is_night, raw["Time"],
         log_amount, high_amount, amt_bucket,
         v14_x_v17, v14_x_v12, v14_squared,
         risk_score_val, risk_tier_val]
    )
    return np.array(feature_vector).reshape(1, -1), risk_score_val


def get_risk_tier(prob: float) -> str:
    if prob >= 0.7: return "High"
    if prob >= 0.35: return "Medium"
    return "Low"


def get_confidence(prob: float) -> str:
    distance = abs(prob - 0.5)
    if distance >= 0.35: return "High"
    if distance >= 0.15: return "Medium"
    return "Low"


def predict_single(txn: TransactionInput, txn_id: int = 0) -> PredictionResult:
    t0 = time.time()
    raw = txn.dict()
    feat_vec, rule_score = engineer_features(raw)

    chosen_model = raw.get("model", "ensemble")
    probas = {}

    for name, model in models.items():
        if model is None:
            continue
        try:
            probas[name] = float(model.predict_proba(feat_vec)[0, 1])
        except Exception as e:
            logger.error(f"Model {name} error: {e}")

    if not probas:
        raise HTTPException(status_code=503, detail="No models available")

    if chosen_model == "ensemble" or chosen_model not in probas:
        weights = {"xgboost": 0.5, "rf": 0.35, "logistic": 0.15}
        fraud_prob = sum(probas.get(k, 0) * w for k, w in weights.items()) / \
                     sum(w for k, w in weights.items() if k in probas)
        model_used = "Ensemble (XGB 50% + RF 35% + LR 15%)"
    else:
        fraud_prob = probas[chosen_model]
        model_used = chosen_model

    is_fraud   = fraud_prob >= best_threshold
    risk_tier  = get_risk_tier(fraud_prob)
    confidence = get_confidence(fraud_prob)
    proc_ms    = (time.time() - t0) * 1000

    explanation = {
        "individual_model_probabilities": {k: round(v, 4) for k, v in probas.items()},
        "rule_based_risk_score": rule_score,
        "threshold_used": best_threshold,
        "key_signals": {
            "high_amount": raw["Amount"] > 1000,
            "night_transaction": (raw["Time"] % 86400) // 3600 < 5,
            "v14_anomaly": raw["V14"] < -5,
            "v17_anomaly": raw["V17"] < -5,
        }
    }

    return PredictionResult(
        transaction_id    = txn_id,
        fraud_probability = round(fraud_prob, 4),
        is_fraud          = is_fraud,
        risk_tier         = risk_tier,
        risk_score        = rule_score,
        confidence        = confidence,
        model_used        = model_used,
        processing_ms     = round(proc_ms, 2),
        explanation       = explanation
    )


# =============================================================================
# Routes
# =============================================================================
@app.get("/health", tags=["System"])
def health_check():
    loaded = {k: v is not None for k, v in models.items()}
    return {
        "status": "healthy" if any(loaded.values()) else "degraded",
        "models_loaded": loaded,
        "threshold": best_threshold
    }


@app.get("/model-info", tags=["System"])
def model_info():
    return {
        "models": ["Logistic Regression", "Random Forest (300 trees)", "XGBoost (500 rounds)"],
        "dataset": "ULB Credit Card Fraud — 284,807 transactions",
        "fraud_rate": "0.17%",
        "imbalance_handling": "SMOTE (sampling_strategy=0.15) + class_weight",
        "validation_metrics": {
            "logistic_regression": {"roc_auc": 0.972, "pr_auc": 0.721, "f1": 0.782},
            "random_forest":       {"roc_auc": 0.979, "pr_auc": 0.859, "f1": 0.851},
            "xgboost":             {"roc_auc": 0.983, "pr_auc": 0.878, "f1": 0.869},
        },
        "ensemble_strategy": "Weighted average: XGB 50% + RF 35% + LR 15%",
        "features_engineered": [
            "Log_Amount", "Is_Night", "High_Amount", "Amt_Bucket",
            "V14_x_V17", "V14_x_V12", "V14_squared",
            "Risk_Score", "Risk_Tier"
        ],
        "author": "Chahat Thakur",
        "github": "github.com/chahatthakur24"
    }


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(txn: TransactionInput):
    """
    Predict fraud probability for a single transaction.

    Returns fraud probability, risk tier (Low/Medium/High),
    model confidence, and key signal explanations.
    """
    return predict_single(txn, txn_id=0)


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(batch: BatchInput):
    """
    Score up to 100 transactions in one call.
    Returns list of predictions with same schema as /predict.
    """
    if len(batch.transactions) > 100:
        raise HTTPException(status_code=400, detail="Max 100 transactions per batch")

    results = []
    for i, txn in enumerate(batch.transactions):
        try:
            results.append(predict_single(txn, txn_id=i).dict())
        except Exception as e:
            results.append({"transaction_id": i, "error": str(e)})

    fraud_count = sum(1 for r in results if r.get("is_fraud"))
    return {
        "total": len(results),
        "fraud_detected": fraud_count,
        "fraud_rate_pct": round(fraud_count / len(results) * 100, 2),
        "predictions": results
    }
