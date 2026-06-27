# =============================================================================
# CREDIT CARD FRAUD DETECTION — API Tests
# Run: pytest tests/test_api.py -v
# Author: Chahat Thakur | github.com/chahatthakur24
# =============================================================================

import pytest
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.main import app

client = TestClient(app)

# ── Sample transaction payloads ───────────────────────────────────────────────
LEGIT_TXN = {
    "Time": 5000, "Amount": 12.50,
    "V1": 0.2, "V2": 0.1, "V3": 0.5, "V4": 0.3,
    "V5": 0.1, "V6": 0.2, "V7": 0.0, "V8": 0.1,
    "V9": 0.2, "V10": 0.1, "V11": 0.3, "V12": 0.2,
    "V13": 0.1, "V14": 0.5, "V15": 0.2, "V16": 0.1,
    "V17": 0.3, "V18": 0.2, "V19": 0.1, "V20": 0.0,
    "V21": 0.1, "V22": 0.2, "V23": 0.0, "V24": 0.1,
    "V25": 0.2, "V26": 0.1, "V27": 0.0, "V28": 0.1,
    "model": "ensemble"
}

FRAUD_TXN = {
    "Time": 406, "Amount": 1837.20,
    "V1": -3.04, "V2": -3.16, "V3": 1.09, "V4": 2.29,
    "V5": -1.35, "V6": -1.72, "V7": -2.68, "V8": -0.07,
    "V9": -1.87, "V10": -5.19, "V11": -2.83, "V12": -9.44,
    "V13": -2.05, "V14": -9.48, "V15": -3.05, "V16": -0.72,
    "V17": -8.26, "V18": -4.04, "V19": 0.13, "V20": -0.23,
    "V21": -0.53, "V22": -0.63, "V23": -0.29, "V24": 0.03,
    "V25": 0.66, "V26": -0.37, "V27": -0.15, "V28": -0.08,
    "model": "ensemble"
}


# ── Health check ──────────────────────────────────────────────────────────────
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data


# ── Model info ────────────────────────────────────────────────────────────────
def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "validation_metrics" in data
    assert "xgboost" in data["validation_metrics"]


# ── Predict endpoint schema ───────────────────────────────────────────────────
def test_predict_returns_correct_schema():
    response = client.post("/predict", json=LEGIT_TXN)
    assert response.status_code == 200
    data = response.json()

    required_fields = [
        "fraud_probability", "is_fraud", "risk_tier",
        "risk_score", "confidence", "model_used",
        "processing_ms", "explanation"
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"


# ── Probability range ─────────────────────────────────────────────────────────
def test_fraud_probability_in_range():
    for txn in [LEGIT_TXN, FRAUD_TXN]:
        response = client.post("/predict", json=txn)
        assert response.status_code == 200
        prob = response.json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"


# ── Risk tier values ──────────────────────────────────────────────────────────
def test_risk_tier_valid():
    response = client.post("/predict", json=LEGIT_TXN)
    assert response.json()["risk_tier"] in ["Low", "Medium", "High"]


# ── High-risk transaction flags correctly ─────────────────────────────────────
def test_high_amount_flagged_in_explanation():
    response = client.post("/predict", json=FRAUD_TXN)
    assert response.status_code == 200
    signals = response.json()["explanation"]["key_signals"]
    assert signals["high_amount"] is True
    assert signals["v14_anomaly"] is True
    assert signals["v17_anomaly"] is True


# ── Negative amount rejected ──────────────────────────────────────────────────
def test_negative_amount_rejected():
    bad_txn = LEGIT_TXN.copy()
    bad_txn["Amount"] = -50
    response = client.post("/predict", json=bad_txn)
    assert response.status_code == 422  # Validation error


# ── Batch prediction ──────────────────────────────────────────────────────────
def test_batch_prediction():
    batch = {"transactions": [LEGIT_TXN, FRAUD_TXN, LEGIT_TXN]}
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert "fraud_detected" in data
    assert len(data["predictions"]) == 3


# ── Batch too large rejected ──────────────────────────────────────────────────
def test_batch_too_large_rejected():
    batch = {"transactions": [LEGIT_TXN] * 101}
    response = client.post("/predict/batch", json=batch)
    assert response.status_code == 400


# ── Individual model selection ────────────────────────────────────────────────
@pytest.mark.parametrize("model_name", ["logistic", "rf", "xgboost", "ensemble"])
def test_individual_model_selection(model_name):
    txn = LEGIT_TXN.copy()
    txn["model"] = model_name
    response = client.post("/predict", json=txn)
    assert response.status_code == 200
    assert "fraud_probability" in response.json()
