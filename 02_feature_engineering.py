# =============================================================================
# CREDIT CARD FRAUD DETECTION — 02: Feature Engineering & Preprocessing
# Author: Chahat Thakur | github.com/chahatthakur24
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

os.makedirs("models", exist_ok=True)

def load_and_engineer(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load raw data and engineer new features."""
    df = pd.read_csv(path)

    # ── Time features ─────────────────────────────────────────────────────────
    df["Hour"]        = (df["Time"] % 86400) // 3600
    df["Is_Night"]    = df["Hour"].apply(lambda h: 1 if h < 5 or h > 22 else 0)
    df["Time_Since_First"] = df["Time"]  # raw seconds kept for model

    # ── Amount features ───────────────────────────────────────────────────────
    df["Log_Amount"]  = np.log1p(df["Amount"])
    df["High_Amount"] = (df["Amount"] > 1000).astype(int)
    df["Amt_Bucket"]  = pd.cut(df["Amount"],
                               bins=[0, 10, 50, 100, 500, 1000, np.inf],
                               labels=[0, 1, 2, 3, 4, 5]).astype(int)

    # ── Interaction features (top correlated PCA components) ──────────────────
    df["V14_x_V17"]   = df["V14"] * df["V17"]
    df["V14_x_V12"]   = df["V14"] * df["V12"]
    df["V14_squared"] = df["V14"] ** 2

    # ── Rule-based risk score (from Phase 1, retained as a feature) ───────────
    def risk_score(row):
        s = 0
        if row["Amount"] > 1000: s += 30
        elif row["Amount"] > 500: s += 15
        if row["Is_Night"]: s += 20
        if row["V14"] < -5: s += 25
        if row["V17"] < -5: s += 15
        if row["V12"] < -5: s += 10
        return min(s, 100)

    df["Risk_Score"] = df.apply(risk_score, axis=1)
    df["Risk_Tier"]  = pd.cut(df["Risk_Score"],
                               bins=[-1, 30, 60, 100],
                               labels=[0, 1, 2]).astype(int)

    print(f"Feature engineering done. Shape: {df.shape}")
    return df


def preprocess(df: pd.DataFrame, apply_smote: bool = True):
    """Scale features, split, optionally apply SMOTE."""

    drop_cols = ["Class", "Time"]
    X = df.drop(columns=drop_cols)
    y = df["Class"]

    # Scale Amount and Log_Amount with RobustScaler (outlier-robust)
    robust_cols = ["Amount", "Log_Amount", "Time_Since_First"]
    rs = RobustScaler()
    X[robust_cols] = rs.fit_transform(X[robust_cols])

    # Standard scale all V features
    v_cols = [c for c in X.columns if c.startswith("V")]
    ss = StandardScaler()
    X[v_cols] = ss.fit_transform(X[v_cols])

    # Train / validation / test split — stratified
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    print(f"Train : {X_train.shape}  |  fraud: {y_train.sum()}")
    print(f"Val   : {X_val.shape}    |  fraud: {y_val.sum()}")
    print(f"Test  : {X_test.shape}   |  fraud: {y_test.sum()}")

    # ── SMOTE on training set only ────────────────────────────────────────────
    if apply_smote:
        sm = SMOTE(sampling_strategy=0.15, random_state=42, k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"\nAfter SMOTE — Train: {X_train.shape}  |  fraud: {y_train.sum()}")

    # Save scalers
    joblib.dump(rs, "models/robust_scaler.pkl")
    joblib.dump(ss, "models/standard_scaler.pkl")
    print("Scalers saved to models/")

    return X_train, X_val, X_test, y_train, y_val, y_test, list(X.columns)


if __name__ == "__main__":
    df = load_and_engineer()
    splits = preprocess(df, apply_smote=True)
    print("\nPreprocessing complete.")
