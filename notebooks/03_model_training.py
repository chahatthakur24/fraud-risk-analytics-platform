# =============================================================================
# CREDIT CARD FRAUD DETECTION — 03: ML Model Training & Comparison
# Models: Logistic Regression | Random Forest | XGBoost
# Author: Chahat Thakur | github.com/chahatthakur24
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, time, joblib
warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score
)

from notebooks.feature_engineering import load_and_engineer  # type: ignore
# ── OR run standalone: ────────────────────────────────────────────────────────
# from notebooks.02_feature_engineering import load_and_engineer, preprocess
# df = load_and_engineer(); splits = preprocess(df)

sns.set_theme(style="whitegrid")

# =============================================================================
# HELPER: Evaluate any classifier
# =============================================================================
def evaluate_model(name, model, X_val, y_val, threshold=0.5):
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= threshold).astype(int)

    roc_auc  = roc_auc_score(y_val, proba)
    pr_auc   = average_precision_score(y_val, proba)
    f1       = f1_score(y_val, preds)
    prec     = precision_score(y_val, preds)
    rec      = recall_score(y_val, preds)

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"  PR-AUC   : {pr_auc:.4f}   ← key metric for imbalanced data")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print()
    print(classification_report(y_val, preds, target_names=["Legit", "Fraud"]))

    return {
        "model": name, "roc_auc": roc_auc, "pr_auc": pr_auc,
        "f1": f1, "precision": prec, "recall": rec,
        "proba": proba, "preds": preds
    }


# =============================================================================
# 1. LOGISTIC REGRESSION  (baseline)
# =============================================================================
def train_logistic(X_train, y_train):
    print("\nTraining Logistic Regression...")
    t0 = time.time()
    lr = LogisticRegression(
        C=0.01,
        class_weight="balanced",
        solver="saga",
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    lr.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    joblib.dump(lr, "models/logistic_regression.pkl")
    return lr


# =============================================================================
# 2. RANDOM FOREST
# =============================================================================
def train_random_forest(X_train, y_train):
    print("\nTraining Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    joblib.dump(rf, "models/random_forest.pkl")
    return rf


# =============================================================================
# 3. XGBOOST
# =============================================================================
def train_xgboost(X_train, y_train):
    print("\nTraining XGBoost...")
    t0 = time.time()

    # scale_pos_weight handles class imbalance natively
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )
    print(f"  Done in {time.time()-t0:.1f}s")
    joblib.dump(xgb, "models/xgboost.pkl")
    return xgb


# =============================================================================
# 4. THRESHOLD TUNING (for best model — XGBoost)
# =============================================================================
def tune_threshold(model, X_val, y_val, model_name="XGBoost"):
    """Find threshold that maximises F1 on validation set."""
    proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        scores.append({
            "threshold": t,
            "f1": f1_score(y_val, preds),
            "precision": precision_score(y_val, preds),
            "recall": recall_score(y_val, preds)
        })
    df_t = pd.DataFrame(scores)
    best = df_t.loc[df_t["f1"].idxmax()]

    print(f"\nBest threshold for {model_name}:")
    print(f"  Threshold : {best['threshold']:.2f}")
    print(f"  F1        : {best['f1']:.4f}")
    print(f"  Precision : {best['precision']:.4f}")
    print(f"  Recall    : {best['recall']:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(df_t["threshold"], df_t["f1"],       label="F1",        color="#2a78d6")
    plt.plot(df_t["threshold"], df_t["precision"], label="Precision", color="#1baf7a")
    plt.plot(df_t["threshold"], df_t["recall"],    label="Recall",    color="#e24b4a")
    plt.axvline(best["threshold"], color="gray", linestyle="--", alpha=0.7,
                label=f"Best: {best['threshold']:.2f}")
    plt.title(f"Threshold Tuning — {model_name}", fontweight="bold")
    plt.xlabel("Classification Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/06_threshold_tuning.png")
    plt.show()

    return float(best["threshold"])


# =============================================================================
# 5. COMPARISON PLOTS
# =============================================================================
def plot_comparison(results, X_val, y_val, models_dict):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── ROC curves ────────────────────────────────────────────────────────────
    colors = {"Logistic Regression": "#2a78d6",
              "Random Forest": "#1baf7a",
              "XGBoost": "#e24b4a"}
    for name, model in models_dict.items():
        proba = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, proba)
        auc = roc_auc_score(y_val, proba)
        axes[0].plot(fpr, tpr, label=f"{name} ({auc:.3f})", color=colors[name], lw=2)
    axes[0].plot([0,1],[0,1], "k--", alpha=0.4)
    axes[0].set_title("ROC Curves", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=9)

    # ── PR curves ─────────────────────────────────────────────────────────────
    for name, model in models_dict.items():
        proba = model.predict_proba(X_val)[:, 1]
        prec, rec, _ = precision_recall_curve(y_val, proba)
        pr_auc = average_precision_score(y_val, proba)
        axes[1].plot(rec, prec, label=f"{name} (AP={pr_auc:.3f})",
                     color=colors[name], lw=2)
    axes[1].set_title("Precision-Recall Curves", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=9)

    # ── Metric bar chart ──────────────────────────────────────────────────────
    df_r = pd.DataFrame(results)[["model", "roc_auc", "pr_auc", "f1", "precision", "recall"]]
    df_r = df_r.set_index("model")
    df_r.plot(kind="bar", ax=axes[2], colormap="tab10", edgecolor="none", width=0.6)
    axes[2].set_title("Model Comparison", fontweight="bold")
    axes[2].set_ylabel("Score")
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=9)
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig("reports/07_model_comparison.png")
    plt.show()


def plot_confusion_matrices(results, y_val):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_val, res["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legit", "Fraud"],
                    yticklabels=["Legit", "Fraud"],
                    cbar=False, linewidths=0.5)
        ax.set_title(res["model"], fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig("reports/08_confusion_matrices.png")
    plt.show()


def plot_feature_importance(rf, xgb, feature_names, top_n=15):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, model, name in [(axes[0], rf, "Random Forest"),
                             (axes[1], xgb, "XGBoost")]:
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        ax.barh([feature_names[i] for i in idx][::-1],
                importances[idx][::-1],
                color="#2a78d6", edgecolor="none")
        ax.set_title(f"Feature Importance — {name}", fontweight="bold")
        ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("reports/09_feature_importance.png")
    plt.show()


# =============================================================================
# MAIN — run all models
# =============================================================================
if __name__ == "__main__":
    # Load & preprocess (assumes 02_feature_engineering.py ran)
    from notebooks.feature_engineering import load_and_engineer, preprocess  # type: ignore
    df = load_and_engineer()
    X_train, X_val, X_test, y_train, y_val, y_test, feat_names = preprocess(df)

    # Train
    lr  = train_logistic(X_train, y_train)
    rf  = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    # Evaluate on validation set
    models_dict = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb
    }
    results = []
    for name, model in models_dict.items():
        res = evaluate_model(name, model, X_val, y_val, threshold=0.5)
        results.append(res)

    # Plots
    plot_comparison(results, X_val, y_val, models_dict)
    plot_confusion_matrices(results, y_val)
    plot_feature_importance(rf, xgb, feat_names)

    # Threshold tuning on best model (XGBoost)
    best_threshold = tune_threshold(xgb, X_val, y_val, "XGBoost")

    # Final test-set evaluation of XGBoost with tuned threshold
    print("\n" + "="*55)
    print("FINAL TEST SET EVALUATION — XGBoost (tuned threshold)")
    print("="*55)
    res_test = evaluate_model("XGBoost (tuned)", xgb, X_test, y_test,
                              threshold=best_threshold)

    # Save best threshold
    joblib.dump(best_threshold, "models/best_threshold.pkl")
    print(f"\nBest threshold saved: {best_threshold:.2f}")
    print("All models saved to models/")
    print("All plots saved to reports/")
