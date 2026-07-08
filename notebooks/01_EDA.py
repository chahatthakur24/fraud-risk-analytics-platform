# =============================================================================
# CREDIT CARD FRAUD DETECTION — 01: Exploratory Data Analysis
# Dataset: ULB Machine Learning Group (Kaggle) — 284,807 transactions
# Author: Chahat Thakur | github.com/chahatthakur24
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Plot style ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "figure.figsize": (10, 5)})

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_csv("data/creditcard.csv")

print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"Shape          : {df.shape}")
print(f"Columns        : {list(df.columns)}")
print(f"Missing values : {df.isnull().sum().sum()}")
print(f"Duplicates     : {df.duplicated().sum()}")
print()

# =============================================================================
# 2. CLASS DISTRIBUTION
# =============================================================================
fraud_count = df["Class"].sum()
legit_count = len(df) - fraud_count
fraud_rate  = fraud_count / len(df) * 100

print("CLASS DISTRIBUTION")
print("-" * 40)
print(f"  Legitimate   : {legit_count:,}  ({100 - fraud_rate:.4f}%)")
print(f"  Fraudulent   : {fraud_count:,}    ({fraud_rate:.4f}%)")
print()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart
axes[0].bar(["Legitimate", "Fraudulent"], [legit_count, fraud_count],
            color=["#2a78d6", "#e24b4a"], edgecolor="none", width=0.5)
axes[0].set_title("Transaction Class Distribution", fontweight="bold")
axes[0].set_ylabel("Count")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# Pie chart
axes[1].pie([legit_count, fraud_count],
            labels=["Legitimate", "Fraudulent"],
            colors=["#2a78d6", "#e24b4a"],
            autopct="%1.2f%%", startangle=140,
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("Class Imbalance (Severe)", fontweight="bold")

plt.tight_layout()
plt.savefig("reports/01_class_distribution.png")
plt.show()

# =============================================================================
# 3. AMOUNT ANALYSIS
# =============================================================================
fraud_df = df[df["Class"] == 1]
legit_df = df[df["Class"] == 0]

print("AMOUNT STATISTICS")
print("-" * 40)
print(f"  Avg fraud amount  : ${fraud_df['Amount'].mean():.2f}")
print(f"  Avg legit amount  : ${legit_df['Amount'].mean():.2f}")
print(f"  Max fraud amount  : ${fraud_df['Amount'].max():.2f}")
print(f"  Max legit amount  : ${legit_df['Amount'].max():.2f}")
print(f"  Median fraud amt  : ${fraud_df['Amount'].median():.2f}")
print()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(legit_df["Amount"].clip(upper=500), bins=60,
             color="#2a78d6", alpha=0.7, edgecolor="none", label="Legitimate")
axes[0].hist(fraud_df["Amount"].clip(upper=500), bins=60,
             color="#e24b4a", alpha=0.7, edgecolor="none", label="Fraud")
axes[0].set_title("Transaction Amount Distribution (clipped at $500)")
axes[0].set_xlabel("Amount ($)")
axes[0].set_ylabel("Frequency")
axes[0].legend()

# Box plot
axes[1].boxplot([legit_df["Amount"].values, fraud_df["Amount"].values],
                tick_labels=["Legitimate", "Fraud"],
                patch_artist=True,
                boxprops=dict(facecolor="#e8f0fb"),
                medianprops=dict(color="#e24b4a", linewidth=2))
axes[1].set_title("Amount Box Plot by Class")
axes[1].set_ylabel("Amount ($)")
axes[1].set_ylim(0, 500)

plt.tight_layout()
plt.savefig("reports/02_amount_analysis.png")
plt.show()

# =============================================================================
# 4. TIME ANALYSIS
# =============================================================================
df["Hour"] = (df["Time"] % 86400) // 3600

hourly = df.groupby("Hour").agg(
    total=("Class", "count"),
    fraud=("Class", "sum")
).reset_index()
hourly["fraud_rate"] = hourly["fraud"] / hourly["total"] * 100

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

axes[0].plot(hourly["Hour"], hourly["total"], color="#2a78d6", linewidth=2)
axes[0].fill_between(hourly["Hour"], hourly["total"], alpha=0.15, color="#2a78d6")
axes[0].set_title("Hourly Transaction Volume", fontweight="bold")
axes[0].set_ylabel("Transactions")

axes[1].plot(hourly["Hour"], hourly["fraud_rate"], color="#e24b4a", linewidth=2)
axes[1].fill_between(hourly["Hour"], hourly["fraud_rate"], alpha=0.15, color="#e24b4a")
axes[1].set_title("Hourly Fraud Rate (%)", fontweight="bold")
axes[1].set_xlabel("Hour of Day")
axes[1].set_ylabel("Fraud Rate (%)")
axes[1].axhline(y=fraud_rate, color="gray", linestyle="--", alpha=0.6, label=f"Avg {fraud_rate:.2f}%")
axes[1].legend()

plt.tight_layout()
plt.savefig("reports/03_time_analysis.png")
plt.show()

# =============================================================================
# 5. FEATURE CORRELATION WITH FRAUD
# =============================================================================
corr = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False)
top10 = corr.head(10)

plt.figure(figsize=(10, 5))
bars = plt.barh(top10.index[::-1], top10.values[::-1],
                color=["#e24b4a" if v > 0.3 else "#2a78d6" if v > 0.1 else "#888"
                       for v in top10.values[::-1]],
                edgecolor="none")
plt.title("Top 10 Features Correlated with Fraud (absolute)", fontweight="bold")
plt.xlabel("|Correlation with Class|")
plt.axvline(x=0.3, color="red", linestyle="--", alpha=0.5, label="|r| > 0.3 (strong)")
plt.axvline(x=0.1, color="orange", linestyle="--", alpha=0.5, label="|r| > 0.1 (moderate)")
plt.legend()
plt.tight_layout()
plt.savefig("reports/04_feature_correlation.png")
plt.show()

print("TOP 10 FEATURES CORRELATED WITH FRAUD:")
print("-" * 40)
for feat, val in top10.items():
    print(f"  {feat:5s}  |r| = {val:.4f}")

# =============================================================================
# 6. PCA FEATURE DISTRIBUTIONS (V14, V12, V17)
# =============================================================================
key_features = ["V14", "V12", "V17"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, feat in enumerate(key_features):
    axes[i].hist(legit_df[feat], bins=80, alpha=0.6, color="#2a78d6",
                 density=True, label="Legitimate", edgecolor="none")
    axes[i].hist(fraud_df[feat], bins=80, alpha=0.6, color="#e24b4a",
                 density=True, label="Fraud", edgecolor="none")
    axes[i].set_title(f"{feat} Distribution by Class", fontweight="bold")
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel("Density")
    axes[i].legend()
plt.tight_layout()
plt.savefig("reports/05_pca_distributions.png")
plt.show()

print("\nEDA complete. Plots saved to reports/")
