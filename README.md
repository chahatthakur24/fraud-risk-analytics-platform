# 🛡️ Credit Card Fraud Detection & Risk Analytics Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-3.0-red?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?style=flat-square&logo=streamlit)

**[🚀 Live Demo](https://fraud-risk-analytics-nqyt9hwcfcryzwpwk2euhp.streamlit.app)** · **[GitHub](https://github.com/chahatthakur24/fraud-risk-analytics-platform)**

*Built by [Chahat Thakur](https://www.linkedin.com/in/chahat-thakur-0a36b5334/)*

</div>

---

## 📌 Project Overview

An end-to-end credit card fraud detection system built on **284,807 real-world transactions** with a severe **0.17% fraud rate**. The project covers every stage of a production ML pipeline — from raw data exploration to a deployed REST API and interactive Streamlit dashboard.

**The core challenge:** On 0.17% imbalanced data, predicting "legit" always gives 99.83% accuracy — making accuracy meaningless. This system addresses the imbalance through SMOTE, threshold tuning, and PR-AUC as the primary evaluation metric.

---

## 🎯 Results

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.974 | 0.742 | 0.821 | 0.831 | 0.826 |
| Random Forest | 0.981 | 0.863 | 0.877 | 0.839 | 0.858 |
| XGBoost | 0.983 | 0.876 | 0.891 | 0.847 | 0.869 |
| **Ensemble** | **0.984** | **0.881** | **0.896** | **0.849** | **0.872** |

> PR-AUC is the primary metric — ROC-AUC is optimistic on severely imbalanced data.

---

## 🚀 Live Demo

👉 **[fraud-risk-analytics-nqyt9hwcfcryzwpwk2euhp.streamlit.app](https://fraud-risk-analytics-nqyt9hwcfcryzwpwk2euhp.streamlit.app)**

The Streamlit dashboard includes 4 pages:
- **🔍 Score Transaction** — Score any transaction in real-time with all 3 models + ensemble
- **📊 Model Analytics** — Benchmark comparison, feature importance, dataset overview
- **📁 Batch Scoring** — Upload a CSV and score thousands of transactions at once
- **📋 Session History** — Probability trend chart and history of all scored transactions

---

## 🗂️ Project Structure

```
fraud-risk-analytics-platform/
│
├── 📁 api/
│   └── main.py                     # FastAPI REST API (single + batch prediction)
│
├── 📁 notebooks/
│   ├── 01_EDA.py                   # Exploratory data analysis & visualizations
│   ├── 02_feature_engineering.py   # Feature engineering + SMOTE + scaling
│   └── 03_model_training.py        # LR + RF + XGBoost training & comparison
│
├── 📁 streamlit_app/
│   └── app.py                      # Interactive Streamlit dashboard (4 pages)
│
├── 📁 src/
│   ├── analytics_queries.sql       # 17 SQL analytical queries
│   └── dax_measures.dax            # 10 Power BI DAX measures
│
├── 📁 tests/
│   └── test_api.py                 # API unit tests
│
├── 📁 reports/                     # Generated EDA & model plots (PNG)
├── 📁 models/                      # Trained .pkl files (auto-downloaded on startup)
├── 📁 data/                        # Place creditcard.csv here (download from Kaggle)
│
├── 📁 .streamlit/
│   └── config.toml                 # Dark theme configuration
│
├── requirements.txt                # Python dependencies
└── streamlit_requirements.txt      # Streamlit Cloud dependencies
```

---

## ⚙️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/chahatthakur24/fraud-risk-analytics-platform.git
cd fraud-risk-analytics-platform
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/Scripts/activate   # Windows
source venv/bin/activate       # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
Download `creditcard.csv` from [Kaggle ULB Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/`.

> ⚠️ Dataset not included due to size (150MB). Must be downloaded separately.

### 5. Run the pipeline (in order)
```bash
python notebooks/01_EDA.py
python notebooks/02_feature_engineering.py
PYTHONPATH=. python notebooks/03_model_training.py
```

After training, `models/` will contain 6 `.pkl` files.

### 6. Run the Streamlit app
```bash
streamlit run streamlit_app/app.py
```

### 7. Run the FastAPI server (optional)
```bash
uvicorn api.main:app --reload --port 8000
```
Open [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health + models loaded |
| `GET` | `/model-info` | Model metrics and ensemble strategy |
| `POST` | `/predict` | Score one transaction |
| `POST` | `/predict/batch` | Score up to 100 transactions |
| `GET` | `/docs` | Interactive Swagger UI |

### Sample Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406, "Amount": 1837.20,
    "V1": -3.04, "V2": -3.16, "V3": 1.09, "V4": 2.29,
    "V14": -9.48, "V17": -8.26,
    "model": "ensemble"
  }'
```

### Sample Response
```json
{
  "fraud_probability": 0.9847,
  "is_fraud": true,
  "risk_tier": "High",
  "risk_score": 95,
  "confidence": "High",
  "model_used": "Ensemble (XGB + RF + LR)",
  "processing_ms": 12.4
}
```

---

## 🧠 Key Technical Decisions

**Why SMOTE?**
SMOTE generates synthetic minority samples in feature space rather than upweighting existing ones. Applied only on training data to prevent data leakage into validation and test sets.

**Why PR-AUC over ROC-AUC?**
ROC-AUC is optimistic on imbalanced data because the large negative class inflates the true negative rate. PR-AUC focuses on minority class performance — what actually matters for fraud detection.

**Why threshold tuning?**
Default 0.5 threshold is arbitrary. Sweeping thresholds on the validation set found the optimal at ~0.43, better balancing precision and recall for the fraud class.

**Why ensemble?**
Logistic Regression captures linear patterns. Random Forest handles feature interactions. XGBoost is the strongest individually. The weighted ensemble reduces variance and consistently outperforms any single model.

---

## 🔬 Feature Engineering

| Feature | Source | Rationale |
|---|---|---|
| `Log_Amount` | `log1p(Amount)` | Reduces right skew in transaction amounts |
| `Is_Night` | Hour 22–5 | Night transactions have 3.2× baseline fraud rate |
| `High_Amount` | Amount > $1000 | 4.7× higher fraud probability |
| `V14_x_V17` | V14 × V17 | Interaction of two strongest fraud signals |
| `V14_squared` | V14² | Captures non-linear V14 behaviour |
| `Risk_Score` | Rule engine | Rule-based score retained as ML feature |

---

## 📊 SQL Analytics (17 queries)

Covering:
- Fraud rate by transaction amount decile
- Rolling 6-hour fraud average using window functions
- Z-score outlier detection across PCA features
- NTILE-based decile analysis of fraud probability
- Confusion matrix simulation at multiple thresholds
- Top-5% highest-risk transactions using CTEs

See [`src/analytics_queries.sql`](src/analytics_queries.sql) for all 17 queries.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| ML | Scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Data | Pandas, NumPy, Matplotlib, Seaborn |
| Database | PostgreSQL, MySQL |
| DevOps | Git, GitHub, Streamlit Cloud |

---

## 💬 Interview Talking Points

**On class imbalance:**
> "0.17% fraud rate means predicting 'legit' always gives 99.83% accuracy — completely useless. I used SMOTE on training data only, with PR-AUC as my primary evaluation metric."

**On threshold tuning:**
> "Default 0.5 threshold is arbitrary. I swept thresholds from 0.1 to 0.9 on the validation set and picked the one maximising F1. The optimal was 0.43."

**On ensemble:**
> "Each model has different strengths. The weighted ensemble reduces variance and consistently outperforms any single model on both PR-AUC and F1."

**On V14:**
> "V14 is a PCA-transformed component. Its distribution clusters below -5 for fraudulent transactions, making it the strongest individual fraud signal in the dataset."

---

## 📂 Dataset

**Source:** [ULB Machine Learning Group — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions · 492 fraud cases · 0.17% fraud rate
- 28 PCA-anonymised features (V1–V28) + Time + Amount
- Real European credit card transactions (September 2013)

---

## 👤 Author

**Chahat Thakur**
- GitHub: [@chahatthakur24](https://github.com/chahatthakur24)
- LinkedIn: [chahat-thakur](https://www.linkedin.com/in/chahat-thakur-0a36b5334/)
- Email: chahat2404@gmail.com

---

<div align="center">
⭐ If you found this project useful, please consider giving it a star!
</div>
