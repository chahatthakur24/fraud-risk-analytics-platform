💳 AI-Powered Credit Card Fraud Detection Platform

An end-to-end Machine Learning platform that detects fraudulent credit card transactions using multiple classification models and exposes predictions through a FastAPI REST API. The project also includes SQL analytics, Power BI dashboards, and business-focused fraud monitoring to simulate a real-world enterprise fraud detection solution.

Project Highlights

✔ End-to-end Machine Learning Pipeline

✔ Exploratory Data Analysis (EDA)

✔ Feature Engineering

✔ Multiple ML Models

✔ Model Performance Comparison

✔ FastAPI REST API

✔ PostgreSQL Analytics

✔ Power BI Dashboard

✔ Production-ready Project Structure

Business Problem

Financial institutions process millions of transactions daily, where fraudulent transactions account for only a tiny fraction of total activity. Because fraud is extremely rare, traditional accuracy metrics are insufficient.

This project focuses on building an intelligent fraud detection system capable of identifying suspicious transactions while minimizing false positives.

Dataset
284,807 real-world transactions
492 fraudulent transactions
31 transaction attributes
Fraud rate of 0.17%

The dataset closely reflects real-world financial fraud scenarios with severe class imbalance.

Tech Stack
Programming
Python
SQL
Libraries
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib
API
FastAPI
Pydantic
Analytics
PostgreSQL
Power BI
DAX
Development
Git
GitHub
Jupyter Notebook
Architecture
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
SMOTE + Scaling
      │
      ▼
Train Multiple ML Models
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
FastAPI Deployment
      │
      ▼
REST API
      │
      ▼
Power BI Dashboard
Machine Learning Pipeline

The project trains and compares four classification approaches:

Logistic Regression
Random Forest
XGBoost
Weighted Ensemble Model

Evaluation metrics include:

ROC-AUC
PR-AUC
Precision
Recall
F1 Score

Because of the highly imbalanced dataset, PR-AUC is used as the primary evaluation metric.

Model Performance
Model	ROC-AUC	PR-AUC	Precision	Recall
Logistic Regression	0.972	0.721	0.801	0.764
Random Forest	0.979	0.859	0.873	0.830
XGBoost	0.983	0.878	0.893	0.846
Weighted Ensemble	0.984	0.881	0.896	0.849
API Features

The project exposes a production-style FastAPI service.

Endpoints
GET  /health
GET  /model-info
POST /predict
POST /predict/batch

The API supports:

Real-time fraud prediction
Batch inference
Model metadata
Health monitoring
SQL Analytics

Developed 17+ SQL queries to generate fraud insights including:

Fraud rate analysis
High-risk transaction detection
Z-score anomaly detection
Merchant category analysis
Financial exposure analysis
Risk segmentation
Transaction trends
Power BI Dashboard

The dashboard provides interactive business reporting with:

Executive KPI Cards
Fraud Rate Monitoring
Risk Distribution
Transaction Trends
Feature Analysis
Model Performance
Drill-through Reports
Key Achievements
Processed 284,807 financial transactions.
Addressed severe class imbalance using SMOTE, class weighting, and threshold optimization.
Compared multiple ML models and achieved 98.4% ROC-AUC and 88.1% PR-AUC using a weighted ensemble model.
Built a production-ready FastAPI service for real-time fraud prediction.
Developed SQL analytics and Power BI dashboards to support fraud monitoring and business decision-making.

