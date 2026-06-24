# Customer Churn Prediction: Logistic Regression vs Random Forest vs XGBoost

An end-to-end machine learning project predicting customer churn using the IBM Telco Customer Churn dataset. Three classification models are benchmarked and compared using industry-standard metrics.

---

## Project Overview

| | |
|---|---|
| **Domain** | Banking / Telecom / Customer Analytics |
| **Dataset** | IBM Telco Customer Churn (Kaggle) — 7,043 customers | Churn rate: 26.5% |
| **Problem Type** | Binary Classification |
| **Models** | Logistic Regression, Random Forest, XGBoost |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Experiment Tracking** | MLflow |

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 74.10% | 50.78% | 78.61% | 61.70% | **84.60%** |
| Random Forest | 77.15% | 55.02% | 76.20% | 63.90% | 84.14% |
| XGBoost | 75.30% | 52.37% | 76.74% | 62.26% | 83.94% |

**Logistic Regression** achieved the best ROC-AUC (84.60%), making it the recommended model given its interpretability and strong recall of 78.61% — critical for identifying at-risk customers before they churn.

---

## Key Business Insights

- **Month-to-month contract customers** churn at significantly higher rates than long-term contract holders
- **New customers (tenure < 12 months)** represent the highest-risk churn segment
- **Fiber optic internet customers** show elevated churn — likely driven by pricing sensitivity
- **High monthly charges** are a strong predictor of churn across all models
- **Logistic Regression recall of 78.61%** means the model correctly identifies 4 out of 5 customers who will churn

---

## Project Structure