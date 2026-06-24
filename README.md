# Customer Churn Prediction: Logistic Regression vs Random Forest vs XGBoost

An end-to-end machine learning project predicting customer churn using the IBM Telco Customer Churn dataset. Three classification models are benchmarked and compared using industry-standard metrics.

---

## Project Overview

| | |
|---|---|
| **Domain** | Banking / Telecom / Customer Analytics |
| **Dataset** | IBM Telco Customer Churn (Kaggle) |
| **Problem Type** | Binary Classification |
| **Models** | Logistic Regression, Random Forest, XGBoost |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Experiment Tracking** | MLflow |

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 64.09% | 45.37% | 68.47% | 54.58% | **72.96%** |
| Random Forest | 64.58% | 45.76% | 66.89% | 54.35% | 71.86% |
| XGBoost | 64.02% | 45.19% | 66.67% | 53.87% | 71.10% |

**Logistic Regression** achieved the best ROC-AUC (72.96%), making it the recommended model for production deployment given its interpretability and comparable performance.

---

## Key Business Insights

- **Month-to-month contract customers** churn at significantly higher rates than long-term contract holders
- **New customers (tenure < 12 months)** represent the highest-risk churn segment
- **Fiber optic internet customers** show elevated churn — likely driven by pricing sensitivity
- **High monthly charges** are a strong predictor of churn across all models

---

## Project Structure

```
customer-churn-prediction/
├── churn_prediction.py          # Main modeling script
├── eda_churn.png                # EDA visualizations
├── model_comparison_churn.png   # ROC curves + metric comparison
├── feature_importance_churn.png # XGBoost feature importance
└── README.md
```

---

## Feature Engineering

Key features engineered beyond the raw dataset:
- `avg_monthly_spend` — TotalCharges / tenure (spending intensity)
- `is_new_customer` — tenure < 12 months flag
- `is_long_term` — tenure > 48 months flag
- `month_to_month` — contract type binary flag
- `fiber_optic` — internet service binary flag
- `electronic_check` — payment method risk flag
- `no_online_security` — service gap flag
- `no_tech_support` — service gap flag

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib scikit-learn xgboost mlflow

# 2. Download dataset from Kaggle
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# File: WA_Fn-UseC_-Telco-Customer-Churn.csv

# 3. Run the pipeline
python churn_prediction.py

# 4. (Optional) View MLflow UI
mlflow ui
# Open http://localhost:5000
```

---

## Skills Demonstrated

`Python` · `Scikit-learn` · `XGBoost` · `Logistic Regression` · `Random Forest` · `Feature Engineering` · `ROC-AUC` · `MLflow` · `EDA` · `Customer Segmentation` · `Pandas` · `NumPy` · `Matplotlib` · `Classification Modeling`
