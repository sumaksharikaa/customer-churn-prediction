"""
Customer Churn Prediction Project
===================================
Dataset  : IBM Telco Customer Churn (Kaggle)
Author   : Sumaksharika Nainavarapu
Goal     : Predict customer churn using Logistic Regression, XGBoost, Random Forest
Metrics  : Accuracy, Precision, Recall, F1, ROC-AUC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from xgboost import XGBClassifier
import mlflow

ACCENT = "#1F4E79"
BLUE   = "#2E75B6"
GREEN  = "#70AD47"
RED    = "#C00000"

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
# Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# File: WA_Fn-UseC_-Telco-Customer-Churn.csv

try:
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("Real dataset loaded!")
except FileNotFoundError:
    print("Dataset not found — using synthetic data for demo.")
    print("Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n")
    np.random.seed(42)
    n = 7043
    df = pd.DataFrame({
        "customerID":       [f"CUST-{i:05d}" for i in range(n)],
        "gender":           np.random.choice(["Male","Female"], n),
        "SeniorCitizen":    np.random.choice([0,1], n, p=[0.84,0.16]),
        "Partner":          np.random.choice(["Yes","No"], n),
        "Dependents":       np.random.choice(["Yes","No"], n, p=[0.3,0.7]),
        "tenure":           np.random.randint(0, 72, n),
        "PhoneService":     np.random.choice(["Yes","No"], n, p=[0.9,0.1]),
        "MultipleLines":    np.random.choice(["Yes","No","No phone service"], n),
        "InternetService":  np.random.choice(["DSL","Fiber optic","No"], n, p=[0.34,0.44,0.22]),
        "OnlineSecurity":   np.random.choice(["Yes","No","No internet service"], n),
        "TechSupport":      np.random.choice(["Yes","No","No internet service"], n),
        "Contract":         np.random.choice(["Month-to-month","One year","Two year"], n, p=[0.55,0.21,0.24]),
        "PaperlessBilling": np.random.choice(["Yes","No"], n, p=[0.59,0.41]),
        "PaymentMethod":    np.random.choice(["Electronic check","Mailed check","Bank transfer","Credit card"], n),
        "MonthlyCharges":   np.round(np.random.uniform(18, 119, n), 2),
        "TotalCharges":     np.round(np.random.uniform(18, 8684, n), 2),
    })
    churn_prob = (0.1
        + 0.25*(df["Contract"]=="Month-to-month")
        + 0.15*(df["InternetService"]=="Fiber optic")
        + 0.15*(df["tenure"]<12)
        + 0.10*(df["SeniorCitizen"]==1)
        - 0.10*(df["tenure"]>48)).clip(0,1)
    df["Churn"] = np.where(np.random.rand(n) < churn_prob, "Yes", "No")

print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {(df['Churn']=='Yes').mean()*100:.1f}%")

# ── 2. PREPROCESSING ──────────────────────────────────────────────────────────
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# Feature engineering
df["avg_monthly_spend"]  = df["TotalCharges"] / (df["tenure"] + 1)
df["is_new_customer"]    = (df["tenure"] < 12).astype(int)
df["is_long_term"]       = (df["tenure"] > 48).astype(int)
df["month_to_month"]     = (df["Contract"] == "Month-to-month").astype(int)
df["fiber_optic"]        = (df["InternetService"] == "Fiber optic").astype(int)
df["electronic_check"]   = (df["PaymentMethod"] == "Electronic check").astype(int)
df["no_online_security"] = (df["OnlineSecurity"] == "No").astype(int)
df["no_tech_support"]    = (df["TechSupport"] == "No").astype(int)

# Encode categoricals
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

print(f"Preprocessed shape: {df.shape}")

# ── 3. EDA ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Customer Churn Analysis — EDA", fontsize=14, fontweight="bold")

churn_counts = df["Churn"].value_counts()
axes[0,0].bar(["No Churn","Churned"], churn_counts.values, color=[BLUE, RED])
axes[0,0].set_title("Churn Distribution"); axes[0,0].set_ylabel("Count")
for i, v in enumerate(churn_counts.values):
    axes[0,0].text(i, v+50, str(v), ha="center", fontweight="bold")

axes[0,1].hist(df[df["Churn"]==0]["tenure"], bins=30, alpha=0.7, color=BLUE, label="No Churn")
axes[0,1].hist(df[df["Churn"]==1]["tenure"], bins=30, alpha=0.7, color=RED, label="Churned")
axes[0,1].set_title("Tenure by Churn"); axes[0,1].set_xlabel("Tenure (months)"); axes[0,1].legend()

axes[0,2].boxplot([df[df["Churn"]==0]["MonthlyCharges"], df[df["Churn"]==1]["MonthlyCharges"]],
                  labels=["No Churn","Churned"], patch_artist=True,
                  boxprops=dict(facecolor=BLUE, alpha=0.7))
axes[0,2].set_title("Monthly Charges by Churn"); axes[0,2].set_ylabel("Monthly Charges ($)")

c1 = df.groupby("month_to_month")["Churn"].mean()*100
axes[1,0].bar(["Long-term","Month-to-Month"], c1.values, color=[GREEN, RED])
axes[1,0].set_title("Churn Rate by Contract"); axes[1,0].set_ylabel("Churn Rate (%)")
for i, v in enumerate(c1.values): axes[1,0].text(i, v+0.5, f"{v:.1f}%", ha="center", fontweight="bold")

c2 = df.groupby("SeniorCitizen")["Churn"].mean()*100
axes[1,1].bar(["Non-Senior","Senior"], c2.values, color=[BLUE, RED])
axes[1,1].set_title("Churn: Senior vs Non-Senior"); axes[1,1].set_ylabel("Churn Rate (%)")
for i, v in enumerate(c2.values): axes[1,1].text(i, v+0.5, f"{v:.1f}%", ha="center", fontweight="bold")

c3 = df.groupby("fiber_optic")["Churn"].mean()*100
axes[1,2].bar(["Other Internet","Fiber Optic"], c3.values, color=[BLUE, RED])
axes[1,2].set_title("Churn by Internet Service"); axes[1,2].set_ylabel("Churn Rate (%)")
for i, v in enumerate(c3.values): axes[1,2].text(i, v+0.5, f"{v:.1f}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("eda_churn.png", dpi=150, bbox_inches="tight")
print("EDA plot saved → eda_churn.png")
plt.close()

# ── 4. TRAIN/TEST SPLIT ───────────────────────────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ── 5. EVALUATE FUNCTION ──────────────────────────────────────────────────────
def evaluate_model(model, Xt, yt, name):
    preds = model.predict(Xt)
    proba = model.predict_proba(Xt)[:,1]
    m = {
        "model": name,
        "Accuracy":  round(accuracy_score(yt, preds)*100, 2),
        "Precision": round(precision_score(yt, preds)*100, 2),
        "Recall":    round(recall_score(yt, preds)*100, 2),
        "F1":        round(f1_score(yt, preds)*100, 2),
        "ROC_AUC":   round(roc_auc_score(yt, proba)*100, 2),
        "proba":     proba
    }
    print(f"\n{name}")
    for k,v in m.items():
        if k not in ["model","proba"]: print(f"  {k}: {v}%")
    return m

# ── 6. MODELS ─────────────────────────────────────────────────────────────────
print("\n" + "="*50 + "\nMODEL 1: LOGISTIC REGRESSION\n" + "="*50)
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_train_sc, y_train)
lr_m = evaluate_model(lr, X_test_sc, y_test, "Logistic Regression")

print("\n" + "="*50 + "\nMODEL 2: RANDOM FOREST\n" + "="*50)
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
rf_m = evaluate_model(rf, X_test, y_test, "Random Forest")

print("\n" + "="*50 + "\nMODEL 3: XGBOOST\n" + "="*50)
scale_pos = (y_train==0).sum()/(y_train==1).sum()
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=scale_pos, random_state=42, eval_metric="logloss")
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_m = evaluate_model(xgb, X_test, y_test, "XGBoost")

# ── 7. PLOTS ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14,5))

for m, color in [(lr_m, RED), (rf_m, BLUE), (xgb_m, GREEN)]:
    fpr, tpr, _ = roc_curve(y_test, m["proba"])
    axes[0].plot(fpr, tpr, label=f"{m['model']} (AUC {m['ROC_AUC']}%)", color=color, linewidth=2)
axes[0].plot([0,1],[0,1],"k--",alpha=0.4)
axes[0].set_title("ROC Curve Comparison", fontweight="bold")
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].legend(); axes[0].grid(alpha=0.3)

results = pd.DataFrame([{k:v for k,v in m.items() if k!="proba"} for m in [lr_m, rf_m, xgb_m]])
x = np.arange(len(results)); w = 0.15
for i, (metric, color) in enumerate(zip(["Accuracy","Precision","Recall","F1","ROC_AUC"],
                                         [BLUE, GREEN, RED, "#FFC000", ACCENT])):
    axes[1].bar(x+i*w, results[metric], w, label=metric, color=color, alpha=0.85)
axes[1].set_xticks(x+w*2); axes[1].set_xticklabels(results["model"], fontsize=9)
axes[1].set_title("Model Comparison", fontweight="bold"); axes[1].set_ylabel("Score (%)")
axes[1].legend(fontsize=8); axes[1].set_ylim(50,105); axes[1].grid(alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig("model_comparison_churn.png", dpi=150, bbox_inches="tight")
print("Model comparison saved → model_comparison_churn.png")
plt.close()

# Feature importance
imp = pd.DataFrame({"feature": X.columns, "importance": xgb.feature_importances_}).sort_values("importance").tail(15)
fig, ax = plt.subplots(figsize=(9,6))
ax.barh(imp["feature"], imp["importance"], color=ACCENT)
ax.set_title("Top 15 Churn Predictors (XGBoost)", fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance_churn.png", dpi=150, bbox_inches="tight")
print("Feature importance saved → feature_importance_churn.png")
plt.close()

# ── 8. MLFLOW ─────────────────────────────────────────────────────────────────
mlflow.set_experiment("customer_churn_prediction")
for m in [lr_m, rf_m, xgb_m]:
    with mlflow.start_run(run_name=m["model"]):
        mlflow.log_param("model_type", m["model"])
        for k in ["Accuracy","Precision","Recall","F1","ROC_AUC"]:
            mlflow.log_metric(k, m[k])
print("MLflow runs logged.")

# ── 9. RESULTS ────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nFINAL RESULTS\n" + "="*60)
print(results.to_string(index=False))
best = results.loc[results["ROC_AUC"].idxmax(), "model"]
print(f"\nBest model by ROC-AUC: {best}")
print("\nKey Business Insights:")
print("  → Month-to-month customers churn at significantly higher rates")
print("  → New customers (tenure < 12 months) are the highest risk segment")
print("  → Fiber optic customers show elevated churn — likely pricing sensitivity")
