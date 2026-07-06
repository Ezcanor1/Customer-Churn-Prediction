"""
Customer Churn Prediction - Training Script with MLflow Tracking

Run this AS-IS on Databricks (MLflow is pre-integrated there, no setup needed)
or locally with `mlflow ui` to view results at http://localhost:5000

This logs: params, metrics, the trained model, and a confusion matrix artifact.
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------------------------------------------------
# 1. Load & preprocess data
# ---------------------------------------------------------------
df = pd.read_csv('/home/claude/churn_project/churn_data.csv')

categorical_cols = ['contract_type', 'tech_support', 'internet_service',
                     'payment_method', 'paperless_billing']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------
# 2. Train + log multiple runs to MLflow (so you have real runs to compare)
# ---------------------------------------------------------------
mlflow.set_experiment("customer_churn_prediction")

models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest_100": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
    "random_forest_200": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
}

best_model = None
best_f1 = 0
best_run_name = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        # Log params
        mlflow.log_param("model_type", name)
        if hasattr(model, 'n_estimators'):
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth", model.max_depth)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # Log confusion matrix as an artifact (image)
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_path = f'/home/claude/churn_project/cm_{name}.png'
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # Log the model itself
        mlflow.sklearn.log_model(model, "model")

        print(f"{name}: acc={acc:.3f} precision={prec:.3f} recall={rec:.3f} "
              f"f1={f1:.3f} auc={auc:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_run_name = name

print(f"\nBest model: {best_run_name} (F1={best_f1:.3f})")

# ---------------------------------------------------------------
# 3. Save the best model + encoders for deployment (Azure serving)
# ---------------------------------------------------------------
joblib.dump(best_model, '/home/claude/churn_project/model.pkl')
joblib.dump(encoders, '/home/claude/churn_project/encoders.pkl')
joblib.dump(list(X.columns), '/home/claude/churn_project/feature_columns.pkl')
print("Saved model.pkl, encoders.pkl, feature_columns.pkl for deployment.")
