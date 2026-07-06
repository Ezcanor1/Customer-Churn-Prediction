# Customer Churn Prediction — MLflow + Azure Deployment

A binary classification project predicting whether a subscription customer
will churn, based on tenure, contract type, billing, and support interaction
features. Built to practice the full ML lifecycle: experiment tracking with
MLflow and deployment as a live REST API on Azure.

## Project files
- `generate_data.py` — creates a realistic synthetic churn dataset (3000 rows)
- `train.py` — trains 3 models (Logistic Regression, 2x Random Forest configs),
  logs all params/metrics/artifacts/models to MLflow, saves the best model
- `app.py` — Flask REST API serving the trained model
- `model.pkl`, `encoders.pkl`, `feature_columns.pkl` — deployment artifacts
- `requirements.txt` — dependencies for deployment

## Part 1 — Run MLflow tracking on Databricks

1. Create a free Databricks Community Edition account: https://community.cloud.databricks.com
2. Create a new Notebook (Python), attach it to a cluster
3. Upload `churn_data.csv` via Databricks' "Upload Data" UI, or regenerate it
   directly in a notebook cell using the code from `generate_data.py`
4. Copy the contents of `train.py` into notebook cells and run it — Databricks
   has MLflow pre-installed and auto-configured, so `mlflow.start_run()` will
   log directly to the Databricks-managed MLflow tracking server, no setup needed
5. Go to the **Experiments** tab (left sidebar) → open `customer_churn_prediction`
   → you'll see all 3 runs with metrics, the confusion matrix artifact images,
   and the logged model files, all comparable side-by-side in the UI

**What to say in the interview:** "I trained three model variants, logged
params, metrics (accuracy, precision, recall, F1, ROC-AUC), and artifacts to
MLflow on Databricks, then compared runs in the MLflow UI to pick the best
model by F1-score, since this is an imbalanced churn problem where recall
matters."

## Part 2 — Deploy the Flask API to Azure App Service

### Option A: Azure Portal (simplest, no CLI needed)
1. Zip this whole `churn_project` folder (must include `app.py`, `model.pkl`,
   `encoders.pkl`, `feature_columns.pkl`, `requirements.txt`)
2. Go to https://portal.azure.com → Create a resource → **Web App**
3. Runtime stack: Python 3.11, Region: closest to you, Pricing: Free F1 tier
4. Once created, go to **Deployment Center** → choose **Local Git** or
   **ZIP Deploy** → upload your zipped folder
5. Under **Configuration → General Settings**, set the Startup Command to:
   ```
   gunicorn --bind=0.0.0.0 --timeout 600 app:app
   ```
6. Your API will be live at `https://<your-app-name>.azurewebsites.net`
7. Test it:
   ```
   curl -X POST https://<your-app-name>.azurewebsites.net/predict \
     -H "Content-Type: application/json" \
     -d '{"tenure":5,"monthly_charges":85.5,"total_charges":427.5,
          "contract_type":"Month-to-month","tech_support":"No",
          "internet_service":"Fiber optic","payment_method":"Electronic check",
          "num_support_calls":3,"senior_citizen":0,"paperless_billing":"Yes"}'
   ```

### Option B: Azure CLI (if you have it installed)
```bash
az login
az group create --name churn-rg --location centralindia
az appservice plan create --name churn-plan --resource-group churn-rg --sku F1 --is-linux
az webapp create --resource-group churn-rg --plan churn-plan --name YOUR-UNIQUE-APP-NAME --runtime "PYTHON:3.11"
az webapp config set --resource-group churn-rg --name YOUR-UNIQUE-APP-NAME --startup-file "gunicorn --bind=0.0.0.0 --timeout 600 app:app"
az webapp deployment source config-zip --resource-group churn-rg --name YOUR-UNIQUE-APP-NAME --src churn_project.zip
```

**What to say in the interview:** "I deployed the trained model behind a
Flask REST API, containerized via Azure App Service's Python runtime, with
a `/predict` endpoint that takes customer features and returns churn
probability and risk tier. I used the Free tier for this since it was a
learning deployment, but the same setup scales to Azure ML endpoints with
autoscaling for production traffic."

## Talking points on the ML itself
- **Why F1-score, not accuracy:** churn is imbalanced (~40% positive here,
  often much lower in real data) — accuracy can look good while missing
  most actual churners, so F1/recall matter more for business impact
- **Feature signal:** contract type (month-to-month), lack of tech support,
  and fiber-optic service were engineered to correlate with higher churn —
  mirrors real-world churn drivers (price sensitivity, support friction)
- **Why Random Forest over Logistic Regression:** captures non-linear
  interactions (e.g., "long tenure + month-to-month" behaves differently
  than either feature alone) without heavy feature engineering
- **Limitation to mention honestly:** this is synthetic data with
  hand-designed signal, not a real production dataset — good for learning
  the pipeline end-to-end, not a claim of deployed business impact
