"""
Flask serving app for the churn prediction model.
Deploy this to Azure App Service (or Azure ML endpoint) as a REST API.

Local test:
    python app.py
    curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" \
      -d '{"tenure":5,"monthly_charges":85.5,"total_charges":427.5,"contract_type":"Month-to-month",
           "tech_support":"No","internet_service":"Fiber optic","payment_method":"Electronic check",
           "num_support_calls":3,"senior_citizen":0,"paperless_billing":"Yes"}'
"""
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, 'model.pkl'))
encoders = joblib.load(os.path.join(BASE, 'encoders.pkl'))
feature_columns = joblib.load(os.path.join(BASE, 'feature_columns.pkl'))


@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Churn prediction API is running"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Apply the same label encoders used at training time
        for col, le in encoders.items():
            df[col] = le.transform(df[col])

        df = df[feature_columns]  # enforce correct column order

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        return jsonify({
            "churn_prediction": int(pred),
            "churn_probability": round(float(prob), 4),
            "risk_level": "High" if prob > 0.6 else "Medium" if prob > 0.35 else "Low"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    # Azure App Service sets the PORT env variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
