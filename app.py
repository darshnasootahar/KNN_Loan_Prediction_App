"""
Flask REST API — Loan Approval Prediction (KNN)
Run: python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow Next.js to call this API

# ── Load model & scaler ───────────────────────────────────
BASE = os.path.dirname(__file__)
model   = joblib.load(os.path.join(BASE, "knn_model.pkl"))
scaler  = joblib.load(os.path.join(BASE, "scaler.pkl"))
features = joblib.load(os.path.join(BASE, "feature_cols.pkl"))

# ── Encoding maps (mirrors LabelEncoder fit order) ────────
GENDER_MAP       = {"Female": 0, "Male": 1}
MARRIED_MAP      = {"No": 0, "Yes": 1}
DEPENDENTS_MAP   = {"0": 0, "1": 1, "2": 2, "3+": 3}
EDUCATION_MAP    = {"Graduate": 0, "Not Graduate": 1}
SELF_EMP_MAP     = {"No": 0, "Yes": 1}
PROPERTY_MAP     = {"Rural": 0, "Semiurban": 1, "Urban": 2}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # ── Build feature vector ──────────────────────────
        row = [
            GENDER_MAP.get(data["gender"], 1),
            MARRIED_MAP.get(data["married"], 1),
            DEPENDENTS_MAP.get(str(data["dependents"]), 0),
            EDUCATION_MAP.get(data["education"], 0),
            SELF_EMP_MAP.get(data["self_employed"], 0),
            float(data["applicant_income"]),
            float(data["coapplicant_income"]),
            float(data["loan_amount"]),
            float(data["loan_amount_term"]),
            float(data["credit_history"]),
            PROPERTY_MAP.get(data["property_area"], 2),
        ]

        X = np.array(row).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]

        return jsonify({
            "prediction": "Approved" if prediction == 1 else "Rejected",
            "approved":   bool(prediction == 1),
            "confidence": round(float(max(probability)) * 100, 1),
            "prob_approved": round(float(probability[1]) * 100, 1),
            "prob_rejected": round(float(probability[0]) * 100, 1),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "KNN Loan Predictor"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
