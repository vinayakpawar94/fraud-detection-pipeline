from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model components
model = joblib.load("model/rf_with_anomaly_thresh.pkl")
scaler = joblib.load("model/scaler_with_anomaly_thresh.pkl")
iso_forest = joblib.load("model/iso_forest_model_thresh.pkl")
default_threshold = joblib.load("model/best_threshold.pkl")

history = []

@app.route("/", methods=["GET", "POST"])
def index():
    global history
    result = None

    if request.method == "POST":
        try:
            # Inputs
            amount = float(request.form["amount"])
            oldbalanceOrg = float(request.form["oldbalanceOrg"])
            newbalanceOrig = float(request.form["newbalanceOrig"])
            oldbalanceDest = float(request.form["oldbalanceDest"])
            newbalanceDest = float(request.form["newbalanceDest"])
            type_text = request.form["type"]
            threshold = float(request.form["threshold"])

            type_map = {"PAYMENT": 1, "TRANSFER": 2, "CASH_OUT": 3, "DEBIT": 4, "CASH_IN": 5}
            txn_type = type_map.get(type_text, 1)

            # Feature array
            X = np.array([[amount, oldbalanceOrg, newbalanceOrig,
                           oldbalanceDest, newbalanceDest, txn_type]])

            # Anomaly score
            anomaly_score = iso_forest.decision_function(X)
            X = np.hstack((X, anomaly_score.reshape(-1, 1)))
            X_scaled = scaler.transform(X)
            proba = model.predict_proba(X_scaled)[0, 1]

            # Rule-based fraud detection
            rule_reason = None
            if (
                txn_type in [2, 3] and oldbalanceOrg > 0 and newbalanceOrig == 0
                and oldbalanceDest == 0 and newbalanceDest == 0
            ):
                rule_reason = "Classic fraud signature"
            elif txn_type in [2, 4] and amount >= 100000 and newbalanceOrig <= 1000:
                rule_reason = "High value fraud with nearly drained balance"
            elif amount > 0 and oldbalanceDest == 0 and newbalanceDest == 0:
                rule_reason = "Transfer to fake/dormant destination"
            elif (
                amount > 0 and oldbalanceOrg == newbalanceOrig
                and oldbalanceDest == newbalanceDest
            ):
                rule_reason = "No movement in balances"
            elif txn_type == 5 and amount >= 200000 and oldbalanceDest == 0:
                rule_reason = "Suspicious top-up into empty account"
            elif amount > 0 and oldbalanceOrg == newbalanceOrig:
                rule_reason = "Ghost transaction (sender balance unchanged)"
            elif proba >= 0.5 and anomaly_score[0] < -0.3:
                rule_reason = "Model risk + anomaly trigger"

            # Final prediction
            if rule_reason:
                label = "FRAUD"
                rule_based = True
            elif proba >= threshold:
                label = "FRAUD"
                rule_based = False
            else:
                label = "Not Fraud"
                rule_based = False

            result = {
                "label": label,
                "probability": round(proba * 100, 2),
                "anomaly_score": round(anomaly_score[0], 5),
                "type": type_text,
                "amount": amount,
                "rule_based": rule_based,
                "rule_reason": rule_reason
            }

            history.insert(0, result)
            history = history[:10]

        except Exception as e:
            result = {"error": str(e)}

    return render_template("index.html", result=result, history=history)

if __name__ == "__main__":
    app.run(debug=True)
