import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, sqlite3, joblib, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
warnings.filterwarnings('ignore')

# 1. Load Data
conn = sqlite3.connect('Database.db')
df = pd.read_sql_query("SELECT * FROM Fraud_detection", conn)

# 2. Clean & Convert
df.replace('', np.nan, inplace=True)
for col in ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Handle NaNs for fraud rows
fraud_rows = df[df['isFraud'] == 1]
for col in ['type', 'newbalanceOrig', 'oldbalanceDest']:
    fraud_rows[col].fillna(fraud_rows[col].mode()[0] if fraud_rows[col].dtype == 'O' else fraud_rows[col].mean(), inplace=True)

# 4. Combine cleaned data
df = pd.concat([df.dropna(), fraud_rows])
df['type'] = df['type'].map({'PAYMENT': 1 , 'TRANSFER': 2 , 'CASH_OUT': 3 , 'DEBIT': 4, 'CASH_IN': 5})
df.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)

# 5. Split Data
train, temp = train_test_split(df, train_size=3900900, stratify=df['isFraud'], random_state=42)
val, live = train_test_split(temp, test_size=0.5, stratify=temp['isFraud'], random_state=42)

X_train, y_train = train.drop('isFraud', axis=1), train['isFraud']
X_val, y_val = val.drop('isFraud', axis=1), val['isFraud']
X_live, y_live = live.drop('isFraud', axis=1), live['isFraud']

# 6. Add Anomaly Score from Isolation Forest
iso_forest = IsolationForest(contamination=0.001, random_state=42)
iso_forest.fit(X_train)

X_train['anomaly_score'] = iso_forest.decision_function(X_train)
X_val['anomaly_score'] = iso_forest.decision_function(X_val)
X_live['anomaly_score'] = iso_forest.decision_function(X_live)

# 7. SMOTE and Scaling
X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_sm_scaled = scaler.fit_transform(X_train_sm)
X_val_scaled = scaler.transform(X_val)
X_live_scaled = scaler.transform(X_live)

# 8. Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, max_depth=15, max_features='sqrt', random_state=42, n_jobs=-1)
model.fit(X_train_sm_scaled, y_train_sm)

# 9. Threshold Tuning
y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
thresholds = np.arange(0.01, 1.01, 0.01)  # start from 0.01 to skip t=0.00

best = None
for t in thresholds:
    preds = (y_val_proba >= t).astype(int)
    recall = recall_score(y_val, preds)
    precision = precision_score(y_val, preds, zero_division=0)

    # ✅ Require high recall (≥ 0.99), not just 1.0
    if recall >= 0.99:
        if best is None or precision > best['precision']:
            best = {
                'threshold': t,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score(y_val, preds),
                'conf_matrix': confusion_matrix(y_val, preds)
            }

# 10. Report Threshold Results
if best:
    print("\n✅ Best Threshold (Recall = 1.0):")
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"F1 Score : {best['f1_score']:.4f}")
    print("Confusion Matrix:\n", best['conf_matrix'])
    best_threshold = best['threshold']
else:
    print("\n❌ No threshold found with recall = 1.0. Using default 0.5.")
    best_threshold = 0.5

# 11. Final Evaluation Function
def evaluate(X, y, title, threshold):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    print(f"\n📊 {title} Set (Threshold = {threshold})")
    print("Accuracy:", round(accuracy_score(y, y_pred),4))
    print("Recall  :", round(recall_score(y, y_pred),4))
    print("Precision:", round(precision_score(y, y_pred),4))
    print("F1-score:", round(f1_score(y, y_pred),4))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc_score(y, y_proba):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{title} ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# 12. Evaluate on Validation and Live
evaluate(X_val_scaled, y_val, "Validation", best_threshold)
evaluate(X_live_scaled, y_live, "Live", best_threshold)

# 13. Feature Importance
importances = model.feature_importances_
features = X_train.columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=features[indices])
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# 14. Save Model
joblib.dump(model, 'rf_with_anomaly_thresh.pkl')
joblib.dump(scaler, 'scaler_with_anomaly_thresh.pkl')
joblib.dump(iso_forest, 'iso_forest_model_thresh.pkl')
joblib.dump(best_threshold, 'best_threshold.pkl')

# 15. Predictions on Live Set
y_live_proba = model.predict_proba(X_live_scaled)[:, 1]
live['predicted_fraud'] = (y_live_proba >= best_threshold).astype(int)
live['fraud_probability'] = y_live_proba

print("\n🔮 Sample Live Predictions:")
print(live[['isFraud', 'predicted_fraud', 'fraud_probability']].head(15))



