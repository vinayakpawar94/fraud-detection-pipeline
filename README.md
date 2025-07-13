
# 🔍 Fraud Detection Pipeline with Anomaly Scoring

A powerful machine learning pipeline that identifies fraudulent financial transactions using supervised learning, anomaly detection, and threshold tuning — built for **real-world fraud detection at scale**.

![Fraud Detection Banner](https://img.shields.io/badge/Built%20with-Python%20%7C%20Scikit--learn-blue?style=for-the-badge)  
![Stars](https://img.shields.io/github/stars/vinayakpawar94/fraud-detection-pipeline?style=social)  
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🚀 Demo

🎥 **Watch it in action**:  
➡️ [Screen Recording](#) *(Upload your video or YouTube link here)*

📊 **Sample Output**
```
🔮 Sample Live Predictions:
   isFraud  predicted_fraud  fraud_probability
0        0                0              0.035
1        1                1              0.981
2        0                0              0.022
```

---

## 🧠 Features

- ✅ Preprocessing 6M+ transaction records  
- ✅ Isolation Forest for anomaly scoring  
- ✅ SMOTE for class imbalance  
- ✅ Random Forest classifier with grid-tuned parameters  
- ✅ Threshold tuning to **maximize recall** (don’t miss fraud!)  
- ✅ ROC-AUC, Precision/Recall, Confusion Matrix, Feature Importance  
- ✅ Saved model components for easy deployment  
- ✅ Interactive evaluation and plotting

---

## 🛠️ Technologies Used

| Category              | Tools/Libs                                       |
|-----------------------|--------------------------------------------------|
| Language              | Python                                           |
| Data Handling         | Pandas, NumPy, SQLite                            |
| Modeling              | Scikit-learn, SMOTE, Random Forest, IsolationForest |
| Metrics               | Recall, Precision, F1-score, ROC-AUC             |
| Visualization         | Matplotlib, Seaborn                              |
| Deployment Ready      | Joblib for saving models                         |

---

## 📂 Project Structure

```
📁 fraud-detection-pipeline/
│
├── 📄 Fraud Detection Pipeline.py         # Main script
├── 📦 rf_with_anomaly_thresh.pkl         # Trained model
├── 📦 scaler_with_anomaly_thresh.pkl     # Scaler
├── 📦 iso_forest_model_thresh.pkl        # Anomaly detector
├── 📦 best_threshold.pkl                 # Tuned threshold
├── 📽️ Screen Recording.mp4              # Demo video
└── 📑 README.md                          # This file
```

---

## 🧪 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/vinayakpawar94/fraud-detection-pipeline.git
cd fraud-detection-pipeline

# Install requirements
pip install -r requirements.txt

# Run the script
python "Fraud Detection Pipeline.py"
```

---

## 💡 Model Insight

This project focuses on **not missing fraudulent transactions**, even at the cost of slightly lower precision.  
By integrating **Isolation Forest**, anomaly scores complement supervised learning models and help improve overall detection accuracy.

---

## 📈 Results

| Metric         | Value (Validation Set) |
|----------------|------------------------|
| Recall         | ✅ 0.99+               |
| Precision      | 🎯 Optimized via threshold |
| ROC-AUC        | 📈 ~0.98                |

---

## 📬 Connect with Me

👤 **Vinayak Pawar**  
🔗 [GitHub](https://github.com/vinayakpawar94)  
🔗 [LinkedIn](https://linkedin.com/in/vinayak-pawar94)  
📧 vnpawar94@gmail.com

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
