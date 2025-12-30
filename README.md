# 🔐 Cyber Threats & Financial Loss Prediction (2015–2024)

An end-to-end machine learning project predicting high financial losses from cyber threats using structured, data-leak-free global incident data. The project implements a complete ML pipeline, from preprocessing to deployment of the best-performing model for real-world predictions.

---

## 📌 Project Overview

Cybersecurity incidents increasingly cause significant financial losses across industries. Estimating losses is challenging due to complex, non-linear factors such as attack type, affected industry, incident resolution time, and number of affected users.

**This project provides:**

* High-risk cyber incident prediction (binary classification)
* Key feature identification driving financial loss
* Model benchmarking for robust predictions
* Deployment of the best model for production use

---

## 🚨 Problem Statement

Organizations often struggle to quantify financial loss from cyber attacks because of:

* Rapidly evolving cyber threat types
* Variability in incident resolution times
* Industry-specific vulnerabilities and defenses

**Solution:** Reframe as binary classification using a high-loss threshold:

* **High Loss (1):** Incidents above optimized threshold (maximizes Macro F1)  
* **Low/Moderate Loss (0):** Incidents below or equal to threshold  

_No separate low-loss threshold is required._

---

## 🎯 Purpose

Provide organizations with a risk prediction system to:

* Prioritize high-risk cyber incidents
* Allocate cybersecurity resources efficiently
* Reduce financial and operational impact
* Support data-driven cybersecurity investments

---

## 🧾 Dataset & Features

* **Source:** Kaggle – Global Cybersecurity Threats (2015–2024)  
* **Size:** ~3,000 incidents  

**Features:**

* Number of Affected Users
* Incident Resolution Time (Hours)
* Attack Type
* Target Industry
* Attack Source
* Security Vulnerability Type
* Interaction Feature: `AttackType_TargetIndustry`
* Users_per_Hour, Log_Users

_Post-preprocessing, 25–65 features are used depending on the model._

---

## 🔄 End-to-End ML Pipeline

1. **Feature Selection & Binary Target Creation** – Define binary target using high-loss threshold.  
2. **Preprocessing Pipeline** – Scale numeric features, encode categoricals, split train/test datasets.  
3. **Baseline Model Training** – RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost.  
4. **Hyperparameter Tuning & Probability Cutoff** – Optimize models and tune probability thresholds.  
5. **Baseline vs Model Comparison** – Evaluate improvements in Accuracy, Macro F1, Macro Recall.  
6. **Feature Importance & Analysis** – Aggregate expanded features to main business-level features.  
7. **Model Benchmarking** – Train, test, and cross-validate models; compare metrics and inference time.  
8. **Deployment** – Save best model (`production_model.joblib`) and implement prediction workflow.

---

## 🧠 Feature Engineering & Preprocessing

* Handle missing values (median for numeric, mode for categorical)  
* Create interaction features (`AttackType_TargetIndustry`)  
* Scale numeric variables; one-hot encode categoricals  
* Train/test split: 4:1 ratio  
* Binary target defined via high-loss threshold  

---

## 🧪 Machine Learning Models Used

* Random Forest  
* Extra Trees  
* XGBoost  
* LightGBM  
* CatBoost  

---

## 🔍 Evaluation Metrics

* Accuracy  
* Precision  
* Recall  
* Macro F1  
* ROC-AUC  
* Inference Time per Sample  

---

## 📊 Results Summary

**Model Comparison (Step 5–8)**

| Model       | Accuracy | Macro F1 | Macro Recall |
|------------|---------|----------|-------------|
| CatBoost    | 0.545   | 0.527    | 0.531       |
| ExtraTrees  | 0.485   | 0.480    | 0.480       |
| LightGBM    | 0.530   | 0.510    | 0.515       |
| RandomForest| 0.523   | 0.517    | 0.517       |
| XGBoost     | 0.530   | 0.530    | 0.532       |

**Best Model:** CatBoost (based on Macro F1)  

**Key Features Driving Predictions:**

* Number of Affected Users  
* Incident Resolution Time (Hours)  
* `AttackType_TargetIndustry`  
* Attack Source  

---

## 🛠 Handling High Variance & Bias

* Tuned regularization parameters (`l2_leaf_reg`, `max_depth`, etc.)  
* Cross-validation for robust performance  
* Feature aggregation to reduce noise  
* Avoided overfitting by limiting model complexity  

---

## 🚀 Deployment

* Production model saved: `deployment/production_model.joblib`  
* Prediction workflow supports new incident data using preprocessor and probability cutoff  
* Fully reproducible pipeline for operational use  

---

## 📁 Project Structure

```text
CyberThreats_FinancialLoss_Prediction_ML/
│── data/
│   ├── raw/          # Original dataset CSVs
│   ├── interim/      # Cleaned & selected features
│   └── processed/    # Step-wise processed data
│
│── notebooks/        # Step 1 → Step 8 notebooks
│── models/           # Trained models (.joblib)
│── reports/          # eda
│── README.md         # Project documentation

```
---

## 🌍 Business & Social Impact

* Cybersecurity risk assessment  
* High-loss incident forecasting  
* Incident response prioritization  
* Industry-specific risk profiling  
* Data-driven investment for improved cybersecurity  

---

## 🌟 Highlights

* ✅ End-to-end ML pipeline: preprocessing → modeling → deployment  
* 📈 CatBoost: Macro F1 = 0.527, ROC-AUC = 0.535  
* 🔍 Key features identified for interpretability  
* ⚙️ Production-ready with probability thresholding  
* 🔁 Fully reproducible and transparent workflow  

---

## 🛠 Tech Stack

* Python 3.10+  
* Pandas, NumPy  
* Scikit-Learn  
* CatBoost, LightGBM, XGBoost  
* Matplotlib, Seaborn  
* Jupyter Notebook  

---

## 📄 License

Educational & learning purposes only; not for commercial use.

---

## 👤 Author

Nithushan Uthayarasa
