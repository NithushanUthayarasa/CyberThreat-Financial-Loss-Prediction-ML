# 🔐 Cyber Threats & Financial Loss Prediction (2015–2024)

An **end-to-end machine learning project** predicting high financial losses from cyber threats using structured, **data-leak-free** global incident data. The pipeline covers preprocessing, feature engineering, training multiple models, hyperparameter tuning, feature importance analysis, benchmarking, and deployment of the best-performing model for production-ready predictions.

---

## 📌 Project Overview

Cybersecurity incidents increasingly cause significant financial losses across industries. Estimating losses is challenging due to complex, non-linear factors such as:

* Attack type
* Affected industry
* Incident resolution time
* Number of affected users

**This project provides:**

* High-risk cyber incident prediction (binary classification)
* Key feature identification driving financial loss
* Model benchmarking for robust predictions
* Deployment of the best model for production use

---

## 🚨 Problem Statement

Organizations often struggle to quantify financial losses from cyber attacks because traditional risk assessment methods cannot capture complex, non-linear relationships between:

* Rapidly evolving cyber threat types
* Variations in incident resolution time
* Industry-specific vulnerabilities and defense mechanisms

As a result, businesses may face inefficient resource allocation, delayed incident response, and misaligned cybersecurity investments.

**Solution:** Frame the challenge as a binary classification task to predict high-risk cyber incidents and support proactive, data-driven decision-making.

---

## 🎯 Purpose

Provide organizations with a risk prediction system to:

* Prioritize high-risk cyber incidents
* Allocate cybersecurity resources efficiently
* Reduce financial and operational impact
* Support data-driven cybersecurity investments

---

## 🧾 Dataset & Features
**Source:** [Kaggle - Global Cybersecurity Threats, 2015–2024](https://www.kaggle.com/datasets/atharvasoundankar/global-cybersecurity-threats-2015-2024)
* **Size:** ~3,000 incidents  

**Features:**

* Number of Affected Users
* Incident Resolution Time (in Hours)
* Attack Type
* Target Industry
* Attack Source
* Security Vulnerability Type
* Interaction Feature: `AttackType_TargetIndustry`
* Users_per_Hour, Log_Users

> Post-processing: 25–65 features are used depending on the model.

---

## 🔄 End-to-End ML Pipeline

* Feature Selection & Binary Target Creation – High-loss threshold to define binary target
* Preprocessing Pipeline – Scale numeric features, encode categoricals, split train/test datasets
* Baseline Model Training – RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost
* Hyperparameter Tuning & Probability Cutoff – Optimize models and tune probability thresholds
* Baseline vs Tuned Model Comparison – Evaluate Accuracy, Macro F1, Macro Recall
* Feature Importance & Analysis – Aggregate expanded features to main business-level features
* Model Benchmarking – Train, test, cross-validate; compare metrics and inference time
* Deployment – Save best model (`production_model.joblib`) and implement prediction workflow

---

## 🧠 Feature Engineering & Preprocessing

* Handle missing values (median for numeric, mode for categorical)
* Create interaction features (`AttackType_TargetIndustry`)
* Scale numeric variables; one-hot encode categoricals
* Train/test split: 4:1 ratio
* Binary target defined via high-loss percentile threshold

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

**Model Comparison (Test Set)**

| Model       | Accuracy | Macro F1 | Macro Recall |
|------------|---------|-----------|-------------|
| CatBoost   | 0.545   | 0.527     | 0.531       |
| RandomForest | 0.545 | 0.512     | 0.524       |
| LightGBM   | 0.520   | 0.509     | 0.510       |
| XGBoost    | 0.525   | 0.503     | 0.509       |
| ExtraTrees | 0.503   | 0.487     | 0.490       |

**Best Model:** CatBoost (based on Macro F1)

**Key Features Driving Predictions:**

* Number of Affected Users
* Incident Resolution Time (Hours)
* AttackType_TargetIndustry
* Attack Source

---

## 🛠 Handling High Variance & Bias

* Tuned regularization parameters (e.g., `l2_leaf_reg`, `max_depth`)
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
│── notebooks/        # Step 1–8: Jupyter notebooks
│── models/           # Trained models (.joblib)
│── reports/          # EDA & benchmarking
│── README.md         # Project documentation

```
---

# 🔧 How to Run the Project

```bash
git clone https://github.com/NithushanUthayarasa/CyberThreat-Financial-Loss-Prediction-ML
cd CyberThreats_FinancialLoss_Prediction_ML
pip install -r requirements.txt
jupyter notebook
```

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
