
# 📊 Customer Churn Prediction Pipeline

This project builds an end-to-end machine learning pipeline to predict customer churn using the Telco dataset. It includes preprocessing, hyperparameter tuning, model training, evaluation, explainability, and experiment tracking.

---

## 🚀 Features

- Data preprocessing with ColumnTransformer
- Hyperparameter optimization using Optuna
- Model training with XGBoost
- Evaluation using AUC, F1-score, and Accuracy
- Experiment tracking with MLflow
- Model explainability using SHAP
- Pipeline + artifacts saved for reuse

---

## 📁 Project Structure

.
├── data/
│   └── telco_churn.csv
├── models/
│   └── churn_pipeline.pkl
├── main.py
└── README.md

---

## ⚙️ Installation

pip install pandas numpy scikit-learn xgboost optuna shap mlflow

---

## 📌 Usage

python main.py

---

## 🧠 Workflow Overview

### 1. Data Loading
- Loads dataset from data/telco_churn.csv

### 2. Preprocessing
- Converts TotalCharges to numeric
- Drops missing values and customerID
- Encodes categorical variables using OneHotEncoder

### 3. Model Training + Optimization
- Uses Optuna to tune XGBoost hyperparameters
- 5-fold Stratified Cross Validation
- Optimizes for ROC-AUC score

### 4. Final Model
- Trains model with best parameters
- Applies probability threshold (0.3) for classification

### 5. Evaluation Metrics
- ROC-AUC
- F1 Score
- Accuracy

### 6. Experiment Tracking
- Logs parameters, metrics, and model to MLflow

### 7. Explainability
- Uses SHAP to compute feature importance

### 8. Saving Artifacts
- Saves:
  - Full pipeline
  - SHAP explainer
  - Feature names
  - Best parameters

---

## 📊 Output Example

Best AUC: 0.84  
AUC: 0.85  
F1 : 0.72  
Acc: 0.80  

---

## 📦 Saved Artifacts

models/churn_pipeline.pkl

Contains:
- Trained pipeline
- SHAP explainer
- Feature names
- Best hyperparameters

---

## 🔍 Explainability (SHAP)

SHAP values are computed on a sample of test data to interpret model predictions.

---

## 📈 MLflow Tracking

mlflow ui

Then open:
http://localhost:5000

---

## 🛠️ Future Improvements

- Add model deployment (FastAPI / Flask)
- Add feature selection
- Automate threshold tuning
- Use distributed training for large datasets

---

## 📚 Reference

Script source included in project.

EOF
