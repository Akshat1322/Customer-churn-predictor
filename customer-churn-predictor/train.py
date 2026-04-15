import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from xgboost import XGBClassifier
import optuna
import shap
import mlflow
import mlflow.xgboost

# ── 1. LOAD DATA ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/telco_churn.csv")

# ── 2. PREPROCESSING ─────────────────────────────────────────
print("Preprocessing...")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df.drop(columns=["customerID"], inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify columns
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ── 3. OPTUNA ───────────────────────────────────────────────
print("\nRunning Optuna...")

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 4),
        "tree_method": "hist",
        "random_state": 42,
        "eval_metric": "logloss"
    }

    model = XGBClassifier(**params)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipe.fit(
            X_tr, y_tr,
            model__eval_set=[(preprocessor.fit_transform(X_val), y_val)],
            model__early_stopping_rounds=30,
            model__verbose=False
        )

        prob = pipe.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, prob))

    return np.mean(aucs)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print(f"Best AUC: {study.best_value:.4f}")
best_params = study.best_params

# ── 4. FINAL MODEL + MLFLOW ─────────────────────────────────
print("\nTraining final model...")

mlflow.set_experiment("Customer_Churn_Pipeline")

with mlflow.start_run():

    final_model = XGBClassifier(
        **best_params,
        tree_method="hist",
        random_state=42,
        eval_metric="logloss"
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", final_model)
    ])

    pipe.fit(X_train, y_train)

    # Predictions
    prob = pipe.predict_proba(X_test)[:, 1]

    # Threshold tuning
    threshold = 0.3
    pred = (prob > threshold).astype(int)

    auc = roc_auc_score(y_test, prob)
    f1 = f1_score(y_test, pred)
    acc = accuracy_score(y_test, pred)

    print(f"AUC: {auc:.4f}")
    print(f"F1 : {f1:.4f}")
    print(f"Acc: {acc:.4f}")

    # MLflow logging
    mlflow.log_params(best_params)
    mlflow.log_metric("auc", auc)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(pipe, "model")

# ── 5. SHAP (Optimized) ─────────────────────────────────────
print("\nComputing SHAP...")

X_sample = X_test.sample(500, random_state=42)
X_sample_transformed = preprocessor.fit_transform(X_sample)

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_sample_transformed)

# ── 6. SAVE ARTIFACTS ───────────────────────────────────────
os.makedirs("models", exist_ok=True)

artifacts = {
    "pipeline": pipe,
    "explainer": explainer,
    "feature_names": X.columns.tolist(),
    "best_params": best_params
}

with open("models/churn_pipeline.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("\nSaved to models/churn_pipeline.pkl")