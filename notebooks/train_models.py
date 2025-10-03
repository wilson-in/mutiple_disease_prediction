import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# ---------------------------
# Paths setup
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def project_path(*parts):
    return os.path.join(BASE_DIR, *parts)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_confusion(cm, disease_name):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{disease_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    fname = project_path("models", f"{disease_name.lower()}_confusion.png")
    plt.savefig(fname)
    plt.close()
    return fname

def evaluate_model(model, X_test, y_test, disease_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    except Exception:
        roc = None
    cm = confusion_matrix(y_test, y_pred)
    save_confusion(cm, disease_name)
    return {"accuracy": acc, "roc_auc": roc}

def load_prefer_cleaned(raw_path, cleaned_path):
    if os.path.exists(cleaned_path):
        print(f"Loading cleaned: {cleaned_path}")
        return pd.read_csv(cleaned_path)
    elif os.path.exists(raw_path):
        print(f"Loading raw: {raw_path}")
        return pd.read_csv(raw_path)
    else:
        raise FileNotFoundError(f"Neither {cleaned_path} nor {raw_path} exist.")

# ---------------------------
# Setup
# ---------------------------
ensure_dir(project_path("models"))
metadata = {}

# ---------------------------
# 1) Kidney
# ---------------------------
print("🔹 Training Kidney Disease Model...")

kidney = load_prefer_cleaned(
    project_path("data", "raw", "kidney_disease.csv"),
    project_path("data", "cleaned", "kidney_preprocessed.csv")
)

kidney = kidney.replace("?", np.nan).dropna()

if "id" in kidney.columns:
    kidney = kidney.drop(columns=["id"])

if "classification" not in kidney.columns:
    raise KeyError("Kidney dataset must contain 'classification' column")

y_raw = kidney["classification"].astype(str).str.lower().str.strip()

if set(y_raw.unique()) <= {"ckd", "notckd"}:
    y_kidney = y_raw.map({"ckd": 1, "notckd": 0})
elif set(y_raw.unique()) <= {"1", "0"}:
    y_kidney = y_raw.astype(int)
elif set(y_raw.unique()) <= {"yes", "no"}:
    y_kidney = y_raw.map({"yes": 1, "no": 0})
else:
    raise ValueError(f"Unexpected kidney classification labels: {y_raw.unique()}")

mask = ~y_kidney.isna()
kidney = kidney.loc[mask]
y_kidney = y_kidney.loc[mask]

X_kidney = kidney.drop(columns=["classification"])

# Encode categoricals
kidney_cat_mappings = {}
for col in X_kidney.select_dtypes(include=["object", "category"]).columns:
    le = LabelEncoder()
    X_kidney[col] = le.fit_transform(X_kidney[col].astype(str))
    kidney_cat_mappings[col] = {str(val): int(code) for val, code in zip(le.classes_, le.transform(le.classes_))}

kidney_feature_order = X_kidney.columns.tolist()

scaler_kidney = StandardScaler()
X_kidney_scaled = scaler_kidney.fit_transform(X_kidney)

X_train, X_test, y_train, y_test = train_test_split(X_kidney_scaled, y_kidney, test_size=0.2, random_state=42)

kidney_model = RandomForestClassifier(random_state=42)
kidney_model.fit(X_train, y_train)

joblib.dump(kidney_model, project_path("models", "kidney_model.pkl"))
joblib.dump(scaler_kidney, project_path("models", "kidney_scaler.pkl"))

kidney_eval = evaluate_model(kidney_model, X_test, y_test, "Kidney")
metadata["Kidney"] = {
    "feature_order": kidney_feature_order,
    "categorical_mappings": kidney_cat_mappings,
    "model": "kidney_model.pkl",
    "scaler": "kidney_scaler.pkl"
}

# ---------------------------
# 2) Liver
# ---------------------------
print("🔹 Training Liver Disease Model...")

liver = load_prefer_cleaned(
    project_path("data", "raw", "indian_liver_patient.csv"),
    project_path("data", "cleaned", "liver_preprocessed.csv")
)

liver = liver.dropna()

if "Dataset" not in liver.columns:
    raise KeyError("Liver dataset must contain 'Dataset' column")

y_raw = liver["Dataset"].astype(str).str.strip()

if set(y_raw.unique()) <= {"1", "2"}:
    y_liver = y_raw.map({"1":1, "2":0})
elif set(y_raw.unique()) <= {"0", "1"}:
    y_liver = y_raw.astype(int)
else:
    raise ValueError(f"Unexpected liver Dataset labels: {y_raw.unique()}")

mask = ~y_liver.isna()
liver = liver.loc[mask]
y_liver = y_liver.loc[mask]

# Encode Gender
le_gender = LabelEncoder()
liver["Gender"] = le_gender.fit_transform(liver["Gender"].astype(str))
liver_gender_mapping = {str(val): int(code) for val, code in zip(le_gender.classes_, le_gender.transform(le_gender.classes_))}

X_liver = liver.drop(columns=["Dataset"])
liver_feature_order = X_liver.columns.tolist()

scaler_liver = StandardScaler()
X_liver_scaled = scaler_liver.fit_transform(X_liver)

X_train, X_test, y_train, y_test = train_test_split(X_liver_scaled, y_liver, test_size=0.2, random_state=42)

liver_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
liver_model.fit(X_train, y_train)

joblib.dump(liver_model, project_path("models", "liver_model.pkl"))
joblib.dump(scaler_liver, project_path("models", "liver_scaler.pkl"))

liver_eval = evaluate_model(liver_model, X_test, y_test, "Liver")
metadata["Liver"] = {
    "feature_order": liver_feature_order,
    "categorical_mappings": {"Gender": liver_gender_mapping},
    "model": "liver_model.pkl",
    "scaler": "liver_scaler.pkl"
}

# ---------------------------
# 3) Parkinsons
# ---------------------------
print("🔹 Training Parkinson's Disease Model...")

parkinsons = load_prefer_cleaned(
    project_path("data", "raw", "parkinsons.csv"),
    project_path("data", "cleaned", "parkinsons_preprocessed.csv")
)

if "status" not in parkinsons.columns:
    raise KeyError("Parkinsons dataset must contain 'status' column")

if "name" in parkinsons.columns:
    parkinsons = parkinsons.drop(columns=["name"])

y_parkinson = parkinsons["status"].astype(int)
mask = ~y_parkinson.isna()
parkinsons = parkinsons.loc[mask]
y_parkinson = y_parkinson.loc[mask]

X_parkinson = parkinsons.drop(columns=["status"])
parkinson_feature_order = X_parkinson.columns.tolist()

scaler_parkinson = StandardScaler()
X_parkinson_scaled = scaler_parkinson.fit_transform(X_parkinson)

X_train, X_test, y_train, y_test = train_test_split(X_parkinson_scaled, y_parkinson, test_size=0.2, random_state=42)

parkinson_model = LogisticRegression(max_iter=500)
parkinson_model.fit(X_train, y_train)

joblib.dump(parkinson_model, project_path("models", "parkinsons_model.pkl"))
joblib.dump(scaler_parkinson, project_path("models", "parkinsons_scaler.pkl"))

parkinson_eval = evaluate_model(parkinson_model, X_test, y_test, "Parkinsons")
metadata["Parkinsons"] = {
    "feature_order": parkinson_feature_order,
    "categorical_mappings": {},
    "model": "parkinsons_model.pkl",
    "scaler": "parkinsons_scaler.pkl"
}

# ---------------------------
# Save reports
# ---------------------------
results = {
    "Kidney": kidney_eval,
    "Liver": liver_eval,
    "Parkinsons": parkinson_eval
}

with open(project_path("models", "training_reports.json"), "w") as f:
    json.dump(results, f, indent=4)

with open(project_path("models", "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("\n✅ Training Completed! Models & metadata saved in /models/")
