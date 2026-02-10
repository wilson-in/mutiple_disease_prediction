import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer

# ======================================================
# PATHS
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================================================
# FIND FILES
# ======================================================
def find_file(keyword):
    for f in os.listdir(DATA_DIR):
        if keyword in f.lower():
            return os.path.join(DATA_DIR, f)
    raise FileNotFoundError(f"No file found for {keyword}")

KIDNEY_FILE = find_file("kidney")
LIVER_FILE = find_file("liver")
PARK_FILE = find_file("parkinson")

metadata = {}

# ======================================================
# SAVE FUNCTION
# ======================================================
def save_model(name, model, scaler, imputer, features, defaults):
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{name}_scaler.pkl"))
    joblib.dump(imputer, os.path.join(MODEL_DIR, f"{name}_imputer.pkl"))

    metadata[name] = {
        "feature_order": features,
        "defaults": defaults,
        "model": f"{name}_model.pkl",
        "scaler": f"{name}_scaler.pkl",
        "imputer": f"{name}_imputer.pkl"
    }

# ======================================================
# 1️⃣ KIDNEY MODEL 
# ======================================================
print("Training Kidney model...")

kidney = pd.read_csv(KIDNEY_FILE)
kidney.columns = kidney.columns.str.lower().str.strip()

# normalize string cells
for col in kidney.columns:
    if kidney[col].dtype == object:
        kidney[col] = (
            kidney[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace("?", np.nan)
        )

# target
kidney["target"] = kidney["classification"].replace({
    "ckd": 1, "ckd\t": 1,
    "notckd": 0, "notckd\t": 0
})
kidney = kidney.dropna(subset=["target"])
y = kidney["target"].astype(int)

FEATURES = [c for c in [
    "age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo"
] if c in kidney.columns]

X = kidney[FEATURES].copy()

# force numeric
for col in FEATURES:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# defaults MUST come from raw numeric data
defaults = X.median(numeric_only=True).to_dict()

# impute → scale
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

Xtr, Xte, ytr, yte = train_test_split(
    X_scaled,
    y.loc[X.index],
    stratify=y.loc[X.index],
    test_size=0.2,
    random_state=42
)

model = CalibratedClassifierCV(
    LogisticRegression(max_iter=3000),
    method="sigmoid",
    cv=5
)
model.fit(Xtr, ytr)

save_model("Kidney", model, scaler, imputer, FEATURES, defaults)

# ======================================================
# 2️⃣ LIVER MODEL 
# ======================================================
print("Training Liver model...")

liver = pd.read_csv(LIVER_FILE)
target_col = liver.columns[-1]
y = liver[target_col].replace({1: 1, 2: 0})

FEATURES = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Albumin"
]

X = liver[FEATURES].copy()

# robust gender handling
X["Gender"] = (
    X["Gender"]
    .astype(str)
    .str.lower()
    .str.strip()
    .replace({
        "male": 1, "m": 1, "1": 1,
        "female": 0, "f": 0, "0": 0
    })
)

# force numeric BEFORE defaults
for col in FEATURES:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# defaults from raw numeric frame
defaults = X.median(numeric_only=True).to_dict()

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

Xtr, Xte, ytr, yte = train_test_split(
    X_scaled, y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

model = CalibratedClassifierCV(
    LogisticRegression(max_iter=3000),
    method="sigmoid",
    cv=5
)
model.fit(Xtr, ytr)

save_model("Liver", model, scaler, imputer, FEATURES, defaults)

# ======================================================
# 3️⃣ PARKINSONS MODEL
# ======================================================
print("Training Parkinsons model...")

park = pd.read_csv(PARK_FILE)
if "name" in park.columns:
    park = park.drop(columns=["name"])

y = park["status"]
X = park.drop(columns=["status"])

FEATURES = X.columns.tolist()

for col in FEATURES:
    X[col] = pd.to_numeric(X[col], errors="coerce")

defaults = X.median(numeric_only=True).to_dict()

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

Xtr, Xte, ytr, yte = train_test_split(
    X_scaled, y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

model = CalibratedClassifierCV(
    LogisticRegression(max_iter=3000),
    method="sigmoid",
    cv=5
)
model.fit(Xtr, ytr)

save_model("Parkinsons", model, scaler, imputer, FEATURES, defaults)

# ======================================================
# SAVE METADATA
# ======================================================
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ ALL MODELS TRAINED SUCCESSFULLY")