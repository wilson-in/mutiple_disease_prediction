import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ======================================================
# PROJECT ROOT
# ======================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================================================
# AUTO-DETECT FILES
# ======================================================
def find_file(keyword):
    for f in os.listdir(DATA_DIR):
        if keyword in f.lower():
            return f
    raise FileNotFoundError(f"No file found for {keyword} in data/cleaned")

KIDNEY_FILE = find_file("kidney")
LIVER_FILE = find_file("liver")
PARK_FILE = find_file("parkinson")

metadata = {}

def save(name, model, scaler, features):
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{name}_scaler.pkl"))
    metadata[name] = {
        "feature_order": features,
        "model": f"{name}_model.pkl",
        "scaler": f"{name}_scaler.pkl"
    }

# ======================================================
# 1️⃣ KIDNEY
# ======================================================
print("Training Kidney model...")

kidney = pd.read_csv(os.path.join(DATA_DIR, KIDNEY_FILE))
kidney = kidney.replace("?", np.nan).dropna()

# ---- FIXED LABEL HANDLING ----
y_raw = kidney["classification"].astype(str).str.lower().str.strip()
y_raw = y_raw.replace({"ckd\t": "ckd", "notckd\t": "notckd"})
y = y_raw.map({"ckd": 1, "notckd": 0})

mask = y.notna()
kidney = kidney.loc[mask]
y = y.loc[mask].astype(int)

FEATURES = ["age","bp","bgr","bu","sc","sod","pot","hemo","htn","dm"]
X = kidney[FEATURES].copy()

X["htn"] = X["htn"].map({"yes":1,"no":0})
X["dm"] = X["dm"].map({"yes":1,"no":0})

NUM = ["age","bp","bgr","bu","sc","sod","pot","hemo"]
BIN = ["htn","dm"]

scaler = StandardScaler()
Xn = scaler.fit_transform(X[NUM])
X_final = np.hstack([Xn, X[BIN].values])

Xtr,Xte,ytr,yte = train_test_split(
    X_final, y, stratify=y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=3000, C=0.3)
model.fit(Xtr,ytr)

save("Kidney", model, scaler, FEATURES)

# ======================================================
# 2️⃣ LIVER
# ======================================================
print("Training Liver model...")

liver = pd.read_csv(os.path.join(DATA_DIR, LIVER_FILE)).dropna()
y = liver.iloc[:, -1].map({1:1, 2:0})

FEATURES = [
    "Age","Gender",
    "Total_Bilirubin","Direct_Bilirubin",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Albumin"
]

X = liver[FEATURES].copy()
X["Gender"] = X["Gender"].map({"Male":1,"Female":0})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Xtr,Xte,ytr,yte = train_test_split(
    X_scaled, y, stratify=y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=3000, C=0.5)
model.fit(Xtr,ytr)

save("Liver", model, scaler, FEATURES)

# ======================================================
# 3️⃣ PARKINSONS
# ======================================================
print("Training Parkinsons model...")

park = pd.read_csv(os.path.join(DATA_DIR, PARK_FILE))
if "name" in park.columns:
    park = park.drop(columns=["name"])

y = park["status"]
X = park.drop(columns=["status"])
FEATURES = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Xtr,Xte,ytr,yte = train_test_split(
    X_scaled, y, stratify=y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=3000)
model.fit(Xtr,ytr)

save("Parkinsons", model, scaler, FEATURES)

# ======================================================
# SAVE METADATA
# ======================================================
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ ALL MODELS TRAINED SUCCESSFULLY")
