import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Multiple Disease Prediction",
    layout="wide"
)

st.title("🩺 Multiple Disease Prediction System")
st.caption("Educational ML-based disease risk estimation")

# ======================================================
# PATHS (DO NOT CHANGE)
# ======================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ======================================================
# LOAD MODELS & METADATA
# ======================================================
@st.cache_resource
def load_assets():
    metadata_path = os.path.join(MODEL_DIR, "metadata.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"metadata.json not found at {metadata_path}. "
            "Run train_models.py first."
        )

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    models = {}
    scalers = {}
    imputers = {}

    for disease, cfg in meta.items():
        models[disease] = joblib.load(os.path.join(MODEL_DIR, cfg["model"]))
        scalers[disease] = joblib.load(os.path.join(MODEL_DIR, cfg["scaler"]))
        imputers[disease] = joblib.load(os.path.join(MODEL_DIR, cfg["imputer"]))

    return meta, models, scalers, imputers

meta, models, scalers, imputers = load_assets()

# ======================================================
# LABEL DEFINITIONS (ONLY UI CHANGE)
# ======================================================
LABELS = {
    "age": "Age (years: 1–100)",
    "bp": "Blood Pressure (mm Hg: 50–180)",
    "bgr": "Blood Glucose Random (mg/dL: 70–300)",
    "bu": "Blood Urea (mg/dL: 5–150)",
    "sc": "Serum Creatinine (mg/dL: 0.4–15.0)",
    "sod": "Sodium Level (mEq/L: 120–160)",
    "pot": "Potassium Level (mEq/L: 2.5–7.0)",
    "hemo": "Hemoglobin (g/dL: 5–20)",

    "Age": "Age (years: 1–100)",
    "Gender": "Gender (Male = 1, Female = 0)",
    "Total_Bilirubin": "Total Bilirubin (mg/dL: 0.1–30.0)",
    "Direct_Bilirubin": "Direct Bilirubin (mg/dL: 0.0–20.0)",
    "Alamine_Aminotransferase": "Alanine Aminotransferase (ALT) (IU/L: 5–500)",
    "Aspartate_Aminotransferase": "Aspartate Aminotransferase (AST) (IU/L: 5–500)",
    "Albumin": "Albumin (g/dL: 2.0–6.0)"
}

# ======================================================
# SIDEBAR
# ======================================================
disease = st.sidebar.selectbox(
    "Select Disease",
    list(meta.keys())
)

st.header(f"{disease} Prediction")

# ======================================================
# INPUT FORM (LABELS ONLY MODIFIED)
# ======================================================
features = meta[disease]["feature_order"]
defaults = meta[disease]["defaults"]

inputs = {}
cols = st.columns(2)

for i, feature in enumerate(features):
    with cols[i % 2]:
        label = LABELS.get(feature, feature)
        inputs[feature] = st.number_input(
            label=label,
            value=float(defaults.get(feature, 0.0)),
            step=0.1
        )

# ======================================================
# PREDICTION (UNCHANGED)
# ======================================================
if st.button("Predict"):
    df = pd.DataFrame([inputs])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X_imputed = imputers[disease].transform(df)
    X_scaled = scalers[disease].transform(X_imputed)

    prob = models[disease].predict_proba(X_scaled)[0, 1]
    percentage = prob * 100

    if percentage < 25:
        risk = "Low Risk"
    elif percentage < 60:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    st.metric("Estimated Risk Probability", f"{percentage:.2f}%")
    st.info(f"Risk Level: {risk}")

# ======================================================
# FOOTER
# ======================================================
st.caption(
    "⚠️ Educational use only. "
    "This system does not provide medical diagnoses."
)