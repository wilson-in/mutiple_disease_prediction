import os
import json
import joblib
import streamlit as st
import pandas as pd

# ---------------------------
# Resolve project base + models folder
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ---------------------------
# Utility functions
# ---------------------------
def load_models_and_metadata(model_dir=MODEL_DIR):
    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None, f"❌ metadata.json not found in {model_dir}. Run train_models.py first."

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    models = {}
    scalers = {}
    for disease, info in metadata.items():
        model_path = os.path.join(model_dir, info["model"])
        scaler_path = os.path.join(model_dir, info["scaler"])

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, f"❌ Missing model or scaler for {disease}. Run train_models.py again."

        models[disease] = joblib.load(model_path)
        scalers[disease] = joblib.load(scaler_path)

    return {"metadata": metadata, "models": models, "scalers": scalers}, None

def encode_input(df, feature_order, categorical_mappings):
    # Ensure all expected columns exist
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_order].copy()

    # Apply categorical mappings
    for col, mapping in (categorical_mappings or {}).items():
        if col in df.columns:
            val = str(df.at[0, col])
            if val in mapping:
                df[col] = mapping[val]
            else:
                df[col] = 0  # fallback for unseen values
    return df

# ---------------------------
# Load Models + Metadata
# ---------------------------
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide")
st.title("🩺 Multiple Disease Prediction System")

loaded, err = load_models_and_metadata(MODEL_DIR)
if err:
    st.error(err)
    st.stop()

metadata = loaded["metadata"]
models = loaded["models"]
scalers = loaded["scalers"]

# Sidebar
option = st.sidebar.selectbox("Select Disease", list(metadata.keys()))

# ---------------------------
# Prediction UI
# ---------------------------
def prediction_ui(disease):
    st.header(f"{disease} Disease Prediction")

    meta = metadata[disease]
    feature_order = meta["feature_order"]
    cat_maps = meta.get("categorical_mappings", {})

    # Build input widgets dynamically
    inputs = {}
    for feature in feature_order:
        if feature in cat_maps:
            choices = list(cat_maps[feature].keys())
            inputs[feature] = st.selectbox(feature, choices)
        else:
            inputs[feature] = st.number_input(feature, value=0.0)

    if st.button(f"Predict {disease} Disease"):
        df = pd.DataFrame([inputs])
        df_enc = encode_input(df, feature_order, cat_maps)
        scaled = scalers[disease].transform(df_enc)
        prediction = models[disease].predict(scaled)[0]

        if prediction == 1:
            st.success(f"✅ {disease} Disease Detected")
        else:
            st.success(f"❌ No {disease} Disease")

        img_path = os.path.join(MODEL_DIR, f"{disease.lower()}_confusion.png")
        if os.path.exists(img_path):
            st.image(img_path, caption=f"{disease} Confusion Matrix")

# Run UI for selected disease
prediction_ui(option)
