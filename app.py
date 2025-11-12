# app.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Paths
MODEL_PATH = Path("model.joblib")
METADATA_PATH = Path("metadata.json")

st.set_page_config(page_title="Expresso Churn Predictor", layout="centered")

st.title("📡 Expresso Churn Predictor")
st.write("Fill the fields and click **Predict**. This app uses a trained model (RandomForest).")

# Ensure model and metadata exist
if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model or metadata not found. Run `train_model.py` first.")
    st.stop()

# Load model and metadata
model = joblib.load(MODEL_PATH)
with open(METADATA_PATH, "r") as f:
    meta = json.load(f)

numeric_feats = meta.get("_numeric_features", [])
categorical_feats = meta.get("_categorical_features", [])
features_order = meta.get("_features_order", numeric_feats + categorical_feats)

st.sidebar.header("Load sample inputs")
# Optionally load sample defaults from metadata if you saved them
# Build input form
with st.form("input_form"):
    st.write("### Input features")
    inputs = {}
    # Categorical inputs
    for c in categorical_feats:
        opts = meta.get(c, [])
        if len(opts) == 0:
            # fallback to text input
            val = st.text_input(c, value="")
            inputs[c] = val
        else:
            # Allow an empty choice to represent missing
            sel = st.selectbox(c, options=["__missing__"] + opts, index=0)
            inputs[c] = sel if sel != "__missing__" else np.nan

    # Numeric inputs
    for n in numeric_feats:
        # Provide a numeric input with default as median if possible
        default = None
        try:
            # try to get median from model if transformer stores it — otherwise default None
            default = None
        except Exception:
            default = None
        # Leave wide range; user can type
        val = st.number_input(n, value=float(default) if default is not None else 0.0, format="%.6f")
        inputs[n] = float(val)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Create dataframe in the same columns order the model expects
    X_input = pd.DataFrame([inputs], columns=features_order)
    # Convert empty strings to NaN
    X_input = X_input.replace({"": np.nan})
    try:
        proba = model.predict_proba(X_input)[:, 1][0]
        pred = int(model.predict(X_input)[0])
        st.write("## Prediction")
        st.write(f"Predicted churn: **{pred}**")
        st.write(f"Churn probability: **{proba:.4f}**")
        st.progress(int(proba*100))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Make sure you trained the model with `train_model.py` and that metadata.json matches the model.")
