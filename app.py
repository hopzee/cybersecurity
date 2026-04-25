import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("AI Cybersecurity Vulnerability Simulator (10 Features)")

model = joblib.load("models/model.pkl")

# expected features
FEATURES = [f"f{i}" for i in range(1, 11)]

file = st.file_uploader("Upload CSV (must have 10 features)", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("Raw Dataset")
    st.write(df.head())

    # 🔥 Force numeric only
    df = df.select_dtypes(include=["number"])

    # Ensure correct columns exist
    missing = [f for f in FEATURES if f not in df.columns]

    if missing:
        st.error(f"Missing required features: {missing}")
        st.stop()

    # reorder correctly
    X = df[FEATURES].values

    st.success("Dataset aligned with model (10 features OK)")

    # ---------------- BASELINE ----------------
    preds = model.predict(X)

    st.subheader("Predictions")
    st.write(preds)

    # ---------------- ATTACK SIMULATION ----------------
    noise = np.random.normal(0, 0.1, X.shape)
    X_attack = X + noise

    preds_attack = model.predict(X_attack)

    st.subheader("After Adversarial Attack")
    st.write(preds_attack)