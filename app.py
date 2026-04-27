import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

st.title("🔐 AI Cybersecurity Vulnerability Simulator")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

try:
    model = load_model()
except:
    st.error("❌ Model not found. Please upload model.pkl")
    st.stop()

# -------------------------------
# UPLOAD DATA
# -------------------------------
file = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # -------------------------------
    # CHECK LABEL
    # -------------------------------
    if "label" not in df.columns:
        st.error("❌ Dataset must contain 'label' column")
        st.stop()

    # -------------------------------
    # USE ALL FEATURES EXCEPT LABEL
    # -------------------------------
    X = df.drop("label", axis=1)
    y = df["label"]

    # -------------------------------
    # DEBUG INFO
    # -------------------------------
    st.write("🔍 Model expects:", model.n_features_in_)
    st.write("🔍 Input features:", X.shape[1])

    if X.shape[1] != model.n_features_in_:
        st.error("❌ Feature mismatch! Dataset does not match trained model")
        st.stop()

    # -------------------------------
    # BASELINE
    # -------------------------------
    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0

    st.subheader("✅ Baseline Performance")
    st.write(f"Accuracy: {acc:.4f}")
    st.write(f"False Negative Rate: {fnr:.4f}")

    # -------------------------------
    # STRONG ADVERSARIAL ATTACK
    # -------------------------------
    noise = np.random.normal(0, 2.0, X.shape)
    X_adv = X + noise

    # Extra distortion
    X_adv = X_adv * np.random.uniform(0.5, 1.5)

    # Clip values
    X_adv = np.clip(X_adv, X.min().min(), X.max().max())

    preds_adv = model.predict(X_adv)
    acc_adv = accuracy_score(y, preds_adv)

    tn, fp, fn, tp = confusion_matrix(y, preds_adv).ravel()
    fnr_adv = fn / (fn + tp) if (fn + tp) != 0 else 0

    st.subheader("⚠️ After Adversarial Attack")
    st.write(f"Accuracy: {acc_adv:.4f}")
    st.write(f"False Negative Rate: {fnr_adv:.4f}")

    # -------------------------------
    # DEFENSE (LIGHT FIX WITHOUT RETRAINING)
    # -------------------------------
    # Reduce noise slightly to simulate recovery
    X_def = X_adv * 0.7

    preds_def = model.predict(X_def)
    acc_def = accuracy_score(y, preds_def)

    tn, fp, fn, tp = confusion_matrix(y, preds_def).ravel()
    fnr_def = fn / (fn + tp) if (fn + tp) != 0 else 0

    st.subheader("🛡️ After Defense (Simulated Recovery)")
    st.write(f"Accuracy: {acc_def:.4f}")
    st.write(f"False Negative Rate: {fnr_def:.4f}")

    # -------------------------------
    # PERFORMANCE SUMMARY
    # -------------------------------
    st.subheader("📊 Performance Summary")

    st.write("Baseline Accuracy:", acc)
    st.write("Attack Accuracy:", acc_adv)
    st.write("Defense Accuracy:", acc_def)

    st.write("Baseline FNR:", fnr)
    st.write("Attack FNR:", fnr_adv)
    st.write("Defense FNR:", fnr_def)

    # -------------------------------
    # PREDICTION COMPARISON
    # -------------------------------
    st.subheader("🔍 Prediction Comparison")

    st.write("Before Attack:", preds[:10])
    st.write("After Attack:", preds_adv[:10])

    # -------------------------------
    # LABEL DISTRIBUTION
    # -------------------------------
    st.subheader("📌 Label Distribution")
    st.write(df["label"].value_counts())

    # -------------------------------
    # GRAPH
    # -------------------------------
    st.subheader("📉 Accuracy Comparison")

    labels = ["Baseline", "Attack", "Defense"]
    values = [acc, acc_adv, acc_def]

    plt.figure()
    plt.bar(labels, values)
    st.pyplot(plt)
