import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs("models", exist_ok=True)

def generate_dataset(n=500):
    np.random.seed(42)

    X = np.random.rand(n, 10)  # 🔥 10 FEATURES ONLY
    y = (X.sum(axis=1) > 5).astype(int)  # simple attack logic

    cols = [f"f{i}" for i in range(1, 11)]

    df = pd.DataFrame(X, columns=cols)
    df["label"] = y

    return df

def train_model():
    df = generate_dataset()

    X = df.drop("label", axis=1).values
    y = df["label"].values

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "models/model.pkl")

    print("Model trained on 10 features successfully")

if __name__ == "__main__":
    train_model()