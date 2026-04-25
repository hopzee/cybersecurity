import shap
import joblib

def explain_model(X_sample):
    model = joblib.load("models/model.pkl")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    return shap_values