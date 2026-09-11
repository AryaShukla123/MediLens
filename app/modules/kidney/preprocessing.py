import os
import json
import joblib
import pandas as pd

# Load saved artifacts relative to this module's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURE_ORDER_PATH = os.path.join(BASE_DIR, "feature_order.json")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURE_ORDER_PATH, "r") as f:
    feature_order = json.load(f)


def preprocess_kidney_input(raw_data: dict) -> pd.DataFrame:
    """
    Takes raw form data (dict) already in clean numeric form — the form
    collects actual values (age in years, blood pressure category, lab
    numbers, etc.), not the original dataset's binned range strings —
    aligns keys with the exact feature order, and applies the fitted
    StandardScaler.
    """
    df_input = pd.DataFrame([raw_data])
    df_ordered = df_input[feature_order]

    scaled_array = scaler.transform(df_ordered)

    return pd.DataFrame(scaled_array, columns=feature_order)


def predict_kidney_disease(raw_data: dict):
    """
    Processes raw input data, generates a prediction, probability score,
    and formats the output for the application.
    """
    X_processed = preprocess_kidney_input(raw_data)

    prediction = int(model.predict(X_processed)[0])
    probability = float(model.predict_proba(X_processed)[0][1])

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "processed_features": X_processed
    }
