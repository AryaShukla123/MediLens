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


def preprocess_diabetes_input(raw_data: dict) -> pd.DataFrame:
    
    df_input = pd.DataFrame([raw_data])

    df_ordered = df_input[feature_order]

    scaled_array = scaler.transform(df_ordered)

    return pd.DataFrame(scaled_array, columns=feature_order)


def predict_diabetes(raw_data: dict):
   
    X_processed = preprocess_diabetes_input(raw_data)

    prediction = int(model.predict(X_processed)[0])
    probability = float(model.predict_proba(X_processed)[0][1])

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "processed_features": X_processed,
    }
