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


def preprocess_heart_input(raw_data: dict) -> pd.DataFrame:
    """
    Takes raw form data (dict), aligns keys with the exact feature order,
    and applies the fitted StandardScaler.
    """
    # Convert input dict to DataFrame with correct column ordering
    df_input = pd.DataFrame([raw_data])
    
    # Ensure all required features exist and are strictly ordered
    df_ordered = df_input[feature_order]
    
    # Apply the pre-fitted scaler
    scaled_array = scaler.transform(df_ordered)
    
    # Return as DataFrame to retain feature names for SHAP analysis
    return pd.DataFrame(scaled_array, columns=feature_order)


def predict_heart_disease(raw_data: dict):
    """
    Processes raw input data, generates a prediction, probability score,
    and formats the output for the application.
    """
    X_processed = preprocess_heart_input(raw_data)
    
    prediction = int(model.predict(X_processed)[0])
    probability = float(model.predict_proba(X_processed)[0][1])
    
    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "processed_features": X_processed
    }