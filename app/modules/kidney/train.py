"""
Offline training script for the Chronic Kidney Disease module.

Unlike Heart/Diabetes, this dataset needed real cleaning before it was
usable (see notes below) — so this script does the full raw-CSV-to-
artifacts pipeline in one place, matching what was worked out in
notebooks/kidney_eda.ipynb.

Data quality notes (see that notebook for the full investigation):
  - Rows 0-1 of the raw CSV are junk metadata, not data.
  - 'affected' and 'stage' are direct derivatives of the target
    diagnosis (data leakage) and are dropped.
  - 'age', 'al', 'su' were corrupted by Excel's automatic date
    conversion (e.g. a range like "12-20" became the string "20-Dec").
    'age' is fixed via a confident contextual reconstruction; 'al'/'su'
    are ordinal severity grades, so they're re-encoded by their known
    increasing-severity rank rather than by guessing the exact
    original numeric boundaries.
  - Every other clinical column is pre-binned into range strings
    (e.g. "112 - 154", "< 3.65") and is converted to a single
    representative numeric value (the bin midpoint, or the boundary
    itself for open-ended bins).

Run this manually to (re)generate model.pkl / scaler.pkl /
feature_order.json:
    python -m app.modules.kidney.train
"""

import os
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "..", "data", "raw", "kidney.csv")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURE_ORDER_PATH = os.path.join(BASE_DIR, "feature_order.json")

RANDOM_STATE = 42

RANGE_COLS = ["sg", "bgr", "bu", "sod", "sc", "pot", "hemo", "pcv", "rbcc", "wbcc", "grf"]
BINARY_COLS = ["bp (Diastolic)", "bp limit", "rbc", "pc", "pcc", "ba",
               "htn", "dm", "cad", "appet", "pe", "ane"]

# Ordinal severity grades, in known increasing-severity order — the exact
# numeric bin boundaries were lost to Excel's date-mangling, but the
# category order survives, which is what the ordinal encoding needs.
AL_ORDER = ["< 0", "1-Jan", "2-Feb", "3-Mar", "≥ 4"]
SU_ORDER = ["< 0", "2-Jan", "2-Feb", "4-Mar", "4-Apr", "≥ 4"]


def parse_bin(value):
    """Converts a bin-range string like '112 - 154', '< 48.1', or
    '≥ 227.944' into a single representative numeric value."""
    if pd.isna(value):
        return np.nan
    value = str(value).strip()

    if value.startswith("<"):
        return float(value.replace("<", "").strip())
    if value.startswith("≥") or value.startswith(">="):
        return float(value.replace("≥", "").replace(">=", "").strip())
    if "-" in value:
        parts = value.split("-")
        try:
            lo, hi = float(parts[0].strip()), float(parts[1].strip())
            return (lo + hi) / 2
        except ValueError:
            return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.iloc[2:].reset_index(drop=True)  # drop junk metadata rows

    # Drop leakage columns, encode target
    df = df.drop(columns=["affected", "stage"])
    df["class"] = df["class"].map({"ckd": 1, "notckd": 0})

    # Fix the one confidently-recoverable corrupted age bin
    df["age"] = df["age"].replace({"20-Dec": "12 - 20"})

    # Parse true numeric-range columns to their midpoint
    for col in RANGE_COLS:
        df[col] = df[col].apply(parse_bin)
    df[RANGE_COLS] = df[RANGE_COLS].fillna(df[RANGE_COLS].median())

    # Ordinal-encode al/su by known severity rank
    al_map = {label: rank for rank, label in enumerate(AL_ORDER)}
    su_map = {label: rank for rank, label in enumerate(SU_ORDER)}
    df["al"] = df["al"].map(al_map)
    df["su"] = df["su"].map(su_map)

    # age is itself a bin-range column once the corrupted entry is fixed
    df["age"] = df["age"].apply(parse_bin)

    df[BINARY_COLS] = df[BINARY_COLS].astype(int)

    print(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Class balance:\n", df["class"].value_counts())
    return df


def main():
    df = load_and_clean()

    X = df.drop(columns=["class"])
    y = df["class"]
    feature_order = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_order)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_order)

    # Random Forest confirmed as the winner in notebooks/kidney_eda.ipynb
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    print("\n--- Random Forest ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"Recall (class 1): {recall_score(y_test, preds):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]):.4f}")
    print(classification_report(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_ORDER_PATH, "w") as f:
        json.dump(feature_order, f)

    print(f"\nSaved: {MODEL_PATH}")
    print(f"Saved: {SCALER_PATH}")
    print(f"Saved: {FEATURE_ORDER_PATH}")


if __name__ == "__main__":
    main()
