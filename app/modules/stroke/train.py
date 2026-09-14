import os
import json

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "..", "data", "raw", "stroke.csv")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURE_ORDER_PATH = os.path.join(BASE_DIR, "feature_order.json")

RANDOM_STATE = 42

CATEGORICAL_COLS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    df = df.drop(columns=["id"])

    df = df[df["gender"] != "Other"].reset_index(drop=True)

    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    print(f"Cleaned dataset: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns")
    print("Class balance:\n", df_encoded["stroke"].value_counts())
    return df_encoded


def main():
    df_encoded = load_and_clean()

    X = df_encoded.drop(columns=["stroke"])
    y = df_encoded["stroke"]
    feature_order = list(X.columns)
    print("Features:", feature_order)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_order)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_order)

    # SMOTE is applied to the training split only — X_test_scaled/y_test
    # stay untouched so evaluation reflects the real, imbalanced world.
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print("Before SMOTE:", y_train.value_counts().to_dict())
    print("After SMOTE:", y_train_resampled.value_counts().to_dict())

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train_resampled, y_train_resampled)
    preds = model.predict(X_test_scaled)

    print("\n--- Random Forest + SMOTE ---")
    print(classification_report(y_test, preds))
    print("Recall (class 1):", recall_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))

    # Swap this for rf_weighted / xgb_smote etc. if your own run picks a
    # different winner (see the notebook's cells 12-13 comparison).
    best_model = rf_smote

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_ORDER_PATH, "w") as f:
        json.dump(feature_order, f)

    print(f"\nSaved: {MODEL_PATH}")
    print(f"Saved: {SCALER_PATH}")
    print(f"Saved: {FEATURE_ORDER_PATH}")


if __name__ == "__main__":
    main()
