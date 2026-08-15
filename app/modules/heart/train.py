"""
Offline training script for the Heart Disease module.

Run this manually (NOT on every app startup) to (re)generate the three
artifacts that app/modules/heart/preprocessing.py loads at import time:

    model.pkl            - trained classifier (RandomForest or XGBoost, whichever wins)
    scaler.pkl            - StandardScaler fitted on the training split
    feature_order.json   - exact column order the model expects

Usage (from the project root, with your venv active):
    python -m app.modules.heart.train

Mirrors the logic already validated in notebooks/heart_eda.ipynb, just
turned into a reusable, non-interactive script.
"""

import os
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score, roc_auc_score
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "..", "data", "raw", "heart.csv")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURE_ORDER_PATH = os.path.join(BASE_DIR, "feature_order.json")

TARGET_COL = "condition"
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def train_and_select_best(X_train, X_test, y_train, y_test):
    """Trains RandomForest + XGBoost, picks the winner by recall on the
    positive class (missing a sick patient is worse than a false alarm),
    falling back to ROC-AUC as a tiebreaker."""

    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    xgb = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)

    rf_recall = recall_score(y_test, rf_preds, pos_label=1)
    xgb_recall = recall_score(y_test, xgb_preds, pos_label=1)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])

    print("\n--- Random Forest ---")
    print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f} | Recall: {rf_recall:.4f} | ROC-AUC: {rf_auc:.4f}")
    print(classification_report(y_test, rf_preds))

    print("--- XGBoost ---")
    print(f"Accuracy: {accuracy_score(y_test, xgb_preds):.4f} | Recall: {xgb_recall:.4f} | ROC-AUC: {xgb_auc:.4f}")
    print(classification_report(y_test, xgb_preds))

    if xgb_recall > rf_recall or (xgb_recall == rf_recall and xgb_auc >= rf_auc):
        print(f"\nSelected model: XGBoost (recall={xgb_recall:.4f}, auc={xgb_auc:.4f})")
        return xgb
    else:
        print(f"\nSelected model: RandomForest (recall={rf_recall:.4f}, auc={rf_auc:.4f})")
        return rf


def main():
    df = load_data()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_order = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_order)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_order)

    best_model = train_and_select_best(X_train_scaled, X_test_scaled, y_train, y_test)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_ORDER_PATH, "w") as f:
        json.dump(feature_order, f)

    print(f"\nSaved: {MODEL_PATH}")
    print(f"Saved: {SCALER_PATH}")
    print(f"Saved: {FEATURE_ORDER_PATH}")


if __name__ == "__main__":
    main()
