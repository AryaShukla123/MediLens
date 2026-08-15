from typing import Optional

import pandas as pd
import shap


def generate_shap_explanation(
    model,
    processed_features: pd.DataFrame,
    raw_input: dict,
    feature_labels: Optional[dict] = None,
    top_n: int = 6,
) -> dict:

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(processed_features)

    shap_row = _extract_positive_class_row(raw_shap)

    feature_labels = feature_labels or {}
    feature_names = list(processed_features.columns)

    contributions = []
    for name, value in zip(feature_names, shap_row):
        contributions.append(
            {
                "feature": name,
                "label": feature_labels.get(name, name),
                "shap_value": round(float(value), 4),
                "input_value": raw_input.get(name),
                "direction": "increases risk" if value > 0 else "decreases risk",
            }
        )

    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

    return {"top_features": contributions[:top_n]}


def _extract_positive_class_row(raw_shap) -> list:
    
    # Older shap versions on sklearn RandomForest: list of 2 arrays
    # (one per class), each shape (n_samples, n_features).
    if isinstance(raw_shap, list):
        positive_class = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
        return list(positive_class[0])

    arr = raw_shap

    # Newer shap versions: single array shaped
    # (n_samples, n_features, n_classes) for classifiers with proba output.
    if arr.ndim == 3:
        if arr.shape[-1] > 1:
            return list(arr[0, :, 1])
        return list(arr[0, :, 0])

    # XGBoost binary classifier: single array (n_samples, n_features) —
    # already represents the positive-class contribution.
    return list(arr[0])
