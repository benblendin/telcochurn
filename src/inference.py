from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from src.config import METADATA_PATH, MODEL_PATH


@lru_cache(maxsize=1)
def load_model_artifacts() -> tuple[object, dict[str, object]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Артефакт модели не найден: {MODEL_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Файл metadata не найден: {METADATA_PATH}")

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def clear_artifact_cache() -> None:
    load_model_artifacts.cache_clear()


def prepare_feature_frame(records: list[dict[str, object]], metadata: dict[str, object]) -> pd.DataFrame:
    feature_names = metadata["feature_names"]
    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]

    frame = pd.DataFrame(records)
    missing_columns = sorted(set(feature_names).difference(frame.columns))
    unexpected_columns = sorted(set(frame.columns).difference(feature_names))

    if missing_columns:
        raise ValueError(f"Отсутствуют обязательные признаки: {missing_columns}")
    if unexpected_columns:
        raise ValueError(f"Переданы неожиданные признаки: {unexpected_columns}")

    prepared = frame[feature_names].copy()
    for column in numeric_features:
        prepared[column] = pd.to_numeric(prepared[column], errors="raise")
    for column in categorical_features:
        prepared[column] = prepared[column].astype(str).str.strip()

    return prepared


def predict_records(records: list[dict[str, object]]) -> dict[str, object]:
    model, metadata = load_model_artifacts()
    prepared = prepare_feature_frame(records, metadata)
    probabilities = model.predict_proba(prepared)[:, 1]
    threshold = float(metadata.get("threshold", 0.5))

    predictions = []
    for probability in probabilities:
        positive_probability = float(probability)
        predicted_class = int(positive_probability >= threshold)
        predictions.append(
            {
                "churn_probability": round(positive_probability, 4),
                "predicted_class": predicted_class,
                "predicted_label": "Yes" if predicted_class == 1 else "No",
            }
        )

    return {
        "selected_model": metadata.get("selected_model", "unknown"),
        "selection_metric": metadata.get("selection_metric", "roc_auc"),
        "threshold": threshold,
        "predictions": predictions,
    }