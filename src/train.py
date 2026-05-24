from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import ARTIFACTS_DIR, DATA_PATH, METADATA_PATH, METRICS_PATH, MODEL_PATH
from src.data import ID_COLUMN, TARGET_COLUMN, DatasetSplit, build_training_split


def build_baseline_pipeline(dataset: DatasetSplit) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                dataset.numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                dataset.categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced"),
            ),
        ]
    )


def build_boosting_pipeline(dataset: DatasetSplit, random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                dataset.numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-2,
                            ),
                        ),
                    ]
                ),
                dataset.categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=6,
                    max_iter=300,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate_predictions(true_values, probabilities, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(true_values, predictions)), 4),
        "precision": round(float(precision_score(true_values, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(true_values, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(true_values, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(true_values, probabilities)), 4),
    }


def select_best_threshold(true_values, probabilities) -> float:
    precision, recall, thresholds = precision_recall_curve(true_values, probabilities)
    if len(thresholds) == 0:
        return 0.5

    numerator = 2 * precision[:-1] * recall[:-1]
    denominator = np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    f1_scores = numerator / denominator
    best_index = int(np.nanargmax(f1_scores))
    return round(float(thresholds[best_index]), 4)


def summarize_cross_validation(model: Pipeline, features: pd.DataFrame, target: pd.Series, random_state: int) -> dict[str, dict[str, float]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_validate(
        model,
        features,
        target,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        n_jobs=-1,
    )

    summary: dict[str, dict[str, float]] = {}
    for metric_name, metric_values in scores.items():
        if not metric_name.startswith("test_"):
            continue
        normalized_name = metric_name.removeprefix("test_")
        summary[normalized_name] = {
            "mean": round(float(np.mean(metric_values)), 4),
            "std": round(float(np.std(metric_values)), 4),
        }
    return summary


def top_feature_importance(
    pipeline: Pipeline,
    dataset: DatasetSplit,
    random_state: int,
    limit: int = 10,
) -> list[dict[str, float | str]]:
    importance = permutation_importance(
        pipeline,
        dataset.test_features,
        dataset.test_target,
        scoring="roc_auc",
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1,
    )
    paired_importance = zip(dataset.test_features.columns.tolist(), importance.importances_mean, strict=True)
    sorted_importance = sorted(paired_importance, key=lambda item: item[1], reverse=True)
    return [
        {"feature": feature_name, "importance": round(float(score), 4)}
        for feature_name, score in sorted_importance[:limit]
    ]


def train_models(
    *,
    data_path: Path,
    artifacts_dir: Path,
    test_size: float = 0.2,
    validation_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, object]:
    dataset = build_training_split(
        data_path,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )

    model_builders = {
        "baseline_logistic_regression": lambda: build_baseline_pipeline(dataset),
        "hist_gradient_boosting": lambda: build_boosting_pipeline(dataset, random_state),
    }

    validation_metrics: dict[str, dict[str, float]] = {}
    tuned_thresholds: dict[str, float] = {}
    for model_name, build_model in model_builders.items():
        pipeline = build_model()
        pipeline.fit(dataset.train_features, dataset.train_target)
        validation_probabilities = pipeline.predict_proba(dataset.validation_features)[:, 1]
        tuned_threshold = select_best_threshold(dataset.validation_target, validation_probabilities)
        tuned_thresholds[model_name] = tuned_threshold
        validation_metrics[model_name] = evaluate_predictions(
            dataset.validation_target,
            validation_probabilities,
            threshold=tuned_threshold,
        )

    selected_model_name, selected_model_bundle = max(
        validation_metrics.items(),
        key=lambda item: item[1]["roc_auc"],
    )
    selected_threshold = tuned_thresholds[selected_model_name]

    development_features = pd.concat(
        [dataset.train_features, dataset.validation_features],
        ignore_index=True,
    )
    development_target = pd.concat(
        [dataset.train_target, dataset.validation_target],
        ignore_index=True,
    )

    test_metrics: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Pipeline] = {}
    for model_name, build_model in model_builders.items():
        pipeline = build_model()
        pipeline.fit(development_features, development_target)
        test_probabilities = pipeline.predict_proba(dataset.test_features)[:, 1]
        test_metrics[model_name] = evaluate_predictions(
            dataset.test_target,
            test_probabilities,
            threshold=tuned_thresholds[model_name],
        )
        fitted_models[model_name] = pipeline

    selected_pipeline = fitted_models[selected_model_name]
    cross_validation_summary = summarize_cross_validation(
        model_builders[selected_model_name](),
        development_features,
        development_target,
        random_state,
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / MODEL_PATH.name
    metrics_path = artifacts_dir / METRICS_PATH.name
    metadata_path = artifacts_dir / METADATA_PATH.name
    joblib.dump(selected_pipeline, model_path)

    metrics = {
        "selection_metric": "roc_auc",
        "threshold_metric": "f1",
        "selected_model": selected_model_name,
        "selected_threshold": selected_threshold,
        "validation_metrics": validation_metrics,
        "cross_validation": {
            selected_model_name: cross_validation_summary,
        },
        "baseline_logistic_regression": test_metrics["baseline_logistic_regression"],
        "hist_gradient_boosting": test_metrics["hist_gradient_boosting"],
    }
    metadata = {
        "data_path": str(data_path.resolve()),
        "artifacts_dir": str(artifacts_dir.resolve()),
        "id_column": ID_COLUMN,
        "target_column": TARGET_COLUMN,
        "feature_names": dataset.train_features.columns.tolist(),
        "categorical_features": dataset.categorical_features,
        "numeric_features": dataset.numeric_features,
        "training_rows": int(len(development_features)),
        "subtrain_rows": int(len(dataset.train_features)),
        "validation_rows": int(len(dataset.validation_features)),
        "test_rows": int(len(dataset.test_features)),
        "random_state": random_state,
        "test_size": test_size,
        "validation_size": validation_size,
        "threshold": selected_threshold,
        "selection_metric": "roc_auc",
        "threshold_metric": "f1",
        "selected_model": selected_model_name,
        "cross_validation": cross_validation_summary,
        "top_feature_importance": top_feature_importance(
            selected_pipeline,
            dataset,
            random_state,
        ),
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {"metrics": metrics, "metadata": metadata, "model_path": str(model_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение моделей оттока клиентов на датасете Telco")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_models(
        data_path=args.data_path,
        artifacts_dir=args.artifacts_dir,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    print(json.dumps(result["metrics"], indent=2))
    print(f"Модель сохранена в {result['model_path']}")


if __name__ == "__main__":
    main()