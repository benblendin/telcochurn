from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
REQUIRED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


@dataclass(frozen=True)
class DatasetSplit:
    train_features: pd.DataFrame
    validation_features: pd.DataFrame
    test_features: pd.DataFrame
    train_target: pd.Series
    validation_target: pd.Series
    test_target: pd.Series
    categorical_features: list[str]
    numeric_features: list[str]


def load_telco_dataframe(data_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(data_path)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(f"В датасете отсутствуют обязательные колонки: {missing_columns}")

    cleaned = dataframe.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned["SeniorCitizen"] = pd.to_numeric(cleaned["SeniorCitizen"], errors="raise").astype("int64")

    object_columns = cleaned.select_dtypes(include=["object", "string"]).columns.tolist()
    for column in object_columns:
        cleaned[column] = cleaned[column].fillna("Unknown").astype(str).str.strip()

    return cleaned


def build_training_split(
    data_path: str | Path,
    *,
    test_size: float = 0.2,
    validation_size: float = 0.25,
    random_state: int = 42,
) -> DatasetSplit:
    if not 0 < test_size < 1:
        raise ValueError("Параметр test_size должен быть между 0 и 1")
    if not 0 < validation_size < 1:
        raise ValueError("Параметр validation_size должен быть между 0 и 1")

    dataframe = load_telco_dataframe(data_path)

    target = dataframe[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    if target.isna().any():
        raise ValueError("Целевая колонка должна содержать только значения 'Yes' и 'No'")

    features = dataframe.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    categorical_features = features.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = [column for column in features.columns if column not in categorical_features]

    development_features, test_features, development_target, test_target = train_test_split(
        features,
        target.astype("int64"),
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    train_features, validation_features, train_target, validation_target = train_test_split(
        development_features,
        development_target,
        test_size=validation_size,
        random_state=random_state,
        stratify=development_target,
    )

    return DatasetSplit(
        train_features=train_features.reset_index(drop=True),
        validation_features=validation_features.reset_index(drop=True),
        test_features=test_features.reset_index(drop=True),
        train_target=train_target.reset_index(drop=True),
        validation_target=validation_target.reset_index(drop=True),
        test_target=test_target.reset_index(drop=True),
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )