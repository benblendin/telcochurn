from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from src.train import train_models


SAMPLE_RECORD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def load_test_client(tmp_path, monkeypatch) -> TestClient:
    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = tmp_path / "artifacts"
    train_models(data_path=project_root / "telco.csv", artifacts_dir=artifacts_dir)

    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("MODEL_PATH", str(artifacts_dir / "churn_model.joblib"))
    monkeypatch.setenv("METADATA_PATH", str(artifacts_dir / "metadata.json"))

    for module_name in ["api.main", "src.inference", "src.config"]:
        sys.modules.pop(module_name, None)

    api_module = importlib.import_module("api.main")
    return TestClient(api_module.app)


def load_test_client_without_artifacts(tmp_path, monkeypatch) -> TestClient:
    artifacts_dir = tmp_path / "missing_artifacts"

    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("MODEL_PATH", str(artifacts_dir / "churn_model.joblib"))
    monkeypatch.setenv("METADATA_PATH", str(artifacts_dir / "metadata.json"))

    for module_name in ["api.main", "src.inference", "src.config"]:
        sys.modules.pop(module_name, None)

    api_module = importlib.import_module("api.main")
    return TestClient(api_module.app)


def test_healthcheck_reports_ready_model(tmp_path, monkeypatch) -> None:
    client = load_test_client(tmp_path, monkeypatch)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_ready"] is True
    assert payload["selected_model"] in {
        "baseline_logistic_regression",
        "hist_gradient_boosting",
    }


def test_predict_returns_probability_and_label(tmp_path, monkeypatch) -> None:
    client = load_test_client(tmp_path, monkeypatch)

    response = client.post("/predict", json={"records": [SAMPLE_RECORD]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_model"] in {
        "baseline_logistic_regression",
        "hist_gradient_boosting",
    }
    assert len(payload["predictions"]) == 1
    prediction = payload["predictions"][0]
    assert 0.0 <= prediction["churn_probability"] <= 1.0
    assert prediction["predicted_class"] in {0, 1}
    assert prediction["predicted_label"] in {"Yes", "No"}


def test_healthcheck_reports_not_ready_without_artifacts(tmp_path, monkeypatch) -> None:
    client = load_test_client_without_artifacts(tmp_path, monkeypatch)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "not_ready",
        "model_ready": False,
        "selected_model": None,
    }


def test_predict_returns_503_without_artifacts(tmp_path, monkeypatch) -> None:
    client = load_test_client_without_artifacts(tmp_path, monkeypatch)

    response = client.post("/predict", json={"records": [SAMPLE_RECORD]})

    assert response.status_code == 503
    assert "Артефакт модели не найден" in response.json()["detail"]


def test_predict_rejects_extra_fields(tmp_path, monkeypatch) -> None:
    client = load_test_client(tmp_path, monkeypatch)

    response = client.post(
        "/predict",
        json={"records": [{**SAMPLE_RECORD, "api_key": "should-not-be-here"}]},
    )

    assert response.status_code == 422