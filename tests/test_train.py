from __future__ import annotations

import json
from pathlib import Path

from src.train import train_models


def test_train_models_writes_expected_artifacts(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = tmp_path / "artifacts"

    result = train_models(data_path=project_root / "telco.csv", artifacts_dir=artifacts_dir)

    assert Path(result["model_path"]).exists()
    assert (artifacts_dir / "metrics.json").exists()
    assert (artifacts_dir / "metadata.json").exists()

    metrics = json.loads((artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["selected_model"] in {
        "baseline_logistic_regression",
        "hist_gradient_boosting",
    }
    assert metrics["threshold_metric"] == "f1"
    assert 0.0 < metrics["selected_threshold"] < 1.0
    assert "validation_metrics" in metrics
    assert "cross_validation" in metrics
    assert metrics["baseline_logistic_regression"]["roc_auc"] > 0.5