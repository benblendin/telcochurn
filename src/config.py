import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT_DIR / "telco.csv"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", ROOT_DIR / "artifacts"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ARTIFACTS_DIR / "churn_model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", ARTIFACTS_DIR / "metrics.json"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", ARTIFACTS_DIR / "metadata.json"))