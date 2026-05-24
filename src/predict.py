from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.inference import predict_records


def load_records(payload_path: Path) -> list[dict[str, object]]:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(
        "Файл с prediction payload должен быть JSON-объектом, списком объектов "
        "или объектом с ключом 'records'"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск локального churn prediction из JSON payload")
    parser.add_argument("--input", type=Path, required=True, help="Путь до JSON-файла с payload")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    result = predict_records(records)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()