# Сервис предсказания оттока клиентов

ML-проект в стиле production на датасете IBM Telco Customer Churn. Репозиторий обучает несколько моделей, выбирает лучшую по ROC-AUC, сохраняет артефакты и поднимает API для инференса на FastAPI.

Подробный технический разбор находится в `docs/model_analysis.md`.

## Ключевые особенности

- Реальная табличная задача классификации с понятным бизнес-контекстом.
- Есть полноценный workflow `train -> evaluate -> serve`, а не только notebook.
- Есть сравнение baseline-модели и boosting-модели.
- Есть API-слой, поэтому проект выглядит как ML-сервис, а не просто эксперимент.

## Текущий результат обучения

На последнем запуске на `telco.csv` была выбрана `baseline_logistic_regression` по validation `roc_auc`, а итоговый threshold был подобран на validation по `f1`.

| Модель | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6096 | 0.7658 | 0.5458 | 0.7005 | 0.6136 | 0.8415 |
| HistGradientBoosting | 0.3543 | 0.7779 | 0.5670 | 0.6898 | 0.6224 | 0.8322 |

Топ признаков по permutation importance для выбранной модели:

1. `tenure`
2. `InternetService`
3. `Contract`
4. `MonthlyCharges`
5. `TotalCharges`

Более сложная модель не всегда оказывается лучшей. В сервис сохраняется именно та модель, которая реально выигрывает по метрике отбора.

Для выбранной logistic regression также получено `roc_auc = 0.8464 +/- 0.0072` в 5-fold cross-validation на development split.

## Методология

1. Датасет делится на train, validation и test.
2. Обучаются две кандидатные модели: logistic regression и histogram gradient boosting.
3. Продакшен-модель выбирается по validation `roc_auc`.
4. Threshold для выбранной модели подбирается на validation по `f1`.
5. Затем лучшая модель переобучается на объединённых train и validation данных.
6. Финальные метрики считаются на отложенном test split.
7. Вместе с моделью сохраняются metadata, threshold и сводка cross-validation.

## Стек

- Python 3.14
- pandas
- scikit-learn
- FastAPI
- Uvicorn
- pytest
- Docker как опциональная упаковка

## Структура проекта

```text
.
├── api/
├── docs/
├── examples/
├── src/
├── tests/
├── telco.csv
├── Dockerfile
├── README.md
└── requirements.txt
```

## Локальный запуск

```bash
python3.14 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Обучение модели

```bash
./venv/bin/python -m src.train
```

Артефакты сохраняются в `artifacts/` и намеренно не попадают в git:

- `churn_model.joblib`
- `metrics.json`
- `metadata.json`

## Локальный прогноз из JSON

```bash
./venv/bin/python -m src.predict --input examples/sample_request.json
```

## Запуск API локально

Перед запуском API необходимо сначала сформировать артефакты модели:

```bash
./venv/bin/python -m src.train
./venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Swagger UI будет доступен по адресу `http://127.0.0.1:8000/docs`.

Проверка health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Пример запроса на prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

## Запуск тестов

```bash
./venv/bin/python -m pytest -q
```

## Docker

Сборка образа:

```bash
docker build -t churn-service .
```

Запуск контейнера:

```bash
docker run --rm -p 8000:8000 churn-service
```

Docker-образ обучает модель при сборке и запускает FastAPI сервис при старте контейнера.