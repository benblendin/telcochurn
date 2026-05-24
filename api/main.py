from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.inference import load_model_artifacts, predict_records


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: str = Field(min_length=1)
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str = Field(min_length=1)
    Dependents: str = Field(min_length=1)
    tenure: int = Field(ge=0)
    PhoneService: str = Field(min_length=1)
    MultipleLines: str = Field(min_length=1)
    InternetService: str = Field(min_length=1)
    OnlineSecurity: str = Field(min_length=1)
    OnlineBackup: str = Field(min_length=1)
    DeviceProtection: str = Field(min_length=1)
    TechSupport: str = Field(min_length=1)
    StreamingTV: str = Field(min_length=1)
    StreamingMovies: str = Field(min_length=1)
    Contract: str = Field(min_length=1)
    PaperlessBilling: str = Field(min_length=1)
    PaymentMethod: str = Field(min_length=1)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[PredictionRecord] = Field(min_length=1)


class PredictionResult(BaseModel):
    churn_probability: float
    predicted_class: int
    predicted_label: str


class PredictionResponse(BaseModel):
    selected_model: str
    selection_metric: str
    threshold: float
    predictions: list[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    selected_model: str | None = None


app = FastAPI(
    title="API для предсказания оттока клиентов",
    version="0.1.0",
    summary="ML inference сервис в production-style формате для предсказания оттока клиентов",
)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    try:
        _, metadata = load_model_artifacts()
    except FileNotFoundError:
        return HealthResponse(status="not_ready", model_ready=False, selected_model=None)

    return HealthResponse(
        status="ok",
        model_ready=True,
        selected_model=metadata.get("selected_model"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        payload = predict_records([record.model_dump() for record in request.records])
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return PredictionResponse(**payload)