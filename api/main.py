"""FastAPI application exposing the complete Stage-1 + Stage-2 system."""

from __future__ import annotations

import io
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from src.features import ORIGINAL_FEATURES

from .config import Settings
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    NetworkFlow,
    PredictionResponse,
)
from .service import PredictionService, json_records


logger = logging.getLogger("sentiflow.api")
settings = Settings.from_environment()
service = PredictionService(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(service.load)
    app.state.prediction_service = service
    yield


app = FastAPI(
    title="Sentiflow Hybrid Network Intrusion Detection API",
    description="Inference API for the complete Stage-1 and Stage-2 hybrid detector.",
    version="1.0.0",
    lifespan=lifespan,
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Seconds"] = f"{time.perf_counter() - started:.6f}"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = [
        {key: value for key, value in error.items() if key in {"type", "loc", "msg"}}
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "request_id": getattr(request.state, "request_id", None),
            "error": "request_validation_error",
            "details": details,
        },
    )


@app.exception_handler(ValueError)
async def value_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={
            "request_id": getattr(request.state, "request_id", None),
            "error": "invalid_inference_input",
            "details": str(exc),
        },
    )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": app.title,
        "status": "online" if service.ready else "starting",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health():
    info = service.model_info() if service.ready else {}
    return HealthResponse(
        status="ok" if service.ready else "not_ready",
        model_loaded=service.ready,
        stage1_model_version=info.get("stage1_model_version"),
        stage2_model_version=info.get("stage2_model_version"),
    )


@app.get("/v1/model-info", tags=["Operations"])
async def model_info():
    if not service.ready:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return service.model_info()


@app.get("/v1/input-schema", tags=["Operations"])
async def input_schema():
    return {
        "required_columns": list(ORIGINAL_FEATURES),
        "field_count": len(ORIGINAL_FEATURES),
        "json_schema": NetworkFlow.model_json_schema(),
    }


@app.get("/metrics", tags=["Operations"])
async def metrics():
    return service.metrics.snapshot()


@app.post("/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_one(flow: NetworkFlow, request: Request):
    frame = pd.DataFrame([flow.model_dump()])
    result = await run_in_threadpool(service.predict, frame)
    return PredictionResponse(
        request_id=request.state.request_id,
        prediction=json_records(result)[0],
    )


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(payload: BatchPredictionRequest, request: Request):
    if len(payload.records) > settings.max_batch_records:
        raise HTTPException(
            status_code=413,
            detail=f"Maximum batch size is {settings.max_batch_records} records",
        )
    frame = pd.DataFrame([record.model_dump() for record in payload.records])
    result = await run_in_threadpool(service.predict, frame)
    return BatchPredictionResponse(
        request_id=request.state.request_id,
        record_count=len(result),
        predictions=json_records(result),
    )


@app.post("/v1/predict/batch/csv", tags=["Prediction"])
async def predict_csv(
    request: Request,
    output: Literal["json", "csv"] = "json",
):
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in {"text/csv", "application/csv", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Use Content-Type: text/csv")
    body = await request.body()
    if len(body) > settings.max_csv_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"CSV exceeds the {settings.max_csv_bytes}-byte limit",
        )
    try:
        frame = pd.read_csv(io.BytesIO(body))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Unable to parse CSV: {exc}") from exc
    # Apply the same strict schema and domain checks as JSON requests. Pandas
    # NaN represents an empty CSV cell and is converted to an allowed null.
    csv_records = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
    validated_records = [NetworkFlow.model_validate(record).model_dump() for record in csv_records]
    frame = pd.DataFrame(validated_records)
    result = await run_in_threadpool(service.predict, frame)
    if output == "csv":
        rendered = result.to_csv(index=False)
        return Response(
            content=rendered,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=sentiflow_predictions.csv"},
        )
    return {
        "request_id": request.state.request_id,
        "record_count": len(result),
        "predictions": json_records(result),
    }
