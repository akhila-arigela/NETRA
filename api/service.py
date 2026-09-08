"""Thread-safe model service and lightweight operational metrics."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.inference_pipeline import HybridInferencePipeline

from .config import Settings


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert pandas/numpy values and NaN to strict JSON-compatible values."""
    return json.loads(frame.to_json(orient="records", double_precision=15))


@dataclass
class ServiceMetrics:
    prediction_requests: int = 0
    predicted_records: int = 0
    failed_requests: int = 0
    total_inference_seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def success(self, records: int, duration: float) -> None:
        with self._lock:
            self.prediction_requests += 1
            self.predicted_records += records
            self.total_inference_seconds += duration

    def failure(self) -> None:
        with self._lock:
            self.failed_requests += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            average = self.total_inference_seconds / max(self.prediction_requests, 1)
            return {
                "prediction_requests": self.prediction_requests,
                "predicted_records": self.predicted_records,
                "failed_requests": self.failed_requests,
                "total_inference_seconds": self.total_inference_seconds,
                "average_request_inference_seconds": average,
            }


class PredictionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline: HybridInferencePipeline | None = None
        self.metrics = ServiceMetrics()
        self._prediction_lock = threading.Lock()

    def load(self) -> None:
        self.pipeline = HybridInferencePipeline.load(
            self.settings.stage1_model_path,
            self.settings.stage2_model_path,
            root=self.settings.project_root,
        )

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def model_info(self) -> dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not loaded")
        info = self.pipeline.model_info()
        info["max_batch_records"] = self.settings.max_batch_records
        info["max_csv_bytes"] = self.settings.max_csv_bytes
        return info

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not loaded")
        if frame.empty:
            raise ValueError("At least one record is required")
        if len(frame) > self.settings.max_batch_records:
            raise ValueError(
                f"Batch contains {len(frame)} records; maximum is {self.settings.max_batch_records}"
            )
        started = time.perf_counter()
        try:
            # The sklearn objects are read-only at prediction time. Serializing
            # calls also prevents nested n_jobs models from oversubscribing CPU.
            with self._prediction_lock:
                result = self.pipeline.predict(frame)
        except Exception:
            self.metrics.failure()
            raise
        self.metrics.success(len(frame), time.perf_counter() - started)
        return result
