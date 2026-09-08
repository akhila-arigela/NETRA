"""Small HTTP client used by the Streamlit application."""

from __future__ import annotations

from typing import Any

import requests


class SentiflowApiError(RuntimeError):
    pass


class SentiflowClient:
    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _raise(self, response: requests.Response) -> None:
        if response.ok:
            return
        try:
            payload = response.json()
            detail = payload.get("detail") or payload.get("details") or payload
        except ValueError:
            detail = response.text or response.reason
        raise SentiflowApiError(f"API {response.status_code}: {detail}")

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=10)
        self._raise(response)
        return response.json()

    def model_info(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/v1/model-info", timeout=10)
        self._raise(response)
        return response.json()

    def predict_one(self, record: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/v1/predict", json=record, timeout=self.timeout
        )
        self._raise(response)
        return response.json()

    def predict_csv(self, content: bytes) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/v1/predict/batch/csv",
            data=content,
            headers={"Content-Type": "text/csv"},
            timeout=self.timeout,
        )
        self._raise(response)
        return response.json()
