"""Integration tests for the FastAPI inference surface."""

from __future__ import annotations

import io
import unittest

import pandas as pd
from fastapi.testclient import TestClient

from src.features import ORIGINAL_FEATURES

from .main import app


class ApiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data = pd.read_csv("Data/Consolidated_df.csv", nrows=3)
        cls.records = data.loc[:, list(ORIGINAL_FEATURES)].to_dict(orient="records")
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)

    def test_health_and_model_info(self) -> None:
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn("Sentiflow", home.json()["service"])
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["model_loaded"])
        info = self.client.get("/v1/model-info")
        self.assertEqual(info.status_code, 200)
        self.assertEqual(len(info.json()["required_input_columns"]), 41)

    def test_single_prediction(self) -> None:
        response = self.client.post("/v1/predict", json=self.records[0])
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIn("final_decision", payload["prediction"])
        self.assertIn("stage2_decision", payload["prediction"])
        self.assertIn("X-Request-ID", response.headers)

    def test_json_batch_prediction(self) -> None:
        response = self.client.post("/v1/predict/batch", json={"records": self.records})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["record_count"], len(self.records))
        self.assertEqual(len(payload["predictions"]), len(self.records))

    def test_csv_batch_json_and_csv_responses(self) -> None:
        body = pd.DataFrame(self.records).to_csv(index=False).encode()
        json_response = self.client.post(
            "/v1/predict/batch/csv", content=body, headers={"Content-Type": "text/csv"}
        )
        self.assertEqual(json_response.status_code, 200, json_response.text)
        self.assertEqual(json_response.json()["record_count"], len(self.records))
        csv_response = self.client.post(
            "/v1/predict/batch/csv?output=csv",
            content=body,
            headers={"Content-Type": "text/csv"},
        )
        self.assertEqual(csv_response.status_code, 200, csv_response.text)
        parsed = pd.read_csv(io.StringIO(csv_response.text))
        self.assertEqual(len(parsed), len(self.records))
        self.assertIn("final_decision", parsed.columns)

    def test_invalid_rate_is_rejected(self) -> None:
        invalid = dict(self.records[0])
        invalid["serrorrate"] = 2.0
        response = self.client.post("/v1/predict", json=invalid)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "request_validation_error")


if __name__ == "__main__":
    unittest.main()
