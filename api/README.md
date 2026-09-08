# Sentiflow FastAPI backend

The backend loads the complete Stage-1 and Stage-2 model artifacts once during
application startup. Requests supply the 41 original network-flow fields; the
existing `src` inference pipeline handles engineered features and both stages.

## Start locally

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

FastAPI service metadata is available at `http://127.0.0.1:8000/` and
interactive OpenAPI documentation at `http://127.0.0.1:8000/docs`.

The separate Streamlit frontend is documented in `ui/README.md` and runs on
port `8501` by default.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Readiness and loaded model versions |
| GET | `/v1/model-info` | Artifact versions and configured limits |
| GET | `/v1/input-schema` | Required fields and JSON Schema |
| GET | `/metrics` | In-process request and inference counters |
| POST | `/v1/predict` | One JSON flow |
| POST | `/v1/predict/batch` | JSON object containing `records` |
| POST | `/v1/predict/batch/csv` | Raw CSV request body |

### Single prediction

Send one object containing all fields shown by `/v1/input-schema`:

```text
POST /v1/predict
Content-Type: application/json
```

### JSON batch

```json
{
  "records": [
    { "duration": 0, "protocoltype": "tcp", "...": "remaining fields" }
  ]
}
```

### CSV batch

The endpoint deliberately accepts a raw CSV body, so `python-multipart` is not
required:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/predict/batch/csv" `
  -H "Content-Type: text/csv" --data-binary "@input.csv"
```

Add `?output=csv` to download prediction results as CSV.

## Configuration

| Environment variable | Default |
|---|---|
| `SENTIFLOW_PROJECT_ROOT` | Repository root |
| `SENTIFLOW_STAGE1_MODEL` | Final notebook Stage-1 artifact |
| `SENTIFLOW_STAGE2_MODEL` | Final notebook Stage-2 artifact |
| `SENTIFLOW_MAX_BATCH_RECORDS` | `5000` |
| `SENTIFLOW_MAX_CSV_BYTES` | `10485760` |
| `SENTIFLOW_CORS_ORIGINS` | `http://localhost:8501` |

Stage-1 and Stage-2 artifacts must come from the same training run family.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest api.test_api -v
```

## Docker

Build from the repository root so the model artifacts are available in the
Docker build context:

```powershell
docker build -f api/Dockerfile -t sentiflow-api .
docker run --rm -p 8000:8000 sentiflow-api
```
