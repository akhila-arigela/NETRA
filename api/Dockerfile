FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SENTIFLOW_PROJECT_ROOT=/app \
    SENTIFLOW_STAGE1_MODEL=/app/models/hybrid_stage1_router.joblib \
    SENTIFLOW_STAGE2_MODEL=/app/models/complete_stage2_router.joblib

WORKDIR /app

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

COPY api /app/api
COPY src /app/src
COPY Notebooks/hybrid_xgb_iforest_router.py /app/Notebooks/hybrid_xgb_iforest_router.py
COPY Notebooks/stage2_router.py /app/Notebooks/stage2_router.py
COPY Notebooks/mlruns/1/009e649215464979b79cc2bd206e8119/artifacts/router/hybrid_stage1_router.joblib /app/models/hybrid_stage1_router.joblib
COPY Notebooks/mlruns/1/21a3a7a638014352bbc52678c02f79e9/artifacts/router/complete_stage2_router.joblib /app/models/complete_stage2_router.joblib

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
