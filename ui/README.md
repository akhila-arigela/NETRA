# NETRA Streamlit UI

The Streamlit application is a separate presentation layer that calls the
FastAPI inference backend. It includes the model objective, measured impact,
limitations, single-flow analysis, demonstration profiles, batch CSV scoring,
and downloadable prediction results.

## Run locally

Start FastAPI first:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Then start Streamlit in a second terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui\app.py
```

Open `http://localhost:8501`.

To use another API location:

```powershell
$env:SENTIFLOW_API_URL="https://api.example.com"
.\.venv\Scripts\python.exe -m streamlit run ui\app.py
```

The UI does not load model files or perform preprocessing itself. All
predictions are made through FastAPI, keeping one authoritative inference path.
