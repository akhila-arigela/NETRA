# Hybrid IDS pipelines

This package converts `Notebooks/Hybrid.ipynb` into reusable training and
inference workflows for Stage 1 and Stage 2.

## Modules

- `features.py`: canonical 41-field input schema and all 31 deterministic features.
- `training_pipeline.py`: reproduces the notebook's split, models, calibration,
  novelty reference distributions, routing thresholds, and saved artifacts.
- `inference_pipeline.py`: loads both artifacts once and executes the complete
  Stage-1 → Stage-2 prediction path for a record or batch.
- `validate_notebook_parity.py`: checks engineered features and model outputs
  against the notebook's saved external-test samples.

## Train both stages

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.training_pipeline Data\Consolidated_df.csv artifacts\hybrid
```

The output directory receives:

- `hybrid_stage1_router.joblib`
- `complete_stage2_router.joblib`
- `manifest.json`

## Predict a CSV

The input CSV must contain the 41 original fields. Engineered fields are
computed internally.

```powershell
.\.venv\Scripts\python.exe -m src.inference_pipeline input.csv predictions.csv
```

By default, inference uses the exact Stage-1 and Stage-2 artifacts produced by
the final notebook runs. Use `--stage1-model` and `--stage2-model` for newly
trained artifacts.

## Validate notebook parity

```powershell
.\.venv\Scripts\python.exe -m src.validate_notebook_parity --output artifacts\parity_report.json
```

Validation independently rebuilds all engineered features, recreates the
notebook's held-out split, and compares the new inference path with the logged
notebook output tables.
