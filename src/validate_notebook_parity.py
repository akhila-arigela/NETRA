"""Validate deployable inference against the saved Hybrid.ipynb outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .features import ENGINEERED_FEATURES, ORIGINAL_FEATURES, engineer_features
from .inference_pipeline import (
    NOTEBOOK_STAGE1_RUN_ID,
    NOTEBOOK_STAGE2_RUN_ID,
    HybridInferencePipeline,
    project_root,
)


def _table(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload["data"], columns=payload["columns"])


def _compare_frames(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    columns: list[str],
    *,
    atol: float = 2e-9,
) -> dict[str, object]:
    numeric_failures: dict[str, float] = {}
    text_failures: dict[str, int] = {}
    for column in columns:
        left, right = actual[column].reset_index(drop=True), expected[column].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(right) and not pd.api.types.is_bool_dtype(right):
            delta = np.abs(pd.to_numeric(left) - pd.to_numeric(right))
            maximum = float(delta.max()) if len(delta) else 0.0
            if not np.allclose(left, right, rtol=0.0, atol=atol, equal_nan=True):
                numeric_failures[column] = maximum
        else:
            mismatches = int((left.fillna("<NA>").astype(str) != right.fillna("<NA>").astype(str)).sum())
            if mismatches:
                text_failures[column] = mismatches
    return {
        "passed": not numeric_failures and not text_failures,
        "numeric_failures": numeric_failures,
        "text_failures": text_failures,
    }


def validate(root: str | Path | None = None) -> dict[str, object]:
    root = Path(root).resolve() if root else project_root()
    data = pd.read_csv(root / "Data" / "Consolidated_df.csv")

    # 1. Validate every deterministic engineered feature independently.
    generated = engineer_features(data.loc[:, list(ORIGINAL_FEATURES)])
    feature_comparison = _compare_frames(
        generated,
        data,
        list(ENGINEERED_FEATURES),
        # Consolidated_df.csv stores already-computed floating-point values.
        # 1e-7 covers its final decimal serialization ULP at very large byte rates.
        atol=1e-7,
    )

    # Recreate the exact external-test order used by Hybrid.ipynb.
    y = data["binary_target"].map({"Normal": 0, "Attack": 1}).astype(int)
    positions = np.arange(len(data))
    _, test_positions = train_test_split(
        positions, test_size=0.20, stratify=y, random_state=42
    )
    raw_test = data.iloc[test_positions].loc[:, list(ORIGINAL_FEATURES)]
    pipeline = HybridInferencePipeline.load(root=root)

    stage1_artifacts = root / "Notebooks" / "mlruns" / "1" / NOTEBOOK_STAGE1_RUN_ID / "artifacts"
    stage2_artifacts = root / "Notebooks" / "mlruns" / "1" / NOTEBOOK_STAGE2_RUN_ID / "artifacts"
    expected_stage1 = _table(stage1_artifacts / "examples" / "test_router_output_sample.json")
    expected_stage2 = _table(stage2_artifacts / "examples" / "stage2_output_sample.json")
    sample_size = min(len(expected_stage1), len(expected_stage2))
    actual = pipeline.predict(raw_test.iloc[:sample_size])

    stage1_mapping = {
        "xgb_attack_probability": "xgb_attack_probability",
        "xgb_risk_band": "stage1_xgb_risk_band",
        "normal_novelty_percentile": "normal_novelty_percentile",
        "novelty_band": "stage1_novelty_band",
        "route": "stage1_route",
        "route_reason": "stage1_route_reason",
        "xgb_low_threshold": "stage1_xgb_low_threshold",
        "xgb_high_threshold": "stage1_xgb_high_threshold",
        "novelty_high_threshold": "stage1_novelty_high_threshold",
        "novelty_extreme_threshold": "stage1_novelty_extreme_threshold",
        "model_version": "stage1_model_version",
        "threshold_version": "stage1_threshold_version",
    }
    actual_stage1 = pd.DataFrame(
        {expected: actual[observed].to_numpy() for expected, observed in stage1_mapping.items()}
    )
    stage1_comparison = _compare_frames(
        actual_stage1, expected_stage1.iloc[:sample_size], list(stage1_mapping), atol=2e-9
    )

    stage2_columns = [
        column for column in expected_stage2.columns
        if column in actual.columns
    ]
    stage2_comparison = _compare_frames(
        actual, expected_stage2.iloc[:sample_size], stage2_columns, atol=2e-9
    )
    decisions_equal = bool(
        actual["stage2_decision"].reset_index(drop=True).equals(
            expected_stage2.iloc[:sample_size]["stage2_decision"].reset_index(drop=True)
        )
    )

    report = {
        "passed": bool(
            feature_comparison["passed"]
            and stage1_comparison["passed"]
            and stage2_comparison["passed"]
            and decisions_equal
        ),
        "dataset_rows": len(data),
        "notebook_external_test_rows": len(test_positions),
        "notebook_output_rows_compared": sample_size,
        "feature_engineering": feature_comparison,
        "stage1_output": stage1_comparison,
        "stage2_output": stage2_comparison,
        "final_decisions_exact_match": decisions_equal,
    }
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Check src inference against Hybrid.ipynb artifacts")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate()
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
