"""End-to-end inference for the notebook-compatible two-stage hybrid IDS."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .features import ORIGINAL_FEATURES, engineer_features
from .legacy_models import router_types


NOTEBOOK_STAGE1_RUN_ID = "009e649215464979b79cc2bd206e8119"
NOTEBOOK_STAGE2_RUN_ID = "21a3a7a638014352bbc52678c02f79e9"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def notebook_artifact_paths(root: Path | None = None) -> tuple[Path, Path]:
    root = (root or project_root()).resolve()
    base = root / "Notebooks" / "mlruns" / "1"
    stage1 = base / NOTEBOOK_STAGE1_RUN_ID / "artifacts" / "router" / "hybrid_stage1_router.joblib"
    stage2 = base / NOTEBOOK_STAGE2_RUN_ID / "artifacts" / "router" / "complete_stage2_router.joblib"
    return stage1, stage2


@dataclass
class HybridInferencePipeline:
    """Load fitted artifacts once and execute Stage 1 followed by Stage 2."""

    stage1_router: Any
    stage2_router: Any
    model_source: str = "notebook-compatible"

    @classmethod
    def load(
        cls,
        stage1_path: str | Path | None = None,
        stage2_path: str | Path | None = None,
        *,
        root: str | Path | None = None,
    ) -> "HybridInferencePipeline":
        root_path = Path(root).resolve() if root else project_root()
        default_stage1, default_stage2 = notebook_artifact_paths(root_path)
        stage1_path = Path(stage1_path or default_stage1).resolve()
        stage2_path = Path(stage2_path or default_stage2).resolve()
        missing = [str(p) for p in (stage1_path, stage2_path) if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing hybrid model artifacts: {missing}")
        HybridRouter, CompleteStage2Router = router_types(root_path)
        stage1 = joblib.load(stage1_path)
        stage2 = joblib.load(stage2_path)
        if not isinstance(stage1, HybridRouter):
            raise TypeError(f"Unexpected Stage-1 artifact type: {type(stage1)!r}")
        if not isinstance(stage2, CompleteStage2Router):
            raise TypeError(f"Unexpected Stage-2 artifact type: {type(stage2)!r}")
        return cls(stage1, stage2, model_source=f"{stage1_path}|{stage2_path}")

    def predict(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Predict one or more rows containing the 41 original flow fields."""
        engineered = engineer_features(raw_data, validate=False)
        raw = engineered.loc[:, list(ORIGINAL_FEATURES)]

        stage1 = self.stage1_router.score(engineered)
        stage1.index = engineered.index
        stage1_evidence = pd.DataFrame(
            {
                "stage1_route": stage1["route"].astype(str),
                "xgb_attack_probability": stage1["xgb_attack_probability"].astype(float),
                "normal_novelty_percentile": stage1["normal_novelty_percentile"].astype(float),
            },
            index=engineered.index,
        )
        stage2 = self.stage2_router.predict(raw, stage1_evidence)
        stage2.index = engineered.index

        # Stage 2 already carries the core Stage-1 evidence. Add explanatory
        # Stage-1 fields without duplicating columns.
        extra_stage1 = stage1.drop(
            columns=["route", "xgb_attack_probability", "normal_novelty_percentile"],
            errors="ignore",
        ).add_prefix("stage1_")
        result = extra_stage1.join(stage2)
        result.insert(0, "final_decision", result["stage2_decision"])
        return result

    def predict_record(self, record: dict[str, Any]) -> dict[str, Any]:
        output = self.predict(pd.DataFrame([record]))
        return output.iloc[0].where(output.iloc[0].notna(), None).to_dict()

    def model_info(self) -> dict[str, Any]:
        return {
            "source": self.model_source,
            "stage1_model_version": self.stage1_router.model_version,
            "stage1_threshold_version": self.stage1_router.threshold_version,
            "stage2_model_version": self.stage2_router.model_version,
            "category_novelty_version": self.stage2_router.category_novelty_version,
            "required_input_columns": list(ORIGINAL_FEATURES),
        }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Sentiflow hybrid inference pipeline")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--stage1-model", type=Path)
    parser.add_argument("--stage2-model", type=Path)
    parser.add_argument("--print-model-info", action="store_true")
    args = parser.parse_args()
    pipeline = HybridInferencePipeline.load(args.stage1_model, args.stage2_model)
    result = pipeline.predict(pd.read_csv(args.input_csv))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    if args.print_model_info:
        print(json.dumps(pipeline.model_info(), indent=2))
    print(f"Wrote {len(result)} predictions to {args.output_csv}")


if __name__ == "__main__":
    _main()
