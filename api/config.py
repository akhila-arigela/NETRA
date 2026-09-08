"""Environment-driven API configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.inference_pipeline import notebook_artifact_paths, project_root


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class Settings:
    project_root: Path
    stage1_model_path: Path
    stage2_model_path: Path
    max_batch_records: int
    max_csv_bytes: int
    cors_origins: tuple[str, ...]

    @classmethod
    def from_environment(cls) -> "Settings":
        root = Path(os.getenv("SENTIFLOW_PROJECT_ROOT", project_root())).resolve()
        default_stage1, default_stage2 = notebook_artifact_paths(root)
        origins = tuple(
            item.strip()
            for item in os.getenv("SENTIFLOW_CORS_ORIGINS", "http://localhost:8501").split(",")
            if item.strip()
        )
        return cls(
            project_root=root,
            stage1_model_path=Path(os.getenv("SENTIFLOW_STAGE1_MODEL", default_stage1)).resolve(),
            stage2_model_path=Path(os.getenv("SENTIFLOW_STAGE2_MODEL", default_stage2)).resolve(),
            max_batch_records=_positive_int("SENTIFLOW_MAX_BATCH_RECORDS", 5000),
            max_csv_bytes=_positive_int("SENTIFLOW_MAX_CSV_BYTES", 10 * 1024 * 1024),
            cors_origins=origins,
        )
