"""Import compatibility for router classes serialized by the notebook."""

from __future__ import annotations

import sys
from pathlib import Path


def install_notebook_module_path(project_root: Path) -> None:
    notebook_dir = str((project_root / "Notebooks").resolve())
    if notebook_dir not in sys.path:
        sys.path.insert(0, notebook_dir)


def router_types(project_root: Path):
    install_notebook_module_path(project_root)
    from hybrid_xgb_iforest_router import HybridRouter
    from stage2_router import CompleteStage2Router

    return HybridRouter, CompleteStage2Router
