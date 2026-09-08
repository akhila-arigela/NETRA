"""Production training and inference utilities for the Sentiflow hybrid IDS."""

from .features import ENGINEERED_FEATURES, ORIGINAL_FEATURES, engineer_features

__all__ = [
    "ENGINEERED_FEATURES",
    "ORIGINAL_FEATURES",
    "HybridInferencePipeline",
    "engineer_features",
]


def __getattr__(name: str):
    # Keep package import lightweight and avoid pre-importing CLI modules.
    if name == "HybridInferencePipeline":
        from .inference_pipeline import HybridInferencePipeline

        return HybridInferencePipeline
    raise AttributeError(name)
