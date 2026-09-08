"""Leakage-safe XGBoost + Isolation Forest routing for network intrusion detection.

Expected target convention:
    0 = Normal
    1 = Attack

The module deliberately keeps the supervised and novelty representations separate:
    * XGBoost uses ORIGINAL_FEATURES.
    * Isolation Forest uses COMBINED_FEATURES and a correlation filter.

Use an existing external test set only once at the end. The training set is split
internally into model-fit, normal-reference, and routing-validation partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def _ordered_intersection(columns: Sequence[str], candidates: Iterable[str]) -> list[str]:
    candidate_set = set(candidates)
    return [column for column in columns if column in candidate_set]


def _safe_binary_target(y: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=int).ravel()
    observed = set(np.unique(values).tolist())
    if not observed.issubset({0, 1}) or len(observed) < 2:
        raise ValueError(f"Expected binary target containing 0 and 1; observed {sorted(observed)}")
    return values


def split_training_for_hybrid(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    reference_fraction: float = 0.15,
    validation_fraction: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Split an existing training set into model-fit, reference, and validation sets.

    The external test set must remain untouched.

    For temporal data, replace this random stratified split with chronological slices.
    """
    y_array = _safe_binary_target(y_train)
    if reference_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("reference_fraction and validation_fraction must be positive")
    if reference_fraction + validation_fraction >= 1:
        raise ValueError("reference_fraction + validation_fraction must be less than 1")

    positions = np.arange(len(X_train))
    fit_fraction = 1.0 - reference_fraction - validation_fraction

    fit_pos, temp_pos = train_test_split(
        positions,
        train_size=fit_fraction,
        stratify=y_array,
        random_state=random_state,
    )

    validation_share_of_temp = validation_fraction / (reference_fraction + validation_fraction)
    ref_pos, val_pos = train_test_split(
        temp_pos,
        test_size=validation_share_of_temp,
        stratify=y_array[temp_pos],
        random_state=random_state,
    )

    return (
        X_train.iloc[fit_pos].copy(),
        X_train.iloc[ref_pos].copy(),
        X_train.iloc[val_pos].copy(),
        y_array[fit_pos],
        y_array[ref_pos],
        y_array[val_pos],
    )


# -----------------------------------------------------------------------------
# XGBoost probability calibration
# -----------------------------------------------------------------------------


def make_xgboost_preprocessor(
    original_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    categorical = _ordered_intersection(original_features, categorical_features)
    numeric = [column for column in original_features if column not in set(categorical)]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def make_calibrated_xgboost(
    original_features: Sequence[str],
    categorical_features: Sequence[str],
    xgb_params: dict | None = None,
    calibration_folds: int = 5,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """Build a complete raw-data-to-calibrated-probability estimator.

    Calibration uses out-of-fold predictions inside the model-fit partition.
    Sigmoid calibration is a stable first choice for binary classification.
    """
    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": random_state,
    }
    if xgb_params:
        params.update(xgb_params)

    base_pipeline = Pipeline(
        [
            (
                "preprocess",
                make_xgboost_preprocessor(original_features, categorical_features),
            ),
            ("model", XGBClassifier(**params)),
        ]
    )

    return CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=calibration_folds,
        ensemble=False,
        n_jobs=-1,
    )


# -----------------------------------------------------------------------------
# Correlation-filtered normal-only Isolation Forest
# -----------------------------------------------------------------------------


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop later columns that are highly correlated with an earlier retained column.

    The transformer is unsupervised: y is ignored. It is intended to be fitted only
    on the model-fit partition (preferably trusted normal rows for a novelty model).
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")

        matrix = X.toarray() if sparse.issparse(X) else np.asarray(X)
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Expected a 2-dimensional feature matrix")

        correlation = np.corrcoef(matrix, rowvar=False)
        correlation = np.nan_to_num(np.abs(correlation), nan=0.0, posinf=0.0, neginf=0.0)

        keep = np.ones(matrix.shape[1], dtype=bool)
        for current in range(matrix.shape[1]):
            if not keep[current]:
                continue
            later = np.arange(current + 1, matrix.shape[1])
            if len(later) == 0:
                continue
            too_correlated = correlation[current, later] >= self.threshold
            keep[later[too_correlated]] = False

        self.support_ = keep
        self.n_features_in_ = matrix.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "support_"):
            raise RuntimeError("CorrelationFilter is not fitted")
        return X[:, self.support_] if sparse.issparse(X) else np.asarray(X)[:, self.support_]

    def get_support(self, indices: bool = False):
        if not hasattr(self, "support_"):
            raise RuntimeError("CorrelationFilter is not fitted")
        return np.flatnonzero(self.support_) if indices else self.support_.copy()


class NormalNoveltyPercentile:
    """Map raw anomaly scores to empirical percentiles of trusted normal traffic."""

    def fit(self, trusted_normal_scores: np.ndarray):
        scores = np.asarray(trusted_normal_scores, dtype=float).ravel()
        if len(scores) < 100:
            raise ValueError("Use at least 100 trusted-normal reference scores")
        if not np.isfinite(scores).all():
            raise ValueError("Reference scores contain NaN or infinity")
        self.sorted_reference_scores_ = np.sort(scores)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not hasattr(self, "sorted_reference_scores_"):
            raise RuntimeError("NormalNoveltyPercentile is not fitted")
        values = np.asarray(scores, dtype=float).ravel()
        ranks = np.searchsorted(self.sorted_reference_scores_, values, side="right")
        # Finite-sample smoothed percentile in (0, 1].
        return (ranks + 1.0) / (len(self.sorted_reference_scores_) + 1.0)


def make_iforest_preprocessor(
    combined_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    categorical = _ordered_intersection(combined_features, categorical_features)
    numeric = [column for column in combined_features if column not in set(categorical)]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=False,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def make_normal_iforest(
    combined_features: Sequence[str],
    categorical_features: Sequence[str],
    correlation_threshold: float = 0.95,
    iforest_params: dict | None = None,
    random_state: int = 42,
) -> Pipeline:
    params = {
        "n_estimators": 500,
        "max_samples": "auto",
        "max_features": 1.0,
        "contamination": "auto",
        "n_jobs": -1,
        "random_state": random_state,
    }
    if iforest_params:
        params.update(iforest_params)

    return Pipeline(
        [
            (
                "preprocess",
                make_iforest_preprocessor(combined_features, categorical_features),
            ),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("correlation", CorrelationFilter(threshold=correlation_threshold)),
            ("detector", IsolationForest(**params)),
        ]
    )


def raw_iforest_anomaly_score(fitted_pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return scores where larger means more anomalous."""
    representation = fitted_pipeline[:-1].transform(X)
    # IsolationForest.score_samples is lower for more abnormal observations.
    return -fitted_pipeline.named_steps["detector"].score_samples(representation)


# -----------------------------------------------------------------------------
# Threshold learning
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingThresholds:
    xgb_low: float
    xgb_high: float
    novelty_high: float
    novelty_extreme: float


@dataclass
class ThresholdDiagnostics:
    xgb_high_fpr: float
    xgb_high_recall: float
    xgb_high_precision: float
    xgb_low_attack_escape_rate: float
    xgb_low_region_attack_rate: float
    xgb_low_coverage: float
    novelty_high_fpr: float
    novelty_extreme_fpr: float


def _binary_metrics(y_true: np.ndarray, positive_mask: np.ndarray) -> dict[str, float]:
    y = _safe_binary_target(y_true)
    pred = np.asarray(positive_mask, dtype=bool)
    tn, fp, fn, tp = confusion_matrix(y, pred.astype(int), labels=[0, 1]).ravel()
    return {
        "fpr": fp / max(fp + tn, 1),
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
    }


def choose_xgb_high_threshold(
    y_validation: np.ndarray,
    calibrated_probabilities: np.ndarray,
    maximum_fpr: float = 0.001,
    minimum_precision: float | None = None,
) -> float:
    """Maximize recall while respecting an auto-attack FPR budget."""
    y = _safe_binary_target(y_validation)
    scores = np.asarray(calibrated_probabilities, dtype=float).ravel()
    fpr, tpr, thresholds = roc_curve(y, scores)

    feasible: list[tuple[float, float, float]] = []
    for current_fpr, current_tpr, threshold in zip(fpr, tpr, thresholds):
        if not np.isfinite(threshold) or current_fpr > maximum_fpr:
            continue
        pred = scores >= threshold
        current_precision = precision_score(y, pred, zero_division=0)
        if minimum_precision is not None and current_precision < minimum_precision:
            continue
        feasible.append((current_tpr, -threshold, threshold))

    if not feasible:
        # No validation observation is auto-routed as attack.
        return float(np.nextafter(scores.max(), np.inf))

    # Maximum TPR; ties choose the lower/more inclusive threshold.
    feasible.sort(reverse=True)
    return float(feasible[0][2])


def choose_xgb_low_threshold(
    y_validation: np.ndarray,
    calibrated_probabilities: np.ndarray,
    maximum_attack_escape_rate: float = 0.001,
    maximum_low_region_attack_rate: float = 0.001,
) -> float:
    """Choose the largest auto-normal threshold satisfying two miss constraints.

    attack_escape_rate:
        attacks placed in the low-risk region / all attacks

    low_region_attack_rate:
        attacks placed in the low-risk region / all records in that region
    """
    y = _safe_binary_target(y_validation)
    scores = np.asarray(calibrated_probabilities, dtype=float).ravel()

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_y = y[order]

    unique_scores, counts = np.unique(sorted_scores, return_counts=True)
    last_positions = np.cumsum(counts) - 1
    cumulative_attacks = np.cumsum(sorted_y)[last_positions]
    cumulative_rows = last_positions + 1

    total_attacks = max(int(y.sum()), 1)
    escape_rate = cumulative_attacks / total_attacks
    region_attack_rate = cumulative_attacks / cumulative_rows

    valid = (
        (escape_rate <= maximum_attack_escape_rate)
        & (region_attack_rate <= maximum_low_region_attack_rate)
    )

    if not np.any(valid):
        return float(np.nextafter(scores.min(), -np.inf))

    return float(unique_scores[np.flatnonzero(valid)[-1]])


def choose_novelty_threshold(
    y_validation: np.ndarray,
    novelty_percentiles: np.ndarray,
    maximum_normal_fpr: float,
) -> float:
    """Choose the most inclusive novelty cutoff that respects the FPR budget.

    Using unique observed scores avoids exceeding the budget when many normal
    observations share the same percentile. If no observed cutoff is feasible,
    a value just above the maximum normal score is returned.
    """
    y = _safe_binary_target(y_validation)
    novelty = np.asarray(novelty_percentiles, dtype=float).ravel()
    normal = np.sort(novelty[y == 0])
    if len(normal) == 0:
        raise ValueError("Validation data contains no normal observations")

    unique_scores = np.unique(normal)
    first_positions = np.searchsorted(normal, unique_scores, side="left")
    counts_at_or_above = len(normal) - first_positions
    observed_fpr = counts_at_or_above / len(normal)
    feasible = observed_fpr <= maximum_normal_fpr

    if not np.any(feasible):
        return float(np.nextafter(normal.max(), np.inf))

    # Lowest feasible threshold gives the largest alert region under the budget.
    return float(unique_scores[np.flatnonzero(feasible)[0]])


def learn_routing_thresholds(
    y_validation: np.ndarray,
    xgb_probabilities: np.ndarray,
    novelty_percentiles: np.ndarray,
    xgb_auto_attack_fpr: float = 0.001,
    xgb_auto_normal_attack_escape_rate: float = 0.001,
    xgb_auto_normal_region_attack_rate: float = 0.001,
    novelty_high_fpr: float = 0.01,
    novelty_extreme_fpr: float = 0.001,
    xgb_minimum_precision: float | None = None,
) -> tuple[RoutingThresholds, ThresholdDiagnostics]:
    if novelty_extreme_fpr >= novelty_high_fpr:
        raise ValueError("novelty_extreme_fpr must be smaller than novelty_high_fpr")

    y = _safe_binary_target(y_validation)
    xgb = np.asarray(xgb_probabilities, dtype=float).ravel()
    novelty = np.asarray(novelty_percentiles, dtype=float).ravel()

    xgb_high = choose_xgb_high_threshold(
        y,
        xgb,
        maximum_fpr=xgb_auto_attack_fpr,
        minimum_precision=xgb_minimum_precision,
    )
    xgb_low = choose_xgb_low_threshold(
        y,
        xgb,
        maximum_attack_escape_rate=xgb_auto_normal_attack_escape_rate,
        maximum_low_region_attack_rate=xgb_auto_normal_region_attack_rate,
    )
    novelty_high = choose_novelty_threshold(y, novelty, novelty_high_fpr)
    novelty_extreme = choose_novelty_threshold(y, novelty, novelty_extreme_fpr)

    if xgb_low >= xgb_high:
        raise RuntimeError(
            "Learned xgb_low is not below xgb_high. Tighten the auto-normal constraints "
            "or relax the auto-attack FPR constraint."
        )

    high_metrics = _binary_metrics(y, xgb >= xgb_high)
    low_mask = xgb <= xgb_low
    low_attacks = int(y[low_mask].sum())
    total_attacks = max(int(y.sum()), 1)

    diagnostics = ThresholdDiagnostics(
        xgb_high_fpr=high_metrics["fpr"],
        xgb_high_recall=high_metrics["recall"],
        xgb_high_precision=high_metrics["precision"],
        xgb_low_attack_escape_rate=low_attacks / total_attacks,
        xgb_low_region_attack_rate=low_attacks / max(int(low_mask.sum()), 1),
        xgb_low_coverage=float(low_mask.mean()),
        novelty_high_fpr=float((novelty[y == 0] >= novelty_high).mean()),
        novelty_extreme_fpr=float((novelty[y == 0] >= novelty_extreme).mean()),
    )

    return (
        RoutingThresholds(
            xgb_low=xgb_low,
            xgb_high=xgb_high,
            novelty_high=novelty_high,
            novelty_extreme=novelty_extreme,
        ),
        diagnostics,
    )


# -----------------------------------------------------------------------------
# Routing
# -----------------------------------------------------------------------------


def route_from_scores(
    xgb_probability: np.ndarray,
    novelty_percentile: np.ndarray,
    thresholds: RoutingThresholds,
    model_version: str = "stage1-router-v1",
    threshold_version: str = "routing-thresholds-v1",
) -> pd.DataFrame:
    """Implement asymmetric XGBoost/Isolation-Forest routing.

    Automatic decisions:
      * High XGBoost risk -> Stage 2 attack classification.
      * Low XGBoost risk + low novelty -> Normal.
      * Low XGBoost risk + extreme novelty -> Novel-anomaly candidate.

    All remaining combinations are retained as disagreement/review cases. This is
    intentionally conservative; a later OOF-trained meta-model can replace REVIEW.
    """
    p = np.asarray(xgb_probability, dtype=float).ravel()
    n = np.asarray(novelty_percentile, dtype=float).ravel()
    if len(p) != len(n):
        raise ValueError("xgb_probability and novelty_percentile lengths differ")

    risk_band = np.full(len(p), "Intermediate", dtype=object)
    risk_band[p <= thresholds.xgb_low] = "Low"
    risk_band[p >= thresholds.xgb_high] = "High"

    novelty_band = np.full(len(n), "Low", dtype=object)
    novelty_band[n >= thresholds.novelty_high] = "High"
    novelty_band[n >= thresholds.novelty_extreme] = "ExtremelyHigh"

    route = np.full(len(p), "REVIEW_DISAGREEMENT", dtype=object)
    reason = np.full(len(p), "ambiguous score combination", dtype=object)

    high_xgb = p >= thresholds.xgb_high
    low_xgb = p <= thresholds.xgb_low
    low_novelty = n < thresholds.novelty_high
    extreme_novelty = n >= thresholds.novelty_extreme

    route[high_xgb] = "ROUTE_STAGE_2"
    reason[high_xgb] = "High supervised attack risk"

    normal_agreement = low_xgb & low_novelty
    route[normal_agreement] = "RETURN_NORMAL"
    reason[normal_agreement] = "Low supervised risk and low deviation from normal"

    novel_candidate = low_xgb & extreme_novelty
    route[novel_candidate] = "NOVEL_ANOMALY_CANDIDATE"
    reason[novel_candidate] = "Low supervised risk but extreme deviation from normal"

    review = route == "REVIEW_DISAGREEMENT"
    reason[review] = "Ambiguous or disagreement between supervised risk and novelty"

    return pd.DataFrame(
        {
            "xgb_attack_probability": p,
            "xgb_risk_band": risk_band,
            "normal_novelty_percentile": n,
            "novelty_band": novelty_band,
            "route": route,
            "route_reason": reason,
            "xgb_low_threshold": thresholds.xgb_low,
            "xgb_high_threshold": thresholds.xgb_high,
            "novelty_high_threshold": thresholds.novelty_high,
            "novelty_extreme_threshold": thresholds.novelty_extreme,
            "model_version": model_version,
            "threshold_version": threshold_version,
        }
    )


@dataclass
class HybridRouter:
    original_features: tuple[str, ...]
    combined_features: tuple[str, ...]
    calibrated_xgboost: CalibratedClassifierCV
    normal_iforest: Pipeline
    novelty_calibrator: NormalNoveltyPercentile
    thresholds: RoutingThresholds
    model_version: str = "stage1-router-v1"
    threshold_version: str = "routing-thresholds-v1"

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(
            (set(self.original_features) | set(self.combined_features)) - set(X.columns)
        )
        if missing:
            raise ValueError(f"Incoming data is missing required columns: {missing}")

        xgb_probability = self.calibrated_xgboost.predict_proba(
            X.loc[:, list(self.original_features)]
        )[:, 1]

        raw_novelty = raw_iforest_anomaly_score(
            self.normal_iforest,
            X.loc[:, list(self.combined_features)],
        )
        novelty_percentile = self.novelty_calibrator.transform(raw_novelty)

        return route_from_scores(
            xgb_probability,
            novelty_percentile,
            self.thresholds,
            model_version=self.model_version,
            threshold_version=self.threshold_version,
        )

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "HybridRouter":
        loaded = joblib.load(path)
        if not isinstance(loaded, HybridRouter):
            raise TypeError("Artifact is not a HybridRouter")
        return loaded


# -----------------------------------------------------------------------------
# End-to-end fitting
# -----------------------------------------------------------------------------


def fit_hybrid_router(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    original_features: Sequence[str],
    combined_features: Sequence[str],
    categorical_features: Sequence[str] = ("protocoltype", "service", "flag"),
    xgb_params: dict | None = None,
    iforest_params: dict | None = None,
    correlation_threshold: float = 0.95,
    random_state: int = 42,
    # Threshold budgets. Tune these to operational requirements.
    xgb_auto_attack_fpr: float = 0.001,
    xgb_auto_normal_attack_escape_rate: float = 0.001,
    xgb_auto_normal_region_attack_rate: float = 0.001,
    novelty_high_fpr: float = 0.01,
    novelty_extreme_fpr: float = 0.001,
) -> tuple[HybridRouter, ThresholdDiagnostics, pd.DataFrame]:
    """Fit both components and learn routing thresholds without touching test data."""
    (
        X_fit,
        X_reference,
        X_validation,
        y_fit,
        y_reference,
        y_validation,
    ) = split_training_for_hybrid(
        X_train,
        y_train,
        reference_fraction=0.15,
        validation_fraction=0.15,
        random_state=random_state,
    )

    calibrated_xgb = make_calibrated_xgboost(
        original_features,
        categorical_features,
        xgb_params=xgb_params,
        calibration_folds=5,
        random_state=random_state,
    )
    calibrated_xgb.fit(X_fit.loc[:, list(original_features)], y_fit)

    normal_iforest = make_normal_iforest(
        combined_features,
        categorical_features,
        correlation_threshold=correlation_threshold,
        iforest_params=iforest_params,
        random_state=random_state,
    )
    normal_fit_mask = y_fit == 0
    normal_iforest.fit(X_fit.loc[normal_fit_mask, list(combined_features)])

    normal_reference_mask = y_reference == 0
    if int(normal_reference_mask.sum()) < 100:
        raise ValueError("Reference split has fewer than 100 normal records")

    reference_raw_scores = raw_iforest_anomaly_score(
        normal_iforest,
        X_reference.loc[normal_reference_mask, list(combined_features)],
    )
    novelty_calibrator = NormalNoveltyPercentile().fit(reference_raw_scores)

    validation_xgb = calibrated_xgb.predict_proba(
        X_validation.loc[:, list(original_features)]
    )[:, 1]
    validation_raw_novelty = raw_iforest_anomaly_score(
        normal_iforest,
        X_validation.loc[:, list(combined_features)],
    )
    validation_novelty = novelty_calibrator.transform(validation_raw_novelty)

    thresholds, diagnostics = learn_routing_thresholds(
        y_validation,
        validation_xgb,
        validation_novelty,
        xgb_auto_attack_fpr=xgb_auto_attack_fpr,
        xgb_auto_normal_attack_escape_rate=xgb_auto_normal_attack_escape_rate,
        xgb_auto_normal_region_attack_rate=xgb_auto_normal_region_attack_rate,
        novelty_high_fpr=novelty_high_fpr,
        novelty_extreme_fpr=novelty_extreme_fpr,
    )

    router = HybridRouter(
        original_features=tuple(original_features),
        combined_features=tuple(combined_features),
        calibrated_xgboost=calibrated_xgb,
        normal_iforest=normal_iforest,
        novelty_calibrator=novelty_calibrator,
        thresholds=thresholds,
    )

    validation_routes = route_from_scores(
        validation_xgb,
        validation_novelty,
        thresholds,
    )
    validation_routes["binary_target"] = y_validation

    return router, diagnostics, validation_routes
