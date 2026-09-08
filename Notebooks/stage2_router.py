from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM


STAGE1_ROUTE_STAGE2 = "ROUTE_STAGE_2"
STAGE1_NOVEL = "NOVEL_ANOMALY_CANDIDATE"
STAGE1_REVIEW = "REVIEW_DISAGREEMENT"
STAGE1_NORMAL = "RETURN_NORMAL"

STAGE2_DATA_ERROR = "DATA_QUALITY_EXCEPTION"
STAGE2_REVIEW = "REVIEW_REQUIRED"
STAGE2_UNKNOWN = "UNKNOWN_ATTACK_CANDIDATE"
STAGE2_NORMAL_AFTER_REVIEW = "RETURN_NORMAL_AFTER_REVIEW"
STAGE2_STAGE1_NORMAL = "NORMAL_STAGE1_FINAL"

SUPPORTED_STAGE1_ROUTES = {
    STAGE1_ROUTE_STAGE2,
    STAGE1_NOVEL,
    STAGE1_REVIEW,
    STAGE1_NORMAL,
}


@dataclass(frozen=True)
class UnknownScoreWeights:
    """Weights used to combine Stage-2 uncertainty evidence.

    Larger values of the resulting score mean "less like a known attack".
    The weights are normalized internally, so they do not need to sum to 1.
    """

    confidence_deficiency: float = 0.35
    margin_deficiency: float = 0.15
    entropy: float = 0.20
    attack_novelty: float = 0.30

    def normalized(self) -> "UnknownScoreWeights":
        values = np.asarray(
            [
                self.confidence_deficiency,
                self.margin_deficiency,
                self.entropy,
                self.attack_novelty,
            ],
            dtype=float,
        )
        if np.any(values < 0):
            raise ValueError("Unknown-score weights must be nonnegative.")
        total = float(values.sum())
        if total <= 0:
            raise ValueError("At least one unknown-score weight must be positive.")
        values = values / total
        return UnknownScoreWeights(*map(float, values))


@dataclass(frozen=True)
class Stage2PolicyConfig:
    """Policy controls for route-specific Stage-2 decisions.

    These are development starting points, not universal constants. They must be
    tuned using a validation partition and open-set holdout experiments.
    """

    # Maximum fraction of known validation attacks that may be rejected by the
    # route-specific unknown-score threshold.
    route_stage2_known_rejection_rate: float = 0.02
    novel_route_known_rejection_rate: float = 0.10
    review_route_known_rejection_rate: float = 0.05

    # Minimum sample count before a route-specific threshold is trusted.
    minimum_route_calibration_samples: int = 40

    # Strong unknown evidence gates.
    unknown_min_normal_novelty: float = 0.99
    unknown_min_attack_novelty: float = 0.99
    unknown_max_category_confidence: float = 0.60
    unknown_max_category_margin: float = 0.20
    unknown_min_category_entropy: float = 0.70

    # Stricter known-attack recovery gates for records Stage 1 considered novel.
    novel_route_min_confidence: float = 0.80
    novel_route_min_margin: float = 0.40
    novel_route_max_entropy: float = 0.55
    novel_route_max_attack_novelty: float = 0.90

    # Moderate known-category gates for ambiguous Stage-1 records.
    review_route_min_confidence: float = 0.70
    review_route_min_margin: float = 0.25
    review_route_max_entropy: float = 0.70
    review_route_max_attack_novelty: float = 0.95

    # Optional disagreement adjudicator thresholds.
    enable_review_normal_clearance: bool = False
    review_adjudicator_low: float = 0.05
    review_adjudicator_high: float = 0.80
    review_normal_max_normal_novelty: float = 0.95

    unknown_score_weights: UnknownScoreWeights = field(
        default_factory=UnknownScoreWeights
    )

    def validate(self) -> None:
        probability_fields = {
            name: value
            for name, value in asdict(self).items()
            if isinstance(value, (float, int))
            and name
            not in {
                "minimum_route_calibration_samples",
            }
        }
        for name, value in probability_fields.items():
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be between 0 and 1; got {value}.")
        if self.minimum_route_calibration_samples < 1:
            raise ValueError("minimum_route_calibration_samples must be positive.")
        self.unknown_score_weights.normalized()


@dataclass(frozen=True)
class DataQualityConfig:
    """Optional row-level quality checks applied before Stage-2 scoring."""

    rate_columns: tuple[str, ...] = ()
    nonnegative_columns: tuple[str, ...] = ()
    binary_columns: tuple[str, ...] = ()
    maximum_row_missing_fraction: float = 0.25

    def validate(self) -> None:
        if not 0 <= self.maximum_row_missing_fraction <= 1:
            raise ValueError("maximum_row_missing_fraction must be between 0 and 1.")


@dataclass(frozen=True)
class RouteThresholds:
    """Thresholds learned from known validation attacks for one Stage-1 route."""

    route: str
    unknown_score_threshold: float
    calibration_sample_count: int
    source: str
    allowed_known_rejection_rate: float
    diagnostic_min_confidence: float
    diagnostic_min_margin: float
    diagnostic_max_entropy: float
    diagnostic_max_attack_novelty: float


@dataclass
class Stage2FitDiagnostics:
    known_categories: list[str]
    category_training_rows: int
    attack_novelty_training_rows: int
    attack_reference_rows: int
    route_thresholds: dict[str, dict[str, Any]]
    validation_route_counts: dict[str, int]
    validation_known_attack_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop later columns whose absolute Pearson correlation exceeds threshold.

    This transformer expects a dense numeric matrix. A VarianceThreshold should
    generally run before it, so constant columns have already been removed.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X: Any, y: Any = None) -> "CorrelationFilter":
        if not 0 < float(self.threshold) <= 1:
            raise ValueError("threshold must be in (0, 1].")

        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("CorrelationFilter expects a 2D matrix.")

        n_features = matrix.shape[1]
        if n_features == 0:
            raise ValueError("CorrelationFilter received zero features.")

        if n_features == 1:
            self.support_ = np.ones(1, dtype=bool)
            self.n_features_in_ = 1
            return self

        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = np.corrcoef(matrix, rowvar=False)
        correlation = np.nan_to_num(
            np.abs(correlation), nan=0.0, posinf=0.0, neginf=0.0
        )

        support = np.ones(n_features, dtype=bool)
        for current in range(1, n_features):
            if not support[current]:
                continue
            prior_kept = np.flatnonzero(support[:current])
            if prior_kept.size and np.any(
                correlation[current, prior_kept] >= float(self.threshold)
            ):
                support[current] = False

        self.support_ = support
        self.n_features_in_ = n_features
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not hasattr(self, "support_"):
            raise RuntimeError("CorrelationFilter has not been fitted.")
        matrix = np.asarray(X, dtype=float)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                "Input feature count differs from the matrix used during fit."
            )
        return matrix[:, self.support_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        if not hasattr(self, "support_"):
            raise RuntimeError("CorrelationFilter has not been fitted.")
        if indices:
            return np.flatnonzero(self.support_)
        return self.support_.copy()


class Stage2OpenSetRouter:
    """Stage-2 known-category classifier plus known-attack novelty router.

    Training has two phases:

    1. ``fit_components`` trains the calibrated attack-category classifier and
       the known-attack novelty detector.
    2. ``fit_policy`` uses Stage-1 validation outputs and known validation
       attacks to learn route-specific open-set rejection thresholds.

    Prediction expects raw feature rows and a Stage-1 evidence DataFrame with:

    - ``stage1_route``
    - ``xgb_attack_probability``
    - ``normal_novelty_percentile``
    """

    required_stage1_columns = (
        "stage1_route",
        "xgb_attack_probability",
        "normal_novelty_percentile",
    )

    def __init__(
        self,
        category_feature_columns: Sequence[str],
        attack_novelty_feature_columns: Sequence[str],
        categorical_columns: Sequence[str] = (
            "protocoltype",
            "service",
            "flag",
        ),
        category_model: BaseEstimator | None = None,
        attack_novelty_model: BaseEstimator | None = None,
        category_calibration_method: str = "sigmoid",
        category_calibration_folds: int = 3,
        category_onehot_min_frequency: int | float | None = 5,
        attack_onehot_min_frequency: int | float | None = 5,
        attack_correlation_threshold: float = 0.95,
        attack_apply_feature_filters: bool = True,
        attack_numeric_scaler: str = "robust",
        attack_novelty_fit_limit: int | None = None,
        attack_reference_fraction: float = 0.20,
        policy_config: Stage2PolicyConfig | None = None,
        data_quality_config: DataQualityConfig | None = None,
        random_state: int = 42,
    ):
        self.category_feature_columns = tuple(category_feature_columns)
        self.attack_novelty_feature_columns = tuple(
            attack_novelty_feature_columns
        )
        self.categorical_columns = tuple(categorical_columns)
        self.category_model = category_model
        self.attack_novelty_model = attack_novelty_model
        self.category_calibration_method = category_calibration_method
        self.category_calibration_folds = category_calibration_folds
        self.category_onehot_min_frequency = category_onehot_min_frequency
        self.attack_onehot_min_frequency = attack_onehot_min_frequency
        self.attack_correlation_threshold = attack_correlation_threshold
        self.attack_apply_feature_filters = attack_apply_feature_filters
        self.attack_numeric_scaler = attack_numeric_scaler
        self.attack_novelty_fit_limit = attack_novelty_fit_limit
        self.attack_reference_fraction = attack_reference_fraction
        self.policy_config = policy_config or Stage2PolicyConfig()
        self.data_quality_config = data_quality_config or DataQualityConfig()
        self.random_state = random_state

        self.policy_config.validate()
        self.data_quality_config.validate()
        self._validate_feature_lists()

    def _validate_feature_lists(self) -> None:
        if not self.category_feature_columns:
            raise ValueError("category_feature_columns cannot be empty.")
        if not self.attack_novelty_feature_columns:
            raise ValueError("attack_novelty_feature_columns cannot be empty.")
        if len(set(self.category_feature_columns)) != len(
            self.category_feature_columns
        ):
            raise ValueError("Duplicate names in category_feature_columns.")
        if len(set(self.attack_novelty_feature_columns)) != len(
            self.attack_novelty_feature_columns
        ):
            raise ValueError("Duplicate names in attack_novelty_feature_columns.")

    def _make_preprocessor(
        self,
        feature_columns: Sequence[str],
        *,
        scale_numeric: bool,
        dense_output: bool,
        min_frequency: int | float | None,
        numeric_scaler: str = "robust",
    ) -> ColumnTransformer:
        feature_columns = list(feature_columns)
        categorical = [
            column
            for column in self.categorical_columns
            if column in feature_columns
        ]
        numeric = [
            column for column in feature_columns if column not in categorical
        ]

        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if scale_numeric:
            if numeric_scaler == "robust":
                numeric_steps.append(("scaler", RobustScaler()))
            elif numeric_scaler == "standard":
                numeric_steps.append(("scaler", StandardScaler()))
            else:
                raise ValueError("numeric_scaler must be 'robust' or 'standard'.")

        numeric_pipeline = Pipeline(numeric_steps)
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=min_frequency,
                        sparse_output=not dense_output,
                        dtype=np.float32,
                    ),
                ),
            ]
        )

        transformers: list[tuple[str, Any, list[str]]] = []
        if numeric:
            transformers.append(("numeric", numeric_pipeline, numeric))
        if categorical:
            transformers.append(
                ("categorical", categorical_pipeline, categorical)
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.0 if dense_output else 1.0,
            verbose_feature_names_out=True,
        )

    def _make_category_estimator(self) -> CalibratedClassifierCV:
        base_model = self.category_model
        if base_model is None:
            base_model = RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=self.random_state,
            )
        else:
            base_model = clone(base_model)

        category_pipeline = Pipeline(
            [
                (
                    "preprocess",
                    self._make_preprocessor(
                        self.category_feature_columns,
                        scale_numeric=False,
                        dense_output=False,
                        min_frequency=self.category_onehot_min_frequency,
                        numeric_scaler="robust",
                    ),
                ),
                ("constant_filter", VarianceThreshold(threshold=0.0)),
                ("model", base_model),
            ]
        )

        return CalibratedClassifierCV(
            estimator=category_pipeline,
            method=self.category_calibration_method,
            cv=self.category_calibration_folds,
            ensemble=False,
            n_jobs=-1,
        )

    def _make_attack_novelty_pipeline(self) -> Pipeline:
        novelty_model = self.attack_novelty_model
        if novelty_model is None:
            novelty_model = IsolationForest(
                n_estimators=500,
                max_samples="auto",
                max_features=1.0,
                contamination="auto",
                n_jobs=-1,
                random_state=self.random_state,
            )
        else:
            novelty_model = clone(novelty_model)

        steps: list[tuple[str, Any]] = [
                (
                    "preprocess",
                    self._make_preprocessor(
                        self.attack_novelty_feature_columns,
                        scale_numeric=True,
                        dense_output=True,
                        min_frequency=self.attack_onehot_min_frequency,
                        numeric_scaler=self.attack_numeric_scaler,
                    ),
                ),
        ]
        if self.attack_apply_feature_filters:
            steps.extend([
                ("constant_filter", VarianceThreshold(threshold=0.0)),
                (
                    "correlation_filter",
                    CorrelationFilter(
                        threshold=self.attack_correlation_threshold
                    ),
                ),
                ("embedding_scaler", StandardScaler()),
            ])
        steps.append(("detector", novelty_model))
        return Pipeline(steps)

    @staticmethod
    def _normalize_binary_target(y: pd.Series | Sequence[Any]) -> pd.Series:
        series = pd.Series(y).copy()
        if pd.api.types.is_numeric_dtype(series):
            result = pd.to_numeric(series, errors="raise").astype(int)
        else:
            normalized = (
                series.astype("string").str.strip().str.lower()
            )
            mapping = {
                "normal": 0,
                "benign": 0,
                "0": 0,
                "false": 0,
                "attack": 1,
                "anomaly": 1,
                "malicious": 1,
                "1": 1,
                "true": 1,
            }
            result = normalized.map(mapping)
            unknown = sorted(normalized[result.isna()].dropna().unique())
            if unknown:
                raise ValueError(
                    f"Unrecognized binary target values: {unknown[:10]}"
                )
            result = result.astype(int)
        if not set(result.unique()).issubset({0, 1}):
            raise ValueError("Binary target must contain only 0 and 1.")
        return result

    @staticmethod
    def _normalize_category_target(
        y: pd.Series | Sequence[Any],
    ) -> pd.Series:
        series = pd.Series(y).astype("string").str.strip()
        if series.isna().any() or (series == "").any():
            raise ValueError("Attack category target contains missing values.")
        return series

    def _ensure_features(
        self, X: pd.DataFrame, columns: Sequence[str], label: str
    ) -> None:
        missing = sorted(set(columns) - set(X.columns))
        if missing:
            raise ValueError(f"{label} is missing feature columns: {missing}")

    def fit_components(
        self,
        X_train: pd.DataFrame,
        y_binary_train: pd.Series | Sequence[Any],
        y_category_train: pd.Series | Sequence[Any],
        *,
        X_attack_reference: pd.DataFrame | None = None,
    ) -> "Stage2OpenSetRouter":
        """Fit the category classifier and the known-attack novelty model.

        ``X_attack_reference`` should contain trusted known attacks not used to
        fit the attack novelty detector. If omitted, attack training rows are
        split internally into novelty-fit and novelty-reference subsets.
        """

        X_train = pd.DataFrame(X_train).copy()
        self._ensure_features(
            X_train, self.category_feature_columns, "X_train"
        )
        self._ensure_features(
            X_train, self.attack_novelty_feature_columns, "X_train"
        )

        y_binary = self._normalize_binary_target(y_binary_train)
        y_category = self._normalize_category_target(y_category_train)
        y_binary.index = X_train.index
        y_category.index = X_train.index

        attack_mask = y_binary.eq(1)
        if int(attack_mask.sum()) < 10:
            raise ValueError("Too few attack rows to fit Stage 2.")

        X_attack = X_train.loc[attack_mask].copy()
        y_attack_category = y_category.loc[attack_mask].copy()

        category_counts = y_attack_category.value_counts()
        if len(category_counts) < 2:
            raise ValueError(
                "Stage-2 category classification requires at least two categories."
            )
        if int(category_counts.min()) < self.category_calibration_folds:
            raise ValueError(
                "Every attack category needs at least category_calibration_folds "
                "records for cross-validated calibration."
            )

        self.category_classifier_ = self._make_category_estimator()
        self.category_classifier_.fit(
            X_attack.loc[:, list(self.category_feature_columns)],
            y_attack_category,
        )
        self.known_categories_ = [str(value) for value in self.category_classifier_.classes_]

        if X_attack_reference is None:
            novelty_fit, novelty_reference = train_test_split(
                X_attack,
                test_size=self.attack_reference_fraction,
                stratify=y_attack_category,
                random_state=self.random_state,
            )
        else:
            novelty_fit = X_attack
            novelty_reference = pd.DataFrame(X_attack_reference).copy()
            self._ensure_features(
                novelty_reference,
                self.attack_novelty_feature_columns,
                "X_attack_reference",
            )

        if len(novelty_reference) < 20:
            raise ValueError(
                "The known-attack reference set is too small; use at least 20 rows."
            )

        if (
            self.attack_novelty_fit_limit is not None
            and len(novelty_fit) > self.attack_novelty_fit_limit
        ):
            novelty_labels = y_attack_category.loc[novelty_fit.index]
            observed = novelty_labels.value_counts().index.tolist()
            per_category = max(1, self.attack_novelty_fit_limit // len(observed))
            selected_indices: list[Any] = []
            rng = np.random.default_rng(self.random_state)
            for category in observed:
                candidates = novelty_labels.index[novelty_labels.eq(category)]
                selected_indices.extend(
                    rng.choice(
                        candidates.to_numpy(),
                        size=min(per_category, len(candidates)),
                        replace=False,
                    ).tolist()
                )
            remaining = self.attack_novelty_fit_limit - len(selected_indices)
            if remaining > 0:
                unused = novelty_fit.index.difference(pd.Index(selected_indices))
                if len(unused):
                    selected_indices.extend(
                        rng.choice(
                            unused.to_numpy(),
                            size=min(remaining, len(unused)),
                            replace=False,
                        ).tolist()
                    )
            novelty_fit = novelty_fit.loc[selected_indices].copy()

        self.attack_novelty_pipeline_ = self._make_attack_novelty_pipeline()
        self.attack_novelty_pipeline_.fit(
            novelty_fit.loc[:, list(self.attack_novelty_feature_columns)]
        )
        reference_scores = self._raw_attack_anomaly_score(novelty_reference)
        self.attack_reference_scores_ = np.sort(reference_scores.astype(float))

        self.component_fit_metadata_ = {
            "category_training_rows": int(len(X_attack)),
            "attack_novelty_training_rows": int(len(novelty_fit)),
            "attack_reference_rows": int(len(novelty_reference)),
            "known_categories": self.known_categories_,
        }
        return self

    def _require_components(self) -> None:
        required = (
            "category_classifier_",
            "attack_novelty_pipeline_",
            "attack_reference_scores_",
            "known_categories_",
        )
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(
                "Stage-2 components are not fitted. Missing: " + ", ".join(missing)
            )

    def _raw_attack_anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        self._ensure_features(
            X,
            self.attack_novelty_feature_columns,
            "Attack novelty input",
        )
        X_view = X.loc[:, list(self.attack_novelty_feature_columns)]
        pipeline = self.attack_novelty_pipeline_
        if hasattr(pipeline, "decision_function"):
            # IsolationForest decision_function: larger = more normal.
            return -np.asarray(pipeline.decision_function(X_view)).ravel()
        if hasattr(pipeline, "score_samples"):
            return -np.asarray(pipeline.score_samples(X_view)).ravel()
        raise TypeError(
            "Attack novelty pipeline must expose decision_function or score_samples."
        )

    def _attack_novelty_percentile(self, X: pd.DataFrame) -> np.ndarray:
        scores = self._raw_attack_anomaly_score(X)
        reference = self.attack_reference_scores_
        ranks = np.searchsorted(reference, scores, side="right")
        return (ranks + 1.0) / (len(reference) + 1.0)

    @staticmethod
    def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.clip(
            np.asarray(probabilities, dtype=float), 1e-12, 1.0
        )
        n_classes = probabilities.shape[1]
        if n_classes <= 1:
            return np.zeros(probabilities.shape[0], dtype=float)
        entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
        return entropy / math.log(n_classes)

    def _category_evidence(self, X: pd.DataFrame) -> pd.DataFrame:
        self._ensure_features(
            X, self.category_feature_columns, "Category-model input"
        )
        X_view = X.loc[:, list(self.category_feature_columns)]
        probabilities = np.asarray(
            self.category_classifier_.predict_proba(X_view), dtype=float
        )
        classes = np.asarray(self.category_classifier_.classes_, dtype=object)
        order = np.argsort(probabilities, axis=1)
        top_index = order[:, -1]
        second_index = order[:, -2]
        row_index = np.arange(len(X))

        confidence = probabilities[row_index, top_index]
        second_probability = probabilities[row_index, second_index]
        margin = confidence - second_probability
        entropy = self._normalized_entropy(probabilities)
        predicted = classes[top_index]

        evidence = pd.DataFrame(
            {
                "predicted_known_category": predicted.astype(str),
                "category_confidence": confidence,
                "category_second_probability": second_probability,
                "category_margin": margin,
                "category_entropy": entropy,
            },
            index=X.index,
        )
        for class_position, class_name in enumerate(classes):
            safe_name = str(class_name).replace(" ", "_")
            evidence[f"category_probability_{safe_name}"] = probabilities[
                :, class_position
            ]
        return evidence

    def _unknown_score(self, evidence: pd.DataFrame) -> np.ndarray:
        weights = self.policy_config.unknown_score_weights.normalized()
        return (
            weights.confidence_deficiency
            * (1.0 - evidence["category_confidence"].to_numpy())
            + weights.margin_deficiency
            * (1.0 - evidence["category_margin"].to_numpy())
            + weights.entropy * evidence["category_entropy"].to_numpy()
            + weights.attack_novelty
            * evidence["known_attack_novelty_percentile"].to_numpy()
        )

    def _validate_stage1_evidence(
        self, stage1_evidence: pd.DataFrame, expected_index: pd.Index
    ) -> pd.DataFrame:
        frame = pd.DataFrame(stage1_evidence).copy()
        missing = sorted(set(self.required_stage1_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Stage-1 evidence is missing columns: {missing}")
        if len(frame) != len(expected_index):
            raise ValueError("Stage-1 evidence and Stage-2 input row counts differ.")
        if not frame.index.equals(expected_index):
            frame.index = expected_index

        invalid_routes = sorted(
            set(frame["stage1_route"].astype(str)) - SUPPORTED_STAGE1_ROUTES
        )
        if invalid_routes:
            raise ValueError(f"Unsupported Stage-1 routes: {invalid_routes}")

        for column in (
            "xgb_attack_probability",
            "normal_novelty_percentile",
        ):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if frame[column].isna().any():
                raise ValueError(f"Stage-1 evidence column {column} contains NaN.")
            if not frame[column].between(0, 1, inclusive="both").all():
                raise ValueError(f"Stage-1 evidence column {column} must be in [0, 1].")

        frame["stage1_route"] = frame["stage1_route"].astype(str)
        return frame

    def _row_quality_report(self, X: pd.DataFrame) -> pd.DataFrame:
        config = self.data_quality_config
        required = sorted(
            set(self.category_feature_columns)
            | set(self.attack_novelty_feature_columns)
        )
        self._ensure_features(X, required, "Stage-2 input")

        result = pd.DataFrame(index=X.index)
        result["data_quality_valid"] = True
        reasons: list[list[str]] = [[] for _ in range(len(X))]

        missing_fraction = X.loc[:, required].isna().mean(axis=1)
        invalid_missing = missing_fraction > config.maximum_row_missing_fraction
        for position in np.flatnonzero(invalid_missing.to_numpy()):
            reasons[position].append("excessive_missing_values")

        for column in config.rate_columns:
            if column not in X.columns:
                continue
            values = pd.to_numeric(X[column], errors="coerce")
            invalid = values.notna() & ~values.between(0, 1, inclusive="both")
            for position in np.flatnonzero(invalid.to_numpy()):
                reasons[position].append(f"invalid_rate:{column}")

        for column in config.nonnegative_columns:
            if column not in X.columns:
                continue
            values = pd.to_numeric(X[column], errors="coerce")
            invalid = values.notna() & values.lt(0)
            for position in np.flatnonzero(invalid.to_numpy()):
                reasons[position].append(f"negative_value:{column}")

        for column in config.binary_columns:
            if column not in X.columns:
                continue
            values = pd.to_numeric(X[column], errors="coerce")
            invalid = values.notna() & ~values.isin([0, 1])
            for position in np.flatnonzero(invalid.to_numpy()):
                reasons[position].append(f"invalid_binary:{column}")

        numeric = X.loc[:, required].select_dtypes(include=[np.number])
        if not numeric.empty:
            invalid_nonfinite = ~np.isfinite(numeric.to_numpy(dtype=float))
            row_invalid = invalid_nonfinite.any(axis=1)
            for position in np.flatnonzero(row_invalid):
                reasons[position].append("nonfinite_numeric_value")

        result["data_quality_reason"] = [
            ";".join(sorted(set(row_reasons))) for row_reasons in reasons
        ]
        result["data_quality_valid"] = result["data_quality_reason"].eq("")
        result["row_missing_fraction"] = missing_fraction
        return result

    def build_evidence(
        self,
        X: pd.DataFrame,
        stage1_evidence: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create Stage-2 evidence before applying route-specific decisions."""

        self._require_components()
        X = pd.DataFrame(X).copy()
        stage1 = self._validate_stage1_evidence(stage1_evidence, X.index)
        quality = self._row_quality_report(X)
        category = self._category_evidence(X)
        attack_novelty = self._attack_novelty_percentile(X)

        evidence = stage1.join(category).join(quality)
        evidence["known_attack_novelty_percentile"] = attack_novelty
        evidence["unknown_score"] = self._unknown_score(evidence)
        return evidence

    @staticmethod
    def _safe_quantile(values: pd.Series, quantile: float, fallback: float) -> float:
        clean = pd.to_numeric(values, errors="coerce").dropna()
        if clean.empty:
            return float(fallback)
        return float(clean.quantile(quantile))

    def _route_rejection_rate(self, route: str) -> float:
        if route == STAGE1_ROUTE_STAGE2:
            return self.policy_config.route_stage2_known_rejection_rate
        if route == STAGE1_NOVEL:
            return self.policy_config.novel_route_known_rejection_rate
        if route == STAGE1_REVIEW:
            return self.policy_config.review_route_known_rejection_rate
        raise ValueError(f"No Stage-2 rejection rate defined for route {route}.")

    def _learn_route_threshold(
        self,
        route: str,
        route_known: pd.DataFrame,
        global_known: pd.DataFrame,
    ) -> RouteThresholds:
        required = self.policy_config.minimum_route_calibration_samples
        if len(route_known) >= required:
            source_frame = route_known
            source = "route_specific"
        else:
            source_frame = global_known
            source = "global_known_fallback"

        rejection_rate = self._route_rejection_rate(route)
        unknown_threshold = self._safe_quantile(
            source_frame["unknown_score"],
            1.0 - rejection_rate,
            fallback=1.0,
        )

        return RouteThresholds(
            route=route,
            unknown_score_threshold=unknown_threshold,
            calibration_sample_count=int(len(source_frame)),
            source=source,
            allowed_known_rejection_rate=float(rejection_rate),
            diagnostic_min_confidence=self._safe_quantile(
                source_frame["category_confidence"], rejection_rate, 0.0
            ),
            diagnostic_min_margin=self._safe_quantile(
                source_frame["category_margin"], rejection_rate, 0.0
            ),
            diagnostic_max_entropy=self._safe_quantile(
                source_frame["category_entropy"], 1.0 - rejection_rate, 1.0
            ),
            diagnostic_max_attack_novelty=self._safe_quantile(
                source_frame["known_attack_novelty_percentile"],
                1.0 - rejection_rate,
                1.0,
            ),
        )

    def fit_policy(
        self,
        X_validation: pd.DataFrame,
        y_binary_validation: pd.Series | Sequence[Any],
        y_category_validation: pd.Series | Sequence[Any],
        stage1_validation: pd.DataFrame,
    ) -> Stage2FitDiagnostics:
        """Learn route-specific open-set thresholds from known validation attacks."""

        evidence = self.build_evidence(X_validation, stage1_validation)
        y_binary = self._normalize_binary_target(y_binary_validation)
        y_category = self._normalize_category_target(y_category_validation)
        y_binary.index = evidence.index
        y_category.index = evidence.index

        known_attack_mask = y_binary.eq(1) & y_category.isin(self.known_categories_)
        global_known = evidence.loc[known_attack_mask].copy()
        if len(global_known) < self.policy_config.minimum_route_calibration_samples:
            raise ValueError(
                "Too few known validation attacks to calibrate Stage-2 policies."
            )

        thresholds: dict[str, RouteThresholds] = {}
        for route in (
            STAGE1_ROUTE_STAGE2,
            STAGE1_NOVEL,
            STAGE1_REVIEW,
        ):
            route_known = global_known.loc[
                global_known["stage1_route"].eq(route)
            ]
            thresholds[route] = self._learn_route_threshold(
                route, route_known, global_known
            )

        self.route_thresholds_ = thresholds
        self.policy_validation_evidence_ = evidence.copy()

        route_counts = (
            evidence["stage1_route"].value_counts().sort_index().to_dict()
        )
        known_route_counts = (
            global_known["stage1_route"].value_counts().sort_index().to_dict()
        )
        diagnostics = Stage2FitDiagnostics(
            known_categories=list(self.known_categories_),
            category_training_rows=int(
                self.component_fit_metadata_["category_training_rows"]
            ),
            attack_novelty_training_rows=int(
                self.component_fit_metadata_["attack_novelty_training_rows"]
            ),
            attack_reference_rows=int(
                self.component_fit_metadata_["attack_reference_rows"]
            ),
            route_thresholds={
                route: asdict(value) for route, value in thresholds.items()
            },
            validation_route_counts={str(k): int(v) for k, v in route_counts.items()},
            validation_known_attack_counts={
                str(k): int(v) for k, v in known_route_counts.items()
            },
        )
        self.fit_diagnostics_ = diagnostics
        return diagnostics

    def _require_policy(self) -> None:
        if not hasattr(self, "route_thresholds_"):
            raise RuntimeError(
                "Stage-2 policy is not fitted. Call fit_policy before predict."
            )

    def fit_review_adjudicator(
        self,
        review_evidence_oof: pd.DataFrame,
        y_binary_review: pd.Series | Sequence[Any],
        *,
        estimator: BaseEstimator | None = None,
    ) -> "Stage2OpenSetRouter":
        """Fit an optional binary adjudicator on out-of-fold disagreement evidence.

        The caller is responsible for ensuring ``review_evidence_oof`` contains
        out-of-fold Stage-1 and Stage-2 scores. Do not train this model using
        scores produced by components fitted on the same observations.
        """

        feature_columns = [
            "xgb_attack_probability",
            "normal_novelty_percentile",
            "category_confidence",
            "category_margin",
            "category_entropy",
            "known_attack_novelty_percentile",
            "unknown_score",
        ]
        missing = sorted(set(feature_columns) - set(review_evidence_oof.columns))
        if missing:
            raise ValueError(
                f"Review adjudicator evidence is missing columns: {missing}"
            )

        y_binary = self._normalize_binary_target(y_binary_review)
        if estimator is None:
            estimator = LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=self.random_state,
            )
        self.review_adjudicator_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", clone(estimator)),
            ]
        )
        self.review_adjudicator_.fit(
            review_evidence_oof.loc[:, feature_columns], y_binary
        )
        self.review_adjudicator_features_ = feature_columns
        return self

    def _review_probability(self, evidence: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "review_adjudicator_"):
            return np.full(len(evidence), np.nan, dtype=float)
        model = self.review_adjudicator_
        X = evidence.loc[:, self.review_adjudicator_features_]
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(X))[:, 1]
        if hasattr(model, "decision_function"):
            raw = np.asarray(model.decision_function(X)).ravel()
            return 1.0 / (1.0 + np.exp(-raw))
        return np.asarray(model.predict(X), dtype=float)

    def _strong_unknown_condition(self, row: pd.Series) -> bool:
        config = self.policy_config
        weak_category = (
            row["category_confidence"] <= config.unknown_max_category_confidence
            or row["category_margin"] <= config.unknown_max_category_margin
            or row["category_entropy"] >= config.unknown_min_category_entropy
        )
        return bool(
            row["normal_novelty_percentile"]
            >= config.unknown_min_normal_novelty
            and row["known_attack_novelty_percentile"]
            >= config.unknown_min_attack_novelty
            and weak_category
        )

    def _novel_route_known_recovery(self, row: pd.Series) -> bool:
        config = self.policy_config
        threshold = self.route_thresholds_[STAGE1_NOVEL]
        return bool(
            row["unknown_score"] <= threshold.unknown_score_threshold
            and row["category_confidence"] >= config.novel_route_min_confidence
            and row["category_margin"] >= config.novel_route_min_margin
            and row["category_entropy"] <= config.novel_route_max_entropy
            and row["known_attack_novelty_percentile"]
            <= config.novel_route_max_attack_novelty
        )

    def _review_route_known_acceptance(self, row: pd.Series) -> bool:
        config = self.policy_config
        threshold = self.route_thresholds_[STAGE1_REVIEW]
        return bool(
            row["unknown_score"] <= threshold.unknown_score_threshold
            and row["category_confidence"] >= config.review_route_min_confidence
            and row["category_margin"] >= config.review_route_min_margin
            and row["category_entropy"] <= config.review_route_max_entropy
            and row["known_attack_novelty_percentile"]
            <= config.review_route_max_attack_novelty
        )

    def _decide_row(self, row: pd.Series) -> tuple[str, str, str]:
        if not bool(row["data_quality_valid"]):
            return (
                STAGE2_DATA_ERROR,
                "data_quality",
                str(row["data_quality_reason"]),
            )

        route = str(row["stage1_route"])
        predicted = str(row["predicted_known_category"])

        if route == STAGE1_NORMAL:
            return (
                STAGE2_STAGE1_NORMAL,
                "stage1_final_normal",
                "Stage 1 already finalized the record as Normal.",
            )

        if route == STAGE1_ROUTE_STAGE2:
            threshold = self.route_thresholds_[STAGE1_ROUTE_STAGE2]
            if row["unknown_score"] <= threshold.unknown_score_threshold:
                return (
                    f"KNOWN_ATTACK_{predicted}",
                    "known_attack",
                    "Strong Stage-1 attack evidence and acceptable known-attack fit.",
                )
            if self._strong_unknown_condition(row):
                return (
                    STAGE2_UNKNOWN,
                    "unknown_attack",
                    "Poor known-category fit plus high normal and attack novelty.",
                )
            return (
                STAGE2_REVIEW,
                "review",
                "Stage-1 attack evidence is strong, but Stage 2 cannot accept or reject a known category safely.",
            )

        if route == STAGE1_NOVEL:
            if self._novel_route_known_recovery(row):
                return (
                    f"KNOWN_ATTACK_{predicted}",
                    "known_attack_recovered",
                    "Stage 2 recovered a very strong known-attack match from a novelty route.",
                )
            if self._strong_unknown_condition(row):
                return (
                    STAGE2_UNKNOWN,
                    "unknown_attack",
                    "Extreme normal novelty, extreme known-attack novelty, and weak category evidence.",
                )
            return (
                STAGE2_REVIEW,
                "review",
                "Novel event has mixed evidence and cannot be assigned safely.",
            )

        if route == STAGE1_REVIEW:
            if self._review_route_known_acceptance(row):
                return (
                    f"KNOWN_ATTACK_{predicted}",
                    "known_attack_resolved",
                    "Disagreement resolved by strong known-category and low attack-novelty evidence.",
                )
            if self._strong_unknown_condition(row):
                return (
                    STAGE2_UNKNOWN,
                    "unknown_attack",
                    "Disagreement resolved toward unknown because both novelty spaces are high and category evidence is weak.",
                )

            review_probability = row.get("review_adjudicator_probability", np.nan)
            config = self.policy_config
            if (
                config.enable_review_normal_clearance
                and np.isfinite(review_probability)
                and review_probability <= config.review_adjudicator_low
                and row["normal_novelty_percentile"]
                <= config.review_normal_max_normal_novelty
                and row["category_confidence"]
                <= config.unknown_max_category_confidence
            ):
                return (
                    STAGE2_NORMAL_AFTER_REVIEW,
                    "normal_after_review",
                    "Dedicated disagreement adjudicator found very low attack risk.",
                )

            return (
                STAGE2_REVIEW,
                "review",
                "Disagreement remains unresolved after category and attack-novelty scoring.",
            )

        raise ValueError(f"Unexpected Stage-1 route: {route}")

    def predict(
        self,
        X: pd.DataFrame,
        stage1_evidence: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return Stage-2 evidence, decisions, reasons, and route metadata."""

        self._require_policy()
        evidence = self.build_evidence(X, stage1_evidence)
        evidence["review_adjudicator_probability"] = self._review_probability(
            evidence
        )

        decisions: list[str] = []
        decision_types: list[str] = []
        reasons: list[str] = []
        for _, row in evidence.iterrows():
            decision, decision_type, reason = self._decide_row(row)
            decisions.append(decision)
            decision_types.append(decision_type)
            reasons.append(reason)

        result = evidence.copy()
        result["stage2_decision"] = decisions
        result["stage2_decision_type"] = decision_types
        result["stage2_reason"] = reasons
        result["closest_known_category"] = result["predicted_known_category"]
        result["review_priority"] = np.select(
            [
                result["stage2_decision"].eq(STAGE2_UNKNOWN),
                result["stage2_decision"].eq(STAGE2_DATA_ERROR),
                result["stage2_decision"].eq(STAGE2_REVIEW)
                & result["normal_novelty_percentile"].ge(
                    self.policy_config.unknown_min_normal_novelty
                ),
                result["stage2_decision"].eq(STAGE2_REVIEW),
            ],
            ["critical", "high", "high", "medium"],
            default="low",
        )
        return result

    def _fitted_category_pipeline(self) -> Pipeline:
        self._require_components()
        calibrated = getattr(self.category_classifier_, "calibrated_classifiers_", None)
        if calibrated:
            return calibrated[0].estimator
        return self.category_classifier_.estimator

    @staticmethod
    def _preprocessor_state(preprocessor: ColumnTransformer) -> dict[str, Any]:
        state: dict[str, Any] = {
            "encoded_feature_names": preprocessor.get_feature_names_out().tolist(),
        }
        if "numeric" in preprocessor.named_transformers_:
            numeric = preprocessor.named_transformers_["numeric"]
            if hasattr(numeric, "named_steps"):
                imputer = numeric.named_steps.get("imputer")
                scaler = numeric.named_steps.get("scaler")
                if imputer is not None and hasattr(imputer, "statistics_"):
                    state["numeric_imputer_statistics"] = np.asarray(
                        imputer.statistics_
                    ).tolist()
                if scaler is not None:
                    if hasattr(scaler, "center_"):
                        state["numeric_scaler_center"] = np.asarray(
                            scaler.center_
                        ).tolist()
                    if hasattr(scaler, "scale_"):
                        state["numeric_scaler_scale"] = np.asarray(
                            scaler.scale_
                        ).tolist()
        if "categorical" in preprocessor.named_transformers_:
            categorical = preprocessor.named_transformers_["categorical"]
            if hasattr(categorical, "named_steps"):
                imputer = categorical.named_steps.get("imputer")
                encoder = categorical.named_steps.get("onehot")
                if imputer is not None and hasattr(imputer, "statistics_"):
                    state["categorical_imputer_statistics"] = [
                        None if pd.isna(value) else str(value)
                        for value in imputer.statistics_
                    ]
                if encoder is not None and hasattr(encoder, "categories_"):
                    state["onehot_categories"] = [
                        [str(value) for value in values]
                        for values in encoder.categories_
                    ]
                    infrequent = getattr(encoder, "infrequent_categories_", None)
                    if infrequent is not None:
                        state["onehot_infrequent_categories"] = [
                            []
                            if values is None
                            else [str(value) for value in values]
                            for values in infrequent
                        ]
        return state

    def component_state(self) -> dict[str, Any]:
        """Return fitted encoder, filter, and selected-feature metadata."""

        self._require_components()
        category_pipeline = self._fitted_category_pipeline()
        category_pre = category_pipeline.named_steps["preprocess"]
        category_names = np.asarray(category_pre.get_feature_names_out(), dtype=object)
        category_constant = category_pipeline.named_steps["constant_filter"]
        category_support = category_constant.get_support()
        category_kept = category_names[category_support]

        attack_pipeline = self.attack_novelty_pipeline_
        attack_pre = attack_pipeline.named_steps["preprocess"]
        attack_names = np.asarray(attack_pre.get_feature_names_out(), dtype=object)
        if "constant_filter" in attack_pipeline.named_steps:
            attack_constant = attack_pipeline.named_steps["constant_filter"]
            attack_after_constant = attack_names[attack_constant.get_support()]
        else:
            attack_after_constant = attack_names
        if "correlation_filter" in attack_pipeline.named_steps:
            attack_correlation = attack_pipeline.named_steps["correlation_filter"]
            attack_selected = attack_after_constant[attack_correlation.get_support()]
        else:
            attack_selected = attack_after_constant

        return {
            "category": {
                "raw_features": list(self.category_feature_columns),
                "preprocessor": self._preprocessor_state(category_pre),
                "encoded_feature_count": int(len(category_names)),
                "post_constant_feature_count": int(len(category_kept)),
                "post_constant_feature_names": category_kept.tolist(),
                "classes": list(self.known_categories_),
            },
            "attack_novelty": {
                "raw_features": list(self.attack_novelty_feature_columns),
                "preprocessor": self._preprocessor_state(attack_pre),
                "encoded_feature_count": int(len(attack_names)),
                "post_constant_feature_count": int(len(attack_after_constant)),
                "post_correlation_feature_count": int(len(attack_selected)),
                "post_correlation_feature_names": attack_selected.tolist(),
                "reference_score_count": int(len(self.attack_reference_scores_)),
                "reference_score_quantiles": {
                    str(q): float(np.quantile(self.attack_reference_scores_, q))
                    for q in (0.01, 0.05, 0.50, 0.95, 0.99)
                },
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "Stage2OpenSetRouter":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Artifact does not contain {cls.__name__}.")
        return model

    def export_policy(self, path: str | Path) -> None:
        self._require_policy()
        payload = {
            "known_categories": list(self.known_categories_),
            "policy_config": asdict(self.policy_config),
            "route_thresholds": {
                route: asdict(value)
                for route, value in self.route_thresholds_.items()
            },
            "component_fit_metadata": dict(self.component_fit_metadata_),
            "component_state": self.component_state(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class AttackCategoryNoveltyBank:
    """One fitted novelty detector per known attack category.

    Each detector is trained on its category only. A separate category reference
    split supplies the score threshold and empirical novelty percentiles.
    """

    def __init__(
        self,
        feature_columns: Sequence[str],
        algorithm_by_category: Mapping[str, str],
        categorical_columns: Sequence[str] = ("protocoltype", "service", "flag"),
        threshold_quantile: float = 0.99,
        reference_fraction: float = 0.20,
        fit_limit: int | None = 6000,
        random_state: int = 42,
    ):
        self.feature_columns = tuple(feature_columns)
        self.algorithm_by_category = dict(algorithm_by_category)
        self.categorical_columns = tuple(categorical_columns)
        self.threshold_quantile = threshold_quantile
        self.reference_fraction = reference_fraction
        self.fit_limit = fit_limit
        self.random_state = random_state

    def _preprocessor(self) -> ColumnTransformer:
        categorical = [c for c in self.categorical_columns if c in self.feature_columns]
        numeric = [c for c in self.feature_columns if c not in categorical]
        return ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def _detector(self, algorithm: str, fit_rows: int) -> BaseEstimator:
        if algorithm == "IsolationForest":
            return IsolationForest(
                n_estimators=300,
                max_samples=1.0,
                contamination="auto",
                n_jobs=-1,
                random_state=self.random_state,
            )
        if algorithm == "OneClassSVM":
            return OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
        raise ValueError(f"Unsupported category novelty algorithm: {algorithm}")

    def fit(self, X: pd.DataFrame, y_category: Sequence[Any]) -> "AttackCategoryNoveltyBank":
        frame = pd.DataFrame(X).copy()
        missing = sorted(set(self.feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Category novelty input is missing columns: {missing}")
        labels = pd.Series(y_category, index=frame.index).astype(str)
        self.models_: dict[str, dict[str, Any]] = {}
        rng = np.random.default_rng(self.random_state)
        for offset, (category, algorithm) in enumerate(self.algorithm_by_category.items()):
            category_frame = frame.loc[labels.eq(category), list(self.feature_columns)]
            if len(category_frame) < 10:
                raise ValueError(f"Too few rows to fit category novelty for {category}")
            fit_frame, reference_frame = train_test_split(
                category_frame,
                test_size=self.reference_fraction,
                random_state=self.random_state + offset,
            )
            if self.fit_limit is not None and len(fit_frame) > self.fit_limit:
                positions = rng.choice(len(fit_frame), size=self.fit_limit, replace=False)
                fit_frame = fit_frame.iloc[positions].copy()
            preprocessor = self._preprocessor()
            fit_values = preprocessor.fit_transform(fit_frame)
            reference_values = preprocessor.transform(reference_frame)
            detector = self._detector(algorithm, len(fit_frame))
            detector.fit(fit_values)
            reference_scores = -np.asarray(detector.decision_function(reference_values)).ravel()
            self.models_[category] = {
                "algorithm": algorithm,
                "preprocessor": preprocessor,
                "detector": detector,
                "reference_scores": np.sort(reference_scores.astype(float)),
                "threshold": float(np.quantile(reference_scores, self.threshold_quantile)),
                "fit_rows": int(len(fit_frame)),
                "reference_rows": int(len(reference_frame)),
                "encoded_dimensions": int(fit_values.shape[1]),
            }
        return self

    def score(self, X: pd.DataFrame, predicted_categories: Sequence[Any]) -> pd.DataFrame:
        if not hasattr(self, "models_"):
            raise RuntimeError("AttackCategoryNoveltyBank is not fitted.")
        frame = pd.DataFrame(X).copy()
        predicted = pd.Series(predicted_categories, index=frame.index).astype(str)
        output = pd.DataFrame(index=frame.index)
        output["category_novelty_algorithm"] = ""
        output["category_novelty_score"] = np.nan
        output["category_novelty_percentile"] = np.nan
        output["category_novelty_threshold"] = np.nan
        output["category_novelty_is_novel"] = False
        for category, state in self.models_.items():
            mask = predicted.eq(category)
            if not mask.any():
                continue
            values = state["preprocessor"].transform(
                frame.loc[mask, list(self.feature_columns)]
            )
            scores = -np.asarray(state["detector"].decision_function(values)).ravel()
            reference = state["reference_scores"]
            ranks = np.searchsorted(reference, scores, side="right")
            percentiles = (ranks + 1.0) / (len(reference) + 1.0)
            output.loc[mask, "category_novelty_algorithm"] = state["algorithm"]
            output.loc[mask, "category_novelty_score"] = scores
            output.loc[mask, "category_novelty_percentile"] = percentiles
            output.loc[mask, "category_novelty_threshold"] = state["threshold"]
            output.loc[mask, "category_novelty_is_novel"] = scores >= state["threshold"]
        return output

    def state(self) -> dict[str, Any]:
        if not hasattr(self, "models_"):
            raise RuntimeError("AttackCategoryNoveltyBank is not fitted.")
        return {
            category: {
                key: value
                for key, value in state.items()
                if key not in {"preprocessor", "detector", "reference_scores"}
            }
            for category, state in self.models_.items()
        }


@dataclass
class CompleteStage2Router:
    """Combine the open-set router with predicted-category novelty confirmation."""

    open_set_router: Stage2OpenSetRouter
    category_novelty_bank: AttackCategoryNoveltyBank
    model_version: str = "stage2-router-v1"
    category_novelty_version: str = "stage2-category-novelty-v1"

    def predict(self, X: pd.DataFrame, stage1_evidence: pd.DataFrame) -> pd.DataFrame:
        result = self.open_set_router.predict(X, stage1_evidence)
        category_evidence = self.category_novelty_bank.score(
            X, result["predicted_known_category"]
        )
        result = result.join(category_evidence)
        accepted = result["stage2_decision"].astype(str).str.startswith("KNOWN_ATTACK_")
        rejected_by_category = accepted & result["category_novelty_is_novel"].astype(bool)
        result.loc[rejected_by_category, "stage2_decision"] = STAGE2_REVIEW
        result.loc[rejected_by_category, "stage2_decision_type"] = "review"
        result.loc[rejected_by_category, "stage2_reason"] = (
            "Global known-attack evidence passed, but the predicted-category novelty model rejected the family."
        )
        result.loc[rejected_by_category, "review_priority"] = "high"
        result["stage2_model_version"] = self.model_version
        result["category_novelty_version"] = self.category_novelty_version
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "CompleteStage2Router":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Artifact does not contain {cls.__name__}.")
        return model


def evaluate_stage2(
    predictions: pd.DataFrame,
    y_binary: pd.Series | Sequence[Any],
    y_category: pd.Series | Sequence[Any],
    known_categories: Sequence[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate Stage-2 outputs on known and optional held-out unknown classes.

    A category absent from ``known_categories`` is treated as an unknown class.
    """

    y_binary_series = Stage2OpenSetRouter._normalize_binary_target(y_binary)
    y_category_series = Stage2OpenSetRouter._normalize_category_target(y_category)
    y_binary_series.index = predictions.index
    y_category_series.index = predictions.index

    known_set = set(map(str, known_categories))
    true_known_attack = y_binary_series.eq(1) & y_category_series.isin(known_set)
    true_unknown_attack = y_binary_series.eq(1) & ~y_category_series.isin(known_set)
    true_normal = y_binary_series.eq(0)

    decision = predictions["stage2_decision"].astype(str)
    accepted_known = decision.str.startswith("KNOWN_ATTACK_")
    predicted_category = decision.str.replace("KNOWN_ATTACK_", "", regex=False)
    predicted_unknown = decision.eq(STAGE2_UNKNOWN)
    predicted_review = decision.eq(STAGE2_REVIEW)
    predicted_normal = decision.isin(
        [STAGE2_NORMAL_AFTER_REVIEW, STAGE2_STAGE1_NORMAL]
    )

    metrics: dict[str, float] = {
        "record_count": float(len(predictions)),
        "known_attack_count": float(true_known_attack.sum()),
        "unknown_attack_count": float(true_unknown_attack.sum()),
        "normal_count": float(true_normal.sum()),
        "known_attack_acceptance_rate": float(
            accepted_known[true_known_attack].mean()
            if true_known_attack.any()
            else np.nan
        ),
        "known_attack_false_unknown_rate": float(
            predicted_unknown[true_known_attack].mean()
            if true_known_attack.any()
            else np.nan
        ),
        "known_attack_review_rate": float(
            predicted_review[true_known_attack].mean()
            if true_known_attack.any()
            else np.nan
        ),
        "unknown_attack_recall": float(
            predicted_unknown[true_unknown_attack].mean()
            if true_unknown_attack.any()
            else np.nan
        ),
        "normal_unknown_escalation_rate": float(
            predicted_unknown[true_normal].mean() if true_normal.any() else np.nan
        ),
        "normal_known_attack_false_positive_rate": float(
            accepted_known[true_normal].mean() if true_normal.any() else np.nan
        ),
        "normal_after_review_rate": float(
            predicted_normal[true_normal].mean() if true_normal.any() else np.nan
        ),
        "overall_review_rate": float(predicted_review.mean()),
    }

    accepted_known_mask = true_known_attack & accepted_known
    if accepted_known_mask.any():
        metrics["known_category_accuracy_when_accepted"] = float(
            accuracy_score(
                y_category_series.loc[accepted_known_mask],
                predicted_category.loc[accepted_known_mask],
            )
        )
        metrics["known_category_macro_f1_when_accepted"] = float(
            f1_score(
                y_category_series.loc[accepted_known_mask],
                predicted_category.loc[accepted_known_mask],
                average="macro",
                zero_division=0,
            )
        )
    else:
        metrics["known_category_accuracy_when_accepted"] = np.nan
        metrics["known_category_macro_f1_when_accepted"] = np.nan

    route_rows: list[dict[str, Any]] = []
    for route, route_frame in predictions.groupby("stage1_route", dropna=False):
        idx = route_frame.index
        route_rows.append(
            {
                "stage1_route": str(route),
                "record_count": int(len(idx)),
                "true_attack_rate": float(y_binary_series.loc[idx].mean()),
                "known_attack_decision_rate": float(
                    route_frame["stage2_decision"].astype(str).str.startswith(
                        "KNOWN_ATTACK_"
                    ).mean()
                ),
                "unknown_candidate_rate": float(
                    route_frame["stage2_decision"].eq(STAGE2_UNKNOWN).mean()
                ),
                "review_rate": float(
                    route_frame["stage2_decision"].eq(STAGE2_REVIEW).mean()
                ),
                "data_quality_exception_rate": float(
                    route_frame["stage2_decision"].eq(STAGE2_DATA_ERROR).mean()
                ),
            }
        )

    return metrics, pd.DataFrame(route_rows)


def stage2_classification_report(
    predictions: pd.DataFrame,
    y_category: pd.Series | Sequence[Any],
) -> dict[str, Any]:
    """Classification report for records accepted as known attacks."""

    categories = Stage2OpenSetRouter._normalize_category_target(y_category)
    categories.index = predictions.index
    accepted = predictions["stage2_decision"].astype(str).str.startswith(
        "KNOWN_ATTACK_"
    )
    if not accepted.any():
        return {}
    predicted = predictions.loc[accepted, "stage2_decision"].str.replace(
        "KNOWN_ATTACK_", "", regex=False
    )
    return classification_report(
        categories.loc[accepted], predicted, output_dict=True, zero_division=0
    )
