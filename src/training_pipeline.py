"""Notebook-equivalent training pipeline for both hybrid detection stages."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

from .features import (
    CATEGORICAL_FEATURES,
    COMBINED_FEATURES,
    ORIGINAL_FEATURES,
    STAGE1_NOVELTY_FEATURES,
    engineer_features,
)
from .legacy_models import install_notebook_module_path


RANDOM_STATE = 42
REFERENCE_FRACTION = 0.15
VALIDATION_FRACTION = 0.15
CALIBRATION_FOLDS = 5
NORMAL_FIT_LIMIT = 6000
ATTACK_CATEGORIES = ("DoS", "Probe", "R2L", "U2R")

THRESHOLD_BUDGETS = {
    "xgb_auto_attack_fpr": 0.001,
    "xgb_auto_normal_attack_escape_rate": 0.001,
    "xgb_auto_normal_region_attack_rate": 0.001,
    "novelty_high_fpr": 0.01,
    "novelty_extreme_fpr": 0.001,
}

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

IFOREST_PARAMS = {
    "n_estimators": 300,
    "max_samples": 1.0,
    "contamination": "auto",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

CATEGORY_NOVELTY_ALGORITHMS = {
    "DoS": "OneClassSVM",
    "Probe": "IsolationForest",
    "R2L": "OneClassSVM",
    "U2R": "OneClassSVM",
}


def _notebook_classes(root: Path):
    install_notebook_module_path(root)
    from hybrid_xgb_iforest_router import (
        HybridRouter,
        NormalNoveltyPercentile,
        learn_routing_thresholds,
        raw_iforest_anomaly_score,
        split_training_for_hybrid,
    )
    from stage2_router import (
        AttackCategoryNoveltyBank,
        CompleteStage2Router,
        DataQualityConfig,
        Stage2OpenSetRouter,
        Stage2PolicyConfig,
    )
    return locals()


def _preprocessor(feature_columns: tuple[str, ...], *, scale_numeric: bool) -> ColumnTransformer:
    categorical = [x for x in feature_columns if x in CATEGORICAL_FEATURES]
    numeric = [x for x in feature_columns if x not in categorical]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def train_hybrid_system(
    data: pd.DataFrame,
    output_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    """Train notebook-equivalent Stage-1 and Stage-2 artifacts.

    ``data`` must contain the 41 original fields plus ``binary_target`` and
    ``attack_category``. Existing engineered columns are deliberately ignored
    and recomputed by the canonical inference transformer.
    """
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    classes = _notebook_classes(root)
    required_targets = {"binary_target", "attack_category"}
    missing_targets = sorted(required_targets - set(data.columns))
    if missing_targets:
        raise ValueError(f"Training data is missing targets: {missing_targets}")

    X_all = engineer_features(data.loc[:, list(ORIGINAL_FEATURES)])
    y_all = data["binary_target"].astype(str).map({"Normal": 0, "Attack": 1})
    if y_all.isna().any():
        raise ValueError("binary_target must contain only Normal and Attack")
    y_all = y_all.astype(int)
    family_all = data["attack_category"].astype(str)

    positions = np.arange(len(data))
    train_positions, test_positions = train_test_split(
        positions, test_size=0.20, stratify=y_all, random_state=RANDOM_STATE
    )
    X_train, X_test = X_all.iloc[train_positions], X_all.iloc[test_positions]
    y_train, y_test = y_all.iloc[train_positions].to_numpy(), y_all.iloc[test_positions].to_numpy()

    X_fit, X_reference, X_validation, y_fit, y_reference, y_validation = classes[
        "split_training_for_hybrid"
    ](
        X_train,
        y_train,
        reference_fraction=REFERENCE_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
    )

    xgb_pipeline = Pipeline(
        [
            ("preprocessor", _preprocessor(ORIGINAL_FEATURES, scale_numeric=True)),
            ("model", XGBClassifier(**XGB_PARAMS)),
        ]
    )
    calibrated_xgboost = CalibratedClassifierCV(
        estimator=xgb_pipeline,
        method="sigmoid",
        cv=CALIBRATION_FOLDS,
        ensemble=False,
        n_jobs=-1,
    )
    calibrated_xgboost.fit(X_fit.loc[:, list(ORIGINAL_FEATURES)], y_fit)
    validation_xgb_probability = calibrated_xgboost.predict_proba(
        X_validation.loc[:, list(ORIGINAL_FEATURES)]
    )[:, 1]

    normal_iforest = Pipeline(
        [
            ("preprocessor", _preprocessor(STAGE1_NOVELTY_FEATURES, scale_numeric=True)),
            ("detector", IsolationForest(**IFOREST_PARAMS)),
        ]
    )
    normal_fit = X_fit.loc[y_fit == 0, list(STAGE1_NOVELTY_FEATURES)].copy()
    if len(normal_fit) > NORMAL_FIT_LIMIT:
        normal_fit = normal_fit.sample(n=NORMAL_FIT_LIMIT, random_state=RANDOM_STATE)
    normal_iforest.fit(normal_fit)
    normal_reference = X_reference.loc[y_reference == 0, list(STAGE1_NOVELTY_FEATURES)]
    reference_scores = classes["raw_iforest_anomaly_score"](normal_iforest, normal_reference)
    novelty_calibrator = classes["NormalNoveltyPercentile"]().fit(reference_scores)
    validation_scores = classes["raw_iforest_anomaly_score"](
        normal_iforest, X_validation.loc[:, list(STAGE1_NOVELTY_FEATURES)]
    )
    validation_percentiles = novelty_calibrator.transform(validation_scores)
    thresholds, threshold_diagnostics = classes["learn_routing_thresholds"](
        y_validation, validation_xgb_probability, validation_percentiles, **THRESHOLD_BUDGETS
    )
    stage1_router = classes["HybridRouter"](
        original_features=ORIGINAL_FEATURES,
        combined_features=STAGE1_NOVELTY_FEATURES,
        calibrated_xgboost=calibrated_xgboost,
        normal_iforest=normal_iforest,
        novelty_calibrator=novelty_calibrator,
        thresholds=thresholds,
        model_version="stage1-router-v1",
        threshold_version="routing-thresholds-v1",
    )

    family_fit = family_all.loc[X_fit.index].to_numpy()
    family_reference = family_all.loc[X_reference.index].to_numpy()
    family_validation = family_all.loc[X_validation.index].to_numpy()
    y_fit_attack = (family_fit != "Normal").astype(int)
    y_reference_attack = (family_reference != "Normal").astype(int)
    y_validation_attack = (family_validation != "Normal").astype(int)

    quality = classes["DataQualityConfig"](
        rate_columns=(
            "serrorrate", "srvserrorrate", "rerrorrate", "srvrerrorrate",
            "samesrvrate", "diffsrvrate", "srvdiffhostrate",
            "dsthostsamesrvrate", "dsthostdiffsrvrate", "dsthostsamesrcportrate",
            "dsthostsrvdiffhostrate", "dsthostserrorrate", "dsthostsrvserrorrate",
            "dsthostrerrorrate", "dsthostsrvrerrorrate",
        ),
        nonnegative_columns=("duration", "srcbytes", "dstbytes", "count", "srvcount", "dsthostcount", "dsthostsrvcount"),
        binary_columns=("land", "loggedin", "rootshell", "ishostlogin", "isguestlogin"),
        maximum_row_missing_fraction=0.25,
    )
    policy = classes["Stage2PolicyConfig"](
        route_stage2_known_rejection_rate=0.02,
        novel_route_known_rejection_rate=0.10,
        review_route_known_rejection_rate=0.05,
        minimum_route_calibration_samples=40,
    )
    stage2_open = classes["Stage2OpenSetRouter"](
        category_feature_columns=ORIGINAL_FEATURES,
        attack_novelty_feature_columns=ORIGINAL_FEATURES,
        category_model=RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        attack_novelty_model=OneClassSVM(kernel="rbf", nu=0.05, gamma="scale"),
        category_calibration_method="sigmoid",
        category_calibration_folds=5,
        category_onehot_min_frequency=None,
        attack_onehot_min_frequency=None,
        attack_apply_feature_filters=False,
        attack_numeric_scaler="standard",
        attack_novelty_fit_limit=6000,
        policy_config=policy,
        data_quality_config=quality,
        random_state=RANDOM_STATE,
    )
    attack_reference = X_reference.loc[y_reference_attack == 1, list(ORIGINAL_FEATURES)]
    stage2_open.fit_components(
        X_fit.loc[:, list(ORIGINAL_FEATURES)],
        y_fit_attack,
        family_fit,
        X_attack_reference=attack_reference,
    )
    attack_fit_mask = y_fit_attack == 1
    category_bank = classes["AttackCategoryNoveltyBank"](
        feature_columns=ORIGINAL_FEATURES,
        algorithm_by_category=CATEGORY_NOVELTY_ALGORITHMS,
        threshold_quantile=0.99,
        reference_fraction=0.20,
        fit_limit=6000,
        random_state=RANDOM_STATE,
    )
    category_bank.fit(
        X_fit.loc[attack_fit_mask, list(ORIGINAL_FEATURES)], family_fit[attack_fit_mask]
    )
    stage1_validation_details = stage1_router.score(X_validation)
    stage1_validation_details.index = X_validation.index
    stage1_validation = pd.DataFrame(
        {
            "stage1_route": stage1_validation_details["route"],
            "xgb_attack_probability": stage1_validation_details["xgb_attack_probability"],
            "normal_novelty_percentile": stage1_validation_details["normal_novelty_percentile"],
        },
        index=X_validation.index,
    )
    policy_diagnostics = stage2_open.fit_policy(
        X_validation.loc[:, list(ORIGINAL_FEATURES)],
        y_validation_attack,
        family_validation,
        stage1_validation,
    )
    stage2_router = classes["CompleteStage2Router"](
        open_set_router=stage2_open,
        category_novelty_bank=category_bank,
        model_version="stage2-router-v1",
        category_novelty_version="stage2-category-novelty-v1",
    )

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stage1_path = output / "hybrid_stage1_router.joblib"
    stage2_path = output / "complete_stage2_router.joblib"
    joblib.dump(stage1_router, stage1_path)
    joblib.dump(stage2_router, stage2_path)
    manifest = {
        "stage1_artifact": stage1_path.name,
        "stage2_artifact": stage2_path.name,
        "random_state": RANDOM_STATE,
        "input_features": list(ORIGINAL_FEATURES),
        "engineered_feature_count": len(COMBINED_FEATURES) - len(ORIGINAL_FEATURES),
        "stage1_novelty_features": list(STAGE1_NOVELTY_FEATURES),
        "split_rows": {
            "train_pool": len(X_train), "fit": len(X_fit), "reference": len(X_reference),
            "validation": len(X_validation), "external_test": len(X_test),
        },
        "stage1_thresholds": asdict(thresholds),
        "stage1_threshold_diagnostics": asdict(threshold_diagnostics),
        "stage2_policy_diagnostics": policy_diagnostics.to_dict(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"stage1_router": stage1_router, "stage2_router": stage2_router, "manifest": manifest}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Train both stages of the Sentiflow hybrid IDS")
    parser.add_argument("data_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = train_hybrid_system(pd.read_csv(args.data_csv), args.output_dir)
    print(json.dumps(result["manifest"], indent=2))


if __name__ == "__main__":
    _main()
