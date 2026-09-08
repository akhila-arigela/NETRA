"""Canonical raw schema and deterministic feature engineering.

The formulas in this module are the deployable equivalent of the feature
engineering performed before ``Notebooks/Hybrid.ipynb`` reads
``Data/Consolidated_df.csv``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


ORIGINAL_FEATURES = (
    "duration", "protocoltype", "service", "flag", "srcbytes", "dstbytes",
    "land", "wrongfragment", "urgent", "hot", "numfailedlogins", "loggedin",
    "numcompromised", "rootshell", "suattempted", "numroot",
    "numfilecreations", "numshells", "numaccessfiles", "numoutboundcmds",
    "ishostlogin", "isguestlogin", "count", "srvcount", "serrorrate",
    "srvserrorrate", "rerrorrate", "srvrerrorrate", "samesrvrate",
    "diffsrvrate", "srvdiffhostrate", "dsthostcount", "dsthostsrvcount",
    "dsthostsamesrvrate", "dsthostdiffsrvrate", "dsthostsamesrcportrate",
    "dsthostsrvdiffhostrate", "dsthostserrorrate", "dsthostsrvserrorrate",
    "dsthostrerrorrate", "dsthostsrvrerrorrate",
)

CATEGORICAL_FEATURES = ("protocoltype", "service", "flag")
NUMERIC_FEATURES = tuple(x for x in ORIGINAL_FEATURES if x not in CATEGORICAL_FEATURES)

ENGINEERED_FEATURES = (
    "total_bytes", "bytes_per_second", "src_dst_byte_ratio",
    "src_byte_fraction", "dst_byte_fraction", "byte_asymmetry",
    "service_connection_ratio", "host_service_ratio",
    "different_service_connections", "different_host_service_connections",
    "short_term_scan_pressure", "host_scan_pressure",
    "same_source_port_pressure", "short_term_serror_score",
    "short_term_rerror_score", "short_term_error_score", "host_serror_score",
    "host_rerror_score", "host_error_score", "same_service_rate_gap",
    "different_service_rate_gap", "serror_rate_gap", "rerror_rate_gap",
    "authentication_risk_score", "has_failed_login",
    "failed_login_and_logged_in", "suspicious_admin_activity",
    "privileged_activity_score", "content_risk_score",
    "file_and_shell_activity", "root_compromise_ratio",
)

COMBINED_FEATURES = ORIGINAL_FEATURES + ENGINEERED_FEATURES

# Fixed upstream selection used by the notebook's Stage-1 Isolation Forest.
STAGE1_NOVELTY_FEATURES = (
    "duration", "protocoltype", "service", "srcbytes", "dstbytes", "land",
    "wrongfragment", "urgent", "hot", "numfailedlogins", "loggedin",
    "numcompromised", "rootshell", "suattempted", "numfilecreations",
    "numshells", "numaccessfiles", "numoutboundcmds", "ishostlogin",
    "isguestlogin", "count", "serrorrate", "srvserrorrate", "rerrorrate",
    "samesrvrate", "diffsrvrate", "srvdiffhostrate", "dsthostcount",
    "dsthostsrvcount", "dsthostsamesrvrate", "dsthostdiffsrvrate",
    "dsthostsamesrcportrate", "dsthostsrvdiffhostrate", "dsthostserrorrate",
    "dsthostsrvserrorrate", "dsthostrerrorrate", "dsthostsrvrerrorrate",
    "bytes_per_second", "src_dst_byte_ratio", "src_byte_fraction",
    "dst_byte_fraction", "service_connection_ratio", "host_service_ratio",
    "different_service_connections", "different_host_service_connections",
    "short_term_scan_pressure", "host_scan_pressure",
    "same_source_port_pressure", "different_service_rate_gap",
    "serror_rate_gap", "rerror_rate_gap", "authentication_risk_score",
    "has_failed_login", "failed_login_and_logged_in", "content_risk_score",
    "root_compromise_ratio",
)

RATE_FEATURES = (
    "serrorrate", "srvserrorrate", "rerrorrate", "srvrerrorrate",
    "samesrvrate", "diffsrvrate", "srvdiffhostrate", "dsthostsamesrvrate",
    "dsthostdiffsrvrate", "dsthostsamesrcportrate", "dsthostsrvdiffhostrate",
    "dsthostserrorrate", "dsthostsrvserrorrate", "dsthostrerrorrate",
    "dsthostsrvrerrorrate",
)
NONNEGATIVE_FEATURES = (
    "duration", "srcbytes", "dstbytes", "count", "srvcount",
    "dsthostcount", "dsthostsrvcount",
)
BINARY_FEATURES = (
    "land", "loggedin", "rootshell", "ishostlogin", "isguestlogin",
)


def _missing(columns: Iterable[str], frame: pd.DataFrame) -> list[str]:
    return sorted(set(columns) - set(frame.columns))


def coerce_raw_frame(data: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Return a typed 41-column raw frame without learning from incoming data."""
    frame = pd.DataFrame(data).copy()
    missing = _missing(ORIGINAL_FEATURES, frame)
    if missing:
        raise ValueError(f"Incoming data is missing required columns: {missing}")
    frame = frame.loc[:, list(ORIGINAL_FEATURES)]
    for column in NUMERIC_FEATURES:
        original_missing = frame[column].isna()
        converted = pd.to_numeric(frame[column], errors="coerce")
        if strict and (converted.isna() & ~original_missing).any():
            bad_rows = converted.index[converted.isna() & ~original_missing].tolist()[:10]
            raise ValueError(f"Non-numeric values in {column!r} at rows {bad_rows}")
        frame[column] = converted
    for column in CATEGORICAL_FEATURES:
        # Keep missing values for the fitted model imputer; stringify only values.
        frame[column] = frame[column].where(frame[column].isna(), frame[column].astype(str))
    return frame


def validate_raw_frame(data: pd.DataFrame, *, maximum_missing_fraction: float = 0.25) -> pd.DataFrame:
    """Return row-level quality status using the notebook Stage-2 rules."""
    frame = coerce_raw_frame(data)
    report = pd.DataFrame(index=frame.index)
    reasons: list[list[str]] = [[] for _ in range(len(frame))]
    missing_fraction = frame.isna().mean(axis=1)
    for pos in np.flatnonzero((missing_fraction > maximum_missing_fraction).to_numpy()):
        reasons[pos].append("excessive_missing_values")
    for column in RATE_FEATURES:
        invalid = frame[column].notna() & ~frame[column].between(0, 1, inclusive="both")
        for pos in np.flatnonzero(invalid.to_numpy()):
            reasons[pos].append(f"invalid_rate:{column}")
    for column in NONNEGATIVE_FEATURES:
        invalid = frame[column].notna() & frame[column].lt(0)
        for pos in np.flatnonzero(invalid.to_numpy()):
            reasons[pos].append(f"negative_value:{column}")
    for column in BINARY_FEATURES:
        invalid = frame[column].notna() & ~frame[column].isin([0, 1])
        for pos in np.flatnonzero(invalid.to_numpy()):
            reasons[pos].append(f"invalid_binary:{column}")
    numeric = frame.loc[:, list(NUMERIC_FEATURES)].to_numpy(dtype=float)
    for pos in np.flatnonzero((~np.isfinite(numeric) & ~np.isnan(numeric)).any(axis=1)):
        reasons[pos].append("nonfinite_numeric_value")
    report["data_quality_reason"] = [";".join(sorted(set(x))) for x in reasons]
    report["data_quality_valid"] = report["data_quality_reason"].eq("")
    report["row_missing_fraction"] = missing_fraction
    return report


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def engineer_features(data: pd.DataFrame, *, validate: bool = True) -> pd.DataFrame:
    """Add the exact 31 engineered columns used to create the notebook dataset."""
    raw = coerce_raw_frame(data)
    if validate:
        quality = validate_raw_frame(raw)
        invalid = quality.index[~quality["data_quality_valid"]]
        if len(invalid):
            examples = quality.loc[invalid, "data_quality_reason"].head(10).to_dict()
            raise ValueError(f"Invalid incoming rows: {examples}")
    out = raw.copy()
    out["total_bytes"] = out["srcbytes"] + out["dstbytes"]
    out["bytes_per_second"] = _safe_divide(out["total_bytes"], out["duration"] + 1)
    out["src_dst_byte_ratio"] = _safe_divide(out["srcbytes"], out["dstbytes"] + 1)
    out["src_byte_fraction"] = _safe_divide(out["srcbytes"], out["total_bytes"] + 1)
    out["dst_byte_fraction"] = _safe_divide(out["dstbytes"], out["total_bytes"] + 1)
    out["byte_asymmetry"] = _safe_divide(out["srcbytes"] - out["dstbytes"], out["total_bytes"] + 1)
    out["service_connection_ratio"] = _safe_divide(out["srvcount"], out["count"] + 1)
    out["host_service_ratio"] = _safe_divide(out["dsthostsrvcount"], out["dsthostcount"] + 1)
    out["different_service_connections"] = (out["count"] - out["srvcount"]).clip(lower=0)
    out["different_host_service_connections"] = (out["dsthostcount"] - out["dsthostsrvcount"]).clip(lower=0)
    out["short_term_scan_pressure"] = out["count"] * out["diffsrvrate"]
    out["host_scan_pressure"] = out["dsthostcount"] * out["dsthostdiffsrvrate"]
    out["same_source_port_pressure"] = out["dsthostsrvcount"] * out["dsthostsamesrcportrate"]
    out["short_term_serror_score"] = (out["serrorrate"] + out["srvserrorrate"]) / 2
    out["short_term_rerror_score"] = (out["rerrorrate"] + out["srvrerrorrate"]) / 2
    out["short_term_error_score"] = (out["serrorrate"] + out["srvserrorrate"] + out["rerrorrate"] + out["srvrerrorrate"]) / 4
    out["host_serror_score"] = (out["dsthostserrorrate"] + out["dsthostsrvserrorrate"]) / 2
    out["host_rerror_score"] = (out["dsthostrerrorrate"] + out["dsthostsrvrerrorrate"]) / 2
    out["host_error_score"] = (out["dsthostserrorrate"] + out["dsthostsrvserrorrate"] + out["dsthostrerrorrate"] + out["dsthostsrvrerrorrate"]) / 4
    out["same_service_rate_gap"] = (out["samesrvrate"] - out["dsthostsamesrvrate"]).abs()
    out["different_service_rate_gap"] = (out["diffsrvrate"] - out["dsthostdiffsrvrate"]).abs()
    out["serror_rate_gap"] = (out["serrorrate"] - out["dsthostserrorrate"]).abs()
    out["rerror_rate_gap"] = (out["rerrorrate"] - out["dsthostrerrorrate"]).abs()
    out["authentication_risk_score"] = out[["numfailedlogins", "rootshell", "suattempted", "ishostlogin", "isguestlogin"]].sum(axis=1)
    out["has_failed_login"] = out["numfailedlogins"].gt(0).astype(int)
    out["failed_login_and_logged_in"] = (out["numfailedlogins"].gt(0) & out["loggedin"].gt(0)).astype(int)
    out["suspicious_admin_activity"] = (out["rootshell"].gt(0) | out["suattempted"].gt(0) | out["ishostlogin"].gt(0)).astype(int)
    out["privileged_activity_score"] = out[["numroot", "rootshell", "suattempted", "numshells", "numaccessfiles", "numfilecreations"]].sum(axis=1)
    out["content_risk_score"] = out[["hot", "numfailedlogins", "numcompromised", "wrongfragment", "urgent"]].sum(axis=1)
    out["file_and_shell_activity"] = out[["numfilecreations", "numshells", "numaccessfiles"]].sum(axis=1)
    out["root_compromise_ratio"] = _safe_divide(out["numroot"], out["numcompromised"] + 1)
    return out.loc[:, list(COMBINED_FEATURES)]
