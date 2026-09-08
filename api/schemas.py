"""HTTP request and response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.features import BINARY_FEATURES, NONNEGATIVE_FEATURES, RATE_FEATURES


class NetworkFlow(BaseModel):
    """The exact 41 raw flow fields expected by the trained system."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    duration: float | None
    protocoltype: str | None
    service: str | None
    flag: str | None
    srcbytes: float | None
    dstbytes: float | None
    land: float | None
    wrongfragment: float | None
    urgent: float | None
    hot: float | None
    numfailedlogins: float | None
    loggedin: float | None
    numcompromised: float | None
    rootshell: float | None
    suattempted: float | None
    numroot: float | None
    numfilecreations: float | None
    numshells: float | None
    numaccessfiles: float | None
    numoutboundcmds: float | None
    ishostlogin: float | None
    isguestlogin: float | None
    count: float | None
    srvcount: float | None
    serrorrate: float | None
    srvserrorrate: float | None
    rerrorrate: float | None
    srvrerrorrate: float | None
    samesrvrate: float | None
    diffsrvrate: float | None
    srvdiffhostrate: float | None
    dsthostcount: float | None
    dsthostsrvcount: float | None
    dsthostsamesrvrate: float | None
    dsthostdiffsrvrate: float | None
    dsthostsamesrcportrate: float | None
    dsthostsrvdiffhostrate: float | None
    dsthostserrorrate: float | None
    dsthostsrvserrorrate: float | None
    dsthostrerrorrate: float | None
    dsthostsrvrerrorrate: float | None

    @model_validator(mode="after")
    def validate_domain(self) -> "NetworkFlow":
        values = self.model_dump()
        errors: list[str] = []
        for name in RATE_FEATURES:
            value = values[name]
            if value is not None and not 0 <= value <= 1:
                errors.append(f"{name} must be between 0 and 1")
        for name in NONNEGATIVE_FEATURES:
            value = values[name]
            if value is not None and value < 0:
                errors.append(f"{name} must be nonnegative")
        for name in BINARY_FEATURES:
            value = values[name]
            if value is not None and value not in (0, 1):
                errors.append(f"{name} must be 0 or 1")
        missing_fraction = sum(value is None for value in values.values()) / len(values)
        if missing_fraction > 0.25:
            errors.append("no more than 25% of input fields may be null")
        if errors:
            raise ValueError("; ".join(errors))
        return self


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[NetworkFlow] = Field(min_length=1)


class PredictionResponse(BaseModel):
    request_id: str
    prediction: dict[str, Any]


class BatchPredictionResponse(BaseModel):
    request_id: str
    record_count: int
    predictions: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    stage1_model_version: str | None = None
    stage2_model_version: str | None = None
