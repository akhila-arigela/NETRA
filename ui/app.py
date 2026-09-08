"""NETRA Streamlit demonstration UI."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import streamlit as st

from src.features import (
    BINARY_FEATURES,
    NONNEGATIVE_FEATURES,
    ORIGINAL_FEATURES,
    RATE_FEATURES,
)
from ui.api_client import SentiflowApiError, SentiflowClient
from ui.styles import CSS


st.set_page_config(
    page_title="NETRA · Network Traffic Recognition & Anomaly Detection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

API_URL = os.getenv("SENTIFLOW_API_URL", "http://127.0.0.1:8000")

FIELD_GROUPS = {
    "Connection identity": ORIGINAL_FEATURES[:9],
    "Authentication and content": ORIGINAL_FEATURES[9:22],
    "Short-term behavior": ORIGINAL_FEATURES[22:31],
    "Destination-host history": ORIGINAL_FEATURES[31:],
}

METRIC_HELP = {
    "response_priority": "How urgently an analyst should review or respond to this flow based on the final hybrid decision.",
    "attack_probability": "Stage 1 model confidence that this network flow is malicious rather than normal.",
    "normal_novelty": "How unusual this flow is compared with traffic learned as normal. A higher percentile means it is less like known normal behavior.",
    "known_attack_novelty": "How unusual the flow is compared with learned examples of known attack families. A higher value suggests it may not fit a familiar attack pattern.",
    "unknown_score": "Combined evidence that the flow may represent an unfamiliar or open-set attack instead of a confidently recognized family.",
    "closest_family": "The known attack family whose learned behavior is most similar to this flow, even when the final decision requires review.",
    "family_confirmed": "Whether the flow is sufficiently consistent with the closest known attack family to accept that family classification.",
}


def _base_profile() -> dict[str, Any]:
    values = {name: 0.0 for name in ORIGINAL_FEATURES}
    values.update(
        protocoltype="tcp", service="ftp_data", flag="SF", srcbytes=491.0,
        count=2.0, srvcount=2.0, samesrvrate=1.0, dsthostcount=150.0,
        dsthostsrvcount=25.0, dsthostsamesrvrate=0.17,
        dsthostdiffsrvrate=0.03, dsthostsamesrcportrate=0.17,
        dsthostrerrorrate=0.05,
    )
    return values


NORMAL = _base_profile()
PROFILES = {
    "Normal traffic": NORMAL,
    "DoS pattern": {
        **NORMAL, "service": "private", "flag": "S0", "srcbytes": 0.0,
        "count": 123.0, "srvcount": 6.0, "serrorrate": 1.0,
        "srvserrorrate": 1.0, "samesrvrate": 0.05, "diffsrvrate": 0.07,
        "dsthostcount": 255.0, "dsthostsrvcount": 26.0,
        "dsthostsamesrvrate": 0.10, "dsthostdiffsrvrate": 0.05,
        "dsthostsamesrcportrate": 0.0, "dsthostserrorrate": 1.0,
        "dsthostsrvserrorrate": 1.0, "dsthostrerrorrate": 0.0,
    },
    "Probe pattern": {
        **NORMAL, "protocoltype": "icmp", "service": "eco_i", "srcbytes": 18.0,
        "count": 1.0, "srvcount": 1.0, "dsthostcount": 1.0,
        "dsthostsrvcount": 16.0, "dsthostsamesrvrate": 1.0,
        "dsthostdiffsrvrate": 0.0, "dsthostsamesrcportrate": 1.0,
        "dsthostsrvdiffhostrate": 1.0, "dsthostrerrorrate": 0.0,
    },
    "R2L pattern": {
        **NORMAL, "srcbytes": 334.0, "loggedin": 1.0, "dsthostcount": 2.0,
        "dsthostsrvcount": 20.0, "dsthostsamesrvrate": 1.0,
        "dsthostdiffsrvrate": 0.0, "dsthostsamesrcportrate": 1.0,
        "dsthostsrvdiffhostrate": 0.2, "dsthostrerrorrate": 0.0,
    },
    "U2R pattern": {
        **NORMAL, "duration": 98.0, "service": "telnet", "srcbytes": 621.0,
        "dstbytes": 8356.0, "urgent": 1.0, "hot": 1.0, "loggedin": 1.0,
        "numcompromised": 5.0, "rootshell": 1.0, "numroot": 14.0,
        "numfilecreations": 1.0, "count": 1.0, "srvcount": 1.0,
        "dsthostcount": 255.0, "dsthostsrvcount": 4.0,
        "dsthostsamesrvrate": 0.02, "dsthostdiffsrvrate": 0.02,
        "dsthostsamesrcportrate": 0.0, "dsthostrerrorrate": 0.0,
    },
}


@st.cache_resource
def api_client(base_url: str) -> SentiflowClient:
    return SentiflowClient(base_url)


def set_profile(profile: dict[str, Any]) -> None:
    for name, value in profile.items():
        st.session_state[f"flow_{name}"] = value


def ensure_form_state() -> None:
    if "flow_duration" not in st.session_state:
        set_profile(NORMAL)


def friendly_decision(decision: str) -> tuple[str, str]:
    if "NORMAL" in decision:
        return "NORMAL", "normal"
    if decision.startswith("KNOWN_ATTACK_"):
        return f"ATTACK · {decision.removeprefix('KNOWN_ATTACK_')}", "attack"
    if "UNKNOWN" in decision:
        return "UNKNOWN ATTACK CANDIDATE", "unknown"
    return decision.replace("_", " "), "review"


def pct(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2%}"


def render_sidebar(client: SentiflowClient) -> None:
    with st.sidebar:
        st.markdown('<div class="sf-brand">NET<span>RA</span></div>', unsafe_allow_html=True)
        st.caption("Network Traffic Recognition & Anomaly Detection")
        try:
            health = client.health()
            st.markdown('<div class="status"><i></i> MODELS ONLINE</div>', unsafe_allow_html=True)
            st.caption(f"{health['stage1_model_version']}  ·  {health['stage2_model_version']}")
        except (SentiflowApiError, OSError) as exc:
            st.error("Inference API offline")
            st.caption(str(exc))
        st.divider()
        st.markdown("**Decision hierarchy**")
        st.caption("1. Final decision\n\n2. Decision reason\n\n3. Review priority\n\n4. Supporting scores")
        st.divider()
        st.caption(f"API: {API_URL}")


def render_overview() -> None:
    st.markdown(
        """<div class="hero"><div class="eyebrow">TWO-STAGE HYBRID DEFENSE</div>
        <h1>See the threat.<br><span>Understand the signal.</span></h1>
        <p>NETRA combines supervised detection with open-set novelty intelligence—identifying known intrusions without forcing unfamiliar behavior into the wrong category.</p></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="objective"><strong>MODEL OBJECTIVE</strong><br><br>
        Detect whether a network flow is normal or malicious, identify trusted attacks as
        DoS, Probe, R2L, or U2R, and explicitly escalate novel or conflicting behavior for
        investigation instead of producing a falsely confident label.</div>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">EXTERNAL TEST SNAPSHOT</div>', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    for column, value, label in zip(
        metric_cols,
        ("99.97%", "99.92%", "99.53%", "96.67%"),
        ("Conservative attack recall", "Automatic decision accuracy", "Automatic coverage", "Known-attack acceptance"),
    ):
        column.markdown(f'<div class="metric-card"><b>{value}</b><span>{label}</span></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading"><span>01</span><div><h3>About the hybrid model</h3>'
        '<p>Two complementary layers balance recognition, novelty detection, and safe escalation.</p></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="hybrid-summary">
          <p><strong>Why hybrid?</strong> NETRA pairs supervised models that recognize learned threats
          with novelty models that identify traffic outside familiar normal and attack behavior.
          This provides strong known-attack recognition without forcing every uncertain flow into a known class.</p>
        </div>
        <div class="model-stack">
          <div class="model-card stage-one-card">
            <span>STAGE 1 &middot; DETECTION & ROUTING</span>
            <h4>Find suspicious traffic</h4>
            <div class="model-row"><b>Calibrated XGBoost</b><small>Attack probability</small></div>
            <div class="model-row"><b>Isolation Forest</b><small>Normal-space novelty</small></div>
            <p>Normal flows finish here. Suspicious, novel, or conflicting flows continue for deeper analysis.</p>
          </div>
          <div class="model-card stage-two-card">
            <span>STAGE 2 &middot; THREAT RECOGNITION</span>
            <h4>Identify or challenge the threat</h4>
            <div class="model-row"><b>Random Forest</b><small>DoS, Probe, R2L or U2R</small></div>
            <div class="model-row"><b>One-Class SVM + family models</b><small>Global and family novelty</small></div>
            <p>Accept a trusted family, flag an unknown candidate, or request analyst review.</p>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading"><span>02</span><div><h3>Operational impact</h3>'
        '<p>What NETRA can contribute to a network-security workflow.</p></div></div>',
        unsafe_allow_html=True,
    )
    cards = st.columns(4)
    impact = (
        ("01", "Earlier warning", "Surface behavior outside learned normal and known-attack patterns."),
        ("02", "Fewer blind spots", "Pair supervised accuracy with unsupervised anomaly awareness."),
        ("03", "Safer decisions", "Send uncertainty to review instead of forcing the wrong category."),
        ("04", "Faster triage", "Give analysts probabilities, novelty evidence, priority, and reasons."),
    )
    for column, (icon, title, copy) in zip(cards, impact):
        column.markdown(f'<div class="impact"><b>{icon}</b><h3>{title}</h3><p>{copy}</p></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading"><span>03</span><div><h3>Decision flow</h3>'
        '<p>How one incoming network flow becomes an actionable outcome.</p></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="decision-flow">
          <div class="flow-track">
            <div class="flow-node input-node">
              <span class="flow-number">01</span>
              <span class="flow-kicker">INPUT</span>
              <h4>Network flow</h4>
              <p>41 connection, content and traffic-history fields</p>
            </div>
            <div class="flow-arrow"><span>&rarr;</span></div>
            <div class="flow-node stage-one-node">
              <span class="flow-number">02</span>
              <span class="flow-kicker">STAGE 1 &middot; DETECT</span>
              <h4>Is this flow suspicious?</h4>
              <p>Attack probability + distance from learned normal behavior</p>
              <div class="flow-signals"><span>Risk</span><span>Normal novelty</span></div>
            </div>
            <div class="flow-arrow"><span>&rarr;</span></div>
            <div class="flow-node stage-two-node">
              <span class="flow-number">03</span>
              <span class="flow-kicker">STAGE 2 &middot; RECOGNIZE</span>
              <h4>What kind of threat is it?</h4>
              <p>Attack-family classification + open-set novelty checks</p>
              <div class="flow-signals"><span>DoS</span><span>Probe</span><span>R2L</span><span>U2R</span></div>
            </div>
          </div>
          <div class="flow-routing">
            <div class="route-path normal-path">
              <span class="route-source">STAGE 1: NORMAL</span>
              <span class="route-line"></span>
              <strong>Normal traffic</strong>
              <small>Finish without displaying Stage 2 evidence</small>
            </div>
            <div class="route-path investigate-path">
              <span class="route-source">STAGE 1: SUSPICIOUS</span>
              <span class="route-line"></span>
              <div class="outcome-group">
                <strong>Known attack</strong>
                <strong>Unknown candidate</strong>
                <strong>Needs review</strong>
              </div>
              <small>Stage 2 evidence determines identity and analyst action</small>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading"><span>04</span><div><h3>Held-out test performance</h3>'
        '<p>Measured once on traffic excluded from model fitting and threshold selection.</p></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="test-context">
          <div><b>25,195</b><span>Total network flows</span></div>
          <div><b>13,469</b><span>Normal flows</span></div>
          <div><b>11,726</b><span>Known attacks</span></div>
          <div><b>1.91%</b><span>Final review rate</span></div>
        </div>""",
        unsafe_allow_html=True,
    )
    detail_left, detail_right = st.columns(2)
    detail_left.markdown(
        """<div class="evaluation-card"><span>STAGE 1</span><h4>High-recall traffic routing</h4>
        <p><b>99.93%</b> of known attacks reached Stage 2, while <b>99.03%</b> of normal
        traffic was automatically returned as Normal.</p></div>""",
        unsafe_allow_html=True,
    )
    detail_right.markdown(
        """<div class="evaluation-card"><span>STAGE 2</span><h4>Conservative family recognition</h4>
        <p><b>100%</b> family accuracy among accepted predictions; uncertain cases were
        withheld from automatic classification and sent for review.</p></div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div class="family-panel">
          <div class="family-panel-head"><div><span>FAMILY PERFORMANCE</span><h4>Correctly accepted known attacks</h4></div><small>RATE&nbsp;&nbsp;/&nbsp;&nbsp;TEST SUPPORT</small></div>
          <div class="family-row">
            <div class="family-name"><b>DoS</b><small>9,105 test flows</small></div>
            <div class="family-meter"><i style="width:97.56%"></i></div><strong>97.56%</strong>
          </div>
          <div class="family-row">
            <div class="family-name"><b>Probe</b><small>2,389 test flows</small></div>
            <div class="family-meter"><i style="width:94.85%"></i></div><strong>94.85%</strong>
          </div>
          <div class="family-row">
            <div class="family-name"><b>R2L</b><small>226 test flows</small></div>
            <div class="family-meter"><i style="width:82.30%"></i></div><strong>82.30%</strong>
          </div>
          <div class="family-row low-support">
            <div class="family-name"><b>U2R</b><small>6 test flows &middot; low support</small></div>
            <div class="family-meter"><i style="width:0%"></i></div><strong>0.00%</strong>
          </div>
          <p class="family-note">U2R cases were sent for review instead of receiving an unsupported automatic family label.</p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.caption(
        "Evaluation context: results use 25,195 held-out flows containing Normal traffic and four known "
        "attack families. Unknown-attack detection should also be assessed with genuinely unseen families."
    )


def input_widget(name: str) -> Any:
    label = name.replace("dsthost", "dst host ").replace("srv", " srv ").replace("rate", " rate").replace("num", "num ")
    if name in {"protocoltype", "service", "flag"}:
        return st.text_input(label, key=f"flow_{name}")
    if name in BINARY_FEATURES:
        return st.selectbox(label, (0.0, 1.0), key=f"flow_{name}")
    options: dict[str, Any] = {"step": 0.01 if name in RATE_FEATURES else 1.0, "format": "%.4f" if name in RATE_FEATURES else "%.2f"}
    if name in RATE_FEATURES:
        options.update(min_value=0.0, max_value=1.0)
    elif name in NONNEGATIVE_FEATURES:
        options.update(min_value=0.0)
    return st.number_input(label, key=f"flow_{name}", **options)


def render_prediction(result: dict[str, Any]) -> None:
    prediction = result["prediction"]
    label, decision_class = friendly_decision(prediction["final_decision"])
    is_normal = "NORMAL" in prediction["final_decision"]
    st.markdown(
        f'<div class="decision {decision_class}"><small>FINAL HYBRID DECISION</small><h2>{label}</h2></div>',
        unsafe_allow_html=True,
    )
    if is_normal:
        st.write(prediction.get("stage1_route_reason", "Stage 1 classified this flow as normal."))
    else:
        st.write(prediction["stage2_reason"])
    st.metric(
        "Response priority",
        str(prediction["review_priority"]).upper(),
        help=METRIC_HELP["response_priority"],
    )

    st.markdown("#### Stage 1 · Detection")
    left, right = st.columns(2)
    left.metric(
        "Attack probability",
        pct(prediction["xgb_attack_probability"]),
        help=METRIC_HELP["attack_probability"],
    )
    left.progress(float(prediction["xgb_attack_probability"]))
    right.metric(
        "Normal-space novelty",
        pct(prediction["normal_novelty_percentile"]),
        help=METRIC_HELP["normal_novelty"],
    )
    right.progress(float(prediction["normal_novelty_percentile"]))
    st.markdown(
        f'<div class="route"><small>ROUTING RESULT</small><b>{prediction["stage1_route"]}</b>{prediction.get("stage1_route_reason", "")}</div>',
        unsafe_allow_html=True,
    )

    if not is_normal:
        st.markdown("#### Stage 2 · Threat identity")
        probabilities = pd.DataFrame(
            {
                "Attack family": ["DoS", "Probe", "R2L", "U2R"],
                "Probability": [prediction.get(f"category_probability_{name}", 0.0) for name in ("DoS", "Probe", "R2L", "U2R")],
            }
        ).set_index("Attack family")
        st.bar_chart(probabilities, horizontal=True, height=230, color="#ad8aff")
        evidence = st.columns(4)
        evidence[0].metric(
            "Known-attack novelty",
            pct(prediction.get("known_attack_novelty_percentile")),
            help=METRIC_HELP["known_attack_novelty"],
        )
        evidence[1].metric(
            "Unknown score",
            pct(prediction.get("unknown_score")),
            help=METRIC_HELP["unknown_score"],
        )
        evidence[2].metric(
            "Closest family",
            prediction.get("closest_known_category", "—"),
            help=METRIC_HELP["closest_family"],
        )
        evidence[3].metric(
            "Family confirmed",
            "Yes" if prediction.get("category_novelty_is_novel") is False else "No",
            help=METRIC_HELP["family_confirmed"],
        )
        with st.expander("Technical output"):
            st.json(prediction)


def render_single(client: SentiflowClient) -> None:
    st.markdown('<div class="eyebrow">SINGLE-FLOW ANALYSIS</div>', unsafe_allow_html=True)
    st.title("Inspect one network flow")
    st.caption("Use a real demonstration profile or enter all 41 original flow features.")
    profile_col, button_col, _ = st.columns([2, 1, 4])
    profile = profile_col.selectbox("Demo profile", tuple(PROFILES))
    if button_col.button("Load profile", use_container_width=True):
        set_profile(PROFILES[profile])
        st.rerun()

    with st.form("single_flow_form"):
        for group, names in FIELD_GROUPS.items():
            with st.expander(group, expanded=group == "Connection identity"):
                columns = st.columns(3)
                for index, name in enumerate(names):
                    with columns[index % 3]:
                        input_widget(name)
        submitted = st.form_submit_button("Run hybrid analysis", use_container_width=True)
    if submitted:
        record = {name: st.session_state[f"flow_{name}"] for name in ORIGINAL_FEATURES}
        try:
            with st.spinner("Running Stage 1 and Stage 2…"):
                st.session_state["last_prediction"] = client.predict_one(record)
        except (SentiflowApiError, OSError) as exc:
            st.error(str(exc))
    if "last_prediction" in st.session_state:
        st.divider()
        render_prediction(st.session_state["last_prediction"])


def render_batch(client: SentiflowClient) -> None:
    st.markdown('<div class="eyebrow">BATCH INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title("Build an investigation queue")
    st.caption("Upload a CSV containing one flow per row and the 41 original feature columns.")
    upload = st.file_uploader("Network-flow CSV", type=["csv"])
    if upload is None:
        st.markdown('<div class="notice">The API accepts up to 5,000 records and 10 MB per request. Labels and engineered features are not required.</div>', unsafe_allow_html=True)
        return
    content = upload.getvalue()
    try:
        preview = pd.read_csv(pd.io.common.BytesIO(content))
        st.caption(f"{upload.name} · {len(preview):,} flows · {len(content) / 1024:.1f} KB")
        missing = sorted(set(ORIGINAL_FEATURES) - set(preview.columns))
        if missing:
            st.error(f"Missing required columns: {missing}")
            return
        st.dataframe(preview.head(10), use_container_width=True, height=260)
    except Exception as exc:
        st.error(f"Cannot read CSV: {exc}")
        return
    if st.button("Analyze batch", type="primary", use_container_width=True):
        try:
            with st.spinner("Scoring batch through both stages…"):
                st.session_state["batch_result"] = client.predict_csv(content)
                st.session_state["batch_name"] = upload.name
        except (SentiflowApiError, OSError) as exc:
            st.error(str(exc))
    if "batch_result" not in st.session_state:
        return
    predictions = pd.DataFrame(st.session_state["batch_result"]["predictions"])
    decision = predictions["final_decision"].astype(str)
    counts = (
        int(decision.str.contains("NORMAL").sum()),
        int(decision.str.startswith("KNOWN_ATTACK").sum()),
        int(decision.str.contains("UNKNOWN").sum()),
        int(decision.str.contains("REVIEW|DATA_QUALITY").sum()),
    )
    for column, value, label in zip(st.columns(4), counts, ("Normal", "Known attacks", "Unknown candidates", "Review / exceptions")):
        column.metric(label, value)
    display_columns = ["final_decision", "xgb_attack_probability", "closest_known_category", "review_priority", "stage1_route", "stage2_reason"]
    st.dataframe(predictions[display_columns], use_container_width=True, height=430)
    st.download_button(
        "Download complete predictions",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name="netra_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


ensure_form_state()
client = api_client(API_URL)
render_sidebar(client)
overview_tab, single_tab, batch_tab = st.tabs(["Overview & impact", "Single-flow analysis", "Batch prediction"])
with overview_tab:
    render_overview()
with single_tab:
    render_single(client)
with batch_tab:
    render_batch(client)
st.markdown('<div class="footer">NETRA · Network Traffic Recognition & Anomaly Detection · Research demonstration</div>', unsafe_allow_html=True)
