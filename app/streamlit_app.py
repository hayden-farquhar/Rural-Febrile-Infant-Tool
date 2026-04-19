"""Rural ED Febrile Infant Decision Support Tool.

Streamlit application that provides uncertainty-aware IBI probability
estimates for febrile infants <90 days when clinical inputs are incomplete.

DISCLAIMER: Research tool only. Not for clinical use without prospective
local validation.
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import IBI_TRANSFER_THRESHOLD, AGE_TERTILE_LABELS
from src.imputation.substitution_policies import (
    EvidenceGrade,
    SCENARIOS,
    get_scenario_substitutions,
    crp_exceeds_threshold,
)
from src.conformal.mondrian_missing_input import (
    MondrianConformalPredictor,
    assign_age_tertile,
    assign_completeness,
    simulate_calibration_cases,
)
from src.probability import posterior_ibi_multi_rule, RULE_PERFORMANCE
from src.prediction_model import FebrileInfantPredictor, load_and_prepare_pecarn

# ── Page config ───────────────────────────────────────────────

st.set_page_config(
    page_title="Febrile Infant Rural Decision Support",
    page_icon="🏥",
    layout="wide",
)

# ── Disclaimer banner ─────────────────────────────────────────

st.error(
    "**RESEARCH TOOL ONLY** — Not for clinical use without prospective "
    "local validation. This tool does not replace clinical judgement. "
    "Always follow local guidelines and consult senior clinicians."
)

st.title("Rural ED Febrile Infant Decision Support")
st.caption(
    "Uncertainty-aware IBI probability for febrile infants <90 days "
    "with incomplete clinical inputs. "
    "[OSF Pre-registration](https://osf.io/dq5n8/)"
)

# ── Sidebar: Clinical inputs ─────────────────────────────────

st.sidebar.header("Patient Details")

age_days = st.sidebar.number_input(
    "Age (days)", min_value=0, max_value=89, value=30, step=1
)
temp_c = st.sidebar.number_input(
    "Temperature (°C)", min_value=36.0, max_value=42.0, value=38.5, step=0.1,
    format="%.1f"
)
appearance = st.sidebar.selectbox(
    "Clinical appearance", ["Well-appearing", "Unwell / ill-appearing"]
)
well_appearing = appearance == "Well-appearing"

yos_total = st.sidebar.number_input(
    "Yale Observation Scale (6-30)", min_value=6, max_value=30, value=6, step=2,
    help="Sum of 6 components (cry, reaction, state, colour, hydration, response). "
         "Each scored 1-3-5. Total 6 = well, ≤10 = normal, >10 = concerning."
)

st.sidebar.header("Laboratory Results")
st.sidebar.caption("Leave unchecked if not available")

has_wbc = st.sidebar.checkbox("WBC available")
wbc = st.sidebar.number_input(
    "WBC (×10³/µL)", min_value=0.0, max_value=50.0, value=10.0, step=0.5,
    disabled=not has_wbc
) if has_wbc else None

has_anc = st.sidebar.checkbox("ANC available")
anc = st.sidebar.number_input(
    "ANC (×10³/µL)", min_value=0.0, max_value=40.0, value=4.0, step=0.5,
    disabled=not has_anc
) if has_anc else None

has_crp = st.sidebar.checkbox("CRP available")
crp = st.sidebar.number_input(
    "CRP (mg/L)", min_value=0.0, max_value=500.0, value=5.0, step=1.0,
    disabled=not has_crp
) if has_crp else None

has_pct = st.sidebar.checkbox("PCT available")
pct = st.sidebar.number_input(
    "PCT (ng/mL)", min_value=0.0, max_value=200.0, value=0.3, step=0.1,
    format="%.2f", disabled=not has_pct
) if has_pct else None

has_ua = st.sidebar.checkbox("Urinalysis available")
if has_ua:
    ua_le = st.sidebar.selectbox("Leukocyte esterase", ["Negative", "Positive"])
    ua_nitrites = st.sidebar.selectbox("Nitrites", ["Negative", "Positive"])
    ua_le_positive = ua_le == "Positive"
    ua_nitrites_positive = ua_nitrites == "Positive"
else:
    ua_le_positive = None
    ua_nitrites_positive = None

st.sidebar.header("Maternal History")
gbs_status = st.sidebar.selectbox(
    "Maternal GBS status", ["Unknown", "Positive", "Negative"]
)
iap_given = st.sidebar.selectbox(
    "Intrapartum antibiotics", ["Unknown", "Yes", "No"]
)


# ── Determine missing inputs and scenario ─────────────────────

def determine_scenario():
    """Map available inputs to the closest pre-registered scenario."""
    missing = []
    if not has_pct:
        missing.append("PCT")
    if not has_crp:
        missing.append("CRP")
    if not has_ua:
        missing.append("UA")
    if gbs_status == "Unknown":
        missing.append("GBS/IAP")
    if not has_anc:
        missing.append("ANC")

    # Map to pre-registered scenarios
    if not missing:
        return "baseline", missing
    if missing == ["PCT"] and has_crp:
        return "A", missing
    if set(missing) == {"PCT", "GBS/IAP"} and has_crp:
        return "B", missing
    if "PCT" in missing and "UA" in missing:
        return "C", missing
    if "ANC" in missing and "PCT" not in missing:
        return "D", missing
    if "PCT" in missing and "CRP" in missing:
        return "E", missing
    return "A", missing  # default to most common


scenario, missing_inputs = determine_scenario()

# ── Apply decision rules ──────────────────────────────────────

def apply_rules():
    """Apply each decision rule and return results."""
    results = {}

    # Aronson (no PCT/CRP needed)
    if has_anc and has_ua:
        from src.rules.aronson import AronsonInputs, apply as aronson_apply
        r = aronson_apply(AronsonInputs(
            age_days=age_days, temp_c=temp_c,
            ua_le_positive=ua_le_positive,
            ua_nitrites_positive=ua_nitrites_positive,
            anc=anc,
        ))
        if r.applicable:
            results["Aronson"] = r

    # Rochester (needs WBC, band count — bands usually unavailable)
    if has_wbc and has_ua:
        from src.rules.rochester import RochesterInputs, apply as rochester_apply
        r = rochester_apply(RochesterInputs(
            age_days=age_days, temp_c=temp_c,
            well_appearing=well_appearing,
            previously_healthy=True, no_focal_infection=True,
            wbc=wbc, band_count=None,  # bands rarely available
            ua_wbc_hpf=None,
        ))
        results["Rochester"] = r  # will be not-applicable due to missing bands

    # Step-by-Step (needs PCT, CRP, ANC, UA)
    if has_ua:
        from src.rules.step_by_step import StepByStepInputs, apply as sbs_apply
        r = sbs_apply(StepByStepInputs(
            age_days=age_days, temp_c=temp_c,
            well_appearing=well_appearing,
            ua_le_positive=ua_le_positive,
            ua_nitrites_positive=ua_nitrites_positive,
            pct=pct, crp=crp, anc=anc,
        ))
        results["Step-by-Step"] = r

    # PECARN (needs UA, ANC, PCT for 7-28d)
    if has_ua and has_anc:
        from src.rules.pecarn import PECARNInputs, apply as pecarn_apply
        r = pecarn_apply(PECARNInputs(
            age_days=age_days, temp_c=temp_c,
            ua_le_positive=ua_le_positive,
            ua_nitrites_positive=ua_nitrites_positive,
            anc=anc, pct=pct,
        ))
        results["PECARN"] = r

    # AAP 2021
    if has_ua:
        from src.rules.aap_2021 import AAP2021Inputs, apply as aap_apply
        r = aap_apply(AAP2021Inputs(
            age_days=age_days, temp_c=temp_c,
            well_appearing=well_appearing,
            ua_le_positive=ua_le_positive,
            ua_nitrites_positive=ua_nitrites_positive,
            pct=pct, anc=anc, crp=crp,
        ))
        results["AAP 2021"] = r

    return results


# ── Calibrate conformal predictor (cached) ────────────────────

@st.cache_resource
def get_prediction_model():
    """Train prediction model once and cache."""
    from pathlib import Path
    model_path = PROJECT_ROOT / "data" / "interim" / "prediction_model_v6.joblib"
    predictor = FebrileInfantPredictor()
    if model_path.exists():
        predictor.load(model_path)
    else:
        df = load_and_prepare_pecarn()
        predictor.fit(df)
        predictor.save(model_path)
    return predictor


# ── Main display ──────────────────────────────────────────────

col1, col2 = st.columns([2, 1])

with col1:
    # Missing inputs banner
    if missing_inputs:
        missing_str = ", ".join(missing_inputs)
        st.warning(
            f"**Missing inputs:** {missing_str}  \n"
            f"**Scenario:** {scenario} — {SCENARIOS[scenario]['description']}  \n"
            f"Uncertainty intervals widened to reflect missing data."
        )
    else:
        st.success("**All inputs available** — full assessment possible.")

    # Get substitutions
    substitutions = get_scenario_substitutions(
        scenario,
        crp_mg_l=crp,
        ua_available=has_ua,
    )

    # Show substitution details
    if substitutions:
        with st.expander("Substitution policies applied", expanded=False):
            for sub in substitutions:
                grade_color = {
                    EvidenceGrade.STRONG: "🟢",
                    EvidenceGrade.MODERATE: "🟡",
                    EvidenceGrade.WEAK: "🟠",
                    EvidenceGrade.DEMONSTRATION_ONLY: "🔴",
                }[sub.evidence_grade]
                st.markdown(
                    f"{grade_color} **{sub.original_input}** → "
                    f"*{sub.evidence_grade.value}*  \n"
                    f"{sub.rationale}  \n"
                    f"*Source: {sub.source}*"
                )

    # Apply rules (for per-rule breakdown display)
    rule_results = apply_rules()

    # Get prediction model
    model = get_prediction_model()

    # Predict P(IBI) from continuous lab values
    pred = model.predict(
        age_days=age_days,
        temp_c=temp_c,
        wbc=wbc if has_wbc else None,
        anc=anc if has_anc else None,
        ua_positive=ua_le_positive if has_ua else None,
        yos_total=float(yos_total),
        pct=pct if has_pct else None,
    )

    ibi_point = pred.probability

    # Map risk tier to display
    tier_map = {
        "very_low": ("Very low risk", "green"),
        "low": ("Low risk — comparable to published decision rules", "blue"),
        "moderate": ("Moderate risk — above published low-risk thresholds", "orange"),
        "high": ("High risk — IBI workup recommended", "red"),
    }
    decision_text, decision_color = tier_map.get(pred.risk_tier, ("Assess further", "orange"))

    # Display decision
    st.markdown("---")
    st.subheader("Assessment")

    if decision_color == "green":
        st.success(f"### {decision_text}")
    elif decision_color == "blue":
        st.info(f"### {decision_text}")
    elif decision_color == "red":
        st.error(f"### {decision_text}")
    else:
        st.warning(f"### {decision_text}")

    # Probability display
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.metric("IBI Probability", f"{ibi_point:.2%}")
    with mcol2:
        st.metric("Risk", f"1 in {pred.one_in_n}")
    with mcol3:
        st.metric("Missing Inputs Imputed", f"{pred.n_missing_imputed}")

    # Risk description and comparison to published rules
    st.markdown(f"**{pred.risk_description}**")
    st.markdown(f"*{pred.comparison_to_rules}*")

    with st.expander("Age-specific context", expanded=True):
        st.markdown(pred.age_context)

    if pred.calibration_note:
        with st.expander("Calibration data", expanded=False):
            st.caption(pred.calibration_note)

with col2:
    st.subheader("Per-Rule Breakdown")

    if not rule_results:
        st.info("No rules applicable with current inputs.")
    else:
        for name, result in rule_results.items():
            if not result.applicable:
                st.markdown(f"**{name}:** ⚪ Not applicable (missing inputs)")
            elif result.prediction == 0:
                st.markdown(f"**{name}:** 🟢 Low-risk")
            else:
                triggers = ", ".join(result.triggered_criteria)
                st.markdown(f"**{name}:** 🔴 Not low-risk ({triggers})")

    # Age tertile info
    st.markdown("---")
    st.caption(
        f"**Age tertile:** {assign_age_tertile(age_days)}  \n"
        f"**Completeness:** {assign_completeness(len(missing_inputs))}  \n"
        f"**Mondrian stratum:** {assign_age_tertile(age_days)}_{assign_completeness(len(missing_inputs))}"
    )

# ── Footer ────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "*Rural-Calibrated Febrile-Infant Decision Support Tool with "
    "Missing-Input Conformal Imputation. "
    "[Pre-registration](https://osf.io/dq5n8/) | "
    "PRISMA-DTA + TRIPOD+AI*"
)
st.caption(
    "Research tool only. Not validated for clinical use. "
    "Prospective validation via PREDICT network invited."
)
