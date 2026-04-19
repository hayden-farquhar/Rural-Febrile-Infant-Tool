"""NICE NG143 traffic-light system for fever in under 5s (2019, updated 2021).

For infants <3 months with fever ≥38°C:
  - NICE recommends urgent referral and investigation for SBI
  - There is NO validated "low-risk" discharge pathway for <3 months
  - All febrile infants <3 months are effectively "not low-risk"

Assessment markers:
  - WBC <5,000 or >15,000/µL → amber/red
  - CRP ≥20 mg/L → amber

Traffic light features (any red → urgent):
  - Non-blanching rash
  - Weak/high-pitched/continuous cry
  - Ill-appearing / reduced responsiveness
  - Temp ≥38°C in <3 months (this alone is red)
  - Neck stiffness, bulging fontanelle, seizures

In practice, NICE does not provide a rule comparable to Rochester/PECARN.
This implementation returns "not low-risk" for all febrile infants <3 months,
with the traffic-light colour as supplementary information.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NICENG143Inputs:
    age_days: int
    temp_c: float
    well_appearing: bool
    wbc: Optional[float] = None    # ×10³/µL
    crp: Optional[float] = None    # mg/L
    non_blanching_rash: bool = False
    abnormal_cry: bool = False
    reduced_responsiveness: bool = False
    neck_stiffness: bool = False
    bulging_fontanelle: bool = False
    seizures: bool = False


@dataclass
class RuleResult:
    prediction: int
    triggered_criteria: list
    traffic_light: str = ""     # "red", "amber", "green"
    applicable: bool = True


def apply(inputs: NICENG143Inputs) -> RuleResult:
    """Apply NICE NG143. All febrile infants <3mo are 'not low-risk'."""
    if inputs.age_days > 89:
        return RuleResult(prediction=1, triggered_criteria=["age_out_of_range"], applicable=False)

    red_features = []
    amber_features = []

    # Fever ≥38°C in <3 months is itself a red feature
    if inputs.temp_c >= 38.0:
        red_features.append("fever_under_3mo")

    # Clinical red features
    if inputs.non_blanching_rash:
        red_features.append("non_blanching_rash")
    if inputs.abnormal_cry:
        red_features.append("abnormal_cry")
    if inputs.reduced_responsiveness or not inputs.well_appearing:
        red_features.append("ill_appearing")
    if inputs.neck_stiffness:
        red_features.append("neck_stiffness")
    if inputs.bulging_fontanelle:
        red_features.append("bulging_fontanelle")
    if inputs.seizures:
        red_features.append("seizures")

    # Lab markers
    if inputs.wbc is not None:
        if inputs.wbc < 5.0 or inputs.wbc > 15.0:
            amber_features.append("wbc_abnormal")
    if inputs.crp is not None:
        if inputs.crp >= 20.0:
            amber_features.append("crp_elevated")

    # Determine traffic light colour
    if red_features:
        colour = "red"
    elif amber_features:
        colour = "amber"
    else:
        colour = "amber"  # Even without specific features, fever <3mo = at minimum amber

    # NICE has no low-risk pathway for febrile <3 months
    all_triggers = red_features + amber_features
    if not all_triggers:
        all_triggers = ["fever_under_3mo_no_low_risk_pathway"]

    return RuleResult(prediction=1, triggered_criteria=all_triggers, traffic_light=colour)
