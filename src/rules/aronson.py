"""Aronson criteria for febrile infants (Aronson et al. 2018/2019, Pediatrics).

Age: 0–60 days, febrile (≥38.0°C).
Not low-risk (IBI risk elevated) if ANY of:
  - Age ≤21 days
  - Temperature ≥38.5°C
  - Abnormal UA (positive LE or nitrites)
  - ANC ≥5,185/µL

If none of the above → Low-risk (IBI probability <1%).

Simple four-variable rule. No PCT or CRP required.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AronsonInputs:
    age_days: int
    temp_c: float
    ua_le_positive: Optional[bool] = None
    ua_nitrites_positive: Optional[bool] = None
    anc: Optional[float] = None    # ×10³/µL


@dataclass
class RuleResult:
    prediction: int
    triggered_criteria: list
    applicable: bool = True


def apply(inputs: AronsonInputs) -> RuleResult:
    """Apply Aronson criteria. Returns low-risk (0) or not low-risk (1)."""
    # Age eligibility
    if inputs.age_days > 60:
        return RuleResult(prediction=1, triggered_criteria=["age_out_of_range"], applicable=False)

    triggered = []

    # Age ≤21 days
    if inputs.age_days <= 21:
        triggered.append("age_leq_21d")

    # Temperature ≥38.5°C
    if inputs.temp_c >= 38.5:
        triggered.append("temp_geq_38.5")

    # Abnormal UA
    if inputs.ua_le_positive is not None or inputs.ua_nitrites_positive is not None:
        if inputs.ua_le_positive is True or inputs.ua_nitrites_positive is True:
            triggered.append("ua_positive")
    else:
        return RuleResult(prediction=1, triggered_criteria=["ua_missing"], applicable=False)

    # ANC ≥5,185/µL (i.e., ≥5.185 ×10³/µL)
    if inputs.anc is not None:
        if inputs.anc >= 5.185:
            triggered.append("anc_elevated")
    else:
        return RuleResult(prediction=1, triggered_criteria=["anc_missing"], applicable=False)

    if triggered:
        return RuleResult(prediction=1, triggered_criteria=triggered)
    return RuleResult(prediction=0, triggered_criteria=[])
