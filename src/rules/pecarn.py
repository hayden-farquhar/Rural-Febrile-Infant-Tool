"""PECARN febrile infant rule (Kuppermann et al. 2019, JAMA).

Age: 7–60 days, febrile (≥38.0°C).

Two age strata:
  7–28 days: Low-risk if ALL of:
    - UA negative (no LE, no nitrites)
    - ANC <4,090/µL
    - PCT <1.71 ng/mL
  29–60 days: Low-risk if ALL of:
    - UA negative
    - ANC <4,090/µL
    (PCT not required for 29–60d in primary rule)

Note: the original derivation excluded infants <7 days.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PECARNInputs:
    age_days: int
    temp_c: float
    ua_le_positive: Optional[bool] = None
    ua_nitrites_positive: Optional[bool] = None
    anc: Optional[float] = None    # ×10³/µL
    pct: Optional[float] = None    # ng/mL (required for 7–28d only)


@dataclass
class RuleResult:
    prediction: int
    triggered_criteria: list
    age_stratum: str = ""       # "7-28d" or "29-60d"
    applicable: bool = True


def apply(inputs: PECARNInputs) -> RuleResult:
    """Apply PECARN febrile infant rule."""
    # Age eligibility: 7–60 days
    if inputs.age_days < 7 or inputs.age_days > 60:
        return RuleResult(prediction=1, triggered_criteria=["age_out_of_range"], applicable=False)

    # Determine age stratum
    if inputs.age_days <= 28:
        stratum = "7-28d"
    else:
        stratum = "29-60d"

    triggered = []

    # UA negative (no LE, no nitrites) — required for both strata
    if inputs.ua_le_positive is not None or inputs.ua_nitrites_positive is not None:
        if inputs.ua_le_positive is True or inputs.ua_nitrites_positive is True:
            triggered.append("ua_positive")
    else:
        return RuleResult(prediction=1, triggered_criteria=["ua_missing"],
                          age_stratum=stratum, applicable=False)

    # ANC <4,090/µL (i.e., <4.09 ×10³/µL) — required for both strata
    if inputs.anc is not None:
        if inputs.anc >= 4.09:
            triggered.append("anc_elevated")
    else:
        return RuleResult(prediction=1, triggered_criteria=["anc_missing"],
                          age_stratum=stratum, applicable=False)

    # PCT <1.71 ng/mL — required for 7–28d only
    if stratum == "7-28d":
        if inputs.pct is not None:
            if inputs.pct >= 1.71:
                triggered.append("pct_elevated")
        else:
            return RuleResult(prediction=1, triggered_criteria=["pct_missing_required_7_28d"],
                              age_stratum=stratum, applicable=False)

    if triggered:
        return RuleResult(prediction=1, triggered_criteria=triggered, age_stratum=stratum)
    return RuleResult(prediction=0, triggered_criteria=[], age_stratum=stratum)
