"""AAP 2021 clinical practice guideline for febrile infants (Pantell et al. 2021, Pediatrics).

Age: 8–60 days, well-appearing, febrile (≥38.0°C), no focal bacterial infection other than UTI.

Three age strata:
  8–21 days:
    - All infants admitted + empiric parenteral antibiotics regardless of labs
    - Always "not low-risk"

  22–28 days:
    - Inflammatory markers: PCT <0.5 ng/mL, ANC <4,000/µL, CRP <20 mg/L
    - If UA negative AND all inflammatory markers normal → may manage at home
    - If any marker abnormal → LP + empiric antibiotics + admit

  29–60 days:
    - If UA negative AND all inflammatory markers normal → Low-risk
    - If any marker abnormal → LP recommended + consider antibiotics
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AAP2021Inputs:
    age_days: int
    temp_c: float
    well_appearing: bool
    ua_le_positive: Optional[bool] = None
    ua_nitrites_positive: Optional[bool] = None
    pct: Optional[float] = None    # ng/mL
    anc: Optional[float] = None    # ×10³/µL
    crp: Optional[float] = None    # mg/L


@dataclass
class RuleResult:
    prediction: int
    triggered_criteria: list
    age_stratum: str = ""       # "8-21d", "22-28d", "29-60d"
    lp_recommended: bool = False
    applicable: bool = True


def _check_inflammatory_markers(inputs: AAP2021Inputs) -> list[str]:
    """Check inflammatory markers against AAP thresholds."""
    triggered = []
    if inputs.pct is not None:
        if inputs.pct >= 0.5:
            triggered.append("pct_elevated")
    if inputs.anc is not None:
        if inputs.anc >= 4.0:
            triggered.append("anc_elevated")
    if inputs.crp is not None:
        if inputs.crp >= 20.0:
            triggered.append("crp_elevated")
    return triggered


def _markers_available(inputs: AAP2021Inputs) -> bool:
    """Check if at least one inflammatory marker is available."""
    return any(x is not None for x in [inputs.pct, inputs.anc, inputs.crp])


def apply(inputs: AAP2021Inputs) -> RuleResult:
    """Apply AAP 2021 guideline algorithm."""
    # Age eligibility: 8–60 days
    if inputs.age_days < 8 or inputs.age_days > 60:
        return RuleResult(prediction=1, triggered_criteria=["age_out_of_range"], applicable=False)

    # Ill-appearing → not low-risk regardless
    if not inputs.well_appearing:
        return RuleResult(prediction=1, triggered_criteria=["ill_appearing"],
                          age_stratum="", lp_recommended=True)

    # Determine age stratum
    if inputs.age_days <= 21:
        # 8–21 days: admit + empiric antibiotics regardless
        return RuleResult(prediction=1, triggered_criteria=["age_8_21d_always_admit"],
                          age_stratum="8-21d", lp_recommended=True)

    elif inputs.age_days <= 28:
        stratum = "22-28d"
    else:
        stratum = "29-60d"

    # Check UA
    ua_available = inputs.ua_le_positive is not None or inputs.ua_nitrites_positive is not None
    if not ua_available:
        return RuleResult(prediction=1, triggered_criteria=["ua_missing"],
                          age_stratum=stratum, applicable=False)

    ua_positive = inputs.ua_le_positive is True or inputs.ua_nitrites_positive is True

    # Check inflammatory markers
    if not _markers_available(inputs):
        return RuleResult(prediction=1, triggered_criteria=["markers_missing"],
                          age_stratum=stratum, applicable=False)

    marker_triggers = _check_inflammatory_markers(inputs)
    triggered = []

    if ua_positive:
        triggered.append("ua_positive")

    triggered.extend(marker_triggers)

    if triggered:
        lp = True  # LP recommended when any marker abnormal
        return RuleResult(prediction=1, triggered_criteria=triggered,
                          age_stratum=stratum, lp_recommended=lp)

    # All clear
    if stratum == "22-28d":
        # May manage at home but blood culture required; LP unless all markers normal
        return RuleResult(prediction=0, triggered_criteria=[],
                          age_stratum=stratum, lp_recommended=False)
    else:
        # 29–60d: low-risk, no LP required
        return RuleResult(prediction=0, triggered_criteria=[],
                          age_stratum=stratum, lp_recommended=False)
