"""Rochester criteria for febrile infants (Dagan 1985; Jaskiewicz 1994).

Age: ≤60 days, febrile (≥38.0°C).
Low-risk if ALL of:
  - Previously healthy (term, no prior abx, no prior admission, no chronic illness)
  - Well-appearing (non-toxic)
  - No focal bacterial infection (except otitis media)
  - WBC 5,000–15,000/µL
  - Band count ≤1,500/µL
  - UA ≤10 WBC/hpf (no bacteria on Gram stain)
  - Stool (if diarrhoea): ≤5 WBC/hpf
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RochesterInputs:
    age_days: int
    temp_c: float
    well_appearing: bool
    previously_healthy: bool
    no_focal_infection: bool
    wbc: Optional[float] = None          # ×10³/µL
    band_count: Optional[float] = None   # absolute, ×10³/µL
    ua_wbc_hpf: Optional[float] = None   # WBC per high-power field
    ua_bacteria_gram: Optional[bool] = None  # True = bacteria on Gram stain
    has_diarrhoea: bool = False
    stool_wbc_hpf: Optional[float] = None


@dataclass
class RuleResult:
    prediction: int          # 0 = low-risk, 1 = not low-risk
    triggered_criteria: list  # which criteria made the infant not low-risk
    applicable: bool = True  # False if required inputs are missing


def apply(inputs: RochesterInputs) -> RuleResult:
    """Apply Rochester criteria. Returns low-risk (0) or not low-risk (1)."""
    triggered = []

    # Age eligibility
    if inputs.age_days > 60:
        return RuleResult(prediction=1, triggered_criteria=["age_out_of_range"], applicable=False)

    # Clinical criteria (required)
    if not inputs.previously_healthy:
        triggered.append("not_previously_healthy")
    if not inputs.well_appearing:
        triggered.append("ill_appearing")
    if not inputs.no_focal_infection:
        triggered.append("focal_infection")

    # WBC 5,000–15,000/µL
    if inputs.wbc is not None:
        if inputs.wbc < 5.0 or inputs.wbc > 15.0:
            triggered.append("wbc_abnormal")
    else:
        return RuleResult(prediction=1, triggered_criteria=["wbc_missing"], applicable=False)

    # Band count ≤1,500/µL (i.e., ≤1.5 ×10³/µL)
    if inputs.band_count is not None:
        if inputs.band_count > 1.5:
            triggered.append("bands_elevated")
    else:
        return RuleResult(prediction=1, triggered_criteria=["bands_missing"], applicable=False)

    # Urinalysis: ≤10 WBC/hpf, no bacteria on Gram stain
    if inputs.ua_wbc_hpf is not None:
        if inputs.ua_wbc_hpf > 10:
            triggered.append("ua_wbc_elevated")
    else:
        return RuleResult(prediction=1, triggered_criteria=["ua_missing"], applicable=False)

    if inputs.ua_bacteria_gram is True:
        triggered.append("ua_bacteria_positive")

    # Stool (only if diarrhoea present)
    if inputs.has_diarrhoea:
        if inputs.stool_wbc_hpf is not None:
            if inputs.stool_wbc_hpf > 5:
                triggered.append("stool_wbc_elevated")
        else:
            return RuleResult(prediction=1, triggered_criteria=["stool_missing_with_diarrhoea"], applicable=False)

    if triggered:
        return RuleResult(prediction=1, triggered_criteria=triggered)
    return RuleResult(prediction=0, triggered_criteria=[])
