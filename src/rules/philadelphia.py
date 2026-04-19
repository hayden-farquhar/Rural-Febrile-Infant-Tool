"""Philadelphia criteria for febrile infants (Baker 1993/1999).

Age: 29–60 days, febrile (≥38.2°C).
Low-risk if ALL of:
  - Well-appearing (infant observation score ≤10)
  - WBC <15,000/µL
  - Band-to-neutrophil ratio <0.2
  - UA: <10 WBC/hpf, negative Gram stain, negative leukocyte esterase
  - CSF: <8 WBC/µL, negative Gram stain (LP required)
  - CXR (if obtained): no infiltrate
  - Stool (if diarrhoea): no blood, few or no WBC on smear

Note: Philadelphia REQUIRES LP and normal CSF to classify as low-risk.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PhiladelphiaInputs:
    age_days: int
    temp_c: float
    well_appearing: bool          # infant observation score ≤10
    wbc: Optional[float] = None   # ×10³/µL
    band_neutrophil_ratio: Optional[float] = None
    ua_wbc_hpf: Optional[float] = None
    ua_gram_negative: Optional[bool] = None   # True = negative (normal)
    ua_le_negative: Optional[bool] = None     # True = negative (normal)
    csf_wbc: Optional[float] = None           # WBC/µL
    csf_gram_negative: Optional[bool] = None  # True = negative (normal)
    cxr_no_infiltrate: Optional[bool] = None  # None if CXR not done
    has_diarrhoea: bool = False
    stool_no_blood: Optional[bool] = None
    stool_few_wbc: Optional[bool] = None


@dataclass
class RuleResult:
    prediction: int
    triggered_criteria: list
    applicable: bool = True


def apply(inputs: PhiladelphiaInputs) -> RuleResult:
    """Apply Philadelphia criteria. Returns low-risk (0) or not low-risk (1)."""
    triggered = []

    # Age eligibility: 29–60 days
    if inputs.age_days < 29 or inputs.age_days > 60:
        return RuleResult(prediction=1, triggered_criteria=["age_out_of_range"], applicable=False)

    # Temperature must be ≥38.2°C to apply
    if inputs.temp_c < 38.2:
        return RuleResult(prediction=1, triggered_criteria=["temp_below_threshold"], applicable=False)

    if not inputs.well_appearing:
        triggered.append("ill_appearing")

    # WBC <15,000/µL
    if inputs.wbc is not None:
        if inputs.wbc >= 15.0:
            triggered.append("wbc_elevated")
    else:
        return RuleResult(prediction=1, triggered_criteria=["wbc_missing"], applicable=False)

    # Band:neutrophil ratio <0.2
    if inputs.band_neutrophil_ratio is not None:
        if inputs.band_neutrophil_ratio >= 0.2:
            triggered.append("band_ratio_elevated")
    else:
        return RuleResult(prediction=1, triggered_criteria=["band_ratio_missing"], applicable=False)

    # UA: <10 WBC/hpf, negative Gram, negative LE
    if inputs.ua_wbc_hpf is not None:
        if inputs.ua_wbc_hpf >= 10:
            triggered.append("ua_wbc_elevated")
    else:
        return RuleResult(prediction=1, triggered_criteria=["ua_missing"], applicable=False)

    if inputs.ua_gram_negative is False:
        triggered.append("ua_gram_positive")
    if inputs.ua_le_negative is False:
        triggered.append("ua_le_positive")

    # CSF required: <8 WBC/µL, negative Gram stain
    if inputs.csf_wbc is not None:
        if inputs.csf_wbc >= 8:
            triggered.append("csf_wbc_elevated")
    else:
        return RuleResult(prediction=1, triggered_criteria=["csf_missing_lp_required"], applicable=False)

    if inputs.csf_gram_negative is False:
        triggered.append("csf_gram_positive")

    # CXR (only if obtained)
    if inputs.cxr_no_infiltrate is False:
        triggered.append("cxr_infiltrate")

    # Stool (only if diarrhoea)
    if inputs.has_diarrhoea:
        if inputs.stool_no_blood is False:
            triggered.append("stool_blood")
        if inputs.stool_few_wbc is False:
            triggered.append("stool_wbc_elevated")

    if triggered:
        return RuleResult(prediction=1, triggered_criteria=triggered)
    return RuleResult(prediction=0, triggered_criteria=[])
