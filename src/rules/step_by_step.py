"""Step-by-Step algorithm for febrile infants (Gomez et al. 2016, JAMA Pediatrics).

Age: 0–90 days, febrile (≥38.0°C).
Sequential algorithm (stop at first "not low-risk"):
  1. Ill-appearing? → Not low-risk
  2. Age ≤21 days? → Not low-risk
  3. Positive UA? (LE+, nitrites+, or >5 WBC/hpf) → Not low-risk
  4. PCT ≥0.5 ng/mL? → Not low-risk
  5. CRP ≥20 mg/L? → Not low-risk
  6. ANC ≥10,000/µL? → Not low-risk
  7. None of the above → Low-risk

The sequential order matters — PCT is evaluated before CRP and ANC.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StepByStepInputs:
    age_days: int
    temp_c: float
    well_appearing: bool
    ua_le_positive: Optional[bool] = None
    ua_nitrites_positive: Optional[bool] = None
    ua_wbc_hpf: Optional[float] = None
    pct: Optional[float] = None    # ng/mL
    crp: Optional[float] = None    # mg/L
    anc: Optional[float] = None    # ×10³/µL


@dataclass
class RuleResult:
    prediction: int
    triggered_criteria: list
    step_reached: int = 7      # which step the algorithm reached (1-7)
    applicable: bool = True


def apply(inputs: StepByStepInputs) -> RuleResult:
    """Apply Step-by-Step algorithm sequentially."""
    # Age eligibility
    if inputs.age_days > 90:
        return RuleResult(prediction=1, triggered_criteria=["age_out_of_range"],
                          step_reached=0, applicable=False)

    # Step 1: Ill-appearing
    if not inputs.well_appearing:
        return RuleResult(prediction=1, triggered_criteria=["ill_appearing"], step_reached=1)

    # Step 2: Age ≤21 days
    if inputs.age_days <= 21:
        return RuleResult(prediction=1, triggered_criteria=["age_leq_21d"], step_reached=2)

    # Step 3: Positive UA
    ua_positive = False
    if inputs.ua_le_positive is True:
        ua_positive = True
    if inputs.ua_nitrites_positive is True:
        ua_positive = True
    if inputs.ua_wbc_hpf is not None and inputs.ua_wbc_hpf > 5:
        ua_positive = True

    # If all UA components are None, UA is missing
    ua_available = any(x is not None for x in [
        inputs.ua_le_positive, inputs.ua_nitrites_positive, inputs.ua_wbc_hpf
    ])
    if not ua_available:
        return RuleResult(prediction=1, triggered_criteria=["ua_missing"],
                          step_reached=3, applicable=False)

    if ua_positive:
        return RuleResult(prediction=1, triggered_criteria=["ua_positive"], step_reached=3)

    # Step 4: PCT ≥0.5 ng/mL
    if inputs.pct is not None:
        if inputs.pct >= 0.5:
            return RuleResult(prediction=1, triggered_criteria=["pct_elevated"], step_reached=4)
    else:
        return RuleResult(prediction=1, triggered_criteria=["pct_missing"],
                          step_reached=4, applicable=False)

    # Step 5: CRP ≥20 mg/L
    if inputs.crp is not None:
        if inputs.crp >= 20.0:
            return RuleResult(prediction=1, triggered_criteria=["crp_elevated"], step_reached=5)
    else:
        return RuleResult(prediction=1, triggered_criteria=["crp_missing"],
                          step_reached=5, applicable=False)

    # Step 6: ANC ≥10,000/µL (i.e., ≥10.0 ×10³/µL)
    if inputs.anc is not None:
        if inputs.anc >= 10.0:
            return RuleResult(prediction=1, triggered_criteria=["anc_elevated"], step_reached=6)
    else:
        return RuleResult(prediction=1, triggered_criteria=["anc_missing"],
                          step_reached=6, applicable=False)

    # Step 7: All clear → Low-risk
    return RuleResult(prediction=0, triggered_criteria=[], step_reached=7)
