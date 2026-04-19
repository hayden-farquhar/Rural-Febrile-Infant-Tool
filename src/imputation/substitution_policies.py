"""Evidence-graded substitution policies for missing clinical inputs.

Pre-registered substitution policies (OSF registration §4.4). Evidence grades
are FROZEN at registration and must not be changed after analysis begins.

Each policy returns a SubstitutionResult with the imputed value, evidence grade,
and a human-readable rationale for the clinical output.

References:
  - FIDO 2024 (Umana et al.): CRP-for-PCT equivalence, n=1821, 67 IBI
  - NSW Health GBS guidelines: ~25% colonisation, ~70% IAP coverage
  - Briggs et al. 2012: POCT vs lab differential comparison
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EvidenceGrade(Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    DEMONSTRATION_ONLY = "demonstration-only"


@dataclass
class SubstitutionResult:
    original_input: str
    substituted_value: object
    evidence_grade: EvidenceGrade
    source: str
    rationale: str
    variance_inflation: float = 1.0  # multiplier for conformal interval width


# ── PCT → CRP substitution (Strong) ──────────────────────────

def substitute_pct_with_crp(crp_mg_l: Optional[float]) -> Optional[SubstitutionResult]:
    """Substitute PCT with CRP using FIDO 2024 equivalence.

    FIDO showed AAP CDA performs equally with CRP or PCT as biomarker.
    We use the CRP value directly at the AAP threshold (CRP ≥20 mg/L
    maps to "abnormal inflammatory marker", analogous to PCT ≥0.5 ng/mL).

    Evidence grade: STRONG
    Source: Umana et al. 2024 (FIDO), sens diff <2pp, p=1.00
    """
    if crp_mg_l is None:
        return None

    return SubstitutionResult(
        original_input="PCT",
        substituted_value=crp_mg_l,
        evidence_grade=EvidenceGrade.STRONG,
        source="Umana et al. 2024 (FIDO); n=1821, 67 IBI; "
               "AAP CDA sens 0.96 (PCT) vs 1.00 (CRP), p=1.00; "
               "spec 0.15 (PCT) vs 0.16 (CRP), p=0.69",
        rationale="CRP used in place of PCT per FIDO 2024 equivalence. "
                  "CRP ≥20 mg/L treated as equivalent to PCT ≥0.5 ng/mL "
                  "for inflammatory marker abnormality classification.",
        variance_inflation=1.1,  # minimal inflation — strong evidence
    )


def crp_exceeds_threshold(crp_mg_l: float, threshold: float = 20.0) -> bool:
    """Classify CRP as abnormal (equivalent to PCT ≥0.5 in AAP/Step-by-Step)."""
    return crp_mg_l >= threshold


# ── Maternal GBS/IAP → population prior (Moderate) ───────────

@dataclass
class GBSPrior:
    """Population-level GBS colonisation and IAP coverage priors."""
    gbs_colonisation_rate: float = 0.25   # ~25% from NSW Health
    iap_coverage_rate: float = 0.70       # ~70% IAP coverage
    source: str = "NSW Health GBS guidelines; Phares et al. 2008"


def substitute_gbs_status_unknown(
    prior: Optional[GBSPrior] = None,
) -> SubstitutionResult:
    """When maternal GBS status is unknown, use population prior.

    The probability that this infant's mother was GBS-colonised AND did NOT
    receive IAP = colonisation_rate × (1 - iap_coverage).

    Evidence grade: MODERATE
    """
    if prior is None:
        prior = GBSPrior()

    prob_gbs_no_iap = prior.gbs_colonisation_rate * (1 - prior.iap_coverage_rate)

    return SubstitutionResult(
        original_input="maternal_gbs_iap",
        substituted_value={
            "gbs_colonisation_rate": prior.gbs_colonisation_rate,
            "iap_coverage_rate": prior.iap_coverage_rate,
            "prob_gbs_positive_no_iap": prob_gbs_no_iap,
        },
        evidence_grade=EvidenceGrade.MODERATE,
        source=prior.source,
        rationale=f"Maternal GBS status unknown. Using population prior: "
                  f"~{prior.gbs_colonisation_rate:.0%} GBS colonisation, "
                  f"~{prior.iap_coverage_rate:.0%} IAP coverage → "
                  f"P(GBS+ & no IAP) = {prob_gbs_no_iap:.1%}.",
        variance_inflation=1.5,
    )


# ── UA delayed → clinical UA within 4h (Weak) ────────────────

def substitute_ua_delayed(
    ua_available: bool,
    ua_age_hours: Optional[float] = None,
    ua_le_positive: Optional[bool] = None,
    ua_nitrites_positive: Optional[bool] = None,
    ua_wbc_hpf: Optional[float] = None,
) -> Optional[SubstitutionResult]:
    """Use clinical UA if obtained within 4 hours; otherwise treat as missing.

    Evidence grade: WEAK (demonstration-only)
    No direct validation study for delayed UA interpretation.
    """
    if not ua_available:
        return None

    if ua_age_hours is not None and ua_age_hours > 4.0:
        return SubstitutionResult(
            original_input="UA",
            substituted_value=None,  # too old — treat as missing
            evidence_grade=EvidenceGrade.WEAK,
            source="Clinical convention; no direct validation study",
            rationale=f"UA specimen is {ua_age_hours:.1f}h old (>4h). "
                      "Treated as missing due to unreliability.",
            variance_inflation=2.0,
        )

    return SubstitutionResult(
        original_input="UA",
        substituted_value={
            "ua_le_positive": ua_le_positive,
            "ua_nitrites_positive": ua_nitrites_positive,
            "ua_wbc_hpf": ua_wbc_hpf,
        },
        evidence_grade=EvidenceGrade.WEAK,
        source="Clinical convention; no direct validation study",
        rationale=f"UA specimen is {ua_age_hours:.1f}h old (≤4h). "
                  "Used with caution — no validation for delayed specimens.",
        variance_inflation=1.8,
    )


# ── ANC from POCT only (Moderate) ────────────────────────────

def substitute_anc_poct(
    poct_anc: Optional[float],
) -> Optional[SubstitutionResult]:
    """Use POCT ANC reading directly; document known positive bias.

    Evidence grade: MODERATE
    POCT neutrophil counts tend to run slightly higher than lab differentials.
    """
    if poct_anc is None:
        return None

    return SubstitutionResult(
        original_input="ANC",
        substituted_value=poct_anc,
        evidence_grade=EvidenceGrade.MODERATE,
        source="Briggs et al. 2012 (POCT vs lab differential); "
               "device-specific literature",
        rationale=f"ANC from POCT ({poct_anc:.1f} ×10³/µL) used directly. "
                  "POCT neutrophil counts have known positive bias vs "
                  "laboratory differential — may slightly overestimate ANC.",
        variance_inflation=1.3,
    )


# ── PCT + CRP both missing (Demonstration-only) ──────────────

def handle_no_biomarkers() -> SubstitutionResult:
    """When both PCT and CRP are unavailable.

    No immunoassay-based inflammatory marker available.
    Rules requiring PCT or CRP cannot be applied; fall back to
    clinical features, WBC, UA only.

    Evidence grade: DEMONSTRATION-ONLY
    """
    return SubstitutionResult(
        original_input="PCT_and_CRP",
        substituted_value=None,
        evidence_grade=EvidenceGrade.DEMONSTRATION_ONLY,
        source="No direct evidence for complete biomarker absence",
        rationale="Neither PCT nor CRP available. Rules requiring "
                  "inflammatory biomarkers (Step-by-Step, AAP, PECARN) "
                  "cannot be applied. Assessment limited to clinical "
                  "appearance, age, WBC, ANC, and UA.",
        variance_inflation=3.0,  # large inflation — very uncertain
    )


# ── Scenario definitions (pre-registered) ────────────────────

SCENARIOS = {
    "baseline": {
        "description": "Full inputs — no substitution needed",
        "missing": [],
    },
    "A": {
        "description": "PCT missing (most common rural case)",
        "missing": ["PCT"],
        "substitution": "CRP-for-PCT if CRP available; otherwise no biomarker",
    },
    "B": {
        "description": "PCT + maternal GBS/IAP missing",
        "missing": ["PCT", "maternal_gbs_iap"],
        "substitution": "CRP-for-PCT + GBS population prior",
    },
    "C": {
        "description": "PCT + UA delayed/contaminated",
        "missing": ["PCT", "UA"],
        "substitution": "CRP-for-PCT + clinical UA within 4h or missing",
    },
    "D": {
        "description": "ANC from POCT only (no lab differential)",
        "missing": ["ANC_lab"],
        "substitution": "POCT ANC used directly with bias documentation",
    },
    "E": {
        "description": "PCT + CRP both missing (very small rural site)",
        "missing": ["PCT", "CRP"],
        "substitution": "No inflammatory biomarker; WBC/ANC/UA/clinical only",
    },
}


def get_scenario_substitutions(
    scenario: str,
    crp_mg_l: Optional[float] = None,
    poct_anc: Optional[float] = None,
    ua_available: bool = False,
    ua_age_hours: Optional[float] = None,
    ua_le_positive: Optional[bool] = None,
    ua_nitrites_positive: Optional[bool] = None,
    ua_wbc_hpf: Optional[float] = None,
) -> list[SubstitutionResult]:
    """Apply all substitutions for a given pre-registered scenario.

    Returns a list of SubstitutionResults documenting each substitution made.
    """
    results = []

    if scenario == "baseline":
        return results

    scenario_def = SCENARIOS.get(scenario)
    if scenario_def is None:
        raise ValueError(f"Unknown scenario: {scenario}. Must be one of: "
                         f"{list(SCENARIOS.keys())}")

    missing = scenario_def["missing"]

    if "PCT" in missing and "CRP" not in missing:
        sub = substitute_pct_with_crp(crp_mg_l)
        if sub:
            results.append(sub)
        else:
            results.append(handle_no_biomarkers())

    if "PCT" in missing and "CRP" in missing:
        results.append(handle_no_biomarkers())

    if "maternal_gbs_iap" in missing:
        results.append(substitute_gbs_status_unknown())

    if "UA" in missing:
        sub = substitute_ua_delayed(
            ua_available, ua_age_hours,
            ua_le_positive, ua_nitrites_positive, ua_wbc_hpf,
        )
        if sub:
            results.append(sub)

    if "ANC_lab" in missing:
        sub = substitute_anc_poct(poct_anc)
        if sub:
            results.append(sub)

    return results
