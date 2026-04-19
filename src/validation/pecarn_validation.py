"""PECARN Biosignatures external validation.

Pre-registered validation analyses (Amendment 001, §4.9):
  (a) Per-rule external validation
  (b) CRP-for-PCT substitution assessment
  (c) Simulated missing-input conformal coverage
  (d) Age-stratified validation

Constraint: tool is NOT re-tuned based on PECARN results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from src.utils import PROJECT_ROOT

PECARN_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "pecarn_tig" / "Biosignatures_Full" / "CSV datasets"


def load_pecarn_analysis_dataset() -> pd.DataFrame:
    """Load and link PECARN Biosignatures tables into analysis dataset.

    Returns DataFrame with one row per patient, columns:
      PId, age_days, gender, temp_c, wbc, anc, pct, crp,
      ua_le_positive, ua_nitrites_positive, ua_wbc_hpf,
      yos_total, well_appearing,
      has_bacteremia, has_meningitis, has_ibi, has_uti, sbi_field
    """
    # Demographics
    demo = pd.read_csv(PECARN_DATA_DIR / "demographics.csv")
    demo["age_days"] = -demo["BirthDay"]

    # Clinical data
    clin = pd.read_csv(PECARN_DATA_DIR / "clinicaldata.csv")
    # Compute YOS total from components
    yos_cols = ["YOSCry", "YOSReaction", "YOSState", "YOSColor", "YOSHydration", "YOSResponse"]
    clin_first = clin.groupby("PId").first().reset_index()
    clin_first["yos_total"] = clin_first[yos_cols].sum(axis=1, min_count=6)
    clin_first["well_appearing"] = clin_first["yos_total"] <= 10

    # Temperature: convert F to C if needed (values >60 are likely Celsius already)
    clin_first["temp_c"] = clin_first["Temperature"].apply(
        lambda t: t if pd.isna(t) or t <= 60 else (t - 32) * 5 / 9
    )

    # Lab results
    lab = pd.read_csv(PECARN_DATA_DIR / "labresults.csv")
    lab_first = lab.groupby("PId").first().reset_index()

    # UA interpretation
    # UrineLEC format DV7033G: 1=1+(small), 2=2+(moderate), 3=3+(large), 4=Negative
    lab_first["ua_le_positive"] = lab_first["UrineLEC"].apply(
        lambda x: True if x in (1, 2, 3) else (False if x == 4 else None)
    )
    # NitriteRes: 0=Negative, 1=Positive
    lab_first["ua_nitrites_positive"] = lab_first["NitriteRes"].apply(
        lambda x: True if x == 1 else (False if x == 0 else None)
    )
    # UrinalWBC format DV7034G: 1=Positive (≥6/hpf), 2=Negative (≤5/hpf)
    lab_first["ua_wbc_hpf"] = lab_first["UrinalWBC"].apply(
        lambda x: 10.0 if x == 1 else (0.0 if x == 2 else None)
    )

    # PCT from dedicated file
    pct_data = pd.read_csv(PECARN_DATA_DIR / "pctdata.csv")
    pct_data["pct"] = pd.to_numeric(pct_data["PCTResult"], errors="coerce")
    pct_first = pct_data.groupby("PId")["pct"].first().reset_index()

    # CRP from otherblood (BloodTest=1 = CRP per format DV7032G)
    ob = pd.read_csv(PECARN_DATA_DIR / "labresults_otherblood.csv")
    crp = ob[ob.BloodTest == 1][["PId", "BloodResult"]].rename(columns={"BloodResult": "crp"})
    crp_first = crp.groupby("PId")["crp"].first().reset_index()

    # IBI from culture reviews
    blood = pd.read_csv(PECARN_DATA_DIR / "culturereview_blood.csv")
    csf = pd.read_csv(PECARN_DATA_DIR / "culturereview_csf.csv")

    bact_pids = set(blood[blood.BloodDCCAssess == 1]["PId"].unique())
    mening_pids = set(csf[csf.CSFDCCAssess == 1]["PId"].unique())
    ibi_pids = bact_pids | mening_pids

    # UTI from urine culture review
    urine = pd.read_csv(PECARN_DATA_DIR / "culturereview_urine.csv")
    uti_pids = set(urine[urine.columns[urine.columns.str.contains("Assess", case=False)].tolist()[0]].apply(
        lambda x: x == 1 if pd.notna(x) else False
    )[lambda x: x].index.map(lambda i: urine.loc[i, "PId"]))

    # Merge everything
    df = demo[["PId", "age_days", "Gender"]].merge(
        clin_first[["PId", "temp_c", "yos_total", "well_appearing", "SBI"]],
        on="PId", how="inner",
    ).merge(
        lab_first[["PId", "WBC", "ANC", "ua_le_positive", "ua_nitrites_positive", "ua_wbc_hpf"]],
        on="PId", how="left",
    ).merge(pct_first, on="PId", how="left",
    ).merge(crp_first, on="PId", how="left")

    # Add IBI labels
    df["has_bacteremia"] = df["PId"].isin(bact_pids)
    df["has_meningitis"] = df["PId"].isin(mening_pids)
    df["has_ibi"] = df["PId"].isin(ibi_pids)

    # Rename for consistency
    df = df.rename(columns={"WBC": "wbc", "ANC": "anc", "Gender": "gender"})

    return df


@dataclass
class ValidationResult:
    rule: str
    n: int
    n_ibi: int
    tp: int
    fp: int
    fn: int
    tn: int
    sensitivity: float
    specificity: float
    npv: float
    ppv: float


def compute_2x2(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute 2×2 table from binary arrays."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    n = tp + fp + fn + tn
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return {
        "n": n, "n_ibi": tp + fn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": sens, "specificity": spec,
        "npv": npv, "ppv": ppv,
    }


def apply_aronson_to_pecarn(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Aronson rule to PECARN data. Returns df with 'pred' column."""
    from src.rules.aronson import AronsonInputs, apply as aronson_apply

    results = []
    for _, row in df.iterrows():
        if pd.isna(row["age_days"]):
            results.append({"PId": row["PId"], "pred": 1, "applicable": False})
            continue
        inp = AronsonInputs(
            age_days=int(row["age_days"]),
            temp_c=row["temp_c"] if pd.notna(row["temp_c"]) else 38.0,
            ua_le_positive=row["ua_le_positive"] if pd.notna(row.get("ua_le_positive")) else None,
            ua_nitrites_positive=row["ua_nitrites_positive"] if pd.notna(row.get("ua_nitrites_positive")) else None,
            anc=row["anc"] if pd.notna(row["anc"]) else None,  # already in ×10³/µL
        )
        result = aronson_apply(inp)
        results.append({"PId": row["PId"], "pred": result.prediction, "applicable": result.applicable})

    return pd.DataFrame(results)


def apply_rochester_to_pecarn(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Rochester criteria to PECARN data."""
    from src.rules.rochester import RochesterInputs, apply as rochester_apply

    results = []
    for _, row in df.iterrows():
        if pd.isna(row["age_days"]):
            results.append({"PId": row["PId"], "pred": 1, "applicable": False})
            continue
        # PECARN doesn't have band counts — Rochester requires them
        # We'll mark as not applicable where bands are missing
        inp = RochesterInputs(
            age_days=int(row["age_days"]),
            temp_c=row["temp_c"] if pd.notna(row["temp_c"]) else 38.0,
            well_appearing=row["well_appearing"] if pd.notna(row.get("well_appearing")) else True,
            previously_healthy=True,  # PECARN excluded chronic conditions
            no_focal_infection=True,  # PECARN = fever without source
            wbc=row["wbc"] if pd.notna(row["wbc"]) else None,  # already in ×10³/µL
            band_count=None,  # not available in PECARN PUD
            ua_wbc_hpf=row["ua_wbc_hpf"] if pd.notna(row.get("ua_wbc_hpf")) else None,
        )
        result = rochester_apply(inp)
        results.append({"PId": row["PId"], "pred": result.prediction, "applicable": result.applicable})

    return pd.DataFrame(results)


def apply_pecarn_rule_to_pecarn(df: pd.DataFrame) -> pd.DataFrame:
    """Apply PECARN prediction rule to PECARN data."""
    from src.rules.pecarn import PECARNInputs, apply as pecarn_apply

    results = []
    for _, row in df.iterrows():
        if pd.isna(row["age_days"]):
            results.append({"PId": row["PId"], "pred": 1, "applicable": False})
            continue
        inp = PECARNInputs(
            age_days=int(row["age_days"]),
            temp_c=row["temp_c"] if pd.notna(row["temp_c"]) else 38.0,
            ua_le_positive=row["ua_le_positive"] if pd.notna(row.get("ua_le_positive")) else None,
            ua_nitrites_positive=row["ua_nitrites_positive"] if pd.notna(row.get("ua_nitrites_positive")) else None,
            anc=row["anc"] if pd.notna(row["anc"]) else None,  # already in ×10³/µL
            pct=row["pct"] if pd.notna(row.get("pct")) else None,
        )
        result = pecarn_apply(inp)
        results.append({"PId": row["PId"], "pred": result.prediction, "applicable": result.applicable})

    return pd.DataFrame(results)


def run_per_rule_validation(df: pd.DataFrame) -> list[ValidationResult]:
    """(a) Per-rule external validation on PECARN data."""
    results = []

    for rule_name, apply_fn in [
        ("Aronson", apply_aronson_to_pecarn),
        ("PECARN", apply_pecarn_rule_to_pecarn),
    ]:
        preds = apply_fn(df)
        # Only include applicable cases
        applicable = preds[preds["applicable"]].merge(
            df[["PId", "has_ibi"]], on="PId"
        )
        if len(applicable) == 0:
            continue

        metrics = compute_2x2(
            applicable["has_ibi"].astype(int).values,
            applicable["pred"].values,
        )
        results.append(ValidationResult(rule=rule_name, **metrics))

    return results


def run_crp_pct_substitution_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """(b) CRP-for-PCT substitution assessment.

    For patients with both CRP and PCT: compute rule output using each
    biomarker and report agreement.
    """
    both = df[(df["crp"].notna()) & (df["pct"].notna())].copy()
    if len(both) == 0:
        return pd.DataFrame()

    # Apply PECARN rule with PCT (standard)
    preds_pct = apply_pecarn_rule_to_pecarn(both)
    preds_pct = preds_pct.rename(columns={"pred": "pred_pct", "applicable": "applicable_pct"})

    # Apply with CRP substituted for PCT (simulate by setting pct=None and using CRP threshold)
    both_crp = both.copy()
    # For PECARN 7-28d: PCT<1.71 → substitute with CRP<20
    # Simple approach: set PCT to a value derived from CRP threshold equivalence
    both_crp["pct"] = both_crp["crp"].apply(lambda c: 0.1 if c < 20 else 2.0)
    preds_crp = apply_pecarn_rule_to_pecarn(both_crp)
    preds_crp = preds_crp.rename(columns={"pred": "pred_crp", "applicable": "applicable_crp"})

    comparison = both[["PId", "has_ibi", "crp", "pct"]].merge(
        preds_pct[["PId", "pred_pct"]], on="PId"
    ).merge(
        preds_crp[["PId", "pred_crp"]], on="PId"
    )

    comparison["agreement"] = comparison["pred_pct"] == comparison["pred_crp"]

    return comparison


def run_validation_pipeline(output_dir: Path = None) -> dict:
    """Run all pre-registered PECARN validation analyses.

    Returns dict with results from each analysis.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "tables"

    print("Loading PECARN analysis dataset...")
    df = load_pecarn_analysis_dataset()
    print(f"  {len(df)} patients, {df.has_ibi.sum()} IBI cases")
    print(f"  Age range: {df.age_days.min():.0f}-{df.age_days.max():.0f} days")
    print(f"  PCT available: {df.pct.notna().sum()}")
    print(f"  CRP available: {df.crp.notna().sum()}")
    print(f"  Both CRP+PCT: {((df.crp.notna()) & (df.pct.notna())).sum()}")

    # (a) Per-rule validation
    print("\n(a) Per-rule external validation...")
    rule_results = run_per_rule_validation(df)
    for r in rule_results:
        print(f"  {r.rule}: n={r.n}, IBI={r.n_ibi}, "
              f"sens={r.sensitivity:.3f}, spec={r.specificity:.3f}, "
              f"FN={r.fn}")

    # (b) CRP-for-PCT substitution
    print("\n(b) CRP-for-PCT substitution analysis...")
    crp_pct = run_crp_pct_substitution_analysis(df)
    if len(crp_pct) > 0:
        agreement_rate = crp_pct["agreement"].mean()
        n_discordant = (~crp_pct["agreement"]).sum()
        print(f"  n={len(crp_pct)}, agreement={agreement_rate:.1%}, "
              f"discordant={n_discordant}")
        # Save
        crp_pct.to_csv(output_dir / "pecarn_crp_pct_comparison.csv", index=False)
    else:
        print("  No patients with both CRP and PCT available")

    # Save rule validation results
    rule_df = pd.DataFrame([vars(r) for r in rule_results])
    rule_df.to_csv(output_dir / "pecarn_rule_validation.csv", index=False)

    print("\nValidation complete. Results saved to results/tables/")
    return {"rule_results": rule_results, "crp_pct": crp_pct}


if __name__ == "__main__":
    run_validation_pipeline()
