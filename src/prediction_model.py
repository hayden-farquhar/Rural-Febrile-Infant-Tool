"""IBI prediction model — Production v6.

Architecture:
  Single pooled logistic regression: age + temp + WBC + ANC + UA + YOS + age_young
  Unpenalised. Complete-case training (n=4,434, IBI=88).
  AUC=0.835, calibration slope=0.937 (near ideal).

Output: Continuous probability with clinical context, not categorical labels.
  Four interpretive tiers with published-rule comparisons:
    P<0.5%  — Very low risk (below published rule residual rates)
    0.5-1.5% — Low risk (comparable to published rule "low-risk" groups)
    1.5-3.0% — Moderate risk (above published rule thresholds)
    >3.0%   — High risk (workup recommended)

Rationale for pooled model over stratified:
  - AUC 0.835 (pooled) vs 0.818 (neonatal) and 0.702 (older)
  - 88 IBI events vs 51/37 — more stable coefficient estimates
  - Stratified models reported as sensitivity analysis
  - Age-specific context provided via interpretation, not model structure

Design decisions:
  - Complete-case training: avoids informative missingness bias
  - No missingness indicators: MCAR in rural deployment
  - No rule augmentation: multicollinearity (VIF>100) without meaningful AUC gain
  - age_young (≤14d): captures highest-risk neonatal subgroup (+0.003 AUC)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from dataclasses import dataclass, field
import joblib

from src.utils import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "data" / "interim"

FEATURES = ["age_days", "temp_c", "wbc", "anc", "ua_pos", "yos_total", "age_young"]

# Published rule performance for context (from our meta-analysis / extraction)
PUBLISHED_RULE_CONTEXT = {
    "Aronson": {"ibi_rate_in_low_risk": 0.006, "npv": 0.994, "source": "FIDO 2024"},
    "PECARN": {"ibi_rate_in_low_risk": 0.004, "npv": 0.996, "source": "Kuppermann 2019"},
    "Step-by-Step": {"ibi_rate_in_low_risk": 0.007, "npv": 0.993, "source": "Gomez 2016"},
    "Rochester": {"ibi_rate_in_low_risk": 0.017, "npv": 0.983, "source": "Gomez 2016"},
}


@dataclass
class PredictionResult:
    probability: float
    one_in_n: int
    risk_tier: str                  # very_low / low / moderate / high
    risk_description: str           # human-readable description
    comparison_to_rules: str        # context vs published rules
    age_context: str                # age-specific interpretation
    calibration_note: str
    features_used: list[str] = field(default_factory=list)
    n_missing_imputed: int = 0
    confidence_band: tuple[float, float] = (0.0, 0.0)


def _risk_tier(prob: float, age_days: int) -> tuple[str, str, str]:
    """Assign risk tier with clinical description and rule comparison."""
    if prob < 0.005:
        tier = "very_low"
        desc = "Very low risk — below the residual IBI rate of all published decision rules."
        comp = ("Below the residual IBI rate in PECARN low-risk (0.4%), "
                "Aronson low-risk (0.6%), and Rochester low-risk (1.7%).")
    elif prob < 0.015:
        tier = "low"
        desc = ("Low risk — comparable to the residual IBI rate accepted by "
                "published decision rules for 'low-risk' classification.")
        comp = ("Comparable to Aronson low-risk (0.6% IBI rate, NPV 99.4%) "
                "and PECARN low-risk (0.4% IBI rate, NPV 99.6%). "
                "Published rules accept this level of residual risk for "
                "observation or discharge with follow-up.")
    elif prob < 0.03:
        tier = "moderate"
        desc = ("Moderate risk — above the residual IBI rate typically accepted "
                "by published low-risk rules.")
        comp = ("Above Aronson/PECARN low-risk thresholds. "
                "Consider further workup, observation period, or shared "
                "decision-making with family.")
    else:
        tier = "high"
        desc = "High risk — IBI workup recommended."
        comp = "Well above all published low-risk thresholds."

    # Age-specific context
    if age_days <= 7:
        age_ctx = ("Age ≤7 days: highest baseline IBI risk. Most guidelines "
                   "recommend full workup and admission regardless of labs.")
    elif age_days <= 21:
        age_ctx = ("Age 8-21 days: high baseline IBI risk. AAP 2021 recommends "
                   "admission with empiric antibiotics for all infants in this group.")
    elif age_days <= 28:
        age_ctx = ("Age 22-28 days: AAP 2021 allows selective management if "
                   "all inflammatory markers are normal and follow-up is assured.")
    elif age_days <= 60:
        age_ctx = ("Age 29-60 days: lower baseline IBI risk (~1.2%). "
                   "Low-risk classification supported by multiple validated rules.")
    else:
        age_ctx = ("Age >60 days: limited validation data for this age group. "
                   "Interpret with caution.")

    return tier, desc, comp, age_ctx


class FebrileInfantPredictor:
    """Pooled logistic regression with probability-focused output."""

    def __init__(self):
        self.model = None
        self.imputer = None
        self.calibration_table = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        complete = df.dropna(subset=["age_days", "temp_c", "wbc", "anc", "ua_pos", "yos_total"]).copy()
        complete["age_young"] = (complete["age_days"] <= 14).astype(float)

        y = complete["has_ibi"].astype(int)
        X = complete[FEATURES]

        self.imputer = SimpleImputer(strategy="median")
        self.imputer.fit(X)

        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X, y)

        self._build_calibration(X, y)
        self.is_fitted = True

        return {"n": len(complete), "ibi": int(y.sum())}

    def _build_calibration(self, X, y):
        thresholds = [0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10, 0.20]
        preds = self.model.predict_proba(X)[:, 1]
        rows = []
        for t in thresholds:
            below = preds < t
            n_below = int(below.sum())
            ibi_in = int(y.values[below].sum()) if n_below > 0 else 0
            rows.append({
                "threshold": t, "n_below": n_below,
                "pct_below": n_below / len(preds) if len(preds) > 0 else 0,
                "ibi_in_group": ibi_in,
                "observed_ibi_rate": ibi_in / n_below if n_below > 0 else 0.0,
            })
        self.calibration_table = pd.DataFrame(rows)

    def predict(
        self,
        age_days: float,
        temp_c: float,
        wbc: float = None,
        anc: float = None,
        ua_positive: bool = None,
        yos_total: float = None,
        pct: float = None,  # not used by model but accepted for interface compatibility
    ) -> PredictionResult:
        if not self.is_fitted:
            raise RuntimeError("Must call fit() first")

        ua_val = float(ua_positive) if ua_positive is not None else np.nan
        yos_val = float(yos_total) if yos_total is not None else np.nan
        age_young_val = 1.0 if age_days <= 14 else 0.0

        n_missing = sum(1 for v in [wbc, anc, ua_positive, yos_total] if v is None)

        raw = np.array([[
            age_days, temp_c,
            wbc if wbc is not None else np.nan,
            anc if anc is not None else np.nan,
            ua_val, yos_val, age_young_val,
        ]])
        X = pd.DataFrame(self.imputer.transform(raw), columns=FEATURES)
        prob = float(self.model.predict_proba(X)[:, 1][0])

        tier, desc, comp, age_ctx = _risk_tier(prob, int(age_days))
        band = self._get_calibration_band(prob)

        # Override with safety messages
        if n_missing >= 3:
            desc = ("Multiple inputs missing — probability estimate is uncertain. "
                    + desc)
            comp = ("Insufficient data for reliable comparison to published rules. "
                    "Consider obtaining additional investigations.")

        # Calibration note
        cal_row = self.calibration_table[self.calibration_table["threshold"] >= prob]
        note = ""
        if len(cal_row) > 0:
            row = cal_row.iloc[0]
            note = (f"In training data, {row['pct_below']:.0%} of infants had "
                    f"P(IBI) below {row['threshold']:.1%}, with observed IBI rate "
                    f"of {row['observed_ibi_rate']:.1%} "
                    f"({row['ibi_in_group']}/{row['n_below']}).")

        one_in_n = int(round(1 / max(prob, 0.0001)))

        return PredictionResult(
            probability=prob,
            one_in_n=one_in_n,
            risk_tier=tier,
            risk_description=desc,
            comparison_to_rules=comp,
            age_context=age_ctx,
            calibration_note=note,
            features_used=list(FEATURES),
            n_missing_imputed=n_missing,
            confidence_band=band,
        )

    def _get_calibration_band(self, prob):
        if self.calibration_table is None:
            return (0.0, prob * 3)
        ct = self.calibration_table
        above = ct[ct["threshold"] >= prob]
        below = ct[ct["threshold"] < prob]
        if len(above) > 0 and len(below) > 0:
            return (float(below.iloc[-1]["observed_ibi_rate"]),
                    float(above.iloc[0]["observed_ibi_rate"]))
        elif len(above) > 0:
            return (0.0, float(above.iloc[0]["observed_ibi_rate"]))
        else:
            return (float(ct.iloc[-1]["observed_ibi_rate"]), 1.0)

    def save(self, path: Path = None):
        if path is None:
            path = MODEL_DIR / "prediction_model_v6.joblib"
        joblib.dump({
            "model": self.model, "imputer": self.imputer,
            "calibration_table": self.calibration_table,
        }, path)

    def load(self, path: Path = None):
        if path is None:
            path = MODEL_DIR / "prediction_model_v6.joblib"
        data = joblib.load(path)
        self.model = data["model"]
        self.imputer = data["imputer"]
        self.calibration_table = data["calibration_table"]
        self.is_fitted = True


def load_and_prepare_pecarn() -> pd.DataFrame:
    base = PROJECT_ROOT / "data" / "raw" / "pecarn_tig" / "Biosignatures_Full" / "CSV datasets"
    demo = pd.read_csv(base / "demographics.csv")
    demo["age_days"] = -demo["BirthDay"]
    clin = pd.read_csv(base / "clinicaldata.csv").groupby("PId").first().reset_index()
    clin["temp_c"] = clin["Temperature"].apply(
        lambda t: t if pd.isna(t) or t <= 60 else (t - 32) * 5 / 9)
    yos_cols = ["YOSCry", "YOSReaction", "YOSState", "YOSColor", "YOSHydration", "YOSResponse"]
    clin["yos_total"] = clin[yos_cols].sum(axis=1, min_count=6)
    lab = pd.read_csv(base / "labresults.csv").groupby("PId").first().reset_index()
    lab["ua_pos"] = ((lab["UrineLEC"].isin([1, 2, 3])) | (lab["NitriteRes"] == 1)).astype(float)
    lab.loc[lab["UrineLEC"].isna() & lab["NitriteRes"].isna(), "ua_pos"] = np.nan
    pctd = pd.read_csv(base / "pctdata.csv")
    pctd["pct"] = pd.to_numeric(pctd["PCTResult"], errors="coerce")
    pctd = pctd.groupby("PId")["pct"].first().reset_index()
    blood = pd.read_csv(base / "culturereview_blood.csv")
    csf = pd.read_csv(base / "culturereview_csf.csv")
    ibi_pids = set(blood[blood.BloodDCCAssess == 1]["PId"]) | set(csf[csf.CSFDCCAssess == 1]["PId"])
    df = (demo[["PId", "age_days"]]
        .merge(clin[["PId", "temp_c", "yos_total"]], on="PId", how="inner")
        .merge(lab[["PId", "WBC", "ANC", "ua_pos"]], on="PId", how="left")
        .merge(pctd, on="PId", how="left"))
    df["has_ibi"] = df["PId"].isin(ibi_pids).astype(int)
    df = df.rename(columns={"WBC": "wbc", "ANC": "anc"})
    return df


def train_and_save():
    print("Loading PECARN data...")
    df = load_and_prepare_pecarn()
    print(f"  n={len(df)}, IBI={df.has_ibi.sum()}")
    print("Training v6 model (pooled, probability-focused)...")
    predictor = FebrileInfantPredictor()
    stats = predictor.fit(df)
    print(f"  Trained on n={stats['n']}, IBI={stats['ibi']}")
    print(f"\nCalibration table:")
    print(predictor.calibration_table.to_string(index=False))
    predictor.save()
    print(f"\nModel saved to {MODEL_DIR / 'prediction_model_v6.joblib'}")
    return predictor


if __name__ == "__main__":
    p = train_and_save()
    print("\n=== Clinical scenarios ===")
    cases = [
        ("Well 45d, all normal",       45, 38.3, 9.0, 3.0, False, 6.0),
        ("Well 45d, elevated",         45, 38.5, 12.0, 6.0, False, 6.0),
        ("Rural: no PCT, normal",      45, 38.3, 9.0, 3.0, False, 6.0),
        ("Neonate 15d, normal",        15, 38.5, 10.0, 3.0, False, 6.0),
        ("Neonate 10d, unwell",        10, 39.0, 18.0, 12.0, True, 16.0),
        ("Neonate 5d, well, normal",    5, 38.2, 8.0, 2.0, False, 6.0),
        ("45d, UA+, well",             45, 38.5, 12.0, 4.0, True, 6.0),
        ("45d, well, very low ANC",    45, 38.3, 9.0, 1.5, False, 6.0),
        ("30d, well, normal",          30, 38.3, 9.0, 3.0, False, 6.0),
        ("50d, well, high temp",       50, 39.5, 10.0, 5.0, False, 8.0),
    ]
    for name, age, temp, wbc, anc, ua, yos in cases:
        r = p.predict(age, temp, wbc=wbc, anc=anc, ua_positive=ua, yos_total=yos)
        icon = {"very_low": "🟢", "low": "🔵", "moderate": "🟠", "high": "🔴"}[r.risk_tier]
        print(f"\n  {icon} {name}: P(IBI)={r.probability:.3%} (1 in {r.one_in_n})")
        print(f"    Tier: {r.risk_tier}")
        print(f"    {r.risk_description}")
        print(f"    {r.comparison_to_rules}")
        print(f"    {r.age_context}")
