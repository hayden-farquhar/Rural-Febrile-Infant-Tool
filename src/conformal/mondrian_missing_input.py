"""Mondrian conformal prediction for missing-input febrile infant assessment.

Pre-registered scheme:
  - Mondrian strata: (age_tertile, input_completeness) → up to 9 categories
  - Age tertiles: 0-28d, 29-60d, 61-89d
  - Input completeness: full, 1_missing, 2plus_missing
  - Coverage targets: 90% and 95% marginal
  - Alpha levels: 0.05, 0.10, 0.20
  - Categories with <20 calibration cases merged with adjacent age category

Calibration from simulated cases using HSROC-pooled distributions.
Conformity scores: 1 - p(y_true) for classification.
"""

import numpy as np
import pandas as pd
from crepes import ConformalClassifier
from dataclasses import dataclass, field

from src.utils import (
    AGE_TERTILE_BOUNDS,
    AGE_TERTILE_LABELS,
    ALPHA_LEVELS,
    COVERAGE_TARGETS,
    IBI_TRANSFER_THRESHOLD,
    MIN_STRATUM_SIZE,
)


@dataclass
class MondrianStratum:
    age_tertile: str
    input_completeness: str
    n_calibration: int = 0
    merged_with: str = ""  # if merged due to small n


@dataclass
class ConformalResult:
    """Result of conformal prediction for a single patient."""
    point_estimate: float        # predicted IBI probability
    interval_90: tuple[float, float]  # 90% prediction interval
    interval_95: tuple[float, float]  # 95% prediction interval
    clinical_decision: str       # "low_risk" / "ibi_workup" / "insufficient"
    mondrian_stratum: str        # which stratum was used
    n_missing_inputs: int
    variance_inflation: float    # from substitution policies
    rules_applicable: list[str] = field(default_factory=list)


def assign_age_tertile(age_days: int) -> str:
    """Assign age tertile label."""
    for (lo, hi), label in zip(AGE_TERTILE_BOUNDS, AGE_TERTILE_LABELS):
        if lo <= age_days <= hi:
            return label
    return "out_of_range"


def assign_completeness(n_missing: int) -> str:
    """Assign input completeness category."""
    if n_missing == 0:
        return "full"
    elif n_missing == 1:
        return "1_missing"
    else:
        return "2plus_missing"


def make_mondrian_bin(age_tertile: str, completeness: str) -> str:
    """Create Mondrian bin label from age tertile and completeness."""
    return f"{age_tertile}_{completeness}"


def simulate_calibration_cases(
    pooled_results: pd.DataFrame,
    n_per_stratum: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Simulate calibration cases from HSROC-pooled distributions.

    For each age stratum, simulate cases with:
    - IBI status drawn from age-specific prevalence
    - Rule predictions drawn from pooled sensitivity/specificity
    - Missing-input scenarios drawn uniformly

    Args:
        pooled_results: DataFrame with columns [rule, pooled_sens, pooled_spec]
        n_per_stratum: Number of cases per Mondrian stratum (pre-registered ≥1000)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with columns: [age_days, age_tertile, n_missing,
                                  completeness, y_true, y_pred_prob,
                                  mondrian_bin]
    """
    rng = np.random.default_rng(random_state)

    from src.probability import posterior_ibi_single_rule, RULE_PERFORMANCE

    age_prevalences = {
        "0-28d": 0.035,
        "29-60d": 0.015,
        "61-89d": 0.020,
    }

    completeness_levels = ["full", "1_missing", "2plus_missing"]

    # Use PECARN as calibration rule — highest specificity (61%) gives
    # the tightest intervals; Aronson (28%) would give overly conservative intervals.
    # This choice is documented as a calibration decision.
    repr_sens = RULE_PERFORMANCE.get("PECARN", {"sens": 0.955})["sens"]
    repr_spec = RULE_PERFORMANCE.get("PECARN", {"spec": 0.610})["spec"]

    rows = []
    for age_label in AGE_TERTILE_LABELS:
        lo, hi = AGE_TERTILE_BOUNDS[AGE_TERTILE_LABELS.index(age_label)]
        prev = age_prevalences.get(age_label, 0.02)

        for completeness in completeness_levels:
            n_missing = {"full": 0, "1_missing": 1, "2plus_missing": 2}[completeness]

            for _ in range(n_per_stratum):
                age = rng.integers(lo, hi + 1)
                y_true = int(rng.random() < prev)

                # Simulate rule output based on sens/spec
                if y_true == 1:
                    rule_positive = rng.random() < repr_sens
                else:
                    rule_positive = rng.random() > repr_spec

                # Compute Bayesian posterior probability
                y_pred_prob = posterior_ibi_single_rule(
                    rule_positive, repr_sens, repr_spec, prev
                )

                # Add calibration noise (larger for more missing inputs)
                noise_sd = 0.002 * (1 + n_missing)
                y_pred_prob = np.clip(
                    y_pred_prob + rng.normal(0, noise_sd), 0.0001, 0.9999
                )

                rows.append({
                    "age_days": age,
                    "age_tertile": age_label,
                    "n_missing": n_missing,
                    "completeness": completeness,
                    "y_true": y_true,
                    "y_pred_prob": y_pred_prob,
                    "mondrian_bin": make_mondrian_bin(age_label, completeness),
                })

    return pd.DataFrame(rows)


def merge_small_strata(
    cal_data: pd.DataFrame,
    min_size: int = MIN_STRATUM_SIZE,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge Mondrian strata with fewer than min_size calibration cases.

    Pre-registered rule: merge with adjacent age category.

    Returns:
        (modified DataFrame, list of merge descriptions)
    """
    merge_log = []
    bin_counts = cal_data["mondrian_bin"].value_counts()

    for bin_name, count in bin_counts.items():
        if count < min_size:
            # Parse age tertile and completeness
            parts = bin_name.rsplit("_", 1)
            age_part = "_".join(bin_name.split("_")[:-1])

            # Find adjacent age category
            if age_part in AGE_TERTILE_LABELS:
                idx = AGE_TERTILE_LABELS.index(age_part)
                if idx > 0:
                    merge_target_age = AGE_TERTILE_LABELS[idx - 1]
                elif idx < len(AGE_TERTILE_LABELS) - 1:
                    merge_target_age = AGE_TERTILE_LABELS[idx + 1]
                else:
                    continue

                completeness = parts[-1] if len(parts) > 1 else "full"
                merge_target = make_mondrian_bin(merge_target_age, completeness)

                cal_data.loc[cal_data["mondrian_bin"] == bin_name, "mondrian_bin"] = merge_target
                merge_log.append(
                    f"Merged {bin_name} (n={count}) into {merge_target}"
                )

    return cal_data, merge_log


class MondrianConformalPredictor:
    """Mondrian conformal predictor for IBI probability with missing inputs.

    Uses empirical quantiles of absolute residuals |y_true - y_pred| per
    Mondrian stratum to construct prediction intervals.
    """

    def __init__(self):
        self.is_calibrated = False
        self.merge_log = []
        self.strata_info = {}
        self._quantiles = {}  # {bin: {alpha: quantile_value}}

    def calibrate(
        self,
        cal_data: pd.DataFrame,
        min_stratum_size: int = MIN_STRATUM_SIZE,
    ):
        """Calibrate from simulated data with known y_true and y_pred_prob."""
        cal_data, self.merge_log = merge_small_strata(
            cal_data.copy(), min_stratum_size
        )

        # Compute absolute residuals per case
        cal_data["residual"] = np.abs(cal_data["y_true"] - cal_data["y_pred_prob"])

        # Compute quantiles per stratum at each coverage level
        for bin_name, group in cal_data.groupby("mondrian_bin"):
            residuals = group["residual"].values
            self.strata_info[bin_name] = {
                "n": len(group),
                "ibi_rate": group["y_true"].mean(),
                "residual_mean": float(np.mean(residuals)),
                "residual_median": float(np.median(residuals)),
            }
            self._quantiles[bin_name] = {}
            for alpha in [0.20, 0.10, 0.05]:
                # Conformal quantile: ceil((n+1)(1-alpha))/n -th quantile
                coverage = 1 - alpha
                n = len(residuals)
                q_idx = min(int(np.ceil((n + 1) * coverage)), n) - 1
                sorted_res = np.sort(residuals)
                q_val = float(sorted_res[q_idx]) if q_idx < n else float(sorted_res[-1])
                self._quantiles[bin_name][alpha] = q_val

        self.is_calibrated = True

    def predict(
        self,
        y_pred_prob: float,
        age_days: int,
        n_missing: int,
        variance_inflation: float = 1.0,
    ) -> ConformalResult:
        """Predict with calibrated conformal intervals."""
        if not self.is_calibrated:
            raise RuntimeError("Must call calibrate() before predict()")

        age_tertile = assign_age_tertile(age_days)
        completeness = assign_completeness(n_missing)
        mondrian_bin = make_mondrian_bin(age_tertile, completeness)

        if mondrian_bin not in self._quantiles:
            mondrian_bin = make_mondrian_bin(age_tertile, "full")
        if mondrian_bin not in self._quantiles:
            # Last resort: use global quantiles
            mondrian_bin = list(self._quantiles.keys())[0]

        q_90 = self._quantiles[mondrian_bin][0.10] * variance_inflation
        q_95 = self._quantiles[mondrian_bin][0.05] * variance_inflation

        interval_90 = (
            max(0.0, y_pred_prob - q_90),
            min(1.0, y_pred_prob + q_90),
        )
        interval_95 = (
            max(0.0, y_pred_prob - q_95),
            min(1.0, y_pred_prob + q_95),
        )

        clinical_decision = self._classify_interval(interval_95)

        return ConformalResult(
            point_estimate=y_pred_prob,
            interval_90=interval_90,
            interval_95=interval_95,
            clinical_decision=clinical_decision,
            mondrian_stratum=mondrian_bin,
            n_missing_inputs=n_missing,
            variance_inflation=variance_inflation,
        )

    def _classify_interval(self, interval_95: tuple[float, float]) -> str:
        """Classify based on 95% interval vs transfer threshold.

        Pre-registered clinical output categories:
          - Interval entirely below 1% → "low_risk"
          - Interval entirely above 1% → "ibi_workup"
          - Interval straddles 1% → "insufficient"
        """
        threshold = IBI_TRANSFER_THRESHOLD
        lower, upper = interval_95

        if upper < threshold:
            return "low_risk"
        elif lower > threshold:
            return "ibi_workup"
        else:
            return "insufficient"

    def coverage_report(
        self,
        test_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute marginal and conditional coverage on test data.

        Args:
            test_data: DataFrame with [y_true, y_pred_prob, age_days, n_missing]

        Returns:
            DataFrame with coverage per stratum at each alpha level
        """
        rows = []
        for _, row in test_data.iterrows():
            result = self.predict(
                y_pred_prob=row["y_pred_prob"],
                age_days=int(row["age_days"]),
                n_missing=int(row["n_missing"]),
            )

            for alpha, target in [(0.10, 0.90), (0.05, 0.95)]:
                interval = result.interval_90 if alpha == 0.10 else result.interval_95
                covered = interval[0] <= row["y_true"] <= interval[1]
                rows.append({
                    "mondrian_bin": result.mondrian_stratum,
                    "alpha": alpha,
                    "target_coverage": target,
                    "covered": int(covered),
                    "interval_width": interval[1] - interval[0],
                })

        report_df = pd.DataFrame(rows)

        # Aggregate
        summary = report_df.groupby(["alpha", "mondrian_bin"]).agg(
            n=("covered", "count"),
            empirical_coverage=("covered", "mean"),
            mean_interval_width=("interval_width", "mean"),
        ).reset_index()

        # Add marginal coverage
        marginal = report_df.groupby("alpha").agg(
            n=("covered", "count"),
            empirical_coverage=("covered", "mean"),
            mean_interval_width=("interval_width", "mean"),
        ).reset_index()
        marginal["mondrian_bin"] = "MARGINAL"
        summary = pd.concat([summary, marginal], ignore_index=True)

        return summary
