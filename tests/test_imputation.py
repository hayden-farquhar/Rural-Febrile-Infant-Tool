"""Tests for substitution policies and multiple imputation."""

import numpy as np
import pandas as pd
import pytest

from src.imputation.substitution_policies import (
    EvidenceGrade,
    SCENARIOS,
    crp_exceeds_threshold,
    get_scenario_substitutions,
    handle_no_biomarkers,
    substitute_anc_poct,
    substitute_gbs_status_unknown,
    substitute_pct_with_crp,
    substitute_ua_delayed,
)
from src.imputation.multiple_imputation import (
    multiple_impute,
    pool_estimates_rubins,
)


# ── Substitution policies ─────────────────────────────────────

class TestPCTSubstitution:
    def test_crp_available(self):
        result = substitute_pct_with_crp(crp_mg_l=15.0)
        assert result is not None
        assert result.evidence_grade == EvidenceGrade.STRONG
        assert result.substituted_value == 15.0
        assert result.variance_inflation == 1.1

    def test_crp_none(self):
        result = substitute_pct_with_crp(crp_mg_l=None)
        assert result is None

    def test_crp_threshold(self):
        assert crp_exceeds_threshold(20.0) is True
        assert crp_exceeds_threshold(19.9) is False
        assert crp_exceeds_threshold(0.0) is False


class TestGBSSubstitution:
    def test_default_prior(self):
        result = substitute_gbs_status_unknown()
        assert result.evidence_grade == EvidenceGrade.MODERATE
        vals = result.substituted_value
        assert vals["gbs_colonisation_rate"] == 0.25
        assert vals["iap_coverage_rate"] == 0.70
        assert abs(vals["prob_gbs_positive_no_iap"] - 0.075) < 0.001
        assert result.variance_inflation == 1.5


class TestUASubstitution:
    def test_ua_within_4h(self):
        result = substitute_ua_delayed(
            ua_available=True, ua_age_hours=2.0,
            ua_le_positive=False, ua_nitrites_positive=False,
        )
        assert result is not None
        assert result.evidence_grade == EvidenceGrade.WEAK
        assert result.substituted_value is not None

    def test_ua_too_old(self):
        result = substitute_ua_delayed(
            ua_available=True, ua_age_hours=5.0,
        )
        assert result.substituted_value is None  # treated as missing

    def test_ua_not_available(self):
        result = substitute_ua_delayed(ua_available=False)
        assert result is None


class TestANCPOCT:
    def test_poct_value(self):
        result = substitute_anc_poct(poct_anc=3.5)
        assert result.evidence_grade == EvidenceGrade.MODERATE
        assert result.substituted_value == 3.5
        assert result.variance_inflation == 1.3

    def test_poct_none(self):
        result = substitute_anc_poct(poct_anc=None)
        assert result is None


class TestNoBiomarkers:
    def test_demonstration_only(self):
        result = handle_no_biomarkers()
        assert result.evidence_grade == EvidenceGrade.DEMONSTRATION_ONLY
        assert result.substituted_value is None
        assert result.variance_inflation == 3.0


class TestScenarios:
    def test_all_scenarios_defined(self):
        assert set(SCENARIOS.keys()) == {"baseline", "A", "B", "C", "D", "E"}

    def test_baseline_no_substitutions(self):
        results = get_scenario_substitutions("baseline")
        assert results == []

    def test_scenario_a_with_crp(self):
        results = get_scenario_substitutions("A", crp_mg_l=10.0)
        assert len(results) == 1
        assert results[0].evidence_grade == EvidenceGrade.STRONG

    def test_scenario_a_without_crp(self):
        results = get_scenario_substitutions("A", crp_mg_l=None)
        assert len(results) == 1
        assert results[0].evidence_grade == EvidenceGrade.DEMONSTRATION_ONLY

    def test_scenario_b(self):
        results = get_scenario_substitutions("B", crp_mg_l=5.0)
        assert len(results) == 2  # PCT sub + GBS sub

    def test_scenario_e(self):
        results = get_scenario_substitutions("E")
        assert len(results) == 1
        assert results[0].evidence_grade == EvidenceGrade.DEMONSTRATION_ONLY

    def test_invalid_scenario(self):
        with pytest.raises(ValueError):
            get_scenario_substitutions("Z")


# ── Multiple imputation ───────────────────────────────────────

class TestMultipleImputation:
    def test_imputation_produces_m_datasets(self):
        data = pd.DataFrame({
            "wbc": [10.0, np.nan, 12.0, 8.0, np.nan, 11.0, 9.0, 13.0, 7.0, 10.5],
            "anc": [4.0, 3.0, np.nan, 5.0, 2.0, np.nan, 4.5, 6.0, 3.5, 4.0],
            "crp": [5.0, 10.0, 2.0, np.nan, 15.0, 8.0, 3.0, np.nan, 12.0, 6.0],
        })
        completed = multiple_impute(data, ["wbc", "anc", "crp"], n_imputations=5)
        assert len(completed) == 5
        for df in completed:
            assert df.isna().sum().sum() == 0  # no missing values

    def test_imputed_values_within_bounds(self):
        data = pd.DataFrame({
            "wbc": [10.0, np.nan, 12.0, np.nan, 8.0, 11.0, 9.0, 14.0, 7.0, 10.5],
            "temp_c": [38.5, 38.3, np.nan, 38.8, 38.1, np.nan, 38.6, 38.4, 38.2, 38.7],
        })
        completed = multiple_impute(data, ["wbc", "temp_c"], n_imputations=3)
        for df in completed:
            assert df["wbc"].min() >= 0.5
            assert df["wbc"].max() <= 50.0
            assert df["temp_c"].min() >= 36.0
            assert df["temp_c"].max() <= 42.0


class TestRubinsRules:
    def test_no_between_variance(self):
        """When all imputations agree, total variance equals within variance."""
        estimates = np.array([0.5, 0.5, 0.5])
        variances = np.array([0.01, 0.01, 0.01])
        q, t, lo, hi = pool_estimates_rubins(estimates, variances)
        assert q == 0.5
        assert abs(t - 0.01) < 0.001  # no between-imputation variance

    def test_with_between_variance(self):
        """Between-imputation variance inflates total variance."""
        estimates = np.array([0.4, 0.5, 0.6])
        variances = np.array([0.01, 0.01, 0.01])
        q, t, lo, hi = pool_estimates_rubins(estimates, variances)
        assert abs(q - 0.5) < 0.001
        assert t > 0.01  # inflated by between-imputation variance
        assert lo < q < hi
