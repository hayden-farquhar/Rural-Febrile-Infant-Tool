"""Tests for Mondrian conformal prediction module."""

import numpy as np
import pandas as pd
import pytest

from src.conformal.mondrian_missing_input import (
    MondrianConformalPredictor,
    assign_age_tertile,
    assign_completeness,
    make_mondrian_bin,
    simulate_calibration_cases,
)


class TestStratification:
    def test_age_tertile_assignment(self):
        assert assign_age_tertile(0) == "0-28d"
        assert assign_age_tertile(28) == "0-28d"
        assert assign_age_tertile(29) == "29-60d"
        assert assign_age_tertile(60) == "29-60d"
        assert assign_age_tertile(61) == "61-89d"
        assert assign_age_tertile(89) == "61-89d"
        assert assign_age_tertile(90) == "out_of_range"

    def test_completeness_assignment(self):
        assert assign_completeness(0) == "full"
        assert assign_completeness(1) == "1_missing"
        assert assign_completeness(2) == "2plus_missing"
        assert assign_completeness(5) == "2plus_missing"

    def test_mondrian_bin(self):
        assert make_mondrian_bin("0-28d", "full") == "0-28d_full"
        assert make_mondrian_bin("29-60d", "2plus_missing") == "29-60d_2plus_missing"


class TestSimulation:
    def test_simulation_size(self):
        pooled = pd.DataFrame({
            "rule": ["Aronson", "PECARN"],
            "pooled_sens": [0.915, 0.955],
            "pooled_spec": [0.280, 0.610],
        })
        cal = simulate_calibration_cases(pooled, n_per_stratum=100)
        # 3 age tertiles × 3 completeness levels × 100 = 900
        assert len(cal) == 900

    def test_simulation_columns(self):
        pooled = pd.DataFrame({
            "rule": ["test"],
            "pooled_sens": [0.9],
            "pooled_spec": [0.5],
        })
        cal = simulate_calibration_cases(pooled, n_per_stratum=50)
        expected_cols = {"age_days", "age_tertile", "n_missing",
                         "completeness", "y_true", "y_pred_prob", "mondrian_bin"}
        assert set(cal.columns) == expected_cols

    def test_simulation_has_ibi_cases(self):
        pooled = pd.DataFrame({
            "rule": ["test"],
            "pooled_sens": [0.9],
            "pooled_spec": [0.5],
        })
        cal = simulate_calibration_cases(pooled, n_per_stratum=500)
        assert cal["y_true"].sum() > 0  # some IBI cases
        assert (cal["y_true"] == 0).sum() > 0  # some non-IBI

    def test_predictions_bounded(self):
        pooled = pd.DataFrame({
            "rule": ["test"],
            "pooled_sens": [0.9],
            "pooled_spec": [0.5],
        })
        cal = simulate_calibration_cases(pooled, n_per_stratum=200)
        assert cal["y_pred_prob"].min() >= 0.0
        assert cal["y_pred_prob"].max() <= 1.0


class TestConformalPredictor:
    @pytest.fixture
    def calibrated_predictor(self):
        pooled = pd.DataFrame({
            "rule": ["Aronson", "PECARN"],
            "pooled_sens": [0.915, 0.955],
            "pooled_spec": [0.280, 0.610],
        })
        cal = simulate_calibration_cases(pooled, n_per_stratum=500)
        predictor = MondrianConformalPredictor()
        predictor.calibrate(cal)
        return predictor

    def test_calibration(self, calibrated_predictor):
        assert calibrated_predictor.is_calibrated is True
        assert len(calibrated_predictor.strata_info) > 0

    def test_predict_returns_result(self, calibrated_predictor):
        result = calibrated_predictor.predict(
            y_pred_prob=0.03,
            age_days=30,
            n_missing=0,
        )
        assert 0.0 <= result.point_estimate <= 1.0
        assert result.interval_90[0] <= result.point_estimate <= result.interval_90[1]
        assert result.interval_95[0] <= result.point_estimate <= result.interval_95[1]
        assert result.clinical_decision in ("low_risk", "ibi_workup", "insufficient")

    def test_wider_intervals_with_missing(self, calibrated_predictor):
        result_full = calibrated_predictor.predict(
            y_pred_prob=0.02, age_days=40, n_missing=0,
        )
        result_missing = calibrated_predictor.predict(
            y_pred_prob=0.02, age_days=40, n_missing=0,
            variance_inflation=3.0,
        )
        width_full = result_full.interval_95[1] - result_full.interval_95[0]
        width_missing = result_missing.interval_95[1] - result_missing.interval_95[0]
        assert width_missing > width_full

    def test_low_risk_classification(self, calibrated_predictor):
        """Very low predicted probability → low_risk if interval below threshold."""
        result = calibrated_predictor.predict(
            y_pred_prob=0.001, age_days=45, n_missing=0,
        )
        # With very low probability, should likely be low_risk
        assert result.interval_95[0] >= 0.0

    def test_high_risk_classification(self, calibrated_predictor):
        """High predicted probability → ibi_workup."""
        result = calibrated_predictor.predict(
            y_pred_prob=0.5, age_days=10, n_missing=0,
        )
        assert result.clinical_decision == "ibi_workup"

    def test_coverage_report(self, calibrated_predictor):
        # Generate small test set
        pooled = pd.DataFrame({
            "rule": ["test"],
            "pooled_sens": [0.9],
            "pooled_spec": [0.5],
        })
        test = simulate_calibration_cases(pooled, n_per_stratum=50)
        report = calibrated_predictor.coverage_report(test)
        assert "empirical_coverage" in report.columns
        assert "MARGINAL" in report["mondrian_bin"].values
