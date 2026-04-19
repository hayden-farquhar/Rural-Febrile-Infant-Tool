"""Unit tests for decision rule implementations.

Tests use worked examples and boundary cases from source publications.
Each test documents the clinical scenario to enable two-person verification
against the published rule definitions.
"""

import pytest
from src.rules import rochester, philadelphia, step_by_step, pecarn, aronson, aap_2021, nice_ng143


# ── Rochester ──────────────────────────────────────────────────

class TestRochester:
    def test_classic_low_risk(self):
        """45-day-old, well, WBC 10K, bands 0.5K, UA 2 WBC/hpf → low-risk."""
        r = rochester.apply(rochester.RochesterInputs(
            age_days=45, temp_c=38.5, well_appearing=True,
            previously_healthy=True, no_focal_infection=True,
            wbc=10.0, band_count=0.5, ua_wbc_hpf=2, ua_bacteria_gram=False,
        ))
        assert r.prediction == 0
        assert r.triggered_criteria == []

    def test_wbc_too_high(self):
        """Same infant but WBC 18K → not low-risk."""
        r = rochester.apply(rochester.RochesterInputs(
            age_days=45, temp_c=38.5, well_appearing=True,
            previously_healthy=True, no_focal_infection=True,
            wbc=18.0, band_count=0.5, ua_wbc_hpf=2, ua_bacteria_gram=False,
        ))
        assert r.prediction == 1
        assert "wbc_abnormal" in r.triggered_criteria

    def test_wbc_too_low(self):
        """WBC 3K (below 5K) → not low-risk."""
        r = rochester.apply(rochester.RochesterInputs(
            age_days=30, temp_c=38.3, well_appearing=True,
            previously_healthy=True, no_focal_infection=True,
            wbc=3.0, band_count=0.2, ua_wbc_hpf=0, ua_bacteria_gram=False,
        ))
        assert r.prediction == 1
        assert "wbc_abnormal" in r.triggered_criteria

    def test_elevated_bands(self):
        """Bands 2.0K (>1.5K) → not low-risk."""
        r = rochester.apply(rochester.RochesterInputs(
            age_days=30, temp_c=38.3, well_appearing=True,
            previously_healthy=True, no_focal_infection=True,
            wbc=12.0, band_count=2.0, ua_wbc_hpf=0, ua_bacteria_gram=False,
        ))
        assert r.prediction == 1
        assert "bands_elevated" in r.triggered_criteria

    def test_ill_appearing(self):
        """Ill-appearing infant → not low-risk regardless of labs."""
        r = rochester.apply(rochester.RochesterInputs(
            age_days=40, temp_c=38.5, well_appearing=False,
            previously_healthy=True, no_focal_infection=True,
            wbc=10.0, band_count=0.5, ua_wbc_hpf=2, ua_bacteria_gram=False,
        ))
        assert r.prediction == 1
        assert "ill_appearing" in r.triggered_criteria

    def test_age_over_60(self):
        """70-day-old → out of range, not applicable."""
        r = rochester.apply(rochester.RochesterInputs(
            age_days=70, temp_c=38.5, well_appearing=True,
            previously_healthy=True, no_focal_infection=True,
            wbc=10.0, band_count=0.5, ua_wbc_hpf=2, ua_bacteria_gram=False,
        ))
        assert r.prediction == 1
        assert r.applicable is False

    def test_diarrhoea_with_stool_wbc(self):
        """Diarrhoea with stool WBC 8/hpf (>5) → not low-risk."""
        r = rochester.apply(rochester.RochesterInputs(
            age_days=45, temp_c=38.5, well_appearing=True,
            previously_healthy=True, no_focal_infection=True,
            wbc=10.0, band_count=0.5, ua_wbc_hpf=2, ua_bacteria_gram=False,
            has_diarrhoea=True, stool_wbc_hpf=8,
        ))
        assert r.prediction == 1
        assert "stool_wbc_elevated" in r.triggered_criteria

    def test_boundary_wbc_at_15(self):
        """WBC exactly 15.0K → within range, low-risk if all else normal."""
        r = rochester.apply(rochester.RochesterInputs(
            age_days=45, temp_c=38.5, well_appearing=True,
            previously_healthy=True, no_focal_infection=True,
            wbc=15.0, band_count=0.5, ua_wbc_hpf=2, ua_bacteria_gram=False,
        ))
        assert r.prediction == 0

    def test_boundary_wbc_at_5(self):
        """WBC exactly 5.0K → within range, low-risk if all else normal."""
        r = rochester.apply(rochester.RochesterInputs(
            age_days=45, temp_c=38.5, well_appearing=True,
            previously_healthy=True, no_focal_infection=True,
            wbc=5.0, band_count=0.5, ua_wbc_hpf=2, ua_bacteria_gram=False,
        ))
        assert r.prediction == 0


# ── Philadelphia ───────────────────────────────────────────────

class TestPhiladelphia:
    def test_classic_low_risk(self):
        """35-day-old, well, WBC 12K, B:N 0.1, UA clean, CSF 2 → low-risk."""
        r = philadelphia.apply(philadelphia.PhiladelphiaInputs(
            age_days=35, temp_c=38.5, well_appearing=True,
            wbc=12.0, band_neutrophil_ratio=0.1,
            ua_wbc_hpf=3, ua_gram_negative=True, ua_le_negative=True,
            csf_wbc=2, csf_gram_negative=True,
        ))
        assert r.prediction == 0

    def test_age_too_young(self):
        """20-day-old → out of Philadelphia range (29–60d)."""
        r = philadelphia.apply(philadelphia.PhiladelphiaInputs(
            age_days=20, temp_c=38.5, well_appearing=True,
            wbc=10.0, band_neutrophil_ratio=0.1,
            ua_wbc_hpf=2, ua_gram_negative=True, ua_le_negative=True,
            csf_wbc=1, csf_gram_negative=True,
        ))
        assert r.applicable is False

    def test_csf_missing_not_applicable(self):
        """CSF not done → cannot classify as low-risk (LP required)."""
        r = philadelphia.apply(philadelphia.PhiladelphiaInputs(
            age_days=40, temp_c=38.5, well_appearing=True,
            wbc=10.0, band_neutrophil_ratio=0.1,
            ua_wbc_hpf=2, ua_gram_negative=True, ua_le_negative=True,
            csf_wbc=None,
        ))
        assert r.prediction == 1
        assert "csf_missing_lp_required" in r.triggered_criteria

    def test_elevated_csf(self):
        """CSF WBC 12 (≥8) → not low-risk."""
        r = philadelphia.apply(philadelphia.PhiladelphiaInputs(
            age_days=40, temp_c=38.5, well_appearing=True,
            wbc=10.0, band_neutrophil_ratio=0.1,
            ua_wbc_hpf=2, ua_gram_negative=True, ua_le_negative=True,
            csf_wbc=12, csf_gram_negative=True,
        ))
        assert r.prediction == 1
        assert "csf_wbc_elevated" in r.triggered_criteria


# ── Step-by-Step ───────────────────────────────────────────────

class TestStepByStep:
    def test_classic_low_risk(self):
        """40-day-old, well, UA neg, PCT 0.2, CRP 5, ANC 3K → low-risk."""
        r = step_by_step.apply(step_by_step.StepByStepInputs(
            age_days=40, temp_c=38.5, well_appearing=True,
            ua_le_positive=False, ua_nitrites_positive=False, ua_wbc_hpf=2,
            pct=0.2, crp=5.0, anc=3.0,
        ))
        assert r.prediction == 0
        assert r.step_reached == 7

    def test_ill_appearing_stops_at_step_1(self):
        """Ill-appearing → stops at step 1, no labs needed."""
        r = step_by_step.apply(step_by_step.StepByStepInputs(
            age_days=40, temp_c=38.5, well_appearing=False,
            ua_le_positive=False, ua_nitrites_positive=False,
            pct=0.1, crp=2.0, anc=2.0,
        ))
        assert r.prediction == 1
        assert r.step_reached == 1

    def test_age_21d_stops_at_step_2(self):
        """21-day-old, well-appearing → stops at step 2."""
        r = step_by_step.apply(step_by_step.StepByStepInputs(
            age_days=21, temp_c=38.5, well_appearing=True,
            ua_le_positive=False, ua_nitrites_positive=False,
            pct=0.1, crp=2.0, anc=2.0,
        ))
        assert r.prediction == 1
        assert r.step_reached == 2
        assert "age_leq_21d" in r.triggered_criteria

    def test_22d_passes_step_2(self):
        """22-day-old passes step 2 (age >21d)."""
        r = step_by_step.apply(step_by_step.StepByStepInputs(
            age_days=22, temp_c=38.5, well_appearing=True,
            ua_le_positive=False, ua_nitrites_positive=False, ua_wbc_hpf=0,
            pct=0.1, crp=2.0, anc=2.0,
        ))
        assert r.prediction == 0
        assert r.step_reached == 7

    def test_ua_positive_stops_at_step_3(self):
        """Positive LE → stops at step 3."""
        r = step_by_step.apply(step_by_step.StepByStepInputs(
            age_days=40, temp_c=38.5, well_appearing=True,
            ua_le_positive=True, ua_nitrites_positive=False,
            pct=0.1, crp=2.0, anc=2.0,
        ))
        assert r.prediction == 1
        assert r.step_reached == 3

    def test_pct_elevated_stops_at_step_4(self):
        """PCT 0.8 (≥0.5) → stops at step 4."""
        r = step_by_step.apply(step_by_step.StepByStepInputs(
            age_days=40, temp_c=38.5, well_appearing=True,
            ua_le_positive=False, ua_nitrites_positive=False, ua_wbc_hpf=2,
            pct=0.8, crp=2.0, anc=2.0,
        ))
        assert r.prediction == 1
        assert r.step_reached == 4

    def test_crp_elevated_stops_at_step_5(self):
        """CRP 25 (≥20) → stops at step 5."""
        r = step_by_step.apply(step_by_step.StepByStepInputs(
            age_days=40, temp_c=38.5, well_appearing=True,
            ua_le_positive=False, ua_nitrites_positive=False, ua_wbc_hpf=2,
            pct=0.3, crp=25.0, anc=2.0,
        ))
        assert r.prediction == 1
        assert r.step_reached == 5

    def test_anc_elevated_stops_at_step_6(self):
        """ANC 12K (≥10K) → stops at step 6."""
        r = step_by_step.apply(step_by_step.StepByStepInputs(
            age_days=40, temp_c=38.5, well_appearing=True,
            ua_le_positive=False, ua_nitrites_positive=False, ua_wbc_hpf=2,
            pct=0.3, crp=10.0, anc=12.0,
        ))
        assert r.prediction == 1
        assert r.step_reached == 6

    def test_pct_missing_not_applicable(self):
        """PCT missing → not applicable at step 4."""
        r = step_by_step.apply(step_by_step.StepByStepInputs(
            age_days=40, temp_c=38.5, well_appearing=True,
            ua_le_positive=False, ua_nitrites_positive=False, ua_wbc_hpf=2,
            pct=None, crp=5.0, anc=3.0,
        ))
        assert r.prediction == 1
        assert r.applicable is False
        assert r.step_reached == 4

    def test_age_91_out_of_range(self):
        """91-day-old → out of range."""
        r = step_by_step.apply(step_by_step.StepByStepInputs(
            age_days=91, temp_c=38.5, well_appearing=True,
            ua_le_positive=False, pct=0.1, crp=2.0, anc=2.0,
        ))
        assert r.applicable is False


# ── PECARN ─────────────────────────────────────────────────────

class TestPECARN:
    def test_young_infant_low_risk(self):
        """14-day-old, UA neg, ANC 2K, PCT 0.3 → low-risk."""
        r = pecarn.apply(pecarn.PECARNInputs(
            age_days=14, temp_c=38.5,
            ua_le_positive=False, ua_nitrites_positive=False,
            anc=2.0, pct=0.3,
        ))
        assert r.prediction == 0
        assert r.age_stratum == "7-28d"

    def test_young_infant_pct_elevated(self):
        """14-day-old, UA neg, ANC 2K, PCT 2.0 (≥1.71) → not low-risk."""
        r = pecarn.apply(pecarn.PECARNInputs(
            age_days=14, temp_c=38.5,
            ua_le_positive=False, ua_nitrites_positive=False,
            anc=2.0, pct=2.0,
        ))
        assert r.prediction == 1
        assert "pct_elevated" in r.triggered_criteria

    def test_young_infant_pct_missing(self):
        """14-day-old without PCT → not applicable (PCT required for 7-28d)."""
        r = pecarn.apply(pecarn.PECARNInputs(
            age_days=14, temp_c=38.5,
            ua_le_positive=False, ua_nitrites_positive=False,
            anc=2.0, pct=None,
        ))
        assert r.prediction == 1
        assert r.applicable is False

    def test_older_infant_low_risk_without_pct(self):
        """40-day-old, UA neg, ANC 2K, no PCT → low-risk (PCT not required for 29-60d)."""
        r = pecarn.apply(pecarn.PECARNInputs(
            age_days=40, temp_c=38.5,
            ua_le_positive=False, ua_nitrites_positive=False,
            anc=2.0, pct=None,
        ))
        assert r.prediction == 0
        assert r.age_stratum == "29-60d"

    def test_anc_at_boundary(self):
        """ANC exactly 4.09 (≥4.09) → not low-risk."""
        r = pecarn.apply(pecarn.PECARNInputs(
            age_days=40, temp_c=38.5,
            ua_le_positive=False, ua_nitrites_positive=False,
            anc=4.09, pct=None,
        ))
        assert r.prediction == 1
        assert "anc_elevated" in r.triggered_criteria

    def test_age_5d_out_of_range(self):
        """5-day-old → below PECARN range (7–60d)."""
        r = pecarn.apply(pecarn.PECARNInputs(
            age_days=5, temp_c=38.5,
            ua_le_positive=False, ua_nitrites_positive=False,
            anc=2.0, pct=0.1,
        ))
        assert r.applicable is False


# ── Aronson ────────────────────────────────────────────────────

class TestAronson:
    def test_classic_low_risk(self):
        """30-day-old, temp 38.2, UA neg, ANC 3K → low-risk."""
        r = aronson.apply(aronson.AronsonInputs(
            age_days=30, temp_c=38.2,
            ua_le_positive=False, ua_nitrites_positive=False,
            anc=3.0,
        ))
        assert r.prediction == 0

    def test_age_leq_21(self):
        """21-day-old → not low-risk (age ≤21d)."""
        r = aronson.apply(aronson.AronsonInputs(
            age_days=21, temp_c=38.2,
            ua_le_positive=False, ua_nitrites_positive=False,
            anc=3.0,
        ))
        assert r.prediction == 1
        assert "age_leq_21d" in r.triggered_criteria

    def test_22d_passes_age_criterion(self):
        """22-day-old passes the age criterion."""
        r = aronson.apply(aronson.AronsonInputs(
            age_days=22, temp_c=38.2,
            ua_le_positive=False, ua_nitrites_positive=False,
            anc=3.0,
        ))
        assert r.prediction == 0

    def test_temp_at_boundary(self):
        """Temp exactly 38.5 → not low-risk."""
        r = aronson.apply(aronson.AronsonInputs(
            age_days=30, temp_c=38.5,
            ua_le_positive=False, ua_nitrites_positive=False,
            anc=3.0,
        ))
        assert r.prediction == 1
        assert "temp_geq_38.5" in r.triggered_criteria

    def test_temp_just_below(self):
        """Temp 38.4 → passes temperature criterion."""
        r = aronson.apply(aronson.AronsonInputs(
            age_days=30, temp_c=38.4,
            ua_le_positive=False, ua_nitrites_positive=False,
            anc=3.0,
        ))
        assert r.prediction == 0

    def test_anc_at_threshold(self):
        """ANC exactly 5.185 (≥5.185) → not low-risk."""
        r = aronson.apply(aronson.AronsonInputs(
            age_days=30, temp_c=38.2,
            ua_le_positive=False, ua_nitrites_positive=False,
            anc=5.185,
        ))
        assert r.prediction == 1
        assert "anc_elevated" in r.triggered_criteria

    def test_multiple_triggers(self):
        """Age ≤21d AND temp ≥38.5 → both should trigger."""
        r = aronson.apply(aronson.AronsonInputs(
            age_days=15, temp_c=39.0,
            ua_le_positive=False, ua_nitrites_positive=False,
            anc=3.0,
        ))
        assert r.prediction == 1
        assert "age_leq_21d" in r.triggered_criteria
        assert "temp_geq_38.5" in r.triggered_criteria


# ── AAP 2021 ───────────────────────────────────────────────────

class TestAAP2021:
    def test_8_21d_always_admit(self):
        """15-day-old → always admit regardless of labs."""
        r = aap_2021.apply(aap_2021.AAP2021Inputs(
            age_days=15, temp_c=38.5, well_appearing=True,
            ua_le_positive=False, pct=0.1, anc=2.0, crp=5.0,
        ))
        assert r.prediction == 1
        assert r.age_stratum == "8-21d"

    def test_22_28d_low_risk(self):
        """25-day-old, all markers normal, UA neg → low-risk."""
        r = aap_2021.apply(aap_2021.AAP2021Inputs(
            age_days=25, temp_c=38.5, well_appearing=True,
            ua_le_positive=False, ua_nitrites_positive=False,
            pct=0.3, anc=2.0, crp=10.0,
        ))
        assert r.prediction == 0
        assert r.age_stratum == "22-28d"

    def test_22_28d_pct_elevated(self):
        """25-day-old, PCT 0.8 (≥0.5) → not low-risk, LP recommended."""
        r = aap_2021.apply(aap_2021.AAP2021Inputs(
            age_days=25, temp_c=38.5, well_appearing=True,
            ua_le_positive=False, ua_nitrites_positive=False,
            pct=0.8, anc=2.0, crp=10.0,
        ))
        assert r.prediction == 1
        assert r.lp_recommended is True
        assert "pct_elevated" in r.triggered_criteria

    def test_29_60d_low_risk(self):
        """45-day-old, all markers normal → low-risk, no LP needed."""
        r = aap_2021.apply(aap_2021.AAP2021Inputs(
            age_days=45, temp_c=38.5, well_appearing=True,
            ua_le_positive=False, ua_nitrites_positive=False,
            pct=0.2, anc=3.0, crp=8.0,
        ))
        assert r.prediction == 0
        assert r.age_stratum == "29-60d"
        assert r.lp_recommended is False

    def test_29_60d_crp_elevated(self):
        """45-day-old, CRP 25 (≥20) → not low-risk."""
        r = aap_2021.apply(aap_2021.AAP2021Inputs(
            age_days=45, temp_c=38.5, well_appearing=True,
            ua_le_positive=False, ua_nitrites_positive=False,
            pct=0.2, anc=3.0, crp=25.0,
        ))
        assert r.prediction == 1
        assert "crp_elevated" in r.triggered_criteria

    def test_ill_appearing_overrides(self):
        """Ill-appearing → not low-risk regardless of labs."""
        r = aap_2021.apply(aap_2021.AAP2021Inputs(
            age_days=45, temp_c=38.5, well_appearing=False,
            ua_le_positive=False, pct=0.1, anc=2.0, crp=5.0,
        ))
        assert r.prediction == 1
        assert r.lp_recommended is True

    def test_age_7d_out_of_range(self):
        """7-day-old → below AAP range (8–60d)."""
        r = aap_2021.apply(aap_2021.AAP2021Inputs(
            age_days=7, temp_c=38.5, well_appearing=True,
            ua_le_positive=False, pct=0.1, anc=2.0, crp=5.0,
        ))
        assert r.applicable is False

    def test_anc_at_boundary(self):
        """ANC exactly 4.0 (≥4.0) → not low-risk."""
        r = aap_2021.apply(aap_2021.AAP2021Inputs(
            age_days=45, temp_c=38.5, well_appearing=True,
            ua_le_positive=False, ua_nitrites_positive=False,
            pct=0.2, anc=4.0, crp=8.0,
        ))
        assert r.prediction == 1
        assert "anc_elevated" in r.triggered_criteria


# ── NICE NG143 ─────────────────────────────────────────────────

class TestNICENG143:
    def test_all_febrile_under_3mo_not_low_risk(self):
        """Any febrile infant <3mo → not low-risk (no NICE low-risk pathway)."""
        r = nice_ng143.apply(nice_ng143.NICENG143Inputs(
            age_days=45, temp_c=38.5, well_appearing=True,
        ))
        assert r.prediction == 1

    def test_red_features(self):
        """Seizures → red traffic light."""
        r = nice_ng143.apply(nice_ng143.NICENG143Inputs(
            age_days=30, temp_c=39.0, well_appearing=True,
            seizures=True,
        ))
        assert r.prediction == 1
        assert r.traffic_light == "red"
        assert "seizures" in r.triggered_criteria

    def test_wbc_abnormal_amber(self):
        """WBC 18K → amber feature."""
        r = nice_ng143.apply(nice_ng143.NICENG143Inputs(
            age_days=30, temp_c=38.5, well_appearing=True,
            wbc=18.0,
        ))
        assert r.prediction == 1
