"""
Tests for the robustness module.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.robustness import (
    SensitivityGridResult,
    SensitivityResult,
    SampleMaxResult,
    BandpassResult,
    CompletenessResult,
    KSBootstrapResult,
    sample_maximum_test,
    bandpass_energy_test,
    ks_goodness_of_fit,
)


# ============================================================
# Sample-Maximum Consistency Test
# ============================================================

class TestSampleMaximumTest:
    """Tests for the sample-maximum consistency test."""

    def _make_power_law_sample(self, alpha=2.0, xmin=1.0, n=200, seed=0):
        rng = np.random.default_rng(seed)
        u = rng.uniform(0, 1, n)
        return xmin * u ** (-1.0 / (alpha - 1.0))

    def test_returns_correct_type(self):
        energies = self._make_power_law_sample()
        result = sample_maximum_test(
            energies, frb_energy=100.0, alpha=2.0, xmin=1.0,
            n_simulations=500,
        )
        assert isinstance(result, SampleMaxResult)

    def test_consistent_frb_within_tail(self):
        """FRB energy near the sample max should be consistent."""
        energies = self._make_power_law_sample(alpha=2.0, n=200)
        frb_energy = np.max(energies) * 1.1  # just above observed max
        result = sample_maximum_test(
            energies, frb_energy=frb_energy, alpha=2.0, xmin=1.0,
            n_simulations=2000,
        )
        # The model should easily produce maxima this large
        assert result.fraction_max_exceeds_frb > 0.05
        assert result.frb_consistent_with_tail is True

    def test_inconsistent_frb_far_above_tail(self):
        """FRB energy vastly above all samples should be inconsistent."""
        energies = self._make_power_law_sample(alpha=3.0, n=200, seed=1)
        # Set FRB energy absurdly high — steeper alpha means less tail
        frb_energy = np.max(energies) * 1e6
        result = sample_maximum_test(
            energies, frb_energy=frb_energy, alpha=3.0, xmin=1.0,
            n_simulations=2000,
        )
        assert result.fraction_max_exceeds_frb < 0.05
        assert result.frb_consistent_with_tail is False

    def test_exceeds_observed_population_flag(self):
        energies = self._make_power_law_sample()
        result = sample_maximum_test(
            energies, frb_energy=np.max(energies) * 2, alpha=2.0, xmin=1.0,
            n_simulations=500,
        )
        assert result.frb_exceeds_observed_population is True

    def test_to_dict_has_interpretation(self):
        energies = self._make_power_law_sample()
        result = sample_maximum_test(
            energies, frb_energy=10.0, alpha=2.0, xmin=1.0,
            n_simulations=500,
        )
        d = result.to_dict()
        assert "interpretation" in d
        assert isinstance(d["interpretation"], str)
        assert len(d["interpretation"]) > 0
        assert "requires sampling" not in d["interpretation"]
        assert "Tail-consistent" in d["interpretation"]


# ============================================================
# Bandpass Energy Test
# ============================================================

class TestBandpassEnergyTest:

    def test_returns_correct_type(self):
        energies = np.logspace(36, 39, 100)
        result = bandpass_energy_test(energies)
        assert isinstance(result, BandpassResult)

    def test_all_anomalous_when_frb_much_higher(self):
        energies = np.logspace(36, 38, 100)  # max = 1e38
        frb_range = [1e39, 5e39, 1e40]  # all well above population
        result = bandpass_energy_test(energies, frb_energy_range=frb_range)
        assert all(result.is_anomalous)
        assert len(result.percentiles) == 3

    def test_none_anomalous_when_frb_within_population(self):
        energies = np.logspace(36, 40, 200)
        frb_range = [1e37]  # below median
        result = bandpass_energy_test(energies, frb_energy_range=frb_range)
        assert not any(result.is_anomalous)

    def test_to_dict_conclusion(self):
        energies = np.logspace(36, 38, 50)
        result = bandpass_energy_test(
            energies, frb_energy_range=[1e39, 1e40]
        )
        d = result.to_dict()
        assert "conclusion" in d


# ============================================================
# KS Goodness-of-Fit
# ============================================================

class TestKSGoodnessOfFit:

    def _make_power_law_sample(self, alpha=2.0, xmin=1.0, n=200, seed=0):
        rng = np.random.default_rng(seed)
        u = rng.uniform(0, 1, n)
        return xmin * u ** (-1.0 / (alpha - 1.0))

    def test_returns_correct_type(self):
        energies = self._make_power_law_sample()
        result = ks_goodness_of_fit(
            energies, alpha=2.0, xmin=1.0, n_bootstrap=200,
        )
        assert isinstance(result, KSBootstrapResult)

    def test_good_fit_has_high_p_value(self):
        """Data drawn from a power law should pass the KS test."""
        energies = self._make_power_law_sample(alpha=2.0, n=300, seed=7)
        mle_alpha = 1.0 + len(energies) / np.sum(np.log(energies / 1.0))
        result = ks_goodness_of_fit(
            energies, alpha=mle_alpha, xmin=1.0, n_bootstrap=500,
        )
        assert result.p_value > 0.1, (
            f"Data from power law should pass KS test, got p={result.p_value}"
        )

    def test_bad_fit_has_low_p_value(self):
        """Non-power-law data with wrong alpha should fail."""
        # Uniform data is NOT a power law
        rng = np.random.default_rng(42)
        energies = rng.uniform(1.0, 2.0, 300)
        result = ks_goodness_of_fit(
            energies, alpha=3.0, xmin=1.0, n_bootstrap=500,
        )
        assert result.p_value < 0.3  # should be quite low

    def test_to_dict_has_acceptable_flag(self):
        energies = self._make_power_law_sample()
        result = ks_goodness_of_fit(
            energies, alpha=2.0, xmin=1.0, n_bootstrap=100,
        )
        d = result.to_dict()
        assert "fit_acceptable" in d
        assert isinstance(d["fit_acceptable"], bool)


# ============================================================
# Completeness (no-FITS, just data class)
# ============================================================

class TestCompletenessResult:

    def test_to_dict_contains_caveat(self):
        result = CompletenessResult(
            total_observation_span_s=100000.0,
            total_gti_exposure_s=5000.0,
            coverage_fraction=0.05,
            n_gti_intervals=8,
            n_detected_bursts=50,
            observed_burst_rate_per_ks=10.0,
        )
        d = result.to_dict()
        assert "caveat" in d
        assert "non-stationary" in d["caveat"]

    def test_no_extrapolation_in_dict(self):
        result = CompletenessResult(
            total_observation_span_s=100000.0,
            total_gti_exposure_s=5000.0,
            coverage_fraction=0.05,
            n_gti_intervals=8,
            n_detected_bursts=50,
            observed_burst_rate_per_ks=10.0,
        )
        d = result.to_dict()
        # Should NOT have an estimated_total_bursts field
        assert "estimated_total_bursts" not in d


# ============================================================
# SensitivityGridResult
# ============================================================

class TestSensitivityGridResult:

    def test_zscore_matrix_shape(self):
        results = [
            SensitivityResult(p0=0.01, threshold_sigma=2.0, n_bursts=100,
                              frb_percentile=99.0, frb_zscore=3.5,
                              alpha=1.9, alpha_err=0.1),
            SensitivityResult(p0=0.01, threshold_sigma=3.0, n_bursts=80,
                              frb_percentile=99.0, frb_zscore=3.6,
                              alpha=2.0, alpha_err=0.1),
            SensitivityResult(p0=0.05, threshold_sigma=2.0, n_bursts=120,
                              frb_percentile=99.0, frb_zscore=3.4,
                              alpha=1.8, alpha_err=0.1),
            SensitivityResult(p0=0.05, threshold_sigma=3.0, n_bursts=90,
                              frb_percentile=99.0, frb_zscore=3.7,
                              alpha=2.1, alpha_err=0.1),
        ]
        grid = SensitivityGridResult(
            results=results,
            p0_values=[0.01, 0.05],
            sigma_values=[2.0, 3.0],
            varied_parameter="joint_grid",
        )
        mat = grid.zscore_matrix()
        assert mat.shape == (2, 2)
        assert mat[0, 0] == 3.5
        assert mat[1, 1] == 3.7

    def test_to_dict_has_varied_parameter(self):
        grid = SensitivityGridResult(
            results=[], p0_values=[0.05], sigma_values=[3.0],
            varied_parameter="joint_grid",
        )
        d = grid.to_dict()
        assert d["varied_parameter"] == "joint_grid"

    def test_summary_reports_systematic_ranges(self):
        results = [
            SensitivityResult(p0=0.01, threshold_sigma=2.0, n_bursts=100,
                              frb_percentile=99.0, frb_zscore=3.5,
                              alpha=1.7, alpha_err=0.1),
            SensitivityResult(p0=0.05, threshold_sigma=3.0, n_bursts=90,
                              frb_percentile=99.0, frb_zscore=3.7,
                              alpha=2.0, alpha_err=0.1),
        ]
        grid = SensitivityGridResult(
            results=results,
            p0_values=[0.01, 0.05],
            sigma_values=[2.0, 3.0],
            varied_parameter="joint_grid",
        )
        summary = grid.summary()
        assert summary["frb_zscore_min"] == 3.5
        assert summary["frb_zscore_max"] == 3.7
        assert summary["alpha_min"] == 1.7
        assert summary["alpha_max"] == 2.0
        assert summary["default_run"]["p0"] == 0.05
        assert summary["default_run"]["threshold_sigma"] == 3.0
        assert "systematic variation" in summary["interpretation"]
