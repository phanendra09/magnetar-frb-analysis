import json
from pathlib import Path

from pipeline.stats import (
    FRBAnomalyResult,
    PowerLawFitResult,
    SOCResult,
    WaitingTimeFitResult,
    sample_powerlaw_alpha_posterior,
    save_results,
)
from run_analysis import discover_event_files


def test_discover_event_files_includes_evt_gz(tmp_path):
    filenames = [
        "a.fits",
        "b.evt",
        "c.fits.gz",
        "d.evt.gz",
        "ignore.txt",
    ]
    for name in filenames:
        (tmp_path / name).write_text("x", encoding="utf-8")

    discovered = [path.name for path in discover_event_files(tmp_path)]
    assert discovered == ["a.fits", "b.evt", "c.fits.gz", "d.evt.gz"]


def test_save_results_preserves_boolean_types(tmp_path):
    output_path = tmp_path / "fit_results.json"
    save_results(
        energy_fit=PowerLawFitResult(
            alpha=1.9,
            alpha_err=0.1,
            xmin=1.0,
            xmax=10.0,
            n_tail=5,
            ks_statistic=None,
            ks_pvalue=None,
            loglikelihood_ratio={},
        ),
        waiting_fit=WaitingTimeFitResult(
            best_model="lognormal",
            coefficient_of_variation=2.0,
            is_clustered=True,
            exponential_rate=0.5,
            weibull_shape=0.8,
            weibull_scale=2.0,
            lognormal_shape=1.1,
            lognormal_scale=3.0,
            model_comparison={"lognormal": 1.0},
        ),
        frb_result=FRBAnomalyResult(
            frb_energy=7.0,
            frb_duration=0.5,
            frb_waiting_time=None,
            energy_percentile=100.0,
            duration_percentile=40.0,
            waiting_time_percentile=None,
            energy_zscore=3.6,
            duration_zscore=0.1,
            is_energy_outlier=True,
            is_duration_outlier=False,
            conclusion="ok",
        ),
        soc_result=SOCResult(
            alpha=1.9,
            alpha_err=0.1,
            beta=2.8,
            beta_err=0.2,
            delta=0.9,
            delta_err=0.1,
            predicted_alpha=3.0,
            predicted_alpha_err=0.3,
            consistency=2.8,
            is_consistent=False,
            conclusion="not consistent",
        ),
        output_path=str(output_path),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["waiting_time_analysis"]["is_clustered"] is True
    assert payload["frb_anomaly_test"]["is_energy_outlier"] is True
    assert payload["soc_consistency"]["is_consistent"] is False


def test_powerlaw_alpha_mcmc_runs():
    samples = [1.0, 1.2, 1.4, 1.8, 2.2, 2.8, 3.5, 4.0, 5.0]
    result = sample_powerlaw_alpha_posterior(
        samples,
        xmin=1.0,
        n_walkers=12,
        n_steps=120,
        burn_in=20,
        random_seed=1,
    )
    assert result.n_samples > 0
    assert result.alpha_median > 1.0
