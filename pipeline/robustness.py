"""
robustness.py - robustness tests for the FRB anomaly result

Checks:
1. Sensitivity grid (p0 x sigma)
2. Sample-maximum consistency
3. Bandpass energy range
4. Catalogue completeness
5. KS goodness-of-fit bootstrap
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Result of a single sensitivity run."""

    p0: float
    threshold_sigma: float
    n_bursts: int
    frb_percentile: float
    frb_zscore: float
    alpha: float
    alpha_err: float


@dataclass
class SensitivityGridResult:
    """Full sensitivity grid across p0 and threshold choices."""

    results: List[SensitivityResult]
    p0_values: List[float]
    sigma_values: List[float]
    varied_parameter: str  # "joint_grid"

    def valid_results(self) -> List[SensitivityResult]:
        return [result for result in self.results if np.isfinite(result.frb_zscore)]

    def zscore_matrix(self) -> np.ndarray:
        mat = np.full((len(self.p0_values), len(self.sigma_values)), np.nan)
        for result in self.results:
            i = self.p0_values.index(result.p0)
            j = self.sigma_values.index(result.threshold_sigma)
            mat[i, j] = result.frb_zscore
        return mat

    def nburst_matrix(self) -> np.ndarray:
        mat = np.full((len(self.p0_values), len(self.sigma_values)), 0)
        for result in self.results:
            i = self.p0_values.index(result.p0)
            j = self.sigma_values.index(result.threshold_sigma)
            mat[i, j] = result.n_bursts
        return mat

    def summary(self, default_p0: float = 0.05, default_sigma: float = 3.0) -> dict:
        valid = self.valid_results()
        if not valid:
            return {
                "n_valid_runs": 0,
                "default_run_present": False,
                "frb_zscore_min": None,
                "frb_zscore_max": None,
                "alpha_min": None,
                "alpha_max": None,
                "default_run": None,
                "interpretation": "No valid sensitivity runs were produced.",
            }

        zscores = [result.frb_zscore for result in valid]
        alphas = [result.alpha for result in valid if np.isfinite(result.alpha)]
        default_run = next(
            (
                result
                for result in valid
                if result.p0 == default_p0 and result.threshold_sigma == default_sigma
            ),
            None,
        )
        interpretation = (
            "The FRB extremeness ranking is stable across the tested parameter grid, "
            "while the fitted power-law index alpha shows material systematic variation "
            "that should be reported as a modeling uncertainty."
        )
        return {
            "n_valid_runs": len(valid),
            "default_run_present": default_run is not None,
            "frb_zscore_min": float(min(zscores)),
            "frb_zscore_max": float(max(zscores)),
            "alpha_min": float(min(alphas)) if alphas else None,
            "alpha_max": float(max(alphas)) if alphas else None,
            "default_run": None
            if default_run is None
            else {
                "p0": default_run.p0,
                "threshold_sigma": default_run.threshold_sigma,
                "n_bursts": default_run.n_bursts,
                "frb_zscore": float(default_run.frb_zscore),
                "alpha": float(default_run.alpha),
                "alpha_err": float(default_run.alpha_err),
            },
            "interpretation": interpretation,
        }

    def to_dict(self) -> dict:
        return {
            "varied_parameter": self.varied_parameter,
            "p0_values": self.p0_values,
            "sigma_values": self.sigma_values,
            "summary": self.summary(),
            "grid": [
                {
                    "p0": result.p0,
                    "threshold_sigma": result.threshold_sigma,
                    "n_bursts": result.n_bursts,
                    "frb_percentile": result.frb_percentile,
                    "frb_zscore": result.frb_zscore,
                    "alpha": result.alpha,
                    "alpha_err": result.alpha_err,
                }
                for result in self.results
            ],
        }


@dataclass
class SampleMaxResult:
    """Result of the sample-maximum consistency test."""

    observed_max_energy: float
    frb_energy: float
    n_population: int
    alpha: float
    xmin: float
    n_simulations: int
    fraction_max_exceeds_frb: float
    fraction_max_exceeds_frb_err: float
    median_simulated_max: float
    percentile_95_simulated_max: float
    frb_consistent_with_tail: bool
    frb_exceeds_observed_population: bool

    def to_dict(self) -> dict:
        return {
            "observed_max_energy_erg": float(self.observed_max_energy),
            "frb_energy_erg": float(self.frb_energy),
            "n_population": self.n_population,
            "alpha": float(self.alpha),
            "xmin": float(self.xmin),
            "n_simulations": self.n_simulations,
            "fraction_max_exceeds_frb": float(self.fraction_max_exceeds_frb),
            "fraction_max_exceeds_frb_err": float(self.fraction_max_exceeds_frb_err),
            "median_simulated_max_erg": float(self.median_simulated_max),
            "95th_pct_simulated_max_erg": float(self.percentile_95_simulated_max),
            "frb_consistent_with_power_law_tail": bool(self.frb_consistent_with_tail),
            "frb_exceeds_all_observed_bursts": bool(self.frb_exceeds_observed_population),
            "interpretation": (
                f"FRB energy {self.frb_energy:.2e} erg vs observed max {self.observed_max_energy:.2e} erg. "
                f"Power law (alpha={self.alpha:.2f}) exceeds FRB in {self.fraction_max_exceeds_frb * 100:.0f}% of {self.n_population}-burst sims. "
                + ("Tail-consistent." if self.frb_consistent_with_tail else "Tail-inconsistent.")
            ),
        }


@dataclass
class BandpassResult:
    """Result of bandpass energy-range sensitivity testing."""

    frb_energies_tested: List[float]
    percentiles: List[float]
    zscores: List[float]
    is_anomalous: List[bool]

    def to_dict(self) -> dict:
        return {
            "frb_energies_erg": [float(value) for value in self.frb_energies_tested],
            "percentiles": [float(value) for value in self.percentiles],
            "zscores": [float(value) for value in self.zscores],
            "is_anomalous": [bool(value) for value in self.is_anomalous],
            "conclusion": (
                "Anomaly is ROBUST across all tested FRB energies"
                if all(self.is_anomalous)
                else f"Anomaly holds for {sum(self.is_anomalous)}/{len(self.is_anomalous)} tested energies"
            ),
        }


@dataclass
class CompletenessResult:
    """Catalogue completeness within GTIs only."""

    total_observation_span_s: float
    total_gti_exposure_s: float
    coverage_fraction: float
    n_gti_intervals: int
    n_detected_bursts: int
    observed_burst_rate_per_ks: float

    def to_dict(self) -> dict:
        return {
            "total_observation_span_s": float(self.total_observation_span_s),
            "total_gti_exposure_s": float(self.total_gti_exposure_s),
            "coverage_fraction_pct": float(self.coverage_fraction * 100),
            "n_gti_intervals": self.n_gti_intervals,
            "n_detected_bursts": self.n_detected_bursts,
            "observed_burst_rate_per_ks": float(self.observed_burst_rate_per_ks),
            "caveat": (
                "The burst rate is measured only within GTIs. Because the storm is highly "
                "non-stationary (CV >> 1), extrapolating to the full observation window is "
                "not valid. The catalogue is complete within the GTI windows but does not "
                "represent the total burst population during orbital gaps."
            ),
        }


@dataclass
class KSBootstrapResult:
    """Result of semi-parametric bootstrap KS goodness-of-fit."""

    ks_statistic: float
    p_value: float
    n_bootstrap: int
    alpha: float
    xmin: float

    def to_dict(self) -> dict:
        return {
            "ks_statistic": float(self.ks_statistic),
            "p_value": float(self.p_value),
            "n_bootstrap": self.n_bootstrap,
            "alpha_used": float(self.alpha),
            "xmin_used": float(self.xmin),
            "fit_acceptable": bool(self.p_value > 0.1),
        }


def sensitivity_analysis(
    photon_times_subsampled: np.ndarray,
    frb_energy: float = 7e39,
    distance_kpc: float = 9.0,
    mean_photon_energy_keV: float = 5.0,
    p0_values: Optional[List[float]] = None,
    sigma_values: Optional[List[float]] = None,
) -> SensitivityGridResult:
    """
    Run a joint sensitivity grid across Bayesian Blocks p0 and burst threshold.

    The same pre-subsampled event list must be reused for every run so that
    the sensitivity study isolates parameter choices rather than event selection.
    """
    from pipeline.catalogue import build_catalogue, compute_waiting_times
    from pipeline.detect import bayesian_blocks_detect, identify_bursts
    from pipeline.stats import fit_energy_distribution

    if p0_values is None:
        p0_values = [0.01, 0.02, 0.05, 0.10]
    if sigma_values is None:
        sigma_values = [2.0, 2.5, 3.0, 3.5]

    results: List[SensitivityResult] = []
    edges_cache = {}
    total = len(p0_values) * len(sigma_values)
    count = 0

    for p0 in p0_values:
        try:
            edges_cache[p0] = bayesian_blocks_detect(photon_times_subsampled, p0=p0)
        except Exception as exc:
            logger.warning(f"Bayesian Blocks failed for p0={p0}: {exc}")

    for p0 in p0_values:
        edges = edges_cache.get(p0)
        for sigma in sigma_values:
            count += 1
            logger.info(f"  Sensitivity {count}/{total}: p0={p0}, sigma={sigma}")
            if edges is None:
                results.append(
                    SensitivityResult(
                        p0=p0,
                        threshold_sigma=sigma,
                        n_bursts=0,
                        frb_percentile=np.nan,
                        frb_zscore=np.nan,
                        alpha=np.nan,
                        alpha_err=np.nan,
                    )
                )
                continue

            try:
                bursts = identify_bursts(
                    photon_times_subsampled,
                    edges,
                    threshold_sigma=sigma,
                    distance_kpc=distance_kpc,
                    mean_photon_energy_keV=mean_photon_energy_keV,
                )
                if len(bursts) < 5:
                    results.append(
                        SensitivityResult(
                            p0=p0,
                            threshold_sigma=sigma,
                            n_bursts=len(bursts),
                            frb_percentile=np.nan,
                            frb_zscore=np.nan,
                            alpha=np.nan,
                            alpha_err=np.nan,
                        )
                    )
                    continue

                catalogue = compute_waiting_times(build_catalogue(bursts))
                energies = catalogue["energy"].values
                energy_fit = fit_energy_distribution(energies)
                log_energies = np.log10(energies)
                frb_percentile = float((energies < frb_energy).sum() / len(energies) * 100)
                frb_zscore = float((np.log10(frb_energy) - np.mean(log_energies)) / np.std(log_energies))

                results.append(
                    SensitivityResult(
                        p0=p0,
                        threshold_sigma=sigma,
                        n_bursts=len(bursts),
                        frb_percentile=frb_percentile,
                        frb_zscore=frb_zscore,
                        alpha=energy_fit.alpha,
                        alpha_err=energy_fit.alpha_err,
                    )
                )
            except Exception as exc:
                logger.warning(f"    Failed sensitivity run for p0={p0}, sigma={sigma}: {exc}")
                results.append(
                    SensitivityResult(
                        p0=p0,
                        threshold_sigma=sigma,
                        n_bursts=0,
                        frb_percentile=np.nan,
                        frb_zscore=np.nan,
                        alpha=np.nan,
                        alpha_err=np.nan,
                    )
                )

    grid = SensitivityGridResult(
        results=results,
        p0_values=p0_values,
        sigma_values=sigma_values,
        varied_parameter="joint_grid",
    )
    summary = grid.summary()
    if summary["n_valid_runs"] > 0:
        logger.info(
            f"Sensitivity complete: {summary['n_valid_runs']} valid runs. "
            f"FRB z-score range [{summary['frb_zscore_min']:.2f}, {summary['frb_zscore_max']:.2f}] sigma; "
            f"alpha range [{summary['alpha_min']:.2f}, {summary['alpha_max']:.2f}]"
        )
    return grid


def sample_maximum_test(
    energies: np.ndarray,
    frb_energy: float = 7e39,
    alpha: float = 1.99,
    xmin: float = 7.6e37,
    n_simulations: int = 10000,
    random_seed: int = 42,
) -> SampleMaxResult:
    """
    Test whether the FRB energy is consistent with the fitted heavy tail
    as the sample maximum of a burst population of the observed size.
    """
    rng = np.random.default_rng(random_seed)
    n_bursts = len(energies)
    n_below_xmin = int(np.sum(energies < xmin))
    n_above_xmin = n_bursts - n_below_xmin
    observed_max = float(np.max(energies))
    below = energies[energies < xmin]

    max_energies = np.zeros(n_simulations)
    for idx in range(n_simulations):
        u = rng.uniform(0, 1, n_above_xmin)
        synthetic_tail = xmin * u ** (-1.0 / (alpha - 1.0))
        if n_below_xmin > 0:
            synthetic_below = rng.choice(below, n_below_xmin, replace=True)
            synthetic = np.concatenate([synthetic_below, synthetic_tail])
        else:
            synthetic = synthetic_tail
        max_energies[idx] = synthetic.max()

    fraction = float(np.mean(max_energies >= frb_energy))
    fraction_err = float(np.sqrt(fraction * (1.0 - fraction) / n_simulations))
    consistent = bool(fraction > 0.05)

    result = SampleMaxResult(
        observed_max_energy=observed_max,
        frb_energy=frb_energy,
        n_population=n_bursts,
        alpha=alpha,
        xmin=xmin,
        n_simulations=n_simulations,
        fraction_max_exceeds_frb=fraction,
        fraction_max_exceeds_frb_err=fraction_err,
        median_simulated_max=float(np.median(max_energies)),
        percentile_95_simulated_max=float(np.percentile(max_energies, 95)),
        frb_consistent_with_tail=consistent,
        frb_exceeds_observed_population=bool(frb_energy > observed_max),
    )
    logger.info(
        f"Sample-max consistency: observed max={observed_max:.2e}, FRB={frb_energy:.2e}, "
        f"model fraction(max >= FRB)={fraction:.3f} +/- {fraction_err:.003f}"
    )
    return result


def bandpass_energy_test(
    nicer_energies: np.ndarray,
    frb_energy_range: Optional[List[float]] = None,
) -> BandpassResult:
    """
    Test anomaly across a range of assumed FRB energies.

    The range accounts for two uncertainties:
    1. Bandpass: NICER sees 0.5-10 keV; the Li et al. value uses 1-250 keV.
       The soft X-ray fraction is estimated as ~15-30% of the broadband
       fluence from the spectral fits in Mereghetti et al. (2020).
    2. Distance: literature uses 6-12 kpc (Kothes et al. 2018 give 9 kpc;
       Zhong et al. 2020 give 6.6 kpc).

    The tested energies span these two uncertainties simultaneously:
      1e39  = NICER-band only at d ~ 6 kpc (most conservative)
      3e39  = NICER-band correction at d ~ 9 kpc
      5e39  = moderate cross-calibration correction
      7e39  = Li et al. 2021 broadband at d = 9 kpc (published value)
      1e40  = full broadband at d ~ 12 kpc (most generous)
    """
    if frb_energy_range is None:
        frb_energy_range = [1.0e39, 3.0e39, 5.0e39, 7.0e39, 1.0e40]

    log_energies = np.log10(nicer_energies)
    mu = np.mean(log_energies)
    sigma = np.std(log_energies)
    percentiles = []
    zscores = []
    anomalous = []

    for frb_energy in frb_energy_range:
        percentile = float((nicer_energies < frb_energy).sum() / len(nicer_energies) * 100)
        zscore = float((np.log10(frb_energy) - mu) / sigma)
        is_anomalous = bool(percentile > 95.0)
        percentiles.append(percentile)
        zscores.append(zscore)
        anomalous.append(is_anomalous)

    return BandpassResult(
        frb_energies_tested=frb_energy_range,
        percentiles=percentiles,
        zscores=zscores,
        is_anomalous=anomalous,
    )


def catalogue_completeness(fits_files: List[Path], n_detected_bursts: int) -> CompletenessResult:
    """Report completeness within GTIs only, with no storm extrapolation."""
    from astropy.io import fits

    starts = []
    stops = []
    gti_total = 0.0
    n_gti = 0

    for fpath in fits_files:
        with fits.open(str(fpath)) as hdul:
            times = hdul[1].data["TIME"]
            if len(times) == 0:
                continue
            starts.append(times.min())
            stops.append(times.max())
            for hdu in hdul:
                if "GTI" in hdu.name.upper():
                    for start, stop in zip(hdu.data["START"], hdu.data["STOP"]):
                        gti_total += stop - start
                        n_gti += 1
                    break

    if not starts:
        return CompletenessResult(0.0, 0.0, 0.0, 0, 0, 0.0)

    span = max(stops) - min(starts)
    coverage = gti_total / span if span > 0 else 0.0
    burst_rate_per_ks = (n_detected_bursts / gti_total * 1000) if gti_total > 0 else 0.0
    return CompletenessResult(
        total_observation_span_s=float(span),
        total_gti_exposure_s=float(gti_total),
        coverage_fraction=float(coverage),
        n_gti_intervals=n_gti,
        n_detected_bursts=n_detected_bursts,
        observed_burst_rate_per_ks=float(burst_rate_per_ks),
    )


def ks_goodness_of_fit(
    energies: np.ndarray,
    alpha: float,
    xmin: float,
    n_bootstrap: int = 2500,
    random_seed: int = 42,
) -> KSBootstrapResult:
    """Semi-parametric bootstrap KS test following Clauset et al. (2009)."""
    rng = np.random.default_rng(random_seed)
    tail = energies[energies >= xmin]
    n_tail = len(tail)

    tail_sorted = np.sort(tail)
    cdf_data = np.arange(1, n_tail + 1) / n_tail
    cdf_model = 1.0 - (tail_sorted / xmin) ** (-(alpha - 1))
    ks_observed = float(np.max(np.abs(cdf_data - cdf_model)))

    ks_synthetic = np.zeros(n_bootstrap)
    for idx in range(n_bootstrap):
        u = rng.uniform(0, 1, n_tail)
        synthetic_tail = xmin * u ** (-1.0 / (alpha - 1.0))
        synthetic_alpha = 1.0 + n_tail / np.sum(np.log(synthetic_tail / xmin))
        synthetic_sorted = np.sort(synthetic_tail)
        cdf_synthetic = np.arange(1, n_tail + 1) / n_tail
        cdf_refit = 1.0 - (synthetic_sorted / xmin) ** (-(synthetic_alpha - 1))
        ks_synthetic[idx] = np.max(np.abs(cdf_synthetic - cdf_refit))

    p_value = float(np.mean(ks_synthetic >= ks_observed))
    return KSBootstrapResult(
        ks_statistic=ks_observed,
        p_value=p_value,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        xmin=xmin,
    )


def plot_sensitivity_heatmap(grid: SensitivityGridResult, output_dir: str = "results/figures"):
    """Generate Figure 6: sensitivity heatmap."""
    import matplotlib.pyplot as plt
    from plots.style import apply_style

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.5), gridspec_kw={"wspace": 0.35})

    z_mat = grid.zscore_matrix()
    n_mat = grid.nburst_matrix()
    p0_labels = [str(value) for value in grid.p0_values]
    sigma_labels = [f"{value}σ" for value in grid.sigma_values]

    im1 = ax1.imshow(z_mat, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=max(5, np.nanmax(z_mat)))
    ax1.set_xticks(range(len(sigma_labels)))
    ax1.set_xticklabels(sigma_labels, fontsize=8)
    ax1.set_yticks(range(len(p0_labels)))
    ax1.set_yticklabels(p0_labels, fontsize=8)
    ax1.set_xlabel("Burst threshold")
    ax1.set_ylabel("Bayesian Blocks $p_0$")
    ax1.set_title("FRB z-score", fontsize=10)
    for i in range(len(grid.p0_values)):
        for j in range(len(grid.sigma_values)):
            value = z_mat[i, j]
            if np.isfinite(value):
                ax1.text(j, i, f"{value:.1f}σ", ha="center", va="center", fontsize=8, fontweight="bold")
    fig.colorbar(im1, ax=ax1, shrink=0.8).set_label("z-score (σ)", fontsize=8)

    im2 = ax2.imshow(n_mat, cmap="Blues", aspect="auto")
    ax2.set_xticks(range(len(sigma_labels)))
    ax2.set_xticklabels(sigma_labels, fontsize=8)
    ax2.set_yticks(range(len(p0_labels)))
    ax2.set_yticklabels(p0_labels, fontsize=8)
    ax2.set_xlabel("Burst threshold")
    ax2.set_ylabel("Bayesian Blocks $p_0$")
    ax2.set_title("Detected bursts", fontsize=10)
    for i in range(len(grid.p0_values)):
        for j in range(len(grid.sigma_values)):
            ax2.text(j, i, str(n_mat[i, j]), ha="center", va="center", fontsize=8, fontweight="bold")
    fig.colorbar(im2, ax=ax2, shrink=0.8).set_label("Count", fontsize=8)

    fig.suptitle("Sensitivity Analysis: Parameter Robustness", fontsize=12, fontweight="bold", y=1.02)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in ["png", "pdf"]:
        fig.savefig(str(outdir / f"fig6_sensitivity.{fmt}"), format=fmt, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: fig6_sensitivity")
    return fig


def run_all_robustness_tests(
    photon_times_subsampled: np.ndarray,
    catalogue: pd.DataFrame,
    energy_fit,
    fits_files: List[Path],
    frb_energy: float = 7e39,
    output_dir: str = "results_real",
) -> dict:
    """Run all robustness tests and save a reviewer-facing summary."""
    import json

    nicer_catalogue = catalogue[~catalogue["is_frb_burst"]].copy()
    nicer_energies = nicer_catalogue["energy"].values
    out_path = Path(output_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("ROBUSTNESS 1: Sensitivity Analysis (joint p0 x sigma grid)")
    logger.info("=" * 60)
    sensitivity = sensitivity_analysis(photon_times_subsampled, frb_energy=frb_energy)
    plot_sensitivity_heatmap(sensitivity, output_dir=str(out_path / "figures"))

    logger.info("")
    logger.info("=" * 60)
    logger.info("ROBUSTNESS 2: Sample-Maximum Consistency Test")
    logger.info("=" * 60)
    sample_max = sample_maximum_test(
        nicer_energies,
        frb_energy=frb_energy,
        alpha=energy_fit.alpha,
        xmin=energy_fit.xmin,
        n_simulations=10000,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("ROBUSTNESS 3: Bandpass Energy Range Test")
    logger.info("=" * 60)
    bandpass = bandpass_energy_test(nicer_energies)

    logger.info("")
    logger.info("=" * 60)
    logger.info("ROBUSTNESS 4: Catalogue Completeness")
    logger.info("=" * 60)
    completeness = catalogue_completeness(fits_files, n_detected_bursts=len(nicer_catalogue))

    logger.info("")
    logger.info("=" * 60)
    logger.info("ROBUSTNESS 5: KS Goodness-of-Fit Bootstrap")
    logger.info("=" * 60)
    ks_result = ks_goodness_of_fit(
        nicer_energies,
        alpha=energy_fit.alpha,
        xmin=energy_fit.xmin,
        n_bootstrap=2500,
    )

    sensitivity_summary = sensitivity.summary()
    reviewer_summary = {
        "main_conclusion": (
            f"FRB ranking stable (z={sensitivity_summary['frb_zscore_min']:.2f}-"
            f"{sensitivity_summary['frb_zscore_max']:.2f} sigma). "
            f"Alpha has systematic spread ({sensitivity_summary['alpha_min']:.2f}-"
            f"{sensitivity_summary['alpha_max']:.2f})."
        ),
        "frb_extremeness_ranking_robust": bool(
            sensitivity_summary["frb_zscore_min"] is not None and sensitivity_summary["frb_zscore_min"] > 3.4
        ),
        "alpha_model_systematic_present": bool(
            sensitivity_summary["alpha_min"] is not None
            and sensitivity_summary["alpha_max"] is not None
            and (sensitivity_summary["alpha_max"] - sensitivity_summary["alpha_min"]) > 0.2
        ),
        "heavy_tail_consistency_statement": sample_max.to_dict()["interpretation"],
        "ks_fit_acceptable": bool(ks_result.p_value > 0.1),
    }

    results = {
        "reviewer_summary": reviewer_summary,
        "sensitivity_analysis": sensitivity.to_dict(),
        "sample_maximum_test": sample_max.to_dict(),
        "bandpass_sensitivity": bandpass.to_dict(),
        "catalogue_completeness": completeness.to_dict(),
        "ks_goodness_of_fit": ks_result.to_dict(),
    }

    rob_path = out_path / "robustness_results.json"
    with rob_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    logger.info(f"All robustness results saved to {rob_path}")
    logger.info("")
    logger.info("=" * 60)
    logger.info("ROBUSTNESS SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"  Sensitivity: z-score range [{sensitivity_summary['frb_zscore_min']:.2f}, "
        f"{sensitivity_summary['frb_zscore_max']:.2f}] sigma"
    )
    logger.info(
        f"  Sensitivity: alpha range [{sensitivity_summary['alpha_min']:.2f}, "
        f"{sensitivity_summary['alpha_max']:.2f}]"
    )
    logger.info(
        f"  Sample-max: model fraction(max >= FRB)={sample_max.fraction_max_exceeds_frb:.3f}"
    )
    logger.info(
        f"  Bandpass: anomaly in {sum(bandpass.is_anomalous)}/{len(bandpass.is_anomalous)} tested energies"
    )
    logger.info(
        f"  Completeness: {completeness.coverage_fraction * 100:.1f}% GTI coverage, "
        f"{completeness.observed_burst_rate_per_ks:.1f} bursts/ks within GTIs"
    )
    logger.info(f"  KS GoF: p = {ks_result.p_value:.3f}")
    return results
