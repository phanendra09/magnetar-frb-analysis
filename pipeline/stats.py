"""
stats.py - statistical analysis

Power-law MLE, waiting times, FRB anomaly test, SOC consistency.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PowerLawFitResult:
    """Result of a power-law MLE fit."""

    alpha: float
    alpha_err: float
    xmin: float
    xmax: float
    n_tail: int
    ks_statistic: Optional[float]
    ks_pvalue: Optional[float]
    loglikelihood_ratio: dict
    distribution_name: str = "power_law"

    def to_dict(self) -> dict:
        return {
            "alpha": float(self.alpha),
            "alpha_err": float(self.alpha_err),
            "xmin": float(self.xmin),
            "xmax": float(self.xmax),
            "n_tail": int(self.n_tail),
            "ks_statistic": None if self.ks_statistic is None else float(self.ks_statistic),
            "ks_pvalue": None if self.ks_pvalue is None else float(self.ks_pvalue),
            "loglikelihood_ratio": self.loglikelihood_ratio,
        }


@dataclass
class PowerLawMCMCResult:
    """Posterior samples for a simple power-law alpha inference."""

    alpha_median: float
    alpha_minus: float
    alpha_plus: float
    samples: np.ndarray
    xmin: float
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "alpha_median": float(self.alpha_median),
            "alpha_minus": float(self.alpha_minus),
            "alpha_plus": float(self.alpha_plus),
            "xmin": float(self.xmin),
            "n_samples": int(self.n_samples),
        }


@dataclass
class WaitingTimeFitResult:
    """Result of waiting time distribution analysis."""

    best_model: str
    coefficient_of_variation: float
    is_clustered: bool
    exponential_rate: float
    weibull_shape: Optional[float]
    weibull_scale: Optional[float]
    lognormal_shape: Optional[float]
    lognormal_scale: Optional[float]
    model_comparison: dict

    def to_dict(self) -> dict:
        return {
            "best_model": self.best_model,
            "coefficient_of_variation": float(self.coefficient_of_variation),
            "is_clustered": bool(self.is_clustered),
            "exponential_rate": float(self.exponential_rate),
            "weibull_shape": None if self.weibull_shape is None else float(self.weibull_shape),
            "weibull_scale": None if self.weibull_scale is None else float(self.weibull_scale),
            "lognormal_shape": None if self.lognormal_shape is None else float(self.lognormal_shape),
            "lognormal_scale": None if self.lognormal_scale is None else float(self.lognormal_scale),
            "model_comparison": self.model_comparison,
        }


@dataclass
class FRBAnomalyResult:
    """Result of FRB burst anomaly analysis."""

    frb_energy: float
    frb_duration: float
    frb_waiting_time: Optional[float]
    energy_percentile: float
    duration_percentile: float
    waiting_time_percentile: Optional[float]
    energy_zscore: float
    duration_zscore: float
    is_energy_outlier: bool
    is_duration_outlier: bool
    conclusion: str

    def to_dict(self) -> dict:
        return {
            "frb_energy_erg": float(self.frb_energy),
            "frb_duration_s": float(self.frb_duration),
            "frb_waiting_time_s": None if self.frb_waiting_time is None else float(self.frb_waiting_time),
            "energy_percentile": float(self.energy_percentile),
            "duration_percentile": float(self.duration_percentile),
            "waiting_time_percentile": None
            if self.waiting_time_percentile is None
            else float(self.waiting_time_percentile),
            "energy_zscore": float(self.energy_zscore),
            "duration_zscore": float(self.duration_zscore),
            "is_energy_outlier": bool(self.is_energy_outlier),
            "is_duration_outlier": bool(self.is_duration_outlier),
            "conclusion": self.conclusion,
        }


@dataclass
class SOCResult:
    """Result of self-organized criticality consistency test."""

    alpha: float
    alpha_err: float
    beta: float
    beta_err: float
    delta: float
    delta_err: float
    predicted_alpha: float
    predicted_alpha_err: float
    consistency: float
    is_consistent: bool
    conclusion: str

    def to_dict(self) -> dict:
        return {
            "alpha": float(self.alpha),
            "alpha_err": float(self.alpha_err),
            "beta": float(self.beta),
            "beta_err": float(self.beta_err),
            "delta": float(self.delta),
            "delta_err": float(self.delta_err),
            "predicted_alpha": float(self.predicted_alpha),
            "predicted_alpha_err": float(self.predicted_alpha_err),
            "consistency_sigma": float(self.consistency),
            "is_consistent": bool(self.is_consistent),
            "conclusion": self.conclusion,
        }


def fit_energy_distribution(
    energies: np.ndarray,
    xmin: Optional[float] = None,
) -> PowerLawFitResult:
    """
    Fit the burst energy distribution using maximum likelihood estimation
    via the powerlaw package.
    """
    import powerlaw

    logger.info(f"Fitting energy distribution: {len(energies)} bursts")

    fit = powerlaw.Fit(energies, xmin=xmin, verbose=False) if xmin is not None else powerlaw.Fit(
        energies, verbose=False
    )

    alpha = fit.power_law.alpha
    sigma = fit.power_law.sigma
    xmin_fit = fit.power_law.xmin

    try:
        ks_stat = float(fit.power_law.KS())
    except (NameError, AttributeError):
        ks_stat = None
        logger.warning("KS statistic computation failed (powerlaw pkg bug), leaving unset")

    ks_pvalue = None

    comparisons = {}
    for alt_model in ["lognormal", "exponential", "truncated_power_law"]:
        try:
            ratio, p_value = fit.distribution_compare("power_law", alt_model)
            comparisons[alt_model] = {
                "loglikelihood_ratio": float(ratio),
                "p_value": float(p_value),
                "preferred": "power_law" if ratio > 0 else alt_model,
            }
            logger.info(
                f"  power_law vs {alt_model}: R={ratio:.3f}, p={p_value:.4f} -> "
                f"{'power_law' if ratio > 0 else alt_model}"
            )
        except Exception as exc:
            comparisons[alt_model] = {"error": str(exc)}

    n_tail = int(np.sum(energies >= xmin_fit))

    result = PowerLawFitResult(
        alpha=float(alpha),
        alpha_err=float(sigma),
        xmin=float(xmin_fit),
        xmax=float(np.max(energies)),
        n_tail=n_tail,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pvalue,
        loglikelihood_ratio=comparisons,
    )

    logger.info(
        f"Energy MLE fit: alpha = {alpha:.3f} +/- {sigma:.3f}, "
        f"xmin = {xmin_fit:.2e}, n_tail = {n_tail}"
    )
    return result


def fit_waiting_times(dt: np.ndarray) -> WaitingTimeFitResult:
    """
    Analyse waiting time distribution: test Poisson (exponential)
    vs. clustered (Weibull, lognormal) models.
    """
    from scipy import stats as sp_stats

    dt = dt[np.isfinite(dt) & (dt > 0)]
    logger.info(f"Fitting waiting time distribution: {len(dt)} intervals")

    cv = float(np.std(dt) / np.mean(dt))
    is_clustered = bool(cv > 1.0)
    logger.info(
        f"Coefficient of variation: {cv:.3f} "
        f"({'clustered' if is_clustered else 'Poisson-like'})"
    )

    exp_rate = float(1.0 / np.mean(dt))
    exp_loglik = np.sum(sp_stats.expon.logpdf(dt, scale=1.0 / exp_rate))
    exp_aic = float(-2 * exp_loglik + 2 * 1)

    try:
        weibull_shape, _, weibull_scale = sp_stats.weibull_min.fit(dt, floc=0)
        weibull_loglik = np.sum(
            sp_stats.weibull_min.logpdf(dt, weibull_shape, 0, weibull_scale)
        )
        weibull_aic = float(-2 * weibull_loglik + 2 * 2)
        logger.info(f"Weibull fit: shape={weibull_shape:.3f}, scale={weibull_scale:.3f}")
    except Exception as exc:
        logger.warning(f"Weibull fit failed: {exc}")
        weibull_shape, weibull_scale = None, None
        weibull_aic = float("inf")

    try:
        lognormal_shape, _, lognormal_scale = sp_stats.lognorm.fit(dt, floc=0)
        lognormal_loglik = np.sum(
            sp_stats.lognorm.logpdf(dt, lognormal_shape, 0, lognormal_scale)
        )
        lognormal_aic = float(-2 * lognormal_loglik + 2 * 2)
        logger.info(
            f"Lognormal fit: shape={lognormal_shape:.3f}, scale={lognormal_scale:.3f}"
        )
    except Exception as exc:
        logger.warning(f"Lognormal fit failed: {exc}")
        lognormal_shape, lognormal_scale = None, None
        lognormal_aic = float("inf")

    models = {
        "exponential": exp_aic,
        "weibull": weibull_aic,
        "lognormal": lognormal_aic,
    }
    best_model = min(models, key=models.get)
    logger.info(f"Best waiting time model (AIC): {best_model}")

    return WaitingTimeFitResult(
        best_model=best_model,
        coefficient_of_variation=cv,
        is_clustered=is_clustered,
        exponential_rate=exp_rate,
        weibull_shape=None if weibull_shape is None else float(weibull_shape),
        weibull_scale=None if weibull_scale is None else float(weibull_scale),
        lognormal_shape=None if lognormal_shape is None else float(lognormal_shape),
        lognormal_scale=None if lognormal_scale is None else float(lognormal_scale),
        model_comparison=models,
    )


def sample_powerlaw_alpha_posterior(
    values: np.ndarray,
    xmin: Optional[float] = None,
    n_walkers: int = 24,
    n_steps: int = 1200,
    burn_in: int = 300,
    random_seed: int = 42,
) -> PowerLawMCMCResult:
    """
    Sample the posterior on the power-law index alpha using emcee.

    The likelihood assumes a continuous power law for samples x >= xmin:
    p(x | alpha, xmin) = (alpha - 1) xmin^(alpha - 1) x^(-alpha), alpha > 1.
    """
    import emcee

    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    if xmin is None:
        xmin = np.min(data)
    tail = data[data >= xmin]
    if len(tail) < 5:
        raise ValueError("Need at least five samples above xmin for MCMC inference")

    log_x = np.log(tail / xmin)
    rng = np.random.default_rng(random_seed)

    def log_prior(theta):
        alpha = theta[0]
        if 1.01 < alpha < 6.0:
            return 0.0
        return -np.inf

    def log_likelihood(theta):
        alpha = theta[0]
        if alpha <= 1.0:
            return -np.inf
        return len(tail) * np.log(alpha - 1.0) - alpha * np.sum(log_x)

    def log_probability(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    mle_alpha = 1.0 + len(tail) / np.sum(log_x)
    initial = mle_alpha + 1e-2 * rng.standard_normal((n_walkers, 1))

    sampler = emcee.EnsembleSampler(n_walkers, 1, log_probability)
    sampler.run_mcmc(initial, n_steps, progress=False)

    flat_samples = sampler.get_chain(discard=burn_in, flat=True)[:, 0]
    q16, q50, q84 = np.percentile(flat_samples, [16, 50, 84])
    result = PowerLawMCMCResult(
        alpha_median=float(q50),
        alpha_minus=float(q50 - q16),
        alpha_plus=float(q84 - q50),
        samples=flat_samples,
        xmin=float(xmin),
        n_samples=int(len(flat_samples)),
    )

    logger.info(
        f"Power-law alpha MCMC: alpha = {result.alpha_median:.3f} "
        f"+{result.alpha_plus:.3f}/-{result.alpha_minus:.3f} "
        f"(xmin={xmin:.2e}, samples={result.n_samples})"
    )
    return result


def frb_anomaly_test(catalogue: pd.DataFrame) -> FRBAnomalyResult:
    """Test whether the FRB-coincident burst is anomalous."""
    frb_mask = catalogue["is_frb_burst"]
    if not frb_mask.any():
        raise ValueError("No FRB burst found in catalogue")

    frb = catalogue[frb_mask].iloc[0]
    non_frb = catalogue[~frb_mask]

    frb_energy = float(frb["energy"])
    energy_percentile = float((non_frb["energy"] <= frb_energy).mean() * 100)
    log_energies = np.log10(non_frb["energy"].values)
    energy_zscore = float((np.log10(frb_energy) - np.mean(log_energies)) / np.std(log_energies))

    frb_duration = float(frb["duration"])
    duration_percentile = float((non_frb["duration"] <= frb_duration).mean() * 100)
    log_durations = np.log10(non_frb["duration"].values)
    duration_zscore = float(
        (np.log10(frb_duration) - np.mean(log_durations)) / np.std(log_durations)
    )

    frb_wt = frb.get("waiting_time", np.nan)
    waiting_time_percentile = None
    if np.isfinite(frb_wt) and frb_wt > 0:
        valid_wt = non_frb["waiting_time"].dropna()
        valid_wt = valid_wt[valid_wt > 0]
        if len(valid_wt) > 0:
            waiting_time_percentile = float((valid_wt <= frb_wt).mean() * 100)

    is_energy_outlier = bool(energy_percentile > 95.0)
    is_duration_outlier = bool(duration_percentile > 95.0)

    if is_energy_outlier:
        conclusion = (
            f"The FRB burst is the MOST ENERGETIC in the observed sample: "
            f"its energy ({frb_energy:.2e} erg) is at the {energy_percentile:.1f}th "
            f"percentile ({energy_zscore:.1f} sigma above the mean in log-space). "
            f"This places it in the extreme observed high-energy tail of the "
            f"magnetar burst population. Whether that observed extremeness is "
            f"also unusual under the fitted tail model should be assessed with "
            f"the sample-maximum consistency test."
        )
    elif energy_percentile > 75:
        conclusion = (
            f"The FRB burst is in the UPPER TAIL but not a clear outlier: "
            f"energy at the {energy_percentile:.1f}th percentile "
            f"({energy_zscore:.1f} sigma). The FRB burst was energetic but not "
            f"uniquely so — other factors may explain the radio emission."
        )
    else:
        conclusion = (
            f"The FRB burst appears ORDINARY: energy at the "
            f"{energy_percentile:.1f}th percentile ({energy_zscore:.1f} sigma). "
            f"The FRB was produced by a statistically typical burst, "
            f"suggesting radio emission is not determined by burst energy alone."
        )

    result = FRBAnomalyResult(
        frb_energy=frb_energy,
        frb_duration=frb_duration,
        frb_waiting_time=float(frb_wt) if np.isfinite(frb_wt) else None,
        energy_percentile=energy_percentile,
        duration_percentile=duration_percentile,
        waiting_time_percentile=waiting_time_percentile,
        energy_zscore=energy_zscore,
        duration_zscore=duration_zscore,
        is_energy_outlier=is_energy_outlier,
        is_duration_outlier=is_duration_outlier,
        conclusion=conclusion,
    )

    logger.info(f"FRB anomaly test: {conclusion}")
    return result


def fit_duration_distribution(
    durations: np.ndarray,
    xmin: Optional[float] = None,
) -> PowerLawFitResult:
    """Fit the burst duration distribution with a power law."""
    import powerlaw

    logger.info(f"Fitting duration distribution: {len(durations)} bursts")

    fit = powerlaw.Fit(durations, xmin=xmin, verbose=False) if xmin is not None else powerlaw.Fit(
        durations, verbose=False
    )

    alpha = fit.power_law.alpha
    sigma = fit.power_law.sigma
    xmin_fit = fit.power_law.xmin
    try:
        ks_stat = float(fit.power_law.KS())
    except (NameError, AttributeError):
        ks_stat = None
        logger.warning("KS statistic computation failed (powerlaw pkg bug), leaving unset")

    n_tail = int(np.sum(durations >= xmin_fit))

    result = PowerLawFitResult(
        alpha=float(alpha),
        alpha_err=float(sigma),
        xmin=float(xmin_fit),
        xmax=float(np.max(durations)),
        n_tail=n_tail,
        ks_statistic=ks_stat,
        ks_pvalue=None,
        loglikelihood_ratio={},
        distribution_name="duration_power_law",
    )

    logger.info(
        f"Duration MLE fit: beta = {alpha:.3f} +/- {sigma:.3f}, "
        f"xmin = {xmin_fit:.4f}s"
    )
    return result


def energy_duration_scaling(
    energies: np.ndarray,
    durations: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Fit the energy-duration scaling relation in log-log space."""
    from scipy import stats as sp_stats

    log_e = np.log10(energies)
    log_t = np.log10(durations)

    slope, intercept, r_value, _, std_err = sp_stats.linregress(log_t, log_e)

    logger.info(
        f"E-T scaling: delta = {slope:.3f} +/- {std_err:.3f}, "
        f"R^2 = {r_value**2:.3f}"
    )

    return float(slope), float(std_err), float(intercept), float(r_value**2)


def soc_consistency_check(
    catalogue: pd.DataFrame,
    energy_xmin: Optional[float] = None,
    duration_xmin: Optional[float] = None,
) -> SOCResult:
    """
    Full self-organized criticality test:
    1. Fit energy distribution -> alpha
    2. Fit duration distribution -> beta
    3. Fit E-T scaling -> delta
    4. Check consistency: alpha = 1 + (beta - 1) / delta
    """
    energies = catalogue["energy"].values
    durations = catalogue["duration"].values

    energy_fit = fit_energy_distribution(energies, xmin=energy_xmin)
    duration_fit = fit_duration_distribution(durations, xmin=duration_xmin)

    alpha = energy_fit.alpha
    alpha_err = energy_fit.alpha_err
    beta = duration_fit.alpha
    beta_err = duration_fit.alpha_err

    delta, delta_err, _, _ = energy_duration_scaling(energies, durations)

    predicted_alpha = 1.0 + (beta - 1.0) / delta

    pred_err_beta = (1.0 / delta) * beta_err
    pred_err_delta = ((beta - 1.0) / delta**2) * delta_err
    predicted_alpha_err = float(np.sqrt(pred_err_beta**2 + pred_err_delta**2))

    # HACK: error propagation assumes alpha_err and predicted_alpha_err are
    # independent, which is only approximately true since alpha feeds into
    # the SOC prediction. Good enough for a rough consistency check.
    total_err = float(np.sqrt(alpha_err**2 + predicted_alpha_err**2))
    consistency = float(abs(alpha - predicted_alpha) / total_err) if total_err > 0 else float("inf")

    is_consistent = bool(consistency < 2.0)

    if is_consistent:
        conclusion = (
            f"SOC CONSISTENT: Measured alpha={alpha:.3f}+/-{alpha_err:.3f}, "
            f"predicted alpha={predicted_alpha:.3f}+/-{predicted_alpha_err:.3f} "
            f"({consistency:.1f} sigma apart). The magnetar crust behaves "
            f"as a self-organised critical system."
        )
    else:
        conclusion = (
            f"SOC INCONSISTENT: Measured alpha={alpha:.3f}+/-{alpha_err:.3f}, "
            f"predicted alpha={predicted_alpha:.3f}+/-{predicted_alpha_err:.3f} "
            f"({consistency:.1f} sigma apart). The scaling relations do not "
            f"simultaneously hold - SOC may not apply."
        )

    logger.info(f"SOC check: {conclusion}")

    return SOCResult(
        alpha=float(alpha),
        alpha_err=float(alpha_err),
        beta=float(beta),
        beta_err=float(beta_err),
        delta=float(delta),
        delta_err=float(delta_err),
        predicted_alpha=float(predicted_alpha),
        predicted_alpha_err=predicted_alpha_err,
        consistency=consistency,
        is_consistent=is_consistent,
        conclusion=conclusion,
    )


def save_results(
    energy_fit: PowerLawFitResult,
    waiting_fit: Optional[WaitingTimeFitResult],
    frb_result: Optional[FRBAnomalyResult],
    soc_result: Optional[SOCResult],
    output_path: str,
) -> None:
    """Save available analysis results to a JSON file."""
    results = {
        "energy_distribution": energy_fit.to_dict(),
    }
    if waiting_fit is not None:
        results["waiting_time_analysis"] = waiting_fit.to_dict()
    if frb_result is not None:
        results["frb_anomaly_test"] = frb_result.to_dict()
    if soc_result is not None:
        results["soc_consistency"] = soc_result.to_dict()

    outpath = Path(output_path)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    with outpath.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)

    logger.info(f"Results saved to {outpath}")
