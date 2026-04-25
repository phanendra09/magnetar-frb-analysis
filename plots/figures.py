"""
figures.py - generate all paper figures
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from plots.style import COLORS, annotate_frb, apply_style, get_figure

logger = logging.getLogger(__name__)


def save_figure(fig, name: str, output_dir: str = "results/figures") -> None:
    """Save a figure as both PNG and PDF."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in ["png", "pdf"]:
        fig.savefig(str(outdir / f"{name}.{fmt}"), format=fmt)
    logger.info(f"Saved figure: {name}")


def plot_lightcurve(
    photon_times: np.ndarray,
    edges: np.ndarray,
    frb_time: Optional[float] = None,
    bin_size: float = 1.0,
    output_dir: str = "results/figures",
):
    """Plot the burst-storm light curve with Bayesian Blocks overlaid."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.0),
        height_ratios=[3, 1],
        sharex=True,
        gridspec_kw={"hspace": 0.05},
    )

    t0 = photon_times.min()
    t_rel = photon_times - t0

    bins = np.arange(0, t_rel.max() + bin_size, bin_size)
    counts, bin_edges = np.histogram(t_rel, bins=bins)
    rates = counts / bin_size
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    ax1.step(
        centers,
        rates,
        where="mid",
        color=COLORS["neutral"],
        alpha=0.4,
        linewidth=0.5,
        label="Binned light curve",
    )

    edges_rel = edges - t0
    median_rate = np.median(rates) if len(rates) else 0.0
    for idx in range(len(edges_rel) - 1):
        left, right = edges_rel[idx], edges_rel[idx + 1]
        mask = (t_rel >= left) & (t_rel < right)
        n_in_block = np.sum(mask)
        dt_block = right - left
        rate_block = n_in_block / dt_block if dt_block > 0 else 0.0
        color = COLORS["primary"] if rate_block > median_rate * 3 else COLORS["light"]
        ax1.fill_between(
            [left, right],
            rate_block,
            0,
            alpha=0.3,
            color=color,
            edgecolor=COLORS["primary"],
            linewidth=0.8,
        )

    if frb_time is not None:
        frb_rel = frb_time - t0
        ylim = ax1.get_ylim()
        ax1.axvline(
            frb_rel,
            color=COLORS["frb"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            zorder=10,
        )
        ax1.text(
            frb_rel,
            ylim[1] * 0.95,
            "FRB 200428",
            color=COLORS["frb"],
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=COLORS["frb"],
                alpha=0.9,
            ),
        )

    ax1.set_ylabel("Count rate (counts s$^{-1}$)")
    ax1.set_title("SGR 1935+2154 - April 2020 Burst Storm", fontsize=12)

    from matplotlib.patches import Patch

    ax1.legend(
        handles=[
            Patch(
                facecolor=COLORS["primary"],
                alpha=0.3,
                edgecolor=COLORS["primary"],
                label="Bayesian Blocks (burst)",
            ),
            Patch(
                facecolor=COLORS["light"],
                alpha=0.3,
                edgecolor=COLORS["primary"],
                label="Bayesian Blocks (quiescent)",
            ),
        ],
        loc="upper right",
        fontsize=8,
    )

    ax2.plot(t_rel, np.arange(1, len(t_rel) + 1), color=COLORS["dark"], linewidth=0.8)
    ax2.set_ylabel("Cumulative\ncounts")
    ax2.set_xlabel(f"Time (s) since T0 = {t0:.1f} s")

    if frb_time is not None:
        ax2.axvline(frb_rel, color=COLORS["frb"], linestyle="--", linewidth=1.0, alpha=0.6)

    fig.align_ylabels([ax1, ax2])
    save_figure(fig, "fig1_lightcurve", output_dir)
    plt.close(fig)
    return fig


def plot_energy_distribution(
    energies: np.ndarray,
    fit_result,
    frb_energy: Optional[float] = None,
    output_dir: str = "results/figures",
):
    """Plot the burst-energy CCDF with the fitted power law."""
    apply_style()
    fig, ax = get_figure("single_column_tall")

    e_sorted = np.sort(energies)
    ccdf = 1.0 - np.arange(1, len(e_sorted) + 1) / (len(e_sorted) + 1)

    ax.scatter(
        e_sorted,
        ccdf,
        s=12,
        color=COLORS["primary"],
        alpha=0.7,
        edgecolors="none",
        zorder=5,
        label="Observed bursts",
    )

    alpha = fit_result.alpha
    xmin = fit_result.xmin
    e_fit = np.logspace(np.log10(xmin), np.log10(e_sorted.max()), 100)
    ccdf_fit = (e_fit / xmin) ** (-(alpha - 1))
    idx_xmin = np.searchsorted(e_sorted, xmin)
    if idx_xmin < len(ccdf):
        ccdf_fit *= ccdf[idx_xmin]
    ax.plot(
        e_fit,
        ccdf_fit,
        color=COLORS["secondary"],
        linewidth=2,
        linestyle="-",
        label=f"Power law: $\\alpha = {alpha:.2f} \\pm {fit_result.alpha_err:.2f}$",
        zorder=6,
    )

    lower_alpha = alpha - fit_result.alpha_err
    upper_alpha = alpha + fit_result.alpha_err
    lower_band = (e_fit / xmin) ** (-(upper_alpha - 1))
    upper_band = (e_fit / xmin) ** (-(lower_alpha - 1))
    if idx_xmin < len(ccdf):
        lower_band *= ccdf[idx_xmin]
        upper_band *= ccdf[idx_xmin]
    ax.fill_between(e_fit, lower_band, upper_band, color=COLORS["secondary"], alpha=0.1)

    ax.axvline(
        xmin,
        color=COLORS["neutral"],
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
        label=f"$x_{{\\mathrm{{min}}}} = {xmin:.1e}$",
    )

    if frb_energy is not None:
        frb_idx = np.searchsorted(e_sorted, frb_energy)
        frb_ccdf = 1.0 - frb_idx / (len(e_sorted) + 1)
        ax.scatter(
            [frb_energy],
            [frb_ccdf],
            s=80,
            color=COLORS["frb"],
            marker="*",
            zorder=10,
            edgecolors="black",
            linewidths=0.5,
        )
        annotate_frb(ax, frb_energy, frb_ccdf)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (erg)")
    ax.set_ylabel("CCDF: $P(E > E_0)$")
    ax.set_title("Burst Energy Distribution")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_ylim(bottom=1e-3)

    save_figure(fig, "fig2_energy_distribution", output_dir)
    plt.close(fig)
    return fig


def plot_waiting_times(
    waiting_times: np.ndarray,
    fit_result,
    output_dir: str = "results/figures",
):
    """Plot the waiting-time distribution with fitted comparison models."""
    from scipy import stats as sp_stats

    apply_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.5),
        gridspec_kw={"width_ratios": [2, 1], "wspace": 0.35},
    )

    dt = waiting_times[np.isfinite(waiting_times) & (waiting_times > 0)]
    ax_main, ax_qq = axes

    bins = np.logspace(np.log10(dt.min()), np.log10(dt.max()), 30)
    ax_main.hist(
        dt,
        bins=bins,
        density=True,
        alpha=0.3,
        color=COLORS["primary"],
        edgecolor=COLORS["primary"],
        linewidth=0.5,
        label="Data",
    )

    x_fit = np.logspace(np.log10(dt.min()), np.log10(dt.max()), 200)

    exp_rate = fit_result.exponential_rate
    ax_main.plot(
        x_fit,
        sp_stats.expon.pdf(x_fit, scale=1.0 / exp_rate),
        color=COLORS["accent"],
        linewidth=1.5,
        linestyle="--",
        label=f"Exponential ($\\lambda = {exp_rate:.4f}$)",
    )

    if fit_result.weibull_shape is not None and fit_result.weibull_scale is not None:
        ax_main.plot(
            x_fit,
            sp_stats.weibull_min.pdf(
                x_fit,
                fit_result.weibull_shape,
                0,
                fit_result.weibull_scale,
            ),
            color=COLORS["secondary"],
            linewidth=1.5,
            label=f"Weibull ($k = {fit_result.weibull_shape:.2f}$)",
        )

    if fit_result.lognormal_shape is not None and fit_result.lognormal_scale is not None:
        ax_main.plot(
            x_fit,
            sp_stats.lognorm.pdf(
                x_fit,
                fit_result.lognormal_shape,
                0,
                fit_result.lognormal_scale,
            ),
            color=COLORS["dark"],
            linewidth=1.5,
            linestyle=":",
            label=f"Lognormal ($\\sigma = {fit_result.lognormal_shape:.2f}$)",
        )

    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_xlabel("Waiting time $\\Delta t$ (s)")
    ax_main.set_ylabel("Probability density")
    ax_main.set_title("Inter-burst Waiting Times")
    ax_main.legend(fontsize=7)

    cv = fit_result.coefficient_of_variation
    status = "Clustered" if cv > 1 else "Poisson-like"
    ax_main.text(
        0.97,
        0.97,
        f"CV = {cv:.2f}\nBest AIC: {fit_result.best_model}\n({status})",
        transform=ax_main.transAxes,
        fontsize=8,
        va="top",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="lightyellow",
            edgecolor=COLORS["warning"],
            alpha=0.9,
        ),
    )

    dt_sorted = np.sort(dt)
    n = len(dt_sorted)
    theoretical_quantiles = sp_stats.expon.ppf(
        (np.arange(1, n + 1) - 0.5) / n,
        scale=1.0 / exp_rate,
    )

    ax_qq.scatter(
        theoretical_quantiles,
        dt_sorted,
        s=6,
        color=COLORS["primary"],
        alpha=0.6,
        edgecolors="none",
    )

    q_low = min(theoretical_quantiles.min(), dt_sorted.min())
    q_high = max(theoretical_quantiles.max(), dt_sorted.max())
    ax_qq.plot(
        [q_low, q_high],
        [q_low, q_high],
        color=COLORS["secondary"],
        linewidth=1,
        linestyle="--",
        alpha=0.8,
    )

    ax_qq.set_xlabel("Exponential quantiles")
    ax_qq.set_ylabel("Observed quantiles")
    ax_qq.set_title("Q-Q Plot", fontsize=10)
    ax_qq.set_xscale("log")
    ax_qq.set_yscale("log")

    save_figure(fig, "fig3_waiting_times", output_dir)
    plt.close(fig)
    return fig


def plot_frb_anomaly(
    catalogue,
    frb_result,
    energy_fit,
    output_dir: str = "results/figures",
):
    """Plot the FRB burst against the X-ray burst population."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5), gridspec_kw={"wspace": 0.4})

    frb_mask = catalogue["is_frb_burst"]
    non_frb = catalogue[~frb_mask]
    frb = catalogue[frb_mask].iloc[0]

    energies = non_frb["energy"].values
    bins = np.logspace(np.log10(energies.min()), np.log10(energies.max()), 25)
    ax1.hist(
        energies,
        bins=bins,
        alpha=0.4,
        color=COLORS["primary"],
        edgecolor=COLORS["primary"],
        linewidth=0.5,
        label="X-ray bursts",
    )

    ax1.axvline(
        frb["energy"],
        color=COLORS["frb"],
        linewidth=2,
        linestyle="-",
        zorder=10,
        label="FRB burst",
    )

    p95 = np.percentile(energies, 95)
    ax1.axvspan(p95, energies.max() * 2, alpha=0.08, color=COLORS["frb"], label="95th percentile tail")

    ax1.set_xscale("log")
    ax1.set_xlabel("Energy (erg)")
    ax1.set_ylabel("Number of bursts")
    ax1.set_title("Energy Distribution")
    ax1.text(
        0.03,
        0.97,
        f"FRB burst:\n{frb_result.energy_percentile:.1f}th pct\n"
        f"$z = {frb_result.energy_zscore:.1f}\\sigma$",
        transform=ax1.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=COLORS["frb"], alpha=0.9),
    )
    ax1.legend(fontsize=6.5, loc="center right")

    ax2.scatter(
        non_frb["duration"],
        non_frb["energy"],
        s=15,
        color=COLORS["primary"],
        alpha=0.4,
        edgecolors="none",
        label="X-ray bursts",
    )
    ax2.scatter(
        frb["duration"],
        frb["energy"],
        s=120,
        color=COLORS["frb"],
        marker="*",
        edgecolors="black",
        linewidths=0.7,
        zorder=10,
        label="FRB burst",
    )
    annotate_frb(ax2, frb["duration"], frb["energy"], xytext=(15, 15))

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Duration (s)")
    ax2.set_ylabel("Energy (erg)")
    ax2.set_title("Energy-Duration Plane")
    ax2.legend(fontsize=7, loc="lower right")

    verdict = "ANOMALOUS" if frb_result.is_energy_outlier else "ORDINARY"
    verdict_color = COLORS["frb"] if frb_result.is_energy_outlier else COLORS["accent"]
    fig.suptitle(
        f"FRB Burst Assessment: {verdict}",
        fontsize=13,
        fontweight="bold",
        color=verdict_color,
        y=1.02,
    )

    save_figure(fig, "fig4_frb_anomaly", output_dir)
    plt.close(fig)
    return fig


def plot_soc_check(
    catalogue,
    soc_result,
    energy_fit,
    output_dir: str = "results/figures",
):
    """Plot the three-part SOC consistency check."""
    import powerlaw
    from pipeline.stats import energy_duration_scaling

    apply_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.5, 3.5), gridspec_kw={"wspace": 0.45})
    fig.subplots_adjust(bottom=0.25)

    energies = catalogue["energy"].values
    durations = catalogue["duration"].values

    e_sorted = np.sort(energies)
    ccdf_e = 1.0 - np.arange(1, len(e_sorted) + 1) / (len(e_sorted) + 1)
    ax1.scatter(e_sorted, ccdf_e, s=8, color=COLORS["primary"], alpha=0.5, edgecolors="none")

    xmin = energy_fit.xmin
    e_fit = np.logspace(np.log10(xmin), np.log10(e_sorted.max()), 50)
    idx = np.searchsorted(e_sorted, xmin)
    if idx < len(ccdf_e):
        ax1.plot(
            e_fit,
            ccdf_e[idx] * (e_fit / xmin) ** (-(soc_result.alpha - 1)),
            color=COLORS["secondary"],
            linewidth=1.5,
        )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Energy (erg)")
    ax1.set_ylabel("CCDF")
    ax1.set_title(f"$\\alpha = {soc_result.alpha:.2f} \\pm {soc_result.alpha_err:.2f}$", fontsize=10)

    d_sorted = np.sort(durations)
    ccdf_d = 1.0 - np.arange(1, len(d_sorted) + 1) / (len(d_sorted) + 1)
    ax2.scatter(d_sorted, ccdf_d, s=8, color=COLORS["accent"], alpha=0.5, edgecolors="none")

    dur_fit = powerlaw.Fit(durations, verbose=False)
    dxmin = dur_fit.power_law.xmin
    d_fit_x = np.logspace(np.log10(dxmin), np.log10(d_sorted.max()), 50)
    didx = np.searchsorted(d_sorted, dxmin)
    if didx < len(ccdf_d):
        ax2.plot(
            d_fit_x,
            ccdf_d[didx] * (d_fit_x / dxmin) ** (-(soc_result.beta - 1)),
            color=COLORS["secondary"],
            linewidth=1.5,
        )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Duration (s)")
    ax2.set_ylabel("CCDF")
    ax2.set_title(f"$\\beta = {soc_result.beta:.2f} \\pm {soc_result.beta_err:.2f}$", fontsize=10)

    ax3.scatter(durations, energies, s=8, color=COLORS["dark"], alpha=0.4, edgecolors="none")
    delta, _, intercept, _ = energy_duration_scaling(energies, durations)
    d_range = np.logspace(np.log10(durations.min()), np.log10(durations.max()), 50)
    ax3.plot(
        d_range,
        10 ** (intercept + delta * np.log10(d_range)),
        color=COLORS["secondary"],
        linewidth=1.5,
    )

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Duration (s)")
    ax3.set_ylabel("Energy (erg)")
    ax3.set_title(f"$\\delta = {soc_result.delta:.2f} \\pm {soc_result.delta_err:.2f}$", fontsize=10)

    consistency_text = (
        f"SOC check: $\\alpha_{{pred}} = 1 + (\\beta - 1)/\\delta "
        f"= {soc_result.predicted_alpha:.2f}$\n"
        f"Measured $\\alpha = {soc_result.alpha:.2f}$  |  "
        f"$\\Delta = {soc_result.consistency:.1f}\\sigma$"
    )
    verdict = "CONSISTENT" if soc_result.is_consistent else "INCONSISTENT"
    verdict_color = COLORS["accent"] if soc_result.is_consistent else COLORS["frb"]

    fig.text(0.5, 0.12, consistency_text, ha="center", fontsize=9, style="italic")
    fig.text(0.5, 0.04, verdict, ha="center", fontsize=11, fontweight="bold", color=verdict_color)
    fig.suptitle("Self-Organised Criticality Test", fontsize=12, fontweight="bold", y=0.98)

    save_figure(fig, "fig5_soc_check", output_dir)
    plt.close(fig)
    return fig


def generate_all_figures(
    photon_times,
    edges,
    catalogue,
    energy_fit,
    waiting_fit,
    frb_result,
    soc_result,
    frb_time=None,
    output_dir="results/figures",
):
    """Generate all five publication figures."""
    logger.info("Generating all publication figures...")

    figures = [
        plot_lightcurve(photon_times, edges, frb_time, output_dir=output_dir),
        plot_energy_distribution(
            catalogue["energy"].values,
            energy_fit,
            frb_result.frb_energy if frb_result else None,
            output_dir=output_dir,
        ),
        plot_waiting_times(catalogue["waiting_time"].dropna().values, waiting_fit, output_dir=output_dir),
        plot_frb_anomaly(catalogue, frb_result, energy_fit, output_dir=output_dir),
        plot_soc_check(catalogue, soc_result, energy_fit, output_dir=output_dir),
    ]

    logger.info("  Figure 1: Light curve")
    logger.info("  Figure 2: Energy distribution")
    logger.info("  Figure 3: Waiting times")
    logger.info("  Figure 4: FRB anomaly")
    logger.info("  Figure 5: SOC check")
    logger.info(f"All figures saved to {output_dir}")
    return figures
