"""
detect.py - Bayesian Blocks burst detection

Find individual bursts in photon arrival time data using the
Bayesian Blocks algorithm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Burst:
    """Represents a single detected magnetar burst."""

    start_time: float
    end_time: float
    duration: float
    peak_rate: float
    total_counts: int
    fluence: float
    energy: float
    is_frb_burst: bool

    def __repr__(self) -> str:
        frb_tag = " [FRB]" if self.is_frb_burst else ""
        return (
            f"Burst(t={self.start_time:.3f}s, dur={self.duration:.4f}s, "
            f"E={self.energy:.2e} erg, counts={self.total_counts}){frb_tag}"
        )


def bayesian_blocks_detect(photon_times: np.ndarray, p0: float = 0.01) -> np.ndarray:
    """Run Bayesian Blocks on sorted photon arrival times."""
    from astropy.stats import bayesian_blocks

    logger.info(f"Running Bayesian Blocks on {len(photon_times)} events (p0={p0})")
    # NOTE: this can be very slow for >100k events, we subsample upstream
    edges = bayesian_blocks(photon_times, fitness="events", p0=p0)
    logger.info(f"Found {len(edges) - 1} blocks")
    return edges


def compute_block_rates(photon_times: np.ndarray, edges: np.ndarray) -> tuple:
    # rates, counts, durations, centers for each BB block
    n_blocks = len(edges) - 1
    block_rates = np.zeros(n_blocks)
    block_counts = np.zeros(n_blocks, dtype=int)
    block_durations = np.zeros(n_blocks)
    block_centers = np.zeros(n_blocks)

    for idx in range(n_blocks):
        left, right = edges[idx], edges[idx + 1]
        dt = right - left
        mask = (photon_times >= left) & (photon_times < right)
        counts = int(np.sum(mask))

        block_durations[idx] = dt
        block_counts[idx] = counts
        block_rates[idx] = counts / dt if dt > 0 else 0.0
        block_centers[idx] = 0.5 * (left + right)

    return block_rates, block_counts, block_durations, block_centers


def estimate_quiescent_rate(block_rates: np.ndarray) -> float:
    """Estimate quiescent rate from the 25th percentile of block rates."""
    positive = block_rates[block_rates > 0]
    if len(positive) == 0:
        raise ValueError("No positive block rates found")
    q_rate = float(np.percentile(positive, 25))
    logger.info(f"Estimated quiescent rate: {q_rate:.2f} counts/s")
    return q_rate


def identify_bursts(
    photon_times: np.ndarray,
    edges: np.ndarray,
    threshold_sigma: float = 5.0,
    min_counts: int = 10,
    distance_kpc: float = 9.0,
    effective_area_cm2: float = 200.0,
    mean_photon_energy_keV: float = 30.0,
    frb_time: Optional[float] = None,
    frb_time_tolerance: float = 1.0,
) -> List[Burst]:
    """
    Identify bursts from Bayesian Blocks.

    A block is classified as a burst when its rate exceeds the estimated
    quiescent rate by `threshold_sigma * sqrt(quiescent_rate)`.
    Adjacent burst blocks are merged into a single burst candidate.
    """
    block_rates, _, _, _ = compute_block_rates(photon_times, edges)
    quiescent_rate = estimate_quiescent_rate(block_rates)
    threshold_rate = quiescent_rate + threshold_sigma * np.sqrt(quiescent_rate)
    logger.info(
        f"Burst threshold: {threshold_rate:.2f} counts/s "
        f"({threshold_sigma} sigma above quiescent)"
    )

    is_burst_block = block_rates > threshold_rate
    bursts: List[Burst] = []
    idx = 0
    n_blocks = len(block_rates)

    while idx < n_blocks:
        if not is_burst_block[idx]:
            idx += 1
            continue

        start_idx = idx
        while idx < n_blocks and is_burst_block[idx]:
            idx += 1
        end_idx = idx - 1

        t_start = edges[start_idx]
        t_end = edges[end_idx + 1]
        duration = t_end - t_start

        mask = (photon_times >= t_start) & (photon_times < t_end)
        total_counts = int(np.sum(mask))
        if total_counts < min_counts:
            continue

        peak_rate = float(np.max(block_rates[start_idx : end_idx + 1]))
        photon_energy_erg = mean_photon_energy_keV * 1.602e-9
        fluence = total_counts * photon_energy_erg / effective_area_cm2
        distance_cm = distance_kpc * 3.086e21
        energy = fluence * 4 * np.pi * distance_cm**2

        is_frb = False
        if frb_time is not None:
            is_frb = bool(
                abs(t_start - frb_time) < frb_time_tolerance
                or (t_start <= frb_time <= t_end)
            )

        bursts.append(
            Burst(
                start_time=float(t_start),
                end_time=float(t_end),
                duration=float(duration),
                peak_rate=peak_rate,
                total_counts=total_counts,
                fluence=float(fluence),
                energy=float(energy),
                is_frb_burst=is_frb,
            )
        )

    logger.info(f"Detected {len(bursts)} bursts")

    frb_bursts = [burst for burst in bursts if burst.is_frb_burst]
    if frb_bursts:
        logger.info(f"FRB burst found: {frb_bursts[0]}")
    elif frb_time is not None:
        logger.warning("FRB burst time was provided but did not match any detected burst")

    return bursts
