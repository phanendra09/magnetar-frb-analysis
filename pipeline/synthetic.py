"""
synthetic.py - synthetic burst data for testing

Generate fake magnetar burst populations with known parameters
so we can validate the pipeline before running on real data.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class SyntheticBurstData:
    """Container for synthetic photon event data."""
    photon_times: np.ndarray       # All photon arrival times (seconds)
    burst_starts: np.ndarray       # True burst start times
    burst_ends: np.ndarray         # True burst end times
    burst_energies: np.ndarray     # True burst energies (erg)
    burst_durations: np.ndarray    # True burst durations (seconds)
    frb_burst_index: int           # Index of the injected FRB burst
    frb_burst_time: float          # Time of the FRB burst
    quiescent_rate: float          # Background photon rate (counts/s)
    observation_start: float       # Observation window start
    observation_end: float         # Observation window end
    true_alpha: float              # Injected energy power-law index
    true_beta: float               # Injected duration power-law index
    true_delta: float              # Injected E-T scaling index
    metadata: dict = field(default_factory=dict)


def generate_power_law_samples(
    n: int,
    alpha: float,
    xmin: float,
    xmax: float = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate samples from a power-law distribution P(x) ∝ x^{-alpha}
    using inverse CDF sampling.

    Parameters
    ----------
    n : int
        Number of samples.
    alpha : float
        Power-law index (> 1).
    xmin : float
        Minimum value of the distribution.
    xmax : float, optional
        Maximum value (truncated power law). If None, untruncated.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Power-law distributed samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    u = rng.uniform(0, 1, size=n)

    if alpha == 1.0:
        # Special case: log-uniform
        if xmax is None:
            raise ValueError("alpha=1 requires finite xmax")
        return xmin * (xmax / xmin) ** u

    if xmax is None:
        # Untruncated power law via inverse CDF
        return xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))
    else:
        # Truncated power law
        a = 1.0 - alpha
        return (xmin**a + u * (xmax**a - xmin**a)) ** (1.0 / a)


def generate_clustered_waiting_times(
    n: int,
    mean_dt: float,
    shape: float = 0.5,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate clustered (super-Poissonian) waiting times using a
    Weibull distribution. Shape < 1 gives clustering; shape = 1 is
    exponential (Poisson).

    Parameters
    ----------
    n : int
        Number of waiting times.
    mean_dt : float
        Mean waiting time (seconds).
    shape : float
        Weibull shape parameter. < 1 = clustered, 1 = Poisson.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Waiting times in seconds.
    """
    if rng is None:
        rng = np.random.default_rng()

    from scipy.special import gamma as gamma_func

    # Weibull scale from desired mean: mean = scale * Gamma(1 + 1/shape)
    scale = mean_dt / gamma_func(1.0 + 1.0 / shape)
    return scale * rng.weibull(shape, size=n)


def generate_burst_photons(
    start: float,
    duration: float,
    energy: float,
    distance_kpc: float = 9.0,
    effective_area_cm2: float = 200.0,
    mean_photon_energy_keV: float = 30.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate photon arrival times for a single burst with a fast-rise
    exponential-decay (FRED) profile.

    Parameters
    ----------
    start : float
        Burst start time (seconds).
    duration : float
        Burst duration (seconds).
    energy : float
        Total burst energy (erg).
    distance_kpc : float
        Source distance in kpc.
    effective_area_cm2 : float
        Detector effective area.
    mean_photon_energy_keV : float
        Mean photon energy for count conversion.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Photon arrival times within the burst.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Convert energy to approximate photon count
    distance_cm = distance_kpc * 3.086e21  # kpc to cm
    fluence = energy / (4 * np.pi * distance_cm**2)  # erg/cm^2
    photon_energy_erg = mean_photon_energy_keV * 1.602e-9  # keV to erg
    n_photons_expected = fluence * effective_area_cm2 / photon_energy_erg
    n_photons = max(int(n_photons_expected), 5)

    # Cap for performance — Bayesian Blocks is O(n²) so total
    # photon count must stay manageable (~10-20K total)
    n_photons = min(n_photons, 500)

    # FRED profile: fast rise (10% of duration), exponential decay
    rise_time = 0.1 * duration
    tau_decay = 0.4 * duration  # decay timescale

    # Rejection sampling from FRED envelope
    times = []
    while len(times) < n_photons:
        candidates = rng.uniform(0, duration, size=n_photons * 3)
        # FRED profile value
        profile = np.where(
            candidates < rise_time,
            candidates / rise_time,  # linear rise
            np.exp(-(candidates - rise_time) / tau_decay),  # exp decay
        )
        accept = rng.uniform(0, 1, size=len(candidates)) < profile
        times.extend(candidates[accept].tolist())

    times = np.sort(np.array(times[:n_photons])) + start
    return times


def generate_synthetic_dataset(
    n_bursts: int = 300,
    alpha: float = 1.6,
    beta: float = 1.8,
    delta: float = 1.5,
    energy_min: float = 1e37,
    energy_max: float = 1e42,
    duration_min: float = 0.01,
    duration_max: float = 5.0,
    mean_waiting_time: float = 200.0,
    waiting_time_shape: float = 0.5,
    quiescent_rate: float = 0.5,
    frb_energy_percentile: float = 0.85,
    distance_kpc: float = 9.0,
    observation_start: float = 0.0,
    seed: int = 42,
) -> SyntheticBurstData:
    """Generate a full synthetic burst dataset for pipeline testing."""
    rng = np.random.default_rng(seed)
    logger.info(
        f"Generating synthetic dataset: {n_bursts} bursts, "
        f"α={alpha}, β={beta}, δ={delta}"
    )

    # --- Generate burst energies (power-law) ---
    energies = generate_power_law_samples(
        n_bursts, alpha, energy_min, energy_max, rng
    )

    # --- Generate correlated durations ---
    # E ∝ T^delta => T ∝ E^(1/delta)
    #   with scatter
    durations_base = (energies / energy_min) ** (1.0 / delta)
    durations_base = durations_base / durations_base.max() * duration_max
    # Add log-normal scatter
    log_scatter = rng.normal(0, 0.3, size=n_bursts)
    durations = durations_base * np.exp(log_scatter)
    durations = np.clip(durations, duration_min, duration_max)

    # --- Generate clustered waiting times ---
    waiting_times = generate_clustered_waiting_times(
        n_bursts - 1, mean_waiting_time, waiting_time_shape, rng
    )

    # --- Compute burst start times ---
    burst_starts = np.zeros(n_bursts)
    burst_starts[0] = observation_start + rng.uniform(10, 100)
    for i in range(1, n_bursts):
        burst_starts[i] = burst_starts[i - 1] + durations[i - 1] + waiting_times[i - 1]

    burst_ends = burst_starts + durations

    # --- Inject FRB burst ---
    # Place the FRB burst roughly in the middle of the observation
    frb_idx = n_bursts // 2
    frb_energy_value = np.percentile(energies, frb_energy_percentile * 100)
    energies[frb_idx] = frb_energy_value
    frb_time = burst_starts[frb_idx]
    logger.info(
        f"FRB burst injected at index {frb_idx}, time={frb_time:.1f}s, "
        f"energy={frb_energy_value:.2e} erg ({frb_energy_percentile*100:.0f}th percentile)"
    )

    # --- Generate photon arrival times ---
    observation_end = burst_ends[-1] + rng.uniform(100, 500)
    total_obs_time = observation_end - observation_start

    # Background photons (Poisson process)
    n_background = rng.poisson(quiescent_rate * total_obs_time)
    bg_photons = rng.uniform(observation_start, observation_end, size=n_background)

    logger.info(f"Generating photons for {n_bursts} bursts...")

    # Burst photons
    all_photons = [bg_photons]
    for i in range(n_bursts):
        burst_photons = generate_burst_photons(
            burst_starts[i],
            durations[i],
            energies[i],
            distance_kpc=distance_kpc,
            rng=rng,
        )
        all_photons.append(burst_photons)

    photon_times = np.sort(np.concatenate(all_photons))

    logger.info(
        f"Synthetic dataset complete: {len(photon_times)} photons, "
        f"observation window [{observation_start:.0f}, {observation_end:.0f}]s"
    )

    return SyntheticBurstData(
        photon_times=photon_times,
        burst_starts=burst_starts,
        burst_ends=burst_ends,
        burst_energies=energies,
        burst_durations=durations,
        frb_burst_index=frb_idx,
        frb_burst_time=frb_time,
        quiescent_rate=quiescent_rate,
        observation_start=observation_start,
        observation_end=observation_end,
        true_alpha=alpha,
        true_beta=beta,
        true_delta=delta,
        metadata={
            "n_bursts": n_bursts,
            "n_photons": len(photon_times),
            "n_background": n_background,
            "energy_range": [energy_min, energy_max],
            "duration_range": [duration_min, duration_max],
            "mean_waiting_time": mean_waiting_time,
            "waiting_time_shape": waiting_time_shape,
            "frb_energy_percentile": frb_energy_percentile,
            "seed": seed,
        },
    )
