"""
catalogue.py - burst catalogue builder

Converts detected burst objects into a pandas DataFrame.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from pathlib import Path
import logging

from pipeline.detect import Burst

logger = logging.getLogger(__name__)


def build_catalogue(bursts: List[Burst]) -> pd.DataFrame:
    """Convert burst list to a DataFrame catalogue."""
    records = []
    for i, b in enumerate(bursts):
        records.append({
            "burst_id": i,
            "start_time": b.start_time,
            "end_time": b.end_time,
            "duration": b.duration,
            "peak_rate": b.peak_rate,
            "total_counts": b.total_counts,
            "fluence": b.fluence,
            "energy": b.energy,
            "is_frb_burst": b.is_frb_burst,
        })

    catalogue = pd.DataFrame(records)
    catalogue = catalogue.sort_values("start_time").reset_index(drop=True)
    catalogue["burst_id"] = range(len(catalogue))

    logger.info(f"Built catalogue with {len(catalogue)} bursts")
    return catalogue


def compute_waiting_times(catalogue: pd.DataFrame) -> pd.DataFrame:
    """
    Compute inter-burst waiting times (time between end of one burst
    and start of the next).

    Parameters
    ----------
    catalogue : pd.DataFrame
        Burst catalogue sorted by start_time.

    Returns
    -------
    pd.DataFrame
        Catalogue with added 'waiting_time' column.
    """
    cat = catalogue.sort_values("start_time").copy()
    # TODO: should this use end_time -> start_time or start_time -> start_time?
    # Using start-to-start for now, matches Younes+2020 convention
    dt = cat["start_time"].values[1:] - cat["end_time"].values[:-1]
    cat["waiting_time"] = np.concatenate([[np.nan], dt])

    # Remove negative waiting times (overlapping bursts)
    n_negative = np.sum(dt < 0)
    if n_negative > 0:
        logger.warning(
            f"{n_negative} negative waiting times found (overlapping bursts). "
            f"Setting to NaN."
        )
        cat.loc[cat["waiting_time"] < 0, "waiting_time"] = np.nan

    median_dt = np.nanmedian(cat["waiting_time"])
    logger.info(f"Waiting times: median={median_dt:.2f}s, n={np.sum(~np.isnan(cat['waiting_time']))}")
    return cat


def compute_energies(catalogue, distance_kpc=9.0):
    """Recompute energies for a different assumed distance."""
    cat = catalogue.copy()
    distance_cm = distance_kpc * 3.086e21  # kpc → cm
    cat["energy"] = cat["fluence"] * 4 * np.pi * distance_cm**2
    logger.info(
        f"Energies recomputed for d={distance_kpc} kpc: "
        f"range [{cat['energy'].min():.2e}, {cat['energy'].max():.2e}] erg"
    )
    return cat


def flag_frb_burst(
    catalogue: pd.DataFrame,
    frb_time: float,
    tolerance: float = 1.0,
) -> pd.DataFrame:
    """
    Flag the burst coincident with the FRB event.

    Parameters
    ----------
    catalogue : pd.DataFrame
        Burst catalogue.
    frb_time : float
        Time of the FRB event (same time system as burst times).
    tolerance : float
        Time tolerance for matching (seconds).

    Returns
    -------
    pd.DataFrame
        Catalogue with 'is_frb_burst' column updated.
    """
    cat = catalogue.copy()
    cat["is_frb_burst"] = False

    for idx, row in cat.iterrows():
        if (abs(row["start_time"] - frb_time) < tolerance or
                row["start_time"] <= frb_time <= row["end_time"]):
            cat.at[idx, "is_frb_burst"] = True
            logger.info(
                f"FRB burst flagged: burst_id={row['burst_id']}, "
                f"t={row['start_time']:.3f}s, E={row['energy']:.2e} erg"
            )

    n_flagged = cat["is_frb_burst"].sum()
    if n_flagged == 0:
        logger.warning("No burst matched the FRB time!")
    elif n_flagged > 1:
        logger.warning(f"Multiple bursts ({n_flagged}) matched the FRB time")

    return cat


def save_catalogue(catalogue: pd.DataFrame, path: str) -> None:
    """Save burst catalogue to CSV."""
    outpath = Path(path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    catalogue.to_csv(str(outpath), index=False, float_format="%.8e")
    logger.info(f"Catalogue saved to {outpath} ({len(catalogue)} bursts)")


def load_catalogue(path: str) -> pd.DataFrame:
    """Load burst catalogue from CSV."""
    cat = pd.read_csv(path)
    logger.info(f"Loaded catalogue from {path} ({len(cat)} bursts)")
    return cat


def catalogue_summary(catalogue: pd.DataFrame) -> dict:
    """
    Compute summary statistics for the burst catalogue.

    Returns
    -------
    dict
        Summary statistics.
    """
    wt = catalogue["waiting_time"].dropna()
    summary = {
        "n_bursts": len(catalogue),
        "total_duration_s": catalogue["end_time"].max() - catalogue["start_time"].min(),
        "energy_min_erg": catalogue["energy"].min(),
        "energy_max_erg": catalogue["energy"].max(),
        "energy_median_erg": catalogue["energy"].median(),
        "duration_min_s": catalogue["duration"].min(),
        "duration_max_s": catalogue["duration"].max(),
        "duration_median_s": catalogue["duration"].median(),
        "waiting_time_median_s": wt.median() if len(wt) > 0 else None,
        "waiting_time_cv": wt.std() / wt.mean() if len(wt) > 0 else None,
        "frb_burst_found": catalogue["is_frb_burst"].any(),
    }

    if catalogue["is_frb_burst"].any():
        frb = catalogue[catalogue["is_frb_burst"]].iloc[0]
        summary["frb_energy_erg"] = frb["energy"]
        summary["frb_duration_s"] = frb["duration"]
        summary["frb_energy_percentile"] = (
            (catalogue["energy"] <= frb["energy"]).mean() * 100
        )

    logger.info(f"Catalogue summary: {summary['n_bursts']} bursts")
    return summary
