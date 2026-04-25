"""
preprocess.py - FITS event file loading and filtering
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def load_fits_events(filepath: str) -> dict:
    """Load a FITS event file and extract photon event data."""
    from astropy.io import fits

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"FITS file not found: {path}")

    logger.info(f"Loading FITS file: {path.name}")
    with fits.open(str(path)) as hdul:
        primary_header = hdul[0].header
        instrument = primary_header.get("INSTRUME", "UNKNOWN")
        telescope = primary_header.get("TELESCOP", "UNKNOWN")
        logger.info(f"Instrument: {instrument}, Telescope: {telescope}")

        events_hdu = None
        for hdu in hdul:
            if hdu.name.upper() in ("EVENTS", "EVTS"):
                events_hdu = hdu
                break

        if events_hdu is None:
            if len(hdul) > 1:
                events_hdu = hdul[1]
                logger.warning(f"No EVENTS extension found, using extension 1: {events_hdu.name}")
            else:
                raise ValueError(f"No event data found in {path}")

        data = events_hdu.data
        columns = [col.name.upper() for col in events_hdu.columns]

        time_col = None
        for candidate in ("TIME", "TIME_TAG", "BARYTIME"):
            if candidate in columns:
                time_col = candidate
                break
        if time_col is None:
            raise ValueError(f"No time column found in {path}. Columns: {columns}")

        times = np.array(data[time_col], dtype=np.float64)

        energies = None
        for candidate in ("PI", "PHA", "ENERGY", "E"):
            if candidate in columns:
                energies = np.array(data[candidate], dtype=np.float64)
                break

        gti_start = np.array([], dtype=np.float64)
        gti_stop = np.array([], dtype=np.float64)
        # NICER uses STDGTI or STDGTI04 depending on the processing version
        for hdu in hdul:
            if hdu.name.upper() in ("GTI", "STDGTI", "STDGTI04"):
                gti_start = np.array(hdu.data["START"], dtype=np.float64)
                gti_stop = np.array(hdu.data["STOP"], dtype=np.float64)
                logger.info(
                    f"Loaded {len(gti_start)} GTI intervals, "
                    f"total exposure: {np.sum(gti_stop - gti_start):.1f}s"
                )
                break

        logger.info(
            f"Loaded {len(times)} events, time range: "
            f"[{times.min():.3f}, {times.max():.3f}]"
        )

        return {
            "times": times,
            "energies": energies,
            "gti_start": gti_start,
            "gti_stop": gti_stop,
            "header": dict(primary_header),
            "instrument": instrument,
            "telescope": telescope,
        }


def filter_energy_band(times, energies, e_min=2.0, e_max=250.0):
    """Filter photon events by energy band."""
    if energies is None:
        logger.warning("No energy or PI data available, returning all events")
        return times

    mask = (energies >= e_min) & (energies <= e_max)
    filtered = times[mask]
    logger.info(
        f"Energy filter [{e_min}, {e_max}]: "
        f"{len(times)} -> {len(filtered)} events ({100 * len(filtered) / len(times):.1f}%)"
    )
    return filtered


def apply_gti(times, gti_start, gti_stop):
    """Keep only events inside good time intervals."""
    if len(gti_start) == 0:
        logger.warning("No GTI data available, returning all events")
        return times

    mask = np.zeros(len(times), dtype=bool)
    for start, stop in zip(gti_start, gti_stop):
        mask |= (times >= start) & (times <= stop)

    filtered = times[mask]
    logger.info(
        f"GTI filter: {len(times)} -> {len(filtered)} events "
        f"({100 * len(filtered) / len(times):.1f}%)"
    )
    return filtered


def merge_observations(file_list: List[str]) -> np.ndarray:
    """Load and merge multiple FITS event files into a single time series."""
    all_times = []
    for fpath in file_list:
        try:
            data = load_fits_events(fpath)
            times = apply_gti(data["times"], data["gti_start"], data["gti_stop"])
            all_times.append(times)
            logger.info(f"  Merged {len(times)} events from {Path(fpath).name}")
        except Exception as exc:
            logger.error(f"  Failed to load {fpath}: {exc}")

    if not all_times:
        raise ValueError("No events loaded from any file")

    merged = np.sort(np.concatenate(all_times))
    logger.info(f"Total merged events: {len(merged)}")
    return merged
