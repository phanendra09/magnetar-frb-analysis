"""
fetch.py - HEASARC data acquisition helpers
===========================================
Query and save observation manifests relevant to the SGR 1935+2154
FRB 200428 campaign.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SGR1935_COORDS = {"ra_deg": 293.732, "dec_deg": 21.896}

SGR1935_OBSERVATIONS = {
    "nicer": {
        "description": "NICER observations of SGR 1935+2154 in April-May 2020",
        "obs_ids": [
            "3020560101",
            "3020560102",
            "3020560103",
            "3020560104",
            "3020560105",
            "3020560106",
        ],
    },
    "fermi_gbm": {
        "description": "Fermi-GBM triggered bursts from SGR 1935+2154",
        "catalog": "fermigtrig",
    },
    "swift_bat": {
        "description": "Swift-BAT observations near SGR 1935+2154",
        "catalog": "swiftmastr",
    },
}

FRB_200428_INFO = {
    "utc_time": "2020-04-28T14:34:24",
    "mjd": 58967.60722,
    "description": "FRB 200428 - first confirmed Galactic FRB from SGR 1935+2154",
}


def _date_mask(df: pd.DataFrame, column: str, start_mjd: float, end_mjd: float) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    return values.between(start_mjd, end_mjd, inclusive="both")


def query_campaign_manifests(
    start_mjd: float = 58940.0,
    end_mjd: float = 59000.0,
    radius_deg: float = 0.5,
) -> Dict[str, pd.DataFrame]:
    """
    Query HEASARC for Swift-BAT and Fermi-GBM manifests around the source.

    NICER is handled as a curated campaign manifest because the relevant
    observation IDs are already known and public.
    """
    from astroquery.heasarc import Heasarc
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    heasarc = Heasarc()
    coord = SkyCoord(SGR1935_COORDS["ra_deg"], SGR1935_COORDS["dec_deg"], unit="deg")

    manifests: Dict[str, pd.DataFrame] = {}

    nicer_rows = []
    for obs_id in SGR1935_OBSERVATIONS["nicer"]["obs_ids"]:
        month_folder = "2020_04" if obs_id in {"3020560101", "3020560102", "3020560103"} else "2020_05"
        nicer_rows.append(
            {
                "obsid": obs_id,
                "instrument": "NICER",
                "campaign_window": "2020-04 to 2020-05",
                "public_event_hint": (
                    "https://heasarc.gsfc.nasa.gov/FTP/nicer/data/obs/"
                    f"{month_folder}/{obs_id}/xti/event_cl/"
                ),
            }
        )
    manifests["nicer"] = pd.DataFrame(nicer_rows)

    for key, meta in [("swift_bat", "swiftmastr"), ("fermi_gbm", "fermigtrig")]:
        try:
            table = heasarc.query_region(coord, catalog=meta, radius=radius_deg * u.deg, maxrec=500)
            df = table.to_pandas() if table is not None else pd.DataFrame()
            if key == "swift_bat" and not df.empty:
                df = df[_date_mask(df, "start_time", start_mjd, end_mjd)].copy()
            elif key == "fermi_gbm" and not df.empty:
                df = df[_date_mask(df, "trigger_time", start_mjd, end_mjd)].copy()
            manifests[key] = df.reset_index(drop=True)
            logger.info(f"Queried {key}: {len(manifests[key])} rows in campaign window")
        except Exception as exc:
            logger.error(f"Failed to query {key}: {exc}")
            manifests[key] = pd.DataFrame()

    return manifests


def save_campaign_manifests(
    output_dir: str = "data/manifests",
    start_mjd: float = 58940.0,
    end_mjd: float = 59000.0,
) -> Dict[str, str]:
    """Query and save campaign manifests as CSV files."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifests = query_campaign_manifests(start_mjd=start_mjd, end_mjd=end_mjd)
    written = {}
    for name, df in manifests.items():
        path = outdir / f"{name}_campaign_manifest.csv"
        df.to_csv(path, index=False)
        written[name] = str(path)
        logger.info(f"Saved {name} manifest to {path}")
    return written


def download_heasarc_data(
    output_dir: str = "data/manifests",
    instruments: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Save the public observation manifests relevant to this project.

    This intentionally focuses on campaign discovery and reproducibility.
    The local NICER event files used by the manuscript still live in
    `data/raw/`.
    """
    written = save_campaign_manifests(output_dir=output_dir)
    if instruments is None:
        return list(written.values())
    return [written[name] for name in instruments if name in written]


def get_manual_download_instructions() -> str:
    """Return plain-text manual download instructions for the campaign."""
    return (
        "SGR 1935+2154 data download instructions\n"
        "----------------------------------------\n"
        "1. Go to https://heasarc.gsfc.nasa.gov/cgi-bin/W3Browse/\n"
        "2. NICER data:\n"
        "   - Table: nicermaster\n"
        "   - ObsIDs: 3020560101 to 3020560104 are the core campaign files in this repo\n"
        "   - Download cleaned event files (*.evt.gz) into data/raw/\n"
        "3. Swift-BAT data:\n"
        "   - Table: swiftmastr\n"
        "   - Search around SGR 1935+2154 for April-May 2020\n"
        "4. Fermi-GBM data:\n"
        "   - Table: fermigtrig\n"
        "   - Search around SGR 1935+2154 for April-May 2020 triggers\n"
        "5. You can also run pipeline.fetch.save_campaign_manifests() to generate\n"
        "   CSV manifests for the public campaign observations.\n"
        "FRB 200428 time: 2020-04-28T14:34:24 UTC\n"
    )
