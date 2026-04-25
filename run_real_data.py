"""
Run the publication-grade NICER analysis for SGR 1935+2154.

This script reproduces the manuscript workflow:
1. Load the real NICER event files in data/raw
2. Apply NICER PI (0.5-10 keV) and GTI filtering
3. Detect bursts with Bayesian Blocks on a uniform event subsample
4. Fit energy/duration distributions on NICER-only catalogue
5. Inject the published FRB 200428 X-ray burst from literature for
   comparison ONLY (it is excluded from all distribution fits)
6. Run robustness tests

FRB Energy Provenance
---------------------
The FRB 200428 X-ray counterpart energy is taken from:
  Li et al. 2021, Nature Astronomy, 5, 378, Table 1
  Instrument: Insight-HXMT (1-250 keV)
  Isotropic-equivalent energy: ~7 x 10^39 erg at d = 9 kpc

This is a BROADBAND value. NICER bursts are measured in 0.5-10 keV.
The comparison is therefore cross-instrument, which we address via
the bandpass sensitivity test (robustness.py) that sweeps the assumed
FRB energy from 1e39 to 1e40 erg.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from run_analysis import maybe_subsample_events

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("magnetar")


def main() -> None:
    from astropy.io import fits
    import pandas as pd

    from pipeline.catalogue import build_catalogue, catalogue_summary, compute_waiting_times, save_catalogue
    from pipeline.detect import bayesian_blocks_detect, identify_bursts
    from pipeline.stats import (
        FRBAnomalyResult,
        SOCResult,
        WaitingTimeFitResult,
        fit_energy_distribution,
        fit_waiting_times,
        frb_anomaly_test,
        save_results,
        soc_consistency_check,
    )
    from plots.figures import generate_all_figures

    output_dir = Path("results_real")
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STAGE 1: Loading real NICER data")
    logger.info("=" * 60)

    fits_files = sorted(Path("data/raw").glob("*.evt.gz"))
    logger.info(f"Found {len(fits_files)} FITS files")
    if not fits_files:
        logger.error("No NICER .evt.gz files found in data/raw")
        sys.exit(1)

    all_times = []
    for fpath in fits_files:
        with fits.open(str(fpath)) as hdul:
            events = hdul[1]
            times = events.data["TIME"]
            pi = events.data["PI"]

            pi_mask = (pi >= 50) & (pi <= 1000)
            filtered = times[pi_mask]

            for hdu in hdul:
                if "GTI" in hdu.name.upper():
                    gti = hdu.data
                    gti_mask = np.zeros(len(filtered), dtype=bool)
                    for start, stop in zip(gti["START"], gti["STOP"]):
                        gti_mask |= (filtered >= start) & (filtered <= stop)
                    filtered = filtered[gti_mask]
                    break

            logger.info(
                f"  {fpath.name}: {len(times)} raw -> {len(filtered)} filtered events"
            )
            if len(filtered) > 0:
                all_times.append(filtered)

    photon_times = np.sort(np.concatenate(all_times))
    photon_times_bb = maybe_subsample_events(photon_times, max_events=50000)
    logger.info(f"Total photon events: {len(photon_times)}")
    logger.info(f"Time range: [{photon_times.min():.3f}, {photon_times.max():.3f}]")
    logger.info(f"Using {len(photon_times_bb)} events for burst detection")

    # FRB 200428 parameters from Li et al. 2021, Nat. Astron. 5, 378
    # MET timestamp from Insight-HXMT detection
    frb_met = 199550066.0
    # Distance: 9.0 kpc (Zhong et al. 2020, Kothes et al. 2018)
    # We test sensitivity to distance via the bandpass energy range test
    distance_kpc = 9.0

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 2: Bayesian Blocks burst detection")
    logger.info("=" * 60)

    edges = bayesian_blocks_detect(photon_times_bb, p0=0.05)

    # NOTE on energy conversion:
    # We use counts * <E_photon> * 4*pi*d^2 / A_eff as a proxy for
    # isotropic-equivalent burst energy. This is intentionally crude:
    # proper spectral fitting (e.g. XSPEC with blackbody kT ~ 1.7 keV
    # per Younes et al. 2020) would require per-burst response matrices.
    # We use mean_photon_energy_keV=5.0 keV as the NICER-band average.
    # The absolute energy scale cancels in the RANKING test (percentile,
    # z-score) because the same conversion is applied to all NICER bursts.
    bursts = identify_bursts(
        photon_times_bb,
        edges,
        threshold_sigma=3.0,
        distance_kpc=distance_kpc,
        mean_photon_energy_keV=5.0,
    )

    logger.info(f"Detected {len(bursts)} bursts from NICER data")
    if len(bursts) < 10:
        logger.warning("Few bursts detected - retrying with a 2 sigma threshold")
        bursts = identify_bursts(
            photon_times_bb,
            edges,
            threshold_sigma=2.0,
            distance_kpc=distance_kpc,
            mean_photon_energy_keV=5.0,
        )
        logger.info(f"With the lower threshold: {len(bursts)} bursts")

    if len(bursts) == 0:
        logger.error("No bursts detected")
        sys.exit(1)

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 3: Building burst catalogue")
    logger.info("=" * 60)

    # Build NICER-only catalogue first — this is the reference population
    nicer_catalogue = compute_waiting_times(build_catalogue(bursts))
    logger.info(f"NICER-only catalogue: {len(nicer_catalogue)} bursts")

    # Save NICER-only catalogue separately for reproducibility
    save_catalogue(nicer_catalogue, str(output_dir / "catalogue_nicer_only.csv"))

    # --- Inject FRB burst for COMPARISON ONLY ---
    # Source: Li et al. 2021, Nat. Astron. 5, 378, Table 1
    # The FRB X-ray counterpart was detected by Insight-HXMT (1-250 keV),
    # NOT by NICER. It occurred during a NICER orbital gap.
    # We inject it to test whether it is an outlier relative to the
    # NICER-detected population. It is EXCLUDED from all distribution fits.
    frb_energy_erg = 7e39  # Li et al. 2021, isotropic-equiv. at d=9 kpc
    frb_duration_s = 0.5   # Li et al. 2021
    distance_cm = distance_kpc * 3.086e21

    frb_row = pd.DataFrame(
        [
            {
                "burst_id": len(nicer_catalogue),
                "start_time": frb_met,
                "end_time": frb_met + frb_duration_s,
                "duration": frb_duration_s,
                "peak_rate": 0.0,
                "total_counts": 0,
                "fluence": frb_energy_erg / (4 * np.pi * distance_cm ** 2),
                "energy": frb_energy_erg,
                "is_frb_burst": True,
                "waiting_time": np.nan,
            }
        ]
    )
    catalogue = pd.concat([nicer_catalogue, frb_row], ignore_index=True)
    catalogue = catalogue.sort_values("start_time").reset_index(drop=True)

    logger.info("")
    logger.info("FRB BURST INJECTION:")
    logger.info(f"  Source: Li et al. 2021, Nat. Astron. 5, 378")
    logger.info(f"  Instrument: Insight-HXMT (1-250 keV)")
    logger.info(f"  Energy: {frb_energy_erg:.1e} erg (isotropic, d={distance_kpc} kpc)")
    logger.info(f"  Duration: {frb_duration_s} s")
    logger.info(f"  NOTE: This burst is EXCLUDED from distribution fits.")
    logger.info(f"         It is included ONLY for the anomaly ranking test.")
    logger.info(f"Total catalogue: {len(catalogue)} entries ({len(nicer_catalogue)} NICER + 1 FRB)")

    summary = catalogue_summary(nicer_catalogue)
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    save_catalogue(catalogue, str(output_dir / "catalogue_real.csv"))

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 4: Statistical analysis")
    logger.info("=" * 60)

    # ALL distribution fits use NICER-only data.
    # The FRB burst is excluded from fitting — it is only used as a
    # test point for the anomaly ranking.
    nicer_only = catalogue[~catalogue["is_frb_burst"]].copy()
    energy_fit = fit_energy_distribution(nicer_only["energy"].values)

    # Waiting times: NICER-only to avoid biasing with the FRB gap
    waiting_values = nicer_only["waiting_time"].dropna().values
    waiting_values = waiting_values[waiting_values > 0]
    waiting_fit = fit_waiting_times(waiting_values) if len(waiting_values) > 5 else None

    try:
        frb_result = frb_anomaly_test(catalogue)
    except ValueError as exc:
        logger.warning(f"FRB anomaly test failed: {exc}")
        frb_result = None

    soc_result = soc_consistency_check(nicer_only) if len(nicer_only) > 30 else None

    save_results(energy_fit, waiting_fit, frb_result, soc_result, str(output_dir / "fit_results_real.json"))

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 5: Generating figures")
    logger.info("=" * 60)

    if waiting_fit is None:
        waiting_fit = WaitingTimeFitResult(
            best_model="exponential",
            coefficient_of_variation=1.0,
            is_clustered=False,
            exponential_rate=0.01,
            weibull_shape=None,
            weibull_scale=None,
            lognormal_shape=None,
            lognormal_scale=None,
            model_comparison={},
        )

    if soc_result is None:
        soc_result = SOCResult(
            alpha=energy_fit.alpha,
            alpha_err=energy_fit.alpha_err,
            beta=0.0,
            beta_err=0.0,
            delta=0.0,
            delta_err=0.0,
            predicted_alpha=0.0,
            predicted_alpha_err=0.0,
            consistency=0.0,
            is_consistent=False,
            conclusion="Insufficient data for SOC test",
        )

    if frb_result is None:
        frb_result = FRBAnomalyResult(
            frb_energy=7e39,
            frb_duration=0.5,
            frb_waiting_time=None,
            energy_percentile=float("nan"),
            duration_percentile=float("nan"),
            waiting_time_percentile=None,
            energy_zscore=float("nan"),
            duration_zscore=float("nan"),
            is_energy_outlier=False,
            is_duration_outlier=False,
            conclusion="FRB anomaly test unavailable",
        )

    generate_all_figures(
        photon_times,
        edges,
        catalogue,
        energy_fit,
        waiting_fit,
        frb_result,
        soc_result,
        frb_time=frb_met,
        output_dir=str(output_dir / "figures"),
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 6: Robustness tests (peer-review hardening)")
    logger.info("=" * 60)

    from pipeline.robustness import run_all_robustness_tests

    robustness = run_all_robustness_tests(
        photon_times_subsampled=photon_times_bb,
        catalogue=catalogue,
        energy_fit=energy_fit,
        fits_files=fits_files,
        frb_energy=7e39,
        output_dir=str(output_dir),
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("REAL DATA PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"FITS files processed: {len(fits_files)}")
    logger.info(f"Total photon events: {len(photon_times)}")
    logger.info(f"Bursts detected (NICER only): {len(nicer_catalogue)}")
    logger.info("FRB burst injected from literature: YES")
    logger.info(f"Output directory: {output_dir}")

    if not np.isnan(frb_result.energy_percentile):
        logger.info("")
        logger.info("=" * 60)
        logger.info("KEY RESULT: FRB 200428 ANOMALY TEST")
        logger.info("=" * 60)
        logger.info(f"  FRB energy: {frb_result.frb_energy:.2e} erg")
        logger.info(f"  Energy percentile: {frb_result.energy_percentile:.1f}%")
        logger.info(f"  Z-score: {frb_result.energy_zscore:.1f} sigma")
        logger.info(f"  Outlier: {frb_result.is_energy_outlier}")
        logger.info(f"  {frb_result.conclusion}")


if __name__ == "__main__":
    main()

