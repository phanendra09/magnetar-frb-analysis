"""
run_analysis.py - End-to-end magnetar flare analysis pipeline
==============================================================
Orchestrates data loading -> burst detection -> catalogue building ->
statistical analysis -> figure generation.

Usage:
    python run_analysis.py --synthetic --n-bursts 500 --output results/
    python run_analysis.py --data data/raw/ --frb-time 58967.60722 --output results/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("magnetar")


def setup_paths(output_dir: str) -> Path:
    """Create the output directory structure."""
    out = Path(output_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def discover_event_files(data_dir: str | Path) -> List[Path]:
    """Discover supported FITS and event files in a data directory."""
    base = Path(data_dir)
    patterns = ("*.fits", "*.evt", "*.fits.gz", "*.evt.gz")
    discovered = []
    seen = set()
    for pattern in patterns:
        for path in sorted(base.glob(pattern)):
            if path not in seen:
                discovered.append(path)
                seen.add(path)
    return discovered


def maybe_subsample_events(photon_times: np.ndarray, max_events: int = 50000) -> np.ndarray:
    """Uniformly subsample a long event list for Bayesian Blocks speed."""
    if len(photon_times) <= max_events:
        return photon_times
    step = max(1, len(photon_times) // max_events)
    sampled = photon_times[::step]
    logger.info(f"Subsampling from {len(photon_times)} to {len(sampled)} events for Bayesian Blocks")
    return sampled


def run_synthetic_pipeline(args) -> None:
    """Run the full pipeline on synthetic data."""
    from pipeline.catalogue import build_catalogue, catalogue_summary, compute_waiting_times, save_catalogue
    from pipeline.detect import bayesian_blocks_detect, identify_bursts
    from pipeline.stats import (
        fit_energy_distribution,
        fit_waiting_times,
        frb_anomaly_test,
        save_results,
        soc_consistency_check,
    )
    from pipeline.synthetic import generate_synthetic_dataset
    from plots.figures import generate_all_figures

    out = setup_paths(args.output)

    logger.info("=" * 60)
    logger.info("STAGE 1: Generating synthetic data")
    logger.info("=" * 60)

    synth = generate_synthetic_dataset(
        n_bursts=args.n_bursts,
        alpha=args.alpha,
        beta=args.beta,
        delta=args.delta,
        seed=args.seed,
    )
    logger.info(
        f"Generated {synth.metadata['n_photons']} photons and "
        f"{synth.metadata['n_bursts']} injected bursts"
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 2: Bayesian Blocks burst detection")
    logger.info("=" * 60)

    edges = bayesian_blocks_detect(synth.photon_times, p0=args.p0)
    bursts = identify_bursts(
        synth.photon_times,
        edges,
        threshold_sigma=args.threshold,
        frb_time=synth.frb_burst_time,
    )
    logger.info(f"Detected {len(bursts)} bursts")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 3: Building burst catalogue")
    logger.info("=" * 60)

    catalogue = compute_waiting_times(build_catalogue(bursts))
    summary = catalogue_summary(catalogue)
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    save_catalogue(catalogue, str(out / "catalogue.csv"))

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 4: Statistical analysis")
    logger.info("=" * 60)

    energy_fit = fit_energy_distribution(catalogue["energy"].values)
    waiting_fit = fit_waiting_times(catalogue["waiting_time"].dropna().values)
    frb_result = frb_anomaly_test(catalogue)
    soc_result = soc_consistency_check(catalogue)

    save_results(energy_fit, waiting_fit, frb_result, soc_result, str(out / "fit_results.json"))

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 5: Generating publication figures")
    logger.info("=" * 60)

    generate_all_figures(
        photon_times=synth.photon_times,
        edges=edges,
        catalogue=catalogue,
        energy_fit=energy_fit,
        waiting_fit=waiting_fit,
        frb_result=frb_result,
        soc_result=soc_result,
        frb_time=synth.frb_burst_time,
        output_dir=str(out / "figures"),
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {out}")
    logger.info(f"Burst catalogue: {out / 'catalogue.csv'}")
    logger.info(f"Fit results: {out / 'fit_results.json'}")
    logger.info(f"Figures: {out / 'figures'}")
    logger.info(
        f"FRB anomaly result: percentile={frb_result.energy_percentile:.1f}%, "
        f"z={frb_result.energy_zscore:.1f}, outlier={frb_result.is_energy_outlier}"
    )


def run_real_data_pipeline(args) -> None:
    """Run the pipeline on real FITS or event data."""
    from pipeline.catalogue import build_catalogue, compute_waiting_times, save_catalogue
    from pipeline.detect import bayesian_blocks_detect, identify_bursts
    from pipeline.fetch import get_manual_download_instructions
    from pipeline.preprocess import apply_gti, filter_energy_band, load_fits_events
    from pipeline.stats import (
        fit_energy_distribution,
        fit_waiting_times,
        frb_anomaly_test,
        save_results,
        soc_consistency_check,
    )
    from plots.figures import generate_all_figures

    out = setup_paths(args.output)
    fits_files = discover_event_files(args.data)

    if not fits_files:
        logger.error(f"No FITS or event files found in {args.data}")
        print(get_manual_download_instructions())
        sys.exit(1)

    logger.info(f"Found {len(fits_files)} supported files in {args.data}")

    all_times = []
    instruments = set()
    for fpath in fits_files:
        try:
            data = load_fits_events(str(fpath))
            instruments.add(data["instrument"])
            times = apply_gti(data["times"], data["gti_start"], data["gti_stop"])
            if data["energies"] is not None:
                if data["instrument"].upper() == "XTI":
                    times = filter_energy_band(times, data["energies"], e_min=50, e_max=1000)
                else:
                    times = filter_energy_band(times, data["energies"], e_min=2.0, e_max=250.0)
            all_times.append(times)
        except Exception as exc:
            logger.error(f"Failed to load {fpath.name}: {exc}")

    if not all_times:
        logger.error("No photon events were loaded from the provided files")
        sys.exit(1)

    photon_times = np.sort(np.concatenate(all_times))
    photon_times_bb = maybe_subsample_events(photon_times, max_events=args.bb_max_events)
    logger.info(f"Total filtered photon events: {len(photon_times)}")
    logger.info(f"Using {len(photon_times_bb)} events for Bayesian Blocks")
    logger.info(f"Instruments seen: {', '.join(sorted(instruments))}")

    edges = bayesian_blocks_detect(photon_times_bb, p0=args.p0)
    bursts = identify_bursts(
        photon_times_bb,
        edges,
        threshold_sigma=args.threshold,
        distance_kpc=args.distance_kpc,
        mean_photon_energy_keV=args.mean_photon_energy_keV,
        frb_time=args.frb_time,
    )

    catalogue = compute_waiting_times(build_catalogue(bursts))
    save_catalogue(catalogue, str(out / "catalogue.csv"))

    energy_fit = fit_energy_distribution(catalogue["energy"].values)
    waiting_fit = None
    waiting_values = catalogue["waiting_time"].dropna().values
    if len(waiting_values) > 0:
        waiting_fit = fit_waiting_times(waiting_values)

    try:
        frb_result = frb_anomaly_test(catalogue)
    except ValueError as exc:
        logger.warning(f"FRB anomaly test skipped: {exc}")
        frb_result = None

    soc_result = soc_consistency_check(catalogue) if len(catalogue) >= 10 else None

    save_results(energy_fit, waiting_fit, frb_result, soc_result, str(out / "fit_results.json"))

    if frb_result is not None and waiting_fit is not None and soc_result is not None:
        generate_all_figures(
            photon_times=photon_times,
            edges=edges,
            catalogue=catalogue,
            energy_fit=energy_fit,
            waiting_fit=waiting_fit,
            frb_result=frb_result,
            soc_result=soc_result,
            frb_time=args.frb_time,
            output_dir=str(out / "figures"),
        )
    else:
        logger.info("Skipping full figure set because FRB or waiting-time context is unavailable")

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {out}")
    logger.info(f"Burst catalogue: {out / 'catalogue.csv'}")
    logger.info(f"Fit results: {out / 'fit_results.json'}")
    if frb_result is not None:
        logger.info(
            f"FRB anomaly result: percentile={frb_result.energy_percentile:.1f}%, "
            f"z={frb_result.energy_zscore:.1f}, outlier={frb_result.is_energy_outlier}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="SGR 1935+2154 Magnetar Flare Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --synthetic --n-bursts 500
  python run_analysis.py --data data/raw/ --frb-time 199550066
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true", help="Run with synthetic data")
    mode.add_argument("--data", type=str, help="Path to directory containing FITS or event files")

    parser.add_argument("--n-bursts", type=int, default=300, help="Number of synthetic bursts")
    parser.add_argument("--alpha", type=float, default=1.6, help="Synthetic energy power-law index")
    parser.add_argument("--beta", type=float, default=1.8, help="Synthetic duration power-law index")
    parser.add_argument("--delta", type=float, default=1.5, help="Synthetic E-T scaling exponent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--p0", type=float, default=0.01, help="Bayesian Blocks false-positive rate")
    parser.add_argument("--threshold", type=float, default=5.0, help="Burst threshold in sigma")
    parser.add_argument("--frb-time", type=float, default=None, help="FRB burst time in mission seconds")
    parser.add_argument("--distance-kpc", type=float, default=9.0, help="Source distance in kpc")
    parser.add_argument(
        "--mean-photon-energy-keV",
        type=float,
        default=30.0,
        help="Mean photon energy used for count-to-energy conversion",
    )
    parser.add_argument(
        "--bb-max-events",
        type=int,
        default=50000,
        help="Maximum number of events used for Bayesian Blocks before subsampling",
    )
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()

    logger.info("=" * 60)
    logger.info("SGR 1935+2154 MAGNETAR FRB ANOMALY PIPELINE")
    logger.info("Question: is the FRB burst anomalous or ordinary?")
    logger.info("=" * 60)

    if args.synthetic:
        run_synthetic_pipeline(args)
    else:
        run_real_data_pipeline(args)


if __name__ == "__main__":
    main()
