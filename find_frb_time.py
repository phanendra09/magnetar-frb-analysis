"""Find the FRB 200428 time in NICER mission elapsed time."""

from astropy.io import fits
from astropy.time import Time
import numpy as np

with fits.open("data/raw/ni3020560101_cl.evt.gz") as hdul:
    hdr = hdul[1].header

    mjdrefi = hdr.get("MJDREFI", 56658)
    mjdreff = hdr.get("MJDREFF", 0.0)
    timezero = hdr.get("TIMEZERO", 0.0)

    print(f"MJDREFI: {mjdrefi}")
    print(f"MJDREFF: {mjdreff}")
    print(f"TIMEZERO: {timezero}")
    print(f"TIMESYS: {hdr.get('TIMESYS', '?')}")
    print(f"TIMEREF: {hdr.get('TIMEREF', '?')}")
    print(f"TSTART: {hdr.get('TSTART', '?')}")
    print(f"TSTOP: {hdr.get('TSTOP', '?')}")

    mjd_ref = mjdrefi + mjdreff
    t_ref = Time(mjd_ref, format="mjd", scale="tt")
    print(f"\nReference epoch: MJD {mjd_ref} = {t_ref.iso}")

    frb_utc = Time("2020-04-28T14:34:24", scale="utc")
    frb_tt = frb_utc.tt
    frb_met = (frb_tt.mjd - mjd_ref) * 86400.0

    print(f"\nFRB 200428 UTC: {frb_utc.iso}")
    print(f"FRB 200428 TT:  {frb_tt.iso}")
    print(f"FRB 200428 MET: {frb_met:.3f}")

    times = hdul[1].data["TIME"]
    print(f"\nData time range: [{times.min():.3f}, {times.max():.3f}]")
    inside = times.min() <= frb_met <= times.max()
    print(f"FRB MET is {'INSIDE' if inside else 'OUTSIDE'} the data")

    idx = np.argmin(np.abs(times - frb_met))
    print(f"Closest event to FRB: t={times[idx]:.3f}, delta={times[idx] - frb_met:.3f}s")

    window = 10
    near_frb = np.sum((times >= frb_met - window) & (times <= frb_met + window))
    print(f"Events within +/-{window}s of FRB: {near_frb}")
    print(f"Rate near FRB: {near_frb / (2 * window):.1f} counts/s")

    bg_mask = (times >= times.min()) & (times <= times.min() + 100)
    bg_events = np.sum(bg_mask)
    print(f"Background rate (first 100s): {bg_events / 100:.1f} counts/s")
