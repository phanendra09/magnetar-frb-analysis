"""Quick test script to inspect downloaded FITS files."""
from astropy.io import fits
import numpy as np

# Load the FRB-day observation (April 28, 2020)
filepath = "data/raw/ni3020560101_cl.evt.gz"
print(f"Loading: {filepath}")
print()

with fits.open(filepath) as hdul:
    print("=== HDU STRUCTURE ===")
    hdul.info()
    print()
    
    # Events extension
    events = hdul[1]
    print(f"Extension name: {events.name}")
    print(f"Columns: {[c.name for c in events.columns]}")
    print(f"Number of events: {len(events.data)}")
    print()
    
    times = events.data["TIME"]
    pi = events.data["PI"]
    print(f"Time range: [{times.min():.3f}, {times.max():.3f}] s")
    print(f"Duration: {times.max() - times.min():.1f} s")
    print(f"PI range: [{pi.min()}, {pi.max()}]")
    print(f"Mean count rate: {len(times) / (times.max() - times.min()):.1f} counts/s")
    
    # GTI
    for hdu in hdul:
        if "GTI" in hdu.name.upper():
            gti = hdu.data
            print(f"\nGTI intervals: {len(gti)}")
            total_exp = np.sum(gti["STOP"] - gti["START"])
            print(f"Total exposure: {total_exp:.1f} s")
            break
    
    # Header info
    hdr = hdul[0].header
    print(f"\nTelescope: {hdr.get('TELESCOP', '?')}")
    print(f"Instrument: {hdr.get('INSTRUME', '?')}")
    print(f"Object: {hdr.get('OBJECT', '?')}")
    print(f"Date-obs: {hdr.get('DATE-OBS', '?')}")
    print(f"Date-end: {hdr.get('DATE-END', '?')}")
