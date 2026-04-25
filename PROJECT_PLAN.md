# SGR 1935+2154 Project Plan

## Core Question
Was the April 2020 FRB burst actually anomalous compared to the rest of the X-ray storm? 
Lots of papers look at the storm statistics (Younes) or the FRB burst (Li, Mereghetti), but no one seems to put them together in a single statistical pipeline. I want to build a reproducible pipeline to test this.

## Data Sources
- NICER data from HEASARC (obsid 3020560101 mainly)
- 0.5-10 keV band

## Pipeline Steps
1. **Detection**: Use `astropy.stats.bayesian_blocks` instead of fixed binning to find bursts. Should avoid the splitting issues with fixed bins.
2. **Stats**: Fit energy distributions using `powerlaw` MLE.
3. **Waiting Times**: Check if arrival times are Poisson or clustered (Weibull/lognormal).
4. **Anomaly Test**: Inject the Li et al. FRB burst properties. Where does it sit in the distribution? Is it just a sample max, or an actual outlier from the power law?
5. **SOC Check**: Check if energy/duration relations match Self-Organized Criticality expectations.

## Target Outputs
- Lightcurve with BB blocks
- Energy distribution plot (with FRB highlighted)
- Waiting times / clustering plot
- SOC consistency plot

## Rough Timeline
- Week 1-2: FITS processing and Bayesian Blocks
- Week 3-4: Energy/duration fitting
- Week 5: FRB anomaly statistics 
- Week 6: SOC checks and robustness
- Week 7-8: Writing manuscript
