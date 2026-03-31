# Title
Exoplanet Verification and Analysis for TIC 165202476

## Research Goal
To gather data from the TOI catalog using the MAST database, specifically targeting TIC 165202476. We aim to rigorously verify and analyze this data to confidently confirm or refute its status as an exoplanet. The analysis will include spatial pixel-level verification and Bayesian inference to definitively rule out false positives and confidently constrain planetary parameters.

## Domain Context
The analysis deals with identifying and confirming planetary transits from photometric time-series data. 
The core guiding principle of this project is strict **reproducibility**: a competent scientist in the field must be able to read the methodology and repeat the analysis from scratch. If a technique is standard and well-documented, the reference must be cited rather than re-explaining it from scratch.

## Inputs / Resources
- **Target Identifier:** TIC 165202476
- **Data Source:** MAST (Mikulski Archive for Space Telescopes) database. Because a 1D light curve dataset alone is insufficient to definitively rule out astrophysical false positives, we will specifically augment the standard Pre-search Data Conditioning Simple Aperture Photometry (PDCSAP) light curves by downloading 2D Target Pixel Files (TPFs) and Full Frame Images (FFIs).
- **Software/Tools:** Python, `lightkurve`, `astropy`, `astroquery`, and the `allesfitter` module (to perform MCMC and Bayesian inference).

## Success Criteria
- Obtain a conclusive, evidence-based determination of the planetary status of TIC 165202476.
- Definitively rule out false positives (e.g., EBs and BEBs) using comprehensive FITS file aperture analysis.
- Complete and transparent documentation of the end-to-end methodology.

## Constraints
- **Strict Reproducibility:** Must exactingly document what data was used (e.g., data versions, timestamps), what software was written/used, and which parameters were chosen and why.
- **Complication Handling:** Any data anomalies or complications encountered must be explicitly documented along with the steps taken to resolve them.
- **Excluded References:** Must explicitly avoid using or citing the following recent papers for this analysis: [A&A 2026 Paper](https://www.aanda.org/articles/aa/pdf/2026/03/aa57656-25.pdf) and [arXiv:2510.11528v1](https://arxiv.org/html/2510.11528v1).
- Standard techniques must be cited appropriately referencing permitted peer-reviewed literature rather than re-explained.

## Deliverables
- Fully reproducible analysis scripts or Jupyter notebooks.
- A descriptive methodology report tracking aperture analysis logic, parameter inferences, MCMC trace plots, and data provenance.
- The final verification results, declaring and scientifically justifying the status of TIC 165202476.

## Verification Expectations
- **Ruling out False Positives via Spatial Analysis:** To verify astrophysical false positives without relying solely on a 1D time-series dataset, we will perform FITS file aperture analysis utilizing the downloaded Target Pixel Files (TPFs) from MAST. By examining the individual pixels within and surrounding the target aperture, we can track the flux centroid during the transit event. If nearby stars are actually the source of the dimming (causing a noticeable centroid shift during transit), we can confidently rule out Eclipsing Binaries (EBs) and Background Eclipsing Binaries (BEBs).
- **Bayesian Inference for Orbital Parameters:** We will implement and run a Markov Chain Monte Carlo (MCMC) algorithm using the `allesfitter` module. This will leverage Bayesian inference to rigorously extract the true physical parameters for the orbit (e.g., radius ratio, inclination, semi-major axis, transit epoch) and accurately quantify their uncertainties.
- Another researcher must be able to take the raw dataset from MAST and, following our documented deliverables, arrive at the exact same physical parameters, MCMC posteriors, and final conclusion.

## Known Unknowns
- The actual nature of the transit signals (whether genuinely planetary or mimicking phenomena like background eclipsing binaries).
- Potential data quality or systematics issues specific to the individual light curves or pixel masks of TIC 165202476.
- Baseline flux contamination parameters and detrending specifics that will only become clear during modeling.
