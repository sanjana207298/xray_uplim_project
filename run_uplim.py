#!/usr/bin/env python
"""
run_uplim.py
------------
Command-line entry point for nustar_uplim.

Edit the CONFIG block below and run:
    python run_uplim.py

Or import and call from a notebook:
    from nustar_uplim import run_uplim
    run_uplim(base_path=..., obsid=..., ra=..., dec=...)
"""

from nustar_uplim import run_uplim

# =============================================================================
# CONFIG  — edit this block
# =============================================================================

BASE_PATH = "/Users/sanjanagupta/Documents/data/NuSTAR/2017gas/"
OBSID     = "80202052002"

RA_INPUT  = "20:17:11.360"    # "HH:MM:SS.ss"  or decimal degrees
DEC_INPUT = "+58:12:08.10"    # "±DD:MM:SS.ss" or decimal degrees

SRC_RADIUS_ARCSEC = 60.0      # NuSTAR EEF: ~50% at 20", ~60% at 30", ~80% at 60"
BKG_RADIUS_ARCSEC = 200.0     # outer radius of background annulus
BKG_INNER_FACTOR  = 1.2       # inner radius = SRC_RADIUS * this

PSF_FWHM_ARCSEC   = 18.0      # Harrison+13; increase for off-axis sources

ENERGY_BAND       = 'full'    # 'full' | 'soft' | 'hard' | 'ultrahard'
                               # or a custom tuple e.g. (8.0, 30.0)

MODULES           = ['A', 'B']

BKG_MODE          = 'annulus' # 'annulus' or 'manual'
BKG_RA            = ""        # only used if BKG_MODE = 'manual'
BKG_DEC           = ""

EXP_STAT          = 'median'  # 'median' | 'mean' | 'psf_weighted'

CONFIDENCE_LEVELS = [0.9545, 0.9973]  # ~2-sigma and ~3-sigma

SAVE_PLOTS        = True

# =============================================================================

if __name__ == "__main__":
    run_uplim(
        base_path         = BASE_PATH,
        obsid             = OBSID,
        ra                = RA_INPUT,
        dec               = DEC_INPUT,
        src_radius_arcsec = SRC_RADIUS_ARCSEC,
        bkg_radius_arcsec = BKG_RADIUS_ARCSEC,
        bkg_inner_factor  = BKG_INNER_FACTOR,
        psf_fwhm_arcsec   = PSF_FWHM_ARCSEC,
        energy_band       = ENERGY_BAND,
        modules           = MODULES,
        bkg_mode          = BKG_MODE,
        bkg_ra            = BKG_RA,
        bkg_dec           = BKG_DEC,
        exp_stat          = EXP_STAT,
        confidence_levels = CONFIDENCE_LEVELS,
        save_plots        = SAVE_PLOTS,
    )
