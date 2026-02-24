"""
nustar_uplim.config
-------------------
Single dataclass holding every user-facing parameter.
Pass a Config instance to run_uplim() or process_module().
"""

from dataclasses import dataclass, field
from typing import List, Union, Tuple


@dataclass
class Config:
    """
    All user-configurable parameters for a NuSTAR upper-limit calculation.

    Parameters
    ----------
    base_path : str
        Root directory containing the observation folder.
    obsid : str
        NuSTAR observation ID (folder name inside base_path).
    ra : str or float
        Source right ascension.  Accepts "HH:MM:SS.ss" or decimal degrees.
    dec : str or float
        Source declination.  Accepts "±DD:MM:SS.ss" or decimal degrees.
    src_radius_arcsec : float
        Radius of the source extraction circle in arcseconds.
        NuSTAR EEF: ~50% at 20", ~60% at 30", ~80% at 60".
        Divide the final count-rate upper limit by the EEF to correct for
        flux outside the aperture.
    bkg_radius_arcsec : float
        Outer radius of the background annulus in arcseconds.
        Inner radius is set automatically to src_radius_arcsec * bkg_inner_factor.
    bkg_inner_factor : float
        Inner radius of background annulus = src_radius_arcsec * this value.
        Default 1.2 leaves a buffer to exclude PSF wings from the background.
    psf_fwhm_arcsec : float
        PSF FWHM in arcseconds.  Used only for the PSF-weighted exposure
        diagnostic and the radial profile plot.
        Harrison et al. 2013 (ApJ 770, 103): on-axis FWHM ~ 18".
        Increase to ~20-25" for sources more than ~2' off-axis.
    energy_band : str or tuple
        Named band: 'full' (3-79 keV), 'soft' (3-10 keV),
                    'hard' (10-30 keV), 'ultrahard' (30-79 keV).
        Custom band: tuple of (e_lo_kev, e_hi_kev), e.g. (8.0, 30.0).
    modules : list of str
        FPMs to process.  Any subset of ['A', 'B'].
    bkg_mode : str
        'annulus' — automatic annulus centred on the source (recommended).
        'manual'  — user-supplied background circle; set bkg_ra / bkg_dec.
    bkg_ra : str or float
        Background circle centre RA.  Only used when bkg_mode='manual'.
    bkg_dec : str or float
        Background circle centre Dec.  Only used when bkg_mode='manual'.
    exp_stat : str
        Statistic used to summarise exposure-map pixels inside the source
        aperture into a single effective exposure time.
        'median'       — recommended for non-detections; robust, no PSF assumption.
        'mean'         — fine when vignetting variation across aperture is small.
        'psf_weighted' — diagnostic only; unreliable for off-axis sources.
        All three values are always printed regardless of this choice.
    confidence_levels : list of float
        One-sided confidence levels for the upper limits.
        Common choices (Gaussian convention):
            0.9000 → 1.28σ   0.9500 → 1.64σ
            0.9545 ≈ 2σ      0.9973 ≈ 3σ
    save_plots : bool
        Whether to save diagnostic plots to <base_path>/<obsid>/ul_products/.
    """

    # -- Observation ----------------------------------------------------------
    base_path : str   = ""
    obsid     : str   = ""

    # -- Source position ------------------------------------------------------
    ra  : Union[str, float] = ""
    dec : Union[str, float] = ""

    # -- Aperture sizes -------------------------------------------------------
    src_radius_arcsec : float = 60.0
    bkg_radius_arcsec : float = 200.0
    bkg_inner_factor  : float = 1.2

    # -- PSF ------------------------------------------------------------------
    psf_fwhm_arcsec : float = 18.0

    # -- Energy band ----------------------------------------------------------
    energy_band : Union[str, Tuple[float, float]] = 'full'

    # -- Modules --------------------------------------------------------------
    modules : List[str] = field(default_factory=lambda: ['A', 'B'])

    # -- Background mode ------------------------------------------------------
    bkg_mode : str              = 'annulus'
    bkg_ra   : Union[str, float] = ""
    bkg_dec  : Union[str, float] = ""

    # -- Exposure statistic ---------------------------------------------------
    exp_stat : str = 'median'

    # -- Confidence levels ----------------------------------------------------
    confidence_levels : List[float] = field(
        default_factory=lambda: [0.9545, 0.9973])

    # -- Output ---------------------------------------------------------------
    save_plots : bool = True

    # -------------------------------------------------------------------------

    ENERGY_BANDS = {
        'full':      (3.0,  79.0),
        'soft':      (3.0,  10.0),
        'hard':      (10.0, 30.0),
        'ultrahard': (30.0, 79.0),
    }

    def resolve_energy_band(self):
        """Return (e_lo, e_hi) in keV."""
        if isinstance(self.energy_band, tuple):
            return float(self.energy_band[0]), float(self.energy_band[1])
        key = self.energy_band.lower()
        if key not in self.ENERGY_BANDS:
            raise ValueError(
                f"Unknown energy_band '{self.energy_band}'. "
                f"Use one of {list(self.ENERGY_BANDS)} or a (e_lo, e_hi) tuple.")
        return self.ENERGY_BANDS[key]

    def validate(self):
        """Raise ValueError for obviously wrong settings."""
        if not self.base_path:
            raise ValueError("base_path is empty.")
        if not self.obsid:
            raise ValueError("obsid is empty.")
        if not self.ra or not self.dec:
            raise ValueError("ra and dec must be set.")
        if self.bkg_mode == 'manual' and (not self.bkg_ra or not self.bkg_dec):
            raise ValueError(
                "bkg_mode='manual' requires bkg_ra and bkg_dec to be set.")
        if self.exp_stat not in ('median', 'mean', 'psf_weighted'):
            raise ValueError(
                f"exp_stat must be 'median', 'mean', or 'psf_weighted', "
                f"not '{self.exp_stat}'.")
        for cl in self.confidence_levels:
            if not 0.0 < cl < 1.0:
                raise ValueError(
                    f"Confidence level {cl} is outside (0, 1).")
