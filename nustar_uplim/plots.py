"""
nustar_uplim.plots
------------------
Diagnostic plots saved to <base_path>/<obsid>/ul_products/.

radial_profile()
    Log-scale radial surface-density profile of events around the source
    position.  Marks the source aperture, background annulus, and PSF
    half-FWHM.  A flat profile inside the source aperture (matching the
    background level) confirms a non-detection.

exposure_histogram()
    Distribution of exposure-map pixel values inside the source aperture,
    with vertical lines for all three summary statistics.  If median, mean,
    and PSF-weighted mean agree closely (< a few percent), the choice of
    summary statistic does not affect the final upper limit.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def radial_profile(evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
                   module, e_lo, e_hi, obsid, cfg, out_dir):
    """
    Save a log-scale radial surface-density profile.

    Parameters
    ----------
    evt_x, evt_y : array  — event pixel coordinates (energy-filtered)
    cx_evt, cy_evt : float — source pixel position in event coordinates
    pscale_evt   : float  — event pixel scale in arcsec/pix
    module       : str    — 'A' or 'B'
    e_lo, e_hi   : float  — energy band in keV
    obsid        : str
    cfg          : Config
    out_dir      : str    — output directory
    """
    r_arcsec = (np.sqrt((evt_x - cx_evt)**2 + (evt_y - cy_evt)**2)
                * pscale_evt)
    max_r    = cfg.bkg_radius_arcsec * 1.15
    bins     = np.linspace(0, max_r, 45)
    counts, edges = np.histogram(r_arcsec, bins=bins)
    mids   = 0.5 * (edges[:-1] + edges[1:])
    areas  = np.pi * (edges[1:]**2 - edges[:-1]**2)
    surf   = np.where(areas > 0, counts / areas, 0.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.step(mids, np.where(surf > 0, surf, np.nan),
            where='mid', color='steelblue', lw=1.5, label='Binned events')

    r_inner = cfg.src_radius_arcsec * cfg.bkg_inner_factor
    ax.axvline(cfg.src_radius_arcsec, color='tomato', ls='--', lw=1.3,
               label=f'Src aperture ({cfg.src_radius_arcsec:.0f}")')
    ax.axvline(r_inner, color='darkorange', ls=':', lw=1.2,
               label=f'Bkg inner ({r_inner:.0f}")')
    ax.axvline(cfg.bkg_radius_arcsec, color='darkorange', ls='--', lw=1.2,
               label=f'Bkg outer ({cfg.bkg_radius_arcsec:.0f}")')
    ax.axvline(cfg.psf_fwhm_arcsec / 2.0, color='grey', ls=':', lw=1.0,
               label=f'PSF half-FWHM ({cfg.psf_fwhm_arcsec/2:.0f}")')

    ax.set_xlabel('Radius (arcsec)', fontsize=12)
    ax.set_ylabel('Surface density (cts arcsec$^{-2}$)', fontsize=12)
    ax.set_title(
        f'NuSTAR FPM{module}  |  {e_lo:.0f}-{e_hi:.0f} keV  |  OBSID {obsid}',
        fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_yscale('log')
    ax.set_xlim(0, max_r)
    fig.tight_layout()

    fname = os.path.join(
        out_dir, f"nustar_radial_FPM{module}_{e_lo:.0f}-{e_hi:.0f}keV.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Radial profile plot  -> {fname}")


def exposure_histogram(meta, exp_stats, module, cfg, out_dir):
    """
    Save a histogram of exposure-map pixel values inside the source aperture.

    Parameters
    ----------
    meta      : dict   — from compute_exposure_stats()
    exp_stats : dict   — {'median': float, 'mean': float, 'psf_weighted': float}
    module    : str
    cfg       : Config
    out_dir   : str
    """
    vals = meta['exp_values']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals / 1e3, bins=30, color='steelblue',
            edgecolor='white', linewidth=0.5, alpha=0.8,
            label='Exposure-map pixels in aperture')

    styles = {
        'median':       ('tomato',     '--', 'Median'),
        'mean':         ('darkorange', '--', 'Mean'),
        'psf_weighted': ('purple',     ':',  'PSF-wtd mean'),
    }
    for key, (col, ls, lbl) in styles.items():
        tag = '  [PRIMARY]' if key == cfg.exp_stat else ''
        ax.axvline(exp_stats[key] / 1e3, color=col, ls=ls, lw=1.8,
                   label=f"{lbl} = {exp_stats[key]/1e3:.2f} ks{tag}")

    ax.set_xlabel('Exposure time (ks)', fontsize=12)
    ax.set_ylabel('Number of pixels', fontsize=12)
    ax.set_title(
        f'FPM{module} — Exposure-map distribution in '
        f'{cfg.src_radius_arcsec:.0f}" aperture', fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()

    fname = os.path.join(out_dir, f"nustar_expmap_hist_FPM{module}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Exposure histogram   -> {fname}")
