"""
nustar_uplim.pipeline
---------------------
Top-level orchestration: per-module extraction and combined results.

Public API
----------
run_uplim(...)        — convenience wrapper; builds a Config and runs everything
process_module(...)   — extract counts + exposure for one FPM, return results dict
combine_modules(...)  — sum across FPMs and print combined upper limits
print_results_table() — shared formatted output for both per-module and combined
"""

import os
import numpy as np

from .config    import Config
from .coords    import parse_coord, sky_to_evt_pixel, sky_to_img_pixel
from .exposure  import compute_exposure_stats
from .io        import locate_files, load_events, load_expmap
from .statistics import net_count_rate, kraft_upper_limit, gehrels_upper_limit
from .plots     import radial_profile, exposure_histogram


# =============================================================================
# RESULTS TABLE  (shared by process_module and combine_modules)
# =============================================================================

def print_results_table(N_src, B_scaled, t_eff, N_bkg_raw, area_ratio,
                        confidence_levels):
    """
    Compute and print all three methods at every confidence level.

    Columns
    -------
    CL          — one-sided confidence level
    Net CR      — (N_src - B_scaled) / t_eff   [point estimate, not a UL]
    Kraft S_ul  — Kraft+91 upper limit in counts
    Kraft CR_ul — Kraft+91 upper limit in cts/s
    Gehrels S_ul / CR_ul — Gehrels 1986 cross-check

    Returns
    -------
    list of dicts, one per confidence level, with keys:
        cl, CR_net, CR_sigma, S_kraft, CR_kraft, S_gehrels, CR_gehrels
    """
    header  = (f"  {'CL':>8}  {'Net CR (cts/s)':>18}  "
               f"{'Kraft S_ul':>10}  {'Kraft CR_ul':>13}  "
               f"{'Gehrels S_ul':>12}  {'Gehrels CR_ul':>13}")
    divider = "  " + "-" * (len(header) - 2)

    CR_net, CR_sigma = net_count_rate(N_src, B_scaled, t_eff,
                                      N_bkg_raw, area_ratio)

    print(f"\n  Point estimate  (N_src - B) / t_eff  [NOT an upper limit]")
    print(f"    = ({N_src} - {B_scaled:.1f}) / {t_eff:.1f} s")
    print(f"    = {CR_net:+.4e} cts/s  ±  {CR_sigma:.4e}  (1-sigma Poisson)")
    if CR_net < 0:
        print(f"    (Negative — source aperture below expected background: "
              f"clean non-detection)")

    print(f"\n  Upper limits:")
    print(header)
    print(divider)

    results = []
    for cl in confidence_levels:
        S_k  = kraft_upper_limit(N_src, B_scaled, cl)
        S_g  = gehrels_upper_limit(N_src, B_scaled, cl)
        CR_k = S_k / t_eff
        CR_g = S_g / t_eff
        results.append({
            'cl':          cl,
            'CR_net':      CR_net,
            'CR_sigma':    CR_sigma,
            'S_kraft':     S_k,
            'CR_kraft':    CR_k,
            'S_gehrels':   S_g,
            'CR_gehrels':  CR_g,
        })
        print(f"  {cl:8.4f}  {CR_net:+18.4e}  "
              f"{S_k:10.3f}  {CR_k:13.4e}  "
              f"{S_g:12.3f}  {CR_g:13.4e}")

    print(divider)
    print(f"  Net CR is a point estimate. Kraft and Gehrels are upper limits.")
    print(f"  Divide all count rates by the aperture EEF (~0.80 at 60\") "
          f"to correct for flux outside the aperture.")
    return results


# =============================================================================
# PER-MODULE PIPELINE
# =============================================================================

def process_module(module, src_coord, cfg):
    """
    Full extraction and result calculation for one FPM.

    Parameters
    ----------
    module    : str              — 'A' or 'B'
    src_coord : SkyCoord         — source sky position
    cfg       : Config

    Returns
    -------
    dict with keys:
        module, N_src, N_bkg_raw, B_scaled, area_ratio, net_counts,
        t_eff_s, exp_stats, ul, energy
    """
    e_lo, e_hi = cfg.resolve_energy_band()
    out_dir    = os.path.join(cfg.base_path, cfg.obsid, "ul_products")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  FPM{module}")
    print(f"{'='*62}")

    # -- Locate and load files ------------------------------------------------
    evt_file, exp_file = locate_files(cfg.base_path, cfg.obsid, module)
    print(f"  Event file  : {os.path.basename(evt_file)}")
    print(f"  Expo map    : {os.path.basename(exp_file)}")

    evts, evt_hdr, PI_lo, PI_hi = load_events(evt_file, e_lo, e_hi)
    print(f"  Energy filter [{e_lo:.1f}-{e_hi:.1f} keV]  "
          f"PI=[{PI_lo},{PI_hi}]  ->  {len(evts):,} events")

    exp_data, exp_hdr = load_expmap(exp_file)

    # -- Source pixel positions -----------------------------------------------
    cx_evt, cy_evt, pscale_evt = sky_to_evt_pixel(
        src_coord.ra.deg, src_coord.dec.deg, evt_hdr)
    cx_exp, cy_exp, pscale_exp = sky_to_img_pixel(
        src_coord.ra.deg, src_coord.dec.deg, exp_hdr)

    # -- Sanity check ---------------------------------------------------------
    evt_x = evts['X'].astype(float)
    evt_y = evts['Y'].astype(float)
    print(f"\n  Event X range       : [{evt_x.min():.0f}, {evt_x.max():.0f}]")
    print(f"  Event Y range       : [{evt_y.min():.0f}, {evt_y.max():.0f}]")
    print(f"  Source pixel (evt)  : ({cx_evt:.1f}, {cy_evt:.1f})")
    print(f"  Source pixel (exp)  : ({cx_exp:.1f}, {cy_exp:.1f})")
    x_ok = evt_x.min() <= cx_evt <= evt_x.max()
    y_ok = evt_y.min() <= cy_evt <= evt_y.max()
    if not (x_ok and y_ok):
        print(f"  !! WARNING: source pixel is OUTSIDE the event X/Y range — "
              f"check your coordinates!")
    else:
        print(f"  Source position is inside the event image. Good.")

    # -- Pixel radii ----------------------------------------------------------
    r_src_evt        = cfg.src_radius_arcsec / pscale_evt
    r_src_exp        = cfg.src_radius_arcsec / pscale_exp
    fwhm_pix         = cfg.psf_fwhm_arcsec   / pscale_exp
    r_bkg_in_arcsec  = cfg.src_radius_arcsec * cfg.bkg_inner_factor
    r_bkg_out_arcsec = cfg.bkg_radius_arcsec
    r_bkg_in_evt     = r_bkg_in_arcsec  / pscale_evt
    r_bkg_out_evt    = r_bkg_out_arcsec / pscale_evt

    print(f"  Pixel scale (evt)   : {pscale_evt:.3f} \"/pix")
    print(f"  Pixel scale (exp)   : {pscale_exp:.3f} \"/pix")
    print(f"  Src aperture        : {cfg.src_radius_arcsec:.1f}\" = {r_src_evt:.1f} pix")
    print(f"  Bkg annulus         : {r_bkg_in_arcsec:.1f}\" -- {r_bkg_out_arcsec:.1f}\"")

    # -- Source counts --------------------------------------------------------
    d_src = np.sqrt((evt_x - cx_evt)**2 + (evt_y - cy_evt)**2)
    N_src = int(np.sum(d_src <= r_src_evt))

    # -- Background counts ----------------------------------------------------
    if cfg.bkg_mode == 'annulus':
        in_annulus = (d_src > r_bkg_in_evt) & (d_src <= r_bkg_out_evt)
        N_bkg_raw  = int(np.sum(in_annulus))
        area_src   = np.pi * r_src_evt**2
        area_bkg   = np.pi * (r_bkg_out_evt**2 - r_bkg_in_evt**2)

    elif cfg.bkg_mode == 'manual':
        bkg_coord      = parse_coord(cfg.bkg_ra, cfg.bkg_dec)
        cx_b, cy_b, _  = sky_to_evt_pixel(
            bkg_coord.ra.deg, bkg_coord.dec.deg, evt_hdr)
        r_bkg_circ     = cfg.bkg_radius_arcsec / pscale_evt
        d_bkg          = np.sqrt((evt_x - cx_b)**2 + (evt_y - cy_b)**2)
        N_bkg_raw      = int(np.sum(d_bkg <= r_bkg_circ))
        area_src       = np.pi * r_src_evt**2
        area_bkg       = np.pi * r_bkg_circ**2
    else:
        raise ValueError(f"Unknown bkg_mode: '{cfg.bkg_mode}'")

    area_ratio = area_src / area_bkg
    B_scaled   = N_bkg_raw * area_ratio

    print(f"\n  Source counts  (N_src)        : {N_src}")
    print(f"  Background counts (raw)       : {N_bkg_raw}")
    print(f"  Area ratio  (src / bkg)       : {area_ratio:.5f}")
    print(f"  Scaled background B           : {B_scaled:.3f} cts")
    print(f"  Net counts  (N_src - B)       : {N_src - B_scaled:.3f} cts")

    # -- Effective exposure ---------------------------------------------------
    exp_stats, exp_meta = compute_exposure_stats(
        exp_data, cx_exp, cy_exp, r_src_exp, fwhm_pix)

    print(f"\n  Pixels in aperture (total)    : {exp_meta['n_pix_total']}")
    print(f"  Pixels in aperture (non-zero) : {exp_meta['n_pix_nonzero']}")
    print(f"  -- Exposure statistics ----------------------------------------")
    for key, label in [
            ('median',       'Median        [RECOMMENDED]          '),
            ('mean',         'Mean          [diagnostic]           '),
            ('psf_weighted', 'PSF-wtd mean  [on-axis diag. only]   ')]:
        tag = ' <-- PRIMARY' if key == cfg.exp_stat else ''
        print(f"    {label} : {exp_stats[key]/1e3:7.3f} ks{tag}")

    t_eff = exp_stats[cfg.exp_stat]
    print(f"\n  Using t_eff = {t_eff/1e3:.3f} ks  ({cfg.exp_stat})")

    # -- Results table --------------------------------------------------------
    ul_results = print_results_table(
        N_src, B_scaled, t_eff, N_bkg_raw, area_ratio, cfg.confidence_levels)

    # -- Diagnostic plots -----------------------------------------------------
    if cfg.save_plots:
        radial_profile(evt_x, evt_y, cx_evt, cy_evt, pscale_evt,
                       module, e_lo, e_hi, cfg.obsid, cfg, out_dir)
        exposure_histogram(exp_meta, exp_stats, module, cfg, out_dir)

    return {
        'module':     module,
        'N_src':      N_src,
        'N_bkg_raw':  N_bkg_raw,
        'B_scaled':   B_scaled,
        'area_ratio': area_ratio,
        'net_counts': N_src - B_scaled,
        't_eff_s':    t_eff,
        'exp_stats':  exp_stats,
        'ul':         ul_results,
        'energy':     (e_lo, e_hi),
    }


# =============================================================================
# COMBINED (FPM-A + FPM-B)
# =============================================================================

def combine_modules(results_list, cfg):
    """
    Sum counts across FPMs and compute combined results.

    Combining strategy
    ------------------
    N_total  = sum(N_src)       counts are additive across independent detectors
    B_total  = sum(B_scaled)    each B already corrected to source aperture area
    t_comb   = sum(t_eff)       exposures add — correct for additive counts.
                                Using mean(t) would be inconsistent.
    area_ratio is identical for both FPMs (same aperture geometry).

    Parameters
    ----------
    results_list : list of dicts returned by process_module()
    cfg          : Config
    """
    print(f"\n{'='*62}")
    print("  COMBINED  FPM-A + FPM-B")
    print(f"{'='*62}")

    N_total     = sum(r['N_src']     for r in results_list)
    B_total     = sum(r['B_scaled']  for r in results_list)
    N_bkg_total = sum(r['N_bkg_raw'] for r in results_list)
    area_ratio  = results_list[0]['area_ratio']   # same for both FPMs
    t_vals      = [r['t_eff_s'] for r in results_list]
    t_comb      = float(np.sum(t_vals))           # SUM, not mean

    print(f"  Combined N_src    : {N_total}")
    print(f"  Combined B_scaled : {B_total:.3f} cts")
    for r in results_list:
        print(f"  t_eff FPM-{r['module']}       : {r['t_eff_s']/1e3:.3f} ks")
    print(f"  t_eff (combined)  : {t_comb/1e3:.3f} ks  "
          f"[sum — correct for additive counts]")

    print_results_table(
        N_total, B_total, t_comb, N_bkg_total, area_ratio,
        cfg.confidence_levels)


# =============================================================================
# CONVENIENCE ENTRY POINT
# =============================================================================

def run_uplim(base_path, obsid, ra, dec, **kwargs):
    """
    Run the full upper-limit pipeline with minimal boilerplate.

    Parameters
    ----------
    base_path : str   — root data directory
    obsid     : str   — NuSTAR observation ID
    ra        : str or float  — source RA
    dec       : str or float  — source Dec
    **kwargs  : any Config field by name, e.g.
                    src_radius_arcsec=30.0,
                    energy_band='soft',
                    confidence_levels=[0.9973],
                    save_plots=False

    Returns
    -------
    list of result dicts (one per module processed)

    Example
    -------
    >>> from nustar_uplim import run_uplim
    >>> results = run_uplim(
    ...     base_path = "/data/NuSTAR/2017gas/",
    ...     obsid     = "80202052002",
    ...     ra        = "20:17:11.360",
    ...     dec       = "+58:12:08.10",
    ...     energy_band = 'soft',
    ...     confidence_levels = [0.9545, 0.9973],
    ... )
    """
    cfg = Config(base_path=base_path, obsid=obsid, ra=ra, dec=dec, **kwargs)
    cfg.validate()

    e_lo, e_hi = cfg.resolve_energy_band()
    src_coord  = parse_coord(cfg.ra, cfg.dec)

    print("NuSTAR Non-Detection Upper Limit")
    print("=" * 62)
    print(f"Source  :  RA = {src_coord.ra.deg:.6f} deg  "
          f"Dec = {src_coord.dec.deg:.6f} deg")
    if isinstance(cfg.energy_band, tuple):
        band_label = f"{e_lo:.1f}-{e_hi:.1f} keV (custom)"
    else:
        band_label = f"'{cfg.energy_band}'  ({e_lo:.1f}-{e_hi:.1f} keV)"
    print(f"Band    :  {band_label}")
    print(f"Modules :  {', '.join(f'FPM{m}' for m in cfg.modules)}")
    print(f"Exp stat:  {cfg.exp_stat}  (primary)")

    all_results = []
    for mod in cfg.modules:
        all_results.append(process_module(mod, src_coord, cfg))

    if len(all_results) > 1:
        combine_modules(all_results, cfg)

    print("\nDone.")
    return all_results
