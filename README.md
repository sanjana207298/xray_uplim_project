# nustar_uplim

**NuSTAR count-rate upper limits for non-detections.**

A clean Python package for computing statistically rigorous X-ray count-rate upper limits from NuSTAR data when a source is not detected. Implements Kraft et al. (1991) as the primary method, with Gehrels (1986) as a cross-check and a simple net-rate point estimate for comparison.

---

## Methods

### Upper limits

| Method | Reference | Notes |
|--------|-----------|-------|
| **Kraft et al. 1991** | ApJ 374, 344 | Bayesian posterior with uniform prior on S ≥ 0. Solved exactly via the regularised incomplete Gamma function — numerically stable at any count level. Standard method for X-ray non-detections. |
| **Gehrels 1986** | ApJ 303, 336 | Closed-form Poisson approximation. Printed as a cross-check; slightly overestimates at low N. |

### Net count rate (point estimate)

```
CR_net = (N_src - B_scaled) / t_eff
```

This is **not an upper limit** — it is the background-subtracted count rate with its 1-sigma Poisson uncertainty. Printed alongside the proper upper limits for comparison. Can be negative for a clean non-detection.

The 1-sigma uncertainty is propagated correctly through the background area scaling:

```
sigma = sqrt(N_src + N_bkg_raw * area_ratio²) / t_eff
```

### Effective exposure time

Read from the NuSTAR **exposure map** (not the header `LIVETIME`). The exposure map encodes vignetting, dead-time, and chip gaps in a single image. For a non-detection, the **median** of non-zero exposure-map pixels inside the source aperture is recommended:

- Makes no assumption about PSF shape or source centring
- Robust against partially-clipped chip-gap edge pixels
- Easy to justify in a methods section

All three statistics (median, mean, PSF-weighted mean) are always printed for comparison.

> **Note on PSF-weighted mean:** This is provided as a diagnostic only. It assumes an on-axis circular Gaussian PSF, which is not appropriate for off-axis NuSTAR sources (the PSF broadens and becomes asymmetric off-axis).

---

## Installation

```bash
git clone https://github.com/sanjana207298/nustar_uplim.git
cd nustar_uplim
pip install -e .
```

Or without installing:

```bash
pip install -r requirements.txt
python run_uplim.py
```

---

## Quickstart

### As a script

Edit the `CONFIG` block in `run_uplim.py` and run:

```bash
python run_uplim.py
```

### From Python / Jupyter

```python
from nustar_uplim import run_uplim

results = run_uplim(
    base_path = "/data/NuSTAR/2017gas/",
    obsid     = "80202052002",
    ra        = "20:17:11.360",
    dec       = "+58:12:08.10",
    energy_band       = "soft",          # 3-10 keV
    confidence_levels = [0.9545, 0.9973],
)
```

### Custom energy band

```python
results = run_uplim(
    base_path = "/data/NuSTAR/",
    obsid     = "80202052002",
    ra        = "20:17:11.360",
    dec       = "+58:12:08.10",
    energy_band = (8.0, 30.0),   # custom keV range
)
```

### Manual background region

```python
results = run_uplim(
    base_path = "/data/NuSTAR/",
    obsid     = "80202052002",
    ra        = "20:17:11.360",
    dec       = "+58:12:08.10",
    bkg_mode  = "manual",
    bkg_ra    = "20:17:25.0",
    bkg_dec   = "+58:14:00.0",
    bkg_radius_arcsec = 100.0,
)
```

---

## File structure expected

Standard `nupipeline` output:

```
<base_path>/<obsid>/event_cl/
    nu<obsid>A01_cl.evt.gz    ← FPM-A cleaned events
    nu<obsid>B01_cl.evt.gz    ← FPM-B cleaned events
    nu<obsid>A01_ex.img       ← FPM-A exposure map
    nu<obsid>B01_ex.img       ← FPM-B exposure map
```

Outputs are written to `<base_path>/<obsid>/ul_products/`.

---

## Configuration reference

All parameters are fields of the `Config` dataclass. Pass them as keyword arguments to `run_uplim()`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_path` | — | Root data directory |
| `obsid` | — | NuSTAR observation ID |
| `ra` | — | Source RA: `"HH:MM:SS.ss"` or decimal degrees |
| `dec` | — | Source Dec: `"±DD:MM:SS.ss"` or decimal degrees |
| `src_radius_arcsec` | `60.0` | Source circle radius (arcsec) |
| `bkg_radius_arcsec` | `200.0` | Background annulus outer radius (arcsec) |
| `bkg_inner_factor` | `1.2` | Background inner radius = `src_radius × this` |
| `psf_fwhm_arcsec` | `18.0` | PSF FWHM for diagnostic plot and PSF-weighted exposure |
| `energy_band` | `'full'` | `'full'` / `'soft'` / `'hard'` / `'ultrahard'` or `(e_lo, e_hi)` tuple |
| `modules` | `['A','B']` | FPMs to process |
| `bkg_mode` | `'annulus'` | `'annulus'` or `'manual'` |
| `bkg_ra` / `bkg_dec` | `""` | Background circle centre (manual mode only) |
| `exp_stat` | `'median'` | Primary exposure statistic: `'median'` / `'mean'` / `'psf_weighted'` |
| `confidence_levels` | `[0.9545, 0.9973]` | One-sided CLs (~2σ and ~3σ) |
| `save_plots` | `True` | Save diagnostic PNG plots |

### Energy bands

| Name | Range |
|------|-------|
| `'full'` | 3–79 keV |
| `'soft'` | 3–10 keV |
| `'hard'` | 10–30 keV |
| `'ultrahard'` | 30–79 keV |

### Confidence levels

Always quote the CL explicitly in your paper. Common choices (one-sided Gaussian convention):

| CL | Gaussian equiv. |
|----|----------------|
| 0.9000 | 1.28σ |
| 0.9500 | 1.64σ |
| 0.9545 | ~2σ |
| 0.9900 | 2.33σ |
| 0.9973 | ~3σ |

---

## Output

Per module and for the combined FPM-A + FPM-B result, the code prints:

```
  Point estimate  (N_src - B) / t_eff  [NOT an upper limit]
    = (558 - 515.1) / 40190.0 s
    = +1.0672e-03 cts/s  ±  8.2341e-04  (1-sigma Poisson)

  Upper limits:
        CL      Net CR (cts/s)   Kraft S_ul    Kraft CR_ul   Gehrels S_ul  Gehrels CR_ul
  ---------------------------------------------------------------------------
    0.9545      +1.0672e-03       83.412     2.0751e-03        83.809     2.0853e-03
    0.9973      +1.0672e-03      109.109     2.7150e-03       109.622     2.7276e-03
  ---------------------------------------------------------------------------
  Divide all count rates by the aperture EEF (~0.80 at 60") to correct for flux outside the aperture.
```

Two diagnostic plots are saved to `ul_products/`:
- `nustar_radial_FPM{A,B}_{band}keV.png` — log-scale radial surface-density profile
- `nustar_expmap_hist_FPM{A,B}.png` — exposure-map pixel distribution in the source aperture

---

## Important notes

**EEF correction:** The upper limits are for counts inside the extraction aperture. Divide by the encircled-energy fraction to recover the total source rate. For a 60" aperture: EEF ≈ 0.80 (Harrison et al. 2013, ApJ 770, 103).

**Flux conversion:** To convert from count rate to flux, use a spectral model (e.g. an absorbed power law) in PIMMS, WebPIMMS, or XSPEC with the appropriate column density and photon index for your source.

**Marginal detections:** If the net counts are significantly positive (e.g. net > 3σ above background), you may have a marginal detection rather than a non-detection. Check the radial profile plot and consider a proper detection significance test before reporting upper limits.

---

## Package structure

```
nustar_uplim/
├── nustar_uplim/
│   ├── __init__.py      ← public API
│   ├── config.py        ← Config dataclass (all user parameters)
│   ├── coords.py        ← coordinate parsing, sky-to-pixel conversion
│   ├── exposure.py      ← exposure map statistics
│   ├── io.py            ← file discovery and FITS loading
│   ├── statistics.py    ← Kraft, Gehrels, net count rate
│   ├── plots.py         ← diagnostic plots
│   └── pipeline.py      ← orchestration, run_uplim()
├── run_uplim.py         ← standalone script / CLI entry point
├── requirements.txt
├── setup.py
└── README.md
```

---

## Dependencies

- `numpy >= 1.21`
- `scipy >= 1.7`
- `astropy >= 5.0`
- `matplotlib >= 3.4`

---

## Citation

If you use this code, please cite the upper-limit methods used:

- **Kraft et al. 1991** — Kraft, R. P., Burrows, D. N., & Nousek, J. A. 1991, ApJ, 374, 344
- **Gehrels 1986** — Gehrels, N. 1986, ApJ, 303, 336
- **NuSTAR PSF** — Harrison, F. A., et al. 2013, ApJ, 770, 103
