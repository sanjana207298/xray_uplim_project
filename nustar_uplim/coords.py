"""
nustar_uplim.coords
-------------------
Coordinate parsing and sky-to-pixel conversion for NuSTAR data.

The key function here is sky_to_evt_pixel(), which reads the per-column
WCS keywords from NuSTAR EVENTS binary table headers rather than the
primary-array WCS used by astropy.wcs.WCS.  Using WCS(evt_hdr, naxis=2)
on a NuSTAR event file reads wrong or missing CRPIX/CRVAL/CDELT keywords
and produces completely incorrect pixel positions (landing outside the
detector), yielding zero counts in every aperture.
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


def parse_coord(ra_str, dec_str):
    """
    Parse an (RA, Dec) pair into an astropy SkyCoord.

    Accepts
    -------
    Decimal degrees  :  304.297   58.202
    Sexagesimal      : "20:17:11.360"   "+58:12:08.10"
    astropy strings  : "20h17m11.36s"  "+58d12m08.1s"

    Returns
    -------
    astropy.coordinates.SkyCoord (ICRS)
    """
    try:
        return SkyCoord(ra=float(ra_str)*u.deg,
                        dec=float(dec_str)*u.deg, frame='icrs')
    except (ValueError, TypeError):
        pass
    ra_fmt  = str(ra_str).replace(':', 'h', 1).replace(':', 'm', 1) + 's'
    dec_fmt = str(dec_str).replace(':', 'd', 1).replace(':', 'm', 1) + 's'
    return SkyCoord(ra_fmt, dec_fmt, frame='icrs')


def sky_to_evt_pixel(ra_deg, dec_deg, evt_hdr):
    """
    Convert RA/Dec (degrees) to NuSTAR event-file sky pixel (X, Y).

    WHY NOT WCS(evt_hdr, naxis=2)?
    --------------------------------
    NuSTAR event files store the sky projection in *per-column* WCS keywords
    inside the EVENTS binary table: TCTYPn, TCRPXn, TCRVLn, TCDLTn.
    astropy's WCS(header, naxis=2) reads the primary-array WCS keywords
    (CRPIX/CRVAL/CDELT) which are absent or refer to something unrelated in
    event files, producing a completely wrong pixel position — and therefore
    zero counts in any aperture.  This function reads the correct
    column-level keywords directly.

    The NuSTAR FOV is ~13'x13', so a linear TAN approximation is accurate
    to well under one pixel across the whole field.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Source sky position in decimal degrees (ICRS).
    evt_hdr : astropy.io.fits.Header
        Header of the EVENTS binary table extension.

    Returns
    -------
    cx, cy  : float  — pixel coordinates matching the event X/Y column values
    pscale  : float  — pixel scale in arcsec/pix (derived from the Dec/Y axis)
    """
    # Find column indices labelled 'X' and 'Y'
    x_col = y_col = None
    for i in range(1, 300):
        if f'TTYPE{i}' not in evt_hdr:
            break
        name = evt_hdr[f'TTYPE{i}'].strip().upper()
        if name == 'X':
            x_col = i
        elif name == 'Y':
            y_col = i

    if x_col is None or y_col is None:
        raise RuntimeError(
            f"Could not find X or Y columns in EVENTS header "
            f"(X_col={x_col}, Y_col={y_col}). "
            "Is this a NuSTAR cleaned event file?")

    try:
        crpx_x = float(evt_hdr[f'TCRPX{x_col}'])   # reference pixel (RA)
        crvl_x = float(evt_hdr[f'TCRVL{x_col}'])   # reference RA  (deg)
        cdlt_x = float(evt_hdr[f'TCDLT{x_col}'])   # deg/pix — negative for RA
        crpx_y = float(evt_hdr[f'TCRPX{y_col}'])   # reference pixel (Dec)
        crvl_y = float(evt_hdr[f'TCRVL{y_col}'])   # reference Dec (deg)
        cdlt_y = float(evt_hdr[f'TCDLT{y_col}'])   # deg/pix — positive
    except KeyError as exc:
        raise RuntimeError(
            f"Missing column WCS keyword {exc} in EVENTS header. "
            "File may be non-standard or from a different pipeline.") from exc

    # Linear TAN projection — dRA scaled by cos(Dec_ref) for foreshortening
    cos_dec = np.cos(np.radians(crvl_y))
    cx = crpx_x + (ra_deg  - crvl_x) * cos_dec / cdlt_x
    cy = crpx_y + (dec_deg - crvl_y)            / cdlt_y

    pscale = abs(cdlt_y) * 3600.0   # arcsec/pixel from Dec axis

    return cx, cy, pscale


def sky_to_img_pixel(ra_deg, dec_deg, img_hdr):
    """
    Convert RA/Dec to pixel position in a standard FITS image (exposure map).
    The NuSTAR exposure map uses a normal image WCS, so astropy WCS works.

    Parameters
    ----------
    ra_deg, dec_deg : float
    img_hdr         : astropy.io.fits.Header  (primary image HDU)

    Returns
    -------
    cx, cy  : float
    pscale  : float  — arcsec/pix
    """
    wcs    = WCS(img_hdr, naxis=2)
    cx, cy = wcs.all_world2pix([[ra_deg, dec_deg]], 0)[0]

    pscale = None
    for key in ('CDELT2', 'CD2_2'):
        if key in img_hdr:
            pscale = abs(img_hdr[key]) * 3600.0
            break
    if pscale is None:
        import warnings
        warnings.warn(
            "Pixel scale not found in exposure-map header; "
            "using NuSTAR default 2.459 \"/pix.",
            RuntimeWarning, stacklevel=2)
        pscale = 2.459

    return cx, cy, pscale
