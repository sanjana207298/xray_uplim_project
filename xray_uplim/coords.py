"""
xray_uplim.coords
-----------------
Shared coordinate utilities used by all observatory modules.

parse_coord()
    Parse an (RA, Dec) string or float pair into an astropy SkyCoord.

sky_to_img_pixel()
    Convert RA/Dec to pixel position in a standard FITS image (e.g. an
    exposure map or vignetting map) using the primary-array WCS.
    Works for any observatory whose image products carry standard
    CRPIX/CRVAL/CDELT or CD-matrix WCS keywords.
"""

import numpy as np
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


def sky_to_img_pixel(ra_deg, dec_deg, img_hdr):
    """
    Convert RA/Dec to pixel position in a standard FITS image.

    Uses the primary-array WCS (CRPIX/CRVAL/CDELT or CD matrix), which
    works correctly for exposure maps, vignetting maps, and similar image
    products from any observatory.

    Parameters
    ----------
    ra_deg, dec_deg : float
    img_hdr         : astropy.io.fits.Header  (primary image HDU)

    Returns
    -------
    cx, cy  : float
    pscale  : float  — arcsec/pix
    """
    import warnings

    wcs    = WCS(img_hdr, naxis=2)
    cx, cy = wcs.all_world2pix([[ra_deg, dec_deg]], 0)[0]

    pscale = None
    for key in ('CDELT2', 'CD2_2'):
        if key in img_hdr:
            pscale = abs(img_hdr[key]) * 3600.0
            break
    if pscale is None:
        warnings.warn(
            "Pixel scale not found in image header (tried CDELT2, CD2_2). "
            "Falling back to NuSTAR default 2.459 \"/pix — "
            "set the correct value if using a different observatory.",
            RuntimeWarning, stacklevel=2)
        pscale = 2.459

    return cx, cy, pscale
