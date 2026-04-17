"""
xray_uplim.nustar.io
--------------------
File discovery and FITS loading for NuSTAR nupipeline products.

Expected directory layout (standard nupipeline output)
-------------------------------------------------------
<base_path>/<obsid>/event_cl/
    nu<obsid>A01_cl.evt.gz   — FPM-A cleaned events
    nu<obsid>B01_cl.evt.gz   — FPM-B cleaned events
    nu<obsid>A01_ex.img      — FPM-A exposure map
    nu<obsid>B01_ex.img      — FPM-B exposure map
"""

import os
import glob
import numpy as np
from astropy.io import fits


# NuSTAR PI <-> energy:  E_keV = PI_SLOPE * PI + PI_OFFSET
PI_SLOPE  = 0.04
PI_OFFSET = 1.6


def find_file(pattern):
    """
    Glob for a single file matching `pattern`.

    Parameters
    ----------
    pattern : str  — glob pattern

    Returns
    -------
    str  — path to the (alphabetically first) matching file

    Raises
    ------
    FileNotFoundError if nothing matches.
    """
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"\nCould not find a file matching:\n  {pattern}\n"
            "Check base_path, obsid, and the contents of event_cl/.")
    return sorted(matches)[0]


def locate_files(base_path, obsid, module):
    """
    Locate the cleaned event file and exposure map for one FPM.

    Parameters
    ----------
    base_path : str
    obsid     : str
    module    : str  — 'A' or 'B'

    Returns
    -------
    evt_file : str  — path to cleaned event file
    exp_file : str  — path to exposure map
    """
    evt_dir = os.path.join(base_path, obsid, "event_cl")
    M       = module.upper()

    evt_file = find_file(os.path.join(evt_dir, f"nu{obsid}{M}01_cl.evt*"))
    exp_file = find_file(os.path.join(evt_dir, f"nu{obsid}{M}01_ex.img*"))

    return evt_file, exp_file


def load_events(evt_file, energy_lo_kev, energy_hi_kev):
    """
    Load and energy-filter events from a NuSTAR cleaned event file.

    Energy filtering uses floor/ceil to ensure the band is strictly
    inclusive at both edges:
        PI_lo = floor((E_lo - 1.6) / 0.04)
        PI_hi = ceil ((E_hi - 1.6) / 0.04)

    Parameters
    ----------
    evt_file      : str   — path to .evt or .evt.gz file
    energy_lo_kev : float — lower energy bound (keV)
    energy_hi_kev : float — upper energy bound (keV)

    Returns
    -------
    evts    : numpy recarray  — filtered event table
    evt_hdr : fits.Header     — EVENTS extension header
    PI_lo   : int
    PI_hi   : int
    """
    PI_lo = int(np.floor((energy_lo_kev - PI_OFFSET) / PI_SLOPE))
    PI_hi = int(np.ceil ((energy_hi_kev - PI_OFFSET) / PI_SLOPE))

    with fits.open(evt_file) as hdul:
        evts    = hdul['EVENTS'].data.copy()
        evt_hdr = hdul['EVENTS'].header

    mask = (evts['PI'] >= PI_lo) & (evts['PI'] <= PI_hi)
    return evts[mask], evt_hdr, PI_lo, PI_hi


def load_expmap(exp_file):
    """
    Load a NuSTAR exposure map.

    Parameters
    ----------
    exp_file : str  — path to .img file

    Returns
    -------
    exp_data : 2-D float array  — exposure in seconds
    exp_hdr  : fits.Header
    """
    with fits.open(exp_file) as hdul:
        exp_data = hdul[0].data.astype(float)
        exp_hdr  = hdul[0].header
    return exp_data, exp_hdr
