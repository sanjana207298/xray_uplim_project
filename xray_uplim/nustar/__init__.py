"""
xray_uplim.nustar
-----------------
NuSTAR-specific pipeline: dual-FPM processing, column WCS coordinates,
NuSTAR CALDB PSF, and 7 named energy bands (3–79 keV).
"""

from .pipeline import run_uplim, process_module, combine_modules
from .config   import Config

__all__ = ["run_uplim", "process_module", "combine_modules", "Config"]
