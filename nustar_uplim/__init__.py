"""
nustar_uplim
============
NuSTAR count-rate upper limit calculator for non-detections.

Quickstart
----------
>>> from nustar_uplim import run_uplim
>>> run_uplim(
...     base_path="/data/NuSTAR/",
...     obsid="80202052002",
...     ra="20:17:11.360",
...     dec="+58:12:08.10",
... )
"""

from .pipeline import run_uplim, process_module, combine_modules
from .statistics import kraft_upper_limit, gehrels_upper_limit, net_count_rate
from .config import Config

__version__ = "1.0.0"
__author__  = "Sanjana Gupta"
__all__ = [
    "run_uplim",
    "process_module",
    "combine_modules",
    "kraft_upper_limit",
    "gehrels_upper_limit",
    "net_count_rate",
    "Config",
]
