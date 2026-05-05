"""
xray_uplim.cli
--------------
Command-line interface for xray_uplim.

Usage
-----
    xray_uplim-cli config.yaml       — run with a YAML config file
    xray_uplim-cli config.json       — run with a JSON config file
    xray_uplim-cli --template        — print a template YAML config and exit

The config file must have an 'observatory' key set to one of:
    nustar | swift | xmm | chandra

All other keys match the parameter names in run_uplim.py exactly.
"""

import sys
import os

TEMPLATE_YAML = """\
# xray_uplim configuration file
# Edit the values below, then run:  xray_uplim-cli config.yaml
#
# Set observatory to one of: nustar | swift | xmm | chandra

observatory: nustar   # <-- change this first

# ---------------------------------------------------------------------------
# NuSTAR settings
# ---------------------------------------------------------------------------
base_path: "/path/to/NuSTAR/data/"
obsid:
  - "80802504004"
  # - "80802504002"   # add more obs to co-add

caldb_dir: ""         # leave empty to use $CALDB environment variable

ra:  "05:00:13.721"   # HH:MM:SS.ss  or decimal degrees
dec: "-03:20:51.22"   # ±DD:MM:SS.ss or decimal degrees

src_radius_arcsec: 60.0
bkg_radius_arcsec: 200.0
bkg_inner_factor:  1.2
psf_fwhm_arcsec:   18.0

energy_band: "full"   # full | soft | iron | medium | hard | ultra-hard
                      # or custom: [8.0, 24.0]

modules: ["A", "B"]

bkg_mode: "annulus"   # annulus | manual
bkg_ra:   ""          # only used when bkg_mode = manual
bkg_dec:  ""

exp_stat:  "median"   # median | mean | psf_weighted
psf_gamma: 2.0

confidence_levels: [0.6827, 0.9545, 0.9973]   # ~1σ  ~2σ  ~3σ

use_gui:     true
gui_per_obs: false
save_plots:  true

# ---------------------------------------------------------------------------
# Swift settings  (only read when observatory: swift)
# ---------------------------------------------------------------------------
# data_dir: "/path/to/Swift/data/"
# obsid: ["03000397004"]
# ...

# ---------------------------------------------------------------------------
# XMM settings  (only read when observatory: xmm)
# ---------------------------------------------------------------------------
# data_dir: "/path/to/XMM/ODF/"
# obsid: "0881990901"
# instruments: ["MOS1", "MOS2", "PN"]
# ...

# ---------------------------------------------------------------------------
# Chandra settings  (only read when observatory: chandra)
# ---------------------------------------------------------------------------
# base_path: "/path/to/Chandra/data/"
# obsid: "26631"
# ciao_prefix: ""
# run_repro:   true
# use_aprates: true
# ...
"""


def _load_config(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.yaml', '.yml'):
        try:
            import yaml
        except ImportError:
            sys.exit(
                "PyYAML is required for YAML config files.\n"
                "Install with:  pip install pyyaml\n"
                "Or use a JSON config file instead.")
        with open(path) as f:
            return yaml.safe_load(f)
    elif ext == '.json':
        import json
        with open(path) as f:
            return json.load(f)
    else:
        sys.exit(f"Unsupported config format '{ext}'. Use .yaml or .json")


def main():
    args = sys.argv[1:]

    if not args or '--help' in args or '-h' in args:
        print(__doc__)
        print("Options:")
        print("  --template    Print a template YAML config file")
        sys.exit(0)

    if '--template' in args:
        print(TEMPLATE_YAML)
        sys.exit(0)

    config_path = args[0]
    if not os.path.isfile(config_path):
        sys.exit(f"Config file not found: {config_path}")

    cfg = _load_config(config_path)
    observatory = cfg.pop('observatory', '').strip().lower()

    if not observatory:
        sys.exit("Config must contain an 'observatory' key "
                 "(nustar | swift | xmm | chandra).")

    # Convert energy_band list → tuple for custom bands
    if isinstance(cfg.get('energy_band'), list):
        cfg['energy_band'] = tuple(cfg['energy_band'])

    # obsid: single string or list
    if isinstance(cfg.get('obsid'), list) and len(cfg['obsid']) == 1:
        cfg['obsid'] = cfg['obsid'][0]

    if observatory == 'nustar':
        from xray_uplim.nustar.pipeline import run_uplim
    elif observatory == 'swift':
        from xray_uplim.swift.pipeline import run_uplim
    elif observatory == 'xmm':
        from xray_uplim.xmm.pipeline import run_uplim
    elif observatory == 'chandra':
        from xray_uplim.chandra.pipeline import run_uplim
    else:
        sys.exit(f"Unknown observatory '{observatory}'. "
                 "Choose from: nustar | swift | xmm | chandra")

    run_uplim(**cfg)


if __name__ == '__main__':
    main()
