"""
xray_uplim._runner
------------------
Thin subprocess entry point called by the GUI:

    python -m xray_uplim._runner --config /tmp/xray_uplim_XXXX.json

Reads the JSON config written by the GUI, converts types that JSON cannot
represent natively (tuples → lists for energy_band), then routes to the
correct telescope pipeline.
"""

import json
import sys


def main():
    if '--config' not in sys.argv:
        print("Usage: python -m xray_uplim._runner --config <config.json>",
              flush=True)
        sys.exit(1)

    idx = sys.argv.index('--config')
    config_path = sys.argv[idx + 1]

    with open(config_path) as f:
        cfg = json.load(f)

    observatory = cfg.pop('_observatory')

    # JSON has no tuple type — energy_band custom bands arrive as lists
    if isinstance(cfg.get('energy_band'), list):
        cfg['energy_band'] = tuple(cfg['energy_band'])

    # obsid may arrive as a single-element list when user typed one ID
    if isinstance(cfg.get('obsid'), list) and len(cfg['obsid']) == 1:
        cfg['obsid'] = cfg['obsid'][0]

    print(f"  Observatory : {observatory.upper()}", flush=True)
    print(f"  Config keys : {list(cfg.keys())}", flush=True)

    if observatory == 'nustar':
        from xray_uplim.nustar.pipeline import run_uplim
    elif observatory == 'swift':
        from xray_uplim.swift.pipeline import run_uplim
    elif observatory == 'xmm':
        from xray_uplim.xmm.pipeline import run_uplim
    elif observatory == 'chandra':
        from xray_uplim.chandra.pipeline import run_uplim
    else:
        raise ValueError(f"Unknown observatory '{observatory}'")

    run_uplim(**cfg)


if __name__ == '__main__':
    main()
