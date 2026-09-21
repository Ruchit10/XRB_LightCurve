#!/usr/bin/env python3
"""
Run the paper's MCMC fits at full size, one after another, into figures/cache/.

The figure notebook runs these itself when a chain is missing; this script is
the same thing from a shell (or a cluster job), so the fits can run overnight
and the notebook then only draws:

    python figures/run_fits.py                 # all four (about 80 minutes on a laptop)
    python figures/run_fits.py A_fiducial      # one of them
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figlib as L  # noqa: E402

FITS = L.FIT_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", default=list(FITS), help=f"which fits (default: {', '.join(FITS)})")
    args = parser.parse_args()
    for name in args.names:
        if name not in FITS:
            parser.error(f"unknown fit {name!r}; choose from {FITS}")
    for name in args.names:
        t0 = time.time()
        cfg = L.ensure_fit(name)
        print(f"{name}: done in {(time.time() - t0) / 60:.1f} min -> {os.path.relpath(cfg['chain'], L.ROOT)}", flush=True)


if __name__ == "__main__":
    main()
