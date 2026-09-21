#!/usr/bin/env python3
"""
Run the paper's MCMC fits at full size, one after another, into figures/cache/.

The figure notebook runs these itself when a chain is missing; this script is
the same thing from a shell (or a cluster job), so the fits can run overnight
and the notebook then only draws:

    python figures/run_fits.py                 # all four (about an hour on a laptop)
    python figures/run_fits.py A_fiducial      # one of them
    python figures/run_fits.py --quick         # tiny chains, smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figlib as L  # noqa: E402

FITS = ("A_fiducial", "ridge_broad", "ridge_tightR", "ridge_fopa_frozen")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", default=list(FITS), help=f"which fits (default: {', '.join(FITS)})")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    for name in args.names:
        if name not in FITS:
            parser.error(f"unknown fit {name!r}; choose from {FITS}")
        t0 = time.time()
        cfg = L.ensure_fit(name, quick=args.quick)
        print(f"{name}: done in {(time.time() - t0) / 60:.1f} min -> {os.path.relpath(cfg['chain'], L.ROOT)}", flush=True)


if __name__ == "__main__":
    main()
