#!/usr/bin/env python3
"""
Standalone plotting of xrb_lightcurve simulation CSVs.
------------------------------------------------------
Thin CLI over ``utils/plot_utils.py``. The plotting itself lives there so that
``mcmc_lightcurve_fit.py`` can produce the same geometry figures from a
posterior point estimate.

Usage
~~~~~
# Per-band model light curves
$ python plot_results.py sim.csv --output bands.png

# Geometry: projected separation vs the eclipse thresholds, sky-plane
# components, N_H(phase) and the resulting band flux
$ python plot_results.py sim.csv --geometric --R 2.0 --r 0.001 --output geom.png

# Projected orbit / eclipse diagram (needs the geometry parameters)
$ python plot_results.py sim.csv --orbit --R 2.0 --r 0.001 \
    --d1 11.0 --d2 8.0 --i0 26.0 --output orbit.png
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from utils.utils import (
    BAND_INFO,
    detect_energy_bands,
    get_band_display_name,
)
from utils.plot_utils import (
    plot_geometry_vs_phase,
    plot_orbit_geometry,
    plot_simulation_bands,
)

__all__ = [
    "BAND_INFO",
    "detect_energy_bands",
    "get_band_display_name",
    "main",
    "plot_geometry_vs_phase",
    "plot_orbit_geometry",
    "plot_simulation_bands",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot XRB Lightcurve simulation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data_file", type=str, help="CSV file with simulation results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for saving plots (e.g., plots.png)")
    parser.add_argument("--geometric", action="store_true",
                        help="Plot geometry and absorption vs phase instead of band fluxes")
    parser.add_argument("--orbit", action="store_true",
                        help="Plot the projected orbit / eclipse diagram")
    parser.add_argument("--R", type=float, default=None,
                        help="Companion radius in solar radii (for --geometric / --orbit)")
    parser.add_argument("--r", type=float, default=0.001,
                        help="Compact object / disk radius in solar radii")
    parser.add_argument("--d1", type=float, default=None,
                        help="Compact-object distance from the centre of mass (--orbit)")
    parser.add_argument("--d2", type=float, default=None,
                        help="Companion distance from the centre of mass (--orbit)")
    parser.add_argument("--i0", type=float, default=None,
                        help="Orbital inclination in degrees (--orbit)")
    parser.add_argument("--band", type=str, default=None,
                        help="Energy band label for titles and the flux panel")

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.data_file)
    except FileNotFoundError:
        print(f"Error: File {args.data_file} not found!")
        sys.exit(1)

    if args.orbit:
        missing = [n for n in ("R", "d1", "d2", "i0") if getattr(args, n) is None]
        if missing:
            parser.error(f"--orbit requires {', '.join('--' + m for m in missing)}")
        plot_orbit_geometry(
            df, R=args.R, r=args.r, d1=args.d1, d2=args.d2, i0=args.i0,
            output_path=args.output, band=args.band,
        )
    elif args.geometric:
        if args.R is None:
            parser.error("--geometric requires --R (the eclipse thresholds need it)")
        plot_geometry_vs_phase(
            df, R=args.R, r=args.r, band=args.band, output_path=args.output,
        )
    else:
        plot_simulation_bands(df, output_path=args.output)


if __name__ == "__main__":
    main()
