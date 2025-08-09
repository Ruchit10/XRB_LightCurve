#!/usr/bin/env python3
"""
Simulation of column densities for eclipsing binary systems.
This module simulates the column densities obtained as the compact object 
eclipses companion in a Binary System orbiting a common Center of Mass.

The Compact object and the Accretion disk is referred to as Star B
The Companion Star is referred to as Star A
All Distance units are in Solar Radii
All Angle units are converted into radians for trigonometric functions
"""

import argparse
import numpy as np
import pandas as pd
import math
from typing import Tuple, List, Optional


def create_grid(
    r: float, l: float, R: float, gma: float, d2h: float = 6.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create grid for wind integral of eclipsing binaries.

    Args:
        r: Radius of smaller star B (compact object)
        l: Separation along viewing plane
        R: Radius of larger star A (companion)
        gma: Phase angle in radians
        d2h: Angle size for polar grid cell (degrees)

    Returns:
        Tuple of (av_x, av_th, av_db, A) arrays
    """
    # Create polar grid
    g1_r = np.linspace(r / 10, r, 10)
    g1_th = np.linspace(0, 2 * np.pi, int(360 / d2h) + 1)

    # Expand grid
    g1_r_mesh, g1_th_mesh = np.meshgrid(g1_r, g1_th)
    g1_r_flat = g1_r_mesh.flatten()
    g1_th_flat = g1_th_mesh.flatten()

    # Filter points based on conditions
    if gma < np.pi:
        # Calculate distance from center
        nn = np.sqrt(g1_r_flat**2 + l**2 - 2 * g1_r_flat * l * np.cos(g1_th_flat))
        mask = nn >= R
        g1_s_r = g1_r_flat[mask]
        g1_s_th = g1_th_flat[mask]
    else:
        g1_s_r = g1_r_flat
        g1_s_th = g1_th_flat

    if g1_s_r.size < 2:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    # Vectorized segment construction (adjacent pairs in flattened order)
    x1 = g1_s_r[:-1]
    x2 = g1_s_r[1:]
    th1 = g1_s_th[:-1]
    dx = x2 - x1
    valid = dx > 0

    if not np.any(valid):
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    av_x = 0.5 * (x1[valid] + x2[valid])
    av_th = 0.5 * (th1[valid] + (th1[valid] + (d2h * np.pi / 180.0)))
    av_db = np.sqrt(av_x**2 + l**2 - 2.0 * av_x * l * np.cos(av_th))
    A = 0.5 * (d2h * np.pi / 180.0) * ((x2[valid] ** 2) - (x1[valid] ** 2))

    return av_x.astype(float), av_th.astype(float), av_db.astype(float), A.astype(float)


def density_function(
    d: float, l: float, gma: float, i: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate wind density along the line of sight.

    Args:
        d: Separation between stars
        l: Separation along viewing plane
        gma: Phase angle in radians
        i: Inclination angle in radians

    Returns:
        Tuple of (z5, colm) arrays
    """
    dz = 0.1
    d4 = d
    z4 = d * np.sin(gma) * np.cos(i)
    t4 = np.sqrt((2 * d) ** 2 - l**2)

    z5 = []
    colm = []

    while abs(z4) <= t4:
        z5.append(z4)
        colm.append(d4 ** (-2))
        z4 = z4 - 0.1
        d4 = np.sqrt(l**2 + z4**2)

    return np.array(z5), np.array(colm)


def wind_los_integral(
    d: float,
    d1: float,
    d2: float,
    gma: float,
    i: float,
    av_x: np.ndarray,
    av_th: np.ndarray,
    av_db: np.ndarray,
    A: np.ndarray,
    dz: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate column integral along the line of sight.

    Uses vectorized summation over the line-of-sight grid to avoid Python loops.

    Args:
        d: Separation between stars
        d1: Distance of star B from COM
        d2: Distance of star A from COM
        gma: Phase angle in radians
        i: Inclination angle in radians
        av_x, av_th, av_db: Grid arrays
        A: Area array
        dz: Step along line of sight (solar radii)

    Returns:
        Tuple of (lw, lw2, icd, A2) arrays
    """
    if av_db is None or av_db.size == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    # z start and bounds (identical for each cell within a phase)
    z1 = d1 * np.sin(gma) * np.cos(i)
    z2 = d2 * np.sin(gma) * np.cos(i)
    z_start = z1 + z2

    # Per-cell LOS half-extent
    t = np.sqrt((2.0 * d) ** 2 - av_db**2)

    # Determine per-cell validity: original algorithm integrates only if |z_start| <= t
    valid_cells = (np.abs(z_start) <= t)

    # Per-cell end step index; invalid cells get -1 so they contribute zero
    end_k = np.floor((z_start + t) / dz).astype(int)
    end_k = np.where(valid_cells, end_k, -1)

    max_steps = int(end_k.max())
    if max_steps < 0:
        return (
            np.zeros_like(av_db),
            np.zeros_like(av_db),
            np.zeros_like(av_db),
            np.zeros_like(av_db),
        )

    # Broadcasted z values across steps
    k = np.arange(max_steps + 1, dtype=float)  # shape (K+1,)
    z_vals = z_start - dz * k  # shape (K+1,)

    # Broadcast to (N, K+1)
    cl2 = (av_db**2)[:, None]  # (N,1)
    z2_vals = z_vals[None, :] ** 2  # (1, K+1)

    denom = cl2 + z2_vals  # (N, K+1)

    # Masks to include only valid steps per cell (k from 0..end_k inclusive)
    step_mask = (k[None, :] <= end_k[:, None]) & valid_cells[:, None]

    # Compute sums
    con_sum = np.sum((denom ** (-5.0 / 4.0)) * step_mask, axis=1)
    con2_sum = np.sum((denom ** (-1.0)) * step_mask, axis=1)

    los = dz * con_sum
    los2 = dz * con2_sum

    lw = los * A
    lw2 = los2 * A

    # icd is los for each cell, A2 is A
    return lw.astype(float), lw2.astype(float), los.astype(float), A.astype(float)


def simulate_lightcurve(
    r: float = 0.001,
    R: float = 2.0,
    d1: float = 11.0,
    d2: float = 8.0,
    gma0: float = -90.0,
    i0: float = 26.0,
    dth: float = 1.0,
    d2h: float = 6.0,
    dz: float = 0.1,
    verbose: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Main simulation function for lightcurve calculation.

    Args:
        r: Radius of smaller star B (compact object) in solar radii
        R: Radius of larger star A (companion) in solar radii
        d1: Distance of star B from COM in solar radii
        d2: Distance of star A from COM in solar radii
        gma0: Starting phase angle in degrees
        i0: Orbital inclination in degrees
        dth: Orbital increment in degrees
        d2h: Angular cell size (degrees) for the polar grid used in the surface integral
        dz: Step size along the line of sight (solar radii)
        verbose: If True, prints per-phase progress

    Returns:
        DataFrame with simulation results
    """
    # Convert angles to radians
    gma = gma0 * np.pi / 180
    i = i0 * np.pi / 180
    d = d1 + d2

    # Prepare phase values
    n_iterations = int(360 / dth)
    gma_values = gma + (np.arange(n_iterations) * (dth * np.pi / 180.0))

    # Worker to compute one phase
    def _compute_one_phase(cur_gma: float):
        h1 = d1 * np.sin(cur_gma) * np.sin(i)
        h2 = d2 * np.sin(cur_gma) * np.sin(i)
        L1 = d1 * np.cos(cur_gma)
        L2 = d2 * np.cos(cur_gma)
        l1 = np.sqrt(h1**2 + L1**2)
        l2 = np.sqrt(h2**2 + L2**2)
        h = h1 + h2
        L = L1 + L2
        l = l1 + l2

        a = l**2 + r**2 - R**2
        b = 2 * abs(l) * r
        n = l / (R + r)

        if cur_gma <= np.pi:
            if n >= 1:
                av_x, av_th, av_db, A_cells = create_grid(r, l, R, cur_gma, d2h=d2h)
                lw, lw2, icd_val, A2_val = wind_los_integral(
                    d, d1, d2, cur_gma, i, av_x, av_th, av_db, A_cells, dz=dz
                )
                if lw.size > 0:
                    flx_i = float(np.sum(lw) / np.sum(A_cells))
                    flx2_i = float(np.sum(lw2) / np.sum(A_cells))
                    icd_i = float(np.sum(lw))
                    A2_i = float(np.sum(A_cells))
                else:
                    flx_i = 0.0
                    flx2_i = 0.0
                    icd_i = 0.0
                    A2_i = 0.0
            else:
                n2 = a / b
                n3 = l / (R - r)
                if abs(n3) <= 1:
                    flx_i = 0.0
                    flx2_i = 0.0
                    icd_i = 0.0
                    A2_i = 0.0
                else:
                    av_x, av_th, av_db, A_cells = create_grid(r, l, R, cur_gma, d2h=d2h)
                    lw, lw2, icd_val, A2_val = wind_los_integral(
                        d, d1, d2, cur_gma, i, av_x, av_th, av_db, A_cells, dz=dz
                    )
                    if lw.size > 0:
                        flx_i = float(np.sum(lw) / np.sum(A_cells))
                        flx2_i = float(np.sum(lw2) / np.sum(A_cells))
                        icd_i = float(np.sum(lw))
                        A2_i = float(np.sum(A_cells))
                    else:
                        flx_i = 0.0
                        flx2_i = 0.0
                        icd_i = 0.0
                        A2_i = 0.0
        else:
            av_x, av_th, av_db, A_cells = create_grid(r, l, R, cur_gma, d2h=d2h)
            lw, lw2, icd_val, A2_val = wind_los_integral(
                d, d1, d2, cur_gma, i, av_x, av_th, av_db, A_cells, dz=dz
            )
            if lw.size > 0:
                flx_i = float(np.sum(lw) / np.sum(A_cells))
                flx2_i = float(np.sum(lw2) / np.sum(A_cells))
                icd_i = float(np.sum(lw))
                A2_i = float(np.sum(A_cells))
            else:
                flx_i = 0.0
                flx2_i = 0.0
                icd_i = 0.0
                A2_i = 0.0

        deg_i = cur_gma * 180.0 / np.pi
        time_i = deg_i * 348.42
        phase_i = (cur_gma - (gma0 * np.pi / 180.0)) / (2.0 * np.pi)
        return (
            flx_i,
            flx2_i,
            icd_i,
            A2_i,
            cur_gma,
            deg_i,
            phase_i,
            time_i,
            l,
            L,
            h,
        )

    # Compute phases, optionally in parallel
    if n_jobs == 1:
        results_list = []
        for idx, cur_gma in enumerate(gma_values):
            out = _compute_one_phase(cur_gma)
            results_list.append(out)
            if verbose:
                print(f"Phase: {out[5]:.2f} degrees")
    else:
        try:
            from joblib import Parallel, delayed

            results_list = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_compute_one_phase)(float(cur_gma)) for cur_gma in gma_values
            )
            if verbose:
                for out in results_list:
                    print(f"Phase: {out[5]:.2f} degrees")
        except Exception:
            # Fallback to serial if joblib missing or errors
            results_list = []
            for idx, cur_gma in enumerate(gma_values):
                out = _compute_one_phase(cur_gma)
                results_list.append(out)
                if verbose:
                    print(f"Phase: {out[5]:.2f} degrees")

    # Unpack
    flx, flx2, icd_vals, A2_vals, ph, deg, phase, time, l3, L3, h3 = map(
        list, zip(*results_list)
    )

    # Create results DataFrame
    results = pd.DataFrame(
        {
            "deg": deg,
            "ph": ph,
            "phase": phase,
            "A2": A2_vals,
            "flx": flx,
            "flx2": flx2,
            "icd": icd_vals,
            "time": time,
            "l3": l3,
            "L3": L3,
            "h3": h3,
        }
    )

    # Calculate additional flux parameters
    lam = 0.589537 / np.mean(flx) if np.mean(flx) > 0 else 1
    lam2 = 0.589537 / np.mean(flx2) if np.mean(flx2) > 0 else 1

    fl = np.array(flx) * lam
    fl2 = np.array(flx2) * lam2

    # Calculate scaled fluxes
    nfl_hard_av = 9.524 * np.exp(-fl * 0.057)
    nfl_hard_cv = 9.524 * np.exp(-fl2 * 0.057)
    nfl_soft_av = 9.3923 * np.exp(-fl * 2.5062)
    nfl_soft_cv = 9.3923 * np.exp(-fl2 * 2.5062)
    pho_count_hard_av = 0.0001464 * np.exp(-fl * 0.1066818)
    pho_count_soft_av = 0.0005275 * np.exp(-fl * 2.7556631)

    # Add flux columns to results
    results["fl"] = fl
    results["fl2"] = fl2
    results["nfl_hard_av"] = nfl_hard_av
    results["nfl_hard_cv"] = nfl_hard_cv
    results["nfl_soft_av"] = nfl_soft_av
    results["nfl_soft_cv"] = nfl_soft_cv
    results["pho_count_hard_av"] = pho_count_hard_av
    results["pho_count_soft_av"] = pho_count_soft_av

    return results


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Simulation of column densities for eclipsing binary systems",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--r",
        type=float,
        default=0.001,
        help="Radius of smaller star B (compact object) in solar radii",
    )
    parser.add_argument(
        "--R",
        type=float,
        default=2.0,
        help="Radius of larger star A (companion) in solar radii",
    )
    parser.add_argument(
        "--d1",
        type=float,
        default=11.0,
        help="Distance of star B from COM in solar radii",
    )
    parser.add_argument(
        "--d2",
        type=float,
        default=8.0,
        help="Distance of star A from COM in solar radii",
    )
    parser.add_argument(
        "--gma0", type=float, default=-90.0, help="Starting phase angle in degrees"
    )
    parser.add_argument(
        "--i0", type=float, default=26.0, help="Orbital inclination in degrees"
    )
    parser.add_argument(
        "--dth", type=float, default=1.0, help="Orbital increment in degrees"
    )
    parser.add_argument(
        "--d2h",
        type=float,
        default=6.0,
        help="Angular cell size (degrees) for the polar grid used in the surface integral",
    )
    parser.add_argument(
        "--dz",
        type=float,
        default=0.1,
        help="Step size along the line of sight (solar radii)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-phase progress during simulation",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel workers across phases (1 = serial)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="xrb_lightcurve_output.csv",
        help="Output file name for results",
    )

    args = parser.parse_args()

    print("Starting XRB Lightcurve Simulation...")
    print(f"Parameters:")
    print(f"  r (emitter radius): {args.r} solar radii")
    print(f"  R (companion radius): {args.R} solar radii")
    print(f"  d1 (emitter separation): {args.d1} solar radii")
    print(f"  d2 (companion separation): {args.d2} solar radii")
    print(f"  gma0 (starting phase): {args.gma0} degrees")
    print(f"  i0 (inclination): {args.i0} degrees")
    print(f"  dth (orbital increment): {args.dth} degrees")
    print(f"  d2h (polar cell size): {args.d2h} degrees")
    print(f"  dz (LOS step size): {args.dz}")
    print(f"  n_jobs (parallel workers): {args.n_jobs}")
    print(f"  Output file: {args.output}")
    print()

    # Run simulation
    results = simulate_lightcurve(
        r=args.r,
        R=args.R,
        d1=args.d1,
        d2=args.d2,
        gma0=args.gma0,
        i0=args.i0,
        dth=args.dth,
        d2h=args.d2h,
        dz=args.dz,
        verbose=args.verbose,
        n_jobs=args.n_jobs,
    )

    # Save results
    results.to_csv(args.output, index=False)
    print(f"\nSimulation completed! Results saved to {args.output}")
    print(f"Total data points: {len(results)}")
    print(
        f"Phase range: {results['deg'].min():.2f} to {results['deg'].max():.2f} degrees"
    )

    return results


if __name__ == "__main__":
    main()
