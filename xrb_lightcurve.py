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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    # Create area segments
    av_x = []
    av_th = []
    av_db = []
    A = []

    for i in range(len(g1_s_r) - 1):
        x1 = g1_s_r[i]
        x2 = g1_s_r[i + 1]
        th1 = g1_s_th[i]
        th2 = g1_s_th[i] + (d2h * np.pi / 180)

        dx = x2 - x1
        if dx > 0:
            av_x.append((x1 + x2) / 2)
            av_th.append((th1 + th2) / 2)
            av_db_val = np.sqrt(
                av_x[-1] ** 2 + l**2 - 2 * av_x[-1] * l * np.cos(av_th[-1])
            )
            av_db.append(av_db_val)
            A.append(0.5 * d2h * np.pi / 180 * (x2**2 - x1**2))

    return np.array(av_x), np.array(av_th), np.array(av_db), np.array(A)


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate column integral along the line of sight.

    Args:
        d: Separation between stars
        d1: Distance of star B from COM
        d2: Distance of star A from COM
        gma: Phase angle in radians
        i: Inclination angle in radians
        av_x, av_th, av_db: Grid arrays
        A: Area array

    Returns:
        Tuple of (lw, lw2, icd, A2) arrays
    """
    dz = 0.1

    if len(av_db) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    lw = []
    lw2 = []
    icd = []
    A2 = []

    for f, cl in enumerate(av_db):
        z1 = d1 * np.sin(gma) * np.cos(i)
        z2 = d2 * np.sin(gma) * np.cos(i)
        z = z1 + z2
        t = np.sqrt((2 * d) ** 2 - cl**2)

        z3 = []
        con = []
        con2 = []

        dl = d
        while abs(z) <= t:
            z3.append(z)
            con.append(dl ** (-5 / 2))  # Atoms/(solar radius)^5
            con2.append(dl ** (-2))
            z = z - 0.1
            dl = np.sqrt(cl**2 + z**2)

        los = dz * np.sum(con)  # Atoms/(solar radius)^4
        los2 = dz * np.sum(con2)

        lw.append(los * A[f])  # Atoms/(solar radius)^2
        lw2.append(los2 * A[f])
        icd.append(los)
        A2.append(A[f])

    return np.array(lw), np.array(lw2), np.array(icd), np.array(A2)


def simulate_lightcurve(
    r: float = 0.001,
    R: float = 2.0,
    d1: float = 11.0,
    d2: float = 8.0,
    gma0: float = -90.0,
    i0: float = 26.0,
    dth: float = 1.0,
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

    Returns:
        DataFrame with simulation results
    """
    # Convert angles to radians
    gma = gma0 * np.pi / 180
    i = i0 * np.pi / 180
    d = d1 + d2

    # Initialize arrays
    flx = []
    flx2 = []
    icd = []
    A2 = []
    ph = []
    deg = []
    phase = []
    time = []
    l3 = []
    L3 = []
    h3 = []

    # Calculate number of iterations
    n_iterations = int(360 / dth)

    for j in range(n_iterations):
        # Calculate geometric parameters
        h1 = d1 * np.sin(gma) * np.sin(i)
        h2 = d2 * np.sin(gma) * np.sin(i)
        L1 = d1 * np.cos(gma)
        L2 = d2 * np.cos(gma)
        l1 = np.sqrt(h1**2 + L1**2)
        l2 = np.sqrt(h2**2 + L2**2)
        h = h1 + h2
        L = L1 + L2
        l = l1 + l2

        # Calculate eclipse parameters
        a = l**2 + r**2 - R**2
        b = 2 * abs(l) * r
        n = l / (R + r)

        if gma <= np.pi:
            if n >= 1:
                # Non-eclipsed region
                av_x, av_th, av_db, A = create_grid(r, l, R, gma)
                lw, lw2, icd_val, A2_val = wind_los_integral(
                    d, d1, d2, gma, i, av_x, av_th, av_db, A
                )

                if len(lw) > 0:
                    flx.append(np.sum(lw) / np.sum(A))
                    flx2.append(np.sum(lw2) / np.sum(A))
                    icd.append(np.sum(lw))
                    A2.append(np.sum(A))
                else:
                    flx.append(0)
                    flx2.append(0)
                    icd.append(0)
                    A2.append(0)
            else:
                n2 = a / b
                n3 = l / (R - r)

                if abs(n3) <= 1:
                    flx.append(0)
                    flx2.append(0)
                    icd.append(0)
                    A2.append(0)
                else:
                    av_x, av_th, av_db, A = create_grid(r, l, R, gma)
                    lw, lw2, icd_val, A2_val = wind_los_integral(
                        d, d1, d2, gma, i, av_x, av_th, av_db, A
                    )

                    if len(lw) > 0:
                        flx.append(np.sum(lw) / np.sum(A))
                        flx2.append(np.sum(lw2) / np.sum(A))
                        icd.append(np.sum(lw))
                        A2.append(np.sum(A))
                    else:
                        flx.append(0)
                        flx2.append(0)
                        icd.append(0)
                        A2.append(0)
        else:
            # After π phase
            av_x, av_th, av_db, A = create_grid(r, l, R, gma)
            lw, lw2, icd_val, A2_val = wind_los_integral(
                d, d1, d2, gma, i, av_x, av_th, av_db, A
            )

            if len(lw) > 0:
                flx.append(np.sum(lw) / np.sum(A))
                flx2.append(np.sum(lw2) / np.sum(A))
                icd.append(np.sum(lw))
                A2.append(np.sum(A))
            else:
                flx.append(0)
                flx2.append(0)
                icd.append(0)
                A2.append(0)

        # Store phase information
        ph.append(gma)
        deg.append(gma * 180 / np.pi)
        time.append(deg[-1] * 348.42)
        phase.append((gma - (gma0 * np.pi / 180)) / (2 * np.pi))
        l3.append(l)
        L3.append(L)
        h3.append(h)

        print(f"Phase: {deg[-1]:.2f} degrees")

        # Update phase angle
        gma = gma + (dth * np.pi / 180)

    # Create results DataFrame
    results = pd.DataFrame(
        {
            "deg": deg,
            "ph": ph,
            "phase": phase,
            "A2": A2,
            "flx": flx,
            "flx2": flx2,
            "icd": icd,
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
