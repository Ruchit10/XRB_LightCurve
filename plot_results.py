#!/usr/bin/env python3
"""
Visualization script for XRB Lightcurve simulation results.
This script demonstrates how to plot and analyze the simulation outputs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys


def plot_lightcurve(data_file, output_file=None):
    """
    Create comprehensive plots of the lightcurve simulation results.

    Args:
        data_file: Path to the CSV file with simulation results
        output_file: Optional output file for saving plots
    """
    # Read the data
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: File {data_file} not found!")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("XRB Lightcurve Simulation Results", fontsize=16, fontweight="bold")

    # Plot 1: Flux vs Phase (degrees)
    axes[0, 0].plot(df["deg"], df["flx"], "b-", linewidth=2, label="Accelerated Wind")
    axes[0, 0].plot(
        df["deg"], df["flx2"], "r--", linewidth=2, label="Constant Velocity Wind"
    )
    axes[0, 0].set_xlabel("Phase (degrees)")
    axes[0, 0].set_ylabel("Flux")
    axes[0, 0].set_title("Flux vs Phase")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Scaled Flux vs Phase
    axes[0, 1].plot(df["deg"], df["fl"], "g-", linewidth=2, label="Scaled Flux (Acc)")
    axes[0, 1].plot(
        df["deg"], df["fl2"], "m--", linewidth=2, label="Scaled Flux (Const)"
    )
    axes[0, 1].set_xlabel("Phase (degrees)")
    axes[0, 1].set_ylabel("Scaled Flux")
    axes[0, 1].set_title("Scaled Flux vs Phase")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Hard Band Fluxes
    axes[1, 0].plot(
        df["deg"], df["nfl_hard_av"], "b-", linewidth=2, label="Hard Band (Acc)"
    )
    axes[1, 0].plot(
        df["deg"], df["nfl_hard_cv"], "r--", linewidth=2, label="Hard Band (Const)"
    )
    axes[1, 0].set_xlabel("Phase (degrees)")
    axes[1, 0].set_ylabel("Hard Band Flux")
    axes[1, 0].set_title("Hard Band (2-10 keV) Flux")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Soft Band Fluxes
    axes[1, 1].plot(
        df["deg"], df["nfl_soft_av"], "b-", linewidth=2, label="Soft Band (Acc)"
    )
    axes[1, 1].plot(
        df["deg"], df["nfl_soft_cv"], "r--", linewidth=2, label="Soft Band (Const)"
    )
    axes[1, 1].set_xlabel("Phase (degrees)")
    axes[1, 1].set_ylabel("Soft Band Flux")
    axes[1, 1].set_title("Soft Band (0.3-2 keV) Flux")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plots saved to {output_file}")
    else:
        plt.show()


def plot_geometric_parameters(data_file, output_file=None):
    """
    Plot geometric parameters from the simulation.

    Args:
        data_file: Path to the CSV file with simulation results
        output_file: Optional output file for saving plots
    """
    # Read the data
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: File {data_file} not found!")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Geometric Parameters", fontsize=16, fontweight="bold")

    # Plot 1: Separation parameters
    axes[0, 0].plot(df["deg"], df["l3"], "b-", linewidth=2, label="l3 (Separation)")
    axes[0, 0].plot(df["deg"], df["L3"], "r--", linewidth=2, label="L3 (Horizontal)")
    axes[0, 0].plot(df["deg"], df["h3"], "g:", linewidth=2, label="h3 (Vertical)")
    axes[0, 0].set_xlabel("Phase (degrees)")
    axes[0, 0].set_ylabel("Distance (solar radii)")
    axes[0, 0].set_title("Geometric Separations")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Area calculations
    axes[0, 1].plot(df["deg"], df["A2"], "purple", linewidth=2)
    axes[0, 1].set_xlabel("Phase (degrees)")
    axes[0, 1].set_ylabel("Area (solar radii²)")
    axes[0, 1].set_title("Area Calculations")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Integrated column density
    axes[1, 0].plot(df["deg"], df["icd"], "orange", linewidth=2)
    axes[1, 0].set_xlabel("Phase (degrees)")
    axes[1, 0].set_ylabel("Integrated Column Density")
    axes[1, 0].set_title("Integrated Column Density")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Phase vs Time
    axes[1, 1].plot(df["deg"], df["time"], "brown", linewidth=2)
    axes[1, 1].set_xlabel("Phase (degrees)")
    axes[1, 1].set_ylabel("Time")
    axes[1, 1].set_title("Time vs Phase")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Geometric plots saved to {output_file}")
    else:
        plt.show()


def main():
    """Main function for plotting results."""
    parser = argparse.ArgumentParser(
        description="Plot XRB Lightcurve simulation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("data_file", type=str, help="CSV file with simulation results")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for saving plots (e.g., plots.png)",
    )
    parser.add_argument(
        "--geometric",
        action="store_true",
        help="Plot geometric parameters instead of flux",
    )

    args = parser.parse_args()

    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for plotting.")
        print("Install it with: pip install matplotlib")
        sys.exit(1)

    if args.geometric:
        plot_geometric_parameters(args.data_file, args.output)
    else:
        plot_lightcurve(args.data_file, args.output)


if __name__ == "__main__":
    main()
