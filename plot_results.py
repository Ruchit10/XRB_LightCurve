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
import re
from typing import List, Dict, Tuple


# Energy band display names and energy ranges (keV)
BAND_INFO: Dict[str, Tuple[str, str]] = {
    "ultrasoft": ("Ultra-soft", "0.2-0.5 keV"),
    "soft": ("Soft", "0.5-2 keV"),
    "medium": ("Medium", "1.2-2.0 keV"),
    "hard": ("Hard", "2.0-7.0 keV"),
    "broad": ("Broad", "0.5-7.0 keV"),
}


def detect_energy_bands(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect available energy band columns in the DataFrame.
    
    Looks for columns matching patterns:
    - nfl_{band}_av (accelerated velocity wind model)
    - nfl_{band}_cv (constant velocity wind model)
    
    Args:
        df: DataFrame with simulation results
        
    Returns:
        List of detected band names (e.g., ['soft', 'hard', 'broad'])
    """
    bands = set()
    pattern = re.compile(r"nfl_(\w+)_(av|cv)")
    
    for col in df.columns:
        match = pattern.match(col)
        if match:
            bands.add(match.group(1))
    
    # Sort bands in a logical order if they match known bands
    known_order = ["ultrasoft", "soft", "medium", "hard", "broad"]
    sorted_bands = []
    for band in known_order:
        if band in bands:
            sorted_bands.append(band)
            bands.discard(band)
    # Add any remaining unknown bands
    sorted_bands.extend(sorted(bands))
    
    return sorted_bands


def get_band_display_name(band: str) -> Tuple[str, str]:
    """
    Get display name and energy range for a band.
    
    Args:
        band: Band name (e.g., 'soft', 'hard')
        
    Returns:
        Tuple of (display_name, energy_range) or defaults if unknown
    """
    if band in BAND_INFO:
        return BAND_INFO[band]
    # For unknown bands, capitalize the name
    return (band.replace("_", " ").title(), "")


def plot_lightcurve(data_file, output_file=None):
    """
    Create comprehensive plots of the lightcurve simulation results.
    Auto-detects available energy bands and creates appropriate subplots.

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

    # Auto-detect energy bands
    bands = detect_energy_bands(df)
    print(f"Detected energy bands: {bands}")
    
    # Determine number of plots needed:
    # - 2 base plots (Flux vs Phase, Scaled Flux vs Phase) if flx/fl columns exist
    # - 1 plot per detected energy band
    has_base_flux = "flx" in df.columns and "fl" in df.columns
    n_base_plots = 2 if has_base_flux else 0
    n_band_plots = len(bands)
    n_total_plots = n_base_plots + n_band_plots
    
    if n_total_plots == 0:
        print("Error: No plottable columns found in data file!")
        return
    
    # Calculate grid dimensions
    n_cols = min(2, n_total_plots)
    n_rows = int(np.ceil(n_total_plots / n_cols))
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    fig.suptitle("XRB Lightcurve Simulation Results", fontsize=16, fontweight="bold")
    
    # Flatten axes array for easier indexing
    if n_total_plots == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot base flux columns if available
    if has_base_flux:
        # Plot 1: Flux vs Phase (degrees)
        ax = axes[plot_idx]
        ax.plot(df["deg"], df["flx"], "b-", linewidth=2, label="Accelerated Wind")
        if "flx2" in df.columns:
            ax.plot(df["deg"], df["flx2"], "r--", linewidth=2, label="Constant Velocity Wind")
        ax.set_xlabel("Phase (degrees)")
        ax.set_ylabel("Flux")
        ax.set_title("Flux vs Phase")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        # Plot 2: Scaled Flux vs Phase
        ax = axes[plot_idx]
        ax.plot(df["deg"], df["fl"], "g-", linewidth=2, label="Scaled Flux (Acc)")
        if "fl2" in df.columns:
            ax.plot(df["deg"], df["fl2"], "m--", linewidth=2, label="Scaled Flux (Const)")
        ax.set_xlabel("Phase (degrees)")
        ax.set_ylabel("Scaled Flux")
        ax.set_title("Scaled Flux vs Phase")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot each detected energy band
    for band in bands:
        ax = axes[plot_idx]
        display_name, energy_range = get_band_display_name(band)
        
        # Check which columns exist for this band
        av_col = f"nfl_{band}_av"
        cv_col = f"nfl_{band}_cv"
        
        if av_col in df.columns:
            ax.plot(df["deg"], df[av_col], "b-", linewidth=2, label=f"{display_name} (Acc)")
        if cv_col in df.columns:
            ax.plot(df["deg"], df[cv_col], "r--", linewidth=2, label=f"{display_name} (Const)")
        
        ax.set_xlabel("Phase (degrees)")
        ax.set_ylabel(f"{display_name} Band Flux")
        title = f"{display_name} Band"
        if energy_range:
            title += f" ({energy_range})"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Hide any unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

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
