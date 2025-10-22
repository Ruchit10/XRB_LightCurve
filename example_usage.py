#!/usr/bin/env python3
"""
Example usage of the XRB Lightcurve simulation.
This script demonstrates how to use the simulation programmatically
and create custom analysis workflows.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xrb_lightcurve import simulate_lightcurve


def main():
    """Example usage of the XRB lightcurve simulation."""

    print("Running XRB Lightcurve Simulation Example...")

    # Example 1: Default parameters
    print("\n1. Running with default parameters...")
    results_default = simulate_lightcurve()
    print(f"   Generated {len(results_default)} data points")
    print(
        f"   Phase range: {results_default['deg'].min():.1f} to {results_default['deg'].max():.1f} degrees"
    )

    # Example 2: High-resolution simulation
    print("\n2. Running high-resolution simulation...")
    results_high_res = simulate_lightcurve(dth=0.5)
    print(f"   Generated {len(results_high_res)} data points")

    # Example 3: Different binary configuration
    print("\n3. Running with different binary parameters...")
    results_large_binary = simulate_lightcurve(
        r=0.002,  # Larger compact object
        R=3.0,  # Larger companion
        d1=15.0,  # Larger separation
        d2=10.0,  # Larger separation
        dth=2.0,  # Coarser resolution for speed
    )
    print(f"   Generated {len(results_large_binary)} data points")

    # Example 4: Custom phase range
    print("\n4. Running with custom phase range...")
    results_custom_phase = simulate_lightcurve(
        gma0=0.0,  # Start at 0 degrees
        dth=1.0,  # 1-degree increments
        i0=45.0,  # Higher inclination
    )
    print(f"   Generated {len(results_custom_phase)} data points")

    # Save all results
    results_default.to_csv("example_default.csv", index=False)
    results_high_res.to_csv("example_high_res.csv", index=False)
    results_large_binary.to_csv("example_large_binary.csv", index=False)
    results_custom_phase.to_csv("example_custom_phase.csv", index=False)

    print("\nAll results saved to CSV files:")
    print("  - example_default.csv")
    print("  - example_high_res.csv")
    print("  - example_large_binary.csv")
    print("  - example_custom_phase.csv")

    # Example analysis: Compare flux variations
    print("\n5. Analyzing flux variations...")

    # Calculate mean fluxes for comparison
    mean_fluxes = {
        "Default": np.mean(results_default["flx"]),
        "High Resolution": np.mean(results_high_res["flx"]),
        "Large Binary": np.mean(results_large_binary["flx"]),
        "Custom Phase": np.mean(results_custom_phase["flx"]),
    }

    print("   Mean flux values:")
    for name, flux in mean_fluxes.items():
        print(f"     {name}: {flux:.6f}")

    # Example: Find eclipse minimum
    min_flux_idx = results_default["flx"].idxmin()
    min_flux_phase = results_default.loc[min_flux_idx, "deg"]
    min_flux_value = results_default.loc[min_flux_idx, "flx"]

    print(f"\n   Eclipse minimum found at {min_flux_phase:.1f} degrees")
    print(f"   Minimum flux value: {min_flux_value:.6f}")

    # Example: Calculate eclipse depth
    max_flux = results_default["flx"].max()
    eclipse_depth = (max_flux - min_flux_value) / max_flux * 100
    print(f"   Eclipse depth: {eclipse_depth:.2f}%")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
