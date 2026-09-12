#!/usr/bin/env python3
"""
Test script for the two flux conversion methods in xrb_lightcurve.py

This script tests:
1. Interpolation mode (log-log interpolation of the XSPEC flux vs nH table)
2. Refit mode (fits exponentials A*exp(-B*nH) to the same table)

Both require a flux vs nH CSV produced by compute_flux_vs_nH.py.

Usage:
    python utils/test_flux_methods.py --csv data_flux_vs_nH.csv
    python utils/test_flux_methods.py --csv data_flux_vs_nH.csv --mode interpolate
"""

import argparse
import os
import sys

# Running this as a script puts utils/ on sys.path rather than the repo root,
# so add the parent directory explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the simulation module
try:
    from xrb_lightcurve import simulate_lightcurve
except ImportError:
    print("Error: Could not import xrb_lightcurve module")
    print("Make sure you're in the repository root and the correct conda environment")
    sys.exit(1)


# Geometry shared by every mode, so the only difference between runs is the
# nH -> flux mapping under test.
BASE_PARAMS = dict(
    r=0.001,
    R=2.0,
    d1=11.0,
    d2=8.0,
    gma0=-90.0,
    i0=78.0,
    dth=5.0,
    f_opacity=0.02,
    verbose=False,
)


def run_mode(flux_method: str, csv_path: str) -> bool:
    """Run one flux_method and report the resulting column/flux ranges."""
    print("\n" + "=" * 60)
    print(f"Testing {flux_method.upper()} mode")
    print("=" * 60)

    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return False

    try:
        results = simulate_lightcurve(
            flux_method=flux_method,
            flux_csv_path=csv_path,
            **BASE_PARAMS,
        )
    except Exception as e:
        print(f"✗ {flux_method} mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"✓ {flux_method} mode completed successfully")
    print(f"  Generated {len(results)} data points")
    print(f"  fl range: {results['fl'].min():.6g} to {results['fl'].max():.6g}"
          f"  (1e22 cm^-2)")
    flux_cols = sorted(c for c in results.columns if c.startswith("nfl_"))
    if not flux_cols:
        print("✗ No nfl_* flux columns were produced")
        return False
    for col in flux_cols:
        print(f"  {col} range: {results[col].min():.3e} to {results[col].max():.3e}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test flux conversion methods in xrb_lightcurve.py"
    )
    parser.add_argument(
        "--mode",
        choices=["interpolate", "refit", "all"],
        default="all",
        help="Which flux_method to test (default: all)",
    )
    parser.add_argument(
        "--csv",
        default="data_flux_vs_nH.csv",
        help="Path to flux vs nH CSV file from compute_flux_vs_nH.py",
    )

    args = parser.parse_args()

    print("XRB Lightcurve Flux Methods Test Suite")
    print("=" * 60)

    modes = ["interpolate", "refit"] if args.mode == "all" else [args.mode]
    results = [(m, run_mode(m, args.csv)) for m in modes]

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
