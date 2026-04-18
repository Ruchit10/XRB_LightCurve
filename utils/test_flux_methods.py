#!/usr/bin/env python3
"""
Test script for the three flux conversion methods in xrb_lightcurve.py

This script tests:
1. Legacy mode (hardcoded exponentials)
2. Interpolation mode (requires data_flux_vs_nH.csv)
3. Refit mode (fits new exponentials to CSV)

Usage:
    # Test legacy mode only
    python test_flux_methods.py --mode legacy
    
    # Test all modes (requires CSV)
    python test_flux_methods.py --mode all --csv data_flux_vs_nH.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Import the simulation module
try:
    from xrb_lightcurve import simulate_lightcurve
except ImportError:
    print("Error: Could not import xrb_lightcurve module")
    print("Make sure you're in the correct directory and conda environment")
    sys.exit(1)


def test_legacy_mode():
    """Test legacy mode with default exponential coefficients"""
    print("\n" + "="*60)
    print("Testing LEGACY mode (hardcoded exponentials)")
    print("="*60)
    
    try:
        results = simulate_lightcurve(
            r=0.001,
            R=2.0,
            d1=11.0,
            d2=8.0,
            gma0=-90.0,
            i0=26.0,
            dth=1.0,
            flux_method="legacy",
            verbose=False,
        )
        
        print(f"✓ Legacy mode completed successfully")
        print(f"  Generated {len(results)} data points")
        print(f"  fl range: {results['fl'].min():.6f} to {results['fl'].max():.6f}")
        print(f"  Hard flux range: {results['nfl_hard_av'].min():.3e} to {results['nfl_hard_av'].max():.3e}")
        print(f"  Soft flux range: {results['nfl_soft_av'].min():.3e} to {results['nfl_soft_av'].max():.3e}")
        return True
    except Exception as e:
        print(f"✗ Legacy mode failed: {e}")
        return False


def test_interpolate_mode(csv_path):
    """Test interpolation mode from CSV data"""
    print("\n" + "="*60)
    print("Testing INTERPOLATE mode (from CSV)")
    print("="*60)
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return False
    
    try:
        results = simulate_lightcurve(
            r=0.001,
            R=2.0,
            d1=11.0,
            d2=8.0,
            gma0=-90.0,
            i0=26.0,
            dth=1.0,
            flux_method="interpolate",
            flux_csv_path=csv_path,
            verbose=False,
        )
        
        print(f"✓ Interpolate mode completed successfully")
        print(f"  Generated {len(results)} data points")
        print(f"  fl range: {results['fl'].min():.6f} to {results['fl'].max():.6f}")
        print(f"  Hard flux range: {results['nfl_hard_av'].min():.3e} to {results['nfl_hard_av'].max():.3e}")
        print(f"  Soft flux range: {results['nfl_soft_av'].min():.3e} to {results['nfl_soft_av'].max():.3e}")
        return True
    except Exception as e:
        print(f"✗ Interpolate mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_refit_mode(csv_path):
    """Test refit mode - fits new exponentials to CSV"""
    print("\n" + "="*60)
    print("Testing REFIT mode (fit exponentials to CSV)")
    print("="*60)
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        return False
    
    try:
        results = simulate_lightcurve(
            r=0.001,
            R=2.0,
            d1=11.0,
            d2=8.0,
            gma0=-90.0,
            i0=26.0,
            dth=1.0,
            flux_method="refit",
            flux_csv_path=csv_path,
            verbose=False,
        )
        
        print(f"✓ Refit mode completed successfully")
        print(f"  Generated {len(results)} data points")
        print(f"  fl range: {results['fl'].min():.6f} to {results['fl'].max():.6f}")
        print(f"  Hard flux range: {results['nfl_hard_av'].min():.3e} to {results['nfl_hard_av'].max():.3e}")
        print(f"  Soft flux range: {results['nfl_soft_av'].min():.3e} to {results['nfl_soft_av'].max():.3e}")
        return True
    except Exception as e:
        print(f"✗ Refit mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test flux conversion methods in xrb_lightcurve.py"
    )
    parser.add_argument(
        "--mode",
        choices=["legacy", "all"],
        default="legacy",
        help="Test mode: 'legacy' only or 'all' methods"
    )
    parser.add_argument(
        "--csv",
        default="data_flux_vs_nH.csv",
        help="Path to flux vs nH CSV file (required for 'all' mode)"
    )
    
    args = parser.parse_args()
    
    print("XRB Lightcurve Flux Methods Test Suite")
    print("="*60)
    
    results = []
    
    # Always test legacy mode
    results.append(("Legacy", test_legacy_mode()))
    
    # Test interpolate and refit if requested
    if args.mode == "all":
        results.append(("Interpolate", test_interpolate_mode(args.csv)))
        results.append(("Refit", test_refit_mode(args.csv)))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

