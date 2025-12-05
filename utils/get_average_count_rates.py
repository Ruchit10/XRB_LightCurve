#!/usr/bin/env python3
"""
Calculate time-averaged count rates for each energy band.

These are the count rates that correspond to your combined spectrum,
and should be used with XSPEC model flux to compute conversion factors.

Usage:
    python get_average_count_rates.py
"""

import glob
import os

def calculate_average_rate(band):
    """Calculate time-averaged count rate for a given band."""
    pattern = f"data/IC_10_X1_LC/{band}_converted/*_{band.lower()}.txt"
    files = sorted(glob.glob(pattern))
    
    if not files:
        return None, None, None
    
    total_counts = 0
    total_exposure = 0
    
    for f in files:
        with open(f) as file:
            for line in file:
                if line.startswith('#') or not line.strip():
                    continue
                try:
                    parts = line.split('\t')
                    counts = float(parts[1])  # COUNTS column
                    exposure = float(parts[4])  # EXPOSURE column
                    
                    total_counts += counts
                    total_exposure += exposure
                except:
                    pass
    
    avg_rate = total_counts / total_exposure if total_exposure > 0 else 0
    
    return avg_rate, total_counts, total_exposure


def main():
    print()
    print("="*80)
    print("TIME-AVERAGED COUNT RATES FOR CONVERSION FACTOR CALCULATION")
    print("="*80)
    print()
    print("These count rates correspond to your combined spectrum and should be")
    print("used with XSPEC model flux to compute conversion factors.")
    print()
    print("-"*80)
    
    bands = ['Broad', 'Soft', 'Hard']
    rates = {}
    
    for band in bands:
        avg_rate, total_counts, total_exposure = calculate_average_rate(band)
        
        if avg_rate is None:
            print(f"\n{band} Band: No data found")
            print(f"  (Check that {band}_converted/ directory exists)")
            continue
        
        rates[band.lower()] = avg_rate
        
        print(f"\n{band} Band:")
        print(f"  Total counts:        {total_counts:.0f}")
        print(f"  Total exposure:      {total_exposure:.2f} seconds ({total_exposure/3600:.2f} hours)")
        print(f"  Average count rate:  {avg_rate:.6f} counts/s")
    
    print()
    print("-"*80)
    print("HOW TO USE THESE VALUES")
    print("-"*80)
    print()
    print("1. Run XSPEC to get model flux for each band:")
    print("   ./get_conversion_factors.sh")
    print()
    print("2. For each band, XSPEC will report model flux (erg/cm²/s)")
    print()
    print("3. Calculate conversion factors:")
    print()
    
    for band in bands:
        if band.lower() in rates:
            rate = rates[band.lower()]
            print(f"   {band} band:")
            print(f"      conversion_factor = XSPEC_flux_{band.lower()} / {rate:.6f}")
            print()
    
    print("4. Apply conversion factors to light curves:")
    print()
    for band in bands:
        if band.lower() in rates:
            print(f"   python add_flux_simple.py \\")
            print(f"       data/IC_10_X1_LC/{band}_converted/ \\")
            print(f"       data/IC_10_X1_LC/{band}_with_flux/ \\")
            print(f"       <conversion_factor_{band.lower()}>")
            print()
    
    print("="*80)
    
    # Save to file for reference
    with open('average_count_rates.txt', 'w') as f:
        f.write("# Time-averaged count rates for conversion factor calculation\n")
        f.write("# Use these with XSPEC model flux: conversion_factor = flux / count_rate\n")
        f.write("#\n")
        for band in bands:
            if band.lower() in rates:
                f.write(f"{band.lower()}_count_rate = {rates[band.lower()]:.6f} counts/s\n")
    
    print()
    print("Values saved to: average_count_rates.txt")
    print()


if __name__ == '__main__':
    main()

