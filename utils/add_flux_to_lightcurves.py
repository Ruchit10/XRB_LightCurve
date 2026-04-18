#!/usr/bin/env python3
"""
Add flux columns to converted light curve files.

Converts COUNT_RATE (counts/s) to FLUX (erg/cm²/s) using either:
1. A user-specified conversion factor
2. Computed from spectral model assuming fixed nH
3. WebPICT or PIMMS-derived conversion factor

The conversion factor depends on:
- Source spectrum (absorption + continuum model)
- Detector response
- Energy band

For IC 10 X-1, you can get this from:
- XSPEC: flux/count_rate from your spectral fit
- WebPICT: http://heasarc.gsfc.nasa.gov/cgi-bin/Tools/w3pimms/w3pimms.pl
"""

import argparse
import pandas as pd
from pathlib import Path
import numpy as np


def read_flux_vs_nh_csv(csv_path: str) -> pd.DataFrame:
    """Read flux vs nH CSV from compute_flux_vs_nH.py"""
    df = pd.read_csv(csv_path)
    return df


def compute_conversion_factor_from_spectrum(band: str, nH: float, flux_csv_dir: str) -> float:
    """
    Compute count-rate to flux conversion factor from spectral model.
    
    Parameters:
    -----------
    band : str
        Energy band ('broad', 'soft', 'medium', or 'hard')
    nH : float
        Column density in units of 1e22 cm^-2
    flux_csv_dir : str
        Directory containing flux_vs_nH_{band}.csv files
    
    Returns:
    --------
    float : Conversion factor such that flux = count_rate * factor
    """
    csv_path = Path(flux_csv_dir) / f"flux_vs_nH_{band}.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Flux CSV not found: {csv_path}")
    
    df = read_flux_vs_nh_csv(str(csv_path))
    
    # Interpolate to get flux at the specified nH
    # Use energy flux (erg/cm²/s)
    flux_col = f"flux_{band}_erg"
    
    if flux_col not in df.columns:
        raise ValueError(f"Column {flux_col} not found in CSV")
    
    # Log-log interpolation
    log_nH = np.log10(df['nH_1e22'].values)
    log_flux = np.log10(df[flux_col].values)
    
    flux_at_nH = 10 ** np.interp(np.log10(nH), log_nH, log_flux)
    
    return flux_at_nH


def add_flux_column(input_file: str, output_file: str, conversion_factor: float = None, 
                    nH: float = None, band: str = None, flux_csv_dir: str = None):
    """
    Add flux column to light curve file.
    
    Parameters:
    -----------
    input_file : str
        Path to input TXT file (from convert_fits_to_txt.py)
    output_file : str
        Path to output file with flux column added
    conversion_factor : float, optional
        Direct conversion factor (flux = count_rate * factor)
        If provided, this overrides nH-based calculation
    nH : float, optional
        Column density in 1e22 cm^-2 units (required if conversion_factor not given)
    band : str, optional
        Energy band ('broad', 'soft', 'medium', 'hard') - inferred from filename if not provided
    flux_csv_dir : str, optional
        Directory with flux_vs_nH CSV files (default: same directory as script)
    """
    # Read the light curve file
    df = pd.read_csv(input_file, sep='\t', comment='#')
    
    # Infer band from filename if not provided
    if band is None:
        filename = Path(input_file).name.lower()
        if 'broad' in filename:
            band = 'broad'
        elif 'soft' in filename:
            band = 'soft'
        elif 'medium' in filename:
            band = 'medium'
        elif 'hard' in filename:
            band = 'hard'
        else:
            raise ValueError(f"Cannot infer band from filename: {input_file}. Please specify --band")
    
    # Compute or use conversion factor
    if conversion_factor is None:
        if nH is None:
            raise ValueError("Must provide either --conversion-factor or --nH")
        
        if flux_csv_dir is None:
            flux_csv_dir = Path(__file__).parent
        
        # Get count rate at nH=0 (unabsorbed) to compute the conversion
        # This requires knowing the count rate from XSPEC
        # For now, we'll compute the flux and the user needs to provide count rate separately
        print(f"Computing flux for nH={nH}e22 cm^-2 using spectral model...")
        flux_at_nH = compute_conversion_factor_from_spectrum(band, nH, flux_csv_dir)
        
        # For actual conversion, we need: conversion_factor = flux / count_rate
        # Since we don't have the source count rate from spectral fit here,
        # we'll just use the flux value directly if user wants
        print(f"Note: To get conversion factor, divide expected flux ({flux_at_nH:.3e} erg/cm²/s)")
        print(f"      by your source count rate from spectral fitting")
        
        # For simplicity, let's assume a typical conversion
        # User should provide actual conversion factor from their spectral fit
        raise NotImplementedError(
            "To compute conversion factor, we need the source count rate from your spectral fit.\n"
            "Please use WebPICT or XSPEC to compute: conversion_factor = flux / count_rate\n"
            "Then run with: --conversion-factor VALUE"
        )
    
    # Apply conversion
    df['FLUX'] = df['COUNT_RATE'] * conversion_factor
    df['FLUX_ERR'] = df['COUNT_RATE_ERR'] * conversion_factor
    
    # Add metadata as comments
    with open(output_file, 'w') as f:
        f.write(f"# Light curve with flux column added\n")
        f.write(f"# Conversion factor: {conversion_factor:.6e} (erg/cm²/s) / (counts/s)\n")
        f.write(f"# Band: {band}\n")
        if nH is not None:
            f.write(f"# Assumed nH: {nH}e22 cm^-2\n")
        f.write("# " + "\t".join(df.columns) + "\n")
        
        # Write data
        df.to_csv(f, sep='\t', index=False, header=False, float_format='%.10e')
    
    print(f"✓ Added flux column: {Path(output_file).name}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Add flux columns to Chandra light curve files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using a conversion factor from XSPEC/WebPICT
  python add_flux_to_lightcurves.py input.txt output.txt --conversion-factor 2.5e-11
  
  # Process all files in a directory
  python add_flux_to_lightcurves.py \\
      data/IC_10_X1_LC/Broad_converted/ \\
      data/IC_10_X1_LC/Broad_with_flux/ \\
      --conversion-factor 2.5e-11
  
  # To get conversion factor from XSPEC:
  # In XSPEC, after fitting: flux 0.5 7.0
  # Then: print "Conversion = ", <flux_value> / <count_rate_from_data>
  
  # To get conversion factor from WebPICT:
  # Visit: http://heasarc.gsfc.nasa.gov/cgi-bin/Tools/w3pimms/w3pimms.pl
  # Input: Chandra ACIS-I, your spectral model, count rate → get flux
        """
    )
    
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('output', help='Output file or directory')
    parser.add_argument('--conversion-factor', type=float, 
                       help='Conversion factor: flux = count_rate * factor (erg/cm²/s per count/s)')
    parser.add_argument('--nH', type=float,
                       help='Column density in 1e22 cm^-2 (if not using conversion-factor)')
    parser.add_argument('--band', choices=['broad', 'soft', 'medium', 'hard'],
                       help='Energy band (auto-detected from filename if not specified)')
    parser.add_argument('--flux-csv-dir', default=None,
                       help='Directory containing flux_vs_nH_*.csv files')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Process single file
    if input_path.is_file():
        add_flux_column(
            str(input_path), 
            str(output_path),
            conversion_factor=args.conversion_factor,
            nH=args.nH,
            band=args.band,
            flux_csv_dir=args.flux_csv_dir
        )
        print(f"\n✓ Successfully converted 1 file")
        return
    
    # Process directory
    if input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True)
        elif not output_path.is_dir():
            print(f"Error: Output path exists but is not a directory: {output_path}")
            return
        
        txt_files = sorted(input_path.glob('*.txt'))
        
        if not txt_files:
            print(f"No .txt files found in {input_path}")
            return
        
        print(f"Processing {len(txt_files)} files...\n")
        
        success_count = 0
        for txt_file in txt_files:
            output_file = output_path / txt_file.name
            try:
                add_flux_column(
                    str(txt_file),
                    str(output_file),
                    conversion_factor=args.conversion_factor,
                    nH=args.nH,
                    band=args.band,
                    flux_csv_dir=args.flux_csv_dir
                )
                success_count += 1
            except Exception as e:
                print(f"✗ Failed to process {txt_file.name}: {e}")
        
        print(f"\n✓ Successfully converted {success_count}/{len(txt_files)} files")
        return
    
    print(f"Error: Input path not found: {input_path}")


if __name__ == '__main__':
    main()

