#!/usr/bin/env python3
"""
Convert FITS light curve files to plain text format using Astropy.
Processes files in IC_10_X1_LC/Broad/, Soft/, and Hard/ directories.
"""

import os
from pathlib import Path
import numpy as np

try:
    from astropy.io import fits
except ImportError:
    print("Error: astropy is not installed. Install it with: pip install astropy")
    exit(1)

def convert_fits_to_txt(input_file, output_file):
    """
    Convert a FITS file to TXT using Astropy.
    
    Parameters:
    -----------
    input_file : str
        Path to input FITS file
    output_file : str
        Path to output TXT file
    """
    try:
        # Open FITS file
        with fits.open(input_file) as hdul:
            # Light curve data is typically in extension 1 (LIGHTCURVE)
            # Extension 0 is usually just the header
            if len(hdul) < 2:
                print(f"✗ No data extension found in {os.path.basename(input_file)}")
                return False
            
            # Get the light curve data
            data = hdul[1].data
            
            # Extract columns we want
            # Check which columns are available
            available_cols = data.columns.names
            
            # Define columns to extract (in order of preference)
            desired_cols = ['TIME', 'COUNTS', 'COUNT_RATE', 'COUNT_RATE_ERR', 
                           'EXPOSURE', 'NET_COUNTS', 'NET_RATE', 'ERR_RATE']
            
            # Get columns that exist in the file
            cols_to_extract = [col for col in desired_cols if col in available_cols]
            
            if not cols_to_extract:
                print(f"✗ No recognized columns found in {os.path.basename(input_file)}")
                return False
            
            # Write to text file
            with open(output_file, 'w') as f:
                # Write header
                f.write('# ' + '\t'.join(cols_to_extract) + '\n')
                
                # Write data
                for i in range(len(data)):
                    row = []
                    for col in cols_to_extract:
                        value = data[col][i]
                        # Format numbers appropriately
                        if isinstance(value, (int, np.integer)):
                            row.append(str(value))
                        else:
                            row.append(f"{value:.10e}")
                    f.write('\t'.join(row) + '\n')
        
        print(f"✓ Converted: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
        return True
        
    except Exception as e:
        print(f"✗ Error converting {input_file}: {str(e)}")
        return False


def main():
    # Base directory
    base_dir = Path(__file__).parent / 'data' / 'IC_10_X1_LC'
    
    # Subdirectories to process
    subdirs = ['Broad', 'Soft', 'Hard']
    
    total_converted = 0
    total_failed = 0
    
    for subdir in subdirs:
        input_dir = base_dir / subdir
        
        if not input_dir.exists():
            print(f"⚠ Directory not found: {input_dir}")
            continue
        
        # Create output directory
        output_dir = base_dir / f"{subdir}_converted"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Processing {subdir} directory...")
        print(f"{'='*60}")
        
        # Find all .txt files (which are actually FITS files)
        fits_files = sorted(input_dir.glob('*.txt'))
        
        if not fits_files:
            print(f"No files found in {input_dir}")
            continue
        
        print(f"Found {len(fits_files)} files to convert\n")
        
        for fits_file in fits_files:
            # Create output filename (keep same name)
            output_file = output_dir / fits_file.name
            
            # Convert
            if convert_fits_to_txt(str(fits_file), str(output_file)):
                total_converted += 1
            else:
                total_failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully converted: {total_converted} files")
    print(f"Failed: {total_failed} files")
    print(f"\nConverted files saved in:")
    for subdir in subdirs:
        output_dir = base_dir / f"{subdir}_converted"
        if output_dir.exists():
            print(f"  - {output_dir}")


if __name__ == '__main__':
    main()

