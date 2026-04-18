#!/usr/bin/env python3
"""
Simple script to add FLUX column to light curve files.
Uses only standard library - no external dependencies.

Usage:
    python add_flux_simple.py input.txt output.txt CONVERSION_FACTOR
    
Example:
    python add_flux_simple.py \\
        data/IC_10_X1_LC/Broad_converted/11080_100s_broad.txt \\
        data/IC_10_X1_LC/Broad_with_flux/11080_100s_broad.txt \\
        3.0e-11
"""

import sys
import os

def add_flux_column(input_file, output_file, conversion_factor):
    """Add FLUX and FLUX_ERR columns to light curve file."""
    
    try:
        factor = float(conversion_factor)
    except ValueError:
        print(f"Error: Invalid conversion factor: {conversion_factor}")
        return False
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return False
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Read input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Process and write output
    with open(output_file, 'w') as f:
        # Write header with metadata
        f.write(f"# Light curve with FLUX columns added\n")
        f.write(f"# Conversion factor: {factor:.6e} (erg/cm²/s)/(count/s)\n")
        f.write(f"# Source file: {os.path.basename(input_file)}\n")
        f.write("#\n")
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Handle header/comment lines
            if line.startswith('#'):
                # Find the column header line and add FLUX columns
                if 'TIME' in line and 'COUNT_RATE' in line:
                    f.write(line.rstrip() + '\tFLUX\tFLUX_ERR\n')
                else:
                    f.write(line)
                continue
            
            # Process data lines
            try:
                parts = line.strip().split('\t')
                
                # Find COUNT_RATE and COUNT_RATE_ERR columns
                # Typical format: TIME COUNTS COUNT_RATE COUNT_RATE_ERR EXPOSURE ...
                if len(parts) >= 4:
                    count_rate = float(parts[2])
                    count_rate_err = float(parts[3])
                    
                    # Calculate flux
                    flux = count_rate * factor
                    flux_err = count_rate_err * factor
                    
                    # Write line with flux columns
                    f.write(line.rstrip() + f'\t{flux:.10e}\t{flux_err:.10e}\n')
                else:
                    # Write line as-is if format unexpected
                    f.write(line)
                    
            except (ValueError, IndexError) as e:
                # Write line as-is if parsing fails
                f.write(line)
    
    return True


def process_directory(input_dir, output_dir, conversion_factor):
    """Process all .txt files in a directory."""
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .txt files
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    print(f"Processing {len(txt_files)} files...")
    print()
    
    success = 0
    for filename in sorted(txt_files):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        
        if add_flux_column(input_file, output_file, conversion_factor):
            print(f"✓ {filename}")
            success += 1
        else:
            print(f"✗ {filename}")
    
    print()
    print(f"Successfully processed {success}/{len(txt_files)} files")


def main():
    if len(sys.argv) < 4:
        print("Usage: python add_flux_simple.py INPUT OUTPUT CONVERSION_FACTOR")
        print()
        print("Examples:")
        print("  # Single file")
        print("  python add_flux_simple.py input.txt output.txt 3.0e-11")
        print()
        print("  # Directory")
        print("  python add_flux_simple.py \\")
        print("      data/IC_10_X1_LC/Broad_converted/ \\")
        print("      data/IC_10_X1_LC/Broad_with_flux/ \\")
        print("      3.0e-11")
        print()
        print("To get the conversion factor, run:")
        print("  python compute_count_to_flux_factor.py")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    conversion_factor = sys.argv[3]
    
    # Check if input is file or directory
    if os.path.isfile(input_path):
        print(f"Converting: {os.path.basename(input_path)}")
        if add_flux_column(input_path, output_path, conversion_factor):
            print(f"✓ Success! Output saved to: {output_path}")
        else:
            print("✗ Conversion failed")
            sys.exit(1)
    
    elif os.path.isdir(input_path):
        process_directory(input_path, output_path, conversion_factor)
    
    else:
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()

