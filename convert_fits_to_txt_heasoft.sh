#!/bin/bash
# Convert FITS light curve files to plain text format using heasoft's fdump
# Usage: ./convert_fits_to_txt_heasoft.sh
#
# Make sure heasoft is initialized before running this script:
#   source /path/to/heasoft/headas-init.sh   (or .csh depending on your shell)

set -e  # Exit on error

# Base directory
BASE_DIR="data/IC_10_X1_LC"

# Subdirectories to process
SUBDIRS=("Broad" "Soft" "Hard")

# Columns to extract
COLUMNS="TIME,COUNTS,COUNT_RATE,COUNT_RATE_ERR,EXPOSURE"

total_converted=0
total_failed=0

# Check if fdump is available
if ! command -v fdump &> /dev/null; then
    echo "ERROR: fdump command not found!"
    echo "Please initialize heasoft first:"
    echo "  source /path/to/heasoft/headas-init.sh"
    exit 1
fi

echo "fdump found: $(which fdump)"
echo ""

# Process each subdirectory
for subdir in "${SUBDIRS[@]}"; do
    input_dir="${BASE_DIR}/${subdir}"
    output_dir="${BASE_DIR}/${subdir}_converted"
    
    if [ ! -d "$input_dir" ]; then
        echo "⚠ Directory not found: $input_dir"
        continue
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    echo "============================================================"
    echo "Processing ${subdir} directory..."
    echo "============================================================"
    
    # Count files
    file_count=$(ls -1 "$input_dir"/*.txt 2>/dev/null | wc -l)
    echo "Found $file_count files to convert"
    echo ""
    
    # Process each file
    for input_file in "$input_dir"/*.txt; do
        if [ ! -f "$input_file" ]; then
            continue
        fi
        
        filename=$(basename "$input_file")
        output_file="${output_dir}/${filename}"
        
        # Run fdump to convert
        if fdump infile="$input_file" outfile="$output_file" columns="$COLUMNS" rows="-" prhead=no showcol=yes showunit=no showrow=no pagewidth=256 page=no wrap=yes more=no clobber=yes 2>/dev/null; then
            echo "✓ Converted: $filename"
            ((total_converted++))
        else
            echo "✗ Failed: $filename"
            ((total_failed++))
        fi
    done
    
    echo ""
done

# Summary
echo "============================================================"
echo "CONVERSION SUMMARY"
echo "============================================================"
echo "Successfully converted: $total_converted files"
echo "Failed: $total_failed files"
echo ""
echo "Converted files saved in:"
for subdir in "${SUBDIRS[@]}"; do
    output_dir="${BASE_DIR}/${subdir}_converted"
    if [ -d "$output_dir" ]; then
        echo "  - $output_dir"
    fi
done

