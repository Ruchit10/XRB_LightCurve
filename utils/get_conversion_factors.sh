#!/bin/bash
# Shell script to run XSPEC and compute conversion factors for count-rate to flux
#
# This script runs xspec_get_conversion_factors.xcm and captures the output
# to help you determine conversion factors for each energy band.
#
# Usage:
#   ./get_conversion_factors.sh
#
# Requirements:
#   - heasoft must be initialized (heainit)
#   - Spectrum files must exist in data/IC10X1_spec/

set -e

# Check if XSPEC is available
if ! command -v xspec &> /dev/null; then
    echo "ERROR: xspec command not found!"
    echo "Please initialize heasoft first:"
    echo "  source /path/to/heasoft/headas-init.sh"
    exit 1
fi

# Check if spectrum directory exists
SPECDIR="data/IC10X1_spec"
if [ ! -d "$SPECDIR" ]; then
    echo "ERROR: Spectrum directory not found: $SPECDIR"
    echo "Please check the path to your spectrum files."
    exit 1
fi

# Check if spectrum files exist
SRCSPEC="$SPECDIR/X1_spectrum_combined_src.pi"
if [ ! -f "$SRCSPEC" ]; then
    echo "ERROR: Source spectrum not found: $SRCSPEC"
    exit 1
fi

echo "========================================================================"
echo "XSPEC CONVERSION FACTOR CALCULATOR"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Load your IC 10 X-1 spectrum from $SPECDIR"
echo "  2. Fit with tbabs*powerlaw model"
echo "  3. Calculate flux for broad, soft, and hard bands"
echo "  4. Show you how to compute conversion factors"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

echo ""
echo "Running XSPEC..."
echo "------------------------------------------------------------------------"

# Run XSPEC with the script
# Redirect output to both screen and file
LOGFILE="xspec_conversion_factors.log"
xspec < xspec_get_conversion_factors.xcm 2>&1 | tee "$LOGFILE"

echo ""
echo "========================================================================"
echo "COMPLETE!"
echo "========================================================================"
echo ""
echo "Output saved to: $LOGFILE"
echo "Fitted model saved to: xspec_fit_results.xcm"
echo ""
echo "------------------------------------------------------------------------"
echo "NEXT STEPS:"
echo "------------------------------------------------------------------------"
echo ""
echo "1. Look for the 'Model Flux' values in the output above"
echo "   (one each for broad, soft, and hard bands)"
echo ""
echo "2. Find your typical count rate from light curve files:"
echo "   head -50 data/IC_10_X1_LC/Broad_converted/11080_100s_broad.txt | grep -v 0.0000"
echo ""
echo "3. For each band, calculate:"
echo "   conversion_factor = model_flux / observed_count_rate"
echo ""
echo "4. Add flux columns to your light curves:"
echo "   python add_flux_simple.py \\"
echo "       data/IC_10_X1_LC/Broad_converted/ \\"
echo "       data/IC_10_X1_LC/Broad_with_flux/ \\"
echo "       <YOUR_CONVERSION_FACTOR>"
echo ""
echo "For help, see: QUICK_START_FLUX_CONVERSION.md"
echo ""

