#!/usr/bin/env python3
"""
Extract the fitted nH value from XSPEC fit results.

This value should be used as the 'lam' parameter in xrb_lightcurve.py
to scale model nH to match the XSPEC fitted value.

Usage:
    python get_xspec_nH.py
    python get_xspec_nH.py --xcm xspec_fit_results.xcm
"""

import argparse
import re
import sys

def extract_nH_from_xcm(xcm_file):
    """Extract TBabs nH parameter from XSPEC .xcm file."""
    try:
        with open(xcm_file, 'r') as f:
            content = f.read()
        
        # Look for the model line and TBabs parameter
        # Format: model  TBabs*powerlaw
        #         nH_value ...
        
        lines = content.split('\n')
        in_model = False
        
        for i, line in enumerate(lines):
            # Find the model definition
            if 'model' in line.lower() and ('tbabs' in line.lower() or 'TBabs' in line.lower()):
                in_model = True
                continue
            
            # The next non-empty line after model should be nH
            if in_model and line.strip():
                # Parse the parameter value (first number on the line)
                match = re.search(r'([\d.eE+-]+)', line.strip())
                if match:
                    nH = float(match.group(1))
                    return nH
        
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract fitted nH from XSPEC results for use in xrb_lightcurve.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python get_xspec_nH.py
  python get_xspec_nH.py --xcm my_fit.xcm
  
The extracted nH value should be used as the --lam parameter:
  python xrb_lightcurve.py --lam <extracted_nH> ...
        """
    )
    
    parser.add_argument(
        '--xcm',
        type=str,
        default='xspec_fit_results.xcm',
        help='Path to XSPEC .xcm file (default: xspec_fit_results.xcm)'
    )
    
    args = parser.parse_args()
    
    print()
    print("="*70)
    print("EXTRACT nH FROM XSPEC FIT")
    print("="*70)
    print()
    
    nH = extract_nH_from_xcm(args.xcm)
    
    if nH is None:
        print(f"❌ Could not extract nH from {args.xcm}")
        print()
        print("Make sure you have run XSPEC and saved the fit:")
        print("  ./get_conversion_factors.sh")
        print()
        sys.exit(1)
    
    print(f"XSPEC fitted nH: {nH} × 10²² cm⁻²")
    print(f"              or: {nH * 1e22:.3e} cm⁻²")
    print()
    
    # Compare with legacy value
    legacy_lam = 0.589537
    diff_percent = (nH - legacy_lam) / legacy_lam * 100
    
    print("-"*70)
    print("COMPARISON WITH LEGACY VALUE")
    print("-"*70)
    print(f"Legacy lam:  {legacy_lam}")
    print(f"XSPEC nH:    {nH}")
    print(f"Difference:  {diff_percent:+.1f}%")
    print()
    
    if abs(diff_percent) > 10:
        print("⚠️  Significant difference (>10%)!")
        print("   Recommend using XSPEC value for accuracy.")
    elif abs(diff_percent) > 5:
        print("⚠️  Moderate difference (5-10%)")
        print("   Consider using XSPEC value.")
    else:
        print("✓ Small difference (<5%)")
        print("  Both values acceptable.")
    print()
    
    print("-"*70)
    print("HOW TO USE THIS VALUE")
    print("-"*70)
    print()
    print("Option 1: Command line argument")
    print(f"  python xrb_lightcurve.py --lam {nH} --output my_results.csv")
    print()
    print("Option 2: Update default in xrb_lightcurve.py")
    print(f"  Change: lam: float = 0.589537")
    print(f"  To:     lam: float = {nH}")
    print()
    print("Option 3: Save to file for scripting")
    print(f"  echo '{nH}' > xspec_nH.txt")
    print(f"  LAM=$(cat xspec_nH.txt)")
    print(f"  python xrb_lightcurve.py --lam $LAM ...")
    print()
    
    # Save to file
    with open('xspec_nH.txt', 'w') as f:
        f.write(f"{nH}\n")
    
    print("✓ nH value saved to: xspec_nH.txt")
    print()
    print("="*70)
    print()


if __name__ == '__main__':
    main()

