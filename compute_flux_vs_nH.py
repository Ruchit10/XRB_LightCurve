#!/usr/bin/env python3
"""
Compute photon flux as a function of hydrogen column density (nH) for X-ray energy bands
using XSPEC via its Python API.

- Loads spectrum data (PHA + responses) from a directory (default: ./data/IC10X1_spec)
- Fits a simple model (default: tbabs*powerlaw)
- Freezes the continuum and varies nH over a grid
- For each nH, computes photon flux in instrument-specific energy bands
- Saves a CSV and a comparison plot

Supported Instruments and Energy Bands:
  Chandra (default):
    - broad: 0.5–8.0 keV (default)
    - soft: 0.5–2.0 keV
    - hard: 2.0–8.0 keV

Run this under the conda environment that has XSPEC Python (e.g., `henv`).

Example:
  python compute_flux_vs_nH.py \
      --specdir ./data/IC10X1_spec \
      --instrument chandra \
      --bands broad soft hard \
      --out_csv flux_vs_nH.csv \
      --out_png flux_vs_nH.png \
      --nH_min 1e20 --nH_max 1e24 --nH_points 60

Notes:
- XSPEC tbabs.nH parameter is in 1e22 cm^-2. This script accepts nH in cm^-2 and converts accordingly.
- Photon flux is computed by integrating the unfolded model spectrum (photons/cm^2/s/keV) over the band.
  If that fails, we fall back to XSPEC's band energy flux for visibility (also saved in CSV).
"""

import argparse
import glob
import os
import sys
import re
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# XSPEC must be available in the active Python environment
try:
    from xspec import (  # type: ignore
        AllData,
        AllModels,
        Model,
        Fit,
        Xset,
        Plot,
    )
except Exception as exc:
    print("Error: XSPEC Python module not available in this environment.")
    print("Ensure you run this under your conda env (e.g., 'henv') with XSPEC installed.")
    print(f"Details: {exc}")
    sys.exit(1)


# ========== INSTRUMENT-SPECIFIC ENERGY BANDS ==========
# Define energy bands (in keV) for different X-ray instruments
# Structure: {instrument_name: {band_name: (E_min, E_max)}}

INSTRUMENT_BANDS = {
    "chandra": {
        "broad": (0.5, 8.0),    # Default broad band
        "soft": (0.5, 2.0),     # Soft band
        "hard": (2.0, 8.0),     # Hard band
    },
    # Future instruments can be added here, e.g.:
    # "xmm": {
    #     "broad": (0.2, 12.0),
    #     "soft": (0.2, 2.0),
    #     "hard": (2.0, 12.0),
    # },
}

# Default bands to use if none specified
DEFAULT_BANDS = {
    "chandra": ["broad"],
}


def _pick_first_matching(paths: List[str], prefer_keywords: Optional[List[str]] = None) -> Optional[str]:
    if not paths:
        return None
    if prefer_keywords:
        lowered = [p.lower() for p in paths]
        for key in prefer_keywords:
            for idx, lp in enumerate(lowered):
                if key in lp:
                    return paths[idx]
    return paths[0]


def find_spectrum_files(specdir: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Locate source PHA/PI, optional background PHA/PI, and responses in specdir.
    Returns (src_pha_or_pi, bkg_pha_or_pi, rmf, arf)
    """
    # Candidates for spectrum files
    pha_patterns = ["*.pha", "*.pha.gz", "*.pi", "*.pi.gz"]
    all_spec = []
    for pat in pha_patterns:
        all_spec.extend(sorted(glob.glob(os.path.join(specdir, pat))))

    if not all_spec:
        raise FileNotFoundError(f"No PHA/PI files found in {specdir}")

    # Choose source and background
    src = _pick_first_matching(all_spec, prefer_keywords=["_src", "source", "src"]) or all_spec[0]

    # Remove chosen src to avoid picking it for bkg
    remaining = [p for p in all_spec if p != src]
    bkg = _pick_first_matching(remaining, prefer_keywords=["_bkg", "background", "bkg"]) if remaining else None

    # Responses: prefer ones tagged as src, else first available
    rmf_list = sorted(glob.glob(os.path.join(specdir, "*.rmf")))
    arf_list = sorted(glob.glob(os.path.join(specdir, "*.arf")))

    rmf = _pick_first_matching(rmf_list, prefer_keywords=["_src", "source", "src"]) or (rmf_list[0] if rmf_list else None)
    arf = _pick_first_matching(arf_list, prefer_keywords=["_src", "source", "src"]) or (arf_list[0] if arf_list else None)

    return src, bkg, rmf, arf


def load_data(pha_or_pi: str, bkg: Optional[str], rmf: Optional[str], arf: Optional[str]) -> None:
    """Load spectrum into XSPEC and attach background; rely on PHA headers for RMF/ARF.

    To avoid XSPEC interactive prompts (e.g., when response paths are relative),
    we temporarily change into the spectrum directory and load by basenames.
    """
    AllData.clear()
    AllModels.clear()

    specdir = os.path.dirname(os.path.abspath(pha_or_pi))
    src_base = os.path.basename(pha_or_pi)
    bkg_base = os.path.basename(bkg) if bkg else None

    cwd = os.getcwd()
    try:
        os.chdir(specdir)
        AllData(src_base)
        try:
            sp = AllData(1)
            if bkg_base:
                sp.background = bkg_base
            # Do NOT set response/arf manually; let XSPEC use RESPFILE/ANCRFILE from header
        except Exception:
            pass
    finally:
        os.chdir(cwd)


def setup_model_with_params(model_expr: str = "tbabs*powerlaw",
                            nH: float = 0.572385,
                            PhoIndex: float = 1.92146,
                            norm: float = 3.37606e-04) -> None:
    """Set up the model with pre-fitted parameter values (no fitting performed).
    
    Uses user-specified parameter values and freezes the continuum parameters
    so only nH can be varied later.
    
    Args:
        model_expr: XSPEC model expression
        nH: Hydrogen column density in 10^22 cm^-2 (default: 0.572385)
        PhoIndex: Power-law photon index (default: 1.92146)
        norm: Power-law normalization (default: 3.37606e-04)
    """
    # Set abundance and cross-section tables
    Xset.abund = "wilm"
    Xset.xsect = "vern"
    
    Model(model_expr)

    # Ensure ALL channels are noticed (no energy filtering)
    try:
        AllData.notice("all")
        print("Noticed all channels")
    except Exception as e:
        print(f"Warning: Could not notice all channels: {e}")

    # Set pre-fitted parameter values for absorption component (tbabs or phabs)
    try:
        model = AllModels(1)
        if hasattr(model, 'TBabs'):
            model.TBabs.nH = nH
        elif hasattr(model, 'tbabs'):
            model.tbabs.nH = nH
        elif hasattr(model, 'phabs'):
            model.phabs.nH = nH
        elif hasattr(model, 'Phabs'):
            model.Phabs.nH = nH
        else:
            print(f"Warning: No recognized absorption model found (tbabs/phabs)")
    except Exception as e:
        print(f"Warning: Could not set nH: {e}")
    
    try:
        model = AllModels(1)
        if hasattr(model, 'powerlaw'):
            model.powerlaw.PhoIndex = PhoIndex
            model.powerlaw.norm = norm
        elif hasattr(model, 'Powerlaw'):
            model.Powerlaw.PhoIndex = PhoIndex
            model.Powerlaw.norm = norm
    except Exception as e:
        print(f"Warning: Could not set powerlaw parameters: {e}")

    # Print the parameters being used
    print(f"Using pre-fitted model parameters (no fitting):")
    print(f"  nH = {nH:.6f} (10^22 cm^-2)")
    print(f"  PhoIndex = {PhoIndex:.5f}")
    print(f"  norm = {norm:.5e}")
    
    # Verify parameters are set by printing chi-squared statistic
    try:
        # Query the fit statistic without actually fitting
        Fit.statMethod = "chi"
        chi_sq = Fit.statistic
        dof = Fit.dof
        print(f"  Chi-squared (with these params) = {chi_sq:.2f}, DOF = {dof}")
        print(f"  Reduced chi-squared = {chi_sq/dof:.3f}")
    except Exception as e:
        print(f"  (Could not compute chi-squared: {e})")

    # Freeze continuum so only nH varies when computing flux grid
    try:
        model = AllModels(1)
        if hasattr(model, 'powerlaw'):
            model.powerlaw.PhoIndex.frozen = True
            model.powerlaw.norm.frozen = True
        elif hasattr(model, 'Powerlaw'):
            model.Powerlaw.PhoIndex.frozen = True
            model.Powerlaw.norm.frozen = True
        print("Froze powerlaw parameters (PhoIndex, norm)")
    except Exception:
        pass


def integrate_photon_flux(E: np.ndarray, y: np.ndarray, band: Tuple[float, float]) -> float:
    """Integrate photon spectrum y(E) over [E1,E2] in keV using trapezoidal rule."""
    e1, e2 = band
    mask = (E >= e1) & (E <= e2)
    if np.count_nonzero(mask) < 2:
        # Attempt to include nearest bins if band edges are between bin centers
        idx = np.argsort(np.abs(E - np.clip((e1 + e2) / 2.0, E.min(), E.max())))[:3]
        mask[idx] = True
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.trapz(y[mask], E[mask]))


def compute_energy_flux_for_band(band: Tuple[float, float]) -> float:
    """Compute band energy flux (erg/cm^2/s) via XSPEC's native calculator with robust fallbacks."""
    e1, e2 = band

    # Try global calcFlux
    try:
        band_str = f"{e1} {e2}"
        res = AllModels.calcFlux(band_str)
        # When spectra are loaded, calcFlux populates each Spectrum object's `.flux` attribute but
        # often returns `None`.  Retrieve the value explicitly from the first spectrum to make
        # the behaviour version-independent.
        try:
            sp = AllData(1)
            if sp and hasattr(sp, "flux") and len(sp.flux) >= 1 and np.isfinite(sp.flux[0]):
                return float(sp.flux[0])
        except Exception as e:
            pass
        if isinstance(res, (list, tuple, np.ndarray)) and len(res) > 0 and np.isfinite(res[0]):
            return float(res[0])
        if isinstance(res, (int, float)) and np.isfinite(res):
            return float(res)
    except Exception as e:
        pass

    # Try model-specific calcFlux (source/model index 1)
    try:
        mdl = AllModels(1)
        band_str = f"{e1} {e2}"
        res = mdl.calcFlux(band_str)
        try:
            sp = AllData(1)
            if sp and hasattr(sp, "flux") and len(sp.flux) >= 1 and np.isfinite(sp.flux[0]):
                return float(sp.flux[0])
        except Exception as e:
            pass
        if isinstance(res, (list, tuple, np.ndarray)) and len(res) > 0 and np.isfinite(res[0]):
            return float(res[0])
        if isinstance(res, (int, float)) and np.isfinite(res):
            return float(res)
    except Exception as e:
        pass

    # Fallback: capture 'flux e1 e2' command output to a temp log and parse numeric value
    try:
        tmp_log = "_xspec_flux_tmp.log"
        # Remove existing tmp file
        try:
            if os.path.exists(tmp_log):
                os.remove(tmp_log)
        except Exception:
            pass

        Xset.openLog(tmp_log)
        # This prints to the log. XSPEC reports energy flux by default (erg/cm^2/s)
        Xset.command(f"flux {e1} {e2}")
        Xset.closeLog()

        # Parse the last numeric in the file
        val = np.nan
        with open(tmp_log, "r") as fh:
            content = fh.read()
            for line in fh:
                # Look for a number in scientific notation
                m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                if m:
                    try:
                        # take last number on the line
                        candidate = float(m[-1])
                        # XSPEC prints error bars too; keep positive plausible values
                        if np.isfinite(candidate) and candidate > 0:
                            val = candidate
                    except Exception:
                        continue
        # Cleanup
        try:
            os.remove(tmp_log)
        except Exception:
            pass

        return float(val) if np.isfinite(val) else float("nan")
    except Exception as e:
        return float("nan")


def compute_photon_flux_for_band(band: Tuple[float, float]) -> float:
    """
    Compute photon flux by integrating the unfolded model spectrum over the band.
    Returns photons/cm^2/s in the specified energy range.
    Tries XSPEC's native band flux calculator first, then falls back to plot integration.
    """
    e1, e2 = band

    # First try XSPEC's native calculation (energy flux); not photon, but allows a non-empty value
    # We still prefer photon flux from unfolded model below.
    # (We will call this again in the caller as a fallback if photon integration fails.)

    # Configure plot to get unfolded spectrum in photons/cm^2/s/keV
    Xset.chatter = 5
    Plot.xAxis = "keV"
    try:
        Xset.command("setplot energy")
        Xset.command("setplot ufspec")
    except Exception as e:
        # Note: Some XSPEC versions may not have Xset.command method
        return float("nan")

    # Build model arrays
    # Use the special "/null" device to suppress on-screen PGPLOT windows.
    Plot.device = "/null"
    E = np.array([])
    y = np.array([])
    
    try:
        # Primary: use unfolded spectrum directly
        Xset.command("setplot energy")
        Plot("ufspec")
        E = np.array(Plot.x(1), dtype=float)
        y = np.array(Plot.y(1), dtype=float)
        
        if y.size < 2:
            # Secondary: well-sampled energy grid with model values
            Xset.command("energies 0.1 20.0 2000 log")
            Plot("model")
            E = np.array(Plot.x(1), dtype=float)
            y = np.array(Plot.model(1), dtype=float)
    except Exception as e:
        return float("nan")

    if E.size < 2 or y.size < 2:
        return float("nan")

    return integrate_photon_flux(E, y, band)


def fit_exponential_decay(nH_cm2: np.ndarray, flux: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Fit exponential decay function to flux vs nH data in LOG SPACE.
    
    Formula: flux_scaled = A * exp(-B * nH_1e22)
    where nH_1e22 = nH_cm2 / 1e22
    
    Taking log: log(flux_scaled) = log(A) - B * nH_1e22
    This becomes a linear fit in log space, giving equal weight to all points
    regardless of magnitude - appropriate for data spanning many orders of magnitude.
    
    Args:
        nH_cm2: Array of nH values in cm^-2
        flux: Array of flux values (photons/cm^2/s or erg/cm^2/s)
    
    Returns:
        A: Fitted coefficient A (in units of 1e-13)
        B: Fitted coefficient B (in units of 1e22 cm^-2)
        flux_fit: Fitted flux values for the input nH_cm2 array
    """
    # Filter out invalid data points
    mask = (nH_cm2 > 0) & (flux > 0) & np.isfinite(nH_cm2) & np.isfinite(flux)
    
    if np.sum(mask) < 3:
        # Not enough valid points for fitting
        return float('nan'), float('nan'), np.full_like(nH_cm2, np.nan)
    
    nH_valid = nH_cm2[mask]
    flux_valid = flux[mask]
    
    # Convert nH to 1e22 units
    nH_1e22 = nH_valid / 1e22
    
    # Scale flux by 1e-13 to match the legacy approach
    flux_scaled = flux_valid / 1e-13
    
    # Take logarithm for fitting in log space
    log_flux_scaled = np.log(flux_scaled)
    
    # Define linear function for log-space fitting: log(flux) = log(A) - B * nH
    def linear_func(x, log_A, B):
        return log_A - B * x
    
    try:
        # Fit in log space with reasonable initial guesses
        # Initial guess: log(A) ≈ log(10) ≈ 2.3, B ≈ 0.1
        popt, _ = curve_fit(
            linear_func,
            nH_1e22,
            log_flux_scaled,
            p0=[2.3, 0.1],  # Initial guess for log(A) and B
            maxfev=10000
        )
        log_A, B = popt
        A = np.exp(log_A)  # Convert back from log space
        
        # Compute fitted flux for all input nH values
        nH_all_1e22 = nH_cm2 / 1e22
        flux_fit = A * np.exp(-B * nH_all_1e22) * 1e-13
        
        return float(A), float(B), flux_fit
        
    except Exception as e:
        # Fitting failed
        return float('nan'), float('nan'), np.full_like(nH_cm2, np.nan)


def vary_nh_and_compute(specdir: str,
                        nH_values_cm2: np.ndarray,
                        bands: dict,
                        model_expr: str = "tbabs*powerlaw",
                        nH: float = 0.572385,
                        PhoIndex: float = 1.92146,
                        norm: float = 3.37606e-04) -> pd.DataFrame:
    """
    For each nH (cm^-2), set tbabs.nH and compute photon flux in specified bands.
    
    Sets up the model with pre-fitted parameters (no fitting performed),
    then varies nH and computes flux for each specified energy band.
    
    Args:
        specdir: Directory containing spectrum files
        nH_values_cm2: Array of nH values in cm^-2
        bands: Dictionary mapping band names to (E_min, E_max) tuples in keV
        model_expr: XSPEC model expression
        nH: Pre-fitted hydrogen column density in 10^22 cm^-2 (default: 0.572385)
        PhoIndex: Pre-fitted power-law photon index (default: 1.92146)
        norm: Pre-fitted power-law normalization (default: 3.37606e-04)
    
    Returns:
        DataFrame with columns: nH_cm2, nH_1e22, flux_{band}_ph, flux_{band}_erg for each band
    """
    src, bkg, rmf, arf = find_spectrum_files(specdir)
    load_data(src, bkg, rmf, arf)

    setup_model_with_params(model_expr, nH=nH, PhoIndex=PhoIndex, norm=norm)

    results = []
    for nH_cm2 in nH_values_cm2:
        nH_1e22 = float(nH_cm2 / 1.0e22)
        result_row = {
            "nH_cm2": float(nH_cm2),
            "nH_1e22": nH_1e22,
        }
        
        try:
            # Try both possible capitalizations of tbabs/TBabs and phabs/Phabs
            model = AllModels(1)
            if hasattr(model, 'TBabs'):
                model.TBabs.nH = nH_1e22
            elif hasattr(model, 'tbabs'):
                model.tbabs.nH = nH_1e22
            elif hasattr(model, 'phabs'):
                model.phabs.nH = nH_1e22
            elif hasattr(model, 'Phabs'):
                model.Phabs.nH = nH_1e22
            else:
                raise AttributeError("Model has no recognized absorption component (tbabs/phabs)")
        except Exception as e:
            # If model doesn't have recognized absorption, fill with NaNs for all bands
            for band_name in bands.keys():
                result_row[f"flux_{band_name}_ph"] = float("nan")
                result_row[f"flux_{band_name}_erg"] = float("nan")
        else:
            # Compute photon flux for each band
            for band_name, band_range in bands.items():
                # Try to compute photon flux via unfolded model
                flux_ph = compute_photon_flux_for_band(band_range)
                
                # Also compute energy flux via XSPEC calculator for visibility/fallback
                flux_erg = compute_energy_flux_for_band(band_range)
                
                # If photon flux failed, fall back to energy flux (so plot is not empty)
                if not np.isfinite(flux_ph) or flux_ph <= 0:
                    flux_ph = flux_erg
                
                result_row[f"flux_{band_name}_ph"] = flux_ph
                result_row[f"flux_{band_name}_erg"] = flux_erg

        results.append(result_row)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Compute photon flux vs nH using XSPEC for instrument-specific energy bands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--specdir",
        type=str,
        default=os.path.join(os.getcwd(), "data", "IC10X1_spec"),
        help="Directory containing PHA/PI (+RMF/ARF) spectrum files",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="chandra",
        choices=list(INSTRUMENT_BANDS.keys()),
        help="X-ray instrument (determines available energy bands)",
    )
    parser.add_argument(
        "--bands",
        type=str,
        nargs="+",
        default=None,
        help="Energy bands to compute (space-separated). If not specified, uses instrument default.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tbabs*powerlaw",
        choices=["tbabs*powerlaw", "phabs*powerlaw"],
        help="XSPEC absorption model (parameters set from pre-fitted values, no fitting performed)",
    )
    
    # Pre-fitted model parameters (no fitting performed)
    parser.add_argument(
        "--nH",
        type=float,
        default=0.572385,
        help="Pre-fitted nH value in 10^22 cm^-2 (from XSPEC fit)",
    )
    parser.add_argument(
        "--PhoIndex",
        type=float,
        default=1.92146,
        help="Pre-fitted power-law photon index (from XSPEC fit)",
    )
    parser.add_argument(
        "--norm",
        type=float,
        default=3.37606e-04,
        help="Pre-fitted power-law normalization (from XSPEC fit)",
    )
    
    parser.add_argument("--nH_min", type=float, default=1e15, help="Min nH (cm^-2) for flux grid")
    parser.add_argument("--nH_max", type=float, default=1e26, help="Max nH (cm^-2) for flux grid")
    parser.add_argument("--nH_points", type=int, default=1000, help="Number of nH grid points (log-spaced)")

    parser.add_argument("--out_csv", type=str, default="flux_vs_nH.csv", help="Output CSV filename")
    parser.add_argument("--out_png", type=str, default="flux_vs_nH.png", help="Output PNG filename")

    args = parser.parse_args()

    if not os.path.isdir(args.specdir):
        print(f"Error: specdir not found: {args.specdir}")
        sys.exit(1)

    # Get instrument-specific bands
    instrument_bands = INSTRUMENT_BANDS[args.instrument]
    
    # Determine which bands to use
    if args.bands is None:
        # Use default bands for the instrument
        band_names = DEFAULT_BANDS.get(args.instrument, list(instrument_bands.keys()))
    else:
        # Validate requested bands
        band_names = args.bands
        invalid_bands = [b for b in band_names if b not in instrument_bands]
        if invalid_bands:
            print(f"Error: Invalid bands for {args.instrument}: {invalid_bands}")
            print(f"Available bands: {list(instrument_bands.keys())}")
            sys.exit(1)
    
    # Build bands dictionary
    bands = {name: instrument_bands[name] for name in band_names}
    
    print(f"Computing flux vs nH for {args.instrument} instrument")
    print(f"Energy bands:")
    for name, (emin, emax) in bands.items():
        print(f"  {name}: {emin}-{emax} keV")

    # Build nH grid (log-spaced)
    nH_values_cm2 = np.logspace(np.log10(args.nH_min), np.log10(args.nH_max), args.nH_points)

    # Compute flux vs nH using pre-fitted model parameters
    df = vary_nh_and_compute(
        specdir=args.specdir,
        nH_values_cm2=nH_values_cm2,
        bands=bands,
        model_expr=args.model,
        nH=args.nH,
        PhoIndex=args.PhoIndex,
        norm=args.norm,
    )

    # Save CSV
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv} ({len(df)} rows)")

    # Plot
    try:
        import matplotlib.pyplot as plt

        # Clean data for plotting: positive nH and flux only
        dfp = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["nH_cm2"]).copy()

        fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.0))

        # Define colors and markers for different bands
        colors = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
        markers = ["o", "s", "^", "D", "v", "p"]
        
        any_plotted = False
        yvals = []
        fit_text_lines = []  # Store fitted equations for display
        
        # Plot each band
        for idx, (band_name, (emin, emax)) in enumerate(bands.items()):
            col_name_ph = f"flux_{band_name}_ph"
            col_name_erg = f"flux_{band_name}_erg"
            
            # Try photon flux first
            if col_name_ph in dfp.columns:
                mask = (dfp["nH_cm2"] > 0) & (dfp[col_name_ph] > 0)
                if mask.any():
                    # Plot data points
                    ax.plot(
                        dfp.loc[mask, "nH_cm2"],
                        dfp.loc[mask, col_name_ph],
                        label=f"{band_name.capitalize()} {emin}-{emax} keV (data)",
                        color=colors[idx % len(colors)],
                        marker=markers[idx % len(markers)],
                        markersize=3,
                        linewidth=0,
                        alpha=0.7,
                    )
                    yvals.append(dfp.loc[mask, col_name_ph].values)
                    any_plotted = True
                    
                    # Fit exponential decay
                    nH_data = dfp.loc[mask, "nH_cm2"].values
                    flux_data = dfp.loc[mask, col_name_ph].values
                    A, B, flux_fit = fit_exponential_decay(nH_data, flux_data)
                    
                    # Plot fitted curve if successful
                    if np.isfinite(A) and np.isfinite(B):
                        # Sort for smooth line
                        sort_idx = np.argsort(nH_data)
                        ax.plot(
                            nH_data[sort_idx],
                            flux_fit[sort_idx],
                            label=f"{band_name.capitalize()} fit",
                            color=colors[idx % len(colors)],
                            linestyle="--",
                            linewidth=2,
                            alpha=0.9,
                        )
                        yvals.append(flux_fit[sort_idx])
                        
                        # Store equation text
                        fit_text_lines.append(
                            f"{band_name.capitalize()}: $F = {A:.3f} \\times 10^{{-13}} \\cdot e^{{-{B:.4f} \\cdot n_H}}$"
                        )
                    
                # Fall back to energy flux if photon flux not available
                elif col_name_erg in dfp.columns:
                    mask_erg = (dfp["nH_cm2"] > 0) & (dfp[col_name_erg] > 0)
                    if mask_erg.any():
                        # Plot data points
                        ax.plot(
                            dfp.loc[mask_erg, "nH_cm2"],
                            dfp.loc[mask_erg, col_name_erg],
                            label=f"{band_name.capitalize()} {emin}-{emax} keV (erg, data)",
                            color=colors[idx % len(colors)],
                            marker=markers[idx % len(markers)],
                            markersize=3,
                            linewidth=0,
                            alpha=0.7,
                        )
                        yvals.append(dfp.loc[mask_erg, col_name_erg].values)
                        any_plotted = True
                        
                        # Fit exponential decay
                        nH_data = dfp.loc[mask_erg, "nH_cm2"].values
                        flux_data = dfp.loc[mask_erg, col_name_erg].values
                        A, B, flux_fit = fit_exponential_decay(nH_data, flux_data)
                        
                        # Plot fitted curve if successful
                        if np.isfinite(A) and np.isfinite(B):
                            sort_idx = np.argsort(nH_data)
                            ax.plot(
                                nH_data[sort_idx],
                                flux_fit[sort_idx],
                                label=f"{band_name.capitalize()} fit",
                                color=colors[idx % len(colors)],
                                linestyle="--",
                                linewidth=2,
                                alpha=0.9,
                            )
                            yvals.append(flux_fit[sort_idx])
                            
                            # Store equation text
                            fit_text_lines.append(
                                f"{band_name.capitalize()}: $F = {A:.3f} \\times 10^{{-13}} \\cdot e^{{-{B:.4f} \\cdot n_H}}$"
                            )

        # Force log scales for readability
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Compute sensible limits from available data
        if yvals:
            yvals = np.concatenate(yvals)
            yvals = yvals[yvals > 0]
            if yvals.size > 0:
                ymin = yvals.min() * 0.8
                ymax = yvals.max() * 1.2
                if ymin > 0 and ymax > ymin:
                    ax.set_ylim(ymin, ymax)

        # X limits from available nH
        if len(dfp) > 0:
            xmin = dfp["nH_cm2"].min() * 0.8
            xmax = dfp["nH_cm2"].max() * 1.2
            if xmin > 0 and xmax > xmin:
                ax.set_xlim(xmin, xmax)

        ax.set_xlabel("nH (cm$^{-2}$)", fontsize=12)
        ax.set_ylabel("Flux (photons or ergs cm$^{-2}$ s$^{-1}$)", fontsize=12)
        ax.set_title(f"Flux vs nH ({args.instrument.capitalize()})", fontsize=14, fontweight='bold')
        ax.grid(True, which="both", alpha=0.3)
        
        # Add legend
        if any_plotted:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        
        # Add text box with fitted equations
        if fit_text_lines:
            equation_text = "Fitted Equations:\n" + "\n".join(fit_text_lines)
            equation_text += "\n\n$n_H$ in units of $10^{22}$ cm$^{-2}$"
            
            # Position text box in lower left
            ax.text(
                0.02, 0.02,
                equation_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5)
            )
            
            # Also print fitted coefficients to console
            print("\nFitted exponential decay coefficients:")
            print("Formula: F = A × 10^(-13) × exp(-B × nH), where nH is in 10^22 cm^-2")
            for line in fit_text_lines:
                # Extract band name (before the colon)
                band_text = line.split(':')[0]
                print(f"  {line.replace('$', '').replace('\\times', '×').replace('\\cdot', '×').replace('{', '').replace('}', '')}")
        
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=200)
        print(f"Saved: {args.out_png}")
    except Exception as exc:
        print(f"Plotting failed (matplotlib missing?): {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 