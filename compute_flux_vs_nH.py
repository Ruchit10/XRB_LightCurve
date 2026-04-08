#!/usr/bin/env python3
"""
Compute photon flux as a function of hydrogen column density (nH) for X-ray energy bands
using XSPEC via its Python API.

Simplified version supporting three absorption models:
- phabs*powerlaw (photoelectric absorption, Morrison & McCammon 1983)
- tbabs*powerlaw (Tuebingen-Boulder ISM absorption, Wilms et al. 2000)
- wabs*powerlaw (Wisconsin absorption, Morrison & McCammon 1983)

Workflow:
1. Load spectrum data (PHA + responses) from directory
2. Fit specified absorption*powerlaw model to data
3. Log best-fit parameters
4. Freeze powerlaw parameters (PhoIndex, norm)
5. Vary nH over a grid and compute photon flux in each energy band
6. Save CSV and comparison plot

Supported Instruments and Energy Bands:
  Chandra (default):
    - broad: 0.5–7.0 keV
    - soft: 0.5–2.0 keV
    - medium: 1.2–2.0 keV
    - hard: 2.0–7.0 keV

Run this under a conda environment with XSPEC Python (e.g., `henv`).

Example:
  python compute_flux_vs_nH.py \\
      --specdir ./data/IC10X1_spec \\
      --model tbabs \\
      --statistic chi \\
      --fit_emin 0.5 --fit_emax 7.0 \\
      --bands broad soft medium hard \\
      --out_csv flux_vs_nH.csv \\
      --out_png flux_vs_nH.png \\
      --nH_min 1e20 --nH_max 1e24 --nH_points 60

Notes:
- XSPEC absorption nH parameter is in 1e22 cm^-2
- This script accepts nH in cm^-2 and converts accordingly
- Photon flux is computed by integrating the unfolded model spectrum (photons/cm^2/s/keV)
- Fitting range (--fit_emin, --fit_emax) determines which channels are used for parameter fitting
- Flux computation uses band-specific ranges (e.g., soft: 0.5-2.0, hard: 2.0-7.0)
- These can differ: you might fit over 0.5-7.0 keV but compute flux in separate soft/hard bands
"""

import argparse
import glob
import os
import sys
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
INSTRUMENT_BANDS = {
    "chandra": {
        "broad": (0.5, 7.0),
        "soft": (0.5, 2.0),
        "medium": (1.2, 2.0),
        "hard": (2.0, 7.0),
    },
}

DEFAULT_BANDS = {
    "chandra": ["broad"],
}

# Supported absorption models
ABSORPTION_MODELS = ["phabs", "tbabs", "wabs"]


def find_spectrum_files(specdir: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Locate source PHA/PI, optional background, and responses in specdir.
    Returns (src_pha_or_pi, bkg_pha_or_pi, rmf, arf)
    """
    pha_patterns = ["*.pha", "*.pha.gz", "*.pi", "*.pi.gz"]
    all_spec = []
    for pat in pha_patterns:
        all_spec.extend(sorted(glob.glob(os.path.join(specdir, pat))))

    if not all_spec:
        raise FileNotFoundError(f"No PHA/PI files found in {specdir}")

    # Prefer files with 'src' or 'source' in name
    src = None
    for spec in all_spec:
        if any(keyword in spec.lower() for keyword in ["_src", "source", "src"]):
            src = spec
            break
    if src is None:
        src = all_spec[0]

    # Find background
    remaining = [p for p in all_spec if p != src]
    bkg = None
    for spec in remaining:
        if any(keyword in spec.lower() for keyword in ["_bkg", "background", "bkg"]):
            bkg = spec
            break

    # Find responses
    rmf_list = sorted(glob.glob(os.path.join(specdir, "*.rmf")))
    arf_list = sorted(glob.glob(os.path.join(specdir, "*.arf")))

    rmf = None
    for r in rmf_list:
        if any(keyword in r.lower() for keyword in ["_src", "source", "src"]):
            rmf = r
            break
    if rmf is None and rmf_list:
        rmf = rmf_list[0]

    arf = None
    for a in arf_list:
        if any(keyword in a.lower() for keyword in ["_src", "source", "src"]):
            arf = a
            break
    if arf is None and arf_list:
        arf = arf_list[0]

    return src, bkg, rmf, arf


def load_xspec_spectrum(pha_or_pi: str, bkg: Optional[str], rmf: Optional[str], arf: Optional[str]) -> None:
    """Load spectrum into XSPEC and attach background."""
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
        except Exception:
            pass
    finally:
        os.chdir(cwd)


def fit_model(
    model_name: str, 
    statistic: str = "chi",
    init_nH: float = 0.55,
    init_PhoIndex: float = 1.89,
    init_norm: float = 1e-4,
    fit_emin: float = 0.5,
    fit_emax: float = 7.0,
) -> Dict[str, float]:
    """
    Fit absorption*powerlaw model to loaded spectrum.
    
    Args:
        model_name: Absorption model name (phabs, tbabs, or wabs)
        statistic: Fit statistic ('chi' or 'cstat')
        init_nH: Initial nH value in 1e22 cm^-2 (default: 0.55)
        init_PhoIndex: Initial powerlaw photon index (default: 1.89)
        init_norm: Initial powerlaw normalization (default: 1e-4)
        fit_emin: Minimum energy for fitting in keV (default: 0.5)
        fit_emax: Maximum energy for fitting in keV (default: 7.0)
        
    Returns:
        Dictionary of best-fit parameters
    """
    if model_name not in ABSORPTION_MODELS:
        raise ValueError(f"Model must be one of {ABSORPTION_MODELS}, got: {model_name}")

    # Set abundance and cross-section tables
    Xset.abund = "wilm"
    Xset.xsect = "vern"
    
    # Set fit statistic
    if statistic.lower() == "chi":
        Fit.statMethod = "chi"
    elif statistic.lower() == "cstat":
        Fit.statMethod = "cstat"
    else:
        raise ValueError(f"Statistic must be 'chi' or 'cstat', got: {statistic}")

    # Create model
    model_expr = f"{model_name}*powerlaw"
    Model(model_expr)

    # Set energy range for fitting
    try:
        # First ignore all channels
        AllData.ignore("**-**")
        # Then notice only the specified energy range
        AllData.notice(f"{fit_emin}-{fit_emax}")
        print(f"Fitting energy range: {fit_emin}-{fit_emax} keV")
    except Exception as e:
        print(f"Warning: Could not set energy range: {e}")
        # Fallback to notice all
        try:
            AllData.notice("all")
        except:
            pass

    model = AllModels(1)

    # Set initial parameter values
    # Access components by parameter index (more robust than component names)
    # absorption*powerlaw typically has parameters:
    # 1: absorption.nH
    # 2: powerlaw.PhoIndex
    # 3: powerlaw.norm
    
    try:
        # Try to set parameters by index (most reliable method)
        # XSPEC parameters store values as tuples/lists, so we need to preserve that
        par1 = model(1)
        if isinstance(par1.values, (tuple, list)):
            vals = list(par1.values)
            vals[0] = init_nH
            par1.values = vals if isinstance(par1.values, list) else tuple(vals)
        else:
            par1.values = init_nH
        
        par2 = model(2)
        if isinstance(par2.values, (tuple, list)):
            vals = list(par2.values)
            vals[0] = init_PhoIndex
            par2.values = vals if isinstance(par2.values, list) else tuple(vals)
        else:
            par2.values = init_PhoIndex
        
        par3 = model(3)
        if isinstance(par3.values, (tuple, list)):
            vals = list(par3.values)
            vals[0] = init_norm
            par3.values = vals if isinstance(par3.values, list) else tuple(vals)
        else:
            par3.values = init_norm
        
        print(f"Initial parameters: nH={init_nH:.4f} × 10²² cm⁻², PhoIndex={init_PhoIndex:.4f}, norm={init_norm:.4e}")
        
    except Exception as e:
        # Fallback: try component name access
        print(f"Warning: Could not set parameters by index, trying component names: {e}")
        
        # Try various capitalizations for absorption component
        abs_comp = None
        for name_variant in [model_name, model_name.upper(), model_name.capitalize(), 
                             model_name.lower(), f"TBabs" if "tb" in model_name.lower() else None]:
            if name_variant and hasattr(model, name_variant):
                abs_comp = getattr(model, name_variant)
                break
        
        if abs_comp is None:
            raise AttributeError(f"Cannot access {model_name} component. Available: {dir(model)}")
        
        # Set starting nH
        if hasattr(abs_comp, "nH"):
            abs_comp.nH = init_nH
        elif hasattr(abs_comp, "NH"):
            abs_comp.NH = init_nH
        else:
            raise AttributeError(f"Cannot find nH parameter in {model_name} component")

        # Access powerlaw component
        po_comp = None
        for name_variant in ["powerlaw", "Powerlaw", "POWERLAW", "po"]:
            if hasattr(model, name_variant):
                po_comp = getattr(model, name_variant)
                break
        
        if po_comp is None:
            raise AttributeError("Cannot access powerlaw component")

        # Set starting powerlaw parameters
        po_comp.PhoIndex = init_PhoIndex
        po_comp.norm = init_norm
        
        print(f"Initial parameters: nH={init_nH:.4f} × 10²² cm⁻², PhoIndex={init_PhoIndex:.4f}, norm={init_norm:.4e}")

    # Perform fit
    print(f"\nFitting {model_expr} with {statistic} statistic...")
    Fit.method = "leven"
    Fit.query = "yes"
    
    try:
        Fit.perform()
    except Exception as e:
        print(f"Warning: Fit may not have converged: {e}")

    # Extract best-fit parameters using parameter indices (most reliable)
    params = {}
    
    try:
        # Parameter 1: nH
        nh_par = model(1)
        params["nH"] = float(nh_par.values[0] if isinstance(nh_par.values, (tuple, list)) else nh_par.values)
        try:
            err = nh_par.error
            params["nH_error"] = float(err[0] if isinstance(err, (tuple, list)) else err)
        except (AttributeError, IndexError, TypeError):
            params["nH_error"] = 0.0
        
        # Parameter 2: PhoIndex
        phoindex_par = model(2)
        params["PhoIndex"] = float(phoindex_par.values[0] if isinstance(phoindex_par.values, (tuple, list)) else phoindex_par.values)
        try:
            err = phoindex_par.error
            params["PhoIndex_error"] = float(err[0] if isinstance(err, (tuple, list)) else err)
        except (AttributeError, IndexError, TypeError):
            params["PhoIndex_error"] = 0.0
        
        # Parameter 3: norm
        norm_par = model(3)
        params["norm"] = float(norm_par.values[0] if isinstance(norm_par.values, (tuple, list)) else norm_par.values)
        try:
            err = norm_par.error
            params["norm_error"] = float(err[0] if isinstance(err, (tuple, list)) else err)
        except (AttributeError, IndexError, TypeError):
            params["norm_error"] = 0.0
        
    except Exception as e:
        print(f"Warning: Could not extract parameters by index, trying component names: {e}")
        
        # Fallback: try component access
        # Find absorption component
        abs_comp = None
        for name_variant in [model_name, model_name.upper(), model_name.capitalize(), 
                             model_name.lower(), f"TBabs" if "tb" in model_name.lower() else None]:
            if name_variant and hasattr(model, name_variant):
                abs_comp = getattr(model, name_variant)
                break
        
        if abs_comp is None:
            raise AttributeError(f"Cannot access {model_name} component for parameter extraction")
        
        # Get nH
        if hasattr(abs_comp, "nH"):
            nh_par = abs_comp.nH
        else:
            nh_par = abs_comp.NH
        params["nH"] = float(nh_par.values[0] if isinstance(nh_par.values, (tuple, list)) else nh_par.values)
        try:
            err = nh_par.error
            params["nH_error"] = float(err[0] if isinstance(err, (tuple, list)) else err)
        except (AttributeError, IndexError, TypeError):
            params["nH_error"] = 0.0
        
        # Find powerlaw component
        po_comp = None
        for name_variant in ["powerlaw", "Powerlaw", "POWERLAW", "po"]:
            if hasattr(model, name_variant):
                po_comp = getattr(model, name_variant)
                break
        
        if po_comp is None:
            raise AttributeError("Cannot access powerlaw component for parameter extraction")
        
        # Get powerlaw parameters
        params["PhoIndex"] = float(po_comp.PhoIndex.values[0] if isinstance(po_comp.PhoIndex.values, (tuple, list)) else po_comp.PhoIndex.values)
        try:
            err = po_comp.PhoIndex.error
            params["PhoIndex_error"] = float(err[0] if isinstance(err, (tuple, list)) else err)
        except (AttributeError, IndexError, TypeError):
            params["PhoIndex_error"] = 0.0
            
        params["norm"] = float(po_comp.norm.values[0] if isinstance(po_comp.norm.values, (tuple, list)) else po_comp.norm.values)
        try:
            err = po_comp.norm.error
            params["norm_error"] = float(err[0] if isinstance(err, (tuple, list)) else err)
        except (AttributeError, IndexError, TypeError):
            params["norm_error"] = 0.0
    
    # Get fit statistics
    params["statistic"] = Fit.statistic
    params["dof"] = Fit.dof
    params["chi2_red"] = Fit.statistic / Fit.dof if Fit.dof > 0 else 0.0
    
    # Log results
    print(f"\nBest-fit parameters:")
    print(f"  {model_name}.nH = {params['nH']:.6f} ± {params['nH_error']:.6f} × 10²² cm⁻²")
    print(f"  powerlaw.PhoIndex = {params['PhoIndex']:.6f} ± {params['PhoIndex_error']:.6f}")
    print(f"  powerlaw.norm = {params['norm']:.6e} ± {params['norm_error']:.6e}")
    print(f"  Fit statistic = {params['statistic']:.2f} for {params['dof']} dof")
    if statistic.lower() == "chi":
        print(f"  Reduced χ² = {params['chi2_red']:.4f}")
    
    return params


def freeze_powerlaw_params() -> object:
    """
    Freeze powerlaw parameters (PhoIndex, norm) and return the nH parameter object.
    
    Uses parameter indices for robustness:
    - Parameter 1: absorption.nH (keep free)
    - Parameter 2: powerlaw.PhoIndex (freeze)
    - Parameter 3: powerlaw.norm (freeze)
    
    Returns:
        The nH parameter object that will be varied
    """
    model = AllModels(1)
    
    try:
        # Freeze by parameter index (most reliable)
        model(2).frozen = True  # PhoIndex
        model(3).frozen = True  # norm
        
        print("\nFroze powerlaw parameters (PhoIndex, norm) using parameter indices")
        
        # Return nH parameter
        return model(1)
        
    except Exception as e:
        print(f"Warning: Could not freeze by index, trying component names: {e}")
        
        # Fallback: try component name access
        po_comp = None
        for name_variant in ["powerlaw", "Powerlaw", "POWERLAW", "po"]:
            if hasattr(model, name_variant):
                po_comp = getattr(model, name_variant)
                break
        
        if po_comp is None:
            raise AttributeError("Cannot access powerlaw component")
        
        po_comp.PhoIndex.frozen = True
        po_comp.norm.frozen = True
        
        print("\nFroze powerlaw parameters (PhoIndex, norm)")
        
        # Find and return nH parameter
        model_expr = str(model.expression).lower()
        for abs_name in ABSORPTION_MODELS:
            if abs_name in model_expr:
                abs_comp = None
                for name_variant in [abs_name, abs_name.upper(), abs_name.capitalize(), 
                                   abs_name.lower(), f"TBabs" if "tb" in abs_name.lower() else None]:
                    if name_variant and hasattr(model, name_variant):
                        abs_comp = getattr(model, name_variant)
                        break
                
                if abs_comp:
                    if hasattr(abs_comp, "nH"):
                        return abs_comp.nH
                    elif hasattr(abs_comp, "NH"):
                        return abs_comp.NH
        
        raise RuntimeError("Could not find nH parameter in model")


_KEV_TO_ERG = 1.60218e-9  # 1 keV in erg


def integrate_fluxes(
    E: np.ndarray, y: np.ndarray, band: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Integrate model spectrum y(E) [photons/cm^2/s/keV] over band [E1, E2] keV.

    Returns:
        photon_flux : photons/cm^2/s  = trapz(y, E)
        energy_flux : erg/cm^2/s     = trapz(E * y, E) * keV_to_erg
    """
    e1, e2 = band
    mask = (E >= e1) & (E <= e2)
    if np.count_nonzero(mask) < 2:
        # Pull in nearest bins when band edges fall between grid points
        idx = np.argsort(np.abs(E - np.clip((e1 + e2) / 2.0, E.min(), E.max())))[:3]
        mask[idx] = True
    if np.count_nonzero(mask) < 2:
        return float("nan"), float("nan")
    E_b, y_b = E[mask], y[mask]
    ph = float(np.trapz(y_b, E_b))
    erg = float(np.trapz(E_b * y_b, E_b)) * _KEV_TO_ERG
    return ph, erg


def compute_fluxes_for_band(band: Tuple[float, float]) -> Tuple[float, float]:
    """
    Evaluate the current XSPEC model on a fine log-spaced energy grid and
    integrate over the requested band to get photon and energy flux.

    The model is sampled directly (no unfolding / instrument response needed)
    because we want the intrinsic model flux, not a data-derived quantity.

    Args:
        band: (E_min, E_max) in keV

    Returns:
        photon_flux : photons/cm^2/s
        energy_flux : erg/cm^2/s
    """
    e1, e2 = band
    if e1 >= e2:
        print(f"Error: Invalid band {e1}-{e2} keV (min >= max)")
        return float("nan"), float("nan")

    Plot.xAxis = "keV"
    Plot.device = "/null"

    try:
        AllModels.setEnergies("0.1 20.0 2000 log")
        Plot("model")
        E = np.array(Plot.x(1), dtype=float)
        y = np.array(Plot.model(1), dtype=float)
    except Exception as e:
        import traceback
        print(f"[compute_fluxes_for_band] ERROR for band {band}: {e}")
        traceback.print_exc()
        return float("nan"), float("nan")

    if E.size < 2 or y.size < 2:
        return float("nan"), float("nan")

    return integrate_fluxes(E, y, band)


def vary_nh_and_compute(
    nH_values_cm2: np.ndarray,
    bands: dict,
) -> pd.DataFrame:
    """
    For each nH (cm^-2), set absorption.nH and compute photon and energy flux
    in each requested band.

    Assumes model has already been fit and powerlaw parameters frozen.

    Args:
        nH_values_cm2: Array of nH values in cm^-2
        bands: Dictionary mapping band names to (E_min, E_max) tuples in keV

    Returns:
        DataFrame with columns:
            nH_cm2, nH_1e22,
            flux_{band}_ph  [photons/cm^2/s],
            flux_{band}_erg [erg/cm^2/s]
        for each band.
    """
    nh_par = freeze_powerlaw_params()

    results = []
    print(f"\nComputing flux for {len(nH_values_cm2)} nH values...")

    for i, nH_cm2 in enumerate(nH_values_cm2):
        nH_1e22 = float(nH_cm2 / 1.0e22)
        result_row = {
            "nH_cm2": float(nH_cm2),
            "nH_1e22": nH_1e22,
        }

        try:
            # Update nH parameter
            if isinstance(nh_par.values, (tuple, list)):
                vals = list(nh_par.values)
                vals[0] = nH_1e22
                nh_par.values = vals if isinstance(nh_par.values, list) else tuple(vals)
            else:
                nh_par.values = nH_1e22

            # Compute photon and energy flux for each band
            for band_name, band_range in bands.items():
                flux_ph, flux_erg = compute_fluxes_for_band(band_range)
                result_row[f"flux_{band_name}_ph"] = flux_ph
                result_row[f"flux_{band_name}_erg"] = flux_erg

        except Exception as e:
            import traceback
            print(f"[vary_nh_and_compute] ERROR at nH={nH_cm2:.3e} cm⁻² (nH_1e22={nH_1e22:.4f}): {e}")
            traceback.print_exc()
            for band_name in bands.keys():
                result_row[f"flux_{band_name}_ph"] = float("nan")
                result_row[f"flux_{band_name}_erg"] = float("nan")

        # Progress indicator
        if (i + 1) % max(1, len(nH_values_cm2) // 10) == 0:
            print(f"  Progress: {i+1}/{len(nH_values_cm2)}")

        results.append(result_row)

    return pd.DataFrame(results)


def fit_exponential_decay(nH_cm2: np.ndarray, flux: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Fit exponential decay function to flux vs nH data in LOG SPACE.
    
    Formula: flux = A * exp(-B * nH_1e22)
    where nH_1e22 = nH_cm2 / 1e22
    
    Taking log: log(flux) = log(A) - B * nH_1e22
    
    Args:
        nH_cm2: Array of nH values in cm^-2
        flux: Array of flux values (photons/cm^2/s)
    
    Returns:
        A: Fitted coefficient A
        B: Fitted coefficient B (in units of 1e22 cm^-2)
        flux_fit: Fitted flux values for the input nH_cm2 array
    """
    # Filter out invalid data points
    mask = (nH_cm2 > 0) & (flux > 0) & np.isfinite(nH_cm2) & np.isfinite(flux)
    
    if np.sum(mask) < 3:
        return float('nan'), float('nan'), np.full_like(nH_cm2, np.nan)
    
    nH_valid = nH_cm2[mask]
    flux_valid = flux[mask]
    
    # Convert nH to 1e22 units
    nH_1e22 = nH_valid / 1e22
    
    # Take logarithm for fitting in log space
    log_flux = np.log(flux_valid)
    
    # Define linear function for log-space fitting
    def linear_func(x, log_A, B):
        return log_A - B * x
    
    try:
        # Fit in log space
        popt, _ = curve_fit(
            linear_func,
            nH_1e22,
            log_flux,
            p0=[np.log(flux_valid[0]), 0.1],
            maxfev=10000
        )
        log_A, B = popt
        A = np.exp(log_A)
        
        # Compute fitted flux for all input nH values
        nH_all_1e22 = nH_cm2 / 1e22
        flux_fit = A * np.exp(-B * nH_all_1e22)
        
        return float(A), float(B), flux_fit
        
    except Exception as e:
        return float('nan'), float('nan'), np.full_like(nH_cm2, np.nan)


def main():
    parser = argparse.ArgumentParser(
        description="Compute photon flux vs nH using XSPEC (simplified: phabs/tbabs/wabs * powerlaw)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--specdir",
        type=str,
        default=os.path.join(os.getcwd(), "data", "IC10X1_spec"),
        help="Directory containing PHA/PI (+RMF/ARF) spectrum files",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=ABSORPTION_MODELS,
        default="tbabs",
        help="Absorption model: phabs, tbabs, or wabs (will be multiplied by powerlaw)",
    )
    parser.add_argument(
        "--statistic",
        type=str,
        choices=["chi", "cstat"],
        default="chi",
        help="Fit statistic: chi (chi-squared) or cstat (C-statistic for Poisson data)",
    )
    parser.add_argument(
        "--init_nH",
        type=float,
        default=0.55,
        help="Initial nH value for fitting in 10^22 cm^-2 (helps convergence)",
    )
    parser.add_argument(
        "--init_PhoIndex",
        type=float,
        default=1.89,
        help="Initial powerlaw photon index for fitting (helps convergence)",
    )
    parser.add_argument(
        "--init_norm",
        type=float,
        default=1e-4,
        help="Initial powerlaw normalization for fitting (helps convergence)",
    )
    parser.add_argument(
        "--fit_emin",
        type=float,
        default=0.5,
        help="Minimum energy for fitting in keV (default: 0.5)",
    )
    parser.add_argument(
        "--fit_emax",
        type=float,
        default=7.0,
        help="Maximum energy for fitting in keV (default: 7.0)",
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
    parser.add_argument("--nH_min", type=float, default=1e20, help="Min nH (cm^-2) for flux grid")
    parser.add_argument("--nH_max", type=float, default=1e24, help="Max nH (cm^-2) for flux grid")
    parser.add_argument("--nH_points", type=int, default=60, help="Number of nH grid points (log-spaced)")
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
        band_names = DEFAULT_BANDS.get(args.instrument, list(instrument_bands.keys()))
    else:
        band_names = args.bands
        invalid_bands = [b for b in band_names if b not in instrument_bands]
        if invalid_bands:
            print(f"Error: Invalid bands for {args.instrument}: {invalid_bands}")
            print(f"Available bands: {list(instrument_bands.keys())}")
            sys.exit(1)
    
    bands = {name: instrument_bands[name] for name in band_names}
    
    print(f"Computing flux vs nH for {args.instrument} instrument")
    print(f"Model: {args.model}*powerlaw")
    print(f"Statistic: {args.statistic}")
    print(f"Fitting energy range: {args.fit_emin}-{args.fit_emax} keV")
    print(f"\nEnergy bands for flux computation:")
    for name, (emin, emax) in bands.items():
        print(f"  {name}: {emin}-{emax} keV")
    
    # Verify that bands are within reasonable range
    for name, (emin, emax) in bands.items():
        if emin < args.fit_emin or emax > args.fit_emax:
            print(f"\nNote: Band '{name}' ({emin}-{emax} keV) extends beyond fitting range ({args.fit_emin}-{args.fit_emax} keV)")
            print(f"      Flux will still be computed over full band range using extrapolated model.")

    # Find and load spectrum files
    print(f"\nLoading spectrum from {args.specdir}...")
    src, bkg, rmf, arf = find_spectrum_files(args.specdir)
    print(f"  Source: {os.path.basename(src)}")
    if bkg:
        print(f"  Background: {os.path.basename(bkg)}")
    if rmf:
        print(f"  RMF: {os.path.basename(rmf)}")
    if arf:
        print(f"  ARF: {os.path.basename(arf)}")
    
    load_xspec_spectrum(src, bkg, rmf, arf)

    # Fit model
    best_fit_params = fit_model(
        args.model, 
        args.statistic,
        init_nH=args.init_nH,
        init_PhoIndex=args.init_PhoIndex,
        init_norm=args.init_norm,
        fit_emin=args.fit_emin,
        fit_emax=args.fit_emax,
    )

    # Build nH grid (log-spaced)
    nH_values_cm2 = np.logspace(np.log10(args.nH_min), np.log10(args.nH_max), args.nH_points)

    # Compute flux vs nH
    df = vary_nh_and_compute(nH_values_cm2, bands)

    # Save CSV
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved: {args.out_csv} ({len(df)} rows)")

    # Plot
    try:
        import matplotlib.pyplot as plt

        dfp = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["nH_cm2"]).copy()

        colors = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
        markers = ["o", "s", "^", "D", "v", "p"]

        fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        ax_ph, ax_erg = axes

        # Shared title
        title = (
            f"Flux vs $n_H$ ({args.instrument.capitalize()})  —  "
            f"Model: {args.model}*powerlaw, Stat: {args.statistic}, "
        )
        title += (
            f"$\\chi^2_r$ = {best_fit_params['chi2_red']:.2f}"
            if args.statistic == "chi"
            else f"C-stat = {best_fit_params['statistic']:.2f}"
        )
        fig.suptitle(title, fontsize=12, fontweight="bold")

        fit_text_lines_ph = []
        fit_text_lines_erg = []
        any_ph = False
        any_erg = False

        for idx, (band_name, (emin, emax)) in enumerate(bands.items()):
            col_ph  = f"flux_{band_name}_ph"
            col_erg = f"flux_{band_name}_erg"
            color   = colors[idx % len(colors)]
            marker  = markers[idx % len(markers)]
            label   = f"{band_name.capitalize()} {emin}–{emax} keV"

            # --- photon flux panel ---
            if col_ph in dfp.columns:
                mask = (dfp["nH_cm2"] > 0) & (dfp[col_ph] > 0)
                if mask.any():
                    nH_d = dfp.loc[mask, "nH_cm2"].values
                    fl_d = dfp.loc[mask, col_ph].values
                    ax_ph.plot(nH_d, fl_d, color=color, marker=marker,
                               markersize=4, linewidth=0, alpha=0.7, label=label)
                    A, B, fl_fit = fit_exponential_decay(nH_d, fl_d)
                    if np.isfinite(A) and np.isfinite(B):
                        si = np.argsort(nH_d)
                        ax_ph.plot(nH_d[si], fl_fit[si], color=color,
                                   linestyle="--", linewidth=2, alpha=0.9,
                                   label=f"{band_name.capitalize()} fit")
                        fit_text_lines_ph.append(
                            f"{band_name.capitalize()}: $F = {A:.3e}\\,e^{{-{B:.4f}\\,n_H}}$"
                        )
                    any_ph = True

            # --- energy flux panel ---
            if col_erg in dfp.columns:
                mask = (dfp["nH_cm2"] > 0) & (dfp[col_erg] > 0)
                if mask.any():
                    nH_d = dfp.loc[mask, "nH_cm2"].values
                    fl_d = dfp.loc[mask, col_erg].values
                    ax_erg.plot(nH_d, fl_d, color=color, marker=marker,
                                markersize=4, linewidth=0, alpha=0.7, label=label)
                    A, B, fl_fit = fit_exponential_decay(nH_d, fl_d)
                    if np.isfinite(A) and np.isfinite(B):
                        si = np.argsort(nH_d)
                        ax_erg.plot(nH_d[si], fl_fit[si], color=color,
                                    linestyle="--", linewidth=2, alpha=0.9,
                                    label=f"{band_name.capitalize()} fit")
                        fit_text_lines_erg.append(
                            f"{band_name.capitalize()}: $F = {A:.3e}\\,e^{{-{B:.4f}\\,n_H}}$"
                        )
                    any_erg = True

        for ax in (ax_ph, ax_erg):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            if len(dfp) > 0:
                ax.set_xlim(dfp["nH_cm2"].min() * 0.8, dfp["nH_cm2"].max() * 1.2)

        ax_ph.set_ylabel("Photon flux (photons cm$^{-2}$ s$^{-1}$)", fontsize=12)
        ax_erg.set_ylabel("Energy flux (erg cm$^{-2}$ s$^{-1}$)", fontsize=12)
        ax_erg.set_xlabel("$n_H$ (cm$^{-2}$)", fontsize=12)

        if any_ph:
            ax_ph.legend(loc="upper right", fontsize=9, framealpha=0.9)
        if any_erg:
            ax_erg.legend(loc="upper right", fontsize=9, framealpha=0.9)

        # Best-fit parameter box on top panel
        param_text = (
            "Best-fit parameters:\n"
            f"$n_H$ = {best_fit_params['nH']:.4f} $\\times 10^{{22}}$ cm$^{{-2}}$\n"
            f"$\\Gamma$ = {best_fit_params['PhoIndex']:.4f}\n"
            f"norm = {best_fit_params['norm']:.3e}"
        )
        ax_ph.text(0.02, 0.98, param_text, transform=ax_ph.transAxes,
                   fontsize=9, verticalalignment="top",
                   bbox=dict(boxstyle="round", facecolor="lightblue",
                             alpha=0.8, edgecolor="black", linewidth=1.5))

        # Equation boxes
        for ax, lines in ((ax_ph, fit_text_lines_ph), (ax_erg, fit_text_lines_erg)):
            if lines:
                eq_text = "Fitted exponentials:\n" + "\n".join(lines)
                eq_text += "\n\n$n_H$ in units of $10^{22}$ cm$^{-2}$"
                ax.text(0.02, 0.02, eq_text, transform=ax.transAxes,
                        fontsize=8, verticalalignment="bottom",
                        bbox=dict(boxstyle="round", facecolor="wheat",
                                  alpha=0.8, edgecolor="black", linewidth=1.5))

        fig.tight_layout()
        fig.savefig(args.out_png, dpi=200)
        print(f"Saved: {args.out_png}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
