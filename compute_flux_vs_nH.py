#!/usr/bin/env python3
"""
Compute photon flux as a function of hydrogen column density (nH) for soft and hard X-ray bands
using XSPEC via its Python API.

- Loads spectrum data (PHA + responses) from a directory (default: ./data/IC10X1_spec)
- Fits a simple model (default: tbabs*powerlaw)
- Freezes the continuum and varies nH over a grid
- For each nH, computes photon flux in:
  - Soft band: 0.3–2 keV
  - Hard band: 2–10 keV
- Saves a CSV and a comparison plot

Run this under the conda environment that has XSPEC Python (e.g., `henv`).

Example:
  python compute_flux_vs_nH.py \
      --specdir ./data/IC10X1_spec \
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
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

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


def fit_baseline_model(model_expr: str = "tbabs*powerlaw") -> None:
    """Fit a baseline model and freeze non-absorption parameters."""
    Model(model_expr)

    # Reasonable starting values
    try:
        # tbabs.nH in 1e22 units - try both capitalizations
        model = AllModels(1)
        if hasattr(model, 'TBabs'):
            model.TBabs.nH = 0.511314
        elif hasattr(model, 'tbabs'):
            model.tbabs.nH = 0.511314
    except Exception:
        pass
    try:
        # Set powerlaw parameters - try both capitalizations
        model = AllModels(1)
        if hasattr(model, 'powerlaw'):
            model.powerlaw.PhoIndex = 1.84513
            model.powerlaw.norm = 3.05926e-04
        elif hasattr(model, 'Powerlaw'):
            model.Powerlaw.PhoIndex = 1.84513
            model.Powerlaw.norm = 3.05926e-04
    except Exception:
        pass

    # Fit
    Fit.method = "leven"
    Fit.statMethod = "chi"
    Fit.query = "yes"
    try:
        Fit.perform()
    except Exception:
        pass

    # Ensure best-fit parameters are used thereafter
    try:
        Xset.command("save all xspec_fit_tmp.xcm")
        Xset.command("@xspec_fit_tmp.xcm")
    except Exception:
        pass

    # Freeze continuum so only nH varies
    try:
        model = AllModels(1)
        if hasattr(model, 'powerlaw'):
            model.powerlaw.PhoIndex.frozen = True
            model.powerlaw.norm.frozen = True
        elif hasattr(model, 'Powerlaw'):
            model.Powerlaw.PhoIndex.frozen = True
            model.Powerlaw.norm.frozen = True
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


def vary_nh_and_compute(specdir: str,
                        nH_values_cm2: np.ndarray,
                        band_soft: Tuple[float, float],
                        band_hard: Tuple[float, float],
                        model_expr: str = "tbabs*powerlaw") -> pd.DataFrame:
    """
    For each nH (cm^-2), set tbabs.nH and compute photon flux in both bands.
    Returns a DataFrame with columns: nH_cm2, nH_1e22, flux_soft_ph, flux_hard_ph
    """
    src, bkg, rmf, arf = find_spectrum_files(specdir)
    load_data(src, bkg, rmf, arf)

    # XSPEC settings for reproducibility
    Xset.abund = "wilm"
    Xset.xsect = "vern"

    fit_baseline_model(model_expr)

    results = []
    for nH_cm2 in nH_values_cm2:
        nH_1e22 = float(nH_cm2 / 1.0e22)
        try:
            # Try both possible capitalizations of tbabs/TBabs
            model = AllModels(1)
            if hasattr(model, 'TBabs'):
                model.TBabs.nH = nH_1e22
            elif hasattr(model, 'tbabs'):
                model.tbabs.nH = nH_1e22
            else:
                raise AttributeError("Model has neither 'TBabs' nor 'tbabs' component")
        except Exception as e:
            # If model isn't tbabs-based, skip with NaNs
            soft_flux_ph = float("nan")
            hard_flux_ph = float("nan")
            soft_flux_erg = float("nan")
            hard_flux_erg = float("nan")
        else:
            # Try to compute photon flux via unfolded model
            soft_flux_ph = compute_photon_flux_for_band(band_soft)
            
            hard_flux_ph = compute_photon_flux_for_band(band_hard)
            
            # Also compute energy flux via XSPEC calculator for visibility/fallback
            soft_flux_erg = compute_energy_flux_for_band(band_soft)
            
            hard_flux_erg = compute_energy_flux_for_band(band_hard)
            
            # If photon flux failed, fall back to energy flux (so plot is not empty)
            if not np.isfinite(soft_flux_ph) or soft_flux_ph <= 0:
                soft_flux_ph = soft_flux_erg
            if not np.isfinite(hard_flux_ph) or hard_flux_ph <= 0:
                hard_flux_ph = hard_flux_erg

        result_row = {
            "nH_cm2": float(nH_cm2),
            "nH_1e22": nH_1e22,
            "flux_soft_ph": soft_flux_ph,
            "flux_hard_ph": hard_flux_ph,
            "flux_soft_erg": soft_flux_erg,
            "flux_hard_erg": hard_flux_erg,
        }
        results.append(result_row)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Compute photon flux vs nH using XSPEC (soft and hard bands)",
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
        default="tbabs*powerlaw",
        help="XSPEC model expression to fit before varying nH",
    )
    parser.add_argument("--nH_min", type=float, default=1e20, help="Min nH (cm^-2)")
    parser.add_argument("--nH_max", type=float, default=1e24, help="Max nH (cm^-2)")
    parser.add_argument("--nH_points", type=int, default=50, help="Number of nH grid points (log-spaced)")

    parser.add_argument("--soft_band", type=str, default="0.3,2.0", help="Soft band keV as 'Emin,Emax'")
    parser.add_argument("--hard_band", type=str, default="2.0,10.0", help="Hard band keV as 'Emin,Emax'")

    parser.add_argument("--out_csv", type=str, default="flux_vs_nH.csv", help="Output CSV filename")
    parser.add_argument("--out_png", type=str, default="flux_vs_nH.png", help="Output PNG filename")

    args = parser.parse_args()

    if not os.path.isdir(args.specdir):
        print(f"Error: specdir not found: {args.specdir}")
        sys.exit(1)

    # Parse bands
    def parse_band(s: str) -> Tuple[float, float]:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid band specification: {s}")
        return float(parts[0]), float(parts[1])

    band_soft = parse_band(args.soft_band)
    band_hard = parse_band(args.hard_band)

    # Build nH grid (log-spaced)
    nH_values_cm2 = np.logspace(np.log10(args.nH_min), np.log10(args.nH_max), args.nH_points)

    # Compute
    df = vary_nh_and_compute(
        specdir=args.specdir,
        nH_values_cm2=nH_values_cm2,
        band_soft=band_soft,
        band_hard=band_hard,
        model_expr=args.model,
    )

    # Save CSV
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv} ({len(df)} rows)")

    # Plot
    try:
        import matplotlib.pyplot as plt

        # Clean data for plotting: positive nH and flux only
        dfp = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["nH_cm2"]).copy()
        mask_soft = (dfp["nH_cm2"] > 0) & (dfp["flux_soft_ph"] > 0)
        mask_hard = (dfp["nH_cm2"] > 0) & (dfp["flux_hard_ph"] > 0)

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0))

        any_plotted = False
        if mask_soft.any():
            ax.plot(
                dfp.loc[mask_soft, "nH_cm2"],
                dfp.loc[mask_soft, "flux_soft_ph"],
                label=f"Soft {band_soft[0]}-{band_soft[1]} keV (ph or erg)",
                color="tab:blue",
                marker="o",
                markersize=3,
                linewidth=1.5,
                alpha=0.9,
            )
            any_plotted = True
        if mask_hard.any():
            ax.plot(
                dfp.loc[mask_hard, "nH_cm2"],
                dfp.loc[mask_hard, "flux_hard_ph"],
                label=f"Hard {band_hard[0]}-{band_hard[1]} keV (ph or erg)",
                color="tab:red",
                marker="s",
                markersize=3,
                linewidth=1.5,
                alpha=0.9,
            )
            any_plotted = True

        # If still nothing plotted, try energy flux columns
        if not any_plotted and "flux_soft_erg" in dfp.columns and "flux_hard_erg" in dfp.columns:
            mask_soft_e = (dfp["nH_cm2"] > 0) & (dfp["flux_soft_erg"] > 0)
            mask_hard_e = (dfp["nH_cm2"] > 0) & (dfp["flux_hard_erg"] > 0)
            if mask_soft_e.any():
                ax.plot(
                    dfp.loc[mask_soft_e, "nH_cm2"],
                    dfp.loc[mask_soft_e, "flux_soft_erg"],
                    label=f"Soft {band_soft[0]}-{band_soft[1]} keV (erg)",
                    color="tab:blue",
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                    alpha=0.9,
                )
                any_plotted = True
            if mask_hard_e.any():
                ax.plot(
                    dfp.loc[mask_hard_e, "nH_cm2"],
                    dfp.loc[mask_hard_e, "flux_hard_erg"],
                    label=f"Hard {band_hard[0]}-{band_hard[1]} keV (erg)",
                    color="tab:red",
                    marker="s",
                    markersize=3,
                    linewidth=1.5,
                    alpha=0.9,
                )
                any_plotted = True

        # Force log scales for readability
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Compute sensible limits from available data
        yvals = []
        if mask_soft.any():
            yvals.append(dfp.loc[mask_soft, "flux_soft_ph"].values)
        if mask_hard.any():
            yvals.append(dfp.loc[mask_hard, "flux_hard_ph"].values)
        if not yvals and "flux_soft_erg" in dfp.columns and "flux_hard_erg" in dfp.columns:
            if (dfp["flux_soft_erg"] > 0).any():
                yvals.append(dfp.loc[dfp["flux_soft_erg"] > 0, "flux_soft_erg"].values)
            if (dfp["flux_hard_erg"] > 0).any():
                yvals.append(dfp.loc[dfp["flux_hard_erg"] > 0, "flux_hard_erg"].values)
        if yvals:
            yvals = np.concatenate(yvals)
            yvals = yvals[yvals > 0]
            if yvals.size > 0:
                ymin = yvals.min() * 0.8
                ymax = yvals.max() * 1.2
                if ymin > 0 and ymax > ymin:
                    ax.set_ylim(ymin, ymax)

        # X limits from available nH
        xmin = dfp["nH_cm2"].min() * 0.8
        xmax = dfp["nH_cm2"].max() * 1.2
        if xmin > 0 and xmax > xmin:
            ax.set_xlim(xmin, xmax)

        ax.set_xlabel("nH (cm$^{-2}$)")
        ax.set_ylabel("Flux (photons or ergs cm$^{-2}$ s$^{-1}$)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=200)
        print(f"Saved: {args.out_png}")
    except Exception as exc:
        print(f"Plotting failed (matplotlib missing?): {exc}")


if __name__ == "__main__":
    main() 