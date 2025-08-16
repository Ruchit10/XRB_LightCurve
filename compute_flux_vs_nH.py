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
        # tbabs.nH in 1e22 units
        AllModels(1).tbabs.nH = 0.511314
    except Exception:
        pass
    try:
        AllModels(1).powerlaw.PhoIndex = 1.84513
        AllModels(1).powerlaw.norm = 3.05926e-04
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
        AllModels(1).powerlaw.PhoIndex.frozen = True
        AllModels(1).powerlaw.norm.frozen = True
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


def compute_photon_flux_for_band(band: Tuple[float, float]) -> float:
    """
    Compute photon flux by integrating the unfolded model spectrum over the band.
    Returns photons/cm^2/s in the specified energy range.
    Tries XSPEC's native band flux calculator first, then falls back to plot integration.
    """
    e1, e2 = band

    # First try XSPEC's native calculation
    try:
        # Some PyXspec versions return a list/tuple; others may set internals.
        res = AllModels.calcFlux(f"{e1} {e2}")
        if res is None:
            # Try to access via Model object if available
            try:
                # Not standardized; keep as placeholder for older versions
                pass
            except Exception:
                pass
        else:
            # If res is scalar or sequence, extract first element as flux value
            if isinstance(res, (list, tuple, np.ndarray)):
                if len(res) > 0:
                    val = float(res[0])
                    if np.isfinite(val) and val >= 0:
                        return val
            else:
                val = float(res)
                if np.isfinite(val) and val >= 0:
                    return val
    except Exception:
        # Fall through to plot-based method
        pass

    # Configure plot to get unfolded spectrum in photons/cm^2/s/keV
    Xset.chatter = 5
    Plot.xAxis = "keV"
    try:
        Xset.command("setplot energy")
        Xset.command("setplot ufspec")
    except Exception:
        pass

    # Build model arrays
    Plot.device = "/null"
    try:
        # Ensure a well-sampled energy grid for integration
        Xset.command("energies 0.1 20.0 2000 log")
        # Request unfolded spectrum so model is in photons/cm^2/s/keV
        Plot("ufspec")
        E = np.array(Plot.x(1), dtype=float)
        # Prefer model array; if unavailable, fall back to y
        y = np.array(Plot.model(1), dtype=float) if len(Plot.model(1)) else np.array(Plot.y(1), dtype=float)
    except Exception:
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
            AllModels(1).tbabs.nH = nH_1e22
        except Exception:
            # If model isn't tbabs-based, skip with NaNs
            soft_flux = float("nan")
            hard_flux = float("nan")
        else:
            soft_flux = compute_photon_flux_for_band(band_soft)
            hard_flux = compute_photon_flux_for_band(band_hard)

        results.append(
            {
                "nH_cm2": float(nH_cm2),
                "nH_1e22": nH_1e22,
                "flux_soft_ph": soft_flux,
                "flux_hard_ph": hard_flux,
            }
        )

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
                label=f"Soft {band_soft[0]}-{band_soft[1]} keV",
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
                label=f"Hard {band_hard[0]}-{band_hard[1]} keV",
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
        if yvals:
            yvals = np.concatenate(yvals)
            yvals = yvals[yvals > 0]
            if yvals.size > 0:
                ymin = yvals.min() * 0.8
                ymax = yvals.max() * 1.2
                if ymin > 0 and ymax > ymin:
                    ax.set_ylim(ymin, ymax)

        # X limits from available nH
        if any_plotted:
            xmin = dfp["nH_cm2"].min() * 0.8
            xmax = dfp["nH_cm2"].max() * 1.2
            if xmin > 0 and xmax > xmin:
                ax.set_xlim(xmin, xmax)

        ax.set_xlabel("nH (cm$^{-2}$)")
        ax.set_ylabel("Photon flux (photons cm$^{-2}$ s$^{-1}$)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=200)
        print(f"Saved: {args.out_png}")
    except Exception as exc:
        print(f"Plotting failed (matplotlib missing?): {exc}")


if __name__ == "__main__":
    main() 