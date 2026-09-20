#!/usr/bin/env python3
"""
Tabulate the model flux of an absorbed power law against the hydrogen column
density nH, using XSPEC through PyXspec.

Workflow:
1. Load the spectrum (PHA/PI, plus background and responses when present) from
   --specdir.
2. Fit {phabs,tbabs,wabs}*powerlaw over --fit_emin/--fit_emax.
3. Freeze the power-law parameters (PhoIndex, norm).
4. Step nH over a log grid and integrate the model spectrum over the requested
   band: photon flux (photons/cm^2/s) and energy flux (erg/cm^2/s).
5. Save one CSV per band (the light-curve model is run one band at a time) and
   a diagnostic figure with the exponential law F = A exp(-B nH) that the
   simulator's `refit` flux method uses.

Chandra bands: broad 0.5-7.0, soft 0.5-2.0, medium 1.2-2.0, hard 2.0-7.0 keV.

Requires PyXspec (HEASoft): initialise HEASoft first, e.g. under the `henv`
conda environment. Everything else needs only numpy, pandas and matplotlib
(`utils.utils` is numpy/pandas only, so this script never imports numba).

Example:
  python compute_flux_vs_nH.py --specdir ./data/IC10X1_spec --model tbabs \\
      --band broad --out_csv flux_vs_nH_broad.csv --out_png flux_vs_nH_broad.png \\
      --nH_min 1e20 --nH_max 1e24 --nH_points 60

Notes:
- XSPEC's absorption nH is in 1e22 cm^-2; the grid here is given in cm^-2 and
  the CSV carries both (`nH_cm2`, `nH_1e22`).
- The band flux is XSPEC's own ``calcFlux`` of the fitted model (absorbed
  model flux, not a data-derived quantity), read back from the loaded spectrum.
- The fitting range and the flux band may differ (fit 0.5-7 keV, tabulate soft).
"""

import argparse
import glob
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils.utils import CHANDRA_BANDS, fit_exponential

# PyXspec is imported in main(), after argument parsing, so --help and argument
# errors work without an initialised HEASoft.
AllData = AllModels = Fit = Model = Xset = None


def _import_xspec() -> None:
    global AllData, AllModels, Fit, Model, Xset
    try:
        from xspec import AllData as _d, AllModels as _m, Fit as _f, Model as _mo, Xset as _x  # type: ignore
    except Exception as exc:
        print("Error: XSPEC Python module not available in this environment.")
        print("Initialise HEASoft (PyXspec) first: export HEADAS=<heasoft dir>; . $HEADAS/headas-init.sh")
        print(f"Details: {exc}")
        sys.exit(1)
    AllData, AllModels, Fit, Model, Xset = _d, _m, _f, _mo, _x


INSTRUMENT_BANDS = {"chandra": CHANDRA_BANDS}
ABSORPTION_MODELS = ("phabs", "tbabs", "wabs")



# ----------------------------------------------------------------------------
# Spectrum files
# ----------------------------------------------------------------------------

def _pick(paths, keywords) -> Optional[str]:
    """First path whose *file name* contains one of *keywords* (case-insensitive)."""
    for path in paths:
        name = os.path.basename(path).lower()
        if any(key in name for key in keywords):
            return path
    return None


def find_spectrum_files(specdir: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Locate the source PHA/PI, the background and the responses in *specdir*.

    Keywords are matched against file names only: matching the full path
    picked the background as the source whenever a directory was called
    ``src``. The background is identified first so a file such as
    ``acisf_src_bkg.pi`` cannot be taken for the source.

    Returns ``(src, bkg, rmf, arf)``; the last three may be None.
    """
    def files(*patterns):
        return sorted(p for pat in patterns for p in glob.glob(os.path.join(specdir, pat)))

    def source_first(paths):
        # Prefer a name marked as source; never fall back to a background file.
        not_bkg = [p for p in paths if _pick([p], ("bkg", "background")) is None]
        return _pick(not_bkg, ("src", "source")) or (not_bkg[0] if not_bkg else None)

    spectra = files("*.pha", "*.pha.gz", "*.pi", "*.pi.gz")
    if not spectra:
        raise FileNotFoundError(f"No PHA/PI files found in {specdir}")
    src = source_first(spectra)
    if src is None:
        raise FileNotFoundError(f"Only a background spectrum found in {specdir}")
    # The source's own background first (<stem>_bkg.*, as fakeit and CIAO name
    # it), then any background-named file.
    stem = os.path.basename(src).split(".")[0].lower().replace("_src", "").replace("src", "")
    own = [p for p in spectra if os.path.basename(p).lower().startswith(stem) and
           _pick([p], ("bkg", "background")) is not None]
    bkg = (own[0] if own else None) or _pick(spectra, ("bkg", "background"))
    rmf = source_first(files("*.rmf", "*.rmf.gz"))
    arf = source_first(files("*.arf", "*.arf.gz"))
    return src, bkg, rmf, arf


def load_xspec_spectrum(src: str, bkg: Optional[str], rmf: Optional[str], arf: Optional[str]) -> None:
    """Load the spectrum into XSPEC; fill in whatever the PHA header did not supply.

    PyXspec first loads the BACKFILE / RESPFILE / ANCRFILE named in the PHA
    header. Those calibrated pairings are kept: the files found in the
    directory are attached only where the header left a gap, because
    assigning ``spectrum.response`` replaces the Response object and would
    silently drop the header's ARF. A spectrum that ends up without a response
    is an error (the fit would have no effective area). XSPEC resolves file
    names relative to the working directory, hence the temporary chdir.
    """
    AllData.clear()
    AllModels.clear()
    cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(os.path.abspath(src)))
        AllData(os.path.basename(src))
        spectrum = AllData(1)
        if bkg and not spectrum.background:
            spectrum.background = os.path.basename(bkg)
        if not spectrum.responsesUsed:
            if rmf is None:
                raise ValueError(f"no response for {os.path.basename(src)}: the header names none "
                                 "(or names a file missing from its directory) and no *.rmf is "
                                 "present there; the fit needs a response.")
            spectrum.response = os.path.basename(rmf)
        if arf:
            try:
                has_arf = bool(spectrum.response.arf)
            except Exception:           # PyXspec raises when the response has no ARF
                has_arf = False
            if not has_arf:
                spectrum.response.arf = os.path.basename(arf)
        try:
            arf_in_use = spectrum.response.arf
        except Exception:
            arf_in_use = None
        print(f"  Response in use: {spectrum.response.rmf}  ARF: {arf_in_use or 'none'}  "
              f"background: {spectrum.background.fileName if spectrum.background else 'none'}")
    finally:
        os.chdir(cwd)


# ----------------------------------------------------------------------------
# Spectral fit
# ----------------------------------------------------------------------------

def fit_model(
    model_name: str,
    statistic: str = "chi",
    init_nH: float = 0.55,
    init_PhoIndex: float = 1.89,
    init_norm: float = 1e-4,
    fit_emin: float = 0.5,
    fit_emax: float = 7.0,
) -> Dict[str, float]:
    """Fit ``model_name*powerlaw`` to the loaded spectrum; return the best fit.

    PyXspec parameters are addressed by index (1: nH, 2: PhoIndex, 3: norm).
    ``Parameter.values`` is a six-float list and assigning a float sets the
    value alone. The reported uncertainty is ``Parameter.sigma``, the fit
    sigma; ``Parameter.error`` holds the result of a ``Fit.error`` run, which
    this script never performs.
    """
    if model_name not in ABSORPTION_MODELS:
        raise ValueError(f"Model must be one of {ABSORPTION_MODELS}, got: {model_name}")
    statistic = statistic.lower()
    if statistic not in ("chi", "cstat"):
        raise ValueError(f"Statistic must be 'chi' or 'cstat', got: {statistic}")

    Xset.abund = "wilm"
    Xset.xsect = "vern"
    Fit.statMethod = statistic
    AllData.ignore("**-**")
    AllData.notice(f"{fit_emin}-{fit_emax}")
    print(f"Fitting energy range: {fit_emin}-{fit_emax} keV")

    model = Model(f"{model_name}*powerlaw")
    for index, value in enumerate((init_nH, init_PhoIndex, init_norm), start=1):
        model(index).values = value
    print(f"Initial parameters: nH={init_nH:.4f} x 10^22 cm^-2, "
          f"PhoIndex={init_PhoIndex:.4f}, norm={init_norm:.4e}")

    print(f"\nFitting {model_name}*powerlaw with {statistic} statistic...")
    Fit.method = "leven"
    Fit.query = "yes"
    Fit.perform()

    params: Dict[str, float] = {}
    for index, name in enumerate(("nH", "PhoIndex", "norm"), start=1):
        params[name] = float(model(index).values[0])
        params[f"{name}_error"] = float(model(index).sigma)
    params["statistic"] = float(Fit.statistic)
    params["dof"] = float(Fit.dof)
    params["chi2_red"] = params["statistic"] / params["dof"] if params["dof"] > 0 else float("nan")

    print("\nBest-fit parameters:")
    print(f"  {model_name}.nH = {params['nH']:.6f} +/- {params['nH_error']:.6f} x 10^22 cm^-2")
    print(f"  powerlaw.PhoIndex = {params['PhoIndex']:.6f} +/- {params['PhoIndex_error']:.6f}")
    print(f"  powerlaw.norm = {params['norm']:.6e} +/- {params['norm_error']:.6e}")
    print(f"  Fit statistic = {params['statistic']:.2f} for {params['dof']:.0f} dof")
    if statistic == "chi":
        print(f"  Reduced chi^2 = {params['chi2_red']:.4f}")
    return params


def freeze_powerlaw_params():
    """Freeze PhoIndex and norm (parameters 2 and 3); return the nH parameter."""
    model = AllModels(1)
    model(2).frozen = True
    model(3).frozen = True
    print("\nFroze powerlaw parameters (PhoIndex, norm)")
    return model(1)


# ----------------------------------------------------------------------------
# Flux vs nH
# ----------------------------------------------------------------------------

def vary_nh_and_compute(nH_values_cm2: np.ndarray, band_name: str,
                        band: Tuple[float, float]) -> pd.DataFrame:
    """Step nH over the grid and let XSPEC integrate the model flux over *band*.

    Assumes the model has been fitted. ``AllModels.calcFlux`` integrates the
    current model exactly over the band, and the result is read back from the
    loaded spectrum: ``flux[0]`` is the energy flux (erg cm^-2 s^-1),
    ``flux[3]`` the photon flux (photons cm^-2 s^-1). XSPEC clips the band to
    the model energy array, so that array is first extended to 0.1-20 keV
    (``setEnergies``), which covers any band of the supported instruments.
    """
    e1, e2 = band
    if e1 >= e2:
        raise ValueError(f"Invalid band {e1}-{e2} keV (min >= max)")
    if e1 < 0.1 or e2 > 20.0:
        raise ValueError(f"Band {e1}-{e2} keV lies outside the 0.1-20 keV model energy array.")
    nh_par = freeze_powerlaw_params()
    spectrum = AllData(1)
    AllModels.setEnergies("0.1 20.0 2000 log")

    rows = []
    report_every = max(1, len(nH_values_cm2) // 10)
    print(f"\nComputing flux for {len(nH_values_cm2)} nH values...")
    for i, nH_cm2 in enumerate(nH_values_cm2):
        nH_1e22 = float(nH_cm2) / 1.0e22
        nh_par.values = nH_1e22
        AllModels.calcFlux(f"{e1} {e2}")
        flux = spectrum.flux
        rows.append({
            "nH_cm2": float(nH_cm2),
            "nH_1e22": nH_1e22,
            f"flux_{band_name}_ph": float(flux[3]),
            f"flux_{band_name}_erg": float(flux[0]),
        })
        if (i + 1) % report_every == 0:
            print(f"  Progress: {i + 1}/{len(nH_values_cm2)}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

def plot_table(df: pd.DataFrame, band_name: str, band: Tuple[float, float],
               best_fit: Dict[str, float], model_name: str, statistic: str,
               instrument: str, out_png: str) -> None:
    """Photon- and energy-flux panels with the exponential law of the table."""
    import matplotlib.pyplot as plt

    fig, (ax_ph, ax_erg) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    stat_text = (f"$\\chi^2_r$ = {best_fit['chi2_red']:.2f}" if statistic == "chi"
                 else f"C-stat = {best_fit['statistic']:.2f}")
    fig.suptitle(f"Flux vs $n_H$ ({instrument.capitalize()})  -  Model: {model_name}*powerlaw, "
                 f"Stat: {statistic}, {stat_text}", fontsize=12, fontweight="bold")

    nh = df["nH_1e22"].to_numpy(dtype=float)
    label = f"{band_name.capitalize()} {band[0]}-{band[1]} keV"
    panels = (
        (ax_ph, "ph", "Photon flux (photons cm$^{-2}$ s$^{-1}$)"),
        (ax_erg, "erg", "Energy flux (erg cm$^{-2}$ s$^{-1}$)"),
    )
    for ax, kind, ylabel in panels:
        flux = df[f"flux_{band_name}_{kind}"].to_numpy(dtype=float)
        ok = np.isfinite(flux) & (flux > 0) & (nh > 0)
        ax.plot(nh[ok] * 1e22, flux[ok], "o", ms=4, alpha=0.7, label=label)
        if np.count_nonzero(ok) >= 2:
            # Orientation only: one exponential cannot follow a table spanning many
            # decades (equal weights in log space let the absorbed tail dominate),
            # which is why the simulator interpolates the table by default.
            A, B = fit_exponential(nh[ok], flux[ok])
            ax.plot(nh[ok] * 1e22, A * np.exp(-B * nh[ok]), "--", lw=2, alpha=0.9,
                    label=f"refit law $F = {A:.3e}\\,e^{{-{B:.4f}\\,n_H}}$ (orientation only)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_erg.set_xlabel("$n_H$ (cm$^{-2}$)", fontsize=12)
    ax_ph.text(0.02, 0.98,
               "Best-fit parameters:\n"
               f"$n_H$ = {best_fit['nH']:.4f} $\\times 10^{{22}}$ cm$^{{-2}}$\n"
               f"$\\Gamma$ = {best_fit['PhoIndex']:.4f}\n"
               f"norm = {best_fit['norm']:.3e}",
               transform=ax_ph.transAxes, fontsize=9, verticalalignment="top",
               bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8, edgecolor="black"))

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_png}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tabulate the model flux vs nH with XSPEC ({phabs,tbabs,wabs}*powerlaw)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--specdir", type=str, default=os.path.join(os.getcwd(), "data", "IC10X1_spec"),
                        help="Directory containing the PHA/PI (+ background, RMF, ARF) spectrum files")
    parser.add_argument("--model", type=str, choices=ABSORPTION_MODELS, default="tbabs",
                        help="Absorption model, multiplied by a power law")
    parser.add_argument("--statistic", type=str, choices=["chi", "cstat"], default="chi",
                        help="Fit statistic: chi-squared, or the C statistic for Poisson data")
    parser.add_argument("--init_nH", type=float, default=0.55, help="Initial nH (10^22 cm^-2)")
    parser.add_argument("--init_PhoIndex", type=float, default=1.89, help="Initial power-law photon index")
    parser.add_argument("--init_norm", type=float, default=1e-4, help="Initial power-law normalization")
    parser.add_argument("--fit_emin", type=float, default=0.5, help="Lower energy of the fit range (keV)")
    parser.add_argument("--fit_emax", type=float, default=7.0, help="Upper energy of the fit range (keV)")
    parser.add_argument("--instrument", type=str, default="chandra", choices=list(INSTRUMENT_BANDS),
                        help="Instrument (determines the available energy bands)")
    parser.add_argument("--band", type=str, default=None,
                        help="Energy band to tabulate. The light-curve model is run one band at "
                             "a time, so each output CSV holds a single band. Default: the "
                             "instrument's broad band.")
    parser.add_argument("--nH_min", type=float, default=1e20, help="Min nH (cm^-2) of the grid")
    parser.add_argument("--nH_max", type=float, default=1e24, help="Max nH (cm^-2) of the grid")
    parser.add_argument("--nH_points", type=int, default=60, help="Number of log-spaced nH grid points")
    parser.add_argument("--out_csv", type=str, default="flux_vs_nH.csv", help="Output CSV")
    parser.add_argument("--out_png", type=str, default="flux_vs_nH.png", help="Output figure")
    args = parser.parse_args()

    if not os.path.isdir(args.specdir):
        parser.error(f"specdir not found: {args.specdir}")
    _import_xspec()
    bands = INSTRUMENT_BANDS[args.instrument]
    band_name = args.band or ("broad" if "broad" in bands else next(iter(bands)))
    if band_name not in bands:
        parser.error(f"Invalid band for {args.instrument}: {band_name}. Available: {list(bands)}")
    band = bands[band_name]

    print(f"Computing flux vs nH for {args.instrument}: {args.model}*powerlaw, "
          f"{args.statistic} statistic, fit range {args.fit_emin}-{args.fit_emax} keV")
    print(f"Band: {band_name} ({band[0]}-{band[1]} keV)")
    if band[0] < args.fit_emin or band[1] > args.fit_emax:
        print("Note: the band extends beyond the fit range; the flux is computed over the "
              "full band from the extrapolated model.")

    print(f"\nLoading spectrum from {args.specdir}...")
    src, bkg, rmf, arf = find_spectrum_files(args.specdir)
    for role, path in (("Source", src), ("Background", bkg), ("RMF", rmf), ("ARF", arf)):
        if path:
            print(f"  {role}: {os.path.basename(path)}")
    load_xspec_spectrum(src, bkg, rmf, arf)

    best_fit = fit_model(args.model, args.statistic, init_nH=args.init_nH,
                         init_PhoIndex=args.init_PhoIndex, init_norm=args.init_norm,
                         fit_emin=args.fit_emin, fit_emax=args.fit_emax)

    nH_values_cm2 = np.logspace(np.log10(args.nH_min), np.log10(args.nH_max), args.nH_points)
    df = vary_nh_and_compute(nH_values_cm2, band_name, band)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved: {args.out_csv} ({len(df)} rows)")

    plot_table(df, band_name, band, best_fit, args.model, args.statistic, args.instrument, args.out_png)


if __name__ == "__main__":
    main()
