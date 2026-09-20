#!/usr/bin/env python3
"""
Generate a fake absorbed power-law spectrum with PyXspec ``fakeit``.

The fake PHA (plus a fake background when one is given) is written to
``--out-dir`` together with copies of the RMF and ARF, so the fake spectrum's
header (RESPFILE/ANCRFILE, relative names) resolves from that directory and
``compute_flux_vs_nH.py --specdir <out-dir>`` builds the ``flux vs nH`` table
from it exactly as from a real spectrum. The script also
reports, for every Chandra band, the model flux, the fake count rate and their
ratio -- the flux-per-count-rate factor that ``make_lightcurve.py`` needs
(``--flux-per-rate``) -- and writes them to ``<out-dir>/band_factors.json``.

Requires PyXspec (HEASoft). Responses default to the IC 10 X-1 combined
ACIS response in ``data/IC10X1_spec``.

Example:
  python synthetic_data/make_spectrum.py --out-dir synthetic_data/out/spec \\
      --model tbabs --nH 0.75 --PhoIndex 1.86 --norm 1e-4 --exposure 100000 --seed 1
  python compute_flux_vs_nH.py --specdir synthetic_data/out/spec --band broad \\
      --out_csv synthetic_data/out/flux_vs_nH_broad.csv
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import CHANDRA_BANDS  # noqa: E402

# PyXspec is imported in main(), after argument parsing, so --help works without HEASoft.
AllData = AllModels = FakeitSettings = Model = Xset = None


def _import_xspec() -> None:
    global AllData, AllModels, FakeitSettings, Model, Xset
    try:
        from xspec import AllData as _d, AllModels as _m, FakeitSettings as _s, Model as _mo, Xset as _x  # type: ignore
    except Exception as exc:
        print("Error: XSPEC Python module not available in this environment.")
        print("Initialise HEASoft (PyXspec) first: export HEADAS=<heasoft dir>; . $HEADAS/headas-init.sh")
        print(f"Details: {exc}")
        sys.exit(1)
    AllData, AllModels, FakeitSettings, Model, Xset = _d, _m, _s, _mo, _x

ABSORPTION_MODELS = ("phabs", "tbabs", "wabs")
DEFAULT_SPEC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "IC10X1_spec")


def band_factors(spectrum, bands) -> dict:
    """Model flux, net count rate and flux-per-rate of the loaded fake spectrum per band."""
    out = {}
    for name, (e1, e2) in bands.items():
        spectrum.ignore("**-**")
        spectrum.notice(f"{e1}-{e2}")
        AllModels.calcFlux(f"{e1} {e2}")
        flux_erg = float(spectrum.flux[0])       # absorbed model flux, erg cm^-2 s^-1
        rate = float(spectrum.rate[0])           # net count rate in the noticed channels
        out[name] = {"emin_keV": e1, "emax_keV": e2, "flux_erg": flux_erg, "rate_cts_s": rate,
                     "flux_per_rate": flux_erg / rate if rate > 0 else float("nan")}
    spectrum.notice("all")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fake absorbed power-law spectrum via PyXspec fakeit, plus per-band flux-per-rate factors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out-dir", required=True, help="directory for the fake PHA (becomes --specdir)")
    parser.add_argument("--rmf", default=os.path.join(DEFAULT_SPEC_DIR, "X1_spectrum_combined_src.rmf"))
    parser.add_argument("--arf", default=os.path.join(DEFAULT_SPEC_DIR, "X1_spectrum_combined_src.arf"))
    parser.add_argument("--bkg", default=None, help="real background PHA to fake a background from (optional)")
    parser.add_argument("--model", choices=ABSORPTION_MODELS, default="tbabs")
    parser.add_argument("--nH", type=float, default=0.75, help="column density (10^22 cm^-2)")
    parser.add_argument("--PhoIndex", type=float, default=1.86, help="power-law photon index")
    parser.add_argument("--norm", type=float, default=1e-4, help="power-law normalization at 1 keV")
    parser.add_argument("--exposure", type=float, default=1.0e5, help="fake exposure (s)")
    parser.add_argument("--name", default="synthetic_src", help="stem of the fake PHA file")
    parser.add_argument("--seed", type=int, default=None, help="XSPEC random seed")
    parser.add_argument("--no-stats", action="store_true", help="fakeit without Poisson statistics")
    args = parser.parse_args()

    for path in (args.rmf, args.arf) + ((args.bkg,) if args.bkg else ()):
        if not os.path.exists(path):
            parser.error(f"file not found: {path}")
    if args.exposure <= 0 or args.norm <= 0 or args.nH < 0:
        parser.error("--exposure and --norm must be > 0 and --nH >= 0")
    _import_xspec()
    os.makedirs(args.out_dir, exist_ok=True)

    # Copy the responses next to the fake PHA: fakeit records the names it is
    # given in RESPFILE/ANCRFILE, and relative names keep the directory portable.
    local = {}
    for key, path in (("rmf", args.rmf), ("arf", args.arf)):
        target = os.path.join(args.out_dir, os.path.basename(path))
        if os.path.abspath(target) != os.path.abspath(path):
            shutil.copy2(path, target)
        local[key] = os.path.basename(path)
    if args.bkg:
        if os.path.abspath(os.path.dirname(args.bkg)) != os.path.abspath(args.out_dir):
            shutil.copy2(args.bkg, os.path.join(args.out_dir, os.path.basename(args.bkg)))
        local["bkg"] = os.path.basename(args.bkg)

    Xset.abund = "wilm"
    Xset.xsect = "vern"
    if args.seed is not None:
        Xset.seed = int(args.seed)

    AllData.clear()
    AllModels.clear()
    model = Model(f"{args.model}*powerlaw")
    for index, value in enumerate((args.nH, args.PhoIndex, args.norm), start=1):
        model(index).values = value

    fake_pha = f"{args.name}.pha"
    settings = FakeitSettings(response=local["rmf"], arf=local["arf"], background=local.get("bkg", ""),
                              exposure=str(args.exposure), fileName=fake_pha)
    cwd = os.getcwd()
    try:
        os.chdir(args.out_dir)               # fakeit writes into the working directory
        AllData.fakeit(1, settings, applyStats=not args.no_stats)
        spectrum = AllData(1)
        factors = band_factors(spectrum, CHANDRA_BANDS)
    finally:
        os.chdir(cwd)

    summary = {
        "model": f"{args.model}*powerlaw", "nH_1e22": args.nH, "PhoIndex": args.PhoIndex, "norm": args.norm,
        "exposure_s": args.exposure, "rmf": os.path.abspath(args.rmf), "arf": os.path.abspath(args.arf),
        "background": os.path.abspath(args.bkg) if args.bkg else None, "seed": args.seed,
        "fake_background": os.path.join(os.path.abspath(args.out_dir), f"{args.name}_bkg.pha") if args.bkg else None,
        "fake_pha": os.path.join(os.path.abspath(args.out_dir), fake_pha), "bands": factors,
    }
    with open(os.path.join(args.out_dir, "band_factors.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Fake spectrum: {summary['fake_pha']}  ({args.model}*powerlaw, nH={args.nH}, "
          f"Gamma={args.PhoIndex}, norm={args.norm:g}, {args.exposure:g} s)")
    print(f"{'band':8s} {'flux (erg/cm2/s)':>18s} {'rate (cts/s)':>14s} {'flux per rate':>16s}")
    for name, f in factors.items():
        print(f"{name:8s} {f['flux_erg']:18.4e} {f['rate_cts_s']:14.5f} {f['flux_per_rate']:16.4e}")
    print(f"Factors saved to {os.path.join(args.out_dir, 'band_factors.json')}")
    print(f"Next: python compute_flux_vs_nH.py --specdir {args.out_dir} --band <band> --out_csv <table.csv>")


if __name__ == "__main__":
    main()
