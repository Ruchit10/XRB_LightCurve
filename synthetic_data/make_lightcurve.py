#!/usr/bin/env python3
"""
Generate a synthetic Chandra-style light curve from the forward model.

The model band flux is evaluated at known parameters (the same keywords as
``xrb_lightcurve.py``), shifted in phase and lifted by a scattered-flux floor,
converted to an expected count rate with a flux-per-count-rate factor, and
Poisson-sampled in ``--dt`` s bins over one or more observing visits with
optional gaps. The output uses the CIAO layout the fitters read unchanged::

    # Columns: dt, t_raw, mjd, phase, counts, rate, rate_err, flux_t

so ``mcmc_lightcurve_fit.py --data-dir <dir> --obs-column flux_t --time-column
t_raw`` and ``chandra_phase_analysis.py`` work on it as on real data. A
``<output>_truth.json`` next to the file records every injected value.

Examples
~~~~~~~~
# One full orbit at the IC 10 X-1 working parameters, 100 s bins:
python synthetic_data/make_lightcurve.py --flux-csv flux_vs_nH_broad.csv --band broad \\
    --R 2 --r 0.001 --d1 11 --d2 8 --i0 78 --f-opacity 0.02 \\
    --phase-shift 0.985 --scatter 3e-13 --seed 1 \\
    --output synthetic_data/out/broad/synth_broad.txt

# Three visits (start:duration in s after REF_EPOCH) with 10 % of the bins
# lost to random 3 ks gaps, beta_law wind, no noise:
python synthetic_data/make_lightcurve.py --flux-csv flux_vs_nH_broad.csv --band broad \\
    --wind-model beta_law --beta 0.8 --H 1.5 --visits 0:150000,400000:60000,900000:90000 \\
    --gap-fraction 0.1 --gap-duration 3000 --noiseless --output out/synth.txt

Conventions (matching the CIAO files): ``dt`` is elapsed time from the first
bin, ``t_raw`` mission time (``REF_EPOCH`` + offset), ``mjd = 50814 + t/86400``,
``rate = net counts / dt``, ``rate_err = sqrt(counts) / dt`` (0 for empty bins),
``flux_t = rate * flux_per_rate``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xrb_lightcurve import (  # noqa: E402
    SIM_DEFAULTS,
    WIND_MODEL_IDS,
    WIND_MODEL_PARAM_KEYS,
    default_wind_params,
    simulate_band_flux,
)
from utils.utils import (  # noqa: E402
    CHANDRA_BANDS,
    MJDREF_CHANDRA,
    ORBITAL_PERIOD,
    REF_EPOCH,
    eval_periodic,
    frac,
    periodic_model,
)

# Flux per unit count rate (flux_t / rate), erg cm^-2 s^-1 per count s^-1, of
# the broad-band IC 10 X-1 light curve of ObsID 15803 (other ObsIDs span
# 0.85-1.49e-11). Replace with the value make_spectrum.py reports for your
# synthetic spectrum and band.
DEFAULT_FLUX_PER_RATE = 1.13e-11


def add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    """The forward-model keywords, with the simulator's own defaults."""
    D = SIM_DEFAULTS
    geo = parser.add_argument_group("Geometry (solar radii, degrees)")
    for name, help_text in (("r", "emitter (compact object / disk) radius"),
                            ("R", "companion photospheric radius"),
                            ("d1", "compact-object distance from the centre of mass"),
                            ("d2", "companion distance from the centre of mass"),
                            ("i0", "inclination from the orbital-plane normal (90 = edge-on)"),
                            ("gma0", "starting phase angle"),
                            ("dth", "model phase step"),
                            ("d2h", "angular cell size of the emitter grid")):
        geo.add_argument(f"--{name}", type=float, default=D[name], help=help_text)
    wind = parser.add_argument_group("Wind")
    wind.add_argument("--wind-model", type=str, choices=list(WIND_MODEL_IDS), default=D["wind_model"],
                      help="wind density profile")
    shape_defaults: Dict[str, float] = {}
    for model in WIND_MODEL_IDS:
        shape_defaults.update({k: v for k, v in default_wind_params(model, D["R"]).items() if k != "R_star"})
    for name, value in shape_defaults.items():
        owners = ", ".join(m for m, keys in WIND_MODEL_PARAM_KEYS.items() if name in keys)
        wind.add_argument(f"--{name}", type=float, default=value, help=f"shape parameter of the {owners} profile")
    wind.add_argument("--mdot", type=float, default=D["mdot"], help="mass-loss rate (Msun/yr)")
    wind.add_argument("--v-inf", type=float, default=D["v_inf"], help="terminal velocity (km/s)")
    wind.add_argument("--mu-wind", type=float, default=D["mu_wind"], help="mean mass per H-equivalent nucleus")
    wind.add_argument("--f-opacity", type=float, default=D["f_opacity"], help="effective-opacity factor")
    parser.add_argument("--flux-method", type=str, choices=["interpolate", "refit"], default=D["flux_method"],
                        help="nH -> flux conversion: table interpolation or the fitted exponential")
    parser.add_argument("--flux-type", type=str, choices=["erg", "ph"], default=D["flux_type"],
                        help="table column to use: energy flux (erg) or photon flux (ph)")


def simulation_kwargs(args) -> Dict[str, object]:
    """simulate_band_flux keywords from the parsed arguments (R_star is tied to R)."""
    wind_params = {k: getattr(args, k) for k in WIND_MODEL_PARAM_KEYS[args.wind_model] if k != "R_star"}
    if "R_star" in WIND_MODEL_PARAM_KEYS[args.wind_model]:
        wind_params["R_star"] = args.R
    return dict(r=args.r, R=args.R, d1=args.d1, d2=args.d2, i0=args.i0, gma0=args.gma0,
                dth=args.dth, d2h=args.d2h, flux_method=args.flux_method, flux_type=args.flux_type,
                flux_csv_path=args.flux_csv, band=args.band, wind_model=args.wind_model,
                wind_params=wind_params, mdot=args.mdot, v_inf=args.v_inf,
                mu_wind=args.mu_wind, f_opacity=args.f_opacity)


def parse_visits(spec: str | None, t_start: float, n_orbits: float) -> List[Tuple[float, float]]:
    """``start:duration,start:duration,...`` in seconds after REF_EPOCH, or one
    visit of *n_orbits* orbits starting at *t_start*. Visits must not overlap."""
    if not spec:
        if n_orbits <= 0:
            raise ValueError("--n-orbits must be > 0")
        return [(float(t_start), float(n_orbits) * ORBITAL_PERIOD)]
    visits = []
    for item in spec.split(","):
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"visit '{item}': expected start:duration (seconds after REF_EPOCH)")
        try:
            start, duration = float(parts[0]), float(parts[1])
        except ValueError:
            raise ValueError(f"visit '{item}': start and duration must be numbers") from None
        if duration <= 0:
            raise ValueError(f"visit '{item}': duration must be > 0")
        visits.append((start, duration))
    visits.sort()
    for (s0, d0), (s1, _) in zip(visits, visits[1:]):
        if s1 < s0 + d0:
            raise ValueError(f"visits overlap: {s0}:{d0} and {s1}:...; timestamps would repeat")
    return visits


def visit_times(visits: List[Tuple[float, float]], dt: float) -> List[np.ndarray]:
    """Bin start offsets (s after REF_EPOCH) of every visit, one array per visit."""
    return [start + dt * np.arange(int(np.floor(duration / dt))) for start, duration in visits]


def apply_gaps(blocks: List[np.ndarray], dt: float, fraction: float, duration: float,
               rng: np.random.Generator) -> np.ndarray:
    """Remove random contiguous gaps of *duration* s from each visit until
    *fraction* of its bins is gone; a gap never crosses a visit boundary and
    every visit keeps at least one bin."""
    if fraction <= 0.0:
        return np.concatenate(blocks)
    n_per_gap = max(1, int(round(duration / dt)))
    kept = []
    for t in blocks:
        keep = np.ones(t.size, dtype=bool)
        target = int(round(fraction * t.size))
        n_gap = min(n_per_gap, max(1, t.size - 1))          # leave at least one bin
        removed, guard = 0, 0
        while removed < target and guard < 100 * t.size:
            guard += 1
            start = int(rng.integers(0, t.size - n_gap + 1))
            block = keep[start:start + n_gap]
            new = int(block.sum())
            if keep.sum() - new < 1:
                continue
            removed += new
            block[:] = False
        kept.append(t[keep])
    return np.concatenate(kept)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic CIAO-layout light curve from the forward model at known parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--flux-csv", required=True, help="flux vs nH table (compute_flux_vs_nH.py)")
    parser.add_argument("--band", type=str, choices=list(CHANDRA_BANDS), default="broad",
                        help="energy band; the table must contain it")
    parser.add_argument("--output", required=True, help="output light-curve file (.txt); truth goes to <stem>_truth.json")
    add_simulation_arguments(parser)

    obs = parser.add_argument_group("Observation")
    obs.add_argument("--flux-per-rate", type=float, default=DEFAULT_FLUX_PER_RATE,
                     help="flux per unit count rate for this band, erg cm^-2 s^-1 per count s^-1 "
                          "(make_spectrum.py reports it per band)")
    obs.add_argument("--dt", type=float, default=100.0, help="time bin (s)")
    obs.add_argument("--visits", type=str, default=None,
                     help="visits as start:duration[,start:duration...] in seconds after REF_EPOCH; "
                          "default one visit of --n-orbits orbits from --t-start")
    obs.add_argument("--t-start", type=float, default=0.0, help="start of the single default visit (s after REF_EPOCH)")
    obs.add_argument("--n-orbits", type=float, default=1.0, help="length of the single default visit in orbits")
    obs.add_argument("--gap-fraction", type=float, default=0.0, help="fraction of bins removed by random gaps")
    obs.add_argument("--gap-duration", type=float, default=3000.0, help="length of each random gap (s)")
    obs.add_argument("--phase-shift", type=float, default=0.0,
                     help="phase shift applied to the model (data phase of mid-eclipse = 0.5 + shift)")
    obs.add_argument("--scatter", type=float, default=0.0, help="additive scattered-flux floor (erg cm^-2 s^-1)")
    obs.add_argument("--bkg-rate", type=float, default=0.0,
                     help="background count rate added before sampling and subtracted from the net rate")
    obs.add_argument("--noiseless", action="store_true", help="write expected counts instead of Poisson draws")
    obs.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()

    if args.dt <= 0:
        parser.error("--dt must be > 0")
    if args.flux_per_rate <= 0:
        parser.error("--flux-per-rate must be > 0")
    if not (0.0 <= args.gap_fraction < 1.0):
        parser.error("--gap-fraction must be in [0, 1)")
    if args.bkg_rate < 0 or args.scatter < 0:
        parser.error("--bkg-rate and --scatter must be >= 0")
    try:
        visits = parse_visits(args.visits, args.t_start, args.n_orbits)
    except ValueError as e:
        parser.error(str(e))

    rng = np.random.default_rng(args.seed)
    sim = simulation_kwargs(args)

    # Time grid and phases exactly as the loaders compute them.
    blocks = visit_times(visits, args.dt)
    if any(b.size == 0 for b in blocks):
        parser.error("a visit is shorter than one --dt bin")
    offsets = apply_gaps(blocks, args.dt, args.gap_fraction, args.gap_duration, rng)
    t_raw = REF_EPOCH + offsets
    phase = frac((t_raw - REF_EPOCH) / ORBITAL_PERIOD)

    # Model band flux at the observed phases: native curve, shifted, plus the floor.
    try:
        model_phase, model_flux = simulate_band_flux(**sim)
    except (FileNotFoundError, ValueError, KeyError) as e:
        parser.error(str(e))
    flux_true = eval_periodic(*periodic_model(model_phase, model_flux), phase,
                              shift=args.phase_shift, offset=args.scatter)

    expected_src = flux_true / args.flux_per_rate * args.dt       # source counts per bin
    bkg_counts = args.bkg_rate * args.dt
    expected = expected_src + bkg_counts
    counts = expected if args.noiseless else rng.poisson(expected).astype(float)
    net_rate = (counts - bkg_counts) / args.dt
    # Poisson error of the total counts plus the variance of the subtracted
    # background estimate; without background this is sqrt(counts)/dt and a
    # zero-count bin carries a zero error, as in the CIAO files.
    rate_err = np.sqrt(counts + bkg_counts) / args.dt
    flux_t = net_rate * args.flux_per_rate

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    header = ("# Synthetic light curve from xrb_lightcurve (synthetic_data/make_lightcurve.py)\n"
              f"# band {args.band}; wind_model {args.wind_model}; phase_shift {args.phase_shift}; "
              f"scatter {args.scatter:g}; flux_per_rate {args.flux_per_rate:g}; seed {args.seed}\n"
              "# Columns: dt, t_raw, mjd, phase, counts, rate, rate_err, flux_t, exposure\n# \n")
    # The CIAO layout plus an explicit exposure column: with it the loaders know
    # that a zero-count row was observed (a CIAO file cannot tell a GTI gap
    # from an empty bin) and the binners weight rows by exposure.
    table = np.column_stack([offsets - offsets[0], t_raw, MJDREF_CHANDRA + t_raw / 86400.0, phase,
                             counts, net_rate, rate_err, flux_t, np.full(offsets.size, float(args.dt))])
    with open(args.output, "w") as fh:
        fh.write(header)
        np.savetxt(fh, table, fmt="%.18e")

    stem = os.path.splitext(args.output)[0]
    truth = {
        "simulation": {k: (v if not isinstance(v, dict) else dict(v)) for k, v in sim.items()},
        "phase_shift": args.phase_shift,
        "mid_eclipse_data_phase": float((0.5 + args.phase_shift) % 1.0),
        "scatter": args.scatter,
        "flux_per_rate": args.flux_per_rate,
        "bkg_rate": args.bkg_rate,
        "dt": args.dt,
        "visits_s_after_ref_epoch": visits,
        "gap_fraction": args.gap_fraction,
        "gap_duration": args.gap_duration,
        "noiseless": args.noiseless,
        "seed": args.seed,
        "ephemeris": {"REF_EPOCH": REF_EPOCH, "ORBITAL_PERIOD": ORBITAL_PERIOD},
        "n_bins": int(offsets.size),
        "total_counts": float(counts.sum()),
        "zero_count_bins": int(np.sum(counts == 0)),
        "output": os.path.abspath(args.output),
    }
    with open(f"{stem}_truth.json", "w") as fh:
        json.dump(truth, fh, indent=2)

    print(f"Wrote {offsets.size} bins ({len(visits)} visit(s), {args.gap_fraction:.0%} gaps) to {args.output}")
    print(f"  total counts {counts.sum():.0f}, zero-count bins {int(np.sum(counts == 0))}, "
          f"mean rate {net_rate.mean():.4f} cts/s, mid-eclipse at data phase {truth['mid_eclipse_data_phase']:.3f}")
    print(f"  truth: {stem}_truth.json")
    print(f"Fit with: python mcmc_lightcurve_fit.py --band {args.band} --flux-csv {args.flux_csv} "
          f"--data-dir {out_dir} --obs-column flux_t --time-column t_raw ...")


if __name__ == "__main__":
    main()
