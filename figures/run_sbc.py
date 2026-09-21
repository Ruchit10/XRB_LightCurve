#!/usr/bin/env python3
"""
Simulation-based calibration (SBC) of the CLOAK posterior on System A.

For each draw k = 0 .. N-1 (seed k): draw a parameter vector from the fit's
own priors, simulate a System A observation with it, fit it with the same
priors, and record the rank of every true value among the posterior draws.
Calibrated posteriors give uniform ranks (Talts et al. 2018). The figure
notebook turns ``figures/results/sbc_ranks.csv`` into the rank-ECDF panels.

Design choices (state them in the paper):
- Parameterization ``--kepler-mtot``; ``q_m`` is drawn but frozen at its true
  value (its posterior would equal its prior by construction), and the
  scattered floor ``f_scatter`` is frozen at the injected value because the
  fitter's floor prior is data-driven, which SBC cannot use.
- Sampled: M_tot, R, r, i0, Rb, p, log10 f_opa. Likelihood chi2 (no intrinsic
  variability is injected). Data are generated at the data notebook's model
  step (1 degree) and fitted at the paper's default (2 degrees), so the test
  covers what a real fit does, including the interpolation error of Figure 4.

Resumable: draws already in the ranks file are skipped. Each draw's fit lives
in ``figures/cache/sbc/draw_<k>/`` (chains are not tracked).

    python figures/run_sbc.py --n-draws 100                 # ~15 min per draw on a laptop
    python figures/run_sbc.py --n-draws 1 --n-walkers 20 --n-steps 40 --n-burn 10 --thin 2   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cloak.mcmc_fit import (  # noqa: E402
    FOPACITY_PRIOR, I0_PRIOR, MODES, R_PRIOR, SMALL_R_PRIOR, WIND_SHAPE_PRIORS,
)
from cloak.synthetic import fiducial  # noqa: E402

PY = sys.executable
CACHE = os.path.join(ROOT, "figures", "cache", "sbc")
RESULTS = os.path.join(ROOT, "figures", "results")
RANKS_CSV = os.path.join(RESULTS, "sbc_ranks.csv")
SYSTEM = fiducial.SYSTEMS["A"]
SAMPLED = ["M_tot", "R", "r", "i0", "Rb", "p", "log_fopa"]
PRIORS = {
    "M_tot": MODES["kepler_mtot"]["scale_priors"]["M_tot"],
    "q_m": MODES["kepler_mtot"]["scale_priors"]["q_m"],
    "R": R_PRIOR, "r": SMALL_R_PRIOR, "i0": I0_PRIOR,
    "Rb": WIND_SHAPE_PRIORS["Rb"], "p": WIND_SHAPE_PRIORS["p"], "log_fopa": FOPACITY_PRIOR,
}


def draw_truncated_normal(rng: np.random.Generator, prior: dict) -> float:
    """One draw from N(mean, std) truncated to (min, max) by rejection."""
    for _ in range(100000):
        x = rng.normal(prior["mean"], prior["std"])
        if prior["min"] < x < prior["max"]:
            return float(x)
    raise RuntimeError(f"could not draw inside {prior}")


def draw_truth(rng: np.random.Generator) -> dict:
    """A parameter vector from the fit's priors, honouring the fit's constraints (r < R, Rb >= R)."""
    while True:
        t = {name: draw_truncated_normal(rng, PRIORS[name]) for name in SAMPLED + ["q_m"]}
        if t["r"] < t["R"] and t["Rb"] >= t["R"]:
            return t


def run(cmd, log_path):
    with open(log_path, "w") as log:
        res = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"command failed (rc={res.returncode}); see {log_path}\n  {' '.join(cmd)}")


def existing_draws() -> set:
    if not os.path.exists(RANKS_CSV):
        return set()
    with open(RANKS_CSV) as fh:
        return {int(row["draw"]) for row in csv.DictReader(fh)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-draws", type=int, default=100)
    parser.add_argument("--start", type=int, default=0, help="first draw index (seeds are the draw index)")
    parser.add_argument("--flux-csv", default=os.path.join(ROOT, "synthetic_data", "flux_vs_nH_tbabs_broad.csv"))
    parser.add_argument("--band", default="broad")
    parser.add_argument("--dth", type=float, default=2.0, help="model step of the fit (the paper's default)")
    parser.add_argument("--gen-dth", type=float, default=1.0, help="model step used to generate the data")
    parser.add_argument("--n-walkers", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=2500)
    parser.add_argument("--n-burn", type=int, default=500)
    parser.add_argument("--thin", type=int, default=20, help="keep every thin-th post-burn step for the ranks")
    parser.add_argument("--n-threads", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    done = existing_draws()
    obs = SYSTEM["observation"]
    period = float(SYSTEM["period_s"])
    K = fiducial.kepler_prefactor(period)
    header_needed = not os.path.exists(RANKS_CSV)

    for k in range(args.start, args.start + args.n_draws):
        if k in done:
            print(f"draw {k}: already recorded, skipping")
            continue
        rng = np.random.default_rng(k)
        truth = draw_truth(rng)
        a = K * truth["M_tot"] ** (1.0 / 3.0)
        d1, d2 = a * truth["q_m"], a * (1.0 - truth["q_m"])
        draw_dir = os.path.join(CACHE, f"draw_{k:03d}")
        os.makedirs(draw_dir, exist_ok=True)
        lc_dir = os.path.join(draw_dir, "lc")
        lc_file = os.path.join(lc_dir, f"sbc_{args.band}.txt")

        # Out-of-eclipse flux of this system sets the floor and the count rate,
        # exactly as the data notebook does for the fiducial systems.
        from cloak.kernel import simulate_band_flux
        sim = dict(r=truth["r"], R=truth["R"], d1=d1, d2=d2, i0=truth["i0"], dth=args.gen_dth,
                   wind_model="smooth_pl", wind_params={"Rb": truth["Rb"], "p": truth["p"], "Delta": 2.0},
                   mdot=SYSTEM["mdot"], v_inf=SYSTEM["v_inf"], mu_wind=SYSTEM["mu_wind"],
                   f_opacity=10.0 ** truth["log_fopa"], flux_csv_path=args.flux_csv, band=args.band)
        _, flux = simulate_band_flux(**sim)
        f_out = float(np.max(flux))
        floor = obs["scatter_fraction"] * f_out
        flux_per_rate = f_out / obs["target_rate"]
        visits = fiducial.visits_arg(SYSTEM)

        if not os.path.exists(lc_file):
            gen = [PY, "-m", "cloak.synthetic.lightcurve", "--flux-csv", args.flux_csv, "--band", args.band,
                   "--r", f"{truth['r']:.8g}", "--R", f"{truth['R']:.8g}", "--d1", f"{d1:.8g}", "--d2", f"{d2:.8g}",
                   "--i0", f"{truth['i0']:.8g}", "--wind-model", "smooth_pl", "--Rb", f"{truth['Rb']:.8g}",
                   "--p", f"{truth['p']:.8g}", "--Delta", "2.0", "--mdot", f"{SYSTEM['mdot']:g}",
                   "--v-inf", f"{SYSTEM['v_inf']:g}", "--f-opacity", f"{10.0 ** truth['log_fopa']:.8g}",
                   "--dth", f"{args.gen_dth:g}", "--orbital-period", f"{period:g}",
                   "--phase-shift", f"{obs['phase_shift']:g}", "--scatter", f"{floor:.8g}",
                   "--flux-per-rate", f"{flux_per_rate:.8g}", "--dt", f"{obs['dt']:g}", "--visits", visits,
                   "--gap-fraction", f"{obs['gap_fraction']:g}", "--gap-duration", f"{obs['gap_duration']:g}",
                   "--seed", str(k), "--output", lc_file]
            run(gen, os.path.join(draw_dir, "generate.log"))

        fit_dir = os.path.join(draw_dir, "fit")
        chain = os.path.join(fit_dir, f"{args.band}_smooth_pl_chain.npz")
        if not os.path.exists(chain):
            fit = [PY, "-m", "cloak.mcmc_fit", "--band", args.band, "--flux-csv", args.flux_csv,
                   "--data-dir", lc_dir, "--obs-column", "flux_t", "--time-column", "t_raw",
                   "--counts-per-bin", "100", "--keep-zero-flux", "--kepler-mtot",
                   "--orbital-period", f"{period:g}", "--fit-wind-shape", "--fit-fopacity",
                   "--freeze", f"q_m={truth['q_m']:.8g},f_scatter={floor:.8g}",
                   "--likelihood", "chi2", "--dth", f"{args.dth:g}", "--n-walkers", str(args.n_walkers),
                   "--n-steps", str(args.n_steps), "--n-burn", str(args.n_burn), "--seed", str(k),
                   "--quiet", "--no-plots", "--no-csv-output", "--output-dir", fit_dir]
            if args.n_threads > 1:
                fit += ["--n-threads", str(args.n_threads)]
            run(fit, os.path.join(draw_dir, "fit.log"))

        meta = np.load(chain, allow_pickle=False)
        names = [str(n) for n in meta["param_names"]]
        post = meta["chain"][::args.thin].reshape(-1, len(names))     # (step, walker) order, thinned
        with open(os.path.join(draw_dir, "truth.json"), "w") as fh:
            json.dump({**truth, "a": a, "d1": d1, "d2": d2, "floor": floor, "flux_per_rate": flux_per_rate}, fh, indent=2)
        with open(RANKS_CSV, "a", newline="") as fh:
            writer = csv.writer(fh)
            if header_needed:
                writer.writerow(["draw", "param", "truth", "rank", "n_post"])
                header_needed = False
            for j, name in enumerate(names):
                if name not in truth:
                    continue
                rank = int(np.sum(post[:, j] < truth[name]))
                writer.writerow([k, name, f"{truth[name]:.10g}", rank, post.shape[0]])
        print(f"draw {k}: ranks recorded ({post.shape[0]} posterior draws, {len(names)} parameters)")
    print(f"Ranks: {RANKS_CSV}")


if __name__ == "__main__":
    main()
