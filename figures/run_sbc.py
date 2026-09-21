#!/usr/bin/env python3
"""
Simulation-based calibration (SBC) of the CLOAK posterior on System A.

For each draw k (seed k): draw a parameter vector from the fit's own priors,
simulate a System A observation with it, fit it with the same model and the
same priors, and record the rank of every true value among the posterior
draws. Calibrated posteriors give uniform ranks (Talts et al. 2018). The
figure notebook turns ``figures/results/sbc_ranks.csv`` into rank-ECDF panels.

The fit is the paper's fit: ``--kepler-mtot`` with wind shape, opacity and the
scattered floor free, the jitter likelihood, the same walkers/steps/burn-in as
``figlib.FIT_SETTINGS``, data generated at 1 degree and fitted at 2 degrees.
Two things a calibration test cannot take from the paper's fits, stated here
and in the caption:
- the floor gets a fixed prior (``figlib.floor_prior()``, via ``--prior-fscatter``)
  instead of the data-driven one, and the injected floor is drawn from it;
- ``q_m`` is drawn but frozen at its true value (its posterior equals its
  prior by construction, so ranking it tests nothing).
Intrinsic variability of ``figlib.SBC_INTRINSIC_SCATTER`` is injected per time
row; the jitter parameter's "truth" is only defined up to the sqrt(n) dilution
over the rows of a bin, so its ranks are recorded as a diagnostic (flagged in
the ``note`` column) and excluded from the calibration verdict.

Each row also records the fitter's convergence verdict, and the notebook uses
converged draws only. The ranks file carries a configuration digest; a run with
different settings refuses to append to it. Resumable: recorded draws are
skipped. Each draw's fit lives in ``figures/cache/sbc/draw_<k>/`` (untracked).

    python figures/run_sbc.py --n-draws 50            # ~25-30 min per draw on a laptop
    python figures/run_sbc.py --n-draws 1 --n-walkers 20 --n-steps 60 --n-burn 20 --thin 2 \\
        --ranks /tmp/sbc_smoke.csv                    # smoke test (never into results/)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import figlib as L  # noqa: E402
from cloak import utils as U  # noqa: E402
from cloak.kernel import simulate_band_flux  # noqa: E402
from cloak.mcmc_fit import (  # noqa: E402
    FOPACITY_PRIOR, I0_PRIOR, MODES, R_PRIOR, SMALL_R_PRIOR, WIND_SHAPE_PRIORS,
)
from cloak.synthetic import fiducial  # noqa: E402

PY = sys.executable
CACHE = os.path.join(L.CACHE, "sbc")
SYSTEM = fiducial.SYSTEMS["A"]
FLOOR_PRIOR = L.floor_prior()                                     # follows the flux scale of the table in use
SAMPLED = ["M_tot", "R", "r", "i0", "Rb", "p", "log_fopa"]          # ranked and judged
PRIORS = {
    "M_tot": MODES["kepler_mtot"]["scale_priors"]["M_tot"],
    "q_m": MODES["kepler_mtot"]["scale_priors"]["q_m"],
    "R": R_PRIOR, "r": SMALL_R_PRIOR, "i0": I0_PRIOR,
    "Rb": WIND_SHAPE_PRIORS["Rb"], "p": WIND_SHAPE_PRIORS["p"], "log_fopa": FOPACITY_PRIOR,
    "f_scatter": FLOOR_PRIOR,
}
COLUMNS = ["draw", "param", "truth", "rank", "n_post", "converged", "note", "config"]


def draw_truncated_normal(rng: np.random.Generator, prior: dict) -> float:
    for _ in range(100000):
        x = rng.normal(prior["mean"], prior["std"])
        if prior["min"] < x < prior["max"]:
            return float(x)
    raise RuntimeError(f"could not draw inside {prior}")


def draw_truth(rng: np.random.Generator) -> dict:
    """A parameter vector from the fit's priors, honouring the fit's constraints (r < R, Rb >= R)."""
    while True:
        t = {name: draw_truncated_normal(rng, PRIORS[name]) for name in SAMPLED + ["q_m", "f_scatter"]}
        if t["r"] < t["R"] and t["Rb"] >= t["R"]:
            return t


def run(cmd, log_path):
    with open(log_path, "w") as log:
        res = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"command failed (rc={res.returncode}); see {log_path}\n  {' '.join(cmd)}")


def config_digest(args) -> str:
    settings = dict(band=args.band, dth=args.dth, gen_dth=args.gen_dth, n_walkers=args.n_walkers,
                    n_steps=args.n_steps, n_burn=args.n_burn, thin=args.thin, floor_prior=FLOOR_PRIOR,
                    intrinsic_scatter=L.SBC_INTRINSIC_SCATTER, priors={k: v for k, v in PRIORS.items()},
                    table=os.path.basename(args.flux_csv), extra_args=list(L.FIT_SETTINGS["extra_args"]))
    return hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]


def read_ranks(path: str):
    """Existing rows (list of dicts) and the file's configuration digest, or ([], None)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return [], None
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "config" not in rows[0]:
        raise RuntimeError(f"{path} is not a ranks file written by this script; move it away.")
    return rows, rows[0]["config"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-draws", type=int, default=50)
    parser.add_argument("--start", type=int, default=0, help="first draw index (seeds are the draw index)")
    parser.add_argument("--flux-csv", default=None, help="flux-vs-nH table (default: the broad table figlib uses)")
    parser.add_argument("--band", default="broad")
    parser.add_argument("--dth", type=float, default=2.0, help="model step of the fit (the paper's default)")
    parser.add_argument("--gen-dth", type=float, default=1.0, help="model step used to generate the data")
    parser.add_argument("--n-walkers", type=int, default=L.FIT_SETTINGS["n_walkers"])
    parser.add_argument("--n-steps", type=int, default=L.FIT_SETTINGS["n_steps"])
    parser.add_argument("--n-burn", type=int, default=L.FIT_SETTINGS["n_burn"])
    parser.add_argument("--thin", type=int, default=20, help="keep every thin-th post-burn step for the ranks")
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--ranks", default=os.path.join(L.RESULTS, "sbc_ranks.csv"),
                        help="ranks file (point a smoke test elsewhere so it never pollutes results/)")
    args = parser.parse_args()
    if args.flux_csv is None:
        args.flux_csv = L.require_table(args.band)
    digest = config_digest(args)

    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.ranks)), exist_ok=True)
    rows, file_digest = read_ranks(args.ranks)
    if file_digest is not None and file_digest != digest:
        raise SystemExit(f"{args.ranks} was written with a different configuration ({file_digest}, now {digest}); "
                         f"move it away or point --ranks elsewhere.")
    done = {int(r["draw"]) for r in rows}
    obs = SYSTEM["observation"]
    period = float(SYSTEM["period_s"])
    K = fiducial.kepler_prefactor(period)
    fp = FLOOR_PRIOR
    floor_prior_arg = f"--prior-fscatter={fp['mean']:.6g},{fp['std']:.6g},{fp['min']:.6g},{fp['max']:.6g}"

    for k in range(args.start, args.start + args.n_draws):
        if k in done:
            print(f"draw {k}: already recorded, skipping")
            continue
        rng = np.random.default_rng(k)
        truth = draw_truth(rng)
        a = K * truth["M_tot"] ** (1.0 / 3.0)
        d1, d2 = a * truth["q_m"], a * (1.0 - truth["q_m"])
        draw_dir = os.path.join(CACHE, f"draw_{k:03d}_{digest}")
        os.makedirs(draw_dir, exist_ok=True)
        lc_dir = os.path.join(draw_dir, "lc")
        lc_file = os.path.join(lc_dir, f"sbc_{args.band}.txt")

        sim = dict(r=truth["r"], R=truth["R"], d1=d1, d2=d2, i0=truth["i0"], dth=args.gen_dth,
                   wind_model="smooth_pl", wind_params={"Rb": truth["Rb"], "p": truth["p"], "Delta": 2.0},
                   mdot=SYSTEM["mdot"], v_inf=SYSTEM["v_inf"], mu_wind=SYSTEM["mu_wind"],
                   f_opacity=10.0 ** truth["log_fopa"], flux_csv_path=args.flux_csv, band=args.band)
        _, flux = simulate_band_flux(**sim)
        f_out = float(np.max(flux))
        flux_per_rate = f_out / obs["target_rate"]
        if not os.path.exists(lc_file):
            gen = [PY, "-m", "cloak.synthetic.lightcurve", "--flux-csv", args.flux_csv, "--band", args.band,
                   "--r", f"{truth['r']:.8g}", "--R", f"{truth['R']:.8g}", "--d1", f"{d1:.8g}", "--d2", f"{d2:.8g}",
                   "--i0", f"{truth['i0']:.8g}", "--wind-model", "smooth_pl", "--Rb", f"{truth['Rb']:.8g}",
                   "--p", f"{truth['p']:.8g}", "--Delta", "2.0", "--mdot", f"{SYSTEM['mdot']:g}",
                   "--v-inf", f"{SYSTEM['v_inf']:g}", "--mu-wind", f"{SYSTEM['mu_wind']:g}",
                   "--f-opacity", f"{10.0 ** truth['log_fopa']:.8g}",
                   "--dth", f"{args.gen_dth:g}", "--orbital-period", f"{period:g}",
                   "--phase-shift", f"{obs['phase_shift']:g}", "--scatter", f"{truth['f_scatter']:.8g}",
                   "--intrinsic-scatter", f"{L.SBC_INTRINSIC_SCATTER:g}",
                   "--flux-per-rate", f"{flux_per_rate:.8g}", "--dt", f"{obs['dt']:g}",
                   "--visits", fiducial.visits_arg(SYSTEM),
                   "--gap-fraction", f"{obs['gap_fraction']:g}", "--gap-duration", f"{obs['gap_duration']:g}",
                   "--seed", str(k), "--output", lc_file]
            run(gen, os.path.join(draw_dir, "generate.log"))

        fit_dir = os.path.join(draw_dir, "fit")
        chain = os.path.join(fit_dir, f"{args.band}_smooth_pl_chain.npz")
        if not os.path.exists(chain):
            fit = [PY, "-m", "cloak.mcmc_fit", "--band", args.band, "--flux-csv", args.flux_csv,
                   "--data-dir", lc_dir, "--obs-column", "flux_t", "--time-column", "t_raw",
                   "--counts-per-bin", "100", "--keep-zero-flux", "--kepler-mtot",
                   "--orbital-period", f"{period:g}", "--wind-model", "smooth_pl",
                   "--mdot", f"{SYSTEM['mdot']:g}", "--v-inf", f"{SYSTEM['v_inf']:g}", "--mu-wind", f"{SYSTEM['mu_wind']:g}",
                   "--fit-wind-shape", "--fit-fopacity", "--fit-scatter", floor_prior_arg,
                   "--freeze", f"q_m={truth['q_m']:.8g}", "--likelihood", "jitter", "--dth", f"{args.dth:g}",
                   "--n-walkers", str(args.n_walkers), "--n-steps", str(args.n_steps), "--n-burn", str(args.n_burn),
                   "--seed", str(k), "--quiet", "--no-plots", "--no-csv-output", "--output-dir", fit_dir] \
                  + list(L.FIT_SETTINGS["extra_args"])
            if args.n_threads > 1:
                fit += ["--n-threads", str(args.n_threads)]
            run(fit, os.path.join(draw_dir, "fit.log"))

        meta = np.load(chain, allow_pickle=False)
        names = [str(n) for n in meta["param_names"]]
        post = meta["chain"][::args.thin].reshape(-1, len(names))     # (step, walker) order, thinned
        diag_path = os.path.join(fit_dir, f"{args.band}_smooth_pl_diagnostics.json")
        converged = None
        if os.path.exists(diag_path):
            with open(diag_path) as fh:
                converged = json.load(fh).get("converged")
        # Jitter reference: the injected per-row variability diluted over the rows of a bin.
        data = U.load_data(lc_dir, obs_column="flux_t", time_column="t_raw", period=period)
        binned = U.phase_bin_data_snr(data, counts_per_bin=100, verbose=False)
        n_bar = float(np.median(binned["n_points"]))
        truth["log_f"] = float(np.log(L.SBC_INTRINSIC_SCATTER / np.sqrt(n_bar)))
        with open(os.path.join(draw_dir, "truth.json"), "w") as fh:
            json.dump({**truth, "a": a, "d1": d1, "d2": d2, "flux_per_rate": flux_per_rate, "rows_per_bin": n_bar,
                       "converged": converged, "config": digest}, fh, indent=2)
        new_file = not os.path.exists(args.ranks) or os.path.getsize(args.ranks) == 0
        with open(args.ranks, "a", newline="") as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(COLUMNS)
            for j, name in enumerate(names):
                if name not in truth:
                    continue
                rank = int(np.sum(post[:, j] < truth[name]))
                note = "diagnostic: reference approximate" if name == "log_f" else ""
                writer.writerow([k, name, f"{truth[name]:.10g}", rank, post.shape[0], converged, note, digest])
        print(f"draw {k}: ranks recorded ({post.shape[0]} posterior draws, {len(names)} parameters, "
              f"converged={converged})")
    print(f"Ranks: {args.ranks}")


if __name__ == "__main__":
    main()
