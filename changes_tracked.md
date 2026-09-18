# XRB Lightcurve Project — Change Log

Tracks the evolution of the IC 10 X-1 X-ray binary lightcurve simulation,
fitting, and inference stack since the original R port.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Phase 1 — R → Python Migration](#phase-1--r--python-migration)
3. [Phase 2 — Flux Integration & XSPEC Support](#phase-2--flux-integration--xspec-support)
4. [Phase 3 — Light-Curve Data Pipeline](#phase-3--light-curve-data-pipeline)
5. [Phase 4 — Spectral Model Analysis](#phase-4--spectral-model-analysis)
6. [Phase 5 — Phase Folding & χ² Fitting](#phase-5--phase-folding--χ²-fitting)
7. [Phase 6 — Eclipse Geometry & LOS Cutoff Fixes](#phase-6--eclipse-geometry--los-cutoff-fixes)
8. [Phase 7 — Unified Wind Model](#phase-7--unified-wind-model)
9. [Phase 8 — MCMC Pipeline & Performance](#phase-8--mcmc-pipeline--performance)
10. [Phase 9 — Convergence Improvements (Reparameterization)](#phase-9--convergence-improvements-reparameterization)
11. [Phase 10 — Wind-Shape MCMC Parameters](#phase-10--wind-shape-mcmc-parameters)
12. [Phase 11 — N-D Precomputed Grid for Shape-Fit MCMC](#phase-11--n-d-precomputed-grid-for-shape-fit-mcmc) *(later removed)*
13. [Phase 12 — flux_t Errors & First-Class Unbinned MCMC](#phase-12--flux_t-errors--first-class-unbinned-mcmc)
14. [Phase 13 — Wind Normalization Constants](#phase-13--wind-normalization-constants)
15. [Phase 14 — Pooled Direct-Model MCMC](#phase-14--pooled-direct-model-mcmc)
16. [Phase 15 — Speed/Memory Pass & BIC Replaces WAIC/LOO](#phase-15--speedmemory-pass--bic-replaces-waicloo)
17. [Phase 16 — Frozen Parameters & Kepler Mode (`ParamSpec`)](#phase-16--frozen-parameters--kepler-mode-paramspec)
18. [Phase 17 — Per-Sample Phase-Shift Alignment](#phase-17--per-sample-phase-shift-alignment)
19. [Phase 18 — Adaptive Constant-SNR Binning & Grid Removal](#phase-18--adaptive-constant-snr-binning--grid-removal)
20. [Phase 19 — Gaussian Phase Smoothing, Scattered Flux & Residual Panels](#phase-19--gaussian-phase-smoothing-scattered-flux--residual-panels)
21. [Phase 20 — Remove Multiplicative Flux Scale from the Single-Model χ² Fit](#phase-20--remove-multiplicative-flux-scale-from-the-single-model-χ-fit)
22. [Phase 21 — MCMC Scatter-Path Audit](#phase-21--mcmc-scatter-path-audit)
23. [Phase 22 — Adaptive Binning in the Single-Model CLI](#phase-22--adaptive-binning-in-the-single-model-cli)
24. [Phase 23 — `utils/` Extraction and a Single Plotting Routine](#phase-23--utils-extraction-and-a-single-plotting-routine)
25. [Phase 24 — Run-Config Persistence and a Replot-Mode Fix](#phase-24--run-config-persistence-and-a-replot-mode-fix)
26. [Phase 25 — MCMC Script Slimming](#phase-25--mcmc-script-slimming)
27. [Phase 26 — Binary-Geometry Diagnostic Plots](#phase-26--binary-geometry-diagnostic-plots)
28. [Phase 27 — Standard Inclination Convention at the Public API](#phase-27--standard-inclination-convention-at-the-public-api)
29. [Phase 28 — Physical Wind Normalization & Per-Cell Flux Conversion](#phase-28--physical-wind-normalization--per-cell-flux-conversion)
30. [Phase 29 — Mass Reparameterization & Error-Column Fix](#phase-29--mass-reparameterization--error-column-fix)
31. [Phase 30 — Physical Norm in the Single-Model CLI, Model-LC Dump & χ²_eff Fix](#phase-30--physical-norm-in-the-single-model-cli-model-lc-dump--χ_eff-fix)
32. [Phase 31 — Release Trim: `lam`, Wind Models and Flux Methods](#phase-31--release-trim-lam-wind-models-and-flux-methods)
33. [Phase 32 — `beta_law` Wind Profile Restored](#phase-32--beta_law-wind-profile-restored)
34. [Side Investigation — Reference Epoch Recalibration](#side-investigation--reference-epoch-recalibration)
35. [Current File Inventory](#current-file-inventory)
36. [Current Status & Quick Commands](#current-status--quick-commands)

---

## Project Overview

**Target:** IC 10 X-1 — eclipsing X-ray binary in the Local Group galaxy IC 10.

**System (working values):** WR companion R ≈ 2 R☉, accretion-disk r ≈ 0.001 R☉,
inclination i₀ ≈ 26°, separation d ≈ 19 R☉.

**Best spectral model:** TBabs × powerlaw — nH ≈ 0.75×10²² cm⁻², Γ ≈ 1.86,
χ²_red ≈ 1.52 (preferred over phabs by Δχ² ≈ 8.5).

---

## Phase 1 — R → Python Migration

Ported `new11.R`, `grid4.R`, `wind_los2.R`, `density_fnc.R` (≈260 R lines)
to a single `xrb_lightcurve.py` with full type hints, an argparse CLI, and
fully vectorized NumPy core (10–100× faster than R). Public surface:
`simulate_lightcurve`, `create_grid`, `wind_los_integral`, `density_function`.
Companion files: `example_usage.py`, `plot_results.py`, `requirements.txt`.

---

## Phase 2 — Flux Integration & XSPEC Support

Added three flux-conversion paths selectable via `--flux_method`:
- `legacy` — hardcoded exponential fits.
- `interpolate` — log-log interpolation from an XSPEC `flux vs nH` CSV.
- `refit` — re-fit exponentials to the XSPEC table.

Helpers: `load_flux_vs_nh_csv`, `interpolate_flux_from_nh`,
`fit_exponential_to_csv`. CLI gained `--flux_csv` and `--lam`
(target mean nH; `--lam2` retired in Phase 7). XSPEC table is generated by
`compute_flux_vs_nH.py --specdir … --out_csv data_flux_vs_nH.csv`.

Units convention: `flx` (atoms / R☉⁴, raw column-density integral),
`fl` (10²² cm⁻²), `nfl_{band}` (photons/cm²/s).

---

## Phase 3 — Light-Curve Data Pipeline

Chandra `.txt` light curves were actually FITS binary; built a small
toolchain to (a) convert FITS→TXT and (b) attach a calibrated FLUX column.
Tools (in `utils/`): `convert_fits_to_txt.py` and the heasoft wrapper
`convert_fits_to_txt_heasoft.sh`, `add_flux_simple.py`,
`add_flux_to_lightcurves.py`, `compute_count_to_flux_factor.py`,
`get_average_count_rates.py`. Output layout:
`data/IC_10_X1_LC/{Broad,Soft,Hard}{_converted,_with_flux}/`.

Time-averaged count rates (cts/s): broad 0.1132, soft 0.0635, hard 0.0497.

---

## Phase 4 — Spectral Model Analysis

XSPEC tooling: `xspec_get_conversion_factors_tbabs.xcm`,
`get_conversion_factors.sh`, `get_xspec_nH.py`,
`compare_absorption_models.xcm`, `compare_models.sh`.
Comparison run picked **TBabs × powerlaw** over phabs × powerlaw
(Δχ² = 8.54 in favor of TBabs; nH 0.75 vs 0.78). 0.5–7 keV model fluxes:
TBabs 1.032×10⁻¹², phabs 1.031×10⁻¹² erg/cm²/s.

---

## Phase 5 — Phase Folding & χ² Fitting

`chandra_phase_analysis.py` (≈1.0k lines) folds observations onto orbital
phase using the ephemeris (`REF_EPOCH = 278801348 s`, `P = 125431 s`) and
fits simulation models via χ² minimization with optional phase-shift /
scale rescaling. Auto-detects flux columns (`nfl_*`, `pho_count_*`),
handles both standard whitespace and CIAO `#Columns:` formats, supports
multi-column simultaneous fitting.

---

## Phase 6 — Eclipse Geometry & LOS Cutoff Fixes

Two physics fixes plus a runtime knob in `xrb_lightcurve.py`:

1. **Front-emitter spurious eclipse** — eclipse gating moved from
   `gma < π` to `sin(gma) > 0` so emitter-in-front geometries are not
   incorrectly occulted.
2. **Total-eclipse flux** — `is_eclipsed` flag added; during eclipse all
   `nfl_*` and `pho_count_*` columns are forced to 0 (previously they
   collapsed to *maximum* flux because `flx=0` → `e⁰=1`).
3. **`--Rmax` and `--converge-rmax`** — configurable LOS cutoff. Converged
   mode integrates to −∞ using a closed form for the constant-velocity
   term plus a cached quadrature + asymptotic tail for the accelerating-wind
   term, comparable in speed to the legacy `Rmax = 2d` heuristic.
   *(The cached `_ACCEL_*` table was retired in Phase 7 once a single Numba
   kernel covered all profiles.)*

---

## Phase 7 — Unified Wind Model

Plan: `mcmc_wind_shape_params_8b9c89d2.plan.md` *(precursor)*,
implementation under `unified_wind_model_77726ced.plan.md`.

Goal was to drop the hardcoded **AV / CV** wind duality (`flx2`, `lam2`,
`*_cv` columns) in favour of a pluggable density profile selected at runtime.

**New profile registry** in `xrb_lightcurve.py` (each exposes a dimensionless
`g(r)` consumed inline by a single Numba kernel):

| `wind_model`   | id | Free shape params              | Notes                                      |
| -------------- | -- | ------------------------------ | ------------------------------------------ |
| `broken_pl`    | 0  | `Rb, p`                        | Piecewise PL (kept for back-compat).       |
| `smooth_pl`    | 1  | `Rb, p, Delta`                 | Default; smoothly broken PL.               |
| `beta_law`     | 2  | `R_star, beta, H`              | CAK velocity profile, `g = 1/(r²·v(r))`.   |
| `confinement`  | 3  | `R_star, fconf, ell`           | `1/r²` with inner exp. compression.        |

**Kernel rewrite** — `_wind_los_profile_numba` (and a Gauss-Legendre variant
`_los_gl_quadrature`) replaces the old converged + fixed-Rmax pair, and
`_simulate_phases_numba` runs the full per-phase sweep under
`@njit(parallel=True, prange)`. Cached `_ACCEL_U_GRID/_ACCEL_F_GRID`
deleted.

**API changes:**
- `simulate_lightcurve(...)` lost `lam2` and gained
  `wind_model: str = "smooth_pl"` and `wind_params: dict | None`.
- Output collapsed to a single `flx`/`fl` plus one `nfl_{band}` column per
  band (no more `_av`/`_cv` suffix).
- New helpers: `pack_wind_params`, `default_wind_params`, `evaluate_g_profile`,
  `compute_surface_density(sim_df, lam, R_star, wind_model, wind_params)`,
  and `wind_density_posterior(...)` for converting MCMC `lam` posteriors
  into surface-density `n₀` posteriors.
- CLI: removed `--lam2`; added `--wind-model {broken_pl, smooth_pl, beta_law,
  confinement}` plus per-model `--Rb / --p / --Delta / --beta / --H /
  --fconf / --ell`.

**Downstream cleanup** — `plot_results.py`, `chandra_phase_analysis.py`,
`chandra_analysis_combined_flux.py`, `utils/test_flux_methods.py` all
updated to look for `nfl_{band}` (no `_av`/`_cv` filter).

---

## Phase 8 — MCMC Pipeline & Performance

Plan: `mcmc_performance_and_statistics_8989cd39.plan.md`.

`mcmc_lightcurve_fit.py` (≈3.1k lines) wraps `xrb_lightcurve.py` in a full
emcee/zeus pipeline. Headline pieces:

- **Forward-model paths**
  - `PrecomputedModelGrid` — geometry-only 6-D grid (`d1, d2, r, R, i0, phase`)
    with vectorized `RegularGridInterpolator` lookup. Multi-process build via
    `_compute_single_model` + `Pool`. Supports `--save-grid` / `--load-grid`.
  - `DirectLightCurveModel` — wraps `simulate_lightcurve` directly; required
    when shape params are sampled (Phase 10).
- **Samplers** — `emcee` (default, stretch move) and `zeus` (ensemble slice)
  via the same harness.
- **Likelihoods** (`--likelihood`) — `chi2` (Gaussian, default), `jitter`
  (Gaussian + free `log_f` systematic term), `studentt` (heavy-tailed,
  configurable `--studentt-nu`). Cash/Poisson was scoped out.
  *(`PrecomputedModelGrid` was deleted in Phase 18, along with `studentt`;
  only `chi2` and `jitter` remain.)*
- **Diagnostics** — ArviZ summary (`r_hat`, `mcse_*`, `eti89_*`),
  autocorrelation, optional WAIC/LOO via `--compute-waic`, corner plots,
  best-fit overlay with reduced χ².
  *(WAIC/LOO replaced by BIC in Phase 15; `--compute-waic` removed in Phase 18.)*

**Performance work** done in `xrb_lightcurve.py`:

- **Pre-optimization state (post Phase 7).** Introducing the unified wind
  profile registry dropped per-LOS Python dispatch into the hot loop: the
  LOS integral called back into Python for `g(r)` at every angular cell ×
  every step on the `dz` grid × every phase. Combined with the fixed-step
  trapezoid quadrature and per-cell eclipse checks, a *single*
  `simulate_lightcurve` call had ballooned to **several seconds** (≈3–8 s
  depending on `dz` / `d2h`), which made Phase 8-style MCMC
  (10⁴–10⁵ model evaluations) completely infeasible without a precomputed
  grid — and even the grid build was painfully slow.
- **Fixes** (together ≈ 2 orders of magnitude):
  1. `@njit(cache=True[, parallel=True])` on `_g_profile`, the LOS
     integrand, `_los_gl_quadrature`, `_wind_los_profile_numba`, and the
     full per-phase sweep `_simulate_phases_numba` (using `prange`).
  2. **Gauss-Legendre quadrature** replaces the old fixed-step trapezoid;
     far fewer integrand evaluations for the same accuracy, and the
     GL nodes/weights are precomputed once.
  3. **Inlined eclipse test** — eclipse gating collapsed into the kernel
     so eclipsed phases early-exit without a Python round-trip.
  4. All `wind_params` / sim params flattened to scalar JIT arguments at
     the Python/Numba boundary; no dict lookups in the hot path.
- **Result:** a single `simulate_lightcurve` call is now **< 1 s
  (≈ 63 ms)** on a laptop, i.e. ~50–100× faster than the post-Phase-7
  regression and 10⁵–10⁶× the sustained rate needed for MCMC. The
  direct evaluator became viable for MCMC without a precomputed grid,
  and the grid build itself dropped from minutes to seconds.

GPU acceleration was evaluated and rejected (Intel Iris ≠ CUDA / JAX-Metal /
PyTorch-MPS targets). NumPyro/NUTS port was scoped out for the same reason
plus the non-differentiable interpolator in the fast path.

---

## Phase 9 — Convergence Improvements (Reparameterization)

Plan: `mcmc_convergence_improvements_c648afb3.plan.md`.

`(d1, d2)` are strongly correlated because the wind absorption sees only
their sum. `--reparam` swaps the sampling space to:

- `a = d1 + d2` (separation, well-constrained)
- `q = d1 / (d1 + d2)` (ratio, weakly constrained)

Implementation lives entirely in `mcmc_lightcurve_fit.py`:
`REPARAM_PRIORS`, `get_param_config(reparam=True)` returns
`['a','q','r','R','i0']`, `_evaluate_model` inverts `(a,q) → (d1,d2)`,
`log_prior` applies priors in `(a,q)` space with the `+log(a)` Jacobian,
walker init / corner / summaries report derived `d1, d2`. Grid build
itself is unchanged (still indexed in physical `d1, d2`).

HMC / NUTS was assessed and **not pursued** — piecewise-linear
`RegularGridInterpolator` gradients, hard eclipse branches, and Numba
kernels all break autodiff. Future paths (smooth GP/NN emulator, cubic grid
+ finite differences, JAX rewrite) are documented in the plan.

---

## Phase 10 — Wind-Shape MCMC Parameters

Plan: `mcmc_wind_shape_params_8b9c89d2.plan.md`.

After Phase 8 brought the direct evaluator to ~63 ms/LC, wind-shape
parameters are now first-class MCMC dimensions, gated by `--fit-wind-shape`.

**Per-model active set** (in `WIND_SHAPE_FIT`):

| `--wind-model` | Free       | Fixed                  | Tied to geometry |
| -------------- | ---------- | ---------------------- | ---------------- |
| `smooth_pl`    | `Rb, p`    | `Delta = 2.0`          | —                |
| `beta_law`     | `beta`     | `H = 1.0`              | `R_star = R`     |
| `confinement`  | `fconf, ell` | —                    | `R_star = R`     |

`broken_pl` is intentionally skipped (`smooth_pl` generalizes it).
`lam` (overall normalization) stays fixed from spectral fits — shape
params are constrained only by the LC shape.

**Mechanics in `mcmc_lightcurve_fit.py`:**
- New registries: `WIND_MODELS`, `WIND_SHAPE_FIT`, `WIND_SHAPE_FIXED`,
  `WIND_SHAPE_LABELS`, `WIND_SHAPE_PRIORS`, plus helpers
  `get_active_priors(...)` and `_to_wind_params(theta, active_names,
  wind_model, R_value, fit_wind_shape)`.
- `get_param_config` and walker init are now fully dynamic in the
  `active_names` list (no more hardcoded `n_phys=5` / `theta[5]` /
  `pos[:,5]`; the jitter `log_f` index is looked up by name).
- `log_prior`, `log_likelihood_*`, `_evaluate_model`,
  `compute_chi2_for_samples`, `compute_pointwise_loglik`, `plot_best_fit`,
  `run_arviz_diagnostics`, `run_single_fit`, and `replot_from_existing`
  all accept `wind_model`, `fit_wind_shape`, `active_names` and route the
  wind-shape sample correctly.
- `load_existing_results` accepts any column set (geometry columns
  required, extras kept) and returns `(samples, stats, loaded_names)` so
  re-plots work for both geometry-only and shape-fit chains.
- `all_results` keys switched from `f"{band}_{wind_model}"` to
  `(band, wind_model)` tuples (the underscore in `smooth_pl` etc. broke
  the previous `rsplit('_', 1)` summary writer).

**CLI additions:**
- `--wind-model {smooth_pl, beta_law, confinement}` (default `smooth_pl`).
  The legacy `av/cv/both` choices and the `'both'` loop are gone.
- `--fit-wind-shape` — adds the wind model's active shape params to the
  MCMC vector. Works with either the precomputed grid (see Phase 11) or
  `--no-grid` (direct evaluator).
- Per-shape prior overrides: `--prior-Rb / -p / -beta / -fconf / -ell`
  using the same `mean,std,min,max` format as the geometry priors.
- `--lam2` removed everywhere.

**Smoke verified** — 50-step chains for `smooth_pl` (geom-only and
`+Rb,p`), `beta_law` (`+beta`), `confinement` (`+fconf, ell`), plus a
grid-path geometry-only run. All produce valid corner data, ArviZ
summaries, samples CSV, and `mcmc_summary.txt`.

---

## Phase 11 — N-D Precomputed Grid for Shape-Fit MCMC

> **Superseded:** the entire precomputed-grid path described below was deleted in
> [Phase 18](#phase-18--adaptive-constant-snr-binning--grid-removal). MCMC now
> always uses the direct evaluator. Kept here for history.

After Phase 10 a short (`~1k-step`) shape-fit chain still took ~2 h
because `--fit-wind-shape` auto-forced the direct evaluator
(≈ 63 ms × 32 walkers × 1000 steps × all phases ≈ hours). The grid
class was extended to cover wind-shape axes dynamically instead of
forcing a slow path.

- **`PrecomputedModelGrid` is now N-D.** `self.axis_names` holds the
  ordered list of axes (geometry first, then the active shape axes for
  the chosen `wind_model`). `self.param_grids` and `self.flux_grid`
  extend correspondingly (geometry-only grids are unchanged shape-wise,
  so the refactor is zero-cost for existing workflows).
- **Worker path.** `_precompute_models` now iterates via `np.ndindex`
  over the full axis set, resolves a per-combo `wind_params` dict
  (fixed template + per-point shape values + `R_star = R` when the
  model ties it), and hands it to `_compute_single_model`. The
  `r >= R` geometry filter still applies.
- **Evaluation.** A single `RegularGridInterpolator` spans
  `(axes…, phase)`; `evaluate()` looks shape values up from the caller's
  `wind_params` dict when shape axes are present, and falls back to the
  fixed template for geometry-only grids.
- **I/O** — `.npz` files persist `axis_names`, `shape_axes`,
  `fit_wind_shape`, and one `<name>_grid` array per axis. Legacy
  geometry-only `.npz` files (missing `axis_names`) still load unchanged
  via a backward-compat branch in `_load_grid`.
- **CLI** — `--fit-wind-shape` no longer force-disables the grid;
  instead:
  - Grid size expands by `(shape_grid_points)^k` for `k` shape axes, with
    the new `--shape-grid-points N` knob (default 5). A `[notice]`
    prints the expected total grid size up front.
  - `--save-grid` / `--load-grid` work as usual and are now actually
    useful for shape-fit MCMC (build once, run many chains).
  - `--no-grid` stays as the escape hatch for short debugging chains.
- **Logging cleanup.** `load_flux_vs_nh_csv(..., verbose=verbose)` and
  `interpolate_flux_from_nh(..., warn_extrapolation=verbose)` now honor
  the `verbose` flag that `simulate_lightcurve` already threads through.
  MCMC calls with `verbose=False`, so the per-call "Detected energy
  bands in CSV: …" print and repeated "nH outside CSV range /
  Extrapolation will be used" `UserWarning` are gone (they only fire in
  notebooks / interactive runs now). A belt-and-suspenders
  `warnings.filterwarnings(...)` in `mcmc_lightcurve_fit.py` keeps the
  extrapolation warning quiet even if a future callsite forgets to pass
  `verbose=False`.
- **Smoke verified** — N-D grid build + save/load round-trip (bit-exact),
  end-to-end MCMC with `--fit-wind-shape` on `smooth_pl` (+ `Rb, p`) and
  `beta_law` (+ `beta`), `--no-grid` shape-fit path, and legacy
  geometry-only grid `.npz` backward compatibility all pass.

---

## Phase 12 — flux_t Errors & First-Class Unbinned MCMC

Plan: `flux_t_error_and_unbinned_mcmc_1bf81aa9.plan.md`. Commit `fa672e3`.

CIAO files carry `flux_t` but **no** `flux_t_err`, so the MCMC's
`error_column = obs_column + "_ERR"` guess (`flux_t_ERR`) never matched and
silently fell back to `0.1·|flux|`.

- **`chandra_phase_analysis.py`** — new `_derive_err_from_rate_err(df, obs_col)`,
  invoked only when no error column is matched. Derives per-row
  `err = rate_err · (obs/rate)` for `rate > 0`, falling back to a file-level
  `cf = median(obs/rate)` for the rest (verified constant per file, e.g.
  `cf ≈ 4.56e-12` for soft 11080).
- **Master-file code removed** — `verify_master_contains_individual()`, the
  `master_file` parameter in `load_data`/`read_observation`, and the
  `--master-file` / `--verify-master` CLI args are gone.
- **`mcmc_lightcurve_fit.py`** — passes `obs_error_column=None` when the user
  didn't set it explicitly so auto-derivation fires; keeps the `flux > 0` drop in
  *both* binned and unbinned modes (zero-flux rows are GTI gaps, and with jitter
  `σ²_eff = σ_obs² + (f·model)² ≈ 0` would blow up `log σ²`), and prints the drop
  count. The invalid-error patch tightened from `0.1·|flux|` to
  `max(0.1·|flux|, median(valid obs_err))`.
- **`is_binned`** threaded from `main()` through `run_single_fit` /
  `replot_from_existing` / `plot_best_fit` so labels and marker style match
  reality ("Observed (phase-binned)" with error bars vs "Observed (raw 100s)" as
  translucent scatter).
- **Jitter-aware χ²** — `plot_best_fit` and `compute_chi2_for_samples` report the
  effective-variance χ² alongside the classical measurement-error χ² when
  `--likelihood jitter` is active.
- `--no-phase-bin` help now recommends pairing with `--likelihood jitter`, and
  the module docstring gained an unbinned `flux_t` example.

---

## Phase 13 — Wind Normalization Constants

Plan: `wind_normalization_constants_76f1ef51.plan.md`. Commit `fa672e3`.

Since every profile is coded as a dimensionless `g(r)`, the physical amplitude has
to be recovered after the fit. Added to `xrb_lightcurve.py`:

- Constants `M_H_G`, `M_SUN_G`, `KM_TO_CM` alongside the existing `R_SUN_CM`.
- **`compute_wind_normalization_constants(lam, flx_mean, wind_model, wind_params,
  v_inf=None, mu=1.4)`** — computes `n0 = lam·1e22/(R_sun·flx_mean)` then, per
  model: `smooth_pl` → `g_break`, `n_break_cm3`, `rho_b_g_cm3`;
  `beta_law`/`confinement` → `n_surface_cm3`, `rho_surface_g_cm3`,
  `mdot_over_vinf_g_per_cm`, plus `mdot_g_s` / `mdot_msun_yr` when `v_inf`
  (km/s) is supplied. `v_inf` is not fitted — it was absorbed into `n0` by the
  `lam` normalization — so the caller must provide it for a mass-loss rate.
- **`wind_normalization_constants_posterior(...)`** — mirrors
  `wind_density_posterior`, looping the point estimator over posterior samples
  and returning `{samples, median, p16, p84}` per constant.

---

## Phase 14 — Pooled Direct-Model MCMC

Commit `36c8c8b`.

The direct evaluator's `simulate_lightcurve` is already Numba
`parallel=True`, so naively adding `--n-threads` worker processes caused severe
thread oversubscription (workers × numba threads ≫ cores), while pinning workers
to one thread made each LC so slow that process parallelism barely beat serial.

- `_init_numba_worker(max_numba_threads)` is used as the `Pool` initializer and
  calls `numba.set_num_threads(...)` inside each worker.
- `--numba-threads-per-worker` exposes the knob; the default `auto` is
  `max(1, cpu_count // n_threads)`, so workers collectively use ≈ one thread per
  logical CPU.
- The pool uses the `spawn` context (safest cross-platform, avoids inheriting
  heavy state) and is only enabled for `DirectLightCurveModel`; requesting
  `--n-threads > 1` with any other model type prints a `[notice]` and runs
  serial.
- Also fixed an indentation bug in `xrb_lightcurve.py`.

---

## Phase 15 — Speed/Memory Pass & BIC Replaces WAIC/LOO

Plan: `mcmc_speed_memory_optimization_68fc497a.plan.md`. Commit `7116302`.

A staged, accuracy-preserving optimization program with a reproducible benchmark
harness (`utils/benchmark_mcmc_performance.py`, documented in
`PERFORMANCE_VALIDATION_REPORT.md`; acceptance thresholds: parameter median drift
< 0.1σ, reduced-χ² change ≤ 2%, BIC ranking consistency).

**Runtime (`xrb_lightcurve.py`):**
- Module-level `_FLUX_CACHE` keyed by `(abs csv path, flux_type)` caching cleaned
  arrays, prebuilt log-log `interp1d` objects, and per-band exponential fits
  (`_build_flux_context`, `_interpolate_flux_from_context`). Previously every
  `simulate_lightcurve` call re-read and re-sorted the CSV.
- Mega-kernel results assembled straight from NumPy arrays into the DataFrame
  (no `tolist()`/`zip()` round-trips).
- Theta-ring trig tables precomputed once per `_simulate_phases_numba` call
  instead of per phase.

**Memory (`mcmc_lightcurve_fit.py`):** grid precompute streamed worker results
directly into `flux_grid` instead of materializing a full `list(...)`, and
interpolator setup avoided whole-grid copies during NaN cleanup. *(Both became
moot once the grid path was deleted in Phase 18.)*

**Hot path:** likelihood invariants precomputed once in `run_mcmc` and passed as
a `like_terms` dict (`obs_err2`, `jitter_logf_index`), replacing repeated
`active_names.index('log_f')` lookups and per-call `obs_err**2` allocations.
`_interp_periodic_phases` gained a monotonic fast path that skips the sort.

**Output:** `save_samples_csv_chunked` writes samples in configurable chunks
(`--csv-chunk-size`, default 50000); `--compact-output` adds an NPZ companion;
`--no-csv-output` skips the large CSV entirely.

**Model comparison:** WAIC/LOO removed in favor of BIC. `compute_bic_metrics`
computes `BIC = k·ln n - 2 ln L̂` with `k = len(active_names)`,
`n = len(obs_flux)` after all binning/filtering, and `L̂` obtained by calling the
run's own likelihood at the max-log-prob sample (not by subtracting priors from
`log_prob`, avoiding Jacobian bookkeeping). Reports `bic`, `logL_hat`,
`k_params`, `n_obs`, `theta_source` (`map_log_prob` | `median_fallback`) to the
console, `mcmc_summary.txt`, and `*_model_metrics.csv`; `ΔBIC` is computed
against the best model in the run. ArviZ is now used for convergence
diagnostics only.

---

## Phase 16 — Frozen Parameters & Kepler Mode (`ParamSpec`)

Plan: `freeze_params_and_kepler_3368d554.plan.md`. Commit `be56359`.

Two features plus the refactor that made both tractable.

**`ParamSpec` dataclass** (built once in `main()` by `build_param_spec(...)` and
threaded through every consumer) replaces ad-hoc positional `theta` indexing:
`mode`, `active_names`, `active_labels`, `frozen`, `fit_wind_shape`,
`fit_scatter`, `wind_model`, `likelihood`, `orbital_period_s`, `K_kepler`.
Central resolvers by name, not index:
- `_resolve_geom(theta, spec)` (replaces `_to_physical`, kept as a shim),
- `_resolve_shape(theta, spec)` (generalizes `_to_wind_params`),
- `_theta_value(theta, name, names, frozen)`.

**`--freeze NAME=VAL[,NAME=VAL,…]`** — pins parameters and drops them from the
chain. Valid: `d1, d2, a, q, r, R, i0, M_X, M_RH, f_scatter, Rb, p, beta, fconf,
ell`. Shape params can be frozen even without `--fit-wind-shape`. `log_f`
cannot be frozen. Unknown names are rejected with the allowed list; values
outside the prior box warn but proceed; `Rb < R` with both frozen fails fast
before sampling. `get_active_priors` drops frozen entries.

**`--kepler`** — samples `(M_X, M_RH)` in M☉ with `KEPLER_PARAM_NAMES` /
`KEPLER_PARAM_LABELS` / `KEPLER_PRIORS`, then derives
`a = K·M_tot^{1/3}` (with `K = (G·M☉·P²/4π²)^{1/3}/R☉` precomputed by
`_compute_kepler_prefactor`) and `q = M_RH/M_tot` from the lever arm
`d1·M_X = d2·M_RH`. `--orbital-period` sets `P` (default `ORBITAL_PERIOD`);
`--prior-MX` / `--prior-MRH` override the mass priors. Mutually exclusive with
`--reparam`. Composes with freezing (e.g. `--kepler --freeze M_RH=20`).

**Prior rewrite** — `log_prior` iterates `spec.active_names`, applies box +
Gaussian per dim, then enforces constraints on *resolved* values so they hold
under freezing and Kepler mode: `r < R` (was a positional `theta[2] >= theta[3]`)
and `Rb ≥ R` for `smooth_pl`. `_log_jacobian` contributes `+log(a)` only in
`reparam` mode (Kepler priors are already in mass space).

**Stats / persistence** — `compute_statistics` derives `(a, q, d1, d2)` in Kepler
mode and `(d1, d2)` in reparam mode, and records a MAP entry (max-log-prob single
sample) which — unlike the marginal medians — exactly satisfies `d1+d2 = a` and
`d1/(d1+d2) = q`. `*_chain.npz` gained `mode`, `frozen_names`, `frozen_values`,
`orbital_period_s`; `replot_from_existing` reads them back to rebuild the spec,
with old chains defaulting to previous behavior. `plot_best_fit` gained
`_value_from_stats_or_frozen` so frozen parameters no longer break the overlay.

Also in this commit: `xrb_lightcurve.py --Delta` default changed from `2.0` to
`1.0`. Note this was *not* mirrored in `default_wind_params` or
`WIND_SHAPE_FIXED`, which still use `2.0` — see PROJECT.md "Known rough edges".

---

## Phase 17 — Per-Sample Phase-Shift Alignment

Commit `67448c3`.

Rather than trusting the ephemeris to align model and data, every likelihood call
now minimizes weighted χ² over a phase shift, so the fit is insensitive to
residual epoch error.

- `_build_phase_shift_terms(enabled, obs_phase, grid_size, eval_points,
  refine_points)` precomputes the coarse shift grid, the dense model evaluation
  grid, and the shifted observation-phase matrix once per run.
- `_apply_best_phase_shift(...)` does a two-stage search: a coarse uniform grid
  over `[0,1)`, then a local refinement across ±1 coarse step around the best
  shift — near-fine-grid accuracy at a fraction of the cost. The model is
  evaluated once on the dense grid and re-interpolated per trial shift.
- Defaults: `DEFAULT_PHASE_SHIFT_GRID_SIZE = 25`,
  `DEFAULT_PHASE_SHIFT_EVAL_POINTS = 240`,
  `DEFAULT_PHASE_SHIFT_REFINE_POINTS = 9`. CLI:
  `--no-fit-phase-shift`, `--phase-shift-grid-size`,
  `--phase-shift-eval-points`.
- Because the shift is a per-sample nuisance minimization rather than a sampled
  parameter, it is applied consistently in `log_likelihood_chi2`,
  `log_likelihood_jitter`, `compute_chi2_for_samples`,
  `compute_pointwise_loglik`, `compute_bic_metrics`, and `plot_best_fit` (which
  also draws the model shifted by the best-fit value and prints it in the
  annotation box).
- `mcmc_summary.txt` gained a "Run configuration" block recording
  `fit_phase_shift`, grid size, eval points, and wall time
  (`stats['_run_meta']`), plus a chain-diagnostics block from
  `print_diagnostics` (`stats['_diagnostics']`).

---

## Phase 18 — Adaptive Constant-SNR Binning & Grid Removal

Plan: `adaptive_constant-snr_binning_249c8cba.plan.md`. Commit `b376585`.

**Constant-counts binning.** `counts` is now carried through
`read_observation` / `load_data` (`counts_column='counts'`) and
`load_observed_lightcurves`. New `phase_bin_data_snr(df, counts_per_bin=100, …)`
in `chandra_phase_analysis.py` (with a flux-naming wrapper in the MCMC module)
sorts by phase and greedily accumulates `counts` until each bin reaches the
target, so every binned point carries roughly equal Poisson weight
(100 counts ⇒ SNR ≈ 10) and low-signal eclipse troughs merge into wide bins
instead of many noisy narrow ones. Per bin it returns the counts-weighted phase
center, inverse-variance weighted flux, `error = √(1/Σw)`, `n_points`,
`total_counts`, and `phase_lo`/`phase_hi`/`width`; a trailing under-target bin is
merged into its predecessor.

Mode is selected by argument *presence*, not a `--bin-mode` flag:
`--no-phase-bin` (raw) > `--counts-per-bin N` (adaptive) > `--n-phase-bins N`
(fixed) > neither (50 fixed bins, backward compatible). `--n-phase-bins` default
changed from `50` to `None`; supplying both binners is an error. Bin widths are
threaded as `obs_phase_width` into `plot_best_fit` and drawn as horizontal error
bars.

**Precomputed grid deleted.** `PrecomputedModelGrid` and its
`_precompute_models` / `_compute_single_model` / `_setup_interpolators` /
`_save_grid` / `_load_grid` machinery are gone, along with `--save-grid`,
`--load-grid`, `--no-grid`, `--grid-points`, and `--shape-grid-points`. MCMC
always uses `DirectLightCurveModel`: at ~60 ms/LC it is fast enough, it avoids
grid interpolation artifacts, and it is the only path that keeps the likelihood
physically faithful for every sample (including per-step wind shape).

**Likelihood CLI simplified.** Student-t and the deprecated WAIC shim removed:
`log_likelihood_studentt()`, `'studentt'` from `LIKELIHOOD_TYPES` and
`--likelihood` choices, `--studentt-nu`, all `studentt_nu` plumbing through
`log_probability` / `run_mcmc` / `compute_chi2_for_samples` / BIC /
`run_single_fit` / save / replot, the `scipy.special.gammaln` import, and
`--compute-waic` plus its shim mapping to `--compute-bic`. Only `chi2` and
`jitter` remain; `--compute-bic` is the model-comparison flag.

---

## Phase 19 — Gaussian Phase Smoothing, Scattered Flux & Residual Panels

Plan: `gaussian_phase_smoothing_reference_d1a46172.plan.md`.
**Status: implemented but uncommitted** on branch `add_generic_wind`.

Three reusable primitives live in `chandra_phase_analysis.py` so both the
single-model and MCMC plot paths share one source of truth:

1. **`smooth_lightcurve(phase, flux, flux_err, sigma=0.01, n_eval=300,
   n_mc=2000, random_state=None)`** — periodic Gaussian-kernel phase smoother.
   Periodic distance `d = |((φ_i-φ_eval+0.5) mod 1) - 0.5|`, weights
   `exp(-½(d/σ)²)`, so it is continuous across `phase = 0/1`. Generalizes the
   MATLAB reference in `temp/LC_MC/*.m` from index windows to a phase-distance
   kernel, so it is correct for fixed-width bins, constant-SNR bins, *and* raw
   unbinned data. The 1σ band is a vectorized Monte Carlo (perturb all points at
   once, one matmul, `std` over realizations) rather than a Python loop. Kernel
   is phase-distance only — no inverse-variance weighting — matching the
   reference. `σ = 0.01` sits well below the ~0.1–0.25 phase scale of real
   features and above the ~0.0002 raw sampling.
2. **`estimate_scattered_flux(phase, flux, window=(0.4, 0.6))`** — mean observed
   flux in the mid-eclipse window (fallback `0.1 × median`, clamped ≥ 0).
3. **`add_residual_panel(ax, phase, obs, model, err, xerr=None)`** — normalized
   pulls `(O-M)/σ` with `0`/`±1` reference lines.

**`xrb_lightcurve.py`** — `simulate_lightcurve(scattered_flux=0.0)` adds a
constant, phase-independent offset to every `nfl_*` column after eclipse
handling, so notebooks can bake a scattered-light floor into a directly generated
model. The fit paths deliberately add scatter at overlay/evaluation time instead,
so `fit_simulation`'s multiplicative `scale` and MCMC's per-step scaling don't
rescale an additive constant.

**`chandra_phase_analysis.py` single-model path** — `fit_simulation(scatter=…)`
adds the constant after scaling inside the inner χ²; `plot_phase` becomes a
2-panel figure (3:1 heights, shared x) with a residual panel whenever a model
overlay is present, plus the optional dashed-green smoothed curve and MC band.
New CLI: `--smooth`, `--smooth-sigma`, `--smooth-n-mc`, `--smooth-seed`,
`--scatter`, `--scatter-eclipse-phase`. When `--fit` runs without `--scatter`,
the value is estimated from the eclipse window.

**`mcmc_lightcurve_fit.py`** — imports the three primitives. Same
`--smooth*` flags, computed once per band and threaded into `plot_best_fit`,
which is now a 2-panel figure with the residual panel clipped to ±5σ. New
`--fit-scatter` promotes `f_scatter` to a free MCMC parameter (mirroring
`log_f`): added by `build_param_spec` / `ParamSpec.fit_scatter`, resolved by
`_resolve_scatter` and applied in `_evaluate_model` so all likelihoods, BIC, and
pointwise log-lik inherit it. Its prior is centered on
`estimate_scattered_flux(...)` with `min = 0` and `max = nanmax(obs_flux)`.
Being phase-invariant it is unaffected by the phase-shift search; being physical
it counts toward `dof` automatically (`n_phys` subtracts only `log_f`). It is
saved as a normal active column, excluded from wind-shape extra-dim detection on
replot, and can be pinned via `--fit-scatter --freeze f_scatter=<v>`.

---

## Phase 20 — Remove Multiplicative Flux Scale from the Single-Model χ² Fit

`chandra_phase_analysis.py`. **Status: uncommitted** on branch `add_generic_wind`.

`fit_simulation` used to fit *two* parameters — a phase shift and a
multiplicative flux `scale` — which meant reduced χ² never tested the model's
absolute normalization. That normalization is not free: it is pinned by `lam`
(the orbit-averaged nH from the spectral fit) together with the XSPEC
`flux vs nH` table. A free y-scale therefore absorbed any normalization error
instead of exposing it. The MCMC path never had such a parameter — it fits a
per-sample phase shift and an *additive* `f_scatter`, nothing multiplicative —
so the two χ² paths were also inconsistent with each other.

Note the additive scattered-flux floor is **not** a substitute for the
multiplicative scale (they are different degrees of freedom); the justification
for dropping `scale` is that the normalization is externally fixed, and the
eclipse-floor offset is what the additive term legitimately covers.

- **`fit_simulation(obs_df, sim_df, sim_column, fit_phase_shift=False,
  scatter=0.0, n_shift_grid=1000)`** now returns `(shift, reduced_chi2)` — the
  `scale` element is gone from both the fit and the return tuple. The model is
  `interp((φ_obs - shift) mod 1) + scatter`.
- **Robust shift search.** χ²(shift) is periodic and strongly multi-modal
  because of the eclipse, and the old `Nelder-Mead` started at `shift = 0`
  routinely settled in the wrong basin — a latent defect that mattered more once
  the shift became the only fitted parameter. Replaced with a vectorized coarse
  scan over the full period (default 1000 nodes, one `np.interp` over all
  `(shift, φ_obs)` pairs) followed by a bounded `minimize_scalar` refinement
  within one coarse step. Verified to recover injected shifts of 0.02, 0.37,
  0.51, and 0.95 to < 1e-3.
- **dof corrected.** Was `N - 2` in *both* branches (wrong even before: the
  no-rescale branch fitted nothing). Now `N - 1` when the shift is fitted and
  `N` when it is not.
- **`plot_phase`** lost its `scale` parameter; the overlay is drawn at native
  normalization plus `scatter`. `rescaled=` renamed to `shift_fitted=`, and the
  title annotation now reports the phase shift instead of a rescaled/not label.
- **`plot_multi_column_fits`** — `fit_results` entries are `(shift, chi2)`;
  `rescaled=` → `shift_fitted=`.
- **CLI** — `--rescale` renamed to `--fit-phase-shift`, with `--rescale` kept as
  a deprecated alias on the same `dest` so existing commands keep working.
- **Unused import removed** — `scipy.optimize.minimize` → `minimize_scalar`.
- **Notebook call sites migrated** (10 cells across
  `notebooks/xrb_toy_wind_models.ipynb`, `xrb_model_analysis.ipynb`,
  `xrb_model_analysis_single_15803.ipynb`); `.bak` copies left alongside.

**Effect on reported fit quality.** On the soft band (12 obs, 100 fixed bins,
`sim_flux_tbabs_15803_smooth_pl_soft.csv`) the old two-parameter fit returned
`scale = 0.413` with χ²/dof = 2.06; with the scale removed the same data/model
give χ²/dof = 20.97. The free scale had been hiding a ≈ 59 % flux-normalization
deficit. Expect previously "acceptable" reduced χ² values to rise across the
board — that is the intended behavior, and a coherent non-zero-centered
residual band is now the diagnostic signature of a normalization mismatch (as
opposed to a shape mismatch).

*Not changed:* `chandra_analysis_combined_flux.py` carries its own older
duplicate of `fit_simulation` / `plot_phase` / `plot_multi_column_fits` (no
`scatter` support at all) and still fits a multiplicative scale.

### Follow-up — model/χ²/residual consistency

Two ways the plot could disagree with the number printed on it, both caused by
"model evaluated at the observed phases" being written out three separate times
with three subtly different formulas:

1. **Overlay could omit the scatter floor.** `plot_phase` takes its own
   `scatter` argument, so a caller that passed `scatter=` to `fit_simulation`
   but not to `plot_phase` got a curve drawn `scatter` too low while the title
   showed the χ² that *included* it. `main()` always threaded both, but the
   notebook call sites did not.
2. **The residual panel ignored the phase shift entirely.** It built its
   interpolation grid from the *unshifted* model
   (`model_interp_phase = phase_sorted`) and evaluated
   `np.interp(obs_phase, …)`, whereas χ² uses
   `interp((obs_phase - shift) mod 1, …)`. The plotted *line* was shifted
   correctly, so only the pulls were wrong — and badly: on a synthetic
   `shift = 0.30` case with an otherwise perfect fit (χ²/dof = 0), the panel
   showed residuals implying χ²/dof ≈ 18050, with 50/120 points disagreeing
   with the χ² model.

Fixed by collapsing all three call sites onto one definition:
- **`_prepare_model_interpolator(sim_df, sim_column)`** — builds the
  wrap-around `(phase_wrap, flux_wrap)` arrays, accepting either a `phase` or a
  `deg` column, and no longer mutates the caller's `sim_df`.
- **`_model_from_wrap(phase_wrap, flux_wrap, phases, shift, scatter)`** — the
  single definition of the model. Accepts an array-valued `shift` so the
  coarse-scan batching in `fit_simulation` uses the *same* expression as a
  single evaluation.
- **`evaluate_model_at_phases(sim_df, sim_column, phases, shift, scatter)`** —
  public one-shot wrapper (useful from notebooks).
- **`_obs_errors(obs_df)`** — shared uncertainty extraction (provided errors
  else `sqrt(|rate|)`, with zero/negative/non-finite floored to `1e-3`), so the
  fit and the residual panel weight points identically. Non-finite errors are
  now floored too; previously a NaN error propagated into a NaN χ².
- `plot_phase` draws the overlay on a dense 721-point grid via the shared
  evaluator instead of shifting the model's own sample points, and computes
  residuals with the same `shift` and `scatter`.

**Added a self-check.** When `plot_phase` is given a `chi2` to display, it
recomputes reduced χ² from the curve it actually drew (matching `dof` via
`shift_fitted`) and warns if the two differ by more than 1 %. This catches a
mismatched `scatter`, `shift`, or `sim_column` at the point of display rather
than leaving a plausible-looking number over the wrong curve — it reproduces
both bugs above as warnings. Verified silent on correct calls.

**Notebook fix.** One genuine mismatch existed —
`notebooks/xrb_toy_wind_models.ipynb` cell 8 fitted with
`scatter=1.92174e-13` but called `plot_phase` without it; now passes it. An
audit of all `fit_simulation` → `plot_phase`/`plot_multi_column_fits` pairs
across the four notebooks found no others.

---

## Phase 21 — MCMC Scatter-Path Audit

`mcmc_lightcurve_fit.py`. **Status: uncommitted** on branch `add_generic_wind`.
All tests run under the `henv` conda env (Python 3.13 / numpy 2.2 / emcee 3.1.6
/ arviz 1.0).

Audit of whether the Phase 20 fixes have analogues in the MCMC path. The MCMC
never had a multiplicative flux scale, so nothing to remove there, and its
per-sample phase-shift search (Phase 17) was already the model for the
`chandra_phase_analysis` one. `plot_best_fit` was already fully consistent — it
applies both the phase shift and `f_scatter` to the overlay curve, the
shift-search model, and the residual basis (verified to agree with the
likelihood to 1.7e-16 relative). Three genuine problems in the `--fit-scatter`
path were found and fixed.

**1. `compute_chi2_for_samples` silently dropped `f_scatter`.** It duplicated
the geometry/shape resolution and called `model.evaluate` directly, never
invoking `_resolve_scatter`. So every per-sample χ² written to
`*_chi2.csv.gz` was computed against a model missing the additive floor
whenever `--fit-scatter` was active. Measured bias on a 40-bin broad-band fit:

| `f_scatter` | correct χ² | χ² with floor dropped | error |
| ----------- | ---------- | --------------------- | ----- |
| 0           | 6228.11    | 6228.11               | 0.0 % |
| 1e-13       | 7697.40    | 6228.11               | −19.1 % |
| 3e-13       | 11120.15   | 6228.11               | −44.0 % |
| 1e-12       | 28183.54   | 6228.11               | −77.9 % |

Fixed by routing it through `_evaluate_model` (the same entry point the
likelihood uses) instead of re-implementing the evaluation, which also picks up
wind-shape resolution and any future additions automatically. The now-dead local
`frozen` binding was removed. Verified: all five reporting paths —
`log_likelihood_chi2`, `compute_chi2_for_samples`, `plot_best_fit`,
`compute_bic_metrics`, `compute_pointwise_loglik` — agree to machine precision
with `f_scatter` active.

**2. `--fit-scatter` could not start at all under emcee.** Walker
initialization clipped each dimension to
`min + 0.01*|min| + 1e-12 … max - 0.01*|max| - 1e-12`. That absolute `1e-12`
epsilon is meaningless for a parameter whose natural scale *is* ~1e-13: with
`f_scatter` prior `min = 0`, the lower bound became exactly `1e-12`, roughly 6×
larger than the entire plausible range, so **every walker was clipped to the
same value**. The column had zero variance, `emcee.walkers_independent` returned
False, and the run died with `ValueError: Initial state has a large condition
number`. The inset is now relative to each parameter's own prior span
(`pad = 1e-9 * (max - min)`), with a fallback to the raw box if the inset
inverts. Regression-checked: every existing parameter's clip bounds move by
~1e-9 relative (numerically identical); only `f_scatter` changes. Added a final
guard that re-spreads any dimension that still collapses to a constant, with a
warning naming the parameter, so a badly scaled prior degrades loudly instead of
aborting inside emcee.

**3. `mcmc_summary.txt` reported `f_scatter: 0.000000`.** The summary and console
writers used fixed-point `%.6f`, which rounds a ~1e-14 flux floor to zero — it
read as "not fitted" even though the posterior was well constrained. Added
`_fmt_val(value, width=0)`, which switches to `%.6e` for non-zero magnitudes
below 1e-4, and applied it to the marginal-posterior, derived-parameter and MAP
blocks plus the `print_results` console table.

**Verified end-to-end** under `henv`: `--fit-scatter` with both `chi2` and
`jitter` likelihoods, `--save-chi2 --compute-bic --smooth`, produces all
artifacts; `f_scatter` is genuinely sampled (230 unique values spanning
8.2e-17 to 1.7e-13, no longer pinned); the χ² table is 100 % finite; and
`plot_best_fit`'s reduced χ² at the MAP matches the χ² table row for that same
sample (239.2630 vs 239.2631).

**4. `plot_best_fit` duplicated the evaluation logic.** It called
`eval_fn = getattr(model, 'evaluate_direct', model.evaluate)` in three places and
added `f_scatter` by hand to each, and it re-implemented geometry resolution
(its own `if reparam: stats['d1']…` branch). Numerically correct, but it was the
same duplication pattern that caused problem 1, and the `evaluate_direct`
fallback was dead — it referenced `PrecomputedModelGrid`, removed in Phase 18.
Now it reconstructs the point-estimate `theta` in active-parameter order and
calls `_evaluate_model` through a local `_eval_at(phases)` helper, so geometry
(phys / reparam / kepler / frozen), wind shape, and `f_scatter` all resolve
through one implementation. `f_scatter_best` is obtained from
`_resolve_scatter` and used for display only. Verified behaviour-preserving:
`plot_best_fit`'s χ² matches the likelihood exactly across eight
configurations — `phys`, `phys + f_scatter`, `reparam`, `kepler`,
`jitter + f_scatter`, `smooth_pl` shape fit, `beta_law` shape fit (with
`R_star` tied to `R`), and `frozen R + f_scatter`.

---

## Phase 22 — Adaptive Binning in the Single-Model CLI

`chandra_phase_analysis.py`. **Status: uncommitted** on branch
`add_generic_wind`.

Phase 18 added `phase_bin_data_snr` to `chandra_phase_analysis.py` but wired the
CLI flag only into `mcmc_lightcurve_fit.py`, so the single-model script could not
use adaptive constant-counts bins from the command line. Added `--counts-per-bin`
with the same argument-presence semantics as the MCMC script:

`--no-phase-bin` > `--counts-per-bin N` > `--n-phase-bins N` > 50 fixed-width
bins.

- `--n-phase-bins` default changed from `50` to `None` so mode selection is
  unambiguous; the effective fallback is still 50, so existing commands behave
  identically.
- Supplying both binners is a `parser.error`, as are non-positive values.
- `--counts-per-bin` on data with no `counts` column fails fast with a message
  pointing at `--n-phase-bins`, rather than raising from inside the binner.
- `--min-points-per-bin` help now notes it applies to fixed-width binning only.
- Variable bin widths already flowed through to horizontal error bars, since
  `plot_phase` picks up the `width` column when `is_binned` — no plotting change
  needed.

Verified on ObsID 15803 soft: `--counts-per-bin 100` gives 87 bins averaging
104.3 counts (vs 60 fixed-width bins), and χ²/dof drops from 8.501 to 6.385 as
the noisy narrow eclipse-trough bins merge.

---

## Phase 23 — `utils/` Extraction and a Single Plotting Routine

`chandra_phase_analysis.py`, `mcmc_lightcurve_fit.py`, new `utils/utils.py`,
`utils/plot_utils.py`, `utils/__init__.py`. **Status: uncommitted** on branch
`add_generic_wind`.

Both analysis scripts carried their own plotting functions, and
`mcmc_lightcurve_fit.py` imported its data-layer helpers *from*
`chandra_phase_analysis.py` — so the CLI script was simultaneously a library and
a front end, and `mcmc` depended on it for reasons unrelated to Chandra data.
Everything shared now lives in a `utils` package and neither script imports the
other.

### New layout

| Module | Contents |
| ------ | -------- |
| `utils/utils.py` (1071 lines) | `REF_EPOCH`, `ORBITAL_PERIOD`, `frac`, `fmt_val`, `band_label_from_column`, `detect_flux_columns`, `validate_sim_columns`, `read_observation`, `load_data`, `phase_bin_data`, `phase_bin_data_snr`, `smooth_lightcurve`, `estimate_scattered_flux`, `prepare_model_interpolator`, `model_from_wrap`, `evaluate_model_at_phases`, `interp_periodic_phases`, `obs_errors`, `fit_simulation`. numpy/pandas/scipy only. |
| `utils/plot_utils.py` (597 lines) | `plot_lightcurve_fit` (the one drawing routine), `plot_phase`, `plot_multi_column_fits`, `plot_corner`, `plot_trace`, `add_residual_panel`, `build_fit_title`, `format_reduced_chi2`, `half_widths`. |

`chandra_phase_analysis.py` drops from 1639 to 457 lines and is now only the
argparse CLI plus an `__all__` re-export block, so `from chandra_phase_analysis
import *` — the notebooks' import style — is unchanged.
`mcmc_lightcurve_fit.py` drops from 4124 to 3952 lines.

### One plotting function

`plot_lightcurve_fit` is the drawing code that used to be inlined in
`mcmc_lightcurve_fit.plot_best_fit`, generalized so both paths reach it. It
draws only what it is handed — observed arrays, an already-shifted overlay
curve, and the model evaluated at the observed phases — which is what makes it
usable from both:

- `plot_best_fit` still owns the MCMC-specific work (MAP-vs-median point
  estimate, `_evaluate_model`, `_apply_best_phase_shift`) and then delegates.
- `plot_phase` is now a thin adapter: interpolate `sim_df` at the given `shift`
  and additive `scatter`, then delegate. Its displayed-χ² self-check warning
  (Phase 20) moved with it.

Feature parity required generalizing three things: an optional `obs_group`
array (so Chandra's per-observation series still get their own colors and
legend entries), an `ax`/`ax_res` pair (so `plot_multi_column_fits` can still
draw into a grid cell, residual-panel-free), and a `title` escape hatch for the
no-model data plot. Two incidental fixes fell out: a model overlay with no error
column no longer produces an empty residual panel, and `--counts-per-bin` widths
reach the x-error bars through one code path instead of two.

### Removed from the figure

Per request, `plot_best_fit`'s wheat-colored annotation box is gone. The title
is now only the energy band and χ²/dof:

```
SOFT band  —  χ²/dof = 13.018
```

The information it carried is not lost — point estimate, `phase_shift`, `f`,
`chi2_eff/dof` and `f_scatter` are printed to stdout next to the existing
parameter table, which is also written to the run summary. `plot_phase` titles
gained the band label the same way: `nfl_soft -> "SOFT band"` via
`band_label_from_column`, so grid panels remain identifiable.

### Deduplicated

- `_interp_periodic_phases` (mcmc) and `_model_from_wrap` (chandra) were two
  spellings of periodic model interpolation; both now live in `utils.utils` as
  `interp_periodic_phases` / `model_from_wrap`, documented as the array-in and
  prepared-interpolator forms of the same operation.
- `plot_corner`, `plot_trace`, `add_residual_panel`, `fmt_val` moved out of
  `mcmc_lightcurve_fit.py` verbatim.
- `mcmc_lightcurve_fit.py` no longer imports `matplotlib.pyplot` or `corner` at
  all — all figure work is behind `utils.plot_utils`.

### Verification (conda env `henv`, Python 3.13.5 / numpy 2.2.6)

- `py_compile` on all five files; `import` of both scripts; star-import exports
  23 names with nothing missing.
- Chandra CLI, four modes on ObsID soft data: adaptive bins + fitted shift +
  smoothing (χ²/dof 30.605, no self-check warning), 3-column grid (98.007 /
  15.071 / 46.100, one title per band), fixed-width no-fit plot, raw unbinned
  fit.
- MCMC: 24 walkers × 60 steps with `--fit-scatter --smooth` produced χ²/dof
  13.0179; `--replot` from the saved chain reproduced 13.0179 exactly.
- Consistency suite, 15 assertions: across `phys` / `phys, shift fixed` /
  `phys + f_scatter`, the χ²/dof `plot_best_fit` reports equals the χ² of the
  arrays it hands the plotter to `rtol=1e-12`, and the drawn overlay
  reinterpolated onto the observed phases matches the residual basis to
  ≤5.0e-4 relative (the Phase 20 regression would blow this up by orders of
  magnitude). `fit_simulation` recovers an injected `shift=0.30` to 5 decimals
  with χ²/dof = 0; the self-check guard stays silent when `shift`/`scatter`
  agree and fires when `scatter` is wrong.
- All five call forms the notebooks use for `plot_phase` /
  `plot_multi_column_fits` run against real data with no self-check warnings.
  (End-to-end `nbconvert` execution still stops at the notebooks' `import
  xspec` cell — a pre-existing `henv` limitation, unrelated to this change.)

Not migrated: `chandra_analysis_combined_flux.py` keeps its own older copies of
`fit_simulation` / `plot_phase` / `plot_multi_column_fits` and still fits a
multiplicative flux scale. See [Known rough edges](PROJECT.md#known-rough-edges).

---

## Phase 24 — Run-Config Persistence and a Replot-Mode Fix

`mcmc_lightcurve_fit.py`. **Status: uncommitted** on branch `add_generic_wind`.

### Bug: `--replot` could not read a kepler- or reparam-mode chain

`load_existing_results` validated the samples CSV against
`REPARAM_PARAM_NAMES if reparam else PARAM_NAMES` — the `kepler` argument it
accepted was never used. Replotting a `--kepler` run therefore failed with

```
Error: Samples file missing required geometry columns: ['d1', 'd2']
```

even though the file correctly held `M_X, M_RH, r, R, i0, ...`. Fixed to select
the geometry block by mode, and two related gaps closed:

- Frozen parameters are not sampled and so are legitimately absent from the CSV;
  they are now excluded from the required set (`--freeze R=2.0` runs were
  unreplottable for the same reason).
- The error now lists the columns actually present and, when they match another
  parameterization, names the flag to use.

### Feature: the CLI of every fit is saved and restored

`--replot` previously recovered only `mode`, frozen values, `orbital_period_s`
and `likelihood` from the chain `.npz`. Everything else fell back to argparse
defaults — including `--data-dir`, `--obs-column`, `--time-column`,
`--counts-per-bin`/`--n-phase-bins`, `--lam`, `--dth`, `--d2h` and all
`--prior-*`. Those options determine the *observed arrays*, so a replot that
missed them reported a χ²/dof for a dataset the posterior had never seen. On the
existing `broad` kepler run, replotting with default `--lam`/`--flux-csv` gave
χ²/dof = 0.916 against the true 1.064.

Every fit now writes `<band>_<wind_model>_run_config.json` into `--output-dir`:

```json
{
  "created": "2026-08-26T16:53:02-0400",
  "command": "mcmc_lightcurve_fit.py --band broad --kepler ...",
  "band": "broad",
  "wind_model": "smooth_pl",
  "args": { "...": "every argparse dest" }
}
```

It is written *before* sampling starts, so it survives an interrupted run. On
`--replot`, `apply_saved_run_config` fills in every option the user did not type:

- **Explicit flags always win.** Which options were typed is determined by
  scanning `sys.argv` against the parser's option strings (including
  unambiguous prefixes), not by comparing against defaults — a user who
  explicitly passes the default value still overrides the saved config.
- `replot` and `output_dir` are never restored: the first would cancel the
  replot (a saved fit always recorded `replot=False`), and the second is defined
  by where the config was found.
- `--band` and `--flux-csv` changed from `required=True` to being validated
  after the restore, so `--replot` alone is a complete command. They are still
  required for a real fit.
- Ambiguity is handled: with several configs in one directory, ones differing
  only by `band` (the `--band all` case) restore identically and the first is
  used; genuinely different configs produce an error listing them.
- Self-healing: a `--replot` that finds no config writes one from the options it
  just used, so pre-existing result directories become self-sufficient after one
  full-CLI replot.

As an independent backstop, `replot_from_existing` now compares the observed
point count against `n_obs` in the chain metadata and warns on a mismatch.
`n_obs` is also now stored unconditionally rather than only when `--compute-bic`
ran, so the check works for every run.

### Verification (conda env `henv`)

- The existing `mcmc_results/broad_smooth_pl` kepler + wind-shape + `f_scatter`
  run replots without error. With the original CLI it reproduces the saved
  numbers exactly: χ²/dof **1.06423** (summary recorded 1.064) and BIC
  **195.673** (`logL_hat=-77.689, k=8, n=154`).
- A second `--replot --output-dir mcmc_results` with **no other arguments**
  reproduces the same 1.06423 / 195.673 from the auto-written config.
- Fresh 24×40-step soft fit: config written, then bare `--replot` round-tripped
  χ²/dof = 13.2843 identically.
- `--replot --lam 0.9` reports `kept from the command line: --lam`, confirming
  precedence.
- Mismatch guard fires as intended: forcing `--counts-per-bin 300` on the broad
  run warns "Replot is using 53 observed points but the saved fit used 154".
- `--band`/`--flux-csv` still error out when missing on a non-replot run.

---

## Phase 25 — MCMC Script Slimming

`mcmc_lightcurve_fit.py`, `utils/utils.py`. **Status: uncommitted** on branch
`add_generic_wind`. Behaviour-preserving throughout — every regression number
below is bit-identical to the pre-cleanup run.

`mcmc_lightcurve_fit.py`: **4246 -> 3553 lines (-693, -16%)**.

### Moved into `utils/utils.py` (generic, not MCMC-specific)

| Moved | Lines |
| ----- | ----- |
| `resolve_band_directory`, `load_observed_lightcurves` | 90 |
| `build_phase_shift_terms`, `apply_best_phase_shift` + the `DEFAULT_PHASE_SHIFT_*` constants | 95 |
| run-config persistence: `save_run_config`, `find_run_configs`, `apply_saved_run_config`, `_explicit_cli_dests`, `_jsonable`, `run_config_path` | 165 |
| `save_samples_csv_chunked` | 23 |

The phase-shift search now sits next to `fit_simulation`, which runs the same
coarse-scan-then-refine algorithm on a tabulated model — both spellings of the
idea are in one place. `utils/utils.py` gained a stdlib-only dependency
footprint (`argparse`, `csv`, `json`, `shlex`, `time`) and stays free of any
import from either analysis script.

### Deleted by making the shared binners honest

`phase_bin_data` / `phase_bin_data_snr` hardcoded `rate`/`error` as their output
column names, so the MCMC script carried two 25-50 line wrappers that renamed
`flux`/`flux_err` in and back out again. The binners now name their value
columns after the `rate_column`/`error_column` arguments they were given, so the
MCMC path calls them directly:

```python
bin_cols = dict(rate_column='flux', error_column='flux_err')
obs_df = phase_bin_data_snr(obs_df, counts_per_bin=..., **bin_cols)
```

`chandra_phase_analysis` passes `rate`/`error` and is unaffected. **-73 lines.**

### Deduplicated

- **Likelihood front half.** `log_likelihood_chi2` and `log_likelihood_jitter`
  shared ~23 lines of identical model-evaluation + phase-shift-alignment
  preamble. Extracted as `_aligned_model_flux`; each likelihood is now its own
  formula plus a call. `f = exp(log_f)` moved after the alignment (it does not
  depend on the model, so the result is unchanged).
- **CLI plumbing.** `_phase_shift_opts(args)` replaces 8 copies of the same
  three-line `fit_phase_shift=` / `phase_shift_grid_size=` /
  `phase_shift_eval_points=` `getattr` triple. `_smooth_plot_kwargs(smoothed,
  args)` replaces two 13-line blocks of `smoothed[...] if smoothed is not None
  else None`.
- **`--prior-*` definitions.** Nine near-identical 8-line `add_argument` calls
  became a table-driven loop reading defaults straight from `DEFAULT_PRIORS` /
  `REPARAM_PRIORS` / `KEPLER_PRIORS`, so the help text cannot drift from the
  values actually used. It already had: `--prior-r` advertised
  `max=0.01` while the code used `0.1`. **-50 lines, one stale-help bug fixed.**
- **Prior parsing.** The geometry and wind-shape override loops were the same
  20 lines twice; now one `_parse_prior_overrides(parser, args, names)`.
- **`replot_from_existing`** opened `*_chain.npz` twice, in two separate
  try/except blocks, to read `mode`/`frozen`/`n_obs` and then `likelihood`.
  Merged into one read.
- **Label construction.** The 17-line "name -> corner label" `if/elif` chain in
  the replot path became `_labels_for_names(names, mode)`, built on the existing
  `get_mode_name_label`.
- `_default_priors(reparam, kepler)` replaces three copies of the
  `if kepler / elif reparam / else` prior-selection block (and upgrades two of
  them from `.copy()` to `deepcopy`, so a prior override can no longer reach the
  module-level dict).
- `_maybe_save_chi2(...)` replaces two 15-line `--save-chi2` blocks.

### Dead code removed

- `compute_pointwise_loglik` (84 lines) — the per-observation log-likelihood
  matrix that WAIC/LOO consumed. Traced before removing: it was added *with* its
  caller in `b867ff1` and called for six commits from inside
  `run_arviz_diagnostics`, gated on `--compute-waic`, feeding
  `log_lik_dict = {"obs": ll[np.newaxis, :, :]}` into `_build_inference_data`
  and thence `az.waic` / `az.loo`. Phase 15 (`7116302`) deleted that call when
  BIC replaced WAIC/LOO — `L̂` from the run's own likelihood at the max-log-prob
  sample needs no pointwise terms — leaving `--compute-waic` as a shim; Phase 18
  (`b376585`) deleted the flag and shim outright. Unreachable ever since, with
  the only remaining trace being `_build_inference_data`'s optional
  `log_lik_dict` parameter, whose sole caller passes `None`. Note it was
  *maintained* dead code (it picked up the Phase 17 phase-shift alignment and
  lost its `studentt` branch with Phase 18), so it is a working starting point
  if LOO is ever wanted back: `git show 19dc637:mcmc_lightcurve_fit.py`.
- `_to_physical` — a "backward-compatible wrapper for legacy callers" with no
  callers anywhere in the repo.
- A dead `import gzip` inside `compute_chi2_for_samples` (the gzip write goes
  through `to_csv(compression='gzip')`), and now-unused module imports
  (`csv`, `json`, `shlex`, `sys`, `glob`, `pickle`).

### Comments trimmed

Thirteen multi-line rationale comments (7-10 lines each) cut to 1-3 lines,
keeping the *why* and dropping the retold debugging narrative — e.g. the
walker-clip block went from eight lines to three while still naming the failure
it prevents (`f_scatter` collapsing and aborting emcee on the condition number).
**-37 lines.**

### Verification (conda env `henv`) — all numbers unchanged

- Bare `--replot --output-dir mcmc_results` on the real broad kepler +
  wind-shape + `f_scatter` run: χ²/dof **1.06423**, BIC **195.673**
  (`logL_hat=-77.689, k=8, n=154`) — identical to Phase 24.
- Consistency suite: **18/18 pass**, including three `plot_best_fit`
  configurations agreeing with the drawn arrays to `rtol=1e-12` at the same
  values as before (1.117909384 / 1.118796619 / 1.392422282), and new checks
  that the moved functions resolve to `utils.utils` and the binners preserve
  column names.
- Fresh fits round-trip through `--replot` exactly in every mode: `chi2` +
  `f_scatter` (12.9794), `jitter` + `f_scatter` + `--save-chi2` +
  `--compute-bic` (16.698, BIC -2011.323), `--reparam` (5.71919), `--kepler
  --fit-wind-shape --freeze R=13.0` (9.82407).
- Chandra CLI, all four modes: 30.605 / (98.007, 15.071, 46.100) / 40 bins /
  3.129 — unchanged.
- All five notebook `plot_phase` / `plot_multi_column_fits` call forms run with
  no self-check warnings. ArviZ, corner and trace paths exercised.
- No unused imports remain in any of the four files.

---

## Phase 26 — Binary-Geometry Diagnostic Plots

`mcmc_lightcurve_fit.py`, `utils/plot_utils.py`, `utils/utils.py`,
`plot_results.py`. **Status: uncommitted** on branch `add_generic_wind`.

`plot_results.py` had a `--geometric` mode that plotted `l3`/`L3`/`h3`, `A2`,
`icd` and time against phase, but only for a simulation CSV and with no
reference to the parameters that produce them — and nothing equivalent existed
on the MCMC side, where a posterior can fit the light curve perfectly with a
geometrically absurd configuration. Its plotting moved into `utils/plot_utils.py`
(leaving a 104-line CLI) and three geometry figures are now produced per fit.

### The key realization

`simulate_lightcurve` already returns everything needed, and `(L3, h3)` are
*exactly* the sky-plane Cartesian coordinates of the compact object relative to
the companion centre, with `l3 = sqrt(L3² + h3²)` the projected separation the
eclipse test compares against `R ± r`. So the projected-orbit diagram is exact
rather than a schematic, and one further consequence follows: the line of sight
from the compact object has impact parameter `l3` relative to the companion
centre, so `[min l3, max l3]` is precisely the range of radii the data probe.

### Three plots, chosen for what they answer

1. **`*_geometry_orbit.png`** — projected orbit over the companion disk, plus a
   to-scale top-down view with `d1`/`d2` and the observer direction. The eclipse
   width constrains a *combination* of `(a, R, i0)`, so this is where a
   well-fitting but implausible parameter set shows up. A footer states the
   verdict numerically: min projected separation vs `R - r` and `R + r` ->
   total / partial / no eclipse.
2. **`*_geometry_phase.png`** — 4 panels: `l3(φ)` against the `R ± r` thresholds
   with the eclipsed interval shaded; the sky-plane components (`h > 0` means
   the emitter is behind, which is what gates the eclipse test, and explains why
   the equally-close conjunction at φ≈0 is *not* eclipsed); `N_H(φ)` with its
   orbit mean annotated (a direct check that it equals `lam`); and the resulting
   band flux.
3. **`*_wind_profile.png`** — `g(r)` with 68/95% posterior credible bands from
   up to 300 draws, an `r⁻²` reference, the companion surface, the
   characteristic radii (`Rb`/`H`/`ell`), and the probed-radius band. The shape
   parameters are only interpretable jointly, so this shows the constraint on
   the quantity that actually enters the model.

Two panels of the old `plot_geometric_parameters` were deliberately dropped:
"Time vs Phase" is linear by construction, and `A2` is the polar-grid cell area
— an artifact of the integration mesh, not physics.

### Wiring

- `plot_geometry_diagnostics()` resolves the point estimate, calls
  `simulate_lightcurve` **once**, and drives all three plots. Called from both
  `run_single_fit` and `replot_from_existing`; `--no-geometry-plots` skips it.
- It passes `scattered_flux=f_scatter` into the simulation, so the flux panel
  shows the curve that was actually fitted rather than one missing the additive
  floor. Verified: the eclipse floor reads 1.70251e-13, matching `plot_best_fit`
  exactly; without it the panel bottomed at 0.
- The MAP-vs-median point-estimate logic was extracted from `plot_best_fit` into
  `_point_estimate_theta()`, now shared by both.
- `BAND_INFO`, `detect_energy_bands`, `get_band_display_name` moved to
  `utils/utils.py` next to `detect_flux_columns`.

### What it immediately showed on the real broad fit

At χ²/dof = 1.064 the MAP geometry is `a = 17.62`, `R = 14.70 R☉`, `i0 = 14.50°`:

- **`R/a = 0.83`** — the companion nearly fills its own orbit. Worth a look
  against the Roche lobe, which for `d1/d2 = 1.57` sits far inside `R`.
- The eclipse is **total across 29.2% of the orbit** (min projected separation
  4.41 vs `R - r` = 14.61), and the flat-bottomed minimum in the data is fitted
  as that total eclipse plus the `f_scatter` floor — self-consistent, and now
  visible as such.
- **`Rb = 27.09 R☉` lies entirely outside the probed range (4.4–17.6 R☉)**, so
  the break radius is unconstrained by these data; only the inner slope `p` is
  doing work over the sampled radii. That is a degeneracy the corner plot does
  not make obvious.

### Verification (conda env `henv`)

- `--replot --output-dir mcmc_results` still reports χ²/dof **1.06423** and BIC
  **195.673**, now writing three extra figures.
- Eclipsing (broad kepler, 29.2% eclipsed) and non-eclipsing (short soft fit,
  min projected separation 9.21 vs `R + r` = 2.69) posteriors both render
  correctly, the latter labelled "No geometric eclipse".
- `--no-geometry-plots` suppresses all three.
- `plot_results.py` exercised in all three modes (`--geometric`, `--orbit`,
  default band grid), including the argument-validation path
  (`--orbit requires --R, --d1, --d2, --i0`).

---

## Phase 27 — Standard Inclination Convention at the Public API

### The mismatch

`simulate_lightcurve`'s `i0` was measured from the **line of sight**: the
geometry expressions want `h = a·sin(γ)·sin(incl)` for the sky-plane offset and
`z = a·sin(γ)·cos(incl)` along the LOS, and the code passed `i0·π/180` straight
into them. So `i0 = 0` meant edge-on and `i0 = 90` meant face-on — the reverse
of the astronomical convention, in which inclination is measured from the
orbital-plane normal (equivalently, from the plane of the sky), `i = 90°` is
edge-on and `i = 0°` face-on.

The consequence was not a bug in the model but a translation tax on every
result: the fitted `i0 ≈ 14.5°` had to be reported as `≈ 75.5°`, and every prior
taken from the literature had to be complemented by hand before being typed as
`--prior-i0`.

### The change

The conversion is confined to a single line at the input boundary:

```python
def inclination_to_internal_rad(i0_deg: float) -> float:
    return (90.0 - float(i0_deg)) * np.pi / 180.0
```

`simulate_lightcurve` calls it in place of `i0 * np.pi / 180`. **No geometry
expression was touched** — `_simulate_phases_numba` and `wind_los_integral`
still take `incl` from the line of sight, which is the right internal choice
because it is the angle that appears directly in the two projection formulas.
Splitting the public convention from the internal one this way means the eclipse
test, the LOS integral and the `(L3, h3)` outputs are provably unchanged.

Defaults and priors were complemented so the *behaviour* is identical, only the
number typed differs:

| Where | Before (from LOS) | After (from normal) |
| ----- | ----------------- | ------------------- |
| `simulate_lightcurve(i0=…)` | `26.0` | `64.0` |
| `xrb_lightcurve.py --i0` | `26.0` | `64.0` |
| `DEFAULT_PRIORS['i0']` | mean 26, σ 20, [10, 85] | mean 64, σ 20, [5, 80] |
| `REPARAM_PRIORS['i0']` | mean 26, σ 20, [10, 85] | mean 64, σ 20, [5, 80] |
| `KEPLER_PRIORS['i0']` | mean 26, σ 20, [10, 85] | mean 64, σ 20, [5, 80] |

The prior mapping is exact rather than approximate: a Gaussian is symmetric
about its mean, so `N(μ, σ)` truncated to `[lo, hi]` becomes `N(90 − μ, σ)`
truncated to `[90 − hi, 90 − lo]` with identical density at the reflected point.
`--freeze i0=…` and `--prior-i0` now both take conventional degrees.

### Stale-chain guard

Chains fitted under the old convention store the complement of what the model
now expects, and nothing in the numbers reveals it — `i0 = 14.5` is a perfectly
legal value in either convention. New run configs therefore carry

```json
"inclination_convention": "i0-from-orbital-normal"
```

and `apply_saved_run_config` warns when a config lacks the stamp, stating that
any χ² from that chain is meaningless. Confirmed on the existing broad kepler
run: `--replot` warns and reports χ²/dof **83.36** instead of 1.064, because
`i0 = 14.5` is now read as nearly face-on. Those results need refitting; no
migration path is provided, by choice.

### Verification (conda env `henv`)

- **Bit-for-bit round trip.** The pre-change `xrb_lightcurve.py` was loaded
  alongside the new one and `old(i0=x)` compared against `new(i0=90−x)` for
  `x ∈ {26, 12, 5, 60}`, over every output column including `flx`, `fl`,
  `nfl_*`, `l3`, `L3`, `h3`, `icd`, `A2` and `is_eclipsed`: **max relative
  difference 0.000e+00** in all four cases. The light-curve geometry is
  untouched, which was the whole requirement.
- `inclination_to_internal_rad` checked at 64→26, 90→0, 0→90, 45→45.
- Sanity of the new sign convention: `i0 = 90` (edge-on) eclipses, `i0 = 0`
  (face-on) does not.
- New `h3`/`L3` match the analytic `a·sin(γ)·sin(26°)` / `a·cos(γ)` to 1.8e-15.
- End-to-end MCMC (short broad kepler fit, `--prior-i0 75,5,40,85`) lands on the
  same physical mode as the original: MAP `i0 = 74.98°` (≈ 90 − 15.0),
  `a = 17.60`, `R = 14.49`, χ²/dof 1.068, BIC 196.2 — versus `i0 = 14.5°`,
  `a = 17.62`, `R = 14.70`, χ²/dof 1.064, BIC 195.673 before.
- A freshly written run config replots with no warning and reproduces its own
  χ²/dof exactly (1.06755).
- Consistency suite: ALL CHECKS PASSED. `xrb_lightcurve.py` CLI and
  `plot_results.py` (`--orbit`, `--geometric`, default) all exercised.
- `chandra_phase_analysis.py` and `chandra_analysis_combined_flux.py` contain no
  reference to `i0` or `incl` (they consume a precomputed simulation CSV), so
  they are unaffected.

### Not changed

`rkp_run_w_mcmc_cmds.sh` is a historical log of commands as they were run, and
already contains flags that no longer exist (`--lam2`, `--load-grid`,
`--compute-waic`, `--wind-model av`). Its `--i0` / `--prior-i0` values are left
in the old convention rather than partially rewriting the record. The notebooks
also call `simulate_lightcurve` with old-convention `i0` and need their values
complemented before they are re-run.

---

## Phase 28 — Physical Wind Normalization & Per-Cell Flux Conversion

### Why: two exact degeneracies in the `lam` normalization

A broad-band kepler fit (35k steps, zeus) reached χ²_red = 1.020 but reported
`converged: False` with autocorrelation times up to 1509 steps, and its MAP
`M_X = 2.53 M☉` fell outside the marginal 16/84 interval `9.17 (+8.20/−5.50)`.
Neither is a sampler defect. Measured directly against the model:

| Test | Result |
| ---- | ------ |
| Scale `(r, R, d1, d2, Rb)` by λ ∈ [0.8, 1.5] | flux changes ≤ 3e-3 (grid round-off); implied `M_tot` swings 14 → 94 M☉ |
| Vary `q = M_RH/M_tot` from 0.30 → 0.95 at fixed `a` | flux changes ~1e-15 (floating-point noise) |

Because `simulate_lightcurve` rescaled the wind integral so that
`mean(fl) = lam`, the absolute column was discarded and the model depended only
on **ratios** — `R/a`, `r/a`, `Rb/a`, `p`, `i0`. The posterior confirms it:
`R/a` = 0.881 at the MAP versus 0.894 at the median, essentially identical,
while `a` itself ranged 13.7 → 16.3 R☉. And since only `a = d1 + d2` enters the
geometry, the mass *split* is invisible.

The consequence is that under `--kepler`, where `a = K·M_tot^(1/3)`, **`M_X` and
`M_RH` were determined entirely by their priors**. The reported `M_X` was never
a measurement, the MAP wandered freely along a flat ridge, and the flat
directions are what produced the 1500-step autocorrelation times.

### Why: the visible-area term was discarded

`A2` (unmasked emitter area) was computed by the kernel and written to the
output frame, but never used to compute `nfl_*` — those came only from `fl`.
Partial occultation therefore did not attenuate the flux at all. Measured at the
posterior median, `A2/A2_max` ramps 1.00 → 0.05 over Δφ ≈ 0.066 (**2.3 h**)
before total eclipse begins, against an observed ingress of Δφ ≈ 0.08 (2.8 h).
The model held flux at full level across that entire ramp and then dropped
abruptly to `f_scatter`, which is also why `R` inflated to 14.6 R☉ (a 10.3 h
total eclipse against ~5–7 h observed): without a penumbra, `R` had to stretch
to cover the observed width.

### The change: `wind_norm`

`simulate_lightcurve` gained a `wind_norm` switch. **`"lam"` remains the
default and is bit-for-bit unchanged.**

```python
if wind_norm == "lam":                    # historical behaviour
    col_scale = lam / mean(flx)
else:                                     # "physical"
    n0 = wind_density_norm_from_mdot(mdot, v_inf, wind_model, wind_params, mu)
    col_scale = f_opacity * n0 * R_SUN_CM / 1e22
```

Two new helpers back it:

- `wind_asymptotic_coefficient(wind_model, wind_params)` — returns `C` with
  `g(r) → C/r²` as `r → ∞` (`Rb²` for the power-law models, 1 for `beta_law` /
  `confinement`), since every supported profile relaxes to a constant-velocity
  `r^-2` wind far from the star.
- `wind_density_norm_from_mdot(...)` — matches that limit to a spherical wind,
  `n_0 = Ṁ / (4π R_☉² v_∞ μ m_H C)`.

This is what breaks the scale degeneracy: `N_H` now depends on the **absolute**
`Rb` in R☉ rather than only on ratios, so scaling all lengths no longer leaves
the light curve invariant.

`R` also recovers its original meaning under `wind_norm="physical"` — the
genuinely opaque photosphere (~2 R☉) — because the extended opaque core now
emerges from the wind column instead of from the geometric cutoff. Under
`"lam"`, `R` remains the *effective* eclipsing radius (~9–14 R☉, comparable to
the 8–10 R☉ that Laycock et al. 2015 derive from the 5 h eclipse).

### The change: per-cell flux conversion

This was required, not cosmetic. The nH → flux mapping is nonlinear, so
`⟨F(N)⟩ ≠ F(⟨N⟩)`. When the column varies steeply across the emitter disk —
near the occulter limb, and everywhere in physical mode — the surviving flux is
dominated by the least-absorbed cells. Converting the *mean* column would be
wrong by many powers of `e` in the eclipse core.

`_simulate_phases_numba` therefore also returns `cell_col`, `cell_area` and
`cell_count` (`n_phases × n_th·n_r_ring`), and the flux block was refactored
into a `band_maps: Dict[str, Callable]` applied either to the per-phase mean
column (`lam`) or per emitter cell followed by an area-weighted average
(`physical`). Falls back with a warning if the numba mega-kernel is unavailable,
since the pure-Python path cannot supply per-cell data.

Physical constants (`R_SUN_CM`, `M_H_G`, `M_SUN_G`, `KM_TO_CM`, `YEAR_S`,
`MU_WIND_DEFAULT = 1.4`) moved from their block below `simulate_lightcurve` to
module top, because they are now used as default argument values.

### MCMC wiring

New CLI group **Wind Normalization**: `--wind-norm {lam,physical}` (default
`lam`), `--mdot` (4e-6 M☉/yr), `--v-inf` (1750 km/s), `--mu-wind`, and
`--fit-fopacity`.

`log_fopa` — log₁₀ of the effective-opacity factor — becomes a fitted dimension,
inserted after `f_scatter` and before the wind-shape parameters. It absorbs wind
photoionization, clumping, and the departure of a He-rich WR wind from the solar
abundances the TBabs `flux_vs_nH` table assumes. `FOPACITY_PRIOR` is centred at
−1.5 because Clark & Crowther's Ṁ predicts N_H ≈ 19–47 ×10²² out of eclipse
against an observed ~0.75 ×10²². Supporting changes: `ParamSpec.wind_norm` /
`.fit_fopacity`, `_resolve_fopacity`, `log_fopa` freezable via `--freeze`,
`get_active_priors(fit_fopacity=…)`, and `DirectLightCurveModel.evaluate`
gaining an `f_opacity` argument. `--fit-fopacity` outside physical mode is
rejected at parse time.

### Verification (conda env `henv`)

- **Backward compatibility exact.** With `wind_norm` at its default,
  `mean(fl)` reproduces `0.589537` to all printed digits and `nfl_*` are
  unchanged. `utils/test_flux_methods.py` passes.
- **Degeneracy break confirmed but partial.** In physical mode with Ṁ fixed,
  the normalized light-curve shape changes by 2.8% at λ = 0.8 and 4.9% at
  λ = 1.5, versus < 0.3% for the same sweep under `lam`. Roughly a 15×
  improvement in scale sensitivity — a real constraint on `a`, not a tight one.
- **Eclipse now from wind opacity.** With `R = 2` (photosphere), geometric
  eclipse phases drop to zero and the core reaches N_H ~ 10⁵ ×10²², genuinely
  Compton-thick.
- End-to-end MCMC smoke tests pass in both modes; `log_fopa` appears in the
  ArviZ summary and is sampled.
- Cost: 13.7 → 18.0 ms per light curve (+31%), from the per-cell interpolation.

### Known limitations

- The degeneracy is broken only weakly (see above). Do **not** quote a
  black-hole mass from a `lam`-mode fit; `M_X` there is a prior artifact.
- `f_opacity` is doing heavy lifting (~0.02–0.04). Physically that is wind
  photoionization stripping the metals that carry photoelectric opacity
  (ξ ~ 10³), but it is phase- and position-dependent — the shadowed sector stays
  neutral, which is the Laycock et al. 2015 He II argument — so folding it into
  one scalar is an approximation. It is also partly degenerate with λ, which is
  why the scale degeneracy is only partially broken.
- `Rb` and `p` fitted under `lam` are **not** valid starting points for physical
  mode: at those values the ingress comes out at 7.6 h against 2.8 h observed.
  They only ever had to reproduce out-of-eclipse modulation. Refit, and move
  `R`'s prior back to the photosphere (`--prior-R 2,0.6,1.2,5`).
- The observed ~12% eclipse floor must still come from `--fit-scatter`: the wind
  core is genuinely opaque, so the residual is scattered light — the same
  conclusion Steiner et al. 2016 reach independently.

---

## Phase 29 — Mass Reparameterization & Error-Column Fix

### `q` is exactly unidentifiable, so `--kepler` cannot fit masses

For a circular two-body orbit the light curve sees only the *relative*
separation. Both geometry expressions,

```
l       = (d1 + d2) · sqrt(sin²γ · sin²i + cos²γ)
z_start = (d1 + d2) · sinγ · cos i
```

depend on the sum alone: however the centre of mass splits `a`, the emitter's
position relative to the companion and its wind is unchanged. Sweeping
`q = M_RH/M_tot` from 0.20 to 0.95 at fixed `a` changes the flux by

| `wind_norm` | max relative change |
| ----------- | ------------------- |
| `lam`       | **0.00e+00** (bit-identical) |
| `physical`  | **3.9e-16** (round-off) |

The physical normalization does **not** help, and the geometry is correct —
there was nothing to fix there. But it means only `M_tot` enters the model, via
`a = K·M_tot^{1/3}`, so sampling `(M_X, M_RH)` lays one *exactly* flat
direction diagonally across both axes. The saved run configs show the symptom
plainly: two otherwise identical runs (25k and 35k steps) returned MAP
`M_X` = 4.50 and 2.53 while their medians agreed (9.04, 9.17), with
autocorrelation times of 707 and 1509 steps and `converged: False` in both.

### `--kepler-mtot`

Samples `(M_tot, q_m)` with `q_m = M_RH/M_tot` — the same reasoning that
motivated `--reparam` for `(d1, d2) → (a, q)`. The flat direction becomes
axis-aligned, so it no longer degrades the mixing of every other parameter;
`q_m`'s posterior comes out equal to its prior, which is honest and visible
rather than hidden inside a mass posterior; and `M_X`, `M_RH`, `a`, `d1`, `d2`
are reported as *derived*. `--freeze q_m=…` drops the dead dimension entirely.

| Mode | Sampled | Derived |
| ---- | ------- | ------- |
| `kepler` (`--kepler`) | `M_X, M_RH, r, R, i0` | `a`, `q`, `d1`, `d2` |
| `kepler_mtot` (`--kepler-mtot`) | `M_tot, q_m, r, R, i0` | `a`, `M_X`, `M_RH`, `d1`, `d2` |

New flags `--kepler-mtot`, `--prior-Mtot`, `--prior-qm`; three-way mutual
exclusion with `--reparam` / `--kepler`, both unchanged. Confirmed on a short
physical-mode run: `q_m` = 0.628 (+0.159/−0.154) against a `0.6 ± 0.15` prior,
i.e. posterior ≡ prior, exactly as predicted.

**Reporting rule.** `M_tot` is a (weak) measurement in `physical` mode; the
`M_X`/`M_RH` split is entirely the `q_m` prior. Quote `q_m` as a stated
assumption, not a result.

### Error-column auto-detection fix

`load_data` mis-detected the error column for lower-case proportional
observables. Two compounding defects:

1. The candidate list contained
   `obs_column.replace("FLUX", "FLUX_ERR")`, a **no-op on a lower-case name**,
   so `"flux_t"` survived in the list and matched *its own column* —
   `error := flux`.
2. `"rate_err"` was accepted for *any* observable. That has the right shape but
   the wrong scale for `flux_t`. `_derive_err_from_rate_err` was written for
   exactly this case and was unreachable.

Candidates are now built upper-cased, any candidate equal to the observable is
skipped, and `RATE_ERR` / `ERR_RATE` / `COUNT_RATE_ERR` are only offered when
the observable *is* the rate; the generic `*_ERR` fallback no longer accepts a
bare rate error for a non-rate observable.

`rate_err = rate/√counts` holds exactly in these files, so `flux_t` now derives
`flux_t/√counts` (matching to 2.7e-13 relative). `obs_column='rate'` still
returns `rate_err` exactly, and an explicit `--obs-error-column` still
overrides.

**Impact.** This affected every run using `--obs-column flux_t` without an
explicit error column — which includes both saved run configs. At
`--counts-per-bin 100` the median fractional error goes **0.386 → 0.101**, so
errors were **3.8× too large** and χ² **too small by ~14.6×** after
inverse-variance binning: a reported χ²/dof of 1.02 is closer to 15. Both the
single-LC and MCMC paths go through `load_data`, so both are corrected.
Expect χ²/dof ≈ 4–5 on the broad band now, reflecting ~20–25% intrinsic
aperiodic variability — which is why `--likelihood jitter` is no longer
optional if the credible intervals are to mean anything.

### Also fixed

A reporting regression from Phase 28: the summary writer called
`get_param_config` without `wind_norm` / `fit_fopacity`, so under
`--wind-norm physical` the sampled `log_fopa` row was silently omitted from
`mcmc_summary.txt` (it was present in the ArviZ table). `get_param_config` now
takes `kepler_mtot`, `wind_norm` and `fit_fopacity`, and the derived-quantity
rows are selected per mode.

### Verification (conda env `henv`)

- `phys`, `--reparam` and `--kepler` all still run and report their own
  parameterizations (regression sweep).
- `--kepler-mtot` runs end-to-end with `--wind-norm physical --fit-fopacity
  --likelihood jitter`, reporting `M_tot`/`q_m` plus derived
  `a`/`M_X`/`M_RH`/`d1`/`d2`.
- `flux_t` → `flux_t/√counts`; `rate` → `rate_err`; explicit override honored.
- `utils/test_flux_methods.py` passes.

### Not changed

`resolve_band_directory` tries `{band}/single` **before** `{band}/`, so
`--data-dir data/IC_10_X1_LC_CIAO --band broad` silently resolves to
`broad/single/` — one observation, not twelve. Every existing run in
`mcmc_results/` is therefore a single-ObsID (15803) fit; the `n=154` in those
BIC lines is exactly the single-obs bin count at 100 counts/bin, against ~432
for all twelve. Left as-is rather than reordering the candidates, which would
silently change the meaning of existing commands. Pass the band directory
explicitly (`.../broad` for all, `.../broad/single` for one). Note there are
**12** broad files, not 10: 3953, 7082, 8458, 11080–11086, 15803, 26188.

---

## Phase 30 — Physical Norm in the Single-Model CLI, Model-LC Dump & χ²_eff Fix

### Physical normalization was unreachable from `xrb_lightcurve.py`

`simulate_lightcurve()` has accepted `wind_norm` / `mdot` / `v_inf` /
`mu_wind` / `f_opacity` since Phase 28, but the script's argument parser never
exposed them, so a physical-norm light curve could only be produced by going
through a full MCMC. (The MCMC does not use this CLI at all:
`DirectLightCurveModel.evaluate()` calls `simulate_lightcurve()` directly as a
Python function, reading `wind_norm`/`mdot`/`v_inf`/`mu_wind` from its
`sim_params` dict and passing `f_opacity` per sample from `log_fopa`.)

Added `--wind-norm {lam,physical}` (default `lam`), `--mdot`, `--v-inf`,
`--mu-wind`, `--f-opacity`, wired through to `simulate_lightcurve` and echoed
in the parameter banner (only the fields relevant to the active mode).

```bash
python xrb_lightcurve.py --wind-norm physical \
    --mdot 4e-6 --v-inf 1750 --f-opacity 0.03 \
    --r 1.4 --R 2.0 --d1 12.2 --d2 8.1 --i0 82 --dth 1 --d2h 6 \
    --wind-model smooth_pl --Rb 12 --p 6.7 --Delta 2 \
    --flux_method interpolate --flux_csv flux_vs_nH_tbabs_broad.csv \
    --output lc_physical.csv
```

### `--Rmax` silently degraded physical mode

The CLI auto-set `--Rmax = 2*(d1+d2)` when omitted, and a *fixed* `Rmax`
disables the numba mega-kernel — the only path that returns the per-cell
columns Phase 28 relies on. Physical mode was therefore falling back (with a
warning that is easy to miss) to converting the **mean** column, which badly
understates eclipse-core leakage.

The auto-default now applies only under `--wind-norm lam`, preserving legacy
behaviour exactly (`Rmax = 40.6` for the reference geometry, `mean(fl)`
reproduced to all printed digits). Under `--wind-norm physical`, `Rmax` is left
adaptive so the mega-kernel is used, and passing `--Rmax` explicitly prints a
warning. The adaptive limits integrate the full z-tail, so this is strictly
more accurate as well as faster.

**Timing** (single LC, `dth=1`, `d2h=6`): lam/default **1.36 s**, lam +
`--converge-rmax` **1.25 s**, physical **1.19 s**. Physical mode is marginally
*faster* because it now takes the mega-kernel path.

### Best-fit model light curve as text

Every MCMC fit now writes `{band}_{wind}_bestfit_model.txt` beside
`_bestfit.png`, so the best-fit curve is usable outside the figure without
re-deriving the point estimate from the chain.

A `#` header records the point-estimate type (MAP or median), the
parameterization, `wind_norm`, χ²/dof and dof, jitter `f` and χ²_eff/dof, the
applied phase shift, every sampled and frozen parameter, the derived geometry
(`d1, d2, a, q, r, R, i0`, plus `M_X`/`M_RH`/`M_tot` where they exist), the
resolved wind-shape parameters, `f_opacity` and `f_scatter` — enough to
reproduce the curve from the file alone. Then two whitespace-delimited tables:

| Block | Rows | Columns |
| ----- | ---- | ------- |
| 1 — dense model curve | 359 | `phase`, `model_flux` |
| 2 — observed bins vs model | one per bin | `phase`, `obs_flux`, `obs_err`, `model_flux`, `resid_sigma` |

Block 1 is already shifted into the observed frame. Both read with
`np.genfromtxt(..., names=True)` after slicing to the block. The model grid
spans phase 0 and 1 inclusive, so wrapping it through the phase shift leaves a
redundant abscissa (the two copies differ only at bit level, and printed
identically at `%.8f`); it is dropped with a 1e-9 tolerance, orders of
magnitude below the ~1/360 grid spacing, leaving 359 strictly increasing rows.

### Per-parameter autocorrelation times

`mcmc_summary.txt` keeps its min/median/max line and adds a per-parameter
breakdown, with the number of τ contained in the chain and a convergence flag,
so a single badly-mixing dimension is attributable instead of hidden inside the
maximum:

```
  autocorr_time_steps per parameter:
    M_tot: 7.91  (7.6 tau in chain, <50 -> unconverged)
    q_m: 7.37  (8.1 tau in chain, <50 -> unconverged)
    ...
```

### χ²_eff was wrong by ~9 orders of magnitude

Surfaced while writing the model-LC header: a jitter run reported
`chi2_eff/dof = 6.87e-10`. Recomputing from the written Block 2 table gives
**1.4768**.

The cause is `sigma2 = np.maximum(sigma2, np.finfo(float).eps)`. That imposes
an **absolute** floor of 2.2e-16 on a flux variance of order 1e-25, so it
clamped *every* bin to the floor and deflated χ²_eff by ~9 dex. It is the same
class of mistake already documented in the walker-init code for `f_scatter`
("an absolute epsilon exceeded f_scatter's entire range"); these two sites
never got the fix. Both now use a positivity-only guard (`np.finfo(float).tiny`),
which is all that is needed since `sigma2` is a sum of squares.

**The likelihood itself was never affected.** `log_likelihood_jitter` uses
`sigma2` directly with no clamp, so existing fits, chains and posteriors are
valid — only the reported χ²_eff/dof diagnostic was wrong, at both the
per-sample χ² path and the best-fit overlay. Now reads 1.19 on a smoke run.

### Verification (conda env `henv`)

- `lam` mode unchanged through the CLI: `mean(fl)` reproduces `--lam` exactly
  and `Rmax` still defaults to `2*(d1+d2)`.
- Physical mode through the CLI runs on the mega-kernel with no fallback
  warning; `N_H` and eclipse depth as expected.
- All four parameterizations (`phys`, `--reparam`, `--kepler`,
  `--kepler-mtot`) run end-to-end, each writing `_bestfit_model.txt` and a
  per-parameter τ block.
- Block 1 verified 359 rows and strictly increasing as printed; both blocks
  round-trip through `np.genfromtxt(names=True)`.
- `xrb_lightcurve.py --help` and `utils/test_flux_methods.py` pass.

---

## Phase 31 — Release Trim: `lam`, Wind Models and Flux Methods

Publication-readiness pass. Removes every alternative that the physical wind
normalization superseded, so there is exactly one supported path through the
code. **`xrb_lightcurve.py`: 2354 → 1196 lines.**

### Removed: the `lam` normalization

`--lam` and `--wind-norm` are gone; the physical `Mdot`/`v_inf` normalization
(Phase 28) is now the only mode. Rationale: under `lam` the orbit-averaged
column was pinned to a spectral-fit constant, which discarded the absolute
scale and left the light curve dependent only on ratios (`R/a`, `r/a`,
`Rb/a`) — so `M_X`, `M_RH` and `R` were prior artifacts rather than
measurements. `simulate_lightcurve` now always takes `mdot`, `v_inf`,
`mu_wind`, `f_opacity`; `ParamSpec.wind_norm` is deleted.

Consequently deleted as unreachable (~300 lines): `compute_surface_density`,
`compute_wind_normalization_constants`, `wind_density_posterior`,
`wind_normalization_constants_posterior` — all existed only to back out `n_0`
*from* `lam`, whereas `n_0` is now an input via `wind_density_norm_from_mdot`.

`--fit-fopacity` no longer requires a mode flag, and `log_fopa` is
unconditionally freezable (`--freeze log_fopa=-1.7`).

### Removed: the fixed-`Rmax` / trapezoid LOS path

`--Rmax`, `--converge-rmax`, `--n_jobs` and `--dz` are gone, along with
`_wind_los_profile_numba`, `create_grid`, `density_function`,
`wind_los_integral`, `_id_to_name`, `_unpack_params`, the per-phase Python
loop and the joblib branch (~450 lines).

Rationale: any `Rmax` disabled the mega-kernel, which is the *only* path that
returns per-cell columns — and without those the nonlinear `nH → flux`
conversion silently degrades to converting the mean column, badly understating
eclipse-core leakage (the hazard already noted in Phase 30). The Gauss-Legendre
quadrature integrates the full `z`-tail anyway, so the cutoff was never needed.

**Numba is now a hard requirement**; the module raises `ImportError` at import
rather than falling back to a degraded integrator.

### Removed: `broken_pl` and `beta_law` wind models

`WIND_MODEL_IDS` is renumbered to `{smooth_pl: 0, confinement: 1}`;
`_g_profile` / `evaluate_g_profile` / `wind_asymptotic_coefficient` /
`default_wind_params` trimmed to match. On the MCMC side `beta` drops out of
`WIND_SHAPE_FIT` / `WIND_SHAPE_FIXED` / `WIND_SHAPE_LABELS` /
`WIND_SHAPE_PRIORS`, and `--prior-beta` is gone. `broken_pl` was always a
special case of `smooth_pl`, and `beta_law` was never used in a production fit.

New `ALL_WIND_SHAPE_NAMES` is derived from `WIND_SHAPE_PRIORS`, so the
`--prior-<name>` flag list can no longer drift from the registry (it was
hardcoded twice before).

### Removed: `--flux_method legacy`

`interpolate` (log-log interpolation of the XSPEC table) is now the default and
`refit` the only alternative. `--flux_csv` becomes **required**. The hardcoded
`nfl_hard = 9.524·e^{-0.057 nH}` / `nfl_soft = 9.3923·e^{-2.5062 nH}`
coefficients predated the XSPEC pipeline and were not tied to any band
definition in current use; the same stale constants are dropped from
`fit_exponential_to_csv`'s exception fallback.

### Bugs found and fixed during the review

- **`plot_geometry_diagnostics` ignored the wind normalization.** It called
  `simulate_lightcurve` without `wind_norm` / `mdot` / `v_inf` / `f_opacity`,
  so under physical mode every `*_geometry_phase.png` plotted an `N_H(φ)` and
  band flux from a *different* model than the one that was fitted. Now resolves
  `f_opacity` from the point estimate and passes the full normalization.
- **`--replot` dropped `fit_fopacity` and `kepler_mtot`.** `build_param_spec`
  was called without either, so a `kepler_mtot` run replotted as `phys` mode
  and `log_fopa` was mishandled. Both are now passed through.
- **`load_existing_results` had no `kepler_mtot` branch**, falling through to
  `PARAM_NAMES` and failing the geometry-column check. Now derives the geometry
  block from `param_spec.mode` via `get_mode_name_label`, and its "columns look
  like X mode" hint covers all four modes.
- **`--Delta` default inconsistency fixed** (a `known rough edge` in
  PROJECT.md): the `xrb_lightcurve.py` CLI defaulted to `1.0` while
  `default_wind_params` and `WIND_SHAPE_FIXED` used `2.0`, so CLI-generated
  models used a different break sharpness than the MCMC fitted. CLI now
  defaults to `2.0`.
- **`plot_wind_profile` marked the removed `H` radius**; now marks `Rb`/`ell`.
- **`--freeze` help** advertised the deleted `beta` and omitted `log_fopa`.

### Docs and support files

- `README.md` rewritten — it still documented the R port and removed API
  (`--lam2`, `flx2`/`fl2`, `nfl_*_av`/`_cv`, `pho_count_*`) and never mentioned
  the MCMC script.
- `requirements.txt` completed (`numba`, `zeus-mcmc`, `arviz`, `astropy`).
- `rkp_run_w_mcmc_cmds.sh` rewritten; it referenced flags removed long ago
  (`--lam2`, `--load-grid`, `--no-grid`, `--wind-model av/cv`, `--n-workers`,
  `--compute-waic`) and had broken line continuations. One-time destructive
  steps are commented out so it cannot clobber the gitignored XSPEC table.
- `utils/test_flux_methods.py` rewritten around `interpolate`/`refit`; it was
  built on the deleted legacy mode and could not import from `utils/`.
- `PROJECT.md` updated throughout.

### Verification (conda env `henv`)

- **Physical-mode output is bit-identical to `HEAD`** — `max_rel_diff = 0.0`
  across all 16 output columns for both `smooth_pl` and `confinement`,
  comparing against `git show HEAD:xrb_lightcurve.py` run with
  `--wind-norm physical --converge-rmax`. This is the key regression check:
  the trim changed no numerics.
- `xrb_lightcurve.py` CLI, `utils/test_flux_methods.py` (both methods),
  a full MCMC fit (reparam + jitter + `--fit-wind-shape` + `--fit-fopacity`)
  through every plot, `--replot` of that run, a `--kepler-mtot` fit and its
  replot, a `confinement` + `--freeze log_fopa=-1.7,ell=0.5` fit, and
  `chandra_phase_analysis.py --fit --write-model` all run clean.
- `--replot` reproduces the original run's `chi2/dof` exactly.
- `pack_wind_params` and `build_param_spec` both reject `beta_law` /
  `broken_pl` rather than silently accepting them.
- Repo-wide grep confirms no `lam` / `wind_norm` / `beta_law` / `broken_pl` /
  `Rmax` / `n_jobs` / `legacy` references remain in tracked Python or shell.

### Not updated

The notebooks still use the removed API — `xrb_model_analysis_single_15803.ipynb`
calls the deleted `compute_surface_density`. Untracked scripts
(`utils/benchmark_mcmc_performance.py`, `chandra_analysis_combined_flux.py`)
also still pass `--lam` and were left alone as legacy.

---

## Phase 32 — `beta_law` Wind Profile Restored

### Why

Phase 31 removed `beta_law` because it had never been used in a production
fit. For the methods paper it is the profile that follows most directly from
the physical normalization: mass continuity gives `n = Ṁ/(4π r² v(r))`, and
with `v̂ = v/v_inf → 1` the dimensionless shape is `g = 1/(r² v̂)` with `C = 1`,
so `n₀ = Ṁ/(4π R_sun² v_inf μ m_H)` is literally the terminal-velocity density.
Three profiles (power-law, confinement, velocity-law) also give the paper a
genuine model-comparison axis.

### The profile (Wind_Density.pdf §5)

```
v̂(r) = (1 − e^{−(r−R★)/H}) · (1 − R★/r)^β ,   g(r) = 1 / (r² v̂(r)) ,  r > R★
```

`g = 0` inside the photosphere. `v̂ → 0` at the surface, so `g` diverges there
(as `(r−R★)^{−(1+β)}`): rays grazing the limb are effectively opaque, which is
the physically expected behaviour of a dense acceleration zone. The effective
break radius is `R★ + 3H`.

### Changes

- `xrb_lightcurve.py`: `beta_law` registered as model id 2 with params
  `(R_star, beta, H)`; scalar branch in `_g_profile` (numba) and vectorized
  branch in `evaluate_g_profile` (uses `r ≤ R★ → inf` so `v̂ → 1`, `g → 0`
  without a negative base); `default_wind_params` → `beta = 1, H = 1`;
  `wind_asymptotic_coefficient` returns 1; new `R_STAR_TIED_MODELS =
  ("beta_law", "confinement")` drives the `R_star` auto-fill; CLI `--beta`,
  `--H`.
- `mcmc_lightcurve_fit.py`: `WIND_MODELS`, `WIND_SHAPE_FIT['beta_law'] =
  ['beta', 'H']` (both free, unlike the pre-Phase-31 version which fixed `H`;
  `--freeze H=…` recovers that), priors `beta ~ N(0.8, 0.3) on [0.3, 2]`,
  `H ~ N(1, 0.7) on [0.1, 10]`, labels, `--prior-beta/--prior-H` (generated
  from `ALL_WIND_SHAPE_NAMES`), `--freeze` help, `R_star` tying via
  `R_STAR_TIED_MODELS` in `_to_wind_params` and `DirectLightCurveModel`.
  The wind-profile diagnostic marks `R★ + 3H`.
- `utils/test_flux_methods.py`: now runs every `flux_method × wind_model`
  and checks that visible phases carry finite, non-negative flux and
  strictly positive column.
- Docs: README, PROJECT.md tables.

### Verification (conda env `henv`, 2026-09-17)

- `utils/test_flux_methods.py`: 6/6 (`interpolate`/`refit` × 3 profiles).
- **Regression:** `smooth_pl` and `confinement` outputs bit-identical to HEAD
  (max |diff| = 0.0 across all 12 numeric columns).
- **Physics:** with `R★ = 2`, `β = 0.8`, `H = 1` the `beta_law` column is
  1.05–1.17× that of a pure `r⁻²` wind of the same `Ṁ/v_inf` (slower inner
  wind is denser); mean N_H 0.185 vs 0.169 ×10²².
- **Quadrature:** per-ray GL16 vs adaptive reference — `beta_law` rel. error
  3e-6 at `b = R★ + 0.5`, 4e-8 at `+1`, ≤1e-10 beyond; degrades to 0.9 % at
  `b = R★ + 0.1` and 28 % at `+0.02` because `g` diverges at the surface.
  At the light-curve level (GL16 vs GL64) the max relative flux change is
  **≤ 9e-9** for `beta_law` (three `(β, H)` combinations), 8e-9 for
  `confinement`, 6e-16 for `smooth_pl`: the limb-grazing cells are opaque
  either way, so the local inaccuracy never reaches the flux.
- **MCMC smoke tests** on ObsID 15803 broad: (i) `--reparam --fit-wind-shape
  --fit-fopacity --likelihood jitter` samples `['a','q','r','R','i0','log_f',
  'log_fopa','beta','H']`, `--prior-beta/--prior-H` honoured, `R_star` tied to
  the fitted `R` (1.565 at the MAP), wind-profile plot marks `R★+3H`;
  (ii) `--kepler-mtot --freeze H=1.0 --likelihood chi2` samples 7 dims with
  `H` frozen and records `"freeze": "H=1.0"` in the run config; `--replot`
  regenerates every figure from the saved chain.
- Scale–opacity invariance (all lengths incl. `R_star`, `H` ×λ, `f_opacity`
  ×λ) holds for `beta_law` to round-off: max rel. flux change 1.2e-15,
  1.5e-15, 0.0 at λ = 0.8, 1.3, 2.0 — the same exact invariance as the other
  two profiles (Proposition 2 of the CLOAK paper).

---

## Side Investigation — Reference Epoch Recalibration

Plan: `reference_epoch_recalibration_ae1cf98a.plan.md`.

Per Laycock et al. 2015 (`stu2151.pdf`, §4), `T0 = 278801348 s = MJD 54040.87` is
the **mid-eclipse** time of ObsID 07082, defined to lie at **phase 0.5**, with a
full eclipse width ~0.2 in phase (~7 h); the paper folds with
`φ = (t - T0 - 100000P)/P`. The codebase formula `frac((t - T0)/P)` instead puts
that reference mid-eclipse at phase **0.0**, and the precomputed `phase` column
stored in the CIAO files reproduces neither exactly.

A read-only study script (`find_reference_epoch.py`) located mid-eclipse two
ways — anchored on ObsID 15803 (the only observation with a full clean eclipse,
~1.73 d > one 1.45 d orbit) via sliding-window / threshold / trapezoid-fit, and
via a joint all-observation scan over trial phase offsets maximizing eclipse
contrast — with the period held fixed at 125431 s.

Outcome: a corrected epoch of **`278800407.267 s`**, which sits **commented out**
next to `REF_EPOCH` in `chandra_phase_analysis.py`; the active value is still
`278801348`. In practice the Phase 17 per-sample phase-shift search absorbs the
offset, so this now mainly affects the interpretability of plotted phases. The
study script itself is **not present** in the working tree.

---

## Current File Inventory

### Core simulation / inference
| File                          | Lines | Status | Description                                            |
| ----------------------------- | ----- | ------ | ------------------------------------------------------ |
| `xrb_lightcurve.py`           | 2115  | active | Forward model: profiles, Numba GL LOS kernels, `simulate_lightcurve`, physical back-calculation helpers. |
| `mcmc_lightcurve_fit.py`      | 3553  | active | emcee/zeus MCMC: `ParamSpec` (phys/reparam/kepler), freeze, wind-shape, `f_scatter`, phase-shift search, BIC, ArviZ, replot. Direct evaluator only. |
| `chandra_phase_analysis.py`   |  457  | active | CLI front end for the single-model χ² fit; re-exports the shared `utils/` API for the notebooks. |
| `utils/utils.py`              | 1474  | active | Shared layer: ephemeris, loading, both binners, smoothing, periodic model interpolation + phase-shift search, `fit_simulation`, run-config persistence. |
| `utils/plot_utils.py`         |  597  | active | All plotting, built on the single `plot_lightcurve_fit` drawing routine. |
| `compute_flux_vs_nH.py`       |  934  | active | XSPEC table generator (flux vs nH).                    |
| `xspec_fit_mcmc.py`           |  702  | active | XSPEC-side MCMC for spectral fits.                     |
| `chandra_analysis_combined_flux.py` | 539 | active | Pre-folded combined-flux phase analysis.           |
| `plot_results.py`             |  104  | active | Thin CLI over `utils/plot_utils.py` (`--geometric`, `--orbit`). |
| `compute_count_to_flux_factor.py` | 147 | active | Count-rate → flux conversion factor.               |
| `example_usage.py`            |   96  | active | Programmatic `simulate_lightcurve` examples.           |

### XSPEC / shell helpers
`compare_models.sh`, `convert_fits_to_txt_heasoft.sh`,
`xspec_tbabs_fit_results.xcm`, `rkp_run_w_mcmc_cmds.sh` (command scrapbook —
contains dead flags from earlier phases).

*Missing from the working tree despite being referenced:*
`compare_absorption_models.xcm` (invoked by `compare_models.sh`),
`xspec_get_conversion_factors_tbabs.xcm`, `get_xspec_nH.py`,
`utils/get_conversion_factors.sh`, `find_reference_epoch.py`.

### Utilities (`utils/`)
Now a package (`__init__.py`). `utils.py` and `plot_utils.py` are library code
imported by both analysis scripts (see Core above). Standalone data-prep
scripts, not part of the package API: `add_flux_simple.py`,
`add_flux_to_lightcurves.py`, `convert_fits_to_txt.py`,
`get_average_count_rates.py`, `test_flux_methods.py`,
`benchmark_mcmc_performance.py`.

### Notebooks
`xrb_toy_wind_models.ipynb` (active wind-profile exploration),
`xrb_model_analysis.ipynb`, `xrb_model_analysis_single_15803.ipynb`,
`xrb_flux_nH_abs.ipynb`.

### Documentation
`PROJECT.md` (current-state reference),
`changes_tracked.md` (this file),
`mcmc_chi2_jitter_explanation.md`, `PERFORMANCE_VALIDATION_REPORT.md`,
`FLUX_INTEGRATION_SUMMARY.md`, `FLUX_METHODS_QUICKREF.md`,
`XSPEC_CONVERSION_GUIDE.md`, `FITS_CONVERSION_README.md`,
`QUICK_START_FLUX_CONVERSION.md`, `CONVERSION_WORKFLOW.md`,
`README_CONVERSION_TOOLS.md`.
Reference PDFs: `Wind_Density.pdf` (profile equations),
`stu2151.pdf` (Laycock et al. 2015 ephemeris).
`README.md` rewritten in Phase 31 as the user-facing overview.
**Stale:** `MIGRATION_SUMMARY.md` — still describes the original R→Python port
and removed API (`--lam2`, `flx2`/`fl2`, `nfl_*_av`/`_cv`, `pho_count_*`).

### Plans (`.cursor/plans/`)
`unified_wind_model_77726ced` (Phase 7),
`mcmc_performance_and_statistics_8989cd39` (Phase 8),
`mcmc_convergence_improvements_c648afb3` (Phase 9),
`mcmc_wind_shape_params_8b9c89d2` (Phase 10),
`flux_t_error_and_unbinned_mcmc_1bf81aa9` (Phase 12),
`wind_normalization_constants_76f1ef51` (Phase 13),
`mcmc_speed_memory_optimization_68fc497a` (Phase 15),
`freeze_params_and_kepler_3368d554` (Phase 16),
`adaptive_constant-snr_binning_249c8cba` (Phase 18),
`gaussian_phase_smoothing_reference_d1a46172` (Phase 19),
`reference_epoch_recalibration_ae1cf98a` (side investigation).

### Legacy (preserved)
`legacy_r_code/` (`new11.R`, `grid4.R`, `wind_los2.R`, `density_fnc.R`),
`light_curve_model_opt_bw.R`.

---

## Current Status & Quick Commands

See [PROJECT.md](PROJECT.md) for the full current-state reference (module-level
API, data layout, outputs, and known rough edges). Summary:

**Environment:** `henv` conda env (heasoft/XSPEC + python deps including
`numba`, `emcee`, `zeus-mcmc`, `arviz`, `corner`, `astropy`, `scipy>=1.12`).
`numba` is a hard requirement — `xrb_lightcurve.py` raises `ImportError`
without it.

**Working spectral model:** TBabs × powerlaw, nH = 0.75×10²² cm⁻²,
Γ = 1.86, χ²_red = 1.52.

**Default forward model:** `smooth_pl` wind (`Rb=5, p=4, Delta=2` everywhere),
single `nfl_{band}` flux column, Gauss-Legendre mega-kernel (~60 ms per light
curve), physical `Mdot`-based column normalization with per-cell flux
conversion. Wind models: `smooth_pl` / `confinement`. Flux methods:
`interpolate` (default) / `refit`.

**MCMC defaults:** `phys` mode (`d1, d2, r, R, i0`), `chi2` likelihood,
`emcee`, 50 fixed-width phase bins, per-sample phase-shift alignment **on**,
direct evaluator (no grid path exists). Parameterizations:
`phys` / `--reparam` / `--kepler` / `--kepler-mtot` (mutually exclusive).
`--fit-fopacity` is strongly recommended.

```bash
# Build/refresh the XSPEC flux-vs-nH table
python compute_flux_vs_nH.py --specdir ./data/IC10X1_spec --model tbabs \
    --bands broad soft medium hard --out_csv flux_vs_nH_tbabs_broad.csv

# FITS pipeline
./convert_fits_to_txt_heasoft.sh
python utils/add_flux_simple.py \
    data/IC_10_X1_LC/Broad_converted/ data/IC_10_X1_LC/Broad_with_flux/ 1.500509e-11

# Simulate one light curve. R is the true photosphere and the eclipse comes
# from wind opacity; f_opacity rescales the Mdot-derived column.
python xrb_lightcurve.py \
    --mdot 4e-6 --v-inf 1750 --f-opacity 0.03 \
    --r 1.4 --R 2.0 --d1 12.2 --d2 8.1 --i0 82 --dth 1 --d2h 6 \
    --wind-model smooth_pl --Rb 12 --p 6.7 --Delta 2 \
    --flux_method interpolate --flux_csv flux_vs_nH_tbabs_broad.csv \
    --output lc_physical.csv

# Single-model χ² fit + smoothed overlay + residual panel
python chandra_phase_analysis.py --data-dir data/IC_10_X1_LC_CIAO/broad \
    --obs-column flux_t --time-column t_raw \
    --fit --sim-file sim_broad.csv --rescale --smooth \
    --n-phase-bins 100 --output fit_broad.png

# MCMC: geometry only, reparameterized, adaptive constant-SNR bins
python mcmc_lightcurve_fit.py --band broad \
    --flux-csv flux_vs_nH_tbabs_broad.csv --data-dir data/IC_10_X1_LC_CIAO \
    --obs-column flux_t --time-column t_raw \
    --wind-model smooth_pl --reparam --likelihood chi2 \
    --counts-per-bin 100 --sampler zeus --dth 4.0 \
    --n-walkers 24 --n-steps 20000 --n-burn 2000 \
    --compute-bic --smooth --output-dir mcmc_results/broad/smooth_pl/geom

# MCMC: Kepler masses + wind shape + free scattered-flux floor
python mcmc_lightcurve_fit.py --band broad \
    --flux-csv flux_vs_nH_tbabs_broad.csv --data-dir data/IC_10_X1_LC_CIAO \
    --obs-column flux_t --time-column t_raw \
    --wind-model smooth_pl --fit-wind-shape --kepler --fit-scatter \
    --likelihood jitter --counts-per-bin 100 \
    --sampler zeus --n-walkers 24 --n-steps 21000 --n-burn 2000 \
    --n-threads 4 --dth 4.0 \
    --prior-MX 30,10,1,100 --prior-MRH 20,10,1,100 \
    --prior-Rb 6,3,3,80 --prior-p 4,2,2,8 \
    --compute-bic --output-dir mcmc_results/broad/smooth_pl/kepler_shape

# MCMC: raw unbinned 100s data (pair with jitter)
python mcmc_lightcurve_fit.py --band soft \
    --flux-csv flux_vs_nH_tbabs_soft.csv --data-dir data/IC_10_X1_LC_CIAO \
    --obs-column flux_t --time-column t_raw \
    --no-phase-bin --likelihood jitter --output-dir mcmc_results/soft/raw_jitter

# Freeze parameters out of the chain
python mcmc_lightcurve_fit.py --band broad --flux-csv flux_vs_nH_tbabs_broad.csv \
    --reparam --freeze q=0.5,Rb=6.0 --n-steps 2000 \
    --output-dir mcmc_results/broad/frozen

# MCMC: mass reparameterization with the wind shape fitted.
# R is the true photosphere here, not the effective eclipsing radius, and the
# eclipse is produced by wind opacity. Keep d2h small: flux is averaged
# *per emitter cell*, and d2h=30 gives only 13 azimuthal cells, which
# under-resolves the eclipse-core leakage.
python mcmc_lightcurve_fit.py --band broad \
    --flux-csv ./analyses/flux_vs_nH_tbabs_600bin_15803_broad.csv \
    --data-dir data/IC_10_X1_LC_CIAO/broad/single/ \
    --obs-column flux_t --time-column t_raw \
    --wind-model smooth_pl --kepler-mtot --fit-wind-shape --fit-scatter \
    --fit-fopacity --mdot 4e-6 --v-inf 1750 \
    --likelihood jitter --counts-per-bin 100 \
    --sampler zeus --n-walkers 24 --n-steps 30000 --n-burn 3000 \
    --n-threads 4 --dth 5.0 --d2h 6.0 --scatter-eclipse-phase 0.4 0.6 \
    --prior-Mtot 45,18,10,110 --prior-qm 0.60,0.12,0.05,0.95 \
    --prior-R 2.0,0.6,1.2,6 --prior-r 1.4,0.8,0.05,4 \
    --prior-i0 78,8,63,89.5 \
    --prior-Rb 8,4,2,40 --prior-p 4,1.5,2,10 \
    --compute-bic --smooth \
    --output-dir mcmc_results/broad/smooth_pl/single_15803_physical_mtot

# Re-plot / recompute BIC from saved results (pass the same data/binning flags)
python mcmc_lightcurve_fit.py --band broad --flux-csv flux_vs_nH_tbabs_broad.csv \
    --data-dir data/IC_10_X1_LC_CIAO --obs-column flux_t --time-column t_raw \
    --wind-model smooth_pl --counts-per-bin 100 \
    --replot --compute-bic --output-dir mcmc_results/broad/smooth_pl/geom
```

**Interpreting `M_X` / `M_RH`:** the light curve is independent of `q` in
*every* mode, so with `--kepler` the split between the two masses is set
entirely by the priors and must not be quoted as a measurement. Prefer
`--kepler-mtot`, which samples `(M_tot, q_m)`: with the physical normalization
`M_tot` is a weak measurement (`a` to ±10–15%, so `M_tot ∝ a³` to ~±35%) while
the split remains the `q_m` prior.
See [Phase 28](#phase-28--physical-wind-normalization--per-cell-flux-conversion),
[Phase 29](#phase-29--mass-reparameterization--error-column-fix) and
[Phase 31](#phase-31--release-trim-lam-wind-models-and-flux-methods).

**`--data-dir` resolves `{band}/single` before `{band}/`.** Pass the band
directory explicitly to control which observations are fitted.

**Notebooks are stale.** They have not been updated for the Phase 31 removals;
`notebooks/xrb_model_analysis_single_15803.ipynb` calls the deleted
`compute_surface_density`.

---

**Last Updated:** September 12, 2026  
**Maintainer:** R. Panchal
