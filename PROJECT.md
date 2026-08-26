# IC 10 X-1 Wind Absorption Light-Curve Project

Current-state reference for the simulation / fitting / inference stack in this
repository. Historical evolution (including code that has since been removed)
lives in [changes_tracked.md](changes_tracked.md).

---

## Contents

1. [Science goal](#science-goal)
2. [Architecture at a glance](#architecture-at-a-glance)
3. [`xrb_lightcurve.py` — forward model](#xrb_lightcurvepy--forward-model)
4. [`utils/` + `chandra_phase_analysis.py` — data, folding, binning, smoothing](#utils--chandra_phase_analysispy--data-folding-binning-smoothing)
5. [`mcmc_lightcurve_fit.py` — Bayesian inference](#mcmc_lightcurve_fitpy--bayesian-inference)
6. [Spectral / XSPEC side](#spectral--xspec-side)
7. [Data layout](#data-layout)
8. [Outputs](#outputs)
9. [Typical workflows](#typical-workflows)
10. [Environment](#environment)
11. [File inventory](#file-inventory)
12. [Known rough edges](#known-rough-edges)

---

## Science goal

**Target:** IC 10 X-1 — an eclipsing X-ray binary in the Local Group galaxy IC 10,
consisting of a compact object (+ accretion disk) orbiting a Wolf-Rayet companion.

The X-ray light curve is modulated by two effects as the compact object orbits:

1. **Geometric eclipse** — the companion occults the emitter around superior
   conjunction.
2. **Wind absorption** — outside eclipse, the line of sight (LOS) passes through
   the WR companion's dense stellar wind, and the phase-dependent hydrogen column
   density `N_H(φ)` attenuates the observed flux.

The project forward-models both effects from binary geometry + a parametric wind
density profile, converts `N_H(φ)` to band flux using an XSPEC-derived
`flux vs nH` table, and fits the result to phase-folded Chandra light curves —
either by simple χ² minimization or by full MCMC over geometry, wind-shape, and
nuisance parameters.

**System working values:** compact-object/disk radius `r ≈ 0.001 R☉`, companion
radius `R ≈ 2 R☉`, `d1 ≈ 11 R☉`, `d2 ≈ 8 R☉` (separation `a = d1 + d2 ≈ 19 R☉`),
inclination `i₀ ≈ 26°`, orbital period `P = 125431 s ≈ 1.45 d`.

**Adopted spectral model:** `TBabs × powerlaw` with `nH ≈ 0.75×10²² cm⁻²`,
`Γ ≈ 1.86`, `χ²_red ≈ 1.52` (preferred over `phabs` by `Δχ² ≈ 8.5`).

---

## Architecture at a glance

```
compute_flux_vs_nH.py  ──►  flux_vs_nH_*.csv        (XSPEC: nH → band flux)
                                    │
FITS light curves ──► utils/  ──►  data/…/*.txt      (time, counts, rate, flux_t)
                                    │                        │
                                    ▼                        ▼
                            xrb_lightcurve.py     chandra_phase_analysis.py
                          (geometry + wind LOS      (CLI front end: single-model
                           → nfl_{band} per phase)    χ² fit of a model CSV)
                                    │                        │
                                    └────────┬───────────────┘
                                             ▼
                                  mcmc_lightcurve_fit.py
                                (emcee/zeus posterior over
                                 geometry + wind shape + nuisance)

                    both analysis scripts sit on top of:
        utils/utils.py       ephemeris, loading, binning, smoothing,
                             periodic model interpolation, fit_simulation
        utils/plot_utils.py  plot_lightcurve_fit  ← the one drawing routine
                             (+ plot_phase / plot_multi_column_fits /
                              plot_corner / plot_trace / add_residual_panel /
                              plot_orbit_geometry / plot_geometry_vs_phase /
                              plot_wind_profile / plot_simulation_bands)
```

Neither analysis script imports the other. Everything they share lives in
`utils/`:

* **`utils/utils.py`** — ephemeris (`REF_EPOCH`, `ORBITAL_PERIOD`, `frac`),
  observation reading (`read_observation`, `load_data`,
  `resolve_band_directory`, `load_observed_lightcurves`), both binners
  (`phase_bin_data`, `phase_bin_data_snr` — their value columns keep the
  caller's names, so `flux`/`flux_err` needs no rename wrapper),
  `smooth_lightcurve`, `estimate_scattered_flux`, periodic model interpolation
  (`prepare_model_interpolator`, `model_from_wrap`, `evaluate_model_at_phases`,
  `interp_periodic_phases`), `obs_errors`, the tabulated-model χ² fit
  `fit_simulation`, the periodic phase-shift search it shares with the MCMC
  likelihood (`build_phase_shift_terms`, `apply_best_phase_shift`),
  `save_samples_csv_chunked`, and CLI run-config persistence
  (`save_run_config`, `apply_saved_run_config`). Standard library plus
  numpy / pandas / scipy only.
* **`utils/plot_utils.py`** — `plot_lightcurve_fit` is the **single** light-curve
  drawing routine (the one that used to be inlined in
  `mcmc_lightcurve_fit.plot_best_fit`). Callers evaluate their own model and hand
  it arrays: `plot_best_fit` resolves the posterior point estimate and the
  per-sample phase shift; `plot_phase` interpolates a simulation CSV. Because
  both routes end in one function, the data, overlay, smoothed curve, residual
  panel and title χ²/dof cannot drift apart. It also holds the geometry figures
  (`plot_orbit_geometry`, `plot_geometry_vs_phase`, `plot_wind_profile`) and the
  per-band simulation grid (`plot_simulation_bands`), which `plot_results.py`
  and `mcmc_lightcurve_fit.py` both drive.

`chandra_phase_analysis.py` is now only the CLI (≈460 lines, down from ~1640)
and re-exports every moved name, so `from chandra_phase_analysis import *` — the
notebooks' import style — is unchanged. It deliberately does **not** import
`xrb_lightcurve.py`: it consumes a model CSV, keeping the data layer decoupled
from the simulator.

Plot titles carry only the **energy band** and **χ²/dof**. Best-fit parameter
values are printed to stdout and written to the run summary rather than
annotated inside the axes.

---

## `xrb_lightcurve.py` — forward model

Pure simulator: binary geometry → per-phase wind column → band flux. No data,
no fitting.

### Wind density profiles

Every profile is expressed as a **dimensionless** shape function `g(r)`
(`r` in solar radii). Absolute amplitude is not part of `g`; it is absorbed by
the `lam` normalization step (below). Registry: `WIND_MODEL_IDS`,
`WIND_MODEL_PARAM_KEYS`.

| `wind_model`  | id | Parameters             | Form |
| ------------- | -- | ---------------------- | ---- |
| `broken_pl`   | 0  | `Rb, p`                | Piecewise power law: `(r/Rb)^-p` inside `Rb`, `(r/Rb)^-2` outside. |
| `smooth_pl`   | 1  | `Rb, p, Delta`         | **Default.** Smoothly broken PL: `(r/Rb)^-2 · [1 + (Rb/r)^Δ]^((p-2)/Δ)`. |
| `beta_law`    | 2  | `R_star, beta, H`      | CAK velocity law: `g = 1/(r²·v(r))` with `v ∝ (1-e^{-(r-R★)/H})(1-R★/r)^β`. |
| `confinement` | 3  | `R_star, fconf, ell`   | `1/r²` with inner exponential compression: `[1 + f_conf·e^{-(r-R★)/ℓ}]/r²`. |

Two implementations kept in lockstep:

- `_g_profile(r, model_id, p1..p4)` — `@njit(cache=True, inline="always")`, the
  version used inside the hot kernels. Scalar.
- `evaluate_g_profile(r, wind_model, wind_params)` — vectorized NumPy mirror, for
  helpers/notebooks.

`pack_wind_params(wind_model, wind_params)` flattens a dict into
`(model_id, p1, p2, p3, p4)` at the Python/Numba boundary (validating required
keys), so nothing dict-shaped enters the hot loop.
`default_wind_params(wind_model, R)` supplies sensible starting values.

### LOS integration kernels

**Fast path — `_simulate_phases_numba`** (`@njit(cache=True, parallel=True)`).
A single "mega-kernel" that computes *all* phases in one call, with `prange`
over phases. Per phase it:

1. Computes orbital geometry (`l`, `L`, `h`, `z_start`).
2. Runs the eclipse test. Gating is on `sin(gma) > 0` so an emitter *in front of*
   the companion is never spuriously occulted. If the compact-object disk lies
   fully behind the companion's projected disk, the phase is flagged
   `is_eclipsed` and short-circuits to zeros.
3. Otherwise walks the polar emitter grid (`n_th = 360/d2h + 1` rings ×
   10 radial cells) inline — trig tables precomputed once per call — masks cells
   blocked by the companion, forms segments from consecutive unmasked cells, and
   integrates each segment's LOS column.
4. Reduces to `flx = Σ(los·A)/ΣA`, `icd = Σ(los·A)`, `A2 = ΣA`.

**Quadrature — `_los_gl_quadrature`.** The LOS integral
`∫_{-∞}^{z_start} g(√(b²+z²)) dz` is evaluated by 16-point Gauss-Legendre
quadrature under the substitution `u = arctan(z/b)`:

```
∫ g(r) dz  =  b · ∫_{-π/2}^{u_start} g(b/cos u) · sec²(u) du
```

This maps the slowly decaying `r^-2` tail onto a bounded, smooth integrand on a
finite interval, so 16 fixed nodes give high accuracy for any profile and any
impact parameter — no `Rmax` heuristic, no special-casing `b` vs `Rb`. Nodes and
weights are module constants (`_GL16_X`, `_GL16_W`).

**Legacy path — `_wind_los_profile_numba`** (fixed-step trapezoid in `z`, with
adaptive convergence stopping or a hard `Rmax` cutoff). Reached only via the
standalone `wind_los_integral()` helper, or when `Rmax` is set explicitly
without `converge_rmax`. `create_grid()` / `density_function()` /
`wind_los_integral()` remain as the debuggable, per-phase Python API (plus a
vectorized NumPy fallback when Numba is unavailable).

Performance: one full `simulate_lightcurve` call is **≈ 60 ms** on a laptop,
which is what makes direct-evaluation MCMC feasible.

### `simulate_lightcurve(...)`

```python
simulate_lightcurve(
    r=0.001, R=2.0, d1=11.0, d2=8.0, gma0=-90.0, i0=26.0,
    dth=1.0, d2h=6.0, dz=0.5,
    flux_method="legacy", flux_csv_path=None, flux_type="erg",
    lam=0.589537,
    Rmax=None, converge_rmax=False,
    wind_model="smooth_pl", wind_params=None,
    scattered_flux=0.0,
    verbose=False, n_jobs=1,
) -> pd.DataFrame
```

Output columns:

| Column | Meaning |
| ------ | ------- |
| `deg`, `ph`, `phase`, `time` | Orbital phase in degrees / radians / 0–1 / seconds. |
| `l3`, `L3`, `h3` | Projected separation components. |
| `A2` | Total unmasked emitter area for that phase. |
| `flx` | Raw dimensionless mean LOS integral `⟨∫g dz⟩`. |
| `icd` | Area-weighted (unnormalized) column integral. |
| `is_eclipsed` | Bool — geometric total eclipse. |
| `fl` | Scaled column density in `10²² cm⁻²`, normalized so `mean(fl) = lam`. |
| `nfl_{band}` | Band flux after `nH → flux` conversion (one column per band). |

**Normalization.** `fl = flx · lam/mean(flx)`, i.e. the orbit-averaged column is
pinned to `lam` (taken from the spectral fit). This is why `g(r)` needs no
absolute scale, and why `lam` is held fixed in MCMC — wind *shape* is constrained
by the light-curve shape, wind *amplitude* by the spectrum.

**Eclipse flux.** During eclipse all `nfl_*` are forced to `0`. (Without this
they would collapse to *maximum* flux, since `flx = 0 ⇒ e⁰ = 1`.)

**`scattered_flux`.** A constant, phase-independent additive offset applied to
every `nfl_*` column after eclipse handling — for baking a scattered-light floor
into a directly generated model. The fit paths deliberately add scatter at
overlay/evaluation time instead, so that multiplicative rescaling doesn't
distort an additive constant.

### Flux conversion (`--flux_method`)

- `legacy` — hardcoded exponentials (`nfl_hard = 9.524·e^{-0.057 fl}`,
  `nfl_soft = 9.3923·e^{-2.5062 fl}`).
- `interpolate` — log-log interpolation of the XSPEC `flux vs nH` CSV
  (recommended). Bands auto-detected from `flux_{band}_{ph|erg}` columns.
- `refit` — refit `A·e^{-B·nH}` to the CSV per band.

`_FLUX_CACHE` (module-level, keyed by `(abs csv path, flux_type)`) caches the
cleaned/sorted arrays and the prebuilt `interp1d` objects, plus a per-band
exponential-fit cache — so an MCMC run reads and prepares the CSV once, not once
per likelihood call. `verbose=False` also suppresses the per-call "detected
bands" print and the extrapolation `UserWarning`.

### Physical back-calculation helpers

Constants: `R_SUN_CM`, `M_H_G`, `M_SUN_G`, `KM_TO_CM`.

Since the simulation only ever needs `g(r)`, the physical amplitude has to be
recovered afterwards from `lam`:

```
n_0 = (lam · 1e22) / (R_sun · mean(flx))        [cm^-3]
n(r) = n_0 · g(r)
```

- `compute_surface_density(sim_df, lam, R_star, wind_model, wind_params)` →
  `n(R_star)` in cm⁻³.
- `compute_wind_normalization_constants(lam, flx_mean, wind_model, wind_params, v_inf=None, mu=1.4)`
  → per-model constants: for `smooth_pl` the break density
  (`n_break_cm3`, `rho_b_g_cm3`); for `beta_law`/`confinement` the surface
  density plus the `Mdot/v_inf` prefactor (`mdot_over_vinf_g_per_cm`), and — if
  `v_inf` is supplied in km/s — `mdot_g_s` and `mdot_msun_yr`. `v_inf` is *not*
  a fitted parameter (it was absorbed into `n_0` by the `lam` normalization), so
  the caller must supply it to get a mass-loss rate.
- `wind_density_posterior(...)` and
  `wind_normalization_constants_posterior(...)` propagate MCMC samples through
  the above and return `{samples, median, p16, p84}` per constant.

---

## `utils/` + `chandra_phase_analysis.py` — data, folding, binning, smoothing

The shared analysis layer (`utils/utils.py`), the shared plotting layer
(`utils/plot_utils.py`), and the single-model (non-MCMC) CLI that drives them
(`chandra_phase_analysis.py`). Every name below is importable either from its
`utils` module or, unchanged, from `chandra_phase_analysis` (which re-exports
them for the notebooks).

### Ephemeris

```python
REF_EPOCH      = 278801348   # s
ORBITAL_PERIOD = 125431      # s
phase = frac((time - REF_EPOCH) / ORBITAL_PERIOD)
```

Phase is always recomputed from timestamps using these constants — the `phase`
column stored in the CIAO files is not trusted. A recalibrated candidate epoch
(`278800407.267`) sits commented out next to `REF_EPOCH`; see
[Known rough edges](#known-rough-edges).

### Reading observations

`read_observation(file_path, label, obs_column, obs_error_column, time_column, counts_column)`
handles three file shapes:

1. CIAO style with a `# Columns: dt, t_raw, mjd, phase, counts, rate, rate_err, flux_t` header.
2. Standard commented header containing TIME/RATE/FLUX-like names.
3. Headerless 3-column `time, rate, error` fallback.

Column resolution is case-insensitive with auto-detection for time
(`TIME`/`T_RAW`/`T`/`MJD`), the observable, the error
(`{col}_ERR`, `ERR_{col}`, `rate_err`, `count_rate_err`, …), and `counts`.

**`flux_t` error derivation.** CIAO files carry `flux_t` but no `flux_t_err`.
When no error column matches, `_derive_err_from_rate_err` derives per-row errors
proportionally, `err = rate_err · (flux_t / rate)`, falling back to a file-level
`cf = median(flux_t/rate)` for rows where `rate ≤ 0`. Output is normalized to
`time, rate, error, counts, phase, obs`. `load_data()` concatenates a directory
of `*.txt`.

### Binning

Two mutually exclusive binners:

- **`phase_bin_data(df, n_bins=50, min_points_per_bin=3, …)`** — fixed-width
  phase bins, inverse-variance weighted mean, `error = √(1/Σw)`; bins below
  `min_points_per_bin` are dropped. Variable counts per bin.
- **`phase_bin_data_snr(df, counts_per_bin=100, …)`** — adaptive
  *constant-counts* bins. Points are sorted by phase and accumulated greedily
  until each bin holds `counts_per_bin` counts, giving every binned point roughly
  equal Poisson weight (100 counts ⇒ SNR ≈ 10) and letting low-signal eclipse
  troughs merge into wide bins instead of many noisy narrow ones. Same weighted
  mean; additionally returns counts-weighted `phase` center, `total_counts`,
  `n_points`, and `phase_lo`/`phase_hi`/`width` for horizontal error bars. A
  trailing under-target bin is merged into its predecessor.

### Smoothing / residual primitives

Shared by both the single-model and MCMC plot paths:

- **`smooth_lightcurve(phase, flux, flux_err, sigma=0.01, n_eval=300, n_mc=2000, random_state=None)`**
  — periodic Gaussian-kernel phase smoother. Periodic distance
  `d = |((φ_i - φ_eval + 0.5) mod 1) - 0.5|`, weights `exp(-½(d/σ)²)`, so it is
  continuous across `phase = 0/1`. The kernel is phase-distance only (no
  inverse-variance weighting), matching the MATLAB reference in `temp/LC_MC/`.
  The 1σ band is a *vectorized* Monte Carlo: perturb all points at once and take
  one matmul, `std` over `n_mc` realizations. Works equally on fixed-width bins,
  constant-SNR bins, and raw unbinned data. `σ = 0.01` sits well below the
  ~0.1–0.25 phase scale of real features and above the ~0.0002 raw sampling.
- **`estimate_scattered_flux(phase, flux, window=(0.4, 0.6))`** — mean observed
  flux inside the mid-eclipse window (fallback `0.1 × median`, clamped ≥ 0).
  Used both as a fixed constant in the single-model path and as the prior center
  for the free `f_scatter` MCMC parameter.
- **`add_residual_panel(ax, phase, obs, model, err, xerr=None)`** — normalized
  pulls `(O-M)/σ` with `0` and `±1` reference lines.

### Single-model fit and plots

- **`fit_simulation(obs_df, sim_df, sim_column, fit_phase_shift=False, scatter=0.0)`**
  — χ² against an interpolated (wrap-around-safe) model curve. **Only the phase
  shift (x-direction) is fitted; there is no multiplicative flux scale.** The
  model's absolute normalization is already fixed by `lam` + the XSPEC
  flux-vs-nH table, so a free y-scale would silently absorb an error in that
  normalization instead of exposing it; the only y-direction freedom is the
  *additive* `scatter` floor, supplied by the caller (measured at mid-eclipse)
  rather than fitted. This matches `mcmc_lightcurve_fit.py`, which likewise
  fits a per-sample phase shift and an additive `f_scatter` but no scale.
  With `--fit-phase-shift` the shift is found by a coarse scan over the full
  period followed by bounded local refinement (χ²(shift) is periodic and
  strongly multi-modal, so a local optimizer started at 0 would settle in the
  wrong basin); otherwise the shift is held at 0. `dof = N - 1` when the shift
  is fitted, `N` otherwise. Returns `(shift, reduced_χ²)`.
- **`evaluate_model_at_phases(sim_df, sim_column, phases, shift, scatter)`** —
  the single definition of "model flux at these phases", built on
  `prepare_model_interpolator` (wrap-around `np.interp` arrays, accepts a
  `phase` or `deg` column) and `model_from_wrap` (which also accepts an
  array-valued `shift` so batched trial-shift scans use the identical
  expression). `fit_simulation`'s χ², the `plot_phase` overlay, and the residual
  panel all route through it, so they cannot silently disagree.
  `interp_periodic_phases(obs_phases, model_phase, model_flux)` is the
  array-in/array-out counterpart, used by the MCMC likelihood which rebuilds the
  curve every sample. `obs_errors(obs_df)` likewise centralizes uncertainty
  extraction (given errors else `sqrt(|rate|)`, with zero/negative/non-finite
  floored) so the fit and the residuals weight points identically.
- **`plot_lightcurve_fit(...)`** (`utils/plot_utils.py`) — **the one light-curve
  drawing routine**, shared with `mcmc_lightcurve_fit.plot_best_fit`. It draws
  only what it is handed (observed arrays, an already-shifted overlay curve, the
  model at the observed phases), which is what lets the MCMC path and the
  tabulated-simulation path share it. Observations get error bars when binned
  and a light scatter when not; adaptive-bin widths become horizontal error
  bars; an optional Gaussian-smoothed green curve carries its MC band. Becomes a
  2-panel figure (3:1 height ratio, shared x) with a normalized-residual panel
  whenever a model *and* errors are present. The title is only the energy band
  and χ²/dof. An optional `obs_group` splits the data into one series per
  observation.
- **`plot_phase(...)`** — the DataFrame/`sim_df` adapter over
  `plot_lightcurve_fit`: it interpolates the model at `shift` and `scatter`, then
  delegates. A coherent, non-zero-centered residual band is the diagnostic
  signature of a flux-normalization mismatch. Note `scatter` and `shift` must
  match the `fit_simulation` call — if a `chi2` is passed for display,
  `plot_phase` recomputes it from the curve it drew and **warns** when the two
  disagree by >1 %, so a mismatch surfaces immediately instead of printing a
  plausible number over the wrong curve.
- **`plot_multi_column_fits(...)`** — grid of `plot_phase` panels, one per
  simulation flux column; each panel's title names its energy band.
- `detect_flux_columns()` auto-detects `nfl_*` columns;
  `validate_sim_columns()` checks user-requested ones;
  `band_label_from_column("nfl_soft") -> "SOFT"` supplies the title label.

---

## `mcmc_lightcurve_fit.py` — Bayesian inference

Wraps the forward model in an emcee/zeus posterior sampler with configurable
parameterization, frozen parameters, wind-shape fitting, nuisance terms, and
diagnostics.

### Forward model: direct only

`DirectLightCurveModel` calls `simulate_lightcurve` per evaluation and
interpolates onto the requested phases. At ~60 ms/LC this is fast enough for
MCMC, it avoids interpolation artifacts, and it is the only path that supports
per-step varying wind shape. There is **no** precomputed-grid path any more —
the old `PrecomputedModelGrid` and its `--save-grid`/`--load-grid`/`--no-grid`
/`--grid-points` flags were removed.

`_interp_periodic_phases` interpolates the model onto observation phases with a
monotonic fast path (falling back to a sort) and a `[-1, 0, +1]` triple-tiling of
the model phase so wrap-around is exact.

### Parameterization: `ParamSpec`

One dataclass, built once in `main()` by `build_param_spec(...)` and threaded
everywhere, replaces ad-hoc positional `theta` indexing:

```python
@dataclass
class ParamSpec:
    mode: str                 # 'phys' | 'reparam' | 'kepler'
    active_names:  List[str]  # MCMC vector dimensions, in order
    active_labels: List[str]
    frozen: Dict[str, float]
    fit_wind_shape: bool
    fit_scatter: bool
    wind_model: str
    likelihood: str
    orbital_period_s: float
    K_kepler: float           # (G·M☉·P²/4π²)^(1/3) / R☉
```

Three geometry modes (mutually exclusive):

| Mode | Sampled | Derived |
| ---- | ------- | ------- |
| `phys` (default) | `d1, d2, r, R, i0` | — |
| `reparam` (`--reparam`) | `a, q, r, R, i0` | `d1 = a·q`, `d2 = a(1-q)` |
| `kepler` (`--kepler`) | `M_X, M_RH, r, R, i0` | `a = K·M_tot^{1/3}`, `q = M_RH/M_tot`, then `d1, d2` |

`--reparam` exists because `d1` and `d2` are strongly correlated — wind
absorption mostly sees their sum — so sampling `a = d1+d2` (well constrained) and
`q = d1/a` (weakly constrained) mixes far better. `--kepler` goes further and
samples component masses, with the separation fixed by Kepler's third law at
`--orbital-period` (default `ORBITAL_PERIOD`) and the lever arm `d1·M_X = d2·M_RH`.

Vector layout: `geometry → [log_f if jitter] → [f_scatter if --fit-scatter] →
[wind-shape params if --fit-wind-shape]`, minus anything frozen.

**Freezing.** `--freeze NAME=VAL[,NAME=VAL,…]` pins parameters and removes them
from the chain. Valid names: `d1, d2, a, q, r, R, i0, M_X, M_RH, f_scatter, Rb,
p, beta, fconf, ell` — shape params can be frozen even without
`--fit-wind-shape`. `log_f` cannot be frozen (use `--likelihood chi2`). Unknown
names are rejected with the allowed list; frozen values outside their prior box
warn but proceed; `Rb < R` with both frozen fails fast.

Resolvers: `_resolve_geom` (theta / frozen / Kepler), `_resolve_shape`
(generalizes `_to_wind_params`), `_resolve_scatter`, all keyed by name rather
than index.

### Wind-shape parameters

`--fit-wind-shape` promotes the chosen model's shape parameters to MCMC
dimensions:

| `--wind-model` | Free         | Fixed         | Tied to geometry |
| -------------- | ------------ | ------------- | ---------------- |
| `smooth_pl`    | `Rb, p`      | `Delta`       | — |
| `beta_law`     | `beta`       | `H = 1.0`     | `R_star = R` |
| `confinement`  | `fconf, ell` | —             | `R_star = R` |

Registries: `WIND_MODELS`, `WIND_SHAPE_FIT`, `WIND_SHAPE_FIXED`,
`WIND_SHAPE_LABELS`, `WIND_SHAPE_PRIORS`. `broken_pl` is not offered here since
`smooth_pl` generalizes it. Priors are overridable via
`--prior-Rb/-p/-beta/-fconf/-ell` using `mean,std,min,max`.

### Likelihoods

- **`chi2`** (default) — Gaussian: `-½ Σ (obs-model)²/σ²`.
- **`jitter`** — adds a free fractional systematic `log_f`; per-point variance
  becomes `σ²_eff = σ_obs² + (f·model)²` with `f = e^{log_f}`, and the
  likelihood carries the `+log σ²_eff` normalization. Recommended when fitting
  raw unbinned data, where formal errors underestimate the real scatter. Prior
  `JITTER_PRIOR = {mean:-3, std:2, min:-10, max:0}`. See
  [mcmc_chi2_jitter_explanation.md](mcmc_chi2_jitter_explanation.md) for the
  full derivation and an emcee-vs-zeus walkthrough.

### Priors

`log_prior` iterates `spec.active_names`: hard box rejection on
`(min, max)` per dimension, then a soft Gaussian penalty
`-½((θ-mean)/std)²`. Physical constraints are applied on *resolved* values (so
they hold under freezing and Kepler mode): `r < R` always, and `Rb ≥ R` for
`smooth_pl`. `reparam` mode adds the change-of-variables Jacobian `+log(a)`;
`phys` and `kepler` need none (their priors are already in the sampled space).
`get_active_priors` merges geometry + jitter + shape + scatter priors and drops
frozen entries.

### Per-sample phase-shift alignment

On by default. Rather than trusting the ephemeris to align model and data,
*every* likelihood call searches for the phase shift that minimizes weighted χ²:

1. Coarse uniform grid of `--phase-shift-grid-size` (default 25) shifts over
   `[0,1)`, with the model evaluated once on a dense
   `--phase-shift-eval-points` grid (default 240) and re-interpolated per shift.
2. Local refinement over 9 points spanning ±1 coarse step around the best shift.

`_build_phase_shift_terms` precomputes the shift grid, the evaluation grid, and
the shifted observation-phase matrix once per run. Because the shift is a
per-sample nuisance minimization (not a sampled parameter), it also applies
consistently in `compute_chi2_for_samples`, `compute_pointwise_loglik`,
`compute_bic_metrics`, and `plot_best_fit`. Disable with `--no-fit-phase-shift`.
`f_scatter` is phase-invariant and so is unaffected by the shift search.

### Samplers and parallelism

- `--sampler emcee` (default, stretch move) or `zeus` (ensemble slice sampler;
  better for correlated posteriors).
- Walkers initialized at `prior['mean'] ± 0.1·prior['std']`, clipped just inside
  each box; `r ≥ R` walkers are repaired when both are free.
- `--n-threads N > 1` opens a `spawn` multiprocessing pool. Because
  `simulate_lightcurve` is itself Numba-`parallel=True`, each worker calls
  `numba.set_num_threads(...)` via `_init_numba_worker` to avoid
  oversubscription; the default is `max(1, cpu_count // n_threads)`, overridable
  with `--numba-threads-per-worker`.

### Data path

`load_observed_lightcurves(band, data_dir, …)` resolves the band directory
through `_resolve_band_directory`, which tries, in order: `data_dir` itself,
`data_dir/{Band}_with_flux/`, `data_dir/{band}/single/`, `data_dir/{band}/`.
It delegates reading to `chandra_phase_analysis.load_data`, remaps to
`time, flux, flux_err, obs_id, counts, phase`, and drops non-positive /
non-finite flux rows (mostly zero-exposure GTI gaps, which carry no information
and would make `σ²_eff ≈ 0` degenerate under the jitter likelihood).

Binning mode is chosen by argument presence, not a mode flag:

| Flags | Behavior |
| ----- | -------- |
| `--no-phase-bin` | raw 100 s points (pair with `--likelihood jitter`) |
| `--counts-per-bin N` | adaptive constant-counts bins (recommended `100`) |
| `--n-phase-bins N` | fixed-width bins |
| neither | 50 fixed-width bins (backward-compatible default) |

Supplying both `--n-phase-bins` and `--counts-per-bin` is an error. Any residual
non-finite / non-positive errors are patched to
`max(0.1·|flux|, median(valid errors))` with a warning.

### Reporting and diagnostics

- `compute_statistics` — per-parameter `median`, `±1σ` from 16/84 percentiles,
  `mean`, `std`; derived `d1, d2` (and `a, q` in Kepler mode); plus a **MAP**
  entry (highest-log-prob single sample) when `log_prob` is available. The MAP
  point is used for overlays because it is algebraically self-consistent —
  `median(a·q) ≠ median(a)·median(q)`, so median rows generally do *not* satisfy
  `d1+d2 = a`.
- `print_diagnostics` — acceptance fraction, integrated autocorrelation times,
  effective independent samples, convergence flag (`n_steps > 50·max τ`).
- `run_arviz_diagnostics` — ArviZ summary (`r_hat`, `ess_*`, `mcse_*`, HDI),
  written to `*_arviz_summary.csv`. Version-agnostic via `_build_inference_data`.
- `compute_bic_metrics` — `BIC = k·ln n - 2 ln L̂`, with `k = len(active_names)`,
  `n = len(obs_flux)` *after* binning/filtering, and `L̂` evaluated by calling
  the run's actual likelihood at the max-log-prob sample (`theta_source =
  map_log_prob`, or `median_fallback`). BIC is the model-comparison metric;
  `ΔBIC` is reported relative to the best model in the run. Enable with
  `--compute-bic`.
- `compute_chi2_for_samples` (`--save-chi2`) — per-sample χ² and reduced χ² for
  the whole chain, gzip CSV. For jitter runs it emits *both* the classical
  measurement-error χ² (comparable across likelihood choices) and the
  effective-variance `chi2_eff`.
- `compute_pointwise_loglik` — per-observation log-likelihood matrix.

### Plots

All three live in `utils/plot_utils.py`.

- `plot_corner` — posterior corner plot with 16/50/84 quantiles.
- `plot_trace` — per-parameter walker traces with the burn-in marker.
- `plot_geometry_diagnostics` — three geometry figures at the point estimate,
  from one extra `simulate_lightcurve` call (skip with `--no-geometry-plots`):
  - **`*_geometry_orbit.png`** — the projected orbit against the companion disk,
    plus a to-scale top-down view. `(L3, h3)` from the simulation *are* the
    sky-plane coordinates of the compact object relative to the companion
    centre, so this is exact, not a sketch. The eclipse width constrains a
    *combination* of `(a, R, i0)`, so this is where an implausible-but-well-
    fitting parameter set becomes obvious; the footer states the numeric verdict
    (`min projected separation` vs `R ± r` → total / partial / no eclipse).
  - **`*_geometry_phase.png`** — projected separation `l3(φ)` against the
    `R ± r` thresholds with the eclipse shaded, the sky-plane components
    (`h > 0` ⇒ emitter behind, which is what gates the eclipse test),
    `N_H(φ)` with its orbit mean (should equal `lam`), and the band flux. Turns
    the eclipse from an emergent light-curve feature into a stated geometric
    condition with visible margin.
  - **`*_wind_profile.png`** — `g(r)` with 68/95% posterior credible bands, an
    `r⁻²` reference, the companion surface, characteristic radii (`Rb`/`H`/`ell`)
    and — the important part — the band of radii the line of sight actually
    probes. That band is `[min l3, max l3]`: the LOS impact parameter relative
    to the companion centre *equals* the projected separation, so the profile
    inside `min l3` is unconstrained by the data. Shape parameters are only
    interpretable jointly (`Rb` and `p` trade off strongly), so the constraint
    reads far more clearly here than in a corner plot.
- `plot_best_fit` (in `mcmc_lightcurve_fit.py`) — resolves the point estimate
  (MAP when available, else per-parameter medians), evaluates the model through
  `_evaluate_model` (the same entry point the likelihood uses, so geometry mode,
  wind shape, frozen values and the additive `f_scatter` are resolved once),
  finds the best phase shift, then hands the arrays to `plot_lightcurve_fit`.
  Result: a 2-panel (3:1) figure with the MAP overlay over the data, the optional
  smoothed green curve + MC band, and a normalized-residual panel clipped to
  ±5σ. The title is only the energy band and χ²/dof; the point estimate,
  `phase_shift`, `f`, `chi2_eff/dof` and `f_scatter` are printed to stdout
  (parameter values with their 1σ come from `print_results` and the summary
  file) rather than annotated on the figure. Returns the reduced χ² of the drawn
  model.

`--replot` regenerates everything from saved results without re-running MCMC:
`replot_from_existing` reads `*_samples.csv` for the posterior and
`*_chain.npz` for run metadata (`mode`, `frozen_names`/`frozen_values`,
`orbital_period_s`, `likelihood`, `wind_model`, `fit_wind_shape`), rebuilds the
`ParamSpec`, and auto-detects whether the saved chain contained shape params or
`f_scatter` from its column names.

**Every option not given explicitly is restored from `*_run_config.json`**, so
`python mcmc_lightcurve_fit.py --replot` on its own reproduces the original
band, wind model, `--flux-csv`, `--data-dir`, `--obs-column`/`--time-column`,
binning, `--lam`/`--dth`/`--d2h`, priors and model flags. This matters because
those options change the *observed arrays*: replotting with different binning
silently reports a χ²/dof for a dataset the posterior never saw. Explicit flags
always win over the saved values, so a single option can be overridden in place
(`--replot --smooth-sigma 0.02`). As a backstop, `replot_from_existing` compares
the observed point count against `n_obs` in the chain metadata and warns on a
mismatch. `--band` and `--flux-csv` are therefore only required when *not*
replotting.

---

## Spectral / XSPEC side

- **`compute_flux_vs_nH.py`** — the key upstream product. Loads a spectrum
  (PHA + background + responses) from `--specdir`, fits
  `{phabs,tbabs,wabs}×powerlaw` over `--fit_emin/--fit_emax`, freezes the
  powerlaw, then sweeps `nH` over a log grid and integrates band flux at each
  point. Emits `flux_vs_nH*.csv` with `nH_1e22`, `flux_{band}_ph`,
  `flux_{band}_erg` — exactly what `--flux_method interpolate` consumes.
  Chandra bands: `broad` 0.5–7.0, `soft` 0.5–2.0, `medium` 1.2–2.0,
  `hard` 2.0–7.0 keV.
- **`xspec_fit_mcmc.py`** — spectral-side MCMC for `phabs×(powerlaw+diskbb)`
  over 0.5–7 keV using XSPEC's Goodman-Weare chain engine, with optional corner
  plot.
- **`compute_count_to_flux_factor.py`** — derives the count-rate → flux
  conversion factor, either from an XSPEC fit or from manually supplied values.
- **`compare_models.sh`** — driver for the TBabs-vs-phabs comparison (logs to
  `model_comparison.log`).
- **`xspec_tbabs_fit_results.xcm`** — saved TBabs×powerlaw best fit.

Requires XSPEC in the active Python environment (the `henv` conda env).

---

## Data layout

```
data/
├── IC10X1_spec/                     spectra (PHA/PI + RMF/ARF + background)
├── ind_spectrum/                    per-observation spectra
├── IC_10_X1_LC/                     original + converted light curves
│   ├── {Broad,Soft,Hard}/           original FITS-masquerading-as-.txt
│   ├── {Broad,Soft,Hard}_converted/ plain text after FITS conversion
│   └── {Broad,Soft,Hard}_with_flux/ + FLUX / FLUX_ERR columns
├── IC_10_X1_LC_CIAO/                CIAO-reduced, preferred input
│   ├── {broad,soft,medium,hard,ultra_soft}/
│   │     <obsid>_100s_<band>_data_plus_flux.txt
│   │     └── single/                single-observation subsets (e.g. 15803)
└── combined_flux/                   pre-folded combined light curves
```

CIAO file columns: `dt, t_raw, mjd, phase, counts, rate, rate_err, flux_t`
(no `flux_t_err` — derived, see above). ObsIDs in play: 3953, 7082, 8458,
11080–11086, 15803, 26188 (+ 29793). 15803 (~1.73 d, longer than one orbit) is
the only observation containing a complete clean eclipse, which makes it the
natural single-observation test case.

Time-averaged count rates (cts/s): broad 0.1132, soft 0.0635, hard 0.0497.

Conversion pipeline: `utils/convert_fits_to_txt.py` (or
`convert_fits_to_txt_heasoft.sh`) → `utils/add_flux_simple.py` /
`utils/add_flux_to_lightcurves.py`, with factors from
`compute_count_to_flux_factor.py` and rates from
`utils/get_average_count_rates.py`.

---

## Outputs

Per `(band, wind_model)` in `--output-dir`, prefixed `{band}_{wind_model}_`:

| File | Contents |
| ---- | -------- |
| `*_samples.csv` | Flat post-burn-in samples + `log_prob` (chunked writer; skip with `--no-csv-output`). |
| `*_samples.npz` | Same, compact binary (`--compact-output`). |
| `*_chain.npz` | Full chain, log-prob, and run metadata (`mode`, frozen params, `likelihood`, `wind_model`, `n_obs`, …) for `--replot` / post-hoc BIC. |
| `*_run_config.json` | The complete CLI configuration of the fit (`created`, `command`, every argparse value). Written before sampling starts, so it survives an interrupted run. `--replot` restores from it. |
| `*_corner.png`, `*_trace.png`, `*_bestfit.png` | Diagnostic plots. |
| `*_geometry_orbit.png`, `*_geometry_phase.png`, `*_wind_profile.png` | Binary-geometry figures at the point estimate (`--no-geometry-plots` to skip). |
| `*_arviz_summary.csv` | ArviZ convergence table. |
| `*_model_metrics.csv` | `bic`, `logL_hat`, `k_params`, `n_obs`, `theta_source`. |
| `*_chi2.csv.gz` | Per-sample χ² (`--save-chi2`). |
| `mcmc_summary.txt` | Human-readable roll-up: run config, marginal posteriors, MAP block with a `d1+d2 == a` consistency check, reduced χ², BIC/ΔBIC, chain diagnostics. |

---

## Typical workflows

```bash
# 1. Build the XSPEC flux-vs-nH table (needs XSPEC / henv)
python compute_flux_vs_nH.py --specdir ./data/IC10X1_spec --model tbabs \
    --bands broad soft medium hard \
    --out_csv flux_vs_nH_tbabs_broad.csv --out_png flux_vs_nH_tbabs_broad.png \
    --nH_min 1e20 --nH_max 1e24 --nH_points 60

# 2. Generate a single simulated light curve
python xrb_lightcurve.py --flux_method interpolate \
    --flux_csv flux_vs_nH_tbabs_broad.csv \
    --wind-model smooth_pl --Rb 5 --p 4 --Delta 1 \
    --i0 12.0 --lam 0.572385 --output sim_broad.csv

# 3. Fold the data and χ²-fit that one model (phase shift free; flux never rescaled)
python chandra_phase_analysis.py \
    --data-dir data/IC_10_X1_LC_CIAO/broad \
    --obs-column flux_t --time-column t_raw \
    --fit --sim-file sim_broad.csv --fit-phase-shift \
    --smooth --n-phase-bins 100 --output fit_broad.png

# 4. MCMC — geometry only, adaptive constant-SNR bins
python mcmc_lightcurve_fit.py --band broad \
    --flux-csv flux_vs_nH_tbabs_broad.csv \
    --data-dir data/IC_10_X1_LC_CIAO \
    --obs-column flux_t --time-column t_raw \
    --wind-model smooth_pl --reparam --likelihood chi2 \
    --counts-per-bin 100 --sampler zeus --dth 4.0 \
    --n-walkers 24 --n-steps 20000 --n-burn 2000 \
    --compute-bic --smooth --output-dir mcmc_results/broad/smooth_pl/geom

# 5. MCMC — Kepler masses + wind shape + scattered-flux floor
python mcmc_lightcurve_fit.py --band broad \
    --flux-csv flux_vs_nH_tbabs_broad.csv \
    --data-dir data/IC_10_X1_LC_CIAO \
    --obs-column flux_t --time-column t_raw \
    --wind-model smooth_pl --fit-wind-shape --kepler --fit-scatter \
    --likelihood jitter --counts-per-bin 100 \
    --sampler zeus --n-walkers 24 --n-steps 21000 --n-burn 2000 \
    --n-threads 4 --dth 4.0 \
    --prior-MX 30,10,1,100 --prior-MRH 20,10,1,100 \
    --prior-Rb 6,3,3,80 --prior-p 4,2,2,8 \
    --compute-bic --output-dir mcmc_results/broad/smooth_pl/kepler_shape

# 6. Raw unbinned + jitter (no binning at all)
python mcmc_lightcurve_fit.py --band soft \
    --flux-csv flux_vs_nH_tbabs_soft.csv \
    --data-dir data/IC_10_X1_LC_CIAO \
    --obs-column flux_t --time-column t_raw \
    --no-phase-bin --likelihood jitter --wind-model smooth_pl \
    --output-dir mcmc_results/soft/raw_jitter

# 7. Freeze a parameter (1-D chain on the rest)
python mcmc_lightcurve_fit.py --band broad --flux-csv flux_vs_nH_tbabs_broad.csv \
    --reparam --freeze q=0.5,Rb=6.0 --n-steps 2000 \
    --output-dir mcmc_results/broad/frozen

# 8. Re-plot / recompute BIC from saved results (no sampling).
#    Everything is restored from <band>_<wind>_run_config.json, so this is the
#    whole command -- band, flux table, data selection, binning and priors all
#    come from the original fit:
python mcmc_lightcurve_fit.py --replot --output-dir mcmc_results/broad/smooth_pl/geom

#    Override a single option in place (explicit flags beat the saved config):
python mcmc_lightcurve_fit.py --replot --output-dir mcmc_results --smooth-sigma 0.02

#    For results predating run-config saving, pass the original options once;
#    a config is then written automatically for next time.
python mcmc_lightcurve_fit.py --band broad --flux-csv flux_vs_nH_tbabs_broad.csv \
    --data-dir data/IC_10_X1_LC_CIAO --obs-column flux_t --time-column t_raw \
    --wind-model smooth_pl --counts-per-bin 100 \
    --replot --compute-bic --output-dir mcmc_results/broad/smooth_pl/geom
```

> When re-plotting, pass the *same* data/binning flags as the original run —
> `--replot` re-loads and re-bins from `args`, and BIC depends on `n_obs`.

---

## Environment

Conda env `henv` (heasoft/XSPEC + Python deps). Beyond
[requirements.txt](requirements.txt) (`numpy`, `pandas`, `matplotlib`, `scipy`,
`emcee`, `corner`, `tqdm`), the current code also uses:

- **`numba`** — effectively required; there is a pure-NumPy fallback but it is
  orders of magnitude slower.
- **`arviz`** — convergence summaries (optional; degrades gracefully).
- **`zeus-mcmc`** — `--sampler zeus` (optional).
- **`astropy`** — FITS conversion utilities.
- **`joblib`** — only the legacy per-phase parallel path.
- **XSPEC Python (`pyxspec`)** — `compute_flux_vs_nH.py`, `xspec_fit_mcmc.py`,
  `compute_count_to_flux_factor.py`.

Note `requirements.txt` predates the numba/arviz/zeus dependencies.

---

## File inventory

### Core
| File | Lines | Role |
| ---- | ----- | ---- |
| [xrb_lightcurve.py](xrb_lightcurve.py) | 2115 | Forward model: profiles, Numba LOS kernels, `simulate_lightcurve`, physical back-calculation. |
| [mcmc_lightcurve_fit.py](mcmc_lightcurve_fit.py) | 3553 | emcee/zeus MCMC: `ParamSpec`, likelihoods, phase-shift search, BIC, `plot_best_fit`, replot. |
| [chandra_phase_analysis.py](chandra_phase_analysis.py) | 457 | CLI front end for the single-model χ² fit; re-exports the shared `utils/` API. |
| [utils/utils.py](utils/utils.py) | 1474 | Shared layer: ephemeris, loading, both binners, smoothing, periodic model interpolation + phase-shift search, `fit_simulation`, run-config persistence. |
| [utils/plot_utils.py](utils/plot_utils.py) | 597 | All plotting, built on the single `plot_lightcurve_fit`. |
| [compute_flux_vs_nH.py](compute_flux_vs_nH.py) | 934 | XSPEC `flux vs nH` table generator. |
| [xspec_fit_mcmc.py](xspec_fit_mcmc.py) | 702 | XSPEC-side spectral MCMC. |
| [chandra_analysis_combined_flux.py](chandra_analysis_combined_flux.py) | 539 | Fit pre-folded combined-flux files. |
| [plot_results.py](plot_results.py) | 104 | Thin CLI over `utils/plot_utils.py` for simulation CSVs (`--geometric`, `--orbit`). |
| [compute_count_to_flux_factor.py](compute_count_to_flux_factor.py) | 147 | Count-rate → flux factor. |
| [example_usage.py](example_usage.py) | 96 | Programmatic `simulate_lightcurve` examples. |

### Utilities (`utils/`)
`utils/` is a package (`__init__.py`). Two modules are library code imported by
the analysis scripts — `utils.py` and `plot_utils.py` (see Core above). The rest
are standalone data-prep scripts, not part of the package API:
`convert_fits_to_txt.py`, `add_flux_simple.py`, `add_flux_to_lightcurves.py`,
`get_average_count_rates.py`, `test_flux_methods.py`,
`benchmark_mcmc_performance.py`.

### Shell / XSPEC
`convert_fits_to_txt_heasoft.sh`, `compare_models.sh`,
`rkp_run_w_mcmc_cmds.sh` (command scrapbook),
`xspec_tbabs_fit_results.xcm`.

### Notebooks
`xrb_toy_wind_models.ipynb` (wind-profile exploration — the active one),
`xrb_model_analysis.ipynb`, `xrb_model_analysis_single_15803.ipynb`,
`xrb_flux_nH_abs.ipynb`.

### Documentation
| File | Contents |
| ---- | -------- |
| `PROJECT.md` | This file — current state. |
| [changes_tracked.md](changes_tracked.md) | Full change log, including removed features. |
| [mcmc_chi2_jitter_explanation.md](mcmc_chi2_jitter_explanation.md) | Likelihood/jitter math and emcee-vs-zeus internals. |
| [PERFORMANCE_VALIDATION_REPORT.md](PERFORMANCE_VALIDATION_REPORT.md) | Benchmark harness and parity thresholds. |
| `Wind_Density.pdf` | Source equations for the four wind profiles. |
| `stu2151.pdf` | Laycock et al. 2015 — ephemeris and eclipse properties. |
| `FLUX_INTEGRATION_SUMMARY.md`, `FLUX_METHODS_QUICKREF.md`, `XSPEC_CONVERSION_GUIDE.md`, `FITS_CONVERSION_README.md`, `QUICK_START_FLUX_CONVERSION.md`, `CONVERSION_WORKFLOW.md`, `README_CONVERSION_TOOLS.md` | Flux-conversion and FITS-pipeline guides. |
| [README.md](README.md), [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md) | **Stale** — describe the original R→Python port. |

### Plans (`.cursor/plans/`)
One `*.plan.md` per feature increment: `unified_wind_model`,
`mcmc_performance_and_statistics`, `mcmc_convergence_improvements`,
`mcmc_wind_shape_params`, `mcmc_speed_memory_optimization`,
`flux_t_error_and_unbinned_mcmc`, `freeze_params_and_kepler`,
`wind_normalization_constants`, `adaptive_constant-snr_binning`,
`gaussian_phase_smoothing_reference`, `reference_epoch_recalibration`.

### Legacy
`legacy_r_code/` (`new11.R`, `grid4.R`, `wind_los2.R`, `density_fnc.R`),
`light_curve_model_opt_bw.R`, `temp/LC_MC/*.m` (MATLAB smoothing reference).

---

## Known rough edges

- **`Delta` default is inconsistent.** `xrb_lightcurve.py --Delta` defaults to
  `1.0`, but `default_wind_params("smooth_pl")` and
  `mcmc_lightcurve_fit.WIND_SHAPE_FIXED['smooth_pl']` both use `2.0` (and the
  MCMC module docstring says "Delta is fixed at 2"). CLI runs and MCMC runs
  therefore use different break sharpness unless `--Delta` is passed explicitly.
- **`dz` default differs** between `xrb_lightcurve.py` (`0.5`) and the MCMC
  simulation group (`0.1`). Low impact: `dz` only affects the legacy
  trapezoid path, not the Gauss-Legendre mega-kernel used in practice.
- **Reference epoch is unresolved.** Laycock et al. define `T0` as the
  *mid-eclipse* time of ObsID 07082 at **phase 0.5**, whereas
  `frac((t-T0)/P)` puts it at phase 0.0. A recalibration study
  (`.cursor/plans/reference_epoch_recalibration_*.plan.md`) derived
  `278800407.267`, which sits commented out beside `REF_EPOCH`. In practice the
  MCMC's per-sample phase-shift search absorbs the offset, so this mostly
  affects the interpretability of plotted phases.
- **`README.md` and `MIGRATION_SUMMARY.md` are stale**, documenting removed
  API (`--lam2`, `flx2`/`fl2`, `nfl_*_av`/`_cv`, `pho_count_*`).
- **`rkp_run_w_mcmc_cmds.sh` contains dead flags** from earlier versions
  (`--wind-model av`, `--load-grid`, `--no-grid`, `--lam2`, `--compute-waic`,
  `--n-workers`). Use the [workflow examples](#typical-workflows) above instead.
- **Some referenced helper files are absent** from the working tree:
  `find_reference_epoch.py`, `compare_absorption_models.xcm` (which
  `compare_models.sh` invokes), `xspec_get_conversion_factors_tbabs.xcm`,
  `get_xspec_nH.py`, `utils/get_conversion_factors.sh`.
- **`.gitignore` excludes `*.csv`, `*.txt`, `*.png`**, so data, XSPEC tables,
  and figures are not version-controlled — inputs must be regenerated or copied
  in on a fresh clone.
- **`requirements.txt` is incomplete** (missing `numba`, `arviz`, `zeus-mcmc`,
  `astropy`).
- **`chandra_analysis_combined_flux.py` is an unmigrated fork.** It still
  carries its own older copies of `detect_flux_columns`, `validate_sim_columns`,
  `fit_simulation`, `plot_phase` and `plot_multi_column_fits`, has no `scatter`
  support, and still fits a **multiplicative flux scale** (`--rescale`) — the
  degree of freedom deliberately removed everywhere else. It should either
  import from `utils/` or be retired; until then its χ² values are not
  comparable to the main path's.
- **Uncommitted work in progress:** the Gaussian-smoothing / `f_scatter` /
  residual-panel feature set, plus the `utils/` extraction, are
  modified-but-uncommitted in `chandra_phase_analysis.py`,
  `mcmc_lightcurve_fit.py`, `xrb_lightcurve.py`, `utils/utils.py`,
  `utils/plot_utils.py` and the notebooks on branch `add_generic_wind`.
