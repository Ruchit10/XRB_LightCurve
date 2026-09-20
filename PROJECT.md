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
inclination `i₀ ≈ 78°` (standard convention, from the orbital-plane normal —
equivalently 12° from the line of sight, which is how the geometry kernels
measure it; the simulator's own default is 64°), orbital period
`P = 125431 s ≈ 1.45 d`. At these values the emitter is never geometrically
occulted (`min l ≈ 3.9 R☉ > R + r`): the dip is wind absorption.

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
                             (+ plot_phase / plot_corner / plot_trace /
                              add_residual_panel / plot_orbit_geometry /
                              plot_geometry_vs_phase / plot_wind_profile /
                              plot_simulation_bands)
```

The model is run **one energy band at a time**: each flux-vs-nH table holds one
band, each simulation yields one `nfl_{band}` column, and each MCMC run fits one
band with one wind model.

Neither analysis script imports the other. Everything they share lives in
`utils/`:

* **`utils/utils.py`** — ephemeris (`REF_EPOCH`, `ORBITAL_PERIOD`, `frac`),
  observation reading (`read_observation`, `load_data`,
  `resolve_band_directory`, `load_observed_lightcurves`), both binners
  (`phase_bin_data`, `phase_bin_data_snr` — their value columns keep the
  caller's names, so `flux`/`flux_err` needs no rename wrapper),
  `smooth_lightcurve`, `estimate_scattered_flux`, the single periodic model
  interpolator (`periodic_model`, `eval_periodic`, `prepare_model_interpolator`,
  `tabulated_model_arrays`), `sanitize_errors`, `obs_errors`, the shared row
  filters (`drop_invalid_flux_rows`, `apply_phase_window`), the tabulated-model χ² fit
  `fit_simulation`, the periodic phase-shift search it shares with the MCMC
  likelihood (`PhaseShiftSearch`, `build_phase_shift_search`,
  `best_phase_shift`), `save_samples_csv_chunked`, and CLI run-config
  persistence (`save_run_config`, `apply_saved_run_config`). Standard library
  plus numpy / pandas only.
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
(`r` in solar radii). Absolute amplitude is not part of `g`; it is supplied by
the physical normalization step (below). Registry: `WIND_MODEL_IDS`,
`WIND_MODEL_PARAM_KEYS`.

| `wind_model`  | id | Parameters             | Form |
| ------------- | -- | ---------------------- | ---- |
| `smooth_pl`   | 0  | `Rb, p, Delta`         | **Default.** Smoothly broken PL: `(r/Rb)^-2 · [1 + (Rb/r)^Δ]^((p-2)/Δ)`. |
| `confinement` | 1  | `R_star, fconf, ell`   | `1/r²` with inner exponential compression: `[1 + f_conf·e^{-(r-R★)/ℓ}]/r²`. |
| `beta_law`    | 2  | `R_star, beta, H`      | Velocity-based (Wind_Density.pdf §5): `g = 1/(r² v̂)`, `v̂ = (1 − e^{-(r-R★)/H})(1 − R★/r)^β`. Zero inside `R★`; diverges at the surface, so limb-grazing rays are opaque. Effective break `R★ + 3H`. |

All three relax to a constant-velocity `r⁻²` wind at large radius, which is
what lets `wind_asymptotic_coefficient()` tie `g` to a physical mass-loss rate.
`R_STAR_TIED_MODELS = ("beta_law", "confinement")` lists the profiles whose
`R_star` is auto-filled from the geometry `R`.

One implementation: `_g_profile(r, model_id, p1..p3)` (`@njit(cache=True,
inline="always")`, scalar, used inside the kernel) and the thin array wrapper
`evaluate_g_profile(r, wind_model, wind_params)` → `_g_profile_array`, which
loops over the same compiled function, so helpers and notebooks cannot drift
from the kernel.

`pack_wind_params(wind_model, wind_params)` flattens a dict into
`(model_id, p1, p2, p3)` at the Python/Numba boundary (validating required
keys), so nothing dict-shaped enters the hot loop.
`default_wind_params(wind_model, R)` supplies sensible starting values.

### LOS integration kernel

**`_simulate_phases_numba`** (`@njit(cache=True, parallel=True)`) is the only
integrator. A single kernel computes *all* phases in one call, with `prange`
over phases. Per phase it:

1. Computes orbital geometry (`l`, `L`, `h`, `z_start`) — all functions of the
   separation `a = d1 + d2` alone.
2. Runs the eclipse test. Gating is on `sin(gma) > 0` so an emitter *in front of*
   the companion is never spuriously occulted. If the compact-object disk lies
   fully behind the companion's projected disk, the phase is flagged
   `is_eclipsed` and short-circuits.
3. Otherwise walks the polar emitter grid: `n_th = 360/d2h` equal angular
   sectors × 10 radii. Each sector's occultation mask and impact parameter are
   both evaluated at the **sector centre**, and a radial segment between two
   consecutive unmasked radii gets one LOS integral and its annular-sector area
   (tables precomputed once per call). Sectors mirrored about the star–star
   line have identical geometry (the impact parameter depends on the sector
   angle only through its cosine, and the cosine table is made exactly
   symmetric), so only half of them are integrated and each result is recorded
   twice.
4. Reduces to `flx = Σ(los·A)/ΣA` and `A2 = ΣA`, and returns the **per-cell**
   column and area arrays that the nonlinear `nH → flux` conversion needs.

**Phase reflection (`_mirror_indices`).** The kernel sees the phase only
through `sin(gma)` and `|cos(gma)|` (`l`, `z_start`, the occultation test), so
`gma` and `π − gma` give identical columns and an `L` of opposite sign. On the
uniform grid `gma_k = gma0 + k·dth` the reflection maps index `k` to
`(m − k) mod n` with `m = (180 − 2·gma0)/dth`; whenever `m` is an integer (the
default `gma0 = −90` with any `dth` dividing 360) `_simulate_core` runs the
kernel and the flux conversion on one member of each pair only (181 of 360
phases at `dth = 1`) and copies the rest. The two halves of a full computation
already agreed only to trig round-off (≤ 2e-15, 6e-13 for rays grazing a
`beta_law` photosphere), so the copy changes nothing measurable.

The pre-Phase-33 kernel used `360/d2h + 1` rings, so the θ = 360° ring
duplicated θ = 0° (sector 0 carried double weight, `ΣA = 61/60` of the area),
and tested visibility at the sector's leading edge while integrating at its
centre. Both are gone. The occultation test is made at the centre of each
radial segment, the point its impact parameter is evaluated at. Testing both
bounding radii instead (the Phase-33 rule) dropped every segment straddling
the limb, which left the visible area and mean column of a partially eclipsed
extended emitter 40–60 % low at `r ~ R` at every `d2h`; against a Monte Carlo
of the disc the centre test is within ~1 %. Irrelevant for the default
`r = 0.001 R☉`, whose partial phases span 0.006° and are never sampled.

**Quadrature — `_los_gl_quadrature`.** The LOS integral
`∫_{-∞}^{z_start} g(√(b²+z²)) dz` is evaluated by 16-point Gauss-Legendre
quadrature under the substitution `u = arctan(z/b)`:

```
∫ g(r) dz  =  b · ∫_{-π/2}^{u_start} g(b/cos u) · sec²(u) du
```

This maps the slowly decaying `r^-2` tail onto a bounded, smooth integrand on a
finite interval, so 16 fixed nodes give better than 1e-5 for `smooth_pl` and
`confinement` at any impact parameter — the full `z`-tail is always
integrated, with no cutoff radius to choose and no special-casing of `b` vs
`Rb`. The `beta_law` profile diverges at the photosphere, so a ray grazing the
limb (`b − R⋆ < 0.3 R☉`) has an integrand peaked at closest approach that 16
nodes over the whole interval under-resolve (−30 % at `b − R⋆ = 0.01`, −90 %
at 0.001); such rays are integrated piecewise around the peak (`_gl_piece`),
which brings the error below 1e-7 against adaptive quadrature. Rays that cross
the photosphere (`b < R⋆`) have a divergent column and are opaque either way.
Nodes and weights are module constants (`_GL16_X`, `_GL16_W`).

**Numba is a hard requirement.** The module raises `ImportError` at import if it
is missing. There is no trapezoid fallback: the kernel is also the only path
that returns per-cell columns, and converting the *mean* column instead badly
understates eclipse-core leakage.

**Per-cell flux conversion is compiled too.** (The `refit` exponential is an
orientation-level approximation only: fitted with equal weights in log space
over a table spanning many decades, it misses the low-`nH` plateau by ~2× for
the broad band; `interpolate` is the quantitative path.) `_cell_flux_loglog` (linear
interpolation in log–log space, extrapolated from the upper end segment and
held at the first tabulated flux below the table — absorption cannot exceed 1,
and extrapolating the first segment returned up to 28 % above the plateau for
tables starting at 0.01 — with the column clipped to `[1e-6, 1e6] × 1e22`) and `_cell_flux_exp` (`A·e^{-B·N}`) each take
the kernel's per-cell columns and areas and return the area-averaged flux per
phase in one `prange` pass; the former reproduces the previous
`scipy.interp1d` path to 5e-15.

Performance: one full light curve (`smooth_pl`, `dth=1`, `d2h=6`) is
**≈ 12 ms** on a laptop (8 threads): ≈ 70 ms before Phase 33, 30 ms after it,
and 12 ms after Phase 34 halved the phases by reflection and cut the
`smooth_pl` profile from three `pow` calls per quadrature node to two (`x⁻²`
as a division, `x^−Δ` as the only pow besides the bracket). At the MCMC
default `dth = 2` a curve costs ≈ 7 ms, at `dth = 5` ≈ 4 ms.

### `simulate_lightcurve(...)` and `simulate_band_flux(...)`

```python
simulate_lightcurve(verbose=False, **kwargs) -> pd.DataFrame
simulate_band_flux(**kwargs)              -> (phase, flux)   # likelihood fast path

# keywords and their defaults, defined once on _simulate_core (= SIM_DEFAULTS):
#   r=0.001, R=2.0, d1=11.0, d2=8.0, gma0=-90.0, i0=64.0, dth=1.0, d2h=6.0,
#   flux_method="interpolate", flux_csv_path=None, flux_type="erg",
#   wind_model="smooth_pl", wind_params=None, scattered_flux=0.0,
#   mdot=4.0e-6, v_inf=1750.0, mu_wind=1.4, f_opacity=1.0, band=None
```

Both entry points forward their keywords to `_simulate_core`, whose keyword-only
signature is the single definition of the defaults: the `xrb_lightcurve.py`
CLI, the MCMC's `DirectLightCurveModel.sim_kwargs` and its argparse defaults
all read `SIM_DEFAULTS` (and the wind-shape CLI defaults come from
`default_wind_params`). A misspelled keyword raises `TypeError` — until Phase
34 `simulate_band_flux` pulled every argument with `kwargs.get(name, default)`
and silently ran the default for an unknown key.

`band` may be omitted when the flux table holds a single band; with a
multi-band table it is required. Both entry points share `_simulate_core`, so
the likelihood sees exactly the curve the DataFrame reports
(`utils/test_flux_methods.py` asserts this).

**Inclination convention.** `i0` is the standard astronomical inclination:
degrees from the orbital-plane normal, so `i0 = 90°` is edge-on (eclipses
possible) and `i0 = 0°` is face-on (the orbit lies in the plane of the sky and
never eclipses). The geometry kernel (`_simulate_phases_numba`) instead measures
`incl` from the *line of sight*, because
that is the angle appearing directly in `h = a·sin(γ)·sin(incl)` (sky-plane) and
`z = a·sin(γ)·cos(incl)` (along the LOS). `simulate_lightcurve` bridges the two
with `inclination_to_internal_rad(i0) = (90 − i0)·π/180`, at the input boundary
only — no geometry expression changed.

Output columns:

| Column | Meaning |
| ------ | ------- |
| `deg`, `phase` | Orbital phase in degrees / 0–1. |
| `l3`, `L3`, `h3` | Projected separation and its sky-plane components. |
| `A2` | Total visible emitter area for that phase (grid units). |
| `is_eclipsed` | Bool — geometric total eclipse. |
| `flx` | Raw dimensionless mean LOS integral `⟨∫g dz⟩`. |
| `fl` | Absolute mean column density `N_H` in `10²² cm⁻²`. |
| `nfl_{band}` | Band flux after the per-cell `nH → flux` conversion (one band per run). |

**Normalization.** `fl = flx · f_opacity · n₀ · R_sun / 1e22`, with `n₀` fixed
from `mdot`/`v_inf` (see below). The column therefore carries real units, so the
light curve constrains the *absolute* scale of the system rather than only
ratios such as `R/a`, the eclipse emerges from wind opacity instead of a
geometric cutoff, and `R` means the true photosphere.

**Per-cell flux conversion.** `nfl_{band}` is `⟨F(N)⟩` over the emitter disk,
*not* `F(⟨N⟩ ) = F(fl)`. The `nH → flux` map is strongly nonlinear, and during
ingress/egress and in the eclipse core the column varies by orders of magnitude
across the disk, so the surviving flux is dominated by the least-absorbed cells.
This is why the kernel returns per-cell columns and areas.

**Eclipse flux.** Eclipsed phases have no visible cells, so the area average
is 0 there by construction (before `scattered_flux` is added).

**`scattered_flux`.** A constant, phase-independent additive offset applied to
the band flux after eclipse handling — for baking a scattered-light floor into a
directly generated model. The fit paths add scatter at evaluation time instead.

### Flux conversion (`--flux_method`)

`--flux_csv` is required for both methods; the table's bands are detected from
its `flux_{band}_{flux_type}` columns.

- `interpolate` — **default.** Log-log interpolation of the XSPEC `flux vs nH`
  table.
- `refit` — `A·e^{-B·nH}` fitted to the table in log space (`fit_exponential`,
  a plain least-squares line fit, once per table and then cached). Smoother,
  but a single exponential cannot follow a table spanning many decades: with
  equal weights in log space it misses the low-`nH` plateau by ~2× for the
  broad band (worse for soft). Orientation only; `interpolate` is the
  quantitative path.

`_FLUX_CACHE` (module-level, keyed by `(abs csv path, flux_type)`) holds the
cleaned per-band arrays, their log10 grids for the compiled interpolator and the
exponential fit, so an MCMC run reads the CSV once, not once per likelihood
call (the `refit` path used to re-read it every call).

### Physical wind normalization

Constants: `R_SUN_CM`, `M_H_G`, `M_SUN_G`, `KM_TO_CM`, `YEAR_S`,
`MU_WIND_DEFAULT = 1.4`.

The absolute density is an **input**, derived from the mass-loss rate. Far from
the star every supported profile relaxes to `g(r) → C/r²`; matching that limit
to a spherical constant-velocity wind
`n(r) = Ṁ / (4π (r R_sun)² v_inf μ m_H)` gives

```
n_0 = Mdot / (4π · R_sun² · v_inf · μ · m_H · C)   [cm^-3]
n(r) = n_0 · g(r)
```

- `wind_asymptotic_coefficient(wind_model, wind_params)` → `C`: `Rb²` for
  `smooth_pl`, `1.0` for `confinement` and `beta_law` (both are written as
  `Mdot/(4π r² v_inf)` times a factor that tends to 1).
- `wind_density_norm_from_mdot(mdot_msun_yr, v_inf_kms, wind_model, wind_params, mu)`
  → `n_0`, called once per `simulate_lightcurve`.

`mu_wind = 1.4` converts the wind *mass* column into the equivalent-hydrogen
column the solar-abundance TBabs table expects.

**`f_opacity`.** A WR wind is hyper-ionized, clumped and He-rich, so its
effective photoelectric opacity is far below what its mass column implies.
`f_opacity` multiplies the Ṁ-derived column to absorb that difference. The
Clark & Crowther (2004) Ṁ overpredicts the observed `N_H` for IC 10 X-1 by
~1.5–2 dex, so values of ~0.01–0.03 are expected. In MCMC it is fitted as
`log10 f_opa` (`--fit-fopacity`) rather than assumed.

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
3. Headerless `time, rate[, error]` (2 or 3 columns; any other count is an
   error, because pandas would silently promote a surplus leading column to
   the index and shift every field by one) — used **only** when no header is
   found. Errors in the header path (unknown observable, header/data
   column-count mismatch, no time column) are raised; they used to be
   swallowed and the file silently re-read as three headerless columns.

Column resolution is case-insensitive (`find_column`) with auto-detection for
time (`TIME`/`T_RAW`/`T`/`MJD`), the observable, the error
(`_detect_error_column`: `{col}_ERR`, `ERR_{col}`, … — `rate_err` only when the
observable *is* the rate), and `counts`.

**`flux_t` error derivation.** CIAO files carry `flux_t` but no `flux_t_err`.
When no error column matches, `_derive_err_from_rate_err` derives per-row errors
proportionally, `err = rate_err · (flux_t / rate)`, falling back to a file-level
`cf = median(flux_t/rate)` for rows where `rate ≤ 0`. An error column whose
median is a thousand times the observable's (`--obs-error-column rate_err`
with `flux_t`) is refused as a unit mismatch. Output is normalized to
`time, rate, error, counts, exposure, phase, obs`: the exposure per row comes
from an `EXPOSURE`/`EXPTIME` column when present, else from `counts / rate`
(zero-count rows take the file's median exposure); an `mjd` time column is
converted to seconds since `MJDREF_CHANDRA` before the ephemeris.
`load_data()` concatenates a directory of `*.txt`.

### Binning

Two mutually exclusive binners, both reducing each bin with the shared
`bin_estimate(values, errors, exposure)`. With an exposure per row (count
data) the bin is the **exposure-weighted mean** `Σ(v t)/Σt` with the propagated
error `√Σ(t e)²/Σt`, which for `v = cN/t`, `e = c√N/t` is `c ΣN/Σt` with error
`c √ΣN/Σt` — the estimate one long exposure would give. Weighting rows by their
own Poisson errors instead (`1/e² ∝ t²/N`) made every bin the harmonic mean of
its counts, low by ~1/λ: −25 % at 3 counts per row, −12 % at 10, −6 % at 20
(measured −6 % to −27 % on synthetic data, 0.72 in-eclipse on the real broad
light curve), and it turned a zero-count row into infinite weight or, after
error repair, into a row weighted like any other. Zero-count rows now
contribute exposure and no variance; a bin without any counts gets error 0,
which the fitters' `sanitize_errors` pass turns into the median bin error.
Without an exposure the binners repair the row errors themselves and take the
inverse-variance mean (`weighted_mean`); without any error column they fall
back to the mean and `std/√n` (ddof = 1) of each bin, which both fitters
report with a warning (the χ² is then computed against scatter-derived errors,
which only `--no-phase-bin` refuses outright):

- **`phase_bin_data(df, n_bins=50, min_points_per_bin=3, …)`** — fixed-width
  phase bins; bins below `min_points_per_bin` are dropped. Variable counts per
  bin. Rows without a finite phase or value are dropped first (`np.digitize`
  would file a NaN phase in the last bin), and an empty result raises instead
  of returning a frame without a `phase` column.
- **`phase_bin_data_snr(df, counts_per_bin=100, …)`** — adaptive
  *constant-counts* bins. Points are sorted by phase and accumulated greedily
  until each bin holds `counts_per_bin` counts, giving every binned point roughly
  equal Poisson weight (100 counts ⇒ SNR ≈ 10) and letting low-signal eclipse
  troughs merge into wide bins instead of many noisy narrow ones. Same bin
  estimate; additionally returns the exposure-weighted `phase` centre (where
  the telescope looked, not where the counts were), `total_counts`,
  `n_points`, `exposure`, and `phase_lo`/`phase_hi`/`width` for horizontal
  error bars. A
  trailing under-target bin is merged into its predecessor. The counts must be
  finite, non-negative and not all zero, otherwise it raises: a NaN or
  all-zero column never reaches the target and used to collapse the whole
  light curve into a single bin.

### Smoothing / residual primitives

Shared by both the single-model and MCMC plot paths:

- **`smooth_lightcurve(phase, flux, flux_err, sigma=0.01, n_eval=300)`**
  — periodic Gaussian-kernel phase smoother. Periodic distance
  `d = |((φ_i - φ_eval + 0.5) mod 1) - 0.5|`, weights `exp(-½(d/σ)²)`, so it is
  continuous across `phase = 0/1`. The kernel is phase-distance only (no
  inverse-variance weighting), matching the MATLAB reference in `temp/LC_MC/`.
  The smoother is linear in the data, so its 1σ band is the exact
  `√(Σ w_i² σ_i²) / Σ w_i` (this replaced a 2000-draw Monte Carlo that was 25×
  slower and itself noisy at the 1–6 % level). Works equally on fixed-width
  bins, constant-SNR bins, and raw unbinned data. `σ = 0.01` sits well below the
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
  model's absolute normalization is already fixed by the wind mass-loss rate
  and the XSPEC flux-vs-nH table, so a free y-scale would silently absorb an error in that
  normalization instead of exposing it; the only y-direction freedom is the
  *additive* `scatter` floor, supplied by the caller (measured at mid-eclipse)
  rather than fitted. This matches `mcmc_lightcurve_fit.py`, which likewise
  fits a per-sample phase shift and an additive `f_scatter` but no scale.
  With `--fit-phase-shift` the shift comes from the shared `best_phase_shift`
  search (see *Per-sample phase-shift alignment* below); otherwise it is held
  at `fixed_shift` (`--phase-shift`, default 0). `dof = N - 1` when the shift
  is fitted, `N` otherwise. Returns `(shift, reduced_χ²)`. `--phase-window`
  restricts the observed points, with the same fixed-shift requirement as the
  MCMC.
- **`periodic_model(phase, flux)` / `eval_periodic(phase_ext, flux_ext, phases, shift, offset)`**
  — the single periodic interpolator: the curve folded into `[0, 1)`, sorted,
  duplicate abscissae removed, with one wrap point on each side so every query
  in `[0, 1)` is bracketed; `eval_periodic` accepts an array-valued `shift` so
  batched trial-shift scans use the identical expression.
  `prepare_model_interpolator(sim_df, column)` is the CSV front end (accepts a
  `phase` or `deg` column) and `tabulated_model_arrays` evaluates the overlay
  curve and the model at the observed phases once for both the plot and the
  model dump. `fit_simulation`'s χ², the `plot_phase`
  overlay, the residual panel, `write_model_lightcurve` and the MCMC likelihood
  all route through it, so they cannot silently disagree. (Before Phase 34 the
  MCMC and the tabulated path had separate interpolators whose wrap ranges
  differed: the `[0, 2)` tiling clamped queries below the first model phase.)
  `obs_errors(obs_df)` likewise centralizes uncertainty
  extraction: it requires an error column and applies the one repair rule,
  `sanitize_errors` (non-finite or non-positive errors replaced by the median
  valid error, with a warning; no absolute floor and no `sqrt(|rate|)`
  fallback, both of which depend on the units of the data and zero-weighted or
  "perfectly fitted" flux points), so the fit and the residuals weight points
  identically.
- **`plot_lightcurve_fit(...)`** (`utils/plot_utils.py`) — **the one light-curve
  drawing routine**, shared with `mcmc_lightcurve_fit.plot_best_fit`. It draws
  only what it is handed (observed arrays, an already-shifted overlay curve, the
  model at the observed phases), which is what lets the MCMC path and the
  tabulated-simulation path share it. Observations get error bars when binned
  and a light scatter when not; adaptive-bin widths become horizontal error
  bars; an optional Gaussian-smoothed green curve carries its analytic 1σ band. Becomes a
  2-panel figure (3:1 height ratio, shared x) with a normalized-residual panel
  whenever a model *and* errors are present. The title is only the energy band
  and χ²/dof. An optional `obs_group` splits the data into one series per
  observation.
- **`plot_phase(...)`** — the DataFrame/`sim_df` adapter over
  `plot_lightcurve_fit`: it interpolates the model at `shift` and `scatter`, then
  delegates. A coherent, non-zero-centered residual band is the diagnostic
  signature of a flux-normalization mismatch. `scatter` and `shift` must match
  the `fit_simulation` call; both fitters pass the values they fitted, so the
  χ² in the title belongs to the curve drawn.
- `detect_flux_columns()` lists the `nfl_*` columns of a simulation CSV — the
  CLI uses the single one present unless `--sim-column` names it;
  `band_label_from_column("nfl_soft") -> "SOFT"` supplies the title label.

---

## `mcmc_lightcurve_fit.py` — Bayesian inference

Wraps the forward model in an emcee/zeus posterior sampler with configurable
parameterization, frozen parameters, wind-shape fitting, nuisance terms, and
diagnostics. One band and one wind model per run (≈ 1900 lines after the
Phase 33 consolidation).

### Forward model: direct only

`DirectLightCurveModel` holds the run's band, flux table, wind model, `dth`
(default 2°) and simulation constants, and validates the band against the
table at construction (a band missing from the table used to surface only as
`-inf` on every likelihood call, which emcee samples to completion with frozen
walkers); `curve(d1, d2, r, R, i0, wind_params, f_opacity)` calls
`simulate_band_flux` and returns the kernel's native `(phase, flux)` arrays. Nothing is resampled: the likelihood interpolates that
curve once, directly onto the (shifted) observed phases. At ≈ 7 ms per curve
this is fast enough for MCMC and is the only path that supports per-sample
wind-shape parameters.

`FitData` bundles the observed arrays the likelihood needs (`phase`, `flux`,
`err`, `err2`, bin widths) and the precomputed `PhaseShiftSearch` (`None` when
the shift is held fixed at `fixed_shift`).

### Parameterization: `ParamSpec`

One dataclass, built once in `main()` by `build_param_spec(...)`, is the single
answer to "which value does this name have for this sample":

```python
@dataclass
class ParamSpec:
    mode: str                 # 'phys' | 'reparam' | 'kepler' | 'kepler_mtot'
    active_names: List[str]   # MCMC vector dimensions, in order
    frozen: Dict[str, float]
    fit_wind_shape: bool; fit_scatter: bool; fit_fopacity: bool
    wind_model: str; likelihood: str
    orbital_period_s: float; K_kepler: float   # a = K·M_tot^(1/3), R☉

    value(theta, name)          # sampled or frozen value, else None
    geometry(theta)             # -> (d1, d2, r, R, i0), NaNs if unphysical
    wind_params(theta, R)       # shape dict for the simulator (unfitted names: simulator defaults, R_star = R)
    f_scatter(theta); f_opacity(theta)
    derived(rows)               # vectorized a/q/d1/d2/M_X/M_RH per mode
```

Every consumer — `log_prior`, `log_likelihood`, `compute_statistics`,
`compute_chi2_for_samples`, `compute_bic_metrics`, `plot_best_fit`,
`plot_geometry_diagnostics`, `write_summary`, `replot_from_existing` — takes the
`ParamSpec` and nothing else; there is no parallel `reparam/kepler/active_names`
argument path. `model_curve(theta, spec, model)` is the one place the physical
model (including the additive `f_scatter`) is evaluated, and
`aligned_model_flux` / `chi2_terms` sit on top of it for the phase-shift search
and the χ² reports.

The `MODES` registry defines the four parameterizations (scale parameters,
their priors, the CLI flag and the derived quantities); `r`, `R`, `i0` and their
priors (`SMALL_R_PRIOR`, `R_PRIOR`, `I0_PRIOR`) are shared by all of them:

| Mode | Sampled | Derived | Flat direction |
| ---- | ------- | ------- | -------------- |
| `phys` (default) | `d1, d2, r, R, i0` | `a, q` | diagonal in `(d1, d2)` |
| `reparam` (`--reparam`) | `a, q, r, R, i0` | `d1, d2` | `q` axis |
| `kepler` (`--kepler`) | `M_X, M_RH, r, R, i0` | `a, q, d1, d2` | diagonal in `(M_X, M_RH)` |
| `kepler_mtot` (`--kepler-mtot`) | `M_tot, q_m, r, R, i0` | `a, M_X, M_RH, d1, d2` | `q_m` axis |

For a circular orbit the light curve depends on `a = d1 + d2` alone, so `q`
and `q_m` are *exactly* unidentifiable (their posteriors equal their priors);
the reparameterized modes put that flat direction on its own axis. Under the
physical normalization, scaling every length together with `f_opacity` is also
an exact invariance, so `M_tot` is anchored only by the priors on `R` and
`f_opacity`.

Vector layout: `geometry → [log_f if jitter] → [f_scatter] → [log_fopa] →
[wind-shape params]`, minus anything frozen.

**Freezing.** `--freeze NAME=VAL[,NAME=VAL,…]` pins parameters and removes them
from the chain. Valid names: the mode's geometry parameters, `f_scatter`,
`log_fopa`, and the wind model's shape parameters (freezable even without
`--fit-wind-shape`). `log_f` cannot be frozen (use `--likelihood chi2`). Unknown
names are rejected with the allowed list; frozen values outside their prior box
warn but proceed; `Rb < R` with both frozen fails fast.

**Priors.** `R ~ N(2, 0.5)` on `[1, 5]` R☉ (the photosphere — the eclipse comes
from wind opacity), `r ~ N(0.001, 0.001)` on `[1e-4, 0.1]`, and
`i0 ~ N(78, 8)` on `[40, 89.9]`° in every mode. (Before Phase 33 the
`kepler_mtot` defaults still carried lam-era values, `R ~ N(9.5, 2.5)` on
`[3, 20]`, which excluded the photosphere, and the other modes capped `i0` at
80°.)

### Wind-shape parameters

`--fit-wind-shape` promotes the chosen model's shape parameters to MCMC
dimensions:

| `--wind-model` | Free         | Fixed         | Tied to geometry |
| -------------- | ------------ | ------------- | ---------------- |
| `smooth_pl`    | `Rb, p`      | `Delta = 2.0` | — |
| `confinement`  | `fconf, ell` | —             | `R_star = R` |
| `beta_law`     | `beta, H`    | —             | `R_star = R` |

Registries: `WIND_MODELS`, `WIND_SHAPE_FIT`,
`WIND_SHAPE_PRIORS`, `ALL_WIND_SHAPE_NAMES`; plot labels for every parameter
live in `PARAM_LABELS`. Priors are overridable via
`--prior-Rb/-p/-fconf/-ell/-beta/-H` using `mean,std,min,max`.
`beta_law` frees both `beta` and `H` so that every profile has two shape
dimensions; `--freeze H=1.0` recovers a one-parameter beta-law fit.

`--fit-fopacity` additionally promotes `log10 f_opacity` to a free dimension
(prior `FOPACITY_PRIOR`, centred at `-1.5`; override with `--prior-fopa`).

### Likelihoods

- **`chi2`** (default) — Gaussian: `-½ Σ (obs-model)²/σ²`.
- **`jitter`** — adds a free fractional systematic `log_f`; per-point variance
  becomes `σ²_eff = σ_obs² + (f·model)²` with `f = e^{log_f}`, and the
  likelihood carries the `+log σ²_eff` normalization. Recommended when fitting
  raw unbinned data, where formal errors underestimate the real scatter, and
  for binned real data, which carries ~20–25 % intrinsic variability. Prior
  `JITTER_PRIOR = {mean:-3, std:2, min:-10, max:0}`.

### Priors

`log_prior` iterates `spec.active_names`: hard box rejection on
`(min, max)` per dimension, then a soft Gaussian penalty
`-½((θ-mean)/std)²`. Physical constraints are applied on *resolved* values (so
they hold under freezing and Kepler mode): `r < R` always, and `Rb ≥ R` for
`smooth_pl`. Priors are stated directly on the sampled parameters of every
mode, so no change-of-variables Jacobian is applied. (Before Phase 34 `reparam`
added `+log(a)` on top of Gaussian priors already stated on `a` and `q`, which
tilted the effective prior to `a·N(a)`, +4.3 % in the mode of `a`; because `a`
is prior-anchored through the exact scale invariance, that leaked ~+13 % into
a derived `M_tot`. The Kepler modes never had such a term.) `get_active_priors`
merges geometry + jitter + shape + scatter priors and drops frozen entries.

Unfitted, unfrozen wind-shape parameters take the **simulator defaults**
(`default_wind_params`, which also ties `R_star` to `R`), never prior means:
freezing one shape parameter therefore leaves the others unchanged, and
`xrb_lightcurve.py` with the same `--freeze` values reproduces the fitted
curve exactly. The registries are checked against `xrb_lightcurve` at import
(`WIND_MODELS` keys = `WIND_MODEL_IDS`, `WIND_SHAPE_FIT` ⊆
`WIND_MODEL_PARAM_KEYS`).

### Per-sample phase-shift alignment

On by default. Rather than trusting the ephemeris to align model and data,
*every* likelihood call searches for the phase shift that maximizes the
likelihood being sampled, with `utils.best_phase_shift`, the same routine
`fit_simulation` uses (the objective is the weighted χ² under the `chi2`
likelihood; under `jitter` it is `Σ[(d−m)²/s² + ln s²]` with `s² = σ² +
(f·m)²`, so the profiled shift is the jitter likelihood's own — profiling it on
the classical χ² instead differed by 16 log-units and 0.007 in phase at
`f = 0.45`):

1. One `np.interp` over the precomputed `(n_grid, n_obs)` matrix of shifted
   observed phases gives χ² at every coarse shift. The coarse step must not
   exceed the narrowest feature of χ²(shift), which is set by the data spacing
   (a bin crossing the eclipse edge) and the model spacing (one `dth`), so
   `n_grid = max(n_obs, 360/dth)` clipped to `[25, 400]`
   (`--phase-shift-grid-size` overrides).
2. Two dense passes of 33 points, each spanning ±1 previous step around the
   best shift, bring the resolution to `step/256` (2e-5 for 180 coarse shifts).

Measured against a brute-force minimum on the real 150-bin light curve, the
profiled χ² is within 0.006 of the true minimum for every sample. The
pre-Phase-34 search (25-point grid, 9-point refinement) quantised the shift to
0.01, coarser than the bin width, and sat a median 1.9 and up to 24 χ² units
above the minimum; its intermediate 240-point resample of the model added a
further ±7 units. The search costs 0.6 ms per call.

`build_phase_shift_search` precomputes the trial shifts once per run (a frozen
`PhaseShiftSearch` stored in `FitData`). Because the shift is a per-sample
nuisance minimization (not a sampled parameter), every consumer goes through
`aligned_model_flux`, so the likelihood, `compute_chi2_for_samples`,
`compute_bic_metrics` and `plot_best_fit` apply it identically. Disable with
`--no-fit-phase-shift` (shift held at 0) or hold it at a chosen value with
`--phase-shift SHIFT`. `f_scatter` is phase-invariant and so is unaffected by
the shift search.

**Fitting part of the orbit.** `--phase-window LO HI` (default `0 1`; `LO > HI`
wraps through phase 0) keeps only the observed points inside the window; the
model is still evaluated over the full orbit, which costs nothing extra because
the kernel already computes only the unique half. The purpose is to fit ingress
and egress separately: the model is exactly symmetric about mid-eclipse, so two
independent half-orbit fits of asymmetric data are a clean asymmetry test.
Because of that same symmetry a partial window **requires a fixed shift**: with
only one eclipse edge in the data, a narrow eclipse centred nearby and a wide
one centred further away fit the edge equally well, so the eclipse width (the
combination of `a`, `R`, `i0`) is degenerate with a free shift. `main()` refuses
a partial window unless `--phase-shift` or `--no-fit-phase-shift` is given; the
intended workflow is a full-orbit fit first, then half-orbit fits with its shift
frozen. The split should sit at the fitted mid-eclipse phase (≈ 0.485 for the
current ephemeris), and the `f_scatter` prior window must overlap the data
window. Both options are fit-defining and are restored by `--replot`.
`chandra_phase_analysis.py` has the same two options for the tabulated fit.

### Samplers and parallelism

- `--sampler emcee` (default, stretch move) or `zeus` (ensemble slice sampler;
  better for correlated posteriors).
- Walkers initialized at `prior['mean'] ± 0.1·prior['std']`, clipped just inside
  each box; `r ≥ R` walkers are repaired when both are free. The initial
  ensemble is evaluated before sampling: the run aborts if every walker is at
  `-inf`; walkers at `-inf` (outside `r < R`, `Rb ≥ R` or a box) are redrawn
  up to 20 times, because zeus refuses to start from any `-inf` walker where
  emcee would merely reject moves from it; emcee starts from the evaluated
  `State`.
  `main()` rejects `--n-burn ≥ --n-steps`, which used to fail only after the
  full run with an empty chain.
- `--seed N` makes a run reproducible: `np.random.seed` covers the initial
  ball, zeus, the `--save-chi2` subsample and the wind-profile draws, and emcee
  is handed the same global state (`sampler.random_state`). The seed is
  recorded in the run metadata and the summary; it is also accepted on
  `--replot`, whose chi2 subsample and wind-profile draws are random too.
- `main()` exits with status 1 on any failure (2 for argument errors), so shell
  chains and batch jobs can tell a failed fit from a finished one.
- `--n-threads N > 1` opens a `spawn` multiprocessing pool. The fit context
  (`ParamSpec`, priors, model, `FitData`) is sent to each worker once through
  the pool initializer `_init_worker`, and the sampler calls
  `_log_probability_worker(theta)`, so only `theta` crosses the pipe per task
  (emcee and zeus would otherwise pickle every argument into every task, ~900 KB
  per sample for unbinned data). Because the kernel is itself
  Numba-`parallel=True`, the initializer also calls `numba.set_num_threads(...)`
  to avoid oversubscription; the default is `max(1, cpu_count // n_threads)`,
  overridable with `--numba-threads-per-worker`.

### Data path

`load_fit_data(args, band)` wraps `load_observed_lightcurves(band, data_dir, …)`,
which resolves the band directory through `resolve_band_directory` (tries, in
order: `data_dir` itself, `data_dir/{Band}_with_flux/`,
`data_dir/{band}/single/`, `data_dir/{band}/`), reads via `utils.load_data`,
remaps to `time, flux, flux_err, obs_id, phase` (plus `counts` only when the
files carry them: an all-NaN column would pass the constant-counts binner's
presence check), drops rows with a zero or undefined exposure (nothing was
observed; only possible when the file states the exposure) and non-finite
flux rows, and drops `flux ≤ 0` rows unless `--keep-zero-flux`. The zero rows
of a CIAO-layout file are ambiguous: of the 374 zero-count rows of the broad
light curves, 235 are GTI gaps with `EXPOSURE = 0` in the legacy layout and 139
are observed empty bins, and the CIAO layout carries no exposure to tell them
apart — hence the choice. For Poisson count data whose exposure is known an
observed empty bin is a measurement the exposure-weighted bin needs (dropping
zeros biases the faintest bins high, +30 % at one count per row), so the
synthetic generator writes an `exposure` column and synthetic data are fitted
with `--keep-zero-flux`; the loader prints a note when it drops observed zero
bins. On the real broad light curve 15 % of the in-eclipse bins are genuine
zero-count 100 s bins, and dropping them raises the eclipse-window mean — the
`f_scatter` prior centre — by 18 %. Row errors are kept as read (a zero error
is a zero-count row to the exposure-weighted binner); `sanitize_errors` runs
on whatever enters the χ², after binning. Both fitters apply the same rules
through `drop_unobserved_rows` and `drop_invalid_flux_rows`. The loader
then applies the `--phase-window`, bins (constant-counts bins are formed along
the phase measured from the window's lower bound, so a wrapping window never
merges the points on either side of its seam), builds the `FitData`, and
computes the smoothed curve (inside the window only) and the data-driven
`f_scatter` prior when requested.

Binning mode is chosen by argument presence, not a mode flag:

| Flags | Behavior |
| ----- | -------- |
| `--no-phase-bin` | raw 100 s points (pair with `--likelihood jitter`) |
| `--counts-per-bin N` | adaptive constant-counts bins (recommended `100`) |
| `--n-phase-bins N` | fixed-width bins |
| neither | 50 fixed-width bins (backward-compatible default) |

Supplying both `--n-phase-bins` and `--counts-per-bin` is an error. Errors are
repaired once, by `sanitize_errors` (median valid error, with a warning), on
the values that enter the χ² — the bins, or the rows under `--no-phase-bin`;
unbinned data without any valid error is rejected, and binned data without an
error column proceed on `std/√n` bin errors with a warning.

### Argument validation

`validate_args` (and `_validate_args` in `chandra_phase_analysis.py`) rejects
argument combinations that contradict each other or have no effect, using
`utils.explicit_cli_dests` to tell options the user typed from defaults and
values restored by `--replot`. The rules, all `parser.error` (exit 2):

- **Binning:** `--no-phase-bin` excludes both binning options; the two binning
  options exclude each other; counts and bin numbers are positive;
  `--min-points-per-bin` (tabulated fit) only with fixed-width bins.
- **Phase shift and window:** a partial `--phase-window` needs a fixed shift;
  `--phase-shift-grid-size` only with the search enabled; `--fit-phase-shift`
  and `--phase-shift` are mutually exclusive (tabulated fit); the
  `--scatter-eclipse-phase` window must overlap the data window when the
  scattered flux is estimated from the data.
- **Contradictions:** `--fit-fopacity` with `--freeze log_fopa`, `--fit-scatter`
  with `--freeze f_scatter`, `--scatter` together with `--scatter-eclipse-phase`
  (tabulated fit).
- **No-effect options:** `--scatter-eclipse-phase` without `--fit-scatter`;
  `--prior-<name>` for a parameter of another parameterization or wind model, or
  a shape prior without `--fit-wind-shape` (unless frozen); `--orbital-period`
  outside the Kepler modes (it only enters Kepler's third law; folding always
  uses `utils.ORBITAL_PERIOD`); `--chi2-n-samples` without `--save-chi2`;
  `--smooth-sigma` without `--smooth`; `--csv-chunk-size` with
  `--no-csv-output`; `--numba-threads-per-worker` without a pool; sampling
  options (`--n-walkers`, `--n-steps`, `--n-burn`, `--sampler`, `--n-threads`,
  …) together with `--replot` (`--seed` is accepted there: the χ² subsample
  and the wind-profile draws are random); fit-only options (`--sim-file`,
  `--sim-column`, `--fit-phase-shift`, `--phase-shift`, `--scatter`,
  `--write-model`) without `--fit` in the tabulated fitter.
- **Ranges:** `--n-walkers` even and ≥ 2·n_dim (emcee's requirement, checked
  before any data is loaded); `0 ≤ n_burn < n_steps`; `--dth` and `--d2h`
  positive divisors of 360; `--mdot`, `--v-inf`, `--mu-wind` positive;
  `--seed` in `[0, 2³²)`; the scatter window and phase window inside `[0, 1]`.

### Reporting and diagnostics

- `compute_statistics` — per-parameter `median`, `±1σ` from 16/84 percentiles,
  `mean`, `std`; the mode's derived quantities via `ParamSpec.derived`; plus a
  **MAP** entry (highest-log-prob single sample) when `log_prob` is available.
  The MAP point is used for overlays because it is algebraically
  self-consistent — `median(a·q) ≠ median(a)·median(q)`, so median rows
  generally do *not* satisfy `d1+d2 = a`.
- `print_diagnostics` — acceptance fraction, integrated autocorrelation times,
  effective independent samples, convergence flag (`n_steps > 50·max τ`).
- `run_arviz_diagnostics` — ArviZ summary (`r_hat`, `ess_*`, `mcse_*`, HDI),
  written to `*_arviz_summary.csv`. Version-agnostic via `_build_inference_data`.
- `compute_bic_metrics` — `BIC = k·ln n − 2 ln L̂`, with `k` the sampled
  dimensions (including `log_f` under `jitter`) plus the profiled phase shift
  (`degrees_of_freedom` also counts the shift but excludes `log_f`),
  `n = len(obs_flux)` *after* binning/filtering, and `L̂` the **maximum
  likelihood over the chain**: `log_prob − log_prior` is exactly each sample's
  log-likelihood, so the best sample is free (`theta_source =
  max_likelihood_sample`). The MAP sample maximises the posterior and can sit
  several log-units lower in likelihood (ΔBIC 5.6 in a 30-step test). Without
  the chain, or on a replot whose priors differ from the sampled ones, the
  likelihood at the point estimate is used (`map_log_prob`, `median_fallback`
  when no log-probabilities exist). BIC is the model-comparison metric; `ΔBIC`
  is reported relative to the best model in the run. Enable with
  `--compute-bic`.
- `compute_chi2_for_samples` (`--save-chi2`) — per-sample χ² and reduced χ²,
  gzip CSV. With the `chi2` likelihood the table is read from the chain
  (`χ² = −2·(log_prob − log_prior)` exactly; no model calls, every sample). For
  jitter runs it emits *both* the classical measurement-error χ² (comparable
  across likelihood choices) and the effective-variance `chi2_eff`, which needs
  a model call per row, so it defaults to a 2000-sample subset
  (`--chi2-n-samples`).
- `postprocess_fit` — the block shared by a fresh fit and `--replot`: ArviZ,
  BIC, corner/trace/best-fit/geometry figures, the chi2 table.
  `write_summary` writes `{band}_{wind_model}_summary.txt` (a second wind
  model in the same directory no longer overwrites the first's summary; a run
  does overwrite its own band/model artefacts, after a warning).
  `run_single_fit` writes the samples CSV and the chain NPZ **immediately
  after sampling**, before statistics and figures, so a plotting or ArviZ
  failure (or a Ctrl-C) can no longer discard the chain; `--replot` needs only
  the chain file. Samples and log-probabilities are flattened in (step,
  walker) order for both samplers — the order `--replot` reshapes the chain
  into (zeus's own `flat=True` is walker-major) — and the autocorrelation
  times and effective-sample count are computed on the post-burn-in chain.
  Ctrl-C on a pooled run terminates the workers (they ignore SIGINT; the
  parent handles it) instead of waiting forever on an in-flight task, and the
  per-worker numba thread count is clamped to numba's own limit, since a Pool
  initializer that raises makes multiprocessing respawn workers forever.
  χ²/dof values reported by the MCMC count the profiled phase shift as one
  fitted parameter, the same convention as `fit_simulation`.

### Plots

`plot_corner`, `plot_trace` and the drawing behind `plot_best_fit` live in
`utils/plot_utils.py`; `plot_geometry_diagnostics` in `mcmc_lightcurve_fit.py`.

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
    `N_H(φ)` with its orbit mean, and the band flux. Turns
    the eclipse from an emergent light-curve feature into a stated geometric
    condition with visible margin.
  - **`*_wind_profile.png`** — `g(r)` with 68/95% posterior credible bands, an
    `r⁻²` reference, the companion surface, characteristic radii (`Rb`/`ell`)
    and — the important part — the band of radii the line of sight actually
    probes. That band is `[min l3, max l3]`: the LOS impact parameter relative
    to the companion centre *equals* the projected separation, so the profile
    inside `min l3` is unconstrained by the data. Shape parameters are only
    interpretable jointly (`Rb` and `p` trade off strongly), so the constraint
    reads far more clearly here than in a corner plot.
- `plot_best_fit` (in `mcmc_lightcurve_fit.py`) — resolves the point estimate
  (MAP when available, else per-parameter medians), evaluates the model through
  `model_curve` (the same entry point the likelihood uses, so geometry mode,
  wind shape, frozen values and the additive `f_scatter` are resolved once),
  finds the best phase shift, then hands the arrays to `plot_lightcurve_fit`.
  Result: a 2-panel (3:1) figure with the MAP overlay over the data, the optional
  smoothed green curve + analytic band, and a normalized-residual panel clipped to
  ±5σ. The title is only the energy band and χ²/dof; the point estimate,
  `phase_shift`, `f`, `chi2_eff/dof` and `f_scatter` are printed to stdout
  (parameter values with their 1σ come from `print_results` and the summary
  file) rather than annotated on the figure. Returns the reduced χ² of the drawn
  model.

`--replot` regenerates everything from saved results without re-running MCMC:
`replot_from_existing` reads **`*_chain.npz`**, the one file every fit writes
right after sampling, for the post-burn chain and log-probabilities, the
sampled parameter names in chain order, the metadata that rebuilds the
`ParamSpec` (`mode`, `frozen_names`/`frozen_values`, `orbital_period_s`,
`likelihood`) and the `wind_normalization` stamp, then runs the same
`postprocess_fit` as a fresh fit. The `*_samples.csv` export is not needed, so
a fit run with `--no-csv-output` replots too, and a directory holding only a
samples CSV (an unstamped legacy result) is refused. Result directories written
before Phase 34 carry no stamp and are refused (see below).

**Every option not given explicitly is restored from `*_run_config.json`**, so
`python mcmc_lightcurve_fit.py --replot` on its own reproduces the original
band, wind model, `--flux-csv`, `--data-dir`, `--obs-column`/`--time-column`,
binning, `--dth`/`--d2h`, the wind normalization, priors and model flags. This matters because
those options change the *observed arrays*: replotting with different binning
silently reports a χ²/dof for a dataset the posterior never saw. Explicit flags
always win over the saved values, so a single option can be overridden in place
(`--replot --smooth --smooth-sigma 0.02`; output options such as `--smooth` are
never restored, so they are typed along with their sub-options). As a
backstop, `replot_from_existing` compares
the observed point count against `n_obs` in the chain metadata and warns on a
mismatch. `--band` and `--flux-csv` are therefore only required when *not*
replotting.

Three rules keep the restore honest (`utils.apply_saved_run_config`):

- **Invocation-only options are never restored.** `build_parser` puts them in
  the *Execution* group (`--n-threads`, `--numba-threads-per-worker`, `--seed`,
  `--quiet`, `--replot`) and the *Output* group (`--output-dir`,
  `--no-plots`, `--no-geometry-plots`, `--smooth`, `--smooth-sigma`,
  `--compute-bic`, `--no-csv-output`, `--csv-chunk-size`, `--save-chi2`,
  `--chi2-n-samples`), and `NEVER_RESTORED_DESTS` is derived from those two
  groups, so a new output option is classified where it is defined.
  `store_true` flags cannot be negated on the command line, so restoring them
  made a fit run with `--no-plots` impossible to replot with figures. Only
  options that define the fit — data, binning, model, priors, sampler settings
  — come back.
- **Mutually exclusive siblings are not restored** when one member was typed:
  `--replot --n-phase-bins 30` on a `--counts-per-bin` run no longer trips the
  exclusivity check (the `n_obs` warning then says the binning differs), and
  likewise for `--reparam`/`--kepler`/`--kepler-mtot`.
- **Results without the `wind_normalization` stamp are refused.** Every run
  config and chain NPZ written since Phase 34 carries
  `wind_normalization = "physical-mdot-vinf"` next to the inclination
  convention. Directories sampled under the retired `lam` normalization lack
  it; re-evaluating their MAP with the physical normalization would report a
  χ²/dof, overlays and BIC for a model the posterior never saw (the old
  `mcmc_results/` fits did exactly that silently, with `N_H` 20–50 × 10²²
  instead of the chain's 0.53), so `--replot` exits with an error asking for a
  refit. The self-healing run config for a directory that has a stamped chain
  but no run config (a deleted or never-written config) is written only after
  a replot has succeeded.

---

## Spectral / XSPEC side

**`compute_flux_vs_nH.py`** is the key upstream product. It loads a spectrum
(PHA + background + responses) from `--specdir`, fits
`{phabs,tbabs,wabs}×powerlaw` over `--fit_emin/--fit_emax`, freezes the
powerlaw, then sweeps `nH` over a log grid and integrates the flux of **one**
band (`--band`, default `broad`) at each point. Emits a CSV with `nH_cm2`,
`nH_1e22`, `flux_{band}_ph`, `flux_{band}_erg` — exactly what `--flux_method interpolate`
consumes. Chandra bands: `broad` 0.5–7.0, `soft` 0.5–2.0, `medium` 1.2–2.0,
`hard` 2.0–7.0 keV. Requires PyXspec in the active environment; the earlier
XSPEC helper scripts (`.xcm` files, model-comparison and conversion-factor
tools) are no longer in the tree.

Phase 34 cut the script from 930 to ~380 lines without changing the table it
produces. PyXspec parameters are addressed by index (`model(i).values = x`
sets the value, `.values[0]` reads it, `.sigma` is the fit sigma — the old
`.error` field held the result of a `Fit.error` run that is never made, so
every printed "±" was 0); the ~200 lines of index-vs-component-name fallbacks
that could not execute are gone; spectrum files are matched on their **file
names** (matching the full path picked the background as the source whenever a
directory was called `src`), with the background identified before the source;
the PHA header's own BACKFILE/RESPFILE/ANCRFILE pairing is kept and the
background, RMF or ARF found in the directory are attached only where the
header left a gap (assigning `spectrum.response` replaces the Response object
and would silently drop the header's ARF; a spectrum without any response is
an error and the files in use are printed); the model energy array is
extended to 0.1–20 keV once and the band flux at each `nH` is XSPEC's own
`AllModels.calcFlux` read back from the loaded spectrum (energy flux
`flux[0]`, photon flux `flux[3]`), exact over the band, instead of a
trapezoid over a plot grid (XSPEC clips a band to the model energy array, so
bands outside 0.1–20 keV are rejected); PyXspec is imported after argument
parsing so `--help` works without HEASoft; and the exponential
law drawn on the figure is `utils.fit_exponential`, the function the
simulator's `refit` method uses. HEASoft is not importable from plain `henv`,
so the script's control flow was exercised end to end against a PyXspec
stand-in (a fake `xspec` module with the same attribute semantics) and its
output table fed to `simulate_band_flux`; the calls it makes of PyXspec are the
documented ones.

---

## Data layout

The repository distributes **synthetic data only**. The Chandra light curves
and spectra of IC 10 X-1 that the model was developed on stay on the author's
disk under `data/` (ignored by git, see `.gitignore`), together with the real
fit results in `mcmc_results/`; a companion paper will publish those fits. What
is tracked is `synthetic_data/`: the example flux table
`flux_vs_nH_tbabs_broad.csv` (TBabs × power law with the IC 10 X-1 spectral
parameters, 1001 `nH` points from 1e15 to 1e26 cm⁻², the table the tests and
examples use) and whatever synthetic light curves, tables and truth records
the generators write there (the blanket `*.csv`/`*.txt`/`*.json` ignore rules
are negated under `synthetic_data/`).

The loaders accept three file layouts, all whitespace-delimited `*.txt`:

| Layout | Columns | Notes |
| ------ | ------- | ----- |
| CIAO (real light curves) | `# Columns: dt, t_raw, mjd, phase, counts, rate, rate_err, flux_t` | `flux_t = rate × c`; no `flux_t_err` (derived from `rate_err`); exposure recovered as `counts / rate`; a zero-count row cannot be told from a GTI gap. |
| Synthetic | CIAO columns + `exposure` | Written by `synthetic_data/make_lightcurve.py`; zero-count rows are known to be observed. |
| Legacy | `TIME COUNTS COUNT_RATE COUNT_RATE_ERR EXPOSURE NET_COUNTS NET_RATE ERR_RATE FLUX FLUX_ERR` | `EXPOSURE` used directly; `EXPOSURE = 0` rows dropped as unobserved. |
| Headerless | `time rate [error]` | Seconds since MJDREF; no counts, so inverse-variance bins. |

`--data-dir` takes a directory of such files or a parent with `{band}/`,
`{band}/single/` or `{Band}_with_flux/` sub-folders. Phase is always
recomputed from the time column with `REF_EPOCH` / `ORBITAL_PERIOD`
(IC 10 X-1 ephemeris; change the constants in `utils/utils.py` for another
system). The one-off scripts that produced the legacy layout from CIAO FITS
products (`convert_fits_to_txt.py`, `add_flux_simple.py`,
`get_average_count_rates.py`) and the author's command log
(`rkp_run_w_mcmc_cmds.sh`) live in `extras/`, untracked.

---

## Synthetic data (`synthetic_data/`)

Generators for injection–recovery tests; everything they write is read by the
pipeline unchanged and the bands are `utils.utils.CHANDRA_BANDS`.

- **`make_spectrum.py`** (PyXspec) fakes an absorbed power law through the IC 10
  X-1 combined ACIS response with `AllData.fakeit` (`--nH`, `--PhoIndex`,
  `--norm`, `--exposure`, optional real background, `--seed` via `Xset.seed`)
  into `--out-dir`, which then serves as `--specdir` for
  `compute_flux_vs_nH.py`. It also reports, per band, the model flux
  (`AllModels.calcFlux`), the fake net count rate and their ratio, the
  flux-per-count-rate factor written to `band_factors.json`.
- **`make_lightcurve.py`** evaluates the forward model at known parameters
  (every `xrb_lightcurve.py` keyword, same defaults), shifts it by
  `--phase-shift` (mid-eclipse lands at data phase `0.5 + shift`), adds
  `--scatter`, converts flux to expected counts per `--dt` bin with
  `--flux-per-rate` (default `1.13e-11`, the real broad-band `flux_t/rate`),
  adds `--bkg-rate`, Poisson-samples (`--noiseless` to skip) over `--visits`
  (`start:duration,...` in seconds after `REF_EPOCH`) or one visit of
  `--n-orbits`, removes random `--gap-fraction`/`--gap-duration` blocks, and
  writes the CIAO layout `# Columns: dt, t_raw, mjd, phase, counts, rate,
  rate_err, flux_t` (`rate_err = √(counts + bkg counts)/dt`, so an empty bin
  carries a zero error like the real files) plus an `exposure` column (`= dt`)
  so the fitters know every zero-count row was observed, and
  `<stem>_truth.json` with every injected value, `mid_eclipse_data_phase`, the
  bin and zero-count counts. Gaps never cross a visit boundary and every visit
  keeps at least one bin; overlapping `--visits` are refused.
- The package README gives the spectrum → table → light curve → fit sequence.
  With the injected floor passed as `--scatter`, the tabulated fit recovers
  the injected shift 0.985 as 0.9848 (constant-counts bins), 0.9852 (100
  bins) and 0.9847 (unbinned) on a synthetic broad light curve with ten times
  ObsID 15803's count rate (`--flux-per-rate 1e-12`), χ²/dof ≈ 1. At the real
  count rate (~14 counts per 100 s bin) the recovery is 0.981 / 0.979 for two
  seeds with χ²/dof ≈ 1.06: the remaining −0.005 is the Neyman bias of a χ²
  whose weights are the observed Poisson errors (see Known rough edges), not
  the binning. The exact degeneracies of the model apply
  to synthetic data too (`q`, and the length scale with `f_opacity`), so
  recovery is judged on `a`, `R`, `i0`, `f_opacity` jointly.

---

## Outputs

Per `(band, wind_model)` in `--output-dir`, prefixed `{band}_{wind_model}_`:

| File | Contents |
| ---- | -------- |
| `*_chain.npz` | Post-burn chain and log-prob plus the run metadata (`param_names`, `mode`, frozen params, `likelihood`, `wind_model`, `orbital_period_s`, `n_obs`, `wind_normalization`). The file `--replot` reads; always written, right after sampling. |
| `*_samples.csv` | Flat post-burn-in samples + `log_prob`, a plain-text export (chunked writer; skip with `--no-csv-output`). |
| `*_run_config.json` | The complete CLI configuration of the fit (`created`, `command`, every argparse value). Written before sampling starts, so it survives an interrupted run. `--replot` restores from it. |
| `*_corner.png`, `*_trace.png`, `*_bestfit.png` | Diagnostic plots. |
| `*_geometry_orbit.png`, `*_geometry_phase.png`, `*_wind_profile.png` | Binary-geometry figures at the point estimate (`--no-geometry-plots` to skip). |
| `*_arviz_summary.csv` | ArviZ convergence table. |
| `*_model_metrics.csv` | `bic`, `logL_hat`, `k_params`, `n_obs`, `theta_source`. |
| `*_chi2.csv.gz` | Per-sample χ² (`--save-chi2`). |
| `*_summary.txt` | Human-readable roll-up per band and wind model: run config, marginal posteriors, MAP block, reduced χ², BIC/ΔBIC, chain diagnostics. |

---

## Typical workflows

```bash
# 1. Build the XSPEC flux-vs-nH table, one band per file (needs XSPEC / henv)
python compute_flux_vs_nH.py --specdir spectra/ic10x1 --model tbabs \
    --band broad \
    --out_csv flux_vs_nH_tbabs_broad.csv --out_png flux_vs_nH_tbabs_broad.png \
    --nH_min 1e20 --nH_max 1e24 --nH_points 60

# 2. Generate a single simulated light curve
python xrb_lightcurve.py --flux_method interpolate \
    --flux_csv flux_vs_nH_tbabs_broad.csv \
    --wind-model smooth_pl --Rb 5 --p 4 --Delta 2 \
    --i0 78.0 --f-opacity 0.02 --output sim_broad.csv

# 3. Fold the data and χ²-fit that one model (phase shift free; flux never rescaled)
python chandra_phase_analysis.py \
    --data-dir lightcurves/broad \
    --obs-column flux_t --time-column t_raw \
    --fit --sim-file sim_broad.csv --fit-phase-shift \
    --smooth --n-phase-bins 100 --output fit_broad.png

# 4. MCMC — geometry only, adaptive constant-SNR bins
python mcmc_lightcurve_fit.py --band broad \
    --flux-csv flux_vs_nH_tbabs_broad.csv \
    --data-dir lightcurves \
    --obs-column flux_t --time-column t_raw \
    --wind-model smooth_pl --reparam --likelihood chi2 \
    --counts-per-bin 100 --sampler zeus --dth 4.0 \
    --n-walkers 24 --n-steps 20000 --n-burn 2000 \
    --compute-bic --smooth --output-dir mcmc_results/broad/smooth_pl/geom

# 5. MCMC — Kepler masses + wind shape + scattered-flux floor
python mcmc_lightcurve_fit.py --band broad \
    --flux-csv flux_vs_nH_tbabs_broad.csv \
    --data-dir lightcurves \
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
    --data-dir lightcurves \
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

#    Override a single option in place (explicit flags beat the saved config;
#    output options are never restored, so type --smooth with its width):
python mcmc_lightcurve_fit.py --replot --output-dir mcmc_results --smooth --smooth-sigma 0.02

#    For a stamped result whose run config is missing, pass the original
#    options once; a config is then written automatically for next time.
python mcmc_lightcurve_fit.py --band broad --flux-csv flux_vs_nH_tbabs_broad.csv \
    --data-dir lightcurves --obs-column flux_t --time-column t_raw \
    --wind-model smooth_pl --counts-per-bin 100 \
    --replot --compute-bic --output-dir mcmc_results/broad/smooth_pl/geom
```

---

## Environment

Conda env `henv` (heasoft/XSPEC + Python deps). [requirements.txt](requirements.txt)
lists everything pip-installable: `numpy`, `pandas`, `numba` (**required**;
`xrb_lightcurve.py` raises `ImportError` without it), `matplotlib`, `emcee`,
`corner`, `tqdm`, and the optional `zeus-mcmc` (`--sampler zeus`), `arviz`
(convergence summaries) and `astropy` (FITS conversion helper). `scipy` is not
imported directly (emcee and arviz pull it in). XSPEC Python (`pyxspec`) comes
from HEASoft and is needed by `compute_flux_vs_nH.py` and the fake-spectrum
generator only. Python ≥ 3.9 (the code compiles under 3.8; the practical floor
is the numba/numpy wheels). Figures are written with the `Agg` backend by the
MCMC fitter; the other scripts show a window only when `--output` is omitted.

---

## File inventory

### Core
| File | Lines | Role |
| ---- | ----- | ---- |
| [xrb_lightcurve.py](xrb_lightcurve.py) | ~1050 | Forward model: profiles, Numba LOS kernel (half the orbit by phase reflection) and per-cell flux conversion, `simulate_lightcurve` / `simulate_band_flux`, `SIM_DEFAULTS`, physical normalization. |
| [mcmc_lightcurve_fit.py](mcmc_lightcurve_fit.py) | ~2140 | emcee/zeus MCMC: `ParamSpec`, `FitData`, prior/likelihood, phase-shift search, BIC, plots, replot, summary. |
| [chandra_phase_analysis.py](chandra_phase_analysis.py) | ~510 | CLI front end for the single-model χ² fit; re-exports the shared `utils/` API. |
| [utils/utils.py](utils/utils.py) | ~1670 | Shared layer: ephemeris, loading, `sanitize_errors`, both binners, smoothing, the periodic interpolator + phase-shift search, `fit_simulation`, model-dump blocks, run-config persistence. |
| [utils/plot_utils.py](utils/plot_utils.py) | ~900 | All plotting, built on the single `plot_lightcurve_fit`. |
| [compute_flux_vs_nH.py](compute_flux_vs_nH.py) | ~360 | XSPEC `flux vs nH` table generator (one band per table). |
| [plot_results.py](plot_results.py) | 104 | Thin CLI over `utils/plot_utils.py` for simulation CSVs (`--geometric`, `--orbit`). |
| [synthetic_data/](synthetic_data/) | ~410 | `make_spectrum.py` (PyXspec `fakeit` + per-band flux-per-rate factors) and `make_lightcurve.py` (CIAO-layout light curves from the forward model with a truth record). |

### Utilities (`utils/`)
`utils/` is a package (`__init__.py`). Two modules are library code imported by
the analysis scripts — `utils.py` and `plot_utils.py` (see Core above).
`test_flux_methods.py` is the regression test (it runs on the tracked
`synthetic_data/flux_vs_nH_tbabs_broad.csv`). `CHANDRA_BANDS` in `utils.py`
is the single definition of the energy bands used by the flux-table generator,
the synthetic-data scripts and the plot labels.

### Not part of the release (on disk, ignored by git)
`.gitignore` lists them: `data/` (Chandra light curves, spectra, responses),
`mcmc_results/` (real fits), `notebooks/` (exploratory notebooks predating
Phases 31–33, still calling removed functions), `legacy_r_code/` (the original
R implementation), `extras/` (the legacy-layout conversion scripts and
`rkp_run_w_mcmc_cmds.sh`), `temp/` (scratch), `paper/` (the MDPI *Algorithms*
manuscript source), the reference PDFs (`Wind_Density.pdf`, `stu2151.pdf`,
`manuscript_1.pdf`) and editor state. Earlier commits on this branch still
contain the real light curves and the notebooks; a public release should
start from fresh history or rewrite it.

### Documentation
| File | Contents |
| ---- | -------- |
| `PROJECT.md` | This file — current state. |
| [changes_tracked.md](changes_tracked.md) | Condensed change log, including removed features. |
| [README.md](README.md) | User-facing overview: pipeline, parameters, output columns. |

---

## Known rough edges

- **MCMC results predating the inclination convention change are stale.** `i0`
  used to be measured from the line of sight and is now measured from the
  orbital-plane normal, so those chains store the complement of what the model
  expects. New run configs carry `"inclination_convention":
  "i0-from-orbital-normal"` and `--replot` warns when the stamp is absent; the
  fix is to refit. The (untracked) notebooks still pass old-convention `--i0`
  / `--prior-i0` values (`extras/rkp_run_w_mcmc_cmds.sh` has been updated).
- **The untracked notebooks have not been updated** for the `lam` /
  `broken_pl` / `beta_law` / `legacy` removals;
  `notebooks/xrb_model_analysis_single_15803.ipynb` in particular calls the
  deleted `compute_surface_density`.
- **Reference epoch is unresolved.** Laycock et al. define `T0` as the
  *mid-eclipse* time of ObsID 07082 at **phase 0.5**, whereas
  `frac((t-T0)/P)` puts it at phase 0.0. A recalibration study
  (`.cursor/plans/reference_epoch_recalibration_*.plan.md`) derived
  `278800407.267`, which sits commented out beside `REF_EPOCH`. In practice the
  MCMC's per-sample phase-shift search absorbs the offset, so this mostly
  affects the interpretability of plotted phases.
- **`.gitignore` excludes `*.csv`, `*.txt`, `*.png` everywhere except under
  `synthetic_data/`**, so real data, XSPEC tables and figures are not
  version-controlled; synthetic products are. A fresh clone has the example
  flux table and needs HEASoft only to make new tables.
- **`chandra_analysis_combined_flux.py` is an unmigrated, untracked fork.** It
  carries its own older copies of `fit_simulation`, `plot_phase` and the
  removed multi-column helpers, has no `scatter` support, and still fits a
  **multiplicative flux scale** (`--rescale`) — the degree of freedom
  deliberately removed everywhere else. Its χ² values are not comparable to the
  main path's; retiring it is recommended.
- **`--data-dir` resolves `{band}/single` before `{band}/`**, so a parent
  directory silently selects the single-observation subset. Pass the band
  directory explicitly.
- **χ² weights are the observed Poisson errors (Neyman's χ²).** Bins with a
  downward fluctuation get a smaller error and more weight, which biases the
  tabulated fit's phase shift by ≈ −0.005 at ~14 counts per 100 s bin
  (0.981/0.979 recovered for 0.985 injected; vanishing at ten times the
  counts). The MCMC likelihoods weight the same way. Model-based (Pearson)
  variances or a Poisson likelihood would remove it; use more counts per bin
  meanwhile.
