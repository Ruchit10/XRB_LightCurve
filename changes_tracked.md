# XRB Lightcurve Project — Change Log

Tracks the evolution of the IC 10 X-1 X-ray binary light-curve simulation,
fitting and inference stack since the original R port. Early phases are
condensed; the numbers that still matter for interpreting results are kept.
[PROJECT.md](PROJECT.md) is the current-state reference.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Phases 1–5 — Foundations](#phases-15--foundations)
3. [Phases 6–7 — Eclipse Geometry Fixes & Unified Wind Model](#phases-67--eclipse-geometry-fixes--unified-wind-model)
4. [Phase 8 — MCMC Pipeline & Performance](#phase-8--mcmc-pipeline--performance)
5. [Phases 9–11 — Reparameterization, Wind-Shape Parameters, N-D Grid](#phases-911--reparameterization-wind-shape-parameters-n-d-grid)
6. [Phases 12–15 — flux_t Errors, Normalization Constants, Pooled MCMC, Speed & BIC](#phases-1215--flux_t-errors-normalization-constants-pooled-mcmc-speed--bic)
7. [Phases 16–18 — `ParamSpec`, Phase-Shift Alignment, Adaptive Binning](#phases-1618--paramspec-phase-shift-alignment-adaptive-binning)
8. [Phases 19–22 — Smoothing, Scattered Flux, No Multiplicative Scale](#phases-1922--smoothing-scattered-flux-no-multiplicative-scale)
9. [Phases 23–26 — `utils/`, Run-Config Persistence, Slimming, Geometry Plots](#phases-2326--utils-run-config-persistence-slimming-geometry-plots)
10. [Phase 27 — Standard Inclination Convention](#phase-27--standard-inclination-convention)
11. [Phase 28 — Physical Wind Normalization & Per-Cell Flux Conversion](#phase-28--physical-wind-normalization--per-cell-flux-conversion)
12. [Phase 29 — Mass Reparameterization & Error-Column Fix](#phase-29--mass-reparameterization--error-column-fix)
13. [Phase 30 — Physical Norm in the CLI, Model-LC Dump & χ²_eff Fix](#phase-30--physical-norm-in-the-cli-model-lc-dump--χ_eff-fix)
14. [Phase 31 — Release Trim](#phase-31--release-trim)
15. [Phase 32 — `beta_law` Wind Profile Restored](#phase-32--beta_law-wind-profile-restored)
16. [Phase 33 — Kernel Symmetry, Compiled Flux Conversion, Single Band & MCMC Consolidation](#phase-33--kernel-symmetry-compiled-flux-conversion-single-band--mcmc-consolidation)
17. [Phase 34 — Release Review: Performance, Correctness, Trimming](#phase-34--release-review-performance-correctness-trimming-2026-09-19)
18. [Side Investigation — Reference Epoch](#side-investigation--reference-epoch)
19. [Current File Inventory](#current-file-inventory)
20. [Current Status & Quick Commands](#current-status--quick-commands)

---

## Project Overview

**Target:** IC 10 X-1 — eclipsing X-ray binary in the Local Group galaxy IC 10;
a compact object (+ accretion disk) orbiting a Wolf-Rayet companion. The
framework (working name **CLOAK**) is general to eclipsing wind-fed binaries
with near-circular orbits.

**System (working values):** WR companion `R ≈ 2 R☉` (photosphere),
accretion-disk `r ≈ 0.001 R☉`, separation `a ≈ 19 R☉`, inclination `i₀ ≈ 78°`
(standard convention, 90° = edge-on), `P = 125431 s ≈ 1.45 d`.

**Best spectral model:** TBabs × powerlaw — nH ≈ 0.75×10²² cm⁻², Γ ≈ 1.86,
χ²_red ≈ 1.52 (preferred over phabs by Δχ² ≈ 8.5).

**Model today:** three dimensionless wind profiles (`smooth_pl`, `confinement`,
`beta_law`) with the absolute density fixed from `Ṁ/v_inf` and an effective
opacity factor `f_opacity`; a Numba Gauss-Legendre kernel with mirror-symmetric
emitter sectors; per-cell `N_H → flux` conversion compiled; ≈ 30 ms per light
curve; one band per run; emcee/zeus MCMC over four parameterizations.

---

## Phases 1–5 — Foundations

- **R → Python port.** `new11.R`, `grid4.R`, `wind_los2.R`, `density_fnc.R`
  (≈260 lines) became `xrb_lightcurve.py` with an argparse CLI and a vectorized
  NumPy core (10–100× faster than R).
- **Flux integration.** `--flux_method {legacy, interpolate, refit}` converting
  column density to band flux; `interpolate`/`refit` consume an XSPEC
  `flux vs nH` table from `compute_flux_vs_nH.py`. `--lam` pinned the
  orbit-averaged nH to the spectral fit (removed in Phase 31). Units: `fl` in
  10²² cm⁻², `nfl_{band}` in flux units.
- **Data pipeline.** Chandra `.txt` light curves were FITS in disguise; built
  `utils/convert_fits_to_txt.py`, `add_flux_simple.py`,
  `add_flux_to_lightcurves.py`, `get_average_count_rates.py` →
  `data/IC_10_X1_LC/{Broad,Soft,Hard}{_converted,_with_flux}/`. Time-averaged
  rates (cts/s): broad 0.1132, soft 0.0635, hard 0.0497.
- **Spectral model.** TBabs × powerlaw chosen over phabs × powerlaw
  (Δχ² = 8.54; nH 0.75 vs 0.78; 0.5–7 keV flux 1.032 vs 1.031×10⁻¹² erg/cm²/s).
- **Phase folding & χ² fitting.** `chandra_phase_analysis.py` folds on
  `REF_EPOCH = 278801348 s`, `P = 125431 s`, handles standard and CIAO
  `#Columns:` headers, and fits a tabulated model by χ² with a phase shift
  (originally also a flux scale — removed in Phase 20).

---

## Phases 6–7 — Eclipse Geometry Fixes & Unified Wind Model

**Geometry fixes.** Eclipse gating moved from `gma < π` to `sin(gma) > 0` so an
emitter *in front of* the companion is never occulted; `is_eclipsed` added and
fluxes forced to 0 in eclipse (previously `flx = 0 ⇒ e⁰ = 1` gave *maximum*
flux). A configurable LOS cutoff (`--Rmax`, `--converge-rmax`) was added and
later removed (Phase 31).

**Unified wind model** (plan `unified_wind_model_77726ced`). The hardcoded
AV/CV wind duality (`flx2`, `lam2`, `*_cv` columns) was replaced by a registry
of dimensionless profiles `g(r)` consumed inline by one Numba kernel:
`broken_pl`, `smooth_pl` (default), `beta_law`, `confinement`.
`simulate_lightcurve` gained `wind_model` / `wind_params`; output collapsed to
one `flx`/`fl` plus one `nfl_{band}` per band; helpers `pack_wind_params`,
`default_wind_params`, `evaluate_g_profile`. `_los_gl_quadrature`
(Gauss-Legendre under `u = arctan(z/b)`) and the per-phase sweep
`_simulate_phases_numba` (`prange` over phases) date from here.

---

## Phase 8 — MCMC Pipeline & Performance

`mcmc_lightcurve_fit.py` wrapped the simulator in emcee/zeus with a
6-D precomputed geometry grid (`PrecomputedModelGrid`, later removed) and a
`DirectLightCurveModel`, `chi2`/`jitter`/`studentt` likelihoods (`studentt`
later removed), ArviZ diagnostics, WAIC/LOO (later BIC), corner and best-fit
plots.

**Performance.** After the profile registry a single light curve took 3–8 s
(Python callbacks for `g(r)` in the hot loop). `@njit` on the profile, the
integrand and the full phase sweep, Gauss-Legendre instead of fixed-step
trapezoid, an inlined eclipse test and scalar kernel arguments brought it to
≈ 63 ms — direct evaluation inside MCMC became viable. GPU and NUTS were
evaluated and rejected (no CUDA; non-differentiable kernel).

---

## Phases 9–11 — Reparameterization, Wind-Shape Parameters, N-D Grid

- **`--reparam`.** `(d1, d2)` are strongly correlated (the wind sees their sum),
  so sample `a = d1 + d2` and `q = d1/a` with the `+log(a)` Jacobian; `d1, d2`
  reported as derived.
- **`--fit-wind-shape`.** Per-model shape parameters become MCMC dimensions
  (`WIND_SHAPE_FIT`, `WIND_SHAPE_FIXED`, `WIND_SHAPE_PRIORS`,
  `--prior-<name>`); the parameter vector became fully name-driven
  (`active_names`, no positional `theta[5]`).
- **N-D precomputed grid** for shape-fit MCMC — superseded and deleted in
  Phase 18; kept here for history only.

---

## Phases 12–15 — flux_t Errors, Normalization Constants, Pooled MCMC, Speed & BIC

- **`flux_t` errors** (commit `fa672e3`). CIAO files carry `flux_t` but no
  `flux_t_err`; `_derive_err_from_rate_err` derives `err = rate_err·(obs/rate)`
  with a file-level `median(obs/rate)` fallback. Zero-flux GTI gaps are dropped
  in binned and unbinned modes; invalid errors patched to
  `max(0.1·|flux|, median(valid))`; `is_binned` threaded to the plots; jitter
  runs report both classical χ² and `chi2_eff`.
- **Wind normalization constants** (`fa672e3`) — helpers to back `n₀`, break
  density and `Ṁ/v_inf` out of `lam` after a fit. Removed in Phase 31 once `n₀`
  became an input.
- **Pooled direct-model MCMC** (`36c8c8b`). `--n-threads` workers each call
  `numba.set_num_threads(cpu_count // n_threads)` (`_init_numba_worker`,
  `--numba-threads-per-worker`) to avoid oversubscribing the `parallel=True`
  kernel; `spawn` context.
- **Speed/memory pass & BIC** (`7116302`; harness
  `utils/benchmark_mcmc_performance.py`). Module-level `_FLUX_CACHE` keyed by
  `(csv path, flux_type)` so the table is read once per run; likelihood
  invariants hoisted out of the hot loop; chunked CSV writer
  (`--csv-chunk-size`, `--compact-output`, `--no-csv-output`). WAIC/LOO
  replaced by **BIC** `= k·ln n − 2 ln L̂`, with `L̂` from the run's own
  likelihood at the max-log-prob sample (`theta_source = map_log_prob`).

---

## Phases 16–18 — `ParamSpec`, Phase-Shift Alignment, Adaptive Binning

**`ParamSpec`** (commit `be56359`, plan `freeze_params_and_kepler_3368d554`):
one dataclass built once in `main()` (`mode`, `active_names`, `frozen`,
`fit_wind_shape`, `fit_scatter`, `wind_model`, `likelihood`,
`orbital_period_s`, `K_kepler`) replaced positional indexing everywhere.
**`--freeze NAME=VAL,…`** pins parameters and drops them from the chain (shape
params freezable without `--fit-wind-shape`; `log_f` never; values outside the
prior box warn). **`--kepler`** samples `(M_X, M_RH)` with
`a = K·M_tot^{1/3}`, `K = (G·M☉·P²/4π²)^{1/3}/R☉`, `q = M_RH/M_tot`.
`log_prior` enforces `r < R` and `Rb ≥ R` on *resolved* values.
`compute_statistics` reports derived quantities and a MAP row (which, unlike
medians, satisfies `d1 + d2 = a` exactly); `*_chain.npz` stores `mode`, frozen
values and period for `--replot`.

**Per-sample phase-shift alignment** (`67448c3`). Every likelihood call
minimizes weighted χ² over a phase shift: a coarse grid of 25 shifts over
`[0, 1)` (model evaluated once on 240 points) then a 9-point local refinement.
Applied consistently in the likelihoods, the per-sample χ², BIC and
`plot_best_fit`; `--no-fit-phase-shift` disables it.
`mcmc_summary.txt` gained run-configuration and chain-diagnostics blocks.

**Adaptive constant-counts binning & grid removal** (`b376585`).
`phase_bin_data_snr(counts_per_bin=100)` accumulates sorted points until each
bin holds the target counts (SNR ≈ 10 per bin), returning counts-weighted
phase centres, inverse-variance means, `error = √(1/Σw)` and bin widths (drawn
as horizontal error bars). Mode by argument presence: `--no-phase-bin` >
`--counts-per-bin` > `--n-phase-bins` > 50 fixed bins. The precomputed grid,
`studentt` and `--compute-waic` were deleted; MCMC always uses the direct
evaluator.

---

## Phases 19–22 — Smoothing, Scattered Flux, No Multiplicative Scale

- **Smoothing & residual panels** (plan `gaussian_phase_smoothing_reference`).
  `smooth_lightcurve` — periodic Gaussian-kernel smoother (`σ = 0.01` in phase)
  with a vectorized Monte-Carlo 1σ band; `estimate_scattered_flux` — mean flux
  in the mid-eclipse window `(0.4, 0.6)`; `add_residual_panel` — `(O−M)/σ`
  pulls. `simulate_lightcurve(scattered_flux=…)`; `--fit-scatter` promotes
  `f_scatter` to a free MCMC parameter with a data-driven prior; CLI
  `--smooth*`, `--scatter*`.
- **Multiplicative flux scale removed** from `fit_simulation`. The model's
  normalization is externally fixed (wind `Ṁ` + XSPEC table), so a free
  y-scale hid normalization errors: on the soft band the old fit returned
  `scale = 0.413`, χ²/dof 2.06; without it χ²/dof = 20.97 (a 59 % deficit).
  The shift search became a vectorized coarse scan + bounded refinement
  (Nelder–Mead from 0 landed in the wrong basin); `dof = N − 1` when the shift
  is fitted. `prepare_model_interpolator` / `model_from_wrap` became the single
  model evaluator for χ², overlay and residuals (the residual panel had ignored
  the shift); `plot_phase` recomputes the displayed χ² and warns on a >1 %
  mismatch.
- **MCMC scatter-path audit.** `compute_chi2_for_samples` had dropped
  `f_scatter` (χ² biased by −19 % to −78 %); walker initialization used an
  absolute `1e-12` inset that collapsed `f_scatter` (~1e-13) and aborted emcee
  — now a span-relative inset with a collapse guard; `fmt_val` switches to
  scientific notation below 1e-4 so `f_scatter` no longer prints as 0.
- **Adaptive binning in the single-model CLI** (`--counts-per-bin`): 87 bins
  averaging 104 counts on ObsID 15803 soft; χ²/dof 8.50 → 6.39.

---

## Phases 23–26 — `utils/`, Run-Config Persistence, Slimming, Geometry Plots

- **`utils/` extraction.** Everything shared moved to `utils/utils.py` (data,
  binning, smoothing, interpolation, `fit_simulation`) and
  `utils/plot_utils.py`; `plot_lightcurve_fit` is the single drawing routine
  used by both `plot_best_fit` and `plot_phase`. `chandra_phase_analysis.py`
  became a ~460-line CLI re-exporting the API for the notebooks. Plot titles
  carry only the band and χ²/dof.
- **Run-config persistence.** Every fit writes
  `<band>_<wind>_run_config.json` *before* sampling; `--replot` restores every
  option not typed explicitly (explicit flags win; `replot`/`output_dir` never
  restored; ambiguous directories error out; missing configs are self-healed).
  Fixed `--replot` for kepler/reparam chains and frozen parameters; a
  chain-vs-data `n_obs` mismatch warns.
- **Script slimming** (4246 → 3553 lines): band-directory resolution,
  phase-shift search, run-config code and the chunked CSV writer moved to
  `utils`; binners keep the caller's column names; `_aligned_model_flux`,
  `_phase_shift_opts`, table-driven `--prior-*` definitions (fixing a stale
  `--prior-r` help), dead `compute_pointwise_loglik` removed.
- **Geometry diagnostics.** `plot_geometry_diagnostics` draws the projected
  orbit against the companion disk (with a total/partial/no-eclipse verdict),
  `l3(φ)` against `R ± r`, `N_H(φ)` and band flux, and `g(r)` with posterior
  bands plus the band of radii the LOS actually probes (`[min l3, max l3]`).
  It immediately showed a `lam`-era fit with `R/a = 0.83` and `Rb` outside the
  probed range.

---

## Phase 27 — Standard Inclination Convention

`i0` had been measured from the line of sight. It is now the standard
astronomical inclination (from the orbital-plane normal; 90° = edge-on),
converted at the input boundary by
`inclination_to_internal_rad(i0) = (90 − i0)·π/180`; no geometry expression
changed (bit-for-bit round trip `old(x) == new(90 − x)`). Defaults and priors
were complemented (`26 → 64`, boxes reflected). Run configs carry
`"inclination_convention": "i0-from-orbital-normal"` and `--replot` warns when
the stamp is absent — chains from before store the complement and must be
refitted.

---

## Phase 28 — Physical Wind Normalization & Per-Cell Flux Conversion

**Why.** Under `mean(fl) = lam` the absolute column was discarded, so the model
depended only on ratios (`R/a`, `r/a`, `Rb/a`, `p`, `i0`): scaling all lengths
by λ ∈ [0.8, 1.5] changed the flux by ≤ 3e-3 while implied `M_tot` swung
14 → 94 M☉, and `q` at fixed `a` changed it by ~1e-15. Under `--kepler` the
masses were pure prior artifacts, the MAP wandered along the flat ridge, and
autocorrelation times reached 1500 steps. Separately, the visible-area term
`A2` was computed but never applied, so partial occultation did not attenuate
the flux at all (the model held full flux across a 2.3 h ramp, then dropped
abruptly, which is why `R` inflated to 14.6 R☉).

**Change.** `n₀ = Ṁ / (4π R☉² v_inf μ m_H C)` with `C` the profile's asymptotic
`r⁻²` coefficient (`wind_asymptotic_coefficient`, `wind_density_norm_from_mdot`),
so `N_H = f_opacity · n₀ · R☉ · ∫g dz` carries real units; `R` is again the
photosphere and the eclipse emerges from wind opacity. The kernel returns
per-cell columns and areas and the flux is converted **per cell before area
averaging** (`⟨F(N)⟩ ≠ F(⟨N⟩)`; converting the mean column is wrong by many
powers of e in the eclipse core). New CLI `--mdot`, `--v-inf`, `--mu-wind`,
`--fit-fopacity` (`log_fopa ~ N(−1.5, 1)`: Clark & Crowther's `Ṁ` predicts
`N_H ≈ 19–47×10²²` against `0.75×10²²` observed — photoionization, clumping and
He-rich abundances). The eclipse floor (~12 %) must come from `--fit-scatter`:
the wind core is genuinely opaque (Steiner et al. 2016 reach the same
conclusion).

**Caveat (sharpened in Phase 33).** With `f_opacity` fixed, physical
normalization changes the light-curve *shape* by 2.8–4.9 % under λ-rescaling
(vs < 0.3 % in `lam` mode); with `f_opacity` *free*, scaling every length
together with `f_opacity` is an exact invariance (verified to 1.5e-15 for all
three profiles), so `M_tot` is anchored only by the priors on `R` and
`f_opacity`.

---

## Phase 29 — Mass Reparameterization & Error-Column Fix

**`q` is exactly unidentifiable.** `l = a·√(sin²γ sin²i + cos²γ)` and
`z_start = a·sinγ·cos i` depend on `a = d1 + d2` alone; sweeping
`q = M_RH/M_tot` from 0.20 to 0.95 at fixed `a` changes the flux by 0.0
(`lam`) / 3.9e-16 (physical). **`--kepler-mtot`** samples `(M_tot, q_m)` so
the flat direction is axis-aligned; `q_m`'s posterior equals its prior
(confirmed: 0.628 ± 0.16 against a 0.6 ± 0.15 prior) and `M_X`, `M_RH`, `a`,
`d1`, `d2` are derived. Reporting rule: quote `q_m` as an assumption.

**Error-column bug.** `load_data` matched `flux_t` against *itself* (the
`.replace("FLUX", "FLUX_ERR")` candidate was a no-op on a lower-case name) and
accepted `rate_err` for any observable. Candidates are now upper-cased,
self-matches skipped, and rate errors offered only when the observable is the
rate, so `flux_t` derives `flux_t/√counts` (matching to 2.7e-13). Impact: at
100 counts/bin the median fractional error was **3.8× too large** and χ²
**too small by ~14.6×** — a reported χ²/dof of 1.02 was really ≈ 15. Expect
χ²/dof ≈ 4–5 on the broad band from ~20–25 % intrinsic variability, so
`--likelihood jitter` is not optional. Also fixed: `log_fopa` was omitted from
`mcmc_summary.txt`.

---

## Phase 30 — Physical Norm in the CLI, Model-LC Dump & χ²_eff Fix

- `xrb_lightcurve.py` gained `--wind-norm`, `--mdot`, `--v-inf`, `--mu-wind`,
  `--f-opacity` (physical mode had been reachable only through the MCMC), and
  the `--Rmax` auto-default no longer disabled the per-cell path in physical
  mode.
- Every fit writes `{band}_{wind}_bestfit_model.txt`: a `#` header with the
  point estimate, parameterization, χ²/dof, jitter `f`, phase shift, sampled /
  frozen / derived parameters, then the dense model curve and the observed
  bins with the model at their phases and normalized residuals.
- `mcmc_summary.txt` reports per-parameter autocorrelation times with the
  number of τ in the chain.
- **χ²_eff was wrong by ~9 dex:** `np.maximum(sigma2, eps)` imposed an
  *absolute* floor of 2.2e-16 on variances of order 1e-25. Now a positivity-only
  guard (`tiny`). The likelihood itself never used the clamp, so posteriors
  were unaffected.

---

## Phase 31 — Release Trim

Publication pass removing every alternative the physical normalization
superseded (`xrb_lightcurve.py` 2354 → 1196 lines): `--lam`/`--wind-norm` and
the `lam`-derived back-calculation helpers; the fixed-`Rmax` trapezoid path
(`--Rmax`, `--converge-rmax`, `--dz`, `--n_jobs`, `wind_los_integral`, …) —
any `Rmax` disabled the per-cell path; `broken_pl` (a special case of
`smooth_pl`) and `beta_law` (restored in Phase 32); `--flux_method legacy`
(`--flux_csv` now required). Numba became a hard requirement. Bugs fixed while
reviewing: `plot_geometry_diagnostics` ignored the wind normalization,
`--replot` dropped `fit_fopacity`/`kepler_mtot`, `load_existing_results` lacked
a `kepler_mtot` branch, the CLI `--Delta` default (1.0) disagreed with the
MCMC's fixed 2.0. README, `requirements.txt`, `rkp_run_w_mcmc_cmds.sh` and
`utils/test_flux_methods.py` rewritten. Physical-mode output bit-identical to
the previous HEAD.

---

## Phase 32 — `beta_law` Wind Profile Restored

The velocity-based profile (Wind_Density.pdf §5) is the one that follows most
directly from the physical normalization: `n = Ṁ/(4π r² v(r))`,
`v̂ = (1 − e^{−(r−R★)/H})(1 − R★/r)^β`, `g = 1/(r² v̂)`, `C = 1`. `g = 0` inside
the photosphere and diverges at the surface (limb-grazing rays are opaque);
effective break `R★ + 3H`. Registered as model id 2 with `(R_star, beta, H)`;
`R_STAR_TIED_MODELS = ("beta_law", "confinement")`; MCMC frees both `beta` and
`H` (`beta ~ N(0.8, 0.3)` on `[0.3, 2]`, `H ~ N(1, 0.7)` on `[0.1, 10]`;
`--freeze H=1` recovers a one-parameter fit); the wind-profile plot marks
`R★ + 3H`. Verification: the two existing profiles bit-identical; `beta_law`
column 1.05–1.17× a pure `r⁻²` wind of the same `Ṁ/v_inf`; per-ray GL16
accuracy 3e-6 at `b = R★ + 0.5` but 28 % at `+0.02` — irrelevant at the
light-curve level (GL16 vs GL64 ≤ 9e-9) because those cells are opaque;
scale–opacity invariance holds to 1.5e-15.

---

## Phase 33 — Kernel Symmetry, Compiled Flux Conversion, Single Band & MCMC Consolidation

Two commits: (1) the forward model — performance, correctness and single-band
simplification — and (2) the `mcmc_lightcurve_fit.py` consolidation. Prompted by
a repo-wide review for dead code and hot spots. All numbers: `henv`, 8 threads,
`dth = 1°`, `d2h = 6°`.

### Where the time went

| Stage | Before | After |
| ----- | ------ | ----- |
| numba kernel | ~50 ms | ~26 ms (mirror symmetry) |
| per-cell `N_H → flux` (`scipy.interp1d`, `10**`, `einsum`) | ~15–20 ms **per band in the CSV** | 1.6 ms (`_cell_flux_loglog`, agrees to 5e-15) |
| DataFrame assembly, clips, `nan_to_num` | ~3 ms | ~0 on the likelihood path (`simulate_band_flux`) |
| phase-shift search + likelihood | ~2 ms | unchanged |
| **one light curve** | **72 ms** | **30 ms** |

### Kernel (`_simulate_phases_numba`)

- **Mirror symmetry.** The impact parameter depends on the sector angle only
  through its cosine, so sectors `i` and `n_th − 1 − i` are geometrically
  identical. The cosine table is made exactly symmetric, half the sectors are
  integrated and each result is recorded twice (per-cell arrays keep their full
  size, so the flux conversion is unchanged).
- **Duplicate ring removed.** `n_th = 360/d2h + 1` put a ring at θ = 360° that
  duplicated θ = 0°: sector 0 carried double weight and `ΣA = 61/60` of the
  area. Now `n_th = 360/d2h` equal sectors of `2π/n_th`.
- **Mask at the sector centre.** Visibility was tested at the sector's leading
  edge while the column was evaluated at its centre; both now use the centre,
  and a radial segment is counted only when both of its bounding radii are
  visible (the old code let a masked interior cell be absorbed into a
  neighbouring segment).
- **Effect.** ≤ 8e-6 relative for the default point-like emitter. For a large
  emitter in partial eclipse (`r = 0.5`, `R = 2`, `i0 = 85°`) the change is a
  few per cent — and it is the *old* kernel that was off. Max relative error
  against the converged answer (`d2h = 0.25°`):

  | `d2h` | old `smooth_pl` | new `smooth_pl` | old `beta_law` | new `beta_law` |
  | ----- | --------------- | --------------- | -------------- | -------------- |
  | 12°   | 3.2e-2 | 2.0e-2 | 1.5e-1 | 5.7e-2 |
  | 6°    | 1.1e-2 | 1.1e-2 | 3.7e-2 | 2.3e-2 |
  | 3°    | 6.1e-3 | 3.9e-3 | 2.5e-2 | 8.6e-3 |
  | 1.5°  | 4.4e-3 | 2.0e-3 | 1.6e-2 | 7.6e-3 |
  | 0.75° | 1.5e-3 | 3.2e-4 | 4.3e-3 | 6.1e-4 |

  Old and new agree at `d2h = 0.25°` to 5e-4 / 1.3e-3 (both converge to the
  same limit; the residual is the fixed 10-cell radial grid).

### Flux conversion and API

- `_cell_flux_loglog` / `_cell_flux_exp` do the per-cell conversion and area
  average in one `prange` pass (`scipy` is no longer imported by
  `xrb_lightcurve.py`; `refit` uses `np.polyfit` in log space, fitted once per
  table and cached — the old path re-read the CSV from disk on **every**
  likelihood call).
- **One band per run.** `simulate_lightcurve(band=…)` / `simulate_band_flux`
  produce a single `nfl_{band}`; a table with one band needs no `band`, one
  with several requires it. `compute_flux_vs_nH.py --bands` → `--band`;
  `chandra_phase_analysis.py --sim-column` takes one column;
  `mcmc_lightcurve_fit.py --band all` is gone. Removed:
  `plot_multi_column_fits`, `validate_sim_columns`, `evaluate_model_at_phases`,
  the `--band all` loop.
- `simulate_band_flux(**kwargs) -> (phase, flux)` is the likelihood entry point
  (no DataFrame); `_simulate_core` is shared, and `utils/test_flux_methods.py`
  asserts the two agree exactly.
- `evaluate_g_profile` now loops the compiled `_g_profile` (`_g_profile_array`)
  instead of maintaining a NumPy mirror that had drifted before.
- Dropped output columns `ph` (duplicate of `deg`), `time` (a hard-coded
  `348.42` s/deg — IC 10 X-1's period/360, unused) and `icd`; the redundant
  eclipse-zeroing block went with them (eclipsed phases have no visible cells).
- `load_flux_vs_nh_csv` keeps rows with *any* valid band (the code ANDed while
  the comment said "at least one").

### Priors

`KEPLER_MTOT_PRIORS` still carried lam-era values — `R ~ N(9.5, 2.5)` on
`[3, 20]` and `r ~ N(2, 1.5)` — describing the old *opaque eclipsing radius*.
Under the physical normalization `R` is the ~2 R☉ photosphere, which the
`min = 3` box excluded, so every default `--kepler-mtot` run fitted with the
photosphere forbidden. The other modes capped `i0` at 80°, excluding edge-on.
All modes now share `R ~ N(2, 0.5)` on `[1, 5]`, `r ~ N(0.001, 0.001)` on
`[1e-4, 0.1]`, `i0 ~ N(78, 8)` on `[40, 89.9]`.

### `mcmc_lightcurve_fit.py` consolidation (4201 → 1896 lines)

Behaviour-preserving: on identical θ vectors across `phys`, `reparam`,
`kepler` and `kepler_mtot` (with frozen parameters, `f_scatter`, `log_fopa`,
jitter) the new prior, likelihood and statistics agree with the previous file to
**1.7e-16**; result directories from before the rewrite replot with identical
χ²/dof and BIC (31.4569 / −8531.977).

- **One parameter path.** Every function used to accept both `param_spec` and
  the legacy `reparam / kepler / active_names / wind_model / fit_wind_shape /
  likelihood` kwargs and reconcile them — the bug class behind the Phase 31
  replot regressions. `ParamSpec` is now the only argument and carries the
  resolution logic as methods: `value`, `geometry`, `wind_params`,
  `f_scatter`, `f_opacity`, `derived`.
- **`MODES` registry** (scale parameters, priors, flag, derived quantities per
  parameterization) plus shared `R_PRIOR` / `SMALL_R_PRIOR` / `I0_PRIOR` and
  one `PARAM_LABELS` dict replace four `*_PRIORS`/`*_PARAM_NAMES`/
  `*_PARAM_LABELS` triples and `get_mode_name_label` / `_default_priors` /
  `_labels_for_names`.
- **One derivation** of `a, q, d1, d2, M_X, M_RH` (`ParamSpec.derived`) serves
  the marginals, the MAP row, the model-curve header and the summary; it was
  written four times before.
- **`FitData`** bundles the observed arrays and the precomputed phase-shift
  terms; `evaluate_model` → `aligned_model_flux` → `chi2_terms` is the single
  evaluation chain under the likelihood, the per-sample χ², BIC and the
  best-fit overlay.
- **`postprocess_fit`** is shared by a fresh fit and `--replot` (ArviZ, BIC,
  figures, chi2 table); `build_parser`, `load_fit_data` and `write_summary`
  split the 899-line `main()`.
- **Replot** rebuilds the spec from the chain metadata and the saved column
  order, detecting `f_scatter`, `log_fopa` and shape parameters by name (the
  old detection counted `log_fopa` as a shape dimension). Chain-file keys are
  unchanged, so old directories load.
- Removed: `get_param_config`, the `log_likelihood` alias, the `TypeError`
  fallback for models without `wind_params`, the `DirectLightCurveModel`-only
  pool guard, an orphaned 15-line comment block, and the `--band all` loop.

### Verification (conda env `henv`, 2026-09-19)

- `utils/test_flux_methods.py` 6/6 (now also asserts `simulate_band_flux ==`
  DataFrame); `xrb_lightcurve.py` CLI → `chandra_phase_analysis.py --fit
  --write-model` → `plot_results.py` (bands and `--geometric`) on the new CSV.
- Forward model vs the pre-Phase-33 HEAD: ≤ 8e-6 relative at the default
  geometry for all three profiles and both flux methods; geometry columns
  identical to 4e-15; resolution study above.
- MCMC smoke tests: `phys`/chi2; `reparam` + `beta_law` + jitter +
  `--fit-wind-shape --fit-fopacity --fit-scatter --freeze H=1.0 --compute-bic
  --save-chi2 --smooth --compact-output`; `kepler_mtot` + `confinement` + zeus
  + `--freeze q_m=0.6`; `kepler` with `--n-threads 2`; `--replot` of the new
  run (χ²/dof and BIC identical) and of two pre-rewrite directories.
- `py_compile` on every changed file; `compute_flux_vs_nH.py` compiles but
  cannot be executed here (no PyXspec in this environment — pre-existing).

### Follow-up commit: `utils/utils.py` dedupe, a swallowed-error fix, stale docs

- **`read_observation` no longer swallows errors.** The whole header path was
  inside `try: … except Exception: pass`, so an unknown `--obs-column`, a
  header whose name count did not match the data columns, or a missing time
  column silently fell through to the headerless reader, which parses a
  CIAO file as three columns of the wrong quantities. Header detection is now
  `_header_columns()`; errors in the header path are raised, and the
  headerless reader is used only when no header exists. Column lookups go
  through one case-insensitive `find_column()` (the same loop was written five
  times) and error detection through `_detect_error_column()`; a requested
  but absent `--obs-error-column` / `--time-column` now warns before
  auto-detecting. Output DataFrames are identical to before on every CIAO file.
- **One `weighted_mean(values, errors)`** for both binners (the two ~25-line
  copies differed only in the degenerate all-errors-invalid fallback, which
  now always uses the guarded version). Binned outputs identical on the real
  data.
- `detect_flux_columns` is defined through `detect_energy_bands`; `frac`
  drops a redundant `abs`; unused `warnings` import removed from
  `xrb_lightcurve.py`.
- **Docs.** `PROJECT.md` referenced 17 files that are not in the tree
  (`xspec_fit_mcmc.py`, `compute_count_to_flux_factor.py`, `example_usage.py`,
  `MIGRATION_SUMMARY.md`, seven conversion guides,
  `mcmc_chi2_jitter_explanation.md`, `PERFORMANCE_VALIDATION_REPORT.md`,
  `compare_models.sh`, `convert_fits_to_txt_heasoft.sh`,
  `xspec_tbabs_fit_results.xcm`); the XSPEC section, data-layout pipeline,
  environment list, file inventory and known rough edges now describe what
  exists. README no longer promises extrapolation warnings the code does not
  emit (columns are clipped and extrapolated silently).

### Left as is

- `fit_simulation` (single-model CLI) and `apply_best_phase_shift` (MCMC)
  both did a coarse scan then refine, but with different refinement targets
  (bounded scalar minimization vs a 9-point grid); kept separate here and
  unified in Phase 34.
- `compute_flux_vs_nH.py` keeps its own small exponential fit for the plot
  annotation so it does not depend on numba in the XSPEC environment.
- `chandra_analysis_combined_flux.py` (untracked) is an unmigrated fork with
  its own multi-column helpers and a multiplicative flux scale; retiring it is
  recommended. The `utils/` data-prep scripts and the notebooks are unchanged
  (the notebooks were already stale).

---

## Phase 34 — Release Review: Performance, Correctness, Trimming (2026-09-19)

A second review round before release (ten review angles plus benchmarks on the
real 150-bin CIAO broad light curve), delivered as the commits in the order
below. All numbers: `henv`, 8 threads unless stated.

### Commit 1 — Halve the kernel by phase reflection, trim pow(), one exact shift search

**Phase reflection.** The kernel depends on the phase only through `sin γ` and
`|cos γ|`, so `γ` and `π − γ` give identical columns (`L` flips sign). With
`gma0 = −90` and any `dth` dividing 360 the uniform grid maps onto itself
(`_mirror_indices`), so `_simulate_core` runs the kernel and the flux
conversion on one member of each pair (181 of 360 phases at `dth = 1`) and
copies the rest. A full computation's two halves already agreed only to trig
round-off (≤ 2e-15; 6e-13 for rays grazing a `beta_law` photosphere).

**`smooth_pl` profile.** `x⁻²` as a division and `x^−Δ` as the only pow besides
the bracket: 3 → 2 `pow` per quadrature node, kernel ×0.75, agreement 5e-16.

**Phase-shift search** (`utils.best_phase_shift`, one implementation for the
MCMC likelihood and `fit_simulation`): a single `np.interp` over the
precomputed `(n_grid, n_obs)` matrix instead of 34 Python calls; the coarse
step tied to the data and model spacing (`n_grid = max(n_obs, 360/dth)`,
clipped to `[25, 400]`); two 33-point dense passes (resolution `step/256`); and
the kernel's native curve interpolated once onto the shifted observed phases,
dropping the intermediate 240-point resample (`--phase-shift-eval-points`
removed). On 25 prior draws the profiled χ² is now within 0.006 of a
brute-force minimum; the old search sat a median 1.9 (max 24) χ² above it and
the resample added ±7 more. `periodic_model` / `eval_periodic` replace the two
periodic interpolators (`interp_periodic_phases` with `[−1, 0, +1]` tiling and
`prepare_model_interpolator` / `model_from_wrap` with `[0, 2)` tiling, which
clamped queries below the first model phase). `FitData` holds a frozen
`PhaseShiftSearch` instead of a dict of terms.

**Also.** `--dth` default 5 → 2: the `dth = 5` discretisation was the largest
likelihood error (−479…+178 χ² across prior draws against a `dth = 0.25`
reference, ≤ 1.6 near the prior means) and `dth = 2` now costs what `dth = 5`
did. `--save-chi2` with the `chi2` likelihood reads χ² from the chain
(`−2(log_prob − log_prior)`, agrees with direct evaluation to 1e-14, no model
calls); jitter runs default to a 2000-sample subset. The smoothing band is the
exact `√(Σw²σ²)/Σw` of a linear smoother instead of a 2000-draw Monte Carlo
(`--smooth-n-mc`, `--smooth-seed` removed; agrees with a 40 000-draw MC within
its noise). Pooled runs send the fit context once through the pool
initializer, so only `theta` is pickled per task. `chandra_phase_analysis`
computes the smoothed overlay once instead of in both branches.

| | HEAD | Commit 1 |
| --- | --- | --- |
| kernel, `smooth_pl`, `dth = 1` | 28.7 ms | 15.1 ms (reflection) → 11.3 ms (+ pow) |
| one light curve, `dth = 1` | 30 ms | 12 ms |
| `log_probability`, 150 bins, `dth = 5` | 8.2 ms | 4.1 ms |
| `log_probability`, `dth = 2` (new default) | 20.8 ms | 7.9 ms |
| shift search | 1.0 ms | 0.6 ms |

Verification: forward model vs HEAD ≤ 6e-13 relative over 6 geometries × 3
profiles × 4 `dth` (including a grid without the symmetry, which takes the
full path); likelihood with the shift held at 0 identical to 5e-13; the
tabulated fit on the CIAO broad data gives the same shift (0.98351) and χ²/dof
(93.410) as HEAD's Brent refinement; tests 6/6; emcee serial with `--save-chi2`
and `--compute-bic`, emcee with a 2-process pool, zeus with `--smooth`, and
`--replot` all run. The MCMC path constructs no pandas DataFrame (checked by
counting constructions over 20 likelihood calls: 0), so the Phase 33 item
"return arrays, not a DataFrame, on the likelihood path" is closed.

### Commit 2 — Persist before plotting, validate up front, one prior and one default per parameter

1. **Chain persisted first.** `run_single_fit` wrote the samples CSV and the
   chain NPZ only after `postprocess_fit` (ArviZ, corner, best-fit and
   geometry figures, the chi2 table), none of which was guarded, so any
   exception there (corner raises on a constant column; ArviZ API mismatch;
   Ctrl-C) discarded a finished run and `--replot` had nothing to read. The
   three save blocks now run straight after sampling. The chain NPZ drops
   the fields replot never read (`reparam`, `fit_wind_shape`, `bic`,
   `logL_hat`, `k_params`); BIC lives in `*_model_metrics.csv` and the summary.
2. **`--n-burn ≥ --n-steps` rejected** at argument time; it used to fail in
   `np.percentile` on an empty chain after the whole run.
3. **Band validated at construction.** `DirectLightCurveModel` checked only
   that the flux CSV existed; a band absent from the table raised inside every
   likelihood call, was caught and turned into `-inf`, and emcee sampled the
   whole run with acceptance 0 and a "posterior" equal to the initial ball.
   The constructor now checks the band (`xrb_lightcurve.flux_table_bands`)
   and the flux method; `run_mcmc` evaluates the initial ensemble, aborts
   when every walker is `-inf`, warns when some are, and hands emcee the
   evaluated `State`.
4. **Reparam Jacobian removed.** `log_prior` added `+log a` in `--reparam`
   mode although the priors are stated directly on `(a, q)`; the effective
   prior was `a·N(a)` (mode +4.3 %, ~+13 % in a derived `M_tot` because `a` is
   prior-anchored through the exact scale invariance) and the Kepler modes,
   equally reparameterizations, had no such term. Priors live in the sampled
   space of every mode.
5. **One default per wind-shape parameter.** `ParamSpec.wind_params` filled
   unfitted, unfrozen shape parameters from the prior means (`fconf` 5,
   `ell` 1.0, `beta` 0.8) while the simulator and CLI use
   `default_wind_params` (10, 0.5, 1.0), so `--freeze ell=0.3` without
   `--fit-wind-shape` silently changed `fconf`. Defaults now come from
   `default_wind_params` (which also ties `R_star`); `WIND_SHAPE_FIXED` is
   gone and the registries are asserted against `xrb_lightcurve` at import.
6. **One dof convention.** `degrees_of_freedom` counts the profiled phase
   shift, as `fit_simulation` always did, so the χ²/dof on the best-fit
   figure, in the model dump and in the chi2 table is comparable with the
   tabulated fit.

### Commit 3 — Replot rules, one error rule, input validation, exit codes, `--seed`

Replot (`utils.apply_saved_run_config`):

7. **Output-control flags are never restored.** `store_true` flags cannot be
   negated on the command line, so a fit run with `--no-plots` could never be
   replotted with figures and a `--save-chi2` run recomputed its table on
   every replot. `_RUN_CONFIG_NEVER_RESTORE` now lists every output option;
   only what defines the fit comes back.
8. **Exclusive-group siblings are not restored** when one member is typed
   (`--replot --n-phase-bins 30` on a `--counts-per-bin` run used to die on
   the exclusivity check); the self-healing run config for pre-config
   directories is written only after a successful replot, not before.
9. **Normalization stamp.** Run configs and chain files carry
   `wind_normalization = "physical-mdot-vinf"`; results without it (every
   pre-Phase-34 directory, including the `lam`-mode `mcmc_results/`) are
   refused with an explicit message. Replotting them silently evaluated the
   MAP under a different model (`N_H` 20–50 × 10²² against the chain's 0.53).

Robustness:

10. `phase_bin_data_snr` validates the counts (finite, non-negative, not all
    zero) instead of `fillna(0)`, which collapsed a light curve without
    counts into one bin; `load_observed_lightcurves` emits `counts` only when
    the files carry it.
11. **One error rule.** `sanitize_errors` (median valid error for non-finite
    or non-positive entries, warning, `ValueError` when no error is valid)
    replaces three rules: the binners' median patch, `load_fit_data`'s
    `max(0.1·|flux|, median)` and `obs_errors`' count-rate constants (a `1e-3`
    absolute floor that zero-weighted a `1e-13` flux point and a `sqrt(|rate|)`
    fallback that turned a 50 %-off model into χ²/dof = 2e-14). `obs_errors`
    now requires an error column.
12. The headerless reader rejects files with other than 2–3 columns; with
    `names=` pandas silently promoted a surplus leading column to the index,
    shifting `time/rate/error` by one.
13. `phase_bin_data` drops NaN phases first (they landed in the last bin) and
    raises when no bin reaches `min_points_per_bin` (the result used to lack a
    `phase` column).
14. The flux table keeps the *numeric* `nH_1e22` column and rejects repeated
    `nH` rows: an object-dtype column sorted lexicographically and the
    compiled interpolator returned NaN or wrong fluxes without a message.
15. `_simulate_core` rejects `r ≥ R` (the eclipse test assumes the emitter
    disk is the smaller one) and `d1 + d2 ≤ 0`.
16. `main()` exits 1 on failure instead of printing and returning 0.
17. `--seed` seeds the initial ball, emcee (handed the global state), zeus
    and every random subset; two runs with the same seed give identical
    samples.
18. `--keep-zero-flux` (both fitters) keeps rows with zero flux on load
    instead of dropping them as gaps. The review measured that 15 % of the
    in-eclipse CIAO bins are genuine zero-count bins and that dropping them
    raises the eclipse-window mean, the `f_scatter` prior centre, by 18 %; the
    default still drops them, the flag makes the choice explicit. Kept rows
    have zero errors, which `sanitize_errors` replaces by the median valid
    error.

Verified in `henv` (30 checks, `g3_verify`): identical samples for equal seeds
and different ones otherwise; a `--no-plots` fit replots with figures; an
exclusive-group override replots with the `n_obs` warning; a config or chain
without the stamp is refused (exit 2 / 1) and the real `mcmc_results/` replot
is refused without touching the directory; NaN/zero counts, 4-column
headerless files, no-error data, empty binning, duplicate `nH` rows and
`r ≥ R` all raise; an object-dtype table sorts numerically; a missing data
directory exits 1; the tabulated fit runs on both data layouts and the tests
pass 6/6.

### Commit 4 — Trimming: one source for the defaults, the XSPEC script, shared helpers

No numerical change: the forward model, the likelihood and every observation
file read identically to the Commit 3 tree (max difference 0.0).

- **Simulator defaults defined once.** `_simulate_core` is keyword-only with
  the defaults in its signature; `SIM_DEFAULTS` exports them and the
  `xrb_lightcurve.py` CLI, `DirectLightCurveModel.sim_kwargs` and the MCMC
  argparse defaults read from it, while the wind-shape CLI defaults come from
  `default_wind_params`. `simulate_lightcurve(verbose=False, **kwargs)` and
  `simulate_band_flux(**kwargs)` forward their keywords, so a misspelled one
  raises `TypeError` (`simulate_band_flux` used to pull every argument with
  `kwargs.get(name, default)`; `f_opa=0.02` ran silently at `f_opacity = 1`).
  The unused fourth profile slot `p4` is gone from `pack_wind_params`,
  `_g_profile`, the quadrature and the kernel signature. `_simulate_core`
  returns the columns in output order, so the DataFrame is one comprehension.
- **`compute_flux_vs_nH.py` 930 → 384 lines.** PyXspec parameters by index
  (`.values = x`, `.values[0]`, `.sigma` instead of the never-populated
  `.error`); the unreachable index-vs-name fallbacks removed; spectrum files
  matched on file names with the background identified first; background and
  RMF/ARF attached explicitly and loudly; the energy grid and plot device set
  once; `np.trapezoid`; the figure's exponential law is `utils.fit_exponential`
  (moved to `utils/utils.py`, which is numpy/pandas only and therefore
  importable in the XSPEC environment; `xrb_lightcurve` imports it from
  there). Exercised end to end against a PyXspec stand-in; the resulting table
  is consumed by `simulate_band_flux`.
- **Shared helpers.** `write_model_blocks` writes the two data blocks of both
  model dumps (`write_model_lightcurve`, `_write_bestfit_model_txt`); the two
  χ² self-checks in `plot_phase` and `write_model_lightcurve`, which guarded a
  shift/scatter mismatch that the single caller cannot produce, are gone, as is
  the `red_chi2` argument they needed.
- **Small items.** `plot_corner`/`plot_trace` draw on their own figure and plot
  all walkers in one call; `getattr(args, …, default)` restating argparse
  defaults replaced by `args.x`; the unused `stats` argument of the model dump
  removed; `_detect_error_column` drops the `.replace` candidates that never
  matched a file; `find_run_configs` drops the `band='all'` case; the `--rescale`
  alias and the duplicated smoothing block in `chandra_phase_analysis` are gone
  and `--obs-column` defaults to `rate` in argparse; the test uses
  `np.isfinite`; f-strings without placeholders and an unused import fixed.
- **Data-prep scripts.** `utils/convert_fits_to_txt.py` looked for the data
  under `utils/data/` since the move into `utils/`; it now resolves the
  repository root. `utils/add_flux_to_lightcurves.py` failed with `KeyError`
  on the converted layout and duplicated `add_flux_simple.py`; removed.
  `get_average_count_rates.py` no longer swallows every exception or points at
  a script that does not exist. `scipy` is no longer imported anywhere and
  leaves `requirements.txt` (emcee and arviz pull it in).

Tracked Python: 7712 → 6945 lines across the four commits.

### Commit 5 — Phase windows, fixed phase shift, argument validation

- **`--phase-window LO HI`** (both fitters, default `0 1`, `LO > HI` wraps)
  keeps only the observed points inside the window; the model is still
  evaluated over the full orbit (free, the kernel already computes the unique
  half). Meant for separate ingress and egress fits of asymmetric data, which
  the exactly symmetric model cannot produce, so the two posteriors are an
  asymmetry test.
- **`--phase-shift SHIFT`** holds the shift at a chosen value (`FitData.fixed_shift`,
  `fit_simulation(fixed_shift=)`); previously the shift was either searched or
  0. A partial window **requires** a fixed shift: with one eclipse edge in the
  data the eclipse width is degenerate with a free shift, so both scripts
  refuse `--phase-window` with the search enabled. Workflow: full-orbit fit,
  then half-orbit fits with its shift. The scattered-flux window must overlap
  the data window.
- **Argument validation** (`validate_args` / `_validate_args`, using
  `utils.explicit_cli_dests` to distinguish typed options from defaults and
  restored values): binning exclusivity incl. `--no-phase-bin` and
  `--min-points-per-bin`; `--fit-fopacity`/`--freeze log_fopa` and
  `--fit-scatter`/`--freeze f_scatter` contradictions; no-effect options
  (`--scatter-eclipse-phase` without `--fit-scatter`, `--prior-<name>` for
  another mode or wind model or without `--fit-wind-shape`, `--orbital-period`
  outside the Kepler modes, `--chi2-n-samples` without `--save-chi2`,
  `--smooth-sigma` without `--smooth`, `--csv-chunk-size` with
  `--no-csv-output`, `--numba-threads-per-worker` without a pool, sampling
  options with `--replot`, fit-only options without `--fit`, `--scatter` with
  `--scatter-eclipse-phase`); ranges (`--n-walkers` even and ≥ 2·n_dim before
  any data is loaded, `--dth`/`--d2h` divisors of 360, positive `--mdot`,
  `--v-inf`, `--mu-wind`, `--seed` in `[0, 2³²)`). The seed is applied after
  validation, so a bad seed is a clean argument error.

Verified in `henv`: with the default window the searched χ² is unchanged
(6774.742 at θ₀) and a fixed shift equal to the searched one reproduces it
exactly; `--phase-window 0 0.5 --phase-shift 0.985` keeps 835 of 1442 points
and replots with both options restored; a wrapping window runs; the tabulated
fit with the fixed searched shift gives the same total χ²; 18 rejected MCMC
combinations and 8 rejected tabulated-fit combinations exit 2 with the
intended message, and the valid freeze-one-fit-one wind-shape case runs.

### Commit 6 — Final pre-release review: replot from the chain, shared helpers, edge cases

A second ten-angle review of the whole tree (one angle ran PyXspec live under
HEASoft) plus a 24-run smoke matrix (four parameterizations × two samplers ×
three wind models, all green). Findings fixed:

- **Replot reads `*_chain.npz` only.** It required `*_samples.csv` (so a
  `--no-csv-output` fit could not be replotted) and checked the normalization
  stamp only if a chain file happened to exist (so a samples-only legacy
  directory replotted unchecked and was then stamped by the self-heal). The
  chain file, always written right after sampling, now supplies chain,
  log-prob, parameter names and metadata; the CSV is an export;
  `--compact-output` (an NPZ nothing read) is gone; chain columns overlapping
  the frozen set are refused.
- **Never-restore set derived from the parser.** `build_parser` groups the
  invocation-only options into *Execution* and *Output* argument groups and
  `NEVER_RESTORED_DESTS` is built from them. `--seed` is accepted on
  `--replot` (the chi2 subsample and wind-profile draws are random);
  `--no-fit-phase-shift` and `--phase-shift` are mutually exclusive, for
  validation and for the restore.
- **From-chain χ² guard.** `-2(log_prob - log_prior)` only holds with the
  sampled priors; on a replot with a typed `--prior-*`, or a rebuilt spec
  whose parameters lack priors, the table falls back to model evaluation with
  a note, and a chain whose mode differs from the command line gets that
  mode's default priors with a warning.
- **zeus start-up.** Initial walkers at `-inf` (outside `r < R`, `Rb ≥ R` or a
  box) are redrawn up to 20 times; emcee only rejected moves from them, zeus
  refused to start. `set_num_threads` failures are no longer swallowed and
  `--numba-threads-per-worker` is checked against numba's limit.
- **Silent inputs made loud.** `DirectLightCurveModel` rejects unknown
  `sim_params` keys and `curve()` catches only the simulator's `ValueError` /
  `ArithmeticError`; `--prior-*` overrides need `STD > 0`, `MIN < MAX` and the
  mean inside the box; `_simulate_core` rejects `dth`/`d2h` that do not divide
  360 (the reflection assumes a closed grid: `--gma0 20 --dth 7` copied 15 of
  51 phases from the wrong partner); a table band with fewer than two usable
  rows raises instead of vanishing; `check_phase_window` rejects `1 0`; the
  scatter window must overlap the data window with positive length.
- **One rule, one place.** `drop_invalid_flux_rows` (both fitters drop `≤ 0`;
  the tabulated fitter used `!= 0`), `apply_phase_window`, `model_dump_path`,
  `dest_to_flag`, `tabulated_model_arrays`, `validate_binning_args` /
  `validate_phase_window_args`; `weighted_mean` lost its own repair rule and
  the tabulated fitter sanitizes before binning (which also removes the
  machine-epsilon errors that gave `--keep-zero-flux` bins weight 10²⁸); BIC's
  `k` counts the profiled shift and uses the overlays' point estimate;
  `--prior-fopa` exists (its override merge was dead code);
  `interp_periodic_phases`, `plot_phase(shift_fitted)`, `stats['wind_model']`
  removed.
- **Windows and figures.** Constant-counts bins are formed along the phase
  measured from the window's lower bound, so a wrapping window no longer merges
  points across its seam into a bin centred in the excluded gap; the smoothed
  curve is evaluated inside the window only; the orbit figure uses
  `sin(gma) > 0` for "behind" (exactly edge-on `h == 0` everywhere) and states
  when the emitter is never behind the companion; the smoothed-band legend no
  longer says "MC".
- **XSPEC script (verified live against HEASoft by the review).** The PHA
  header's BACKFILE/RESPFILE/ANCRFILE pairing is kept and directory files fill
  only the gaps: assigning `spectrum.response` replaces the Response object and
  dropped the header's ARF whenever no `*.arf` matched, and `spectrum.response`
  raises rather than returning None, so the previous guard was dead. The
  RMF/ARF fallbacks no longer pick a background response; `.gz` responses are
  matched; the model energy array is extended once so `calcFlux` is exact over
  the band (XSPEC clips a band to the model array; bands outside 0.1–20 keV
  are rejected); `chi2_red` is NaN, not 0, when dof ≤ 0; PyXspec is imported
  after argument parsing so `--help` works without HEASoft. The `refit`
  exponential is documented as orientation only: fitted with equal weights in
  log space it misses the low-`nH` plateau by ~2× for the broad band.
  `CHANDRA_BANDS` in `utils.utils` is the single band definition.
- Known approximation, documented: under the jitter likelihood the shift is
  profiled on the classical χ²; the difference to profiling on the jitter
  variance is second order in `f`. Efficiency notes recorded for later: the
  coarse shift scan could be an FFT correlation (~0.5 ms → 0.15 ms), the
  log-uniform flux table allows direct indexing instead of bisection
  (0.9 → 0.5 ms), and pool workers import matplotlib/arviz/zeus they never use
  (~2.5 s per worker start).

Verified in `henv`: forward model and likelihood bit-identical to the previous
tree; tests 6/6; the group-3 and group-5 check scripts all green; a
`--no-csv-output` fit replots; a samples-only directory is refused; zeus starts
from a prior that puts half the initial ball at `-inf`; a wrapping window with
`--counts-per-bin` keeps every bin centre inside the window; `--replot
--save-chi2 --prior-R ...` falls back to model evaluation; the XSPEC script
runs against the PyXspec stand-in (response, background, `calcFlux` and
`fakeit` semantics mirrored from the live audit); pyflakes clean.

### Commit 13 — Figure pipeline review: no reduced mode, input-keyed fit caches, cross-band floor

- The quick/smoke mode is gone (`CLOAK_QUICK`, `--quick`): the notebook and
  both scripts always produce the paper's numbers. Smoke tests of the
  calibration script use explicit small `--n-walkers/--n-steps/--n-burn/--thin`.
- Fit caches are keyed by a digest of the flux table, the light-curve files and
  every fit option (`figures/cache/<name>_<digest>/`), so a regenerated table
  or data set can never reuse a stale chain; `run_fits.py` and the notebook
  share `figlib.FIT_NAMES`.
- Cross-band prediction rescales the fitted scattered floor (a flux of the
  fitted band) by the ratio of the bands' out-of-eclipse fluxes; the broad
  floor added unscaled to the soft band was 3.6 times too high.
- `eclipse_width_half_depth` measures the contiguous run around the minimum on
  the curve rolled to phase 0.5 (a dip through phase 0 returned ~1).
- The injection figure uses the jitter likelihood's effective variance at the
  MAP for bars, residuals and the quoted chi2 (System A carries 10 % intrinsic
  variability by construction, so the classical chi2 misrepresented the fit);
  the band is labelled as posterior model curves; the corner plot notes that
  the `ln f` reference is the injected amplitude only approximately.
- Priors table: the floor prior is truncated at the brightest bin as well as
  at zero. Tables from two spectra are never mixed: the generic tables are
  used exclusively once any exists. PDF only (no PNG previews).
  `fiducial.visits()` derives the visits from the period. Notebooks carry the
  `henv` kernel and say so. The SBC batch generates at 1 degree and fits at the
  paper's 2 degrees. Stale loader message about repaired zero-count errors fixed.
- Verified: both notebooks execute at full size with the four cached fits
  (25.8 + 17.5 + 17.1 + 16.1 min); tests 13/13; pyflakes clean.

### Commit 12 — Paper figures: legends in a strip below the panels

Every legend sat inside the axes and covered data; each figure now gets a
legend strip beneath its panels from the layout engine (`figlib.panels` /
`put_legend`), panel letters are left-aligned titles, labels were shortened,
and the quadrature, invariance, per-cell and convergence figures were tidied.

### Commit 11 — Paper figures: fiducial systems, two notebooks, calibration batch

- `cloak/synthetic/fiducial.py`: the paper's two generic systems (A WR-like,
  B OB-like) with observing patterns; `kepler_prefactor`, `total_mass`,
  `geometry`, `simulation_kwargs`, `describe`.
- The orbital period is threaded through: `--orbital-period` folds the
  light curves in both fitters (it used to enter Kepler's law only) and in the
  generator (`--n-orbits` and the fold), `read_observation` / `load_data` /
  `load_observed_lightcurves` take `period`/`epoch`. The generator also gained
  `--intrinsic-scatter EPS` (mean-preserving log-normal variability per bin,
  for the jitter likelihood).
- `cloak.kernel.emitter_cell_columns`: per-cell columns and positions across
  the emitter disk at one phase (tested against the kernel's own mean column).
- `synthetic_data/generate_synthetic_data.ipynb` (tracked): tables under
  HEASoft, light curves of both systems, diagnostics. Wrote System A/B broad
  light curves (593 and 2058 bins of 100 counts after binning).
- `figures/paper_figures.ipynb` + `figures/figlib.py` (tracked): fig02 wind
  profiles, fig03 quadrature (plain rule vs the kernel's limb split), fig04
  convergence (phase step, sector size; interpolated onto the finest grid,
  normalized to the out-of-eclipse flux), fig05 per-cell vs mean column
  (System B, 6 R☉ disk: factor 16), fig06 energy dependence, fig07 + table
  invariances, fig08/fig09 + table injection-recovery of System A (MAP and
  68 % predictive band; corner with injected values), fig10 SBC rank-ECDF
  panels, fig11 the (M_tot, log f_opa) ridge under three priors, fig12 the
  profiled shift against brute force, fig13 bin estimators, fig14 cross-band
  prediction, and the performance table. MCMC fits run as cached
  `python -m cloak.mcmc_fit` subprocesses; `CLOAK_QUICK=1` shrinks them for a
  smoke run. `figures/run_sbc.py`: 100 prior draws (q_m and f_scatter
  frozen, chi2 likelihood, same dth for data and fit), resumable, ranks to
  `figures/results/sbc_ranks.csv`.
- `.gitignore`: `figures/cache/` ignored; figure PDFs/PNGs and
  `figures/results/*` re-included. Manuscript: quadrature caveat and
  observation-model paragraphs rewritten for the limb split and the
  exposure-weighted bins; `\graphicspath{{../}}` in the local root.
- Verified: both notebooks execute end to end under the `henv` kernel (quick
  mode: 13 figures, 5 tables, 13 inline images); the SBC script on two quick
  draws; tests 13/13; pyflakes clean. The radial cell count stays a module
  constant (measured effect < 0.3 % for extended emitters, none for
  point-like ones).

### Commit 10 — Rename into the `cloak` package

The flat scripts became one importable package named after the model
(CLOAK: Column-density and Line-of-sight Occultation & Absorption Kernel).
Pure moves plus reference rewrites; no behaviour change.

| Before | After | Run as |
| ------ | ----- | ------ |
| `xrb_lightcurve.py` | `cloak/kernel.py` | `python -m cloak.kernel` |
| `compute_flux_vs_nH.py` | `cloak/flux_table.py` | `python -m cloak.flux_table` |
| `mcmc_lightcurve_fit.py` | `cloak/mcmc_fit.py` | `python -m cloak.mcmc_fit` |
| `chandra_phase_analysis.py` | `cloak/phase_analysis.py` | `python -m cloak.phase_analysis` |
| `plot_results.py` | `cloak/plot_results.py` | `python -m cloak.plot_results` |
| `utils/utils.py` | `cloak/utils.py` | (library) |
| `utils/plot_utils.py` | `cloak/plots.py` | (library) |
| `utils/__init__.py` | `cloak/__init__.py` | package docstring, `__version__` |
| `synthetic_data/make_spectrum.py` | `cloak/synthetic/spectrum.py` | `python -m cloak.synthetic.spectrum` |
| `synthetic_data/make_lightcurve.py` | `cloak/synthetic/lightcurve.py` | `python -m cloak.synthetic.lightcurve` |
| `synthetic_data/README.md` | `cloak/synthetic/README.md` | |
| `utils/test_flux_methods.py` | `tests/test_flux_methods.py` | `python tests/test_flux_methods.py` |
| — | `tests/test_pipeline.py` | `python -m unittest discover -s tests` |

`synthetic_data/` is now the data directory for synthetic products (its
generators moved into the package). Each CLI module carries a guard that puts
the repository root on `sys.path` when it is run as a plain script, so
`python cloak/mcmc_fit.py` works from any directory as well; spawn-based
worker pools inherit the path. Imports are `from cloak.utils import ...`,
`from cloak.plots import ...`, `from cloak.kernel import ...`. Documentation,
help strings and the synthetic README were rewritten for the new names;
history in this file keeps the old ones. `tests/test_pipeline.py` (12 tests)
exercises the kernel symmetries, the periodic helpers, the window rules, the
tabulated fit's shift recovery, a tiny MCMC fit with replot and two rejected
argument combinations, all on synthetic data; both test modules pass, every
module runs with `-m` and as a script, and `--replot` of an earlier result
works through the new entry point.

### Commit 9 — Release split: real data, notebooks, legacy and helper scripts untracked

- Untracked (kept on disk, now ignored): `data/` (88 Chandra light curves of
  IC 10 X-1 plus the spectra and responses that were never tracked),
  `notebooks/` (4), `legacy_r_code/` (4 R files), and the one-off
  legacy-layout conversion scripts `add_flux_simple.py`,
  `convert_fits_to_txt.py`, `get_average_count_rates.py` together with the
  author's command log `rkp_run_w_mcmc_cmds.sh`, moved into `extras/`.
  `.gitignore` also covers `mcmc_results/`, `temp/`, `paper/`, `*.pdf` and
  editor state, and negates the blanket `*.csv`/`*.txt`/`*.json` rules under
  `synthetic_data/` so synthetic products are released.
- Tracked: `synthetic_data/flux_vs_nH_tbabs_broad.csv`, the TBabs broad-band
  table (IC 10 X-1 spectral parameters, 1001 points); the regression test
  defaults to it.
- No default points at absent data any more: `--data-dir` is required in both
  fitters (except with `--replot`), `--specdir` in the flux-table script,
  `--rmf`/`--arf` in the fake-spectrum generator;
  `load_observed_lightcurves` has no default directory. Docs describe the
  accepted file layouts and the release policy instead of the local data tree.
- Earlier commits still carry the real light curves and the notebooks; a
  public release should start from fresh history (or rewrite it).

### Commit 8 — Third review round: kernel occultation and quadrature, exposure-weighted bins, inference fixes

Six independent reviewers (kernel numerics against quadrature and Monte
Carlo references, statistics, data pipeline, runtime/portability, docs/CLI
drift, release inventory). Confirmed defects fixed:

- **Kernel.** The occultation mask is evaluated at each radial segment's
  centre (the point its impact parameter uses); requiring both bounding radii
  to be visible dropped every segment straddling the limb and left the visible
  area of a partially eclipsed extended emitter 40–60 % low at `r ~ R`
  (within ~1 % of a Monte Carlo now; no effect at the default `r = 0.001`).
  `beta_law` rays grazing the photosphere (`b − R⋆ < 0.3`) are integrated
  piecewise around the closest-approach peak (`_gl_piece`): 16 nodes over the
  whole interval were −30 % at `b − R⋆ = 0.01` and −90 % at 0.001, now < 1e-7.
  Below the table's first `nH` the flux is held at the first tabulated value
  (the log-log end-segment extrapolation returned up to 28 % above the
  unabsorbed plateau). `i0` must lie in [0, 180] (a negative `i0` inverted the
  "behind the companion" test), `mdot > 0`, `f_opacity ≥ 0`, `dth`/`d2h ≤ 180`.
  A model file without a `phase` column is refused (`deg / 360` put mid-eclipse
  at 0.25).
- **Binning.** Both binners weighted rows by their own Poisson errors, making
  every bin the harmonic mean of its counts: −25 % at 3 counts per row, −12 %
  at 10 (0.72 in-eclipse on the real broad light curve, χ²/n 2.3–33 on
  synthetic data). Bins are now exposure-weighted (`bin_estimate`), with the
  exposure from an `EXPOSURE` column or `counts / rate`; the error is the
  propagated Poisson error; zero-count rows contribute exposure and no
  variance; bin centres are exposure-weighted. Unbiased at every count level
  in simulation, χ²/n ≈ 1. Rows with zero exposure are dropped as unobserved;
  the loaders keep row errors as read and the fitters repair the errors that
  enter the χ² (after binning). An `mjd` time column is converted to seconds
  (it was fed to the ephemeris as seconds); an error column a thousand times
  the observable is refused as a unit mismatch; empty data files are named in
  the error. The synthetic generator writes an `exposure` column, applies gaps
  per visit (never across a boundary, never emptying a visit), refuses
  overlapping visits, and gained `--flux-type`.
- **Inference.** `--seed` now also seeds Python's `random` (zeus draws walker
  pairs with it; two zeus runs are byte-identical). BIC uses the maximum
  likelihood over the chain (`log_prob − log_prior`), not the MAP sample's
  likelihood (ΔBIC 5.6 in a 30-step test). Under the jitter likelihood the
  phase shift is profiled on that likelihood (variance term included) instead
  of the classical χ² (16 log-units apart at `f = 0.45`). Autocorrelation
  times and effective samples are computed on the post-burn-in chain. Samples
  and log-probabilities are flattened in (step, walker) order for both
  samplers, matching `--replot`. The summary file is
  `{band}_{wind}_summary.txt` (a second wind model overwrote
  `mcmc_summary.txt`); an existing chain is overwritten with a warning.
- **Runtime.** Ctrl-C on a pooled fit hung forever (`pool.join()` on an
  in-flight task); the pool is terminated and workers ignore SIGINT. The
  per-worker numba thread count is clamped to numba's limit and the
  initializer never raises (a raising initializer made `Pool` respawn workers
  forever). Console output is ASCII (`σ`/`χ²` crashed under a non-UTF-8
  stdout); stdout is line-buffered so logs interleave correctly; expected
  user errors (missing files, bad columns, empty data, bad bands) exit with one
  line instead of a traceback in every CLI; `plot_results.py` shows the
  geometry figures when no `--output` is given; the chain is loaded with
  `allow_pickle=False`; `matplotlib.use("Agg")` in the MCMC fitter.
- **XSPEC/synthetic.** The flux-table script prefers the source's own
  `<stem>_bkg.*` background; a fake-spectrum `--name` containing "bkg" is
  refused. Documentation brought in line with the code (refit error ~2×, BIC
  and shift definitions, `nH_cm2` column, `--seed` on replot, plot locations,
  parameter counts, CIAO column examples, recovery numbers with the count rate
  they were obtained at, requirements list) and a Known rough edge added for
  the Neyman bias of observed-error χ² weights (−0.005 in the shift at ~14
  counts per bin).

Verified in `henv`: unit tests 6/6; the kernel reviewer's reference scripts
(limb quadrature ≤ 2.7e-8, partial-eclipse area within 1–5 % of Monte Carlo,
mirror symmetry, flag checks); the data reviewer's bias script (both binners
unbiased, χ²/n 1.06–1.21); the earlier group-3 and group-5 check scripts;
synthetic light curve → both fitters (shift 0.9848 / 0.9852 / 0.9847 at ten
times the real count rate; χ²/dof 1.05 / 0.92); MCMC chi2 and jitter runs
with BIC and χ² tables; replot; zeus seed reproducibility; Ctrl-C on a pooled
run exits in 0.5 s with no orphans; pyflakes clean.

### Commit 7 — `synthetic_data/`: fake spectra and synthetic light curves

- `make_spectrum.py`: PyXspec `fakeit` of an absorbed power law through the IC
  10 X-1 combined ACIS response (`--nH`, `--PhoIndex`, `--norm`, `--exposure`,
  optional background, `--seed`), written to a directory that
  `compute_flux_vs_nH.py --specdir` consumes; reports per band the model flux
  (`calcFlux`), the fake net count rate and their ratio (`band_factors.json`).
- `make_lightcurve.py`: the forward model at known parameters (every
  `xrb_lightcurve.py` keyword), shifted, lifted by a scattered floor, converted
  to counts with a flux-per-rate factor and `--dt`, Poisson-sampled over
  visits with random gaps and an optional background, written in the CIAO
  layout the fitters read unchanged, plus `<stem>_truth.json`.
- README with the spectrum → table → light curve → fit sequence. Exercised end
  to end: stand-in spectrum → table → light curve → both fitters; with the
  injected floor passed as `--scatter`, the tabulated fit recovered the
  injected shift 0.985 as 0.9848 (constant-counts bins), 0.9850 (100 bins)
  and 0.9847 (unbinned); the default eclipse-window floor estimate assumes a
  total eclipse and biases a partial dip's shift to 0.960, which the README
  now says.

---

## Side Investigation — Reference Epoch

Per Laycock et al. 2015 (`stu2151.pdf`, §4), `T0 = 278801348 s` is the
**mid-eclipse** time of ObsID 07082 at **phase 0.5**, whereas the code's
`frac((t − T0)/P)` puts it at phase 0.0. A read-only study derived a corrected
epoch of `278800407.267 s`, which sits commented out next to `REF_EPOCH`. The
per-sample phase-shift search absorbs the offset, so this mainly affects the
interpretability of plotted phases; the study script is not in the tree.

---

## Current File Inventory

### Core simulation / inference
| File | Lines | Description |
| ---- | ----- | ----------- |
| `xrb_lightcurve.py` | ~1050 | Forward model: profiles, Numba GL kernel (mirror-symmetric sectors, half the orbit by phase reflection), compiled per-cell flux conversion, `simulate_lightcurve` / `simulate_band_flux`, `SIM_DEFAULTS`, physical normalization. |
| `mcmc_lightcurve_fit.py` | ~2140 | emcee/zeus MCMC: `MODES`, `ParamSpec`, `FitData`, prior/likelihood, phase-shift search, BIC, plots, replot, summary. |
| `chandra_phase_analysis.py` | ~510 | CLI for the single-model χ² fit; re-exports the shared `utils/` API. |
| `utils/utils.py` | ~1670 | Ephemeris, loading, `sanitize_errors`, both binners, smoothing, the periodic interpolator + phase-shift search, `fit_simulation`, model-dump blocks, run-config persistence. |
| `utils/plot_utils.py` | ~900 | All plotting on the single `plot_lightcurve_fit`; geometry and wind-profile figures. |
| `compute_flux_vs_nH.py` | ~380 | XSPEC flux-vs-nH table generator, one band per table. |
| `plot_results.py` | 104 | Thin CLI over `utils/plot_utils.py` (`--geometric`, `--orbit`). |
| `synthetic_data/` | ~410 | `make_spectrum.py` (PyXspec `fakeit`, per-band flux-per-rate) and `make_lightcurve.py` (CIAO-layout synthetic light curves + truth JSON). |

### Utilities, scripts, references
`utils/` is a package (`utils.py`, `plot_utils.py`); `test_flux_methods.py` is
the regression test; `convert_fits_to_txt.py`, `add_flux_simple.py`,
`get_average_count_rates.py` are standalone data-prep scripts
(`add_flux_to_lightcurves.py` was removed in Phase 34). `rkp_run_w_mcmc_cmds.sh` is the worked command sequence.
Reference PDFs: `Wind_Density.pdf` (profile equations), `stu2151.pdf`
(Laycock et al. 2015), `manuscript_1.pdf` (2017 MS thesis). `paper/` holds the
MDPI *Algorithms* manuscript skeleton and bibliography. Legacy R code in
`legacy_r_code/`. Notebooks under `notebooks/` predate Phases 31–33.

---

## Current Status & Quick Commands

**Environment:** `henv` conda env (heasoft/XSPEC + `numba`, `emcee`,
`zeus-mcmc`, `arviz`, `corner`, `astropy`, `scipy`). `numba` is a hard
requirement.

**Forward model:** three profiles (`smooth_pl` default with `Rb=5, p=4,
Delta=2`; `confinement`; `beta_law`), physical `Ṁ/v_inf` normalization with
`f_opacity`, per-cell flux conversion, ≈ 11 ms per light curve at `dth = 1`
(5.5 ms at the MCMC default `dth = 2`), one band per run. **MCMC:** `phys` / `--reparam` / `--kepler` / `--kepler-mtot`, `chi2` or
`jitter`, emcee or zeus, per-sample phase-shift alignment on, adaptive
constant-counts binning recommended, `--fit-fopacity` strongly recommended.

```bash
# XSPEC flux-vs-nH table, one band per file (needs XSPEC)
python compute_flux_vs_nH.py --specdir ./data/IC10X1_spec --model tbabs \
    --band broad --out_csv flux_vs_nH_tbabs_broad.csv

# One light curve. R is the photosphere; the eclipse comes from wind opacity.
python xrb_lightcurve.py --flux_csv flux_vs_nH_tbabs_broad.csv \
    --mdot 4e-6 --v-inf 1750 --f-opacity 0.03 \
    --r 0.001 --R 2.0 --d1 11 --d2 8 --i0 78 --dth 1 --d2h 6 \
    --wind-model beta_law --beta 0.8 --H 1.0 --output lc_broad.csv

# Single-model χ² fit with a fitted phase shift, smoothed overlay, model dump
python chandra_phase_analysis.py --data-dir data/IC_10_X1_LC_CIAO/broad/single \
    --obs-column flux_t --time-column t_raw --counts-per-bin 100 \
    --fit --sim-file lc_broad.csv --fit-phase-shift --smooth \
    --output fit_broad.png --write-model

# MCMC: total mass + wind shape + opacity + scattered floor (the recommended form)
python mcmc_lightcurve_fit.py --band broad \
    --flux-csv flux_vs_nH_tbabs_broad.csv \
    --data-dir data/IC_10_X1_LC_CIAO/broad/single/ \
    --obs-column flux_t --time-column t_raw --counts-per-bin 100 \
    --wind-model smooth_pl --kepler-mtot --fit-wind-shape --fit-scatter \
    --fit-fopacity --likelihood jitter --sampler zeus \
    --n-walkers 24 --n-steps 30000 --n-burn 3000 --n-threads 4 --dth 4.0 \
    --prior-Mtot 45,18,10,110 --prior-qm 0.60,0.12,0.05,0.95 \
    --compute-bic --smooth --output-dir mcmc_results/broad/smooth_pl/mtot

# Regenerate every output from a finished run; options come from the run config
python mcmc_lightcurve_fit.py --replot --output-dir mcmc_results/broad/smooth_pl/mtot
```

**Interpreting masses.** The light curve depends on `a = d1 + d2` only, so `q`
/ `q_m` are exactly unidentifiable, and scaling every length together with
`f_opacity` is an exact invariance. Quote the shape ratios (`R/a`, `r/a`,
`Rb/a`, `p`, `i0`), the opacity amplitude and `f_scatter` as measurements;
quote `q_m` as an assumption and `M_tot` only with its anchor (a stated prior
on `R` or a fixed `f_opacity`) made explicit.

**`--data-dir` resolves `{band}/single` before `{band}/`.** Pass the band
directory explicitly to choose between one observation and all of them.

---

**Last Updated:** September 19, 2026  
**Maintainer:** R. Panchal
