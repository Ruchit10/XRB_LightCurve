# XRB Lightcurve — wind-absorption modelling for eclipsing X-ray binaries

Forward model and inference stack for the X-ray light curve of an eclipsing
high-mass X-ray binary, developed for **IC 10 X-1**. The compact object orbits
inside the companion's stellar wind; the observed modulation is the combination
of a geometric eclipse and phase-dependent photoelectric absorption in that
wind.

Originally ported from R; the numerical core is a Numba-parallel
Gauss-Legendre quadrature over the line of sight. One full light curve
(360 phases, 60 angular sectors × 10 radial cells) takes ≈ 30 ms on a laptop,
which is what makes direct-evaluation MCMC practical. The model is run **one
energy band at a time**: each flux-vs-nH table holds a single band and each
simulation produces a single `nfl_{band}` column.

---

## What the model does

For each orbital phase the code

1. builds a polar grid across the projected emitter disk,
2. integrates the wind density along the line of sight from every visible
   grid cell (sectors mirrored about the star–star line share their geometry,
   so only half of them are integrated),
3. converts each cell's column density `N_H` to a band flux using an XSPEC
   `flux vs nH` table, and
4. area-averages the result.

Steps 3–4 run inside one compiled pass, **per cell, before averaging**. The
`N_H → flux` map is strongly nonlinear, so `⟨F(N)⟩ ≠ F(⟨N⟩)` wherever the
column varies steeply across the disk — during ingress/egress and throughout
the eclipse core, where the surviving flux is dominated by the least-absorbed
cells.

### Wind column normalization

The density normalization `n₀` is fixed **physically**, from the mass-loss rate
and terminal velocity, by matching the asymptotic `r⁻²` limit of the profile to
a spherical constant-velocity wind:

```
n₀ = Ṁ / (4π R_sun² v_inf μ m_H C)
```

so `N_H` carries real units. The eclipse therefore emerges from wind opacity
rather than from a geometric cutoff, and `R` means the true photospheric radius.

Because a WR wind is hyper-ionized, clumped, and He-rich rather than
solar-abundance, its *effective* photoelectric opacity is far below what its
mass column implies. The dimensionless factor `--f-opacity` absorbs that
difference; for IC 10 X-1 it lands around 0.01–0.03. In MCMC runs, fit it with
`--fit-fopacity` rather than guessing.

### Wind density profiles

Three dimensionless profiles `g(r)`, selected with `--wind-model`:

| Model | Parameters | Form |
|-------|-----------|------|
| `smooth_pl` (default) | `Rb`, `p`, `Delta` | smoothly broken power law: inner slope `p`, outer `r⁻²`, break at `Rb` with smoothness `Delta` |
| `confinement` | `R_star`, `fconf`, `ell` | `r⁻²` wind with an exponential inner overdensity of amplitude `fconf` and scale `ell` |
| `beta_law` | `R_star`, `beta`, `H` | velocity-based, `n = Ṁ / (4π r² v(r))` with `v = v_inf (1 − e^{−(r−R★)/H}) (1 − R★/r)^β`; dense acceleration zone inside `R★ + 3H`, `r⁻²` beyond |

For `confinement` and `beta_law`, `R_star` is tied to the geometric companion
radius `R`. `beta_law` is the profile that follows most directly from the
`n₀ = Ṁ/(4π R_sun² v_inf μ m_H)` normalization: `g = 1/(r² v̂)` with
`v̂ = v/v_inf → 1`, so `C = 1`.

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+. **Numba is required** — the mega-kernel is the only LOS
integrator and the only path that produces the per-cell columns the flux
conversion needs. `emcee` is required for MCMC; `zeus`, `arviz`, `corner` and
`tqdm` are optional.

---

## Pipeline

### 1. Flux vs nH table (upstream of everything)

Light curves are generated from column densities, so an XSPEC-derived
`flux vs nH` table is required first:

```bash
python compute_flux_vs_nH.py \
    --specdir ./data/IC10X1_spec --band broad \
    --out_csv flux_vs_nH_broad.csv \
    --out_png flux_vs_nH_broad.png \
    --nH_min 1e20 --nH_max 1e24 --nH_points 60
```

Requires XSPEC (PyXspec) in the environment. Each CSV carries `nH_1e22` plus
`flux_{band}_ph` / `flux_{band}_erg` for **one** band; make one table per band
you intend to fit. (A table holding several bands is still accepted, but the
simulator then needs `--band` / `band=` to pick one.)

### 2. Generate a model light curve

```bash
python xrb_lightcurve.py \
    --flux_csv flux_vs_nH_broad.csv \
    --wind-model smooth_pl \
    --R 2.0 --r 0.001 --d1 11.0 --d2 8.0 --i0 78.0 \
    --f-opacity 0.02 \
    --output sim_broad.csv
```

From Python, `simulate_lightcurve(...)` returns the per-phase DataFrame and
`simulate_band_flux(...)` just the `(phase, flux)` arrays the likelihood needs.

### 3. Fit

Single-model χ² fit against observed data:

```bash
python chandra_phase_analysis.py \
    --data-dir data/IC_10_X1_LC/Broad_with_flux/ \
    --sim-file sim_broad.csv --obs-column FLUX \
    --fit --fit-phase-shift --output fit_broad.png --write-model
```

Full posterior via MCMC:

```bash
python mcmc_lightcurve_fit.py \
    --band broad --flux-csv flux_vs_nH_broad.csv \
    --data-dir data/IC_10_X1_LC_CIAO/broad/single/ \
    --obs-column flux_t --time-column t_raw --n-phase-bins 150 \
    --wind-model smooth_pl --fit-wind-shape --fit-fopacity \
    --reparam --sampler zeus --likelihood jitter \
    --n-walkers 32 --n-steps 5000 --n-burn 500
```

See `rkp_run_w_mcmc_cmds.sh` for the full worked sequence, and
`python <script>.py --help` for every option.

Only the **phase shift** is fitted in the x-direction and only an *additive*
scattered-flux floor in the y-direction. There is deliberately no
multiplicative flux scale: the absolute normalization is already set by Ṁ and
the XSPEC table, so a free y-scale would silently absorb an error in that
normalization instead of exposing it.

---

## `xrb_lightcurve.py` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--r` | 0.001 | Radius of the compact object / accretion disk (R☉) |
| `--R` | 2.0 | Companion photospheric radius (R☉) |
| `--d1` | 11.0 | Compact-object distance from the COM (R☉) |
| `--d2` | 8.0 | Companion distance from the COM (R☉) |
| `--gma0` | -90.0 | Starting phase angle (degrees) |
| `--i0` | 64.0 | Inclination from the orbital-plane normal (90 = edge-on, 0 = face-on) |
| `--dth` | 1.0 | Orbital increment (degrees) |
| `--d2h` | 6.0 | Angular cell size of the polar grid (degrees) |
| `--flux_csv` | *required* | Flux vs nH CSV from `compute_flux_vs_nH.py` |
| `--band` | *auto* | Band to simulate; only needed if the CSV holds more than one |
| `--flux_method` | `interpolate` | `interpolate` or `refit` (see below) |
| `--flux_type` | `erg` | `erg` (erg/cm²/s) or `ph` (photons/cm²/s) |
| `--wind-model` | `smooth_pl` | `smooth_pl`, `confinement` or `beta_law` |
| `--Rb`, `--p`, `--Delta` | 5.0, 4.0, 2.0 | `smooth_pl` shape parameters (`Delta` matches the value the MCMC holds fixed) |
| `--fconf`, `--ell` | 10.0, 0.5 | `confinement` shape parameters |
| `--beta`, `--H` | 1.0, 1.0 | `beta_law` shape parameters (CAK exponent, acceleration scale height in R☉) |
| `--mdot` | 4e-6 | WR mass-loss rate (M☉/yr), Clark & Crowther (2004) |
| `--v-inf` | 1750.0 | Wind terminal velocity (km/s) |
| `--mu-wind` | 1.4 | Mean mass per hydrogen-equivalent nucleus |
| `--f-opacity` | 1.0 | Effective-opacity factor (see above) |
| `--output` | `xrb_lightcurve_output.csv` | Output CSV |

### Flux conversion methods

1. **`interpolate`** (default) — log-log interpolation of the XSPEC table.
   Most faithful to the spectral model.
2. **`refit`** — fits `A·exp(−B·nH)` to the same table (once per table, then
   cached) and uses the analytic form. Smoother, at the cost of a small
   systematic error where the true curve departs from a single exponential.

Both conversions are applied per emitter cell inside a compiled kernel.

---

## Output columns

| Column | Meaning |
|--------|---------|
| `deg`, `phase` | Phase angle in degrees, and normalized phase (0–1) |
| `l3`, `L3`, `h3` | Sky-plane separation and its in-plane / out-of-plane components (R☉) |
| `A2` | Visible emitter area (grid units; an integration diagnostic) |
| `is_eclipsed` | Per-phase geometric eclipse flag |
| `flx` | Dimensionless mean wind LOS integral ∫g(r)dz, r in R☉ |
| `fl` | Absolute mean column density `N_H` (10²² cm⁻²) |
| `nfl_{band}` | Absorbed band flux, area-averaged over the emitter disk |

`fl = flx × f_opacity × n₀ × R_sun / 10²²`, and `nfl_{band}` is the per-cell
flux conversion averaged over the disk (**not** the conversion of `fl`).

---

## Repository layout

| Path | Role |
|------|------|
| `xrb_lightcurve.py` | Forward model — generates model light curves |
| `compute_flux_vs_nH.py` | XSPEC flux vs nH table (upstream of the model) |
| `mcmc_lightcurve_fit.py` | Full MCMC posterior inference |
| `chandra_phase_analysis.py` | Single-model χ² fit, CLI front end |
| `plot_results.py` | Standalone plots from a simulation CSV |
| `utils/utils.py` | Data loading, phase binning, smoothing, χ² fit |
| `utils/plot_utils.py` | All plotting routines, shared by both fit scripts |
| `notebooks/` | Exploratory analysis |
| `legacy_r_code/` | Original R implementation, kept for reference |
| `changes_tracked.md` | Development history |

---

## Troubleshooting

**`ImportError: numba is required`** — install Numba; there is no pure-Python
fallback integrator.

**Columns outside the table's `nH` range** are handled silently: the column is
clipped to `[1e-6, 1e6] × 10²² cm⁻²` and the flux is extrapolated linearly in
log–log space from the table's end segments. If your fits reach such columns,
regenerate the table with a wider `--nH_min` / `--nH_max`.

**Model flux orders of magnitude too low** — `f_opacity` is probably at its
default of 1.0. The Clark & Crowther mass-loss rate overpredicts the observed
`N_H` for IC 10 X-1 by ~1.5–2 dex; use `--f-opacity 0.02` or fit it.

**Poor MCMC mixing in mass mode** — use `--kepler-mtot` rather than `--kepler`.
The light curve constrains only `M_tot`; `q_m` is exactly unidentifiable, and
sampling `(M_X, M_RH)` lays that flat direction diagonally across both axes.

## License

Provided as-is for educational and research purposes.
