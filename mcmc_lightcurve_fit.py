#!/usr/bin/env python3
"""
MCMC Light Curve Fitting for X-ray Binary Systems
--------------------------------------------------
This module performs Markov Chain Monte Carlo (MCMC) fitting to find optimal
binary system parameters by fitting model light curves to observed Chandra data.

WIND MODELS (selected via --wind-model):
- smooth_pl   : Smoothly broken power-law density profile (Rb, p, Delta)
- beta_law    : CAK beta-law velocity-based density (R_star, beta, H)
- confinement : Inner-confinement / compression amplification (R_star, fconf, ell)

GEOMETRY parameters (always fit):
- d1: Distance of compact object from center of mass (solar radii)
- d2: Distance of companion star from center of mass (solar radii)
- r:  Radius of compact object/accretion disk (solar radii)
- R:  Radius of companion star (solar radii)
- i0: Orbital inclination (degrees)

WIND-SHAPE parameters (added with --fit-wind-shape; the set depends on the
chosen --wind-model):
- smooth_pl   : Rb (break radius), p (inner slope). Delta is fixed at 2.
- beta_law    : beta (CAK exponent). H is fixed; R_star is tied to R.
- confinement : fconf (compression amplitude), ell (compression scale).
                R_star is tied to R.

MCMC fitting uses direct ``simulate_lightcurve`` evaluations (~60 ms per LC
with the Gauss-Legendre mega-kernel). This avoids interpolation artifacts from
precomputed grids and keeps the likelihood physically faithful for every sample.

Simulation parameters (passed to simulate_lightcurve):
- lam:  Target mean nH in 1e22 cm^-2 units (fixes overall normalization)
- gma0: Starting phase angle
- d2h:  Angular cell size for polar grid
- dz:   Step size along line of sight (legacy fallback path only)

Usage:
    # Geometry-only fit, smooth_pl wind, default shape params
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv \\
        --wind-model smooth_pl

    # Geometry + wind-shape fit (smooth_pl: + Rb, p)
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv \\
        --wind-model smooth_pl --fit-wind-shape

    # beta-law wind with shape fit (adds beta)
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv \\
        --wind-model beta_law --fit-wind-shape

    # Use zeus sampler / jitter likelihood
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv \\
        --wind-model smooth_pl --sampler zeus --likelihood jitter

    # CIAO flux_t workflow (raw 100s bins + jitter)
    python mcmc_lightcurve_fit.py --band soft --flux-csv data_flux_vs_nH.csv \\
        --obs-column flux_t --time-column t_raw --no-phase-bin --likelihood jitter

    # Custom shape-param prior (e.g. tighter Rb)
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv \\
        --wind-model smooth_pl --fit-wind-shape --prior-Rb 5.0,1.0,2.0,15.0
"""

import argparse
import csv
import copy
from dataclasses import dataclass, field
import glob
import multiprocessing as mp
import os
import time
import warnings
import pickle
from typing import Tuple, List, Dict, Optional
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import emcee
except ImportError:
    raise ImportError("emcee is required. Install with: pip install emcee")

try:
    import corner
except ImportError:
    warnings.warn("corner not installed. Corner plots will be disabled. Install with: pip install corner")
    corner = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    warnings.warn("tqdm not installed. Using basic progress output. Install with: pip install tqdm")
    HAS_TQDM = False

try:
    import zeus as zeus_sampler
    HAS_ZEUS = True
except ImportError:
    HAS_ZEUS = False

try:
    import arviz as az
    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False

# Silence noisy, repeated messages that would otherwise flood stdout during
# MCMC (one fire per model evaluation x thousands of evaluations).  The
# underlying causes are benign (extrapolation just outside the flux-vs-nH
# CSV grid, and a one-line "detected bands" print).  simulate_lightcurve is
# already called with verbose=False from the MCMC path, which suppresses
# both at the source; this filter is a belt-and-suspenders catch in case a
# new callsite is added without threading the flag through.
warnings.filterwarnings(
    "ignore",
    message=r"Some nH values are outside CSV range",
    category=UserWarning,
)

from xrb_lightcurve import (
    simulate_lightcurve,
    WIND_MODEL_PARAM_KEYS,
    default_wind_params,
)
from chandra_phase_analysis import (
    REF_EPOCH,
    ORBITAL_PERIOD,
    frac,
    load_data as _load_data_base,
    phase_bin_data as _phase_bin_data_base,
    phase_bin_data_snr as _phase_bin_data_snr_base,
)


def _init_numba_worker(max_numba_threads: int = 1):
    """Pool worker initializer: set Numba thread count inside each worker process.

    In pooled mode, each worker runs full ``simulate_lightcurve`` calls that
    already use ``parallel=True`` / ``prange`` in ``xrb_lightcurve``.

    - Too many threads per worker × many workers ⇒ severe oversubscription.
    - Too few (e.g. 1) per worker ⇒ each LC evaluation is much slower, so
      process-level parallelism barely beats serial ``32 × t_eval``.

    The driver picks ``max_numba_threads`` (see ``--numba-threads-per-worker``
    and its default ``auto`` ≈ ``cpu_count // pool_size``).
    """
    try:
        import numba
        numba.set_num_threads(max(1, int(max_numba_threads)))
    except Exception:
        # If numba is unavailable or thread control fails, proceed with defaults.
        pass

# Default priors based on IC 10 X-1 parameters
DEFAULT_PRIORS = {
    'd1': {'mean': 11.0, 'std': 3.0, 'min': 5.0, 'max': 20.0},      # Solar radii
    'd2': {'mean': 8.0, 'std': 3.0, 'min': 3.0, 'max': 15.0},       # Solar radii
    'r': {'mean': 0.001, 'std': 0.001, 'min': 0.0001, 'max': 0.1}, # Solar radii
    'R': {'mean': 2.0, 'std': 0.5, 'min': 1.0, 'max': 5.0},         # Solar radii
    'i0': {'mean': 26.0, 'std': 20.0, 'min': 10.0, 'max': 85.0},    # Degrees
}

# Parameter names for labeling
PARAM_NAMES = ['d1', 'd2', 'r', 'R', 'i0']
PARAM_LABELS = [
    r'$d_1$ (R$_\odot$)',
    r'$d_2$ (R$_\odot$)', 
    r'$r$ (R$_\odot$)',
    r'$R$ (R$_\odot$)',
    r'$i$ (deg)'
]

# Reparameterized priors: (d1, d2) -> (a = d1+d2, q = d1/(d1+d2))
# a is the orbital separation; q is a mass-ratio proxy bounded to (0, 1).
REPARAM_PRIORS = {
    'a':  {'mean': 19.0, 'std': 4.0,  'min': 8.0,    'max': 35.0},
    'q':  {'mean': 0.58, 'std': 0.15, 'min': 0.01,   'max': 0.99},
    'r':  {'mean': 0.001, 'std': 0.001, 'min': 0.0001, 'max': 0.1},
    'R':  {'mean': 2.0,  'std': 0.5,  'min': 1.0,    'max': 5.0},
    'i0': {'mean': 26.0, 'std': 20.0, 'min': 10.0,   'max': 85.0},
}

REPARAM_PARAM_NAMES = ['a', 'q', 'r', 'R', 'i0']
REPARAM_PARAM_LABELS = [
    r'$a$ (R$_\odot$)',
    r'$q$',
    r'$r$ (R$_\odot$)',
    r'$R$ (R$_\odot$)',
    r'$i$ (deg)'
]

KEPLER_PRIORS = {
    'M_X': {'mean': 30.0, 'std': 10.0, 'min': 1.0, 'max': 100.0},     # Solar masses
    'M_RH': {'mean': 20.0, 'std': 10.0, 'min': 1.0, 'max': 100.0},    # Solar masses
    'r': {'mean': 0.001, 'std': 0.001, 'min': 0.0001, 'max': 0.1},    # Solar radii
    'R': {'mean': 2.0, 'std': 0.5, 'min': 1.0, 'max': 5.0},           # Solar radii
    'i0': {'mean': 26.0, 'std': 20.0, 'min': 10.0, 'max': 85.0},      # Degrees
}

KEPLER_PARAM_NAMES = ['M_X', 'M_RH', 'r', 'R', 'i0']
KEPLER_PARAM_LABELS = [
    r'$M_X$ (M$_\odot$)',
    r'$M_\mathrm{RH}$ (M$_\odot$)',
    r'$r$ (R$_\odot$)',
    r'$R$ (R$_\odot$)',
    r'$i$ (deg)',
]

G_SI = 6.674e-11
R_SUN_M = 6.957e8
M_SUN_KG = 1.989e30

# Per-sample phase-shift optimization defaults.
# Coarse grid + local refinement gives near-fine-grid accuracy at lower cost.
DEFAULT_PHASE_SHIFT_GRID_SIZE = 25
DEFAULT_PHASE_SHIFT_EVAL_POINTS = 240
DEFAULT_PHASE_SHIFT_REFINE_POINTS = 9


@dataclass
class ParamSpec:
    mode: str = 'phys'  # 'phys' | 'reparam' | 'kepler'
    active_names: List[str] = field(default_factory=list)
    active_labels: List[str] = field(default_factory=list)
    frozen: Dict[str, float] = field(default_factory=dict)
    fit_wind_shape: bool = False
    wind_model: str = 'smooth_pl'
    likelihood: str = 'chi2'
    orbital_period_s: float = float(ORBITAL_PERIOD)
    K_kepler: float = 0.0


def _compute_kepler_prefactor(orbital_period_s: float) -> float:
    """Return factor K for a = K * (Mtot/Msun)^(1/3) in solar radii."""
    p = float(orbital_period_s)
    return ((G_SI * M_SUN_KG * p ** 2) / (4.0 * np.pi ** 2)) ** (1.0 / 3.0) / R_SUN_M


def parse_freeze_map(freeze_arg: Optional[str]) -> Dict[str, float]:
    """Parse --freeze NAME=VAL[,NAME=VAL,...] into a dict."""
    out: Dict[str, float] = {}
    if not freeze_arg:
        return out
    for chunk in str(freeze_arg).split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid --freeze entry '{item}'. Expected NAME=VALUE."
            )
        name, value = item.split("=", 1)
        key = name.strip()
        if not key:
            raise ValueError(f"Invalid --freeze entry '{item}': missing parameter name.")
        if key in out:
            raise ValueError(f"Duplicate frozen parameter '{key}' in --freeze.")
        out[key] = float(value)
    return out


def get_mode_name_label(mode: str) -> Tuple[List[str], List[str]]:
    if mode == 'reparam':
        return list(REPARAM_PARAM_NAMES), list(REPARAM_PARAM_LABELS)
    if mode == 'kepler':
        return list(KEPLER_PARAM_NAMES), list(KEPLER_PARAM_LABELS)
    return list(PARAM_NAMES), list(PARAM_LABELS)


def build_param_spec(
    likelihood: str = 'chi2',
    reparam: bool = False,
    kepler: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    frozen: Optional[Dict[str, float]] = None,
    orbital_period_s: float = ORBITAL_PERIOD,
) -> ParamSpec:
    """Build canonical active-parameter layout for this run."""
    if reparam and kepler:
        raise ValueError("--reparam and --kepler are mutually exclusive.")
    mode = 'kepler' if kepler else ('reparam' if reparam else 'phys')
    frozen = dict(frozen or {})

    names, labels = get_mode_name_label(mode)
    if likelihood == 'jitter':
        names.append('log_f')
        labels.append(r'$\ln\,f$')
    if fit_wind_shape:
        if wind_model not in WIND_SHAPE_FIT:
            raise ValueError(
                f"--fit-wind-shape is not supported for wind_model "
                f"'{wind_model}'. Choose one of: {list(WIND_SHAPE_FIT)}"
            )
        for sname in WIND_SHAPE_FIT[wind_model]:
            names.append(sname)
            labels.append(WIND_SHAPE_LABELS.get(sname, sname))

    valid_frozen = set(names)
    # Allow freezing shape parameters even when fit_wind_shape is off.
    valid_frozen.update(WIND_SHAPE_FIT.get(wind_model, []))

    if 'log_f' in frozen:
        raise ValueError("Freezing log_f is not supported. Use --likelihood chi2/jitter.")

    unknown = [k for k in frozen if k not in valid_frozen]
    if unknown:
        allowed = sorted(valid_frozen)
        raise ValueError(
            f"Unknown frozen parameter(s): {unknown}. "
            f"Allowed names for this run: {allowed}"
        )

    active_names: List[str] = []
    active_labels: List[str] = []
    for n, l in zip(names, labels):
        if n not in frozen:
            active_names.append(n)
            active_labels.append(l)

    if ('R' in frozen) and ('Rb' in frozen) and (frozen['Rb'] < frozen['R']):
        raise ValueError(
            "Invalid freeze combination: require Rb >= R when both are frozen."
        )

    return ParamSpec(
        mode=mode,
        active_names=active_names,
        active_labels=active_labels,
        frozen=frozen,
        fit_wind_shape=fit_wind_shape,
        wind_model=wind_model,
        likelihood=likelihood,
        orbital_period_s=float(orbital_period_s),
        K_kepler=_compute_kepler_prefactor(orbital_period_s),
    )


# Wind model descriptions (matches xrb_lightcurve.WIND_MODEL_IDS keys,
# excluding broken_pl which is a special case of smooth_pl).
WIND_MODELS = {
    'smooth_pl':   'Smoothly Broken Power-Law Wind',
    'beta_law':    'CAK Beta-Law (Velocity-Based) Wind',
    'confinement': 'Inner-Confinement / Compression Wind',
}

# Per-model wind-shape parameters that become free MCMC dimensions when the
# user passes --fit-wind-shape. The remaining keys in WIND_SHAPE_FIXED are
# always passed to simulate_lightcurve as constants. R_star (for beta_law /
# confinement) is *tied* to the geometry parameter R and is therefore not
# listed here; it is filled in by _to_wind_params().
WIND_SHAPE_FIT = {
    'smooth_pl':   ['Rb', 'p'],
    'beta_law':    ['beta'],
    'confinement': ['fconf', 'ell'],
}

# Fixed shape parameters that are passed inside wind_params but are NOT
# fitted (poor identifiability or strong degeneracy).
WIND_SHAPE_FIXED = {
    'smooth_pl':   {'Delta': 2.0},
    'beta_law':    {'H': 1.0},
    'confinement': {},
}

# Pretty labels for corner plots / diagnostics.
WIND_SHAPE_LABELS = {
    'Rb':    r'$R_b$ (R$_\odot$)',
    'p':     r'$p$',
    'beta':  r'$\beta$',
    'fconf': r'$f_\mathrm{conf}$',
    'ell':   r'$\ell$ (R$_\odot$)',
}

# Default priors for wind-shape parameters (mean, std, min, max).
# Box bounds are kept generous; the Gaussian acts as a weak preference toward
# physically motivated values. Override with --prior-<name> on the CLI.
WIND_SHAPE_PRIORS = {
    'Rb':    {'mean': 5.0, 'std': 3.0,  'min': 0.5, 'max': 30.0},
    'p':     {'mean': 4.0, 'std': 1.0,  'min': 2.0, 'max': 8.0},
    'beta':  {'mean': 0.8, 'std': 0.3,  'min': 0.3, 'max': 2.0},
    'fconf': {'mean': 5.0, 'std': 5.0,  'min': 0.0, 'max': 50.0},
    'ell':   {'mean': 1.0, 'std': 0.7,  'min': 0.1, 'max': 10.0},
}

# Likelihood configuration
LIKELIHOOD_TYPES = {
    'chi2': 'Chi-squared (Gaussian)',
    'jitter': 'Gaussian with systematic jitter',
}

JITTER_PRIOR = {'mean': -3.0, 'std': 2.0, 'min': -10.0, 'max': 0.0}

SAMPLER_TYPES = {
    'emcee': 'emcee Ensemble Sampler (stretch moves)',
    'zeus': 'zeus Ensemble Slice Sampler',
}


def get_param_config(
    likelihood: str = 'chi2',
    reparam: bool = False,
    kepler: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    frozen: Optional[Dict[str, float]] = None,
    orbital_period_s: float = ORBITAL_PERIOD,
):
    """Return (param_names, param_labels) for the active MCMC vector.

    Layout:
        geometry (5) -> [log_f if jitter] -> wind-shape params (if requested)

    When *reparam* is True the first two geometric parameters are ``(a, q)``
    instead of ``(d1, d2)``.

    When *fit_wind_shape* is True the model-specific shape params from
    ``WIND_SHAPE_FIT[wind_model]`` are appended (in their listed order).
    """
    spec = build_param_spec(
        likelihood=likelihood,
        reparam=reparam,
        kepler=kepler,
        wind_model=wind_model,
        fit_wind_shape=fit_wind_shape,
        frozen=frozen,
        orbital_period_s=orbital_period_s,
    )
    return list(spec.active_names), list(spec.active_labels)


def get_active_priors(
    base_priors: Dict,
    wind_model: str,
    fit_wind_shape: bool,
    likelihood: str,
    shape_prior_overrides: Dict[str, Dict] = None,
    frozen: Optional[Dict[str, float]] = None,
) -> Dict:
    """Build the merged prior dict covering geometry + jitter + shape params.

    *base_priors* is the geometry priors dict (DEFAULT_PRIORS or REPARAM_PRIORS,
    possibly modified by CLI overrides). *shape_prior_overrides* is an
    optional dict of {shape_name: {mean, std, min, max}} to override the
    defaults in WIND_SHAPE_PRIORS.
    """
    out = dict(base_priors)
    if likelihood == 'jitter':
        out.setdefault('log_f', dict(JITTER_PRIOR))
    if fit_wind_shape:
        for name in WIND_SHAPE_FIT.get(wind_model, []):
            prior = dict(WIND_SHAPE_PRIORS[name])
            if shape_prior_overrides and name in shape_prior_overrides:
                prior.update(shape_prior_overrides[name])
            out[name] = prior
    if frozen:
        for name in frozen:
            out.pop(name, None)
    return out


def _to_wind_params(
    theta: np.ndarray,
    active_names: List[str],
    wind_model: str,
    R_value: float,
    fit_wind_shape: bool = False,
    frozen: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Build the wind_params dict for simulate_lightcurve from a sample.

    Pulls fittable shape values from *theta* using their position in
    *active_names*; fills in fixed shape values from WIND_SHAPE_FIXED; and
    ties R_star to the geometry R for the beta_law / confinement models.
    """
    wp: Dict[str, float] = dict(WIND_SHAPE_FIXED.get(wind_model, {}))

    frozen = frozen or {}
    for name in WIND_SHAPE_FIT.get(wind_model, []):
        if name in frozen:
            wp[name] = float(frozen[name])
            continue
        if fit_wind_shape:
            try:
                idx = active_names.index(name)
            except ValueError:
                wp[name] = float(WIND_SHAPE_PRIORS[name]['mean'])
                continue
            wp[name] = float(theta[idx])
        else:
            wp[name] = float(WIND_SHAPE_PRIORS[name]['mean'])

    if wind_model in ('beta_law', 'confinement'):
        wp['R_star'] = float(R_value)

    return wp


# =============================================================================
# Data Loading and Phase Binning
# =============================================================================

def _resolve_band_directory(band: str, data_dir: str) -> str:
    """Resolve the directory containing light curve files for *band*.

    Tries several common layouts in order:
      1. data_dir itself contains .txt files              (direct path)
      2. data_dir/{Band}_with_flux/                       (old converted layout)
      3. data_dir/{band}/single/                          (CIAO single-obs layout)
      4. data_dir/{band}/                                 (CIAO band folder)
    """
    candidates = [
        data_dir,
        os.path.join(data_dir, f"{band.capitalize()}_with_flux"),
        os.path.join(data_dir, band.lower(), "single"),
        os.path.join(data_dir, band.lower()),
    ]
    for path in candidates:
        if os.path.isdir(path) and glob.glob(os.path.join(path, "*.txt")):
            return path

    tried = "\n  ".join(candidates)
    raise FileNotFoundError(
        f"No .txt light-curve files found for band '{band}'. "
        f"Searched:\n  {tried}"
    )


def load_observed_lightcurves(
    band: str,
    data_dir: str = "data/IC_10_X1_LC",
    flux_column: str = "FLUX",
    error_column: Optional[str] = None,
    time_column: str = None
) -> pd.DataFrame:
    """Load all observed light curve files for a given energy band.

    Delegates file reading to :func:`chandra_phase_analysis.load_data`
    and remaps the output columns to the MCMC convention
    (``flux`` / ``flux_err`` / ``obs_id``).

    Parameters
    ----------
    band : str
        Energy band: 'broad', 'soft', 'medium', or 'hard'
    data_dir : str
        Base directory (or direct path) containing light curve data.
        The function tries several sub-directory conventions automatically.
    flux_column : str
        Column name for flux values in the data files
    error_column : str, optional
        Column name for flux errors in the data files. If None, auto-detect.
    time_column : str, optional
        Column name for timestamps (e.g. 'TIME', 't_raw'). Auto-detected if None.

    Returns
    -------
    DataFrame with columns: time, flux, flux_err, phase, obs_id, counts
    """
    band_dir = _resolve_band_directory(band, data_dir)
    print(f"Loading {band} band data from: {band_dir}")

    raw = _load_data_base(
        band_dir,
        obs_column=flux_column,
        obs_error_column=error_column,
        time_column=time_column,
        counts_column='counts',
    )

    combined = pd.DataFrame({
        'time': raw['time'].astype(float),
        'flux': raw['rate'].astype(float),
        'flux_err': raw['error'].astype(float) if 'error' in raw.columns else np.nan,
        'obs_id': raw['obs'] if 'obs' in raw.columns else 'data',
        'counts': raw['counts'].astype(float) if 'counts' in raw.columns else np.nan,
    })

    if 'phase' in raw.columns:
        combined['phase'] = raw['phase'].astype(float)
    else:
        combined['phase'] = frac((combined['time'] - REF_EPOCH) / ORBITAL_PERIOD)

    n_before = len(combined)
    valid = (combined['flux'] > 0) & np.isfinite(combined['flux']) & np.isfinite(combined['time'])
    combined = combined.loc[valid].reset_index(drop=True)
    n_dropped = n_before - len(combined)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} zero/non-finite flux rows before fitting")

    n_files = len(glob.glob(os.path.join(band_dir, "*.txt")))
    print(f"Loaded {len(combined)} data points from {n_files} file(s) for {band} band")

    return combined


def phase_bin_data(
    df: pd.DataFrame,
    n_bins: int = 50,
    min_points_per_bin: int = 3
) -> pd.DataFrame:
    """
    Bin observed data into orbital phase bins.
    
    This is a wrapper around chandra_phase_analysis.phase_bin_data that
    handles the column naming convention used in MCMC fitting (flux/flux_err
    instead of rate/error).
    
    Parameters
    ----------
    df : DataFrame
        Observed data with columns: phase, flux, flux_err
    n_bins : int
        Number of phase bins (default 50)
    min_points_per_bin : int
        Minimum number of data points required per bin
        
    Returns
    -------
    DataFrame with columns: phase, flux, flux_err, n_points
    """
    # Rename columns to match chandra_phase_analysis convention
    df_renamed = df.copy()
    if 'flux' in df_renamed.columns:
        df_renamed['rate'] = df_renamed['flux']
    if 'flux_err' in df_renamed.columns:
        df_renamed['error'] = df_renamed['flux_err']
    
    # Add dummy 'obs' column if not present (required by the base function)
    if 'obs' not in df_renamed.columns:
        df_renamed['obs'] = 'data'
    
    # Call the base function from chandra_phase_analysis
    result = _phase_bin_data_base(
        df_renamed,
        n_bins=n_bins,
        min_points_per_bin=min_points_per_bin,
        rate_column='rate',
        error_column='error',
        verbose=True
    )
    
    # Rename columns back to MCMC convention
    result = result.rename(columns={'rate': 'flux', 'error': 'flux_err'})
    
    return result


def phase_bin_data_snr(
    df: pd.DataFrame,
    counts_per_bin: int = 100,
    counts_column: str = 'counts',
) -> pd.DataFrame:
    """Adaptive phase binning wrapper for constant-counts bins."""
    df_renamed = df.copy()
    if 'flux' in df_renamed.columns:
        df_renamed['rate'] = df_renamed['flux']
    if 'flux_err' in df_renamed.columns:
        df_renamed['error'] = df_renamed['flux_err']
    if 'obs' not in df_renamed.columns:
        df_renamed['obs'] = 'data'

    result = _phase_bin_data_snr_base(
        df_renamed,
        counts_per_bin=counts_per_bin,
        counts_column=counts_column,
        rate_column='rate',
        error_column='error',
        verbose=True,
    )
    return result.rename(columns={'rate': 'flux', 'error': 'flux_err'})


def _interp_periodic_phases(
    obs_phases: np.ndarray,
    model_phase: np.ndarray,
    model_flux: np.ndarray,
) -> np.ndarray:
    """Interpolate periodic model flux to observation phases.

    Uses a monotonic fast path and falls back to sorting when needed.
    """
    if model_phase.size == 0:
        return np.full_like(obs_phases, np.nan, dtype=float)

    if np.all(np.diff(model_phase) >= 0):
        phase_sorted = model_phase
        flux_sorted = model_flux
    else:
        sort_idx = np.argsort(model_phase)
        phase_sorted = model_phase[sort_idx]
        flux_sorted = model_flux[sort_idx]

    phase_ext = np.concatenate([phase_sorted - 1.0, phase_sorted, phase_sorted + 1.0])
    flux_ext = np.concatenate([flux_sorted, flux_sorted, flux_sorted])
    return np.interp(obs_phases, phase_ext, flux_ext)


def _build_phase_shift_terms(
    enabled: bool,
    obs_phase: np.ndarray,
    *,
    grid_size: int = DEFAULT_PHASE_SHIFT_GRID_SIZE,
    eval_points: int = DEFAULT_PHASE_SHIFT_EVAL_POINTS,
    refine_points: int = DEFAULT_PHASE_SHIFT_REFINE_POINTS,
) -> Dict[str, object]:
    """Precompute reusable arrays for per-sample phase-shift optimization.

    The search runs in two stages:
      1) coarse uniform grid over phase shifts in [0, 1)
      2) local refinement around the best coarse shift
    """
    if not enabled:
        return {"enabled": False}
    n_grid = max(3, int(grid_size))
    n_eval = max(16, int(eval_points))
    n_refine = max(0, int(refine_points))
    shift_grid = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    phase_eval_grid = np.linspace(0.0, 1.0, n_eval, endpoint=False)
    shifted_obs_phase = np.mod(obs_phase[None, :] - shift_grid[:, None], 1.0)
    return {
        "enabled": True,
        "shift_grid": shift_grid,
        "phase_eval_grid": phase_eval_grid,
        "shifted_obs_phase": shifted_obs_phase,
        "refine_points": n_refine,
    }


def _apply_best_phase_shift(
    model_phase: np.ndarray,
    model_flux: np.ndarray,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err2: np.ndarray,
    phase_shift_terms: Optional[Dict[str, object]],
) -> Tuple[Optional[np.ndarray], float]:
    """Align model to observations by minimizing weighted chi-square over shift."""
    if not phase_shift_terms or not bool(phase_shift_terms.get("enabled", False)):
        return model_flux, 0.0

    shift_grid = np.asarray(phase_shift_terms.get("shift_grid", []), dtype=float)
    shifted_obs_phase = phase_shift_terms.get("shifted_obs_phase")
    if shift_grid.size == 0:
        return model_flux, 0.0

    if shifted_obs_phase is None or np.shape(shifted_obs_phase) != (shift_grid.size, obs_phase.size):
        shifted_obs_phase = np.mod(obs_phase[None, :] - shift_grid[:, None], 1.0)

    best_model = None
    best_idx = -1
    best_shift = 0.0
    best_chi2 = np.inf

    for i, shift in enumerate(shift_grid):
        shifted_phase = shifted_obs_phase[i]
        shifted_flux = _interp_periodic_phases(shifted_phase, model_phase, model_flux)
        if np.any(~np.isfinite(shifted_flux)):
            continue
        chi2 = np.sum((obs_flux - shifted_flux) ** 2 / obs_err2)
        if chi2 < best_chi2:
            best_chi2 = float(chi2)
            best_model = shifted_flux
            best_idx = i
            best_shift = float(shift)

    if best_model is None:
        return None, 0.0

    # Local refinement around the best coarse shift to recover sub-grid accuracy
    # without paying for a globally dense shift grid.
    n_refine = int(phase_shift_terms.get("refine_points", 0) or 0)
    if n_refine > 1 and shift_grid.size >= 3 and best_idx >= 0:
        coarse_step = 1.0 / float(shift_grid.size)
        fine_shifts = np.linspace(
            best_shift - coarse_step,
            best_shift + coarse_step,
            n_refine,
            endpoint=True,
        )
        for shift in np.mod(fine_shifts, 1.0):
            shifted_phase = np.mod(obs_phase - shift, 1.0)
            shifted_flux = _interp_periodic_phases(shifted_phase, model_phase, model_flux)
            if np.any(~np.isfinite(shifted_flux)):
                continue
            chi2 = np.sum((obs_flux - shifted_flux) ** 2 / obs_err2)
            if chi2 < best_chi2:
                best_chi2 = float(chi2)
                best_model = shifted_flux
                best_shift = float(shift)

    return best_model, best_shift




# =============================================================================
# Direct Model (SLOW - for comparison/debugging)
# =============================================================================

class DirectLightCurveModel:
    """
    Direct light curve model evaluation (calls simulate_lightcurve each time).

    With the Gauss-Legendre mega-kernel in xrb_lightcurve.py a single LC is
    ~60 ms, so this class is now suitable for MCMC. It is also the only path
    that supports per-step varying wind shape parameters (--fit-wind-shape).
    """

    def __init__(
        self,
        band: str,
        flux_csv_path: str,
        wind_model: str = 'smooth_pl',
        dth: float = 5.0,
        flux_method: str = "interpolate",
        sim_params: Dict = None,
        wind_params_default: Dict[str, float] = None,
    ):
        self.band = band.lower()
        self.flux_csv_path = flux_csv_path
        self.wind_model = wind_model.lower()
        self.dth = dth
        self.flux_method = flux_method
        self.flux_column = f"nfl_{self.band}"
        self.sim_params = sim_params or {}

        if self.wind_model not in WIND_MODELS:
            raise ValueError(
                f"wind_model must be one of {list(WIND_MODELS)}, got '{wind_model}'"
            )

        if wind_params_default is None:
            wp = default_wind_params(self.wind_model, R=2.0)
            wp.pop('R_star', None)
            wind_params_default = wp
        self.wind_params_default = dict(wind_params_default)

        if not os.path.exists(flux_csv_path):
            raise FileNotFoundError(f"Flux CSV not found: {flux_csv_path}")

    def evaluate(
        self,
        d1: float,
        d2: float,
        r: float,
        R: float,
        i0: float,
        obs_phases: np.ndarray,
        wind_params: Dict[str, float] = None,
    ) -> np.ndarray:
        """Evaluate model by running simulate_lightcurve.

        ``wind_params`` overrides the default fixed shape parameters when
        provided. R_star is auto-filled from R for beta_law / confinement
        if not present.
        """
        if wind_params is None:
            wp = dict(self.wind_params_default)
        else:
            wp = dict(wind_params)
        if self.wind_model in ('beta_law', 'confinement') and 'R_star' not in wp:
            wp['R_star'] = float(R)

        try:
            results = simulate_lightcurve(
                r=r, R=R, d1=d1, d2=d2,
                gma0=self.sim_params.get('gma0', -90.0),
                i0=i0,
                dth=self.dth,
                d2h=self.sim_params.get('d2h', 6.0),
                dz=self.sim_params.get('dz', 0.5),
                flux_method=self.flux_method,
                flux_csv_path=self.flux_csv_path,
                lam=self.sim_params.get('lam', 0.589537),
                wind_model=self.wind_model,
                wind_params=wp,
                verbose=False,
            )
        except Exception as e:
            warnings.warn(f"Model evaluation failed: {e}")
            return np.full_like(obs_phases, np.nan)

        if self.flux_column not in results.columns:
            return np.full_like(obs_phases, np.nan)

        model_phase = results['phase'].values
        model_flux = results[self.flux_column].values
        return _interp_periodic_phases(obs_phases, model_phase, model_flux)


# =============================================================================
# Likelihood Functions
# =============================================================================

def _theta_value(
    theta: np.ndarray,
    name: str,
    active_names: Optional[List[str]] = None,
    frozen: Optional[Dict[str, float]] = None,
):
    frozen = frozen or {}
    if name in frozen:
        return float(frozen[name])
    if active_names is not None and name in active_names:
        return float(theta[active_names.index(name)])
    return None


def _resolve_geom(
    theta: np.ndarray,
    reparam: bool = False,
    active_names: Optional[List[str]] = None,
    param_spec: Optional[ParamSpec] = None,
) -> Tuple[float, float, float, float, float]:
    """Resolve geometry from theta for phys/reparam/kepler modes."""
    if param_spec is not None:
        mode = param_spec.mode
        names = param_spec.active_names
        frozen = param_spec.frozen
    else:
        mode = 'reparam' if reparam else 'phys'
        names = active_names
        frozen = {}

    if mode == 'phys':
        d1 = _theta_value(theta, 'd1', names, frozen)
        d2 = _theta_value(theta, 'd2', names, frozen)
    elif mode == 'reparam':
        a = _theta_value(theta, 'a', names, frozen)
        q = _theta_value(theta, 'q', names, frozen)
        d1 = a * q
        d2 = a * (1.0 - q)
    elif mode == 'kepler':
        mx = _theta_value(theta, 'M_X', names, frozen)
        mrh = _theta_value(theta, 'M_RH', names, frozen)
        mtot = mx + mrh
        if mtot <= 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        a = float(param_spec.K_kepler) * mtot ** (1.0 / 3.0)
        q = mrh / mtot
        d1 = a * q
        d2 = a * (1.0 - q)
    else:
        raise ValueError(f"Unknown parameter mode '{mode}'")

    r = _theta_value(theta, 'r', names, frozen)
    R = _theta_value(theta, 'R', names, frozen)
    i0 = _theta_value(theta, 'i0', names, frozen)
    return float(d1), float(d2), float(r), float(R), float(i0)


def _to_physical(theta, reparam: bool = False):
    """Backward-compatible wrapper for legacy callers."""
    return _resolve_geom(theta, reparam=reparam)


def _resolve_shape(
    theta: np.ndarray,
    R_value: float,
    active_names: Optional[List[str]],
    wind_model: str,
    fit_wind_shape: bool,
    frozen: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, float]]:
    if (not fit_wind_shape) and not any(
        name in (frozen or {}) for name in WIND_SHAPE_FIT.get(wind_model, [])
    ):
        return None
    return _to_wind_params(
        theta=theta,
        active_names=(active_names or []),
        wind_model=wind_model,
        R_value=R_value,
        fit_wind_shape=fit_wind_shape,
        frozen=frozen,
    )


def _evaluate_model(
    theta,
    model,
    obs_phase,
    reparam: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    active_names: List[str] = None,
    param_spec: Optional[ParamSpec] = None,
):
    """Evaluate the physical model. Returns model_flux or None on failure.

    When *fit_wind_shape* is True, the wind-shape parameters are pulled out
    of *theta* (using their position in *active_names*) and passed inside
    *wind_params* to ``model.evaluate``. When False, the model uses its
    constructor-time defaults.
    """
    if param_spec is not None:
        active_names = param_spec.active_names
        wind_model = param_spec.wind_model
        fit_wind_shape = param_spec.fit_wind_shape
    d1, d2, r, R, i0 = _resolve_geom(
        theta, reparam=reparam, active_names=active_names, param_spec=param_spec
    )
    if not np.all(np.isfinite([d1, d2, r, R, i0])):
        return None

    wind_params = _resolve_shape(
        theta=theta,
        R_value=R,
        active_names=active_names,
        wind_model=wind_model,
        fit_wind_shape=fit_wind_shape,
        frozen=(param_spec.frozen if param_spec is not None else None),
    )

    try:
        model_flux = model.evaluate(
            d1, d2, r, R, i0, obs_phase, wind_params=wind_params,
        )
    except TypeError:
        # Backward compat with any model.evaluate() that doesn't accept
        # wind_params yet.
        try:
            model_flux = model.evaluate(d1, d2, r, R, i0, obs_phase)
        except Exception:
            return None
    except Exception:
        return None
    if np.any(~np.isfinite(model_flux)):
        return None
    return model_flux


def log_likelihood_chi2(
    theta: np.ndarray,
    model,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    reparam: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    active_names: List[str] = None,
    obs_err2: np.ndarray = None,
    phase_shift_terms: Optional[Dict[str, object]] = None,
    param_spec: Optional[ParamSpec] = None,
) -> float:
    """Standard Gaussian log-likelihood (chi-squared)."""
    eval_phases = (
        np.asarray(phase_shift_terms["phase_eval_grid"], dtype=float)
        if (phase_shift_terms and phase_shift_terms.get("enabled", False))
        else obs_phase
    )
    model_flux = _evaluate_model(
        theta, model, eval_phases, reparam=reparam,
        wind_model=wind_model, fit_wind_shape=fit_wind_shape,
        active_names=active_names,
        param_spec=param_spec,
    )
    if model_flux is None:
        return -np.inf
    if obs_err2 is None:
        obs_err2 = obs_err ** 2
    if phase_shift_terms and phase_shift_terms.get("enabled", False):
        model_flux, _ = _apply_best_phase_shift(
            eval_phases, model_flux, obs_phase, obs_flux, obs_err2, phase_shift_terms
        )
        if model_flux is None:
            return -np.inf
    chi2 = np.sum((obs_flux - model_flux) ** 2 / obs_err2)
    return -0.5 * chi2


def log_likelihood_jitter(
    theta: np.ndarray,
    model,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    reparam: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    active_names: List[str] = None,
    obs_err2: np.ndarray = None,
    jitter_logf_index: Optional[int] = None,
    phase_shift_terms: Optional[Dict[str, object]] = None,
    param_spec: Optional[ParamSpec] = None,
) -> float:
    """Gaussian log-likelihood with a free fractional systematic error term.

    The position of the ``log_f`` parameter is looked up from *active_names*.
    The effective variance per point is  sigma_obs^2 + (f * model)^2
    where f = exp(log_f).
    """
    eval_phases = (
        np.asarray(phase_shift_terms["phase_eval_grid"], dtype=float)
        if (phase_shift_terms and phase_shift_terms.get("enabled", False))
        else obs_phase
    )
    model_flux = _evaluate_model(
        theta, model, eval_phases, reparam=reparam,
        wind_model=wind_model, fit_wind_shape=fit_wind_shape,
        active_names=active_names,
        param_spec=param_spec,
    )
    if model_flux is None:
        return -np.inf
    if jitter_logf_index is not None:
        idx_logf = int(jitter_logf_index)
    elif active_names is not None and 'log_f' in active_names:
        idx_logf = active_names.index('log_f')
    else:
        idx_logf = 5  # legacy default
    f = np.exp(theta[idx_logf])
    if obs_err2 is None:
        obs_err2 = obs_err ** 2
    if phase_shift_terms and phase_shift_terms.get("enabled", False):
        model_flux, _ = _apply_best_phase_shift(
            eval_phases, model_flux, obs_phase, obs_flux, obs_err2, phase_shift_terms
        )
        if model_flux is None:
            return -np.inf
    sigma2 = obs_err2 + (f * model_flux) ** 2
    return -0.5 * np.sum((obs_flux - model_flux) ** 2 / sigma2 + np.log(sigma2))


# Keep the old name as an alias so existing imports keep working
log_likelihood = log_likelihood_chi2


# =============================================================================
# Prior & Posterior
# =============================================================================

def log_prior(
    theta: np.ndarray,
    priors: Dict = DEFAULT_PRIORS,
    likelihood: str = 'chi2',
    reparam: bool = False,
    active_names: List[str] = None,
    param_spec: Optional[ParamSpec] = None,
) -> float:
    """Log prior probability for geometry + jitter + wind-shape parameters.

    *priors* must contain entries for every name in *active_names* (the
    helper :func:`get_active_priors` builds such a merged dict). When
    *active_names* is None this falls back to legacy geometry-only
    behavior for backward compatibility.

    When *reparam* is True, theta[:5] = (a, q, r, R, i0) and *priors*
    must contain keys ``'a'`` and ``'q'``. A Jacobian correction
    ``log(a)`` is added to account for the change of variables
    ``d(d1) d(d2) = a · d(a) d(q)``.
    """
    if param_spec is None:
        mode = 'reparam' if reparam else 'phys'
        pnames, _ = get_mode_name_label(mode)
        frozen = {}
        if active_names is None:
            active_names = list(pnames)
            if likelihood == 'jitter':
                active_names.append('log_f')
    else:
        mode = param_spec.mode
        pnames, _ = get_mode_name_label(mode)
        active_names = list(param_spec.active_names)
        frozen = dict(param_spec.frozen)

    # Box check on every active parameter (geometry + jitter + shape).
    for name, value in zip(active_names, theta):
        prior = priors.get(name)
        if prior is None:
            # No prior provided for this dim; treat as improper / skip box.
            continue
        if not (prior['min'] < value < prior['max']):
            return -np.inf

    _, _, r_value, R_value, _ = _resolve_geom(
        theta, reparam=reparam, active_names=active_names, param_spec=param_spec
    )
    # Physical constraint r < R.
    if not np.isfinite(r_value) or not np.isfinite(R_value) or (r_value >= R_value):
        return -np.inf

    log_p = 0.0

    for i, name in enumerate(active_names):
        prior = priors.get(name)
        if prior is None:
            continue
        log_p += -0.5 * ((theta[i] - prior['mean']) / prior['std']) ** 2

    # Jacobian |d(d1,d2)/d(a,q)| = a in reparameterized mode.
    if mode == 'reparam':
        a_value = _theta_value(theta, 'a', active_names, frozen)
        if (a_value is None) or (a_value <= 0):
            return -np.inf
        log_p += np.log(a_value)

    # Physically sensible smooth_pl break radius.
    if (param_spec is not None) and (param_spec.wind_model == 'smooth_pl'):
        rb_val = _theta_value(theta, 'Rb', active_names, frozen)
        if rb_val is not None and rb_val < R_value:
            return -np.inf

    if mode == 'kepler':
        mx = _theta_value(theta, 'M_X', active_names, frozen)
        mrh = _theta_value(theta, 'M_RH', active_names, frozen)
        if (mx is None) or (mrh is None) or ((mx + mrh) <= 0):
            return -np.inf

    return log_p


def log_probability(
    theta: np.ndarray,
    model,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    priors: Dict = DEFAULT_PRIORS,
    likelihood: str = 'chi2',
    reparam: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    active_names: List[str] = None,
    like_terms: Dict[str, object] = None,
    param_spec: Optional[ParamSpec] = None,
) -> float:
    """Log posterior = log prior + log likelihood."""
    lp = log_prior(
        theta, priors, likelihood=likelihood, reparam=reparam,
        active_names=active_names,
        param_spec=param_spec,
    )
    if not np.isfinite(lp):
        return -np.inf

    common_kwargs = dict(
        reparam=reparam,
        wind_model=wind_model,
        fit_wind_shape=fit_wind_shape,
        active_names=active_names,
        param_spec=param_spec,
    )
    like_terms = like_terms or {}
    obs_err2 = like_terms.get('obs_err2')
    jitter_logf_index = like_terms.get('jitter_logf_index')
    phase_shift_terms = like_terms.get('phase_shift_terms')
    if likelihood == 'jitter':
        ll = log_likelihood_jitter(
            theta, model, obs_phase, obs_flux, obs_err, **common_kwargs,
            obs_err2=obs_err2,
            jitter_logf_index=jitter_logf_index,
            phase_shift_terms=phase_shift_terms,
        )
    else:
        ll = log_likelihood_chi2(
            theta, model, obs_phase, obs_flux, obs_err, **common_kwargs,
            obs_err2=obs_err2,
            phase_shift_terms=phase_shift_terms,
        )

    return lp + ll


# =============================================================================
# MCMC Sampler (emcee / zeus)
# =============================================================================

def run_mcmc(
    model,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    n_walkers: int = 32,
    n_steps: int = 5000,
    n_burn: int = 1000,
    priors: Dict = DEFAULT_PRIORS,
    progress: bool = True,
    n_threads: int = 1,
    sampler_type: str = 'emcee',
    likelihood: str = 'chi2',
    reparam: bool = False,
    kepler: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    numba_threads_per_worker: Optional[int] = None,
    frozen: Optional[Dict[str, float]] = None,
    orbital_period_s: float = ORBITAL_PERIOD,
    fit_phase_shift: bool = True,
    phase_shift_grid_size: int = DEFAULT_PHASE_SHIFT_GRID_SIZE,
    phase_shift_eval_points: int = DEFAULT_PHASE_SHIFT_EVAL_POINTS,
    param_spec: Optional[ParamSpec] = None,
) -> Tuple:
    """
    Run MCMC sampling with emcee or zeus.
    
    Parameters
    ----------
    model : DirectLightCurveModel
        Model evaluator
    obs_phase, obs_flux, obs_err : np.ndarray
        Observed data
    n_walkers : int
        Number of MCMC walkers
    n_steps : int
        Number of MCMC steps
    n_burn : int
        Number of burn-in steps to discard
    priors : dict
        Prior specifications
    progress : bool
        Show progress bar
    n_threads : int
        Number of worker processes for parallel MCMC (1 = serial).
    numba_threads_per_worker : int or None
        Numba threads inside each worker when ``n_threads > 1`` and the model is
        ``DirectLightCurveModel``. ``None`` means auto:
        ``max(1, cpu_count // n_threads)``.
    sampler_type : str
        'emcee' or 'zeus'
    likelihood : str
        'chi2' or 'jitter'
    reparam : bool
        If True, sample in (a, q) space instead of (d1, d2).
        
    Returns
    -------
    sampler : emcee.EnsembleSampler or zeus.EnsembleSampler
    samples : np.ndarray
        Flattened chain samples (after burn-in)
    active_param_names : list of str
    active_param_labels : list of str
    """
    if param_spec is None:
        param_spec = build_param_spec(
            likelihood=likelihood,
            reparam=reparam,
            kepler=kepler,
            wind_model=wind_model,
            fit_wind_shape=fit_wind_shape,
            frozen=frozen,
            orbital_period_s=orbital_period_s,
        )
    active_names = list(param_spec.active_names)
    active_labels = list(param_spec.active_labels)
    reparam = (param_spec.mode == 'reparam')
    kepler = (param_spec.mode == 'kepler')
    wind_model = param_spec.wind_model
    fit_wind_shape = param_spec.fit_wind_shape
    likelihood = param_spec.likelihood
    n_dim = len(active_names)

    # Build initial positions and per-dim scatter from the (already merged)
    # priors dict, which must contain entries for every active dim.
    initial = np.empty(n_dim)
    scatter = np.empty(n_dim)
    for i, name in enumerate(active_names):
        prior = priors[name]
        initial[i] = prior['mean']
        scatter[i] = prior['std'] * 0.1

    pos = initial + scatter * np.random.randn(n_walkers, n_dim)

    # Clip walkers inside the box for every dim.
    for i, name in enumerate(active_names):
        prior = priors[name]
        pos[:, i] = np.clip(
            pos[:, i],
            prior['min'] + 0.01 * abs(prior['min']) + 1e-12,
            prior['max'] - 0.01 * abs(prior['max']) - 1e-12,
        )

    # Enforce r < R only when both are free dimensions.
    idx_r = active_names.index('r') if 'r' in active_names else None
    idx_R = active_names.index('R') if 'R' in active_names else None
    if (idx_r is not None) and (idx_R is not None):
        for j in range(n_walkers):
            if pos[j, idx_r] >= pos[j, idx_R]:
                pos[j, idx_r] = pos[j, idx_R] * 0.1

    parallel_info = f", {n_threads} threads" if n_threads > 1 else " (serial)"
    print(f"\nStarting MCMC ({sampler_type}) with {n_walkers} walkers, "
          f"{n_steps} steps{parallel_info}")
    if reparam:
        print("Reparameterization: (d1, d2) -> (a = d1+d2, q = d1/a)")
    elif kepler:
        print("Kepler mode: sample (M_X, M_RH); derive a and q from period.")
    print(f"Likelihood: {LIKELIHOOD_TYPES[likelihood]}")
    print(f"Wind model: {WIND_MODELS.get(wind_model, wind_model)} "
          f"({wind_model})  | fit_wind_shape={fit_wind_shape}")
    if fit_phase_shift:
        print(
            f"Per-sample phase-shift search: enabled "
            f"(grid={phase_shift_grid_size}, eval_points={phase_shift_eval_points})"
        )
    else:
        print("Per-sample phase-shift search: disabled")
    print(f"Active params ({n_dim}): {active_names}")
    print(f"Initial parameter values (first walker): {pos[0]}")

    phase_shift_terms = _build_phase_shift_terms(
        fit_phase_shift,
        obs_phase,
        grid_size=phase_shift_grid_size,
        eval_points=phase_shift_eval_points,
    )

    log_prob_args = (
        # Precomputed invariants to reduce per-call overhead in likelihood eval.
        # Keep formulas unchanged; only move repeated allocations/lookups out of
        # the hot loop.
        # `jitter_logf_index` only used for jitter likelihood.
        #
        # Values are passed as a dict to keep backward-compatible call ordering.
        #
        # NOTE: not including constants dropped in existing formulas.
        #
        # fmt: off
        model, obs_phase, obs_flux, obs_err,
        priors, likelihood, reparam,
        wind_model, fit_wind_shape, active_names,
        {
            'obs_err2': obs_err ** 2,
            'jitter_logf_index': (
                active_names.index('log_f')
                if ('log_f' in active_names) else None
            ),
            'phase_shift_terms': phase_shift_terms,
        },
        param_spec,
        # fmt: on
    )

    start_time = time.time()

    using_direct_model = isinstance(model, DirectLightCurveModel)
    using_pool = n_threads > 1 and using_direct_model

    if n_threads > 1 and not using_direct_model:
        print(
            f"\n[notice] --n-threads={n_threads} requested with model type "
            f"{type(model).__name__}. Pooling is only enabled for "
            f"DirectLightCurveModel; running serial."
        )
        n_threads = 1
        using_pool = False

    if using_pool:
        cpus = int(cpu_count() or 1)
        if numba_threads_per_worker is None:
            ntb = max(1, cpus // int(n_threads))
        else:
            ntb = max(1, int(numba_threads_per_worker))
        print(
            f"[info] Pooled MCMC: {n_threads} worker processes, "
            f"numba.set_num_threads({ntb}) per worker "
            f"(logical CPUs ≈ {cpus}; auto is cpus//workers)."
        )
        # 'spawn' is safest cross-platform and avoids inheriting heavy state.
        mp_ctx = mp.get_context("spawn")
        pool = mp_ctx.Pool(
            processes=n_threads,
            initializer=_init_numba_worker,
            initargs=(ntb,),
        )
    else:
        pool = None

    try:
        if sampler_type == 'zeus':
            if not HAS_ZEUS:
                raise ImportError("zeus not installed. Install with: pip install zeus-mcmc")
            sampler = zeus_sampler.EnsembleSampler(
                n_walkers, n_dim, log_probability, args=log_prob_args, pool=pool
            )
            sampler.run_mcmc(pos, n_steps, progress=progress)
        else:
            sampler = emcee.EnsembleSampler(
                n_walkers, n_dim, log_probability,
                args=log_prob_args, pool=pool,
            )
            if progress and HAS_TQDM:
                for _ in tqdm(sampler.sample(pos, iterations=n_steps),
                              total=n_steps, desc="MCMC Sampling"):
                    pass
            else:
                sampler.run_mcmc(pos, n_steps, progress=progress)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    elapsed = time.time() - start_time

    samples = sampler.get_chain(discard=n_burn, flat=True)

    print(f"\nMCMC completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Time per step: {elapsed/n_steps*1000:.1f} ms")
    print(f"Final chain shape: {samples.shape}")

    return sampler, samples, active_names, active_labels


# =============================================================================
# Output and Diagnostics
# =============================================================================

def compute_statistics(
    samples: np.ndarray,
    param_names: List[str] = None,
    reparam: bool = False,
    kepler: bool = False,
    orbital_period_s: float = ORBITAL_PERIOD,
    param_spec: Optional[ParamSpec] = None,
    log_prob: Optional[np.ndarray] = None,
) -> Dict:
    """Compute summary statistics from MCMC samples.

    When *reparam* is True, derived ``d1`` and ``d2`` statistics are
    appended by transforming each sample: ``d1 = a*q``, ``d2 = a*(1-q)``.

    Per-parameter ``median`` / ``mean`` / ``std`` are *marginal* summaries.
    Note that medians of nonlinear combinations are not the combinations of
    medians, e.g. ``median(a*q) != median(a) * median(q)``, so the ``median``
    rows for ``a, q, d1, d2`` will not algebraically satisfy
    ``d1 + d2 == a`` or ``d1 / (d1 + d2) == q`` in general (they will be
    close only when posteriors are roughly symmetric and uncorrelated).

    If *log_prob* is provided, a single self-consistent point estimate (MAP,
    i.e. the sample with the highest log-probability) is added under the
    ``'map'`` key of every parameter's stats dict. The MAP point *does*
    satisfy ``d1 + d2 == a`` and ``d1 / (d1 + d2) == q`` exactly, since it
    is a single sample.
    """
    if param_names is None:
        param_names = PARAM_NAMES
    if param_spec is not None:
        mode = param_spec.mode
        orbital_period_s = param_spec.orbital_period_s
    else:
        mode = 'kepler' if kepler else ('reparam' if reparam else 'phys')
    stats = {}

    for i, param in enumerate(param_names):
        param_samples = samples[:, i]
        p16, p50, p84 = np.percentile(param_samples, [16, 50, 84])
        stats[param] = {
            'median': p50,
            'lower': p50 - p16,
            'upper': p84 - p50,
            'mean': np.mean(param_samples),
            'std': np.std(param_samples)
        }

    if mode == 'reparam':
        frozen = (param_spec.frozen if param_spec is not None else {})
        if 'a' in param_names:
            a_samples = samples[:, param_names.index('a')]
        elif 'a' in frozen:
            a_samples = np.full(len(samples), float(frozen['a']))
        else:
            raise ValueError("Could not resolve 'a' for reparameterized statistics.")
        if 'q' in param_names:
            q_samples = samples[:, param_names.index('q')]
        elif 'q' in frozen:
            q_samples = np.full(len(samples), float(frozen['q']))
        else:
            raise ValueError("Could not resolve 'q' for reparameterized statistics.")
        derived = [
            ('d1', a_samples * q_samples),
            ('d2', a_samples * (1.0 - q_samples)),
        ]
    elif mode == 'kepler':
        mx_samples = samples[:, param_names.index('M_X')]
        mrh_samples = samples[:, param_names.index('M_RH')]
        mtot = mx_samples + mrh_samples
        K = _compute_kepler_prefactor(orbital_period_s)
        a_samples = K * np.power(mtot, 1.0 / 3.0)
        q_samples = mrh_samples / mtot
        derived = [
            ('a', a_samples),
            ('q', q_samples),
            ('d1', a_samples * q_samples),
            ('d2', a_samples * (1.0 - q_samples)),
        ]
    else:
        derived = []

    for derived_name, derived_vals in derived:
            p16, p50, p84 = np.percentile(derived_vals, [16, 50, 84])
            stats[derived_name] = {
                'median': p50,
                'lower': p50 - p16,
                'upper': p84 - p50,
                'mean': np.mean(derived_vals),
                'std': np.std(derived_vals),
                'derived': True,
            }

    if log_prob is not None and len(log_prob) == len(samples):
        finite = np.isfinite(log_prob)
        if finite.any():
            map_idx = int(np.argmax(np.where(finite, log_prob, -np.inf)))
            map_sample = samples[map_idx]
            for i, param in enumerate(param_names):
                stats[param]['map'] = float(map_sample[i])
            if mode == 'reparam':
                frozen = (param_spec.frozen if param_spec is not None else {})
                if 'a' in param_names:
                    a_map = float(map_sample[param_names.index('a')])
                else:
                    a_map = float(frozen['a'])
                if 'q' in param_names:
                    q_map = float(map_sample[param_names.index('q')])
                else:
                    q_map = float(frozen['q'])
                stats['d1']['map'] = a_map * q_map
                stats['d2']['map'] = a_map * (1.0 - q_map)
            elif mode == 'kepler':
                mx_map = float(map_sample[param_names.index('M_X')])
                mrh_map = float(map_sample[param_names.index('M_RH')])
                mtot_map = mx_map + mrh_map
                a_map = _compute_kepler_prefactor(orbital_period_s) * (mtot_map ** (1.0 / 3.0))
                q_map = mrh_map / mtot_map
                stats['a']['map'] = a_map
                stats['q']['map'] = q_map
                stats['d1']['map'] = a_map * q_map
                stats['d2']['map'] = a_map * (1.0 - q_map)
            stats['_map_meta'] = {
                'index': map_idx,
                'log_prob': float(log_prob[map_idx]),
            }

    return stats


def save_samples_csv_chunked(
    samples: np.ndarray,
    param_names: List[str],
    output_path: str,
    log_prob: Optional[np.ndarray] = None,
    chunk_size: int = 50000,
):
    """Write sample table to CSV in chunks to limit peak memory."""
    headers = list(param_names) + (["log_prob"] if log_prob is not None else [])
    with open(output_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(headers)
        n_rows = len(samples)
        for start in range(0, n_rows, int(chunk_size)):
            stop = min(start + int(chunk_size), n_rows)
            block = samples[start:stop]
            if log_prob is not None:
                lp_block = log_prob[start:stop]
                for row, lp in zip(block, lp_block):
                    writer.writerow(list(np.asarray(row, dtype=float)) + [float(lp)])
            else:
                for row in block:
                    writer.writerow(list(np.asarray(row, dtype=float)))


def load_existing_results(
    output_dir: str,
    band: str,
    wind_model: str,
    reparam: bool = False,
    kepler: bool = False,
    orbital_period_s: float = ORBITAL_PERIOD,
    param_spec: Optional[ParamSpec] = None,
) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[List[str]]]:
    """
    Load existing MCMC results from saved files.

    Returns
    -------
    samples : np.ndarray or None
    stats : Dict or None
    loaded_names : list[str] or None
        Column names used (in order) for the returned samples array.
    """
    suffix = f"{band}_{wind_model}"
    samples_path = os.path.join(output_dir, f"{suffix}_samples.csv")

    if not os.path.exists(samples_path):
        print(f"Samples file not found: {samples_path}")
        return None, None, None

    print(f"Loading existing samples from: {samples_path}")
    samples_df = pd.read_csv(samples_path)

    geom_names = REPARAM_PARAM_NAMES if reparam else PARAM_NAMES
    missing_geom = [p for p in geom_names if p not in samples_df.columns]
    if missing_geom:
        print(f"Error: Samples file missing required geometry columns: {missing_geom}")
        return None, None, None

    # Use whatever columns are present, preserving the file order, and
    # excluding bookkeeping columns like log_prob.
    loaded_names = [c for c in samples_df.columns if c != 'log_prob']
    samples = samples_df[loaded_names].values
    log_prob_loaded = (
        samples_df['log_prob'].values if 'log_prob' in samples_df.columns else None
    )
    stats = compute_statistics(
        samples,
        param_names=loaded_names,
        reparam=reparam,
        kepler=kepler,
        orbital_period_s=orbital_period_s,
        param_spec=param_spec,
        log_prob=log_prob_loaded,
    )

    print(f"  Loaded {len(samples)} samples; columns: {loaded_names}")

    return samples, stats, loaded_names


def compute_chi2_for_samples(
    model,
    samples: np.ndarray,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    output_path: str,
    n_samples: int = None,
    verbose: bool = True,
    reparam: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    active_names: List[str] = None,
    likelihood: str = 'chi2',
    fit_phase_shift: bool = True,
    phase_shift_grid_size: int = DEFAULT_PHASE_SHIFT_GRID_SIZE,
    phase_shift_eval_points: int = DEFAULT_PHASE_SHIFT_EVAL_POINTS,
    param_spec: Optional[ParamSpec] = None,
) -> None:
    """
    Compute chi-square for all (or a subset of) MCMC samples and save to compressed file.
    
    Parameters
    ----------
    model : DirectLightCurveModel
        Model evaluator
    samples : np.ndarray
        MCMC samples array (n_samples, n_params)
    obs_phase, obs_flux, obs_err : np.ndarray
        Observed data
    output_path : str
        Path to save the output file (will be gzip compressed if ends with .gz)
    n_samples : int, optional
        Number of samples to compute chi-square for. If None, use all samples.
    verbose : bool
        Print progress messages
    reparam : bool
        If True, samples are in (a, q, r, R, i0) space.
    """
    import gzip
    
    n_total = len(samples)
    if n_samples is None or n_samples > n_total:
        n_samples = n_total
        sample_indices = np.arange(n_total)
    else:
        sample_indices = np.random.choice(n_total, size=n_samples, replace=False)
        sample_indices = np.sort(sample_indices)
    
    if verbose:
        print(f"Computing chi-square for {n_samples} samples...")

    # Number of free physical parameters = active dims minus log_f (if any).
    if active_names is None:
        n_phys = 5
    else:
        n_phys = len(active_names) - (1 if 'log_f' in active_names else 0)
    dof = len(obs_flux) - n_phys

    results = []
    if param_spec is not None:
        active_names = list(param_spec.active_names)
        wind_model = param_spec.wind_model
        fit_wind_shape = param_spec.fit_wind_shape
        likelihood = param_spec.likelihood
    frozen = (param_spec.frozen if param_spec is not None else None)
    use_jitter = (likelihood == 'jitter' and active_names is not None and 'log_f' in active_names)

    if HAS_TQDM and verbose:
        iterator = tqdm(sample_indices, desc="Computing χ²")
    else:
        iterator = sample_indices

    phase_shift_terms = _build_phase_shift_terms(
        fit_phase_shift,
        obs_phase,
        grid_size=phase_shift_grid_size,
        eval_points=phase_shift_eval_points,
    )
    phase_eval_grid = (
        np.asarray(phase_shift_terms["phase_eval_grid"], dtype=float)
        if phase_shift_terms.get("enabled", False)
        else obs_phase
    )
    obs_err2 = obs_err ** 2

    for idx in iterator:
        sample_params = samples[idx]
        d1, d2, r, R, i0 = _resolve_geom(
            sample_params, reparam=reparam, active_names=active_names, param_spec=param_spec
        )
        wp = _resolve_shape(
            theta=sample_params,
            R_value=R,
            active_names=active_names,
            wind_model=wind_model,
            fit_wind_shape=fit_wind_shape,
            frozen=frozen,
        )

        try:
            try:
                model_flux = model.evaluate(
                    d1, d2, r, R, i0, phase_eval_grid, wind_params=wp,
                )
            except TypeError:
                model_flux = model.evaluate(d1, d2, r, R, i0, phase_eval_grid)

            if np.all(np.isfinite(model_flux)):
                if phase_shift_terms.get("enabled", False):
                    model_flux, _ = _apply_best_phase_shift(
                        phase_eval_grid,
                        model_flux,
                        obs_phase,
                        obs_flux,
                        obs_err2,
                        phase_shift_terms,
                    )
                    if model_flux is None:
                        raise ValueError("phase-shift optimization failed")
                # Always compute classical chi2 on measurement errors for
                # comparability across likelihood choices.
                chi2 = np.sum(((obs_flux - model_flux) / obs_err) ** 2)
                red_chi2 = chi2 / dof if dof > 0 else np.nan

                # For jitter runs, also compute the effective-variance version
                # used in the likelihood; this can be much smaller than 1.
                chi2_eff = np.nan
                red_chi2_eff = np.nan
                if use_jitter:
                    idx_logf = active_names.index('log_f')
                    f = np.exp(sample_params[idx_logf])
                    sigma2 = obs_err2 + (f * model_flux) ** 2
                    sigma2 = np.maximum(sigma2, np.finfo(float).eps)
                    chi2_eff = np.sum((obs_flux - model_flux) ** 2 / sigma2)
                    red_chi2_eff = chi2_eff / dof if dof > 0 else np.nan
            else:
                chi2 = np.nan
                red_chi2 = np.nan
                chi2_eff = np.nan
                red_chi2_eff = np.nan
        except Exception:
            chi2 = np.nan
            red_chi2 = np.nan
            chi2_eff = np.nan
            red_chi2_eff = np.nan

        if use_jitter:
            results.append([idx, d1, d2, r, R, i0, chi2, red_chi2, chi2_eff, red_chi2_eff])
        else:
            results.append([idx, d1, d2, r, R, i0, chi2, red_chi2])

    columns = ['sample_idx', 'd1', 'd2', 'r', 'R', 'i0', 'chi2', 'reduced_chi2']
    if use_jitter:
        columns += ['chi2_eff', 'reduced_chi2_eff']
    results_df = pd.DataFrame(results, columns=columns)
    
    # Save to file (compressed if .gz extension)
    if output_path.endswith('.gz'):
        results_df.to_csv(output_path, index=False, compression='gzip')
    else:
        results_df.to_csv(output_path, index=False)
    
    # Compute summary statistics
    valid_chi2 = results_df['reduced_chi2'].dropna()
    if len(valid_chi2) > 0:
        chi2_median = np.median(valid_chi2)
        chi2_min = np.min(valid_chi2)
        chi2_max = np.max(valid_chi2)
        chi2_std = np.std(valid_chi2)
        
        if verbose:
            print(f"Chi-square statistics (reduced):")
            print(f"  Median: {chi2_median:.3f}")
            print(f"  Min:    {chi2_min:.3f}")
            print(f"  Max:    {chi2_max:.3f}")
            print(f"  Std:    {chi2_std:.3f}")
    
    file_size = os.path.getsize(output_path) / 1024  # KB
    if verbose:
        print(f"Chi-square data saved to: {output_path} ({file_size:.1f} KB)")


def print_results(stats: Dict, band: str, wind_model: str, param_names: List[str] = None,
                   reparam: bool = False):
    """Print formatted results table."""
    if param_names is None:
        param_names = PARAM_NAMES
    print(f"\n{'='*60}")
    print(f"MCMC Results for {band.upper()} band - {WIND_MODELS[wind_model]}")
    print('='*60)
    print(f"{'Parameter':<15} {'Median':<12} {'Lower σ':<12} {'Upper σ':<12}")
    print('-'*60)

    for param in param_names:
        s = stats[param]
        print(f"{param:<15} {s['median']:<12.6f} {s['lower']:<12.6f} {s['upper']:<12.6f}")

    if reparam or ('d1' in stats and 'd2' in stats):
        print('-'*60)
        print("Derived physical parameters:")
        for derived in ('d1', 'd2'):
            if derived in stats:
                s = stats[derived]
                print(f"{derived:<15} {s['median']:<12.6f} {s['lower']:<12.6f} {s['upper']:<12.6f}")

    print('='*60)


def plot_corner(samples: np.ndarray, band: str, wind_model: str, output_path: str,
                param_labels: List[str] = None):
    """Generate corner plot of posterior distributions."""
    if corner is None:
        warnings.warn("corner package not installed, skipping corner plot")
        return
    if param_labels is None:
        param_labels = PARAM_LABELS

    fig = corner.corner(
        samples,
        labels=param_labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        title_fmt=".4f"
    )

    fig.suptitle(f"Posterior Distributions - {band.upper()} band ({wind_model.upper()})",
                 fontsize=14, y=1.02)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Corner plot saved to: {output_path}")


def plot_trace(sampler, band: str, wind_model: str, output_path: str,
               param_labels: List[str] = None, n_burn: int = None):
    """Generate trace plots for convergence diagnostics."""
    chain = sampler.get_chain()
    n_steps, n_walkers, n_dim = chain.shape
    if param_labels is None:
        param_labels = PARAM_LABELS
    if n_burn is None:
        n_burn = n_steps // 5

    fig, axes = plt.subplots(n_dim, 1, figsize=(10, 2*n_dim), sharex=True)
    if n_dim == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        for j in range(n_walkers):
            ax.plot(chain[:, j, i], alpha=0.3, lw=0.5)
        ax.set_ylabel(param_labels[i] if i < len(param_labels) else f"param {i}")
        ax.axvline(x=n_burn, color='r', linestyle='--',
                   alpha=0.5, label='Burn-in' if i == 0 else None)

    axes[-1].set_xlabel("Step")
    axes[0].legend(loc='upper right')
    fig.suptitle(f"MCMC Trace Plots - {band.upper()} band ({wind_model.upper()})", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Trace plot saved to: {output_path}")


def plot_best_fit(
    model,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    obs_phase_width: Optional[np.ndarray],
    stats: Dict,
    band: str,
    wind_model: str,
    output_path: str,
    param_names: List[str] = None,
    reparam: bool = False,
    fit_wind_shape: bool = False,
    is_binned: bool = True,
    likelihood: str = 'chi2',
    fit_phase_shift: bool = True,
    phase_shift_grid_size: int = DEFAULT_PHASE_SHIFT_GRID_SIZE,
    phase_shift_eval_points: int = DEFAULT_PHASE_SHIFT_EVAL_POINTS,
    param_spec: Optional[ParamSpec] = None,
):
    """
    Plot observed data with best-fit model overlay.

    Parameters
    ----------
    model : DirectLightCurveModel
        Model evaluator
    obs_phase, obs_flux, obs_err : np.ndarray
        Observed data
    stats : Dict
        Statistics from MCMC samples
    band : str
        Energy band name
    wind_model : str
        Wind model name
    output_path : str
        Path to save the plot
    reparam : bool
        If True, stats contains (a, q) and derived (d1, d2) keys.
    """
    if param_names is None:
        param_names = PARAM_NAMES

    # Prefer the algebraically self-consistent MAP point (single sample with
    # the highest log-prob) when available; otherwise fall back to per-param
    # medians. Note that median(d1) + median(d2) != median(a) in general
    # (medians of nonlinear combinations are not the combinations of medians),
    # so using medians here would produce a curve whose displayed parameters
    # don't satisfy d1 + d2 = a or d1/(d1+d2) = q. The MAP point does.
    use_map = all(
        ('map' in stats[p]) for p in (PARAM_NAMES if not reparam else
                                       ['r', 'R', 'i0', 'd1', 'd2'])
        if p in stats
    )
    point_key = 'map' if use_map else 'median'

    def _value_from_stats_or_frozen(name: str) -> float:
        if name in stats and point_key in stats[name]:
            return float(stats[name][point_key])
        if param_spec is not None and name in param_spec.frozen:
            return float(param_spec.frozen[name])
        raise KeyError(
            f"Missing '{name}' in statistics for best-fit plotting. "
            f"If this parameter is frozen, pass param_spec with frozen values."
        )

    if reparam:
        best_d1 = _value_from_stats_or_frozen('d1')
        best_d2 = _value_from_stats_or_frozen('d2')
        best_params = [
            best_d1,
            best_d2,
            _value_from_stats_or_frozen('r'),
            _value_from_stats_or_frozen('R'),
            _value_from_stats_or_frozen('i0'),
        ]
    else:
        best_params = [_value_from_stats_or_frozen(p) for p in PARAM_NAMES]

    # Build best-fit wind_params if shape was fitted.
    best_R = best_params[3]
    if fit_wind_shape and wind_model in WIND_SHAPE_FIT:
        best_wp = dict(WIND_SHAPE_FIXED.get(wind_model, {}))
        for sname in WIND_SHAPE_FIT[wind_model]:
            if sname in stats:
                best_wp[sname] = float(stats[sname][point_key])
            elif param_spec is not None and sname in param_spec.frozen:
                best_wp[sname] = float(param_spec.frozen[sname])
        if wind_model in ('beta_law', 'confinement'):
            best_wp['R_star'] = float(best_R)
    else:
        best_wp = None

    eval_fn = getattr(model, 'evaluate_direct', model.evaluate)

    model_phases = np.linspace(0, 1, 360)
    try:
        model_flux = eval_fn(*best_params, model_phases, wind_params=best_wp)
    except TypeError:
        model_flux = eval_fn(*best_params, model_phases)

    phase_shift_terms = _build_phase_shift_terms(
        fit_phase_shift,
        obs_phase,
        grid_size=phase_shift_grid_size,
        eval_points=phase_shift_eval_points,
    )
    best_phase_shift = 0.0
    if phase_shift_terms.get("enabled", False):
        phase_eval_grid = np.asarray(phase_shift_terms["phase_eval_grid"], dtype=float)
        try:
            model_eval = eval_fn(*best_params, phase_eval_grid, wind_params=best_wp)
        except TypeError:
            model_eval = eval_fn(*best_params, phase_eval_grid)
        obs_model, best_phase_shift = _apply_best_phase_shift(
            phase_eval_grid,
            model_eval,
            obs_phase,
            obs_flux,
            obs_err ** 2,
            phase_shift_terms,
        )
        if obs_model is None:
            obs_model = np.full_like(obs_phase, np.nan)
            best_phase_shift = 0.0
    else:
        try:
            obs_model = eval_fn(*best_params, obs_phase, wind_params=best_wp)
        except TypeError:
            obs_model = eval_fn(*best_params, obs_phase)
    f_best = None
    chi2_obs = np.sum(((obs_flux - obs_model) / obs_err) ** 2)
    chi2_eff = np.nan
    if likelihood == 'jitter' and 'log_f' in stats and point_key in stats['log_f']:
        f_best = float(np.exp(stats['log_f'][point_key]))
        sigma2 = obs_err ** 2 + (f_best * obs_model) ** 2
        sigma2 = np.maximum(sigma2, np.finfo(float).eps)
        chi2_eff = np.sum((obs_flux - obs_model) ** 2 / sigma2)
    chi2 = chi2_obs
    n_phys = len(param_names) - (1 if 'log_f' in param_names else 0)
    dof = len(obs_flux) - n_phys
    red_chi2 = chi2 / dof if dof > 0 else np.nan
    red_chi2_eff = chi2_eff / dof if (dof > 0 and np.isfinite(chi2_eff)) else np.nan

    fig, ax = plt.subplots(figsize=(10, 6))

    if is_binned:
        xerr = None
        if obs_phase_width is not None:
            width_arr = np.asarray(obs_phase_width, dtype=float)
            if width_arr.shape == obs_phase.shape:
                xerr = 0.5 * np.clip(width_arr, 0.0, np.inf)
        ax.errorbar(
            obs_phase, obs_flux, yerr=obs_err, xerr=xerr, fmt='o',
            markersize=4, alpha=0.7, label='Observed (phase-binned)',
            capsize=2, elinewidth=1, color='C0', zorder=5
        )
    else:
        ax.scatter(
            obs_phase, obs_flux, s=10, alpha=0.25, color='C0',
            label='Observed (raw 100s)', zorder=4
        )

    if fit_phase_shift:
        phase_overlay = np.mod(model_phases + best_phase_shift, 1.0)
        order = np.argsort(phase_overlay)
        ax.plot(
            phase_overlay[order], model_flux[order], 'r-', lw=2,
            label=f'Best-fit model ({WIND_MODELS[wind_model]})',
            zorder=10,
        )
    else:
        ax.plot(model_phases, model_flux, 'r-', lw=2,
                label=f'Best-fit model ({WIND_MODELS[wind_model]})', zorder=10)

    ax.set_xlabel('Orbital Phase', fontsize=12)
    ax.set_ylabel('Flux (erg/cm²/s)', fontsize=12)
    if np.isfinite(red_chi2) and red_chi2 < 1e-2:
        red_chi2_label = f"{red_chi2:.2e}"
    else:
        red_chi2_label = f"{red_chi2:.3f}"
    ax.set_title(
        f'{band.upper()} Band - {wind_model.upper()} Wind '
        f'(χ²/dof = {red_chi2_label})',
        fontsize=14
    )
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    # Show all sampled parameters plus derived d1/d2 when reparameterized.
    # Display the point estimate that was actually used to evaluate the
    # overlay (MAP if available, else median) and tag the median's symmetric
    # 1-sigma uncertainty for context.
    display_params = list(param_names)
    if reparam:
        display_params += ['d1', 'd2']
    point_label = 'MAP' if point_key == 'map' else 'median'
    rows = [f"point estimate: {point_label}"]
    for p in display_params:
        if p not in stats:
            continue
        s = stats[p]
        sigma = (s['lower'] + s['upper']) / 2
        rows.append(f"{p}: {s[point_key]:.4f}  (median +/- {sigma:.4f})")
    if fit_phase_shift:
        rows.append(f"phase_shift: {best_phase_shift:.4f}")
    if f_best is not None:
        rows.append(f"f: {f_best:.4f}  (from log_f)")
        if np.isfinite(red_chi2_eff):
            rows.append(f"reduced_chi2_eff: {red_chi2_eff:.3e}")
    param_text = '\n'.join(rows)
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Best-fit plot saved to: {output_path}")

    return red_chi2


def print_diagnostics(sampler, sampler_type: str = 'emcee',
                      param_names: List[str] = None):
    """Print MCMC diagnostics (works for emcee and zeus)."""
    diag: Dict[str, object] = {}
    if param_names is None:
        param_names = PARAM_NAMES
    print("\n" + "="*60)
    print(f"MCMC Diagnostics  ({sampler_type})")
    print("="*60)

    # Acceptance fraction (emcee only)
    if sampler_type == 'emcee' and hasattr(sampler, 'acceptance_fraction'):
        acc_frac = np.mean(sampler.acceptance_fraction)
        diag['acceptance_fraction_mean'] = float(acc_frac)
        print(f"Mean acceptance fraction: {acc_frac:.3f}")
        if acc_frac < 0.2:
            print("  WARNING: Low acceptance fraction - consider adjusting priors")
        elif acc_frac > 0.5:
            print("  WARNING: High acceptance fraction - chain may not be mixing well")
        else:
            print("  OK: Acceptance fraction in optimal range (0.2-0.5)")

    # Autocorrelation time
    try:
        if sampler_type == 'emcee' and hasattr(sampler, 'get_autocorr_time'):
            tau = sampler.get_autocorr_time(quiet=True)
        else:
            chain = sampler.get_chain()
            tau = np.array([
                emcee.autocorr.integrated_time(chain[:, :, i].mean(axis=1), quiet=True)[0]
                for i in range(chain.shape[2])
            ])
        print(f"\nAutocorrelation times:")
        diag['autocorr_time'] = {}
        for i, param in enumerate(param_names):
            if i < len(tau):
                print(f"  {param}: {tau[i]:.1f} steps")
                diag['autocorr_time'][param] = float(tau[i])

        chain = sampler.get_chain()
        n_steps = chain.shape[0]
        n_walkers = chain.shape[1]
        n_independent = n_steps / np.max(tau)
        diag['effective_independent_samples'] = int(n_independent * n_walkers)
        print(f"\nEffective independent samples: ~{int(n_independent * n_walkers)}")

        if n_steps < 50 * np.max(tau):
            diag['converged'] = False
            print("  WARNING: Chain may not be converged. Consider running longer.")
        else:
            diag['converged'] = True
            print("  OK: Chain appears well-converged")
    except Exception:
        diag['autocorr_time'] = None
        diag['effective_independent_samples'] = None
        diag['converged'] = None
        print("\nAutocorrelation time: Could not compute (chain too short)")

    print("="*60)
    return diag


# =============================================================================
# ArviZ Diagnostics & Model Comparison
# =============================================================================

def compute_pointwise_loglik(
    samples: np.ndarray,
    model,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    likelihood: str = 'chi2',
    n_samples: int = 200,
    reparam: bool = False,
    kepler: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    active_names: List[str] = None,
    fit_phase_shift: bool = True,
    phase_shift_grid_size: int = DEFAULT_PHASE_SHIFT_GRID_SIZE,
    phase_shift_eval_points: int = DEFAULT_PHASE_SHIFT_EVAL_POINTS,
    param_spec: Optional[ParamSpec] = None,
) -> np.ndarray:
    """Compute per-observation log-likelihood for a subset of posterior samples.

    Returns array of shape (n_samples_used, n_obs).
    """
    n_total = len(samples)
    n_use = min(n_samples, n_total)
    indices = np.random.choice(n_total, size=n_use, replace=False)
    n_obs = len(obs_phase)
    log_lik = np.full((n_use, n_obs), np.nan)

    if param_spec is not None:
        active_names = list(param_spec.active_names)
        reparam = (param_spec.mode == 'reparam')
        kepler = (param_spec.mode == 'kepler')
        wind_model = param_spec.wind_model
        fit_wind_shape = param_spec.fit_wind_shape
        likelihood = param_spec.likelihood

    if active_names is not None and 'log_f' in active_names:
        idx_logf = active_names.index('log_f')
    else:
        idx_logf = 5

    phase_shift_terms = _build_phase_shift_terms(
        fit_phase_shift,
        obs_phase,
        grid_size=phase_shift_grid_size,
        eval_points=phase_shift_eval_points,
    )
    eval_phases = (
        np.asarray(phase_shift_terms["phase_eval_grid"], dtype=float)
        if phase_shift_terms.get("enabled", False)
        else obs_phase
    )
    obs_err2 = obs_err ** 2

    for k, idx in enumerate(indices):
        theta = samples[idx]
        model_flux = _evaluate_model(
            theta, model, eval_phases, reparam=reparam,
            wind_model=wind_model, fit_wind_shape=fit_wind_shape,
            active_names=active_names,
            param_spec=param_spec,
        )
        if model_flux is None:
            continue
        if phase_shift_terms.get("enabled", False):
            model_flux, _ = _apply_best_phase_shift(
                eval_phases, model_flux, obs_phase, obs_flux, obs_err2, phase_shift_terms
            )
            if model_flux is None:
                continue

        if likelihood == 'jitter':
            f = np.exp(theta[idx_logf])
            sigma2 = obs_err2 + (f * model_flux) ** 2
            log_lik[k] = -0.5 * ((obs_flux - model_flux) ** 2 / sigma2
                                  + np.log(sigma2) + np.log(2 * np.pi))
        else:
            log_lik[k] = (
                -0.5 * ((obs_flux - model_flux) / obs_err) ** 2
                - np.log(obs_err) - 0.5 * np.log(2 * np.pi)
            )

    valid = ~np.any(np.isnan(log_lik), axis=1)
    return log_lik[valid]


def compute_bic_metrics(
    samples_flat: np.ndarray,
    param_names: List[str],
    model,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    likelihood: str = 'chi2',
    reparam: bool = False,
    kepler: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    log_prob_flat: Optional[np.ndarray] = None,
    fit_phase_shift: bool = True,
    phase_shift_grid_size: int = DEFAULT_PHASE_SHIFT_GRID_SIZE,
    phase_shift_eval_points: int = DEFAULT_PHASE_SHIFT_EVAL_POINTS,
    param_spec: Optional[ParamSpec] = None,
) -> Dict[str, float]:
    """Compute BIC from the best posterior point under the selected likelihood."""
    if samples_flat is None or len(samples_flat) == 0:
        return {}

    theta_source = "median_fallback"
    theta_hat = np.median(samples_flat, axis=0)

    lp = log_prob_flat
    if lp is not None and len(lp) == len(samples_flat):
        finite = np.isfinite(lp)
        if np.any(finite):
            idx = int(np.argmax(np.where(finite, lp, -np.inf)))
            theta_hat = samples_flat[idx]
            theta_source = "map_log_prob"

    if param_spec is not None:
        reparam = (param_spec.mode == 'reparam')
        kepler = (param_spec.mode == 'kepler')
        wind_model = param_spec.wind_model
        fit_wind_shape = param_spec.fit_wind_shape
        likelihood = param_spec.likelihood
        if param_names is None:
            param_names = list(param_spec.active_names)

    like_kwargs = dict(
        reparam=reparam,
        wind_model=wind_model,
        fit_wind_shape=fit_wind_shape,
        active_names=param_names,
        param_spec=param_spec,
    )
    phase_shift_terms = _build_phase_shift_terms(
        fit_phase_shift,
        obs_phase,
        grid_size=phase_shift_grid_size,
        eval_points=phase_shift_eval_points,
    )
    obs_err2 = obs_err ** 2
    if likelihood == 'jitter':
        logL_hat = log_likelihood_jitter(
            theta_hat, model, obs_phase, obs_flux, obs_err,
            obs_err2=obs_err2,
            jitter_logf_index=(param_names.index('log_f') if 'log_f' in param_names else None),
            phase_shift_terms=phase_shift_terms,
            **like_kwargs,
        )
    else:
        logL_hat = log_likelihood_chi2(
            theta_hat, model, obs_phase, obs_flux, obs_err,
            obs_err2=obs_err2, phase_shift_terms=phase_shift_terms, **like_kwargs,
        )

    k = int(len(param_names))
    n = int(len(obs_flux))
    if n <= 0 or not np.isfinite(logL_hat):
        return {}
    bic = k * np.log(n) - 2.0 * float(logL_hat)
    return {
        "bic": float(bic),
        "logL_hat": float(logL_hat),
        "k_params": float(k),
        "n_obs": float(n),
        "theta_source": theta_source,
    }


def _build_inference_data(posterior_dict, log_lik_dict=None):
    """Construct an ArviZ inference object from plain numpy dicts.

    Works across ArviZ versions: legacy (<= 0.17) uses InferenceData,
    modern (1.0+) uses xarray.DataTree via ``az.from_dict``.

    Parameters
    ----------
    posterior_dict : dict[str, ndarray]
        {param_name: array(chain, draw)}
    log_lik_dict : dict[str, ndarray] or None
        {name: array(chain, draw, obs)}
    """
    groups = {"posterior": posterior_dict}
    if log_lik_dict is not None:
        groups["log_likelihood"] = log_lik_dict

    # ArviZ >= 1.0: from_dict takes a single positional dict
    #   az.from_dict({"posterior": {...}, "log_likelihood": {...}})
    try:
        return az.from_dict(groups)
    except (TypeError, AttributeError):
        pass

    # ArviZ < 1.0: from_dict takes keyword arguments
    #   az.from_dict(posterior={...}, log_likelihood={...})
    try:
        return az.from_dict(**groups)
    except (TypeError, AttributeError):
        pass

    raise RuntimeError(
        f"Could not build inference data with arviz {getattr(az, '__version__', '?')}. "
        "Try: pip install --upgrade arviz"
    )


def run_arviz_diagnostics(
    sampler=None,
    param_names: List[str] = None,
    n_burn: int = 0,
    model=None,
    obs_phase: np.ndarray = None,
    obs_flux: np.ndarray = None,
    obs_err: np.ndarray = None,
    likelihood: str = 'chi2',
    compute_bic: bool = False,
    output_dir: str = None,
    suffix: str = '',
    chain: np.ndarray = None,
    samples_flat: np.ndarray = None,
    log_prob_flat: np.ndarray = None,
    reparam: bool = False,
    kepler: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    fit_phase_shift: bool = True,
    phase_shift_grid_size: int = DEFAULT_PHASE_SHIFT_GRID_SIZE,
    phase_shift_eval_points: int = DEFAULT_PHASE_SHIFT_EVAL_POINTS,
    param_spec: Optional[ParamSpec] = None,
):
    """Run ArviZ convergence diagnostics and optionally BIC.

    Accepts either a live *sampler* object or pre-saved arrays
    (*chain* and/or *samples_flat*) so diagnostics can be run
    without re-running MCMC.

    Parameters
    ----------
    sampler : emcee/zeus sampler, optional
        Live sampler (used when called right after MCMC).
    chain : np.ndarray, optional
        Saved chain array of shape (steps, walkers, dim).
        Used when *sampler* is None (e.g. loaded from disk).
    samples_flat : np.ndarray, optional
        Flat samples array (n_samples, dim). Derived from *chain*
        or *sampler* if not provided.

    Returns the ArviZ inference object (or None if ArviZ is unavailable).
    """
    if not HAS_ARVIZ and not compute_bic:
        print("arviz not installed. Install with: pip install arviz")
        return {"idata": None, "bic": {}}
    if param_names is None:
        param_names = PARAM_NAMES

    # Obtain chain and flat samples from whichever source is available
    if chain is None and sampler is not None:
        chain = sampler.get_chain(discard=n_burn)
    if samples_flat is None:
        if sampler is not None:
            samples_flat = sampler.get_chain(discard=n_burn, flat=True)
        elif chain is not None:
            n_steps, n_walkers, n_dim = chain.shape
            samples_flat = chain.reshape(-1, n_dim)

    if chain is None:
        print("No chain data available for ArviZ diagnostics.")
        return {"idata": None, "bic": {}}

    # posterior_dict: ArviZ wants (chain=walkers, draw=steps)
    posterior_dict = {
        name: chain[:, :, i].T for i, name in enumerate(param_names)
    }

    idata = None
    summary = None
    if HAS_ARVIZ:
        idata = _build_inference_data(posterior_dict, None)
        print("\n--- ArviZ Summary ---")
        summary = az.summary(idata)
        print(summary)

    bic_info = {}
    if compute_bic and model is not None and samples_flat is not None:
        if log_prob_flat is None and sampler is not None:
            try:
                log_prob_flat = sampler.get_log_prob(discard=n_burn, flat=True)
            except Exception:
                log_prob_flat = None
        bic_info = compute_bic_metrics(
            samples_flat=samples_flat,
            param_names=param_names,
            model=model,
            obs_phase=obs_phase,
            obs_flux=obs_flux,
            obs_err=obs_err,
            likelihood=likelihood,
            reparam=reparam,
            kepler=kepler,
            wind_model=wind_model,
            fit_wind_shape=fit_wind_shape,
            fit_phase_shift=fit_phase_shift,
            phase_shift_grid_size=phase_shift_grid_size,
            phase_shift_eval_points=phase_shift_eval_points,
            log_prob_flat=log_prob_flat,
            param_spec=param_spec,
        )
        if bic_info:
            print(
                "\nBIC: {bic:.3f}  (logL_hat={logL_hat:.3f}, k={k_params:.0f}, n={n_obs:.0f}, source={theta_source})".format(
                    **bic_info
                )
            )

    if output_dir and summary is not None:
        csv_path = os.path.join(output_dir, f"{suffix}_arviz_summary.csv" if suffix else "arviz_summary.csv")
        summary.to_csv(csv_path)
        print(f"ArviZ summary saved to: {csv_path}")
    if output_dir and bic_info:
        metrics_path = os.path.join(
            output_dir, f"{suffix}_model_metrics.csv" if suffix else "model_metrics.csv"
        )
        pd.DataFrame([bic_info]).to_csv(metrics_path, index=False)
        print(f"Model metrics saved to: {metrics_path}")

    return {"idata": idata, "bic": bic_info}


# =============================================================================
# Main CLI
# =============================================================================

def run_single_fit(
    band: str,
    wind_model: str,
    args,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    obs_phase_width: Optional[np.ndarray] = None,
    priors: Dict = None,
    sim_params: Dict = None,
    reparam: bool = False,
    kepler: bool = False,
    fit_wind_shape: bool = False,
    shape_prior_overrides: Dict[str, Dict] = None,
    is_binned: bool = True,
) -> Tuple[Dict, object]:
    """Run MCMC fit for a single band/wind_model combination."""

    if priors is None:
        if kepler:
            priors = copy.deepcopy(KEPLER_PRIORS)
        elif reparam:
            priors = REPARAM_PRIORS.copy()
        else:
            priors = DEFAULT_PRIORS.copy()
    if sim_params is None:
        sim_params = {}

    likelihood = getattr(args, 'likelihood', 'chi2')
    frozen_params = dict(getattr(args, 'frozen_params', {}) or {})
    orbital_period_s = float(getattr(args, 'orbital_period', ORBITAL_PERIOD))

    param_spec = build_param_spec(
        likelihood=likelihood,
        reparam=reparam,
        kepler=kepler,
        wind_model=wind_model,
        fit_wind_shape=fit_wind_shape,
        frozen=frozen_params,
        orbital_period_s=orbital_period_s,
    )

    # Active priors include geometry + (optional) jitter + (optional) shape.
    active_priors = get_active_priors(
        base_priors=priors,
        wind_model=wind_model,
        fit_wind_shape=fit_wind_shape,
        likelihood=likelihood,
        shape_prior_overrides=shape_prior_overrides,
        frozen=param_spec.frozen,
    )

    print(f"\n{'#'*60}")
    print(f"# Fitting {band.upper()} band - {WIND_MODELS[wind_model]}")
    print('#'*60)

    if sim_params:
        print(f"# Simulation params: {sim_params}")

    print("\n[info] Using DirectLightCurveModel for MCMC.")
    model = DirectLightCurveModel(
        band=band,
        flux_csv_path=args.flux_csv,
        wind_model=wind_model,
        dth=args.dth,
        sim_params=sim_params,
    )

    sampler_type = getattr(args, 'sampler', 'emcee')

    fit_start = time.time()
    sampler, samples, active_names, active_labels = run_mcmc(
        model=model,
        obs_phase=obs_phase,
        obs_flux=obs_flux,
        obs_err=obs_err,
        n_walkers=args.n_walkers,
        n_steps=args.n_steps,
        n_burn=args.n_burn,
        priors=active_priors,
        progress=not args.quiet,
        n_threads=args.n_threads,
        sampler_type=sampler_type,
        likelihood=likelihood,
        reparam=reparam,
        kepler=kepler,
        wind_model=wind_model,
        fit_wind_shape=fit_wind_shape,
        frozen=param_spec.frozen,
        orbital_period_s=orbital_period_s,
        param_spec=param_spec,
        numba_threads_per_worker=getattr(
            args, "numba_threads_per_worker", None
        ),
        fit_phase_shift=not getattr(args, "no_fit_phase_shift", False),
        phase_shift_grid_size=getattr(args, "phase_shift_grid_size", DEFAULT_PHASE_SHIFT_GRID_SIZE),
        phase_shift_eval_points=getattr(args, "phase_shift_eval_points", DEFAULT_PHASE_SHIFT_EVAL_POINTS),
    )
    fit_elapsed_s = float(time.time() - fit_start)

    try:
        chain_log_prob_flat = sampler.get_log_prob(discard=args.n_burn, flat=True)
    except Exception:
        chain_log_prob_flat = None
    stats = compute_statistics(
        samples,
        param_names=active_names,
        reparam=reparam,
        kepler=kepler,
        orbital_period_s=orbital_period_s,
        param_spec=param_spec,
        log_prob=chain_log_prob_flat,
    )
    print_results(stats, band, wind_model, param_names=active_names, reparam=reparam)
    diag_info = print_diagnostics(sampler, sampler_type=sampler_type, param_names=active_names)
    stats['_run_meta'] = {
        'sampler': sampler_type,
        'likelihood': likelihood,
        'n_walkers': int(args.n_walkers),
        'n_steps': int(args.n_steps),
        'n_burn': int(args.n_burn),
        'fit_elapsed_s': fit_elapsed_s,
        'fit_phase_shift': bool(not getattr(args, "no_fit_phase_shift", False)),
        'phase_shift_grid_size': int(getattr(args, "phase_shift_grid_size", DEFAULT_PHASE_SHIFT_GRID_SIZE)),
        'phase_shift_eval_points': int(getattr(args, "phase_shift_eval_points", DEFAULT_PHASE_SHIFT_EVAL_POINTS)),
    }
    if isinstance(diag_info, dict):
        stats['_diagnostics'] = diag_info

    compute_bic_flag = bool(getattr(args, 'compute_bic', False))
    diag_out = run_arviz_diagnostics(
        sampler, active_names, args.n_burn,
        model=model, obs_phase=obs_phase, obs_flux=obs_flux, obs_err=obs_err,
        likelihood=likelihood,
        compute_bic=compute_bic_flag,
        output_dir=args.output_dir,
        suffix=f"{band}_{wind_model}",
        log_prob_flat=chain_log_prob_flat,
        reparam=reparam,
        kepler=kepler,
        wind_model=wind_model,
        fit_wind_shape=fit_wind_shape,
        fit_phase_shift=not getattr(args, "no_fit_phase_shift", False),
        phase_shift_grid_size=getattr(args, "phase_shift_grid_size", DEFAULT_PHASE_SHIFT_GRID_SIZE),
        phase_shift_eval_points=getattr(args, "phase_shift_eval_points", DEFAULT_PHASE_SHIFT_EVAL_POINTS),
        param_spec=param_spec,
    )
    bic_info = (diag_out or {}).get("bic", {})
    if bic_info:
        stats.update(bic_info)

    # Generate file suffix
    suffix = f"{band}_{wind_model}"

    # Generate plots
    if not args.no_plots:
        plot_corner(
            samples, band, wind_model,
            os.path.join(args.output_dir, f"{suffix}_corner.png"),
            param_labels=active_labels,
        )
        plot_trace(
            sampler, band, wind_model,
            os.path.join(args.output_dir, f"{suffix}_trace.png"),
            param_labels=active_labels,
            n_burn=args.n_burn,
        )
        red_chi2 = plot_best_fit(
            model, obs_phase, obs_flux, obs_err, obs_phase_width, stats, band, wind_model,
            os.path.join(args.output_dir, f"{suffix}_bestfit.png"),
            param_names=active_names,
            reparam=reparam,
            fit_wind_shape=fit_wind_shape,
            is_binned=is_binned,
            likelihood=likelihood,
            fit_phase_shift=not getattr(args, "no_fit_phase_shift", False),
            phase_shift_grid_size=getattr(args, "phase_shift_grid_size", DEFAULT_PHASE_SHIFT_GRID_SIZE),
            phase_shift_eval_points=getattr(args, "phase_shift_eval_points", DEFAULT_PHASE_SHIFT_EVAL_POINTS),
            param_spec=param_spec,
        )
        stats['reduced_chi2'] = red_chi2

    try:
        log_prob = sampler.get_log_prob(discard=args.n_burn, flat=True)
    except Exception:
        log_prob = None

    samples_csv_path = os.path.join(args.output_dir, f"{suffix}_samples.csv")
    if not getattr(args, "no_csv_output", False):
        save_samples_csv_chunked(
            samples=samples,
            param_names=active_names,
            output_path=samples_csv_path,
            log_prob=log_prob,
            chunk_size=getattr(args, "csv_chunk_size", 50000),
        )
        print(f"Samples saved to: {samples_csv_path}")

    if getattr(args, "compact_output", False):
        samples_npz_path = os.path.join(args.output_dir, f"{suffix}_samples.npz")
        np.savez_compressed(
            samples_npz_path,
            samples=samples,
            param_names=np.array(active_names, dtype=str),
            log_prob=(log_prob if log_prob is not None else np.array([])),
        )
        print(f"Compact samples saved to: {samples_npz_path}")

    try:
        chain_full = sampler.get_chain(discard=args.n_burn)
        log_prob_chain = sampler.get_log_prob(discard=args.n_burn)
        chain_path = os.path.join(args.output_dir, f"{suffix}_chain.npz")
        np.savez_compressed(
            chain_path,
            chain=chain_full,
            log_prob=log_prob_chain,
            param_names=active_names,
            n_burn=args.n_burn,
            likelihood=likelihood,
            reparam=reparam,
            mode=param_spec.mode,
            frozen_names=np.array(list(param_spec.frozen.keys()), dtype=str),
            frozen_values=np.array(list(param_spec.frozen.values()), dtype=float),
            orbital_period_s=float(param_spec.orbital_period_s),
            wind_model=wind_model,
            fit_wind_shape=fit_wind_shape,
            bic=(bic_info.get("bic") if bic_info else np.nan),
            logL_hat=(bic_info.get("logL_hat") if bic_info else np.nan),
            k_params=(bic_info.get("k_params") if bic_info else np.nan),
            n_obs=(bic_info.get("n_obs") if bic_info else np.nan),
        )
        print(f"Full chain saved to: {chain_path}")
    except Exception as e:
        warnings.warn(f"Could not save full chain: {e}")

    if getattr(args, 'save_chi2', False):
        chi2_path = os.path.join(args.output_dir, f"{suffix}_chi2.csv.gz")
        compute_chi2_for_samples(
            model, samples, obs_phase, obs_flux, obs_err,
            output_path=chi2_path,
            n_samples=getattr(args, 'chi2_n_samples', None),
            verbose=True,
            reparam=reparam,
            wind_model=wind_model,
            fit_wind_shape=fit_wind_shape,
            active_names=active_names,
            likelihood=likelihood,
            fit_phase_shift=not getattr(args, "no_fit_phase_shift", False),
            phase_shift_grid_size=getattr(args, "phase_shift_grid_size", DEFAULT_PHASE_SHIFT_GRID_SIZE),
            phase_shift_eval_points=getattr(args, "phase_shift_eval_points", DEFAULT_PHASE_SHIFT_EVAL_POINTS),
            param_spec=param_spec,
        )

    stats['wind_model'] = wind_model
    return stats, model


def replot_from_existing(
    args,
    band: str,
    wind_model: str,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    obs_phase_width: Optional[np.ndarray] = None,
    priors: Dict = None,
    sim_params: Dict = None,
    reparam: bool = False,
    kepler: bool = False,
    fit_wind_shape: bool = False,
    shape_prior_overrides: Dict[str, Dict] = None,
    is_binned: bool = True,
) -> Optional[Dict]:
    """Regenerate plots from existing MCMC results without re-running MCMC.

    The fit_wind_shape flag here matches what *the saved chain* contains; it
    is only used to build the right wind_params for forward model evaluation.
    If the saved chain has more dims than the geometry block, fit_wind_shape
    is auto-detected.
    """
    if priors is None:
        if kepler:
            priors = copy.deepcopy(KEPLER_PRIORS)
        elif reparam:
            priors = REPARAM_PRIORS.copy()
        else:
            priors = DEFAULT_PRIORS.copy()
    if sim_params is None:
        sim_params = {}

    print(f"\n{'#'*60}")
    print(f"# Replotting {band.upper()} band - {WIND_MODELS[wind_model]}")
    print('#'*60)

    saved_mode = 'kepler' if kepler else ('reparam' if reparam else 'phys')
    saved_orbital_period_s = float(getattr(args, 'orbital_period', ORBITAL_PERIOD))
    saved_frozen = dict(getattr(args, 'frozen_params', {}) or {})
    chain_path = os.path.join(args.output_dir, f"{band}_{wind_model}_chain.npz")
    if os.path.exists(chain_path):
        try:
            _meta = np.load(chain_path, allow_pickle=True)
            saved_mode = str(_meta.get('mode', saved_mode))
            saved_orbital_period_s = float(_meta.get('orbital_period_s', saved_orbital_period_s))
            fn = list(_meta.get('frozen_names', []))
            fv = list(_meta.get('frozen_values', []))
            if len(fn) == len(fv):
                saved_frozen = {str(k): float(v) for k, v in zip(fn, fv)}
        except Exception:
            pass

    spec_for_replot = build_param_spec(
        likelihood=getattr(args, 'likelihood', 'chi2'),
        reparam=(saved_mode == 'reparam'),
        kepler=(saved_mode == 'kepler'),
        wind_model=wind_model,
        fit_wind_shape=fit_wind_shape,
        frozen=saved_frozen,
        orbital_period_s=saved_orbital_period_s,
    )

    samples, stats, loaded_names = load_existing_results(
        args.output_dir,
        band,
        wind_model,
        reparam=(saved_mode == 'reparam'),
        kepler=(saved_mode == 'kepler'),
        orbital_period_s=saved_orbital_period_s,
        param_spec=spec_for_replot,
    )

    if samples is None or stats is None:
        print(f"Could not load existing results for {band}_{wind_model}")
        return None

    print_results(
        stats, band, wind_model,
        param_names=loaded_names, reparam=(saved_mode == 'reparam')
    )

    # Auto-detect shape-fit from the saved column names.
    if saved_mode == 'reparam':
        geom_names = REPARAM_PARAM_NAMES
    elif saved_mode == 'kepler':
        geom_names = KEPLER_PARAM_NAMES
    else:
        geom_names = PARAM_NAMES
    extra_dims = [n for n in loaded_names
                  if n not in geom_names and n != 'log_f']
    saved_fit_wind_shape = bool(extra_dims) or fit_wind_shape
    spec_for_replot.active_names = list(loaded_names)
    spec_for_replot.fit_wind_shape = bool(saved_fit_wind_shape)

    print("\nUsing DirectLightCurveModel for replot.")
    model = DirectLightCurveModel(
        band=band,
        flux_csv_path=args.flux_csv,
        wind_model=wind_model,
        dth=args.dth,
        sim_params=sim_params,
    )

    suffix = f"{band}_{wind_model}"
    saved_likelihood = getattr(args, 'likelihood', 'chi2')
    chain_path = os.path.join(args.output_dir, f"{suffix}_chain.npz")
    if os.path.exists(chain_path):
        try:
            _meta = np.load(chain_path, allow_pickle=True)
            saved_likelihood = str(_meta.get('likelihood', saved_likelihood))
            spec_for_replot.likelihood = saved_likelihood
        except Exception:
            pass

    if not args.no_plots:
        # Use the loaded column names as the active set.
        active_names = list(loaded_names)
        active_labels = []
        if saved_mode == 'reparam':
            geom_labels = REPARAM_PARAM_LABELS
        elif saved_mode == 'kepler':
            geom_labels = KEPLER_PARAM_LABELS
        else:
            geom_labels = PARAM_LABELS
        for name in active_names:
            if name in geom_names:
                active_labels.append(geom_labels[geom_names.index(name)])
            elif name == 'log_f':
                active_labels.append(r'$\ln\,f$')
            elif name in WIND_SHAPE_LABELS:
                active_labels.append(WIND_SHAPE_LABELS[name])
            else:
                active_labels.append(name)

        plot_corner(
            samples, band, wind_model,
            os.path.join(args.output_dir, f"{suffix}_corner.png"),
            param_labels=active_labels,
        )

        red_chi2 = plot_best_fit(
            model, obs_phase, obs_flux, obs_err, obs_phase_width, stats, band, wind_model,
            os.path.join(args.output_dir, f"{suffix}_bestfit.png"),
            param_names=active_names,
            reparam=(saved_mode == 'reparam'),
            fit_wind_shape=saved_fit_wind_shape,
            is_binned=is_binned,
            likelihood=saved_likelihood,
            fit_phase_shift=not getattr(args, "no_fit_phase_shift", False),
            phase_shift_grid_size=getattr(args, "phase_shift_grid_size", DEFAULT_PHASE_SHIFT_GRID_SIZE),
            phase_shift_eval_points=getattr(args, "phase_shift_eval_points", DEFAULT_PHASE_SHIFT_EVAL_POINTS),
            param_spec=spec_for_replot,
        )
        stats['reduced_chi2'] = red_chi2

    compute_bic_flag = bool(getattr(args, 'compute_bic', False))
    if HAS_ARVIZ or compute_bic_flag:
        if os.path.exists(chain_path):
            print(f"\nLoading saved chain from: {chain_path}")
            chain_data = np.load(chain_path, allow_pickle=True)
            saved_chain = chain_data['chain']
            saved_lp = chain_data['log_prob'] if 'log_prob' in chain_data else None
            saved_names = list(chain_data['param_names'])
            saved_likelihood = str(chain_data.get('likelihood', 'chi2'))
            spec_for_replot.likelihood = saved_likelihood
            saved_reparam = bool(chain_data.get('reparam', False))
            saved_wind_model = str(chain_data.get('wind_model', wind_model))
            saved_fws = bool(chain_data.get('fit_wind_shape', saved_fit_wind_shape))
            print(f"  Chain shape: {saved_chain.shape} "
                  f"(likelihood={saved_likelihood})")

            diag_out = run_arviz_diagnostics(
                param_names=saved_names,
                chain=saved_chain,
                model=model,
                obs_phase=obs_phase,
                obs_flux=obs_flux,
                obs_err=obs_err,
                likelihood=saved_likelihood,
                compute_bic=compute_bic_flag,
                output_dir=args.output_dir,
                suffix=suffix,
                log_prob_flat=(
                    saved_lp.reshape(-1) if isinstance(saved_lp, np.ndarray) else None
                ),
                reparam=saved_reparam,
                kepler=(saved_mode == 'kepler'),
                wind_model=saved_wind_model,
                fit_wind_shape=saved_fws,
                fit_phase_shift=not getattr(args, "no_fit_phase_shift", False),
                phase_shift_grid_size=getattr(args, "phase_shift_grid_size", DEFAULT_PHASE_SHIFT_GRID_SIZE),
                phase_shift_eval_points=getattr(args, "phase_shift_eval_points", DEFAULT_PHASE_SHIFT_EVAL_POINTS),
                param_spec=spec_for_replot,
            )
            bic_info = (diag_out or {}).get("bic", {})
            if bic_info:
                stats.update(bic_info)
        elif compute_bic_flag:
            print(f"Chain file not found: {chain_path}")
            print("  Re-run MCMC to generate it, or skip --compute-bic.")

    if getattr(args, 'save_chi2', False):
        chi2_path = os.path.join(args.output_dir, f"{suffix}_chi2.csv.gz")
        compute_chi2_for_samples(
            model, samples, obs_phase, obs_flux, obs_err,
            output_path=chi2_path,
            n_samples=getattr(args, 'chi2_n_samples', None),
            verbose=True,
            reparam=(saved_mode == 'reparam'),
            wind_model=wind_model,
            fit_wind_shape=saved_fit_wind_shape,
            active_names=loaded_names,
            likelihood=saved_likelihood,
            fit_phase_shift=not getattr(args, "no_fit_phase_shift", False),
            phase_shift_grid_size=getattr(args, "phase_shift_grid_size", DEFAULT_PHASE_SHIFT_GRID_SIZE),
            phase_shift_eval_points=getattr(args, "phase_shift_eval_points", DEFAULT_PHASE_SHIFT_EVAL_POINTS),
            param_spec=spec_for_replot,
        )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="MCMC fitting of XRB light curves to observed Chandra data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--band",
        type=str,
        required=True,
        choices=['broad', 'soft', 'medium', 'hard', 'all'],
        help="Energy band to fit (or 'all' to fit all bands)"
    )
    parser.add_argument(
        "--flux-csv",
        type=str,
        required=True,
        help="Path to flux vs nH CSV file (from compute_flux_vs_nH.py)"
    )
    
    # Wind model option
    parser.add_argument(
        "--wind-model",
        type=str,
        choices=list(WIND_MODELS.keys()),
        default='smooth_pl',
        help=("Wind density model. Choices: " +
              ", ".join(f"{k} ({v})" for k, v in WIND_MODELS.items()))
    )
    parser.add_argument(
        "--fit-wind-shape",
        action="store_true",
        help=(
            "Add the wind-shape parameters of the chosen --wind-model as free "
            "MCMC dimensions (smooth_pl: Rb, p; beta_law: beta; "
            "confinement: fconf, ell). Override priors via --prior-Rb, --prior-p, "
            "--prior-beta, --prior-fconf, --prior-ell."
        ),
    )
    
    # Data options
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/IC_10_X1_LC",
        help="Base directory containing light curve data. Accepts a direct path to "
             ".txt files, or a parent directory with {Band}_with_flux/ or {band}/single/ sub-folders."
    )
    parser.add_argument(
        "--obs-column",
        type=str,
        default="FLUX",
        help="Column name for the observable in data files (e.g. FLUX, flux_t, rate, ECF)"
    )
    parser.add_argument(
        "--obs-error-column",
        type=str,
        default=None,
        help="Column name for errors. If omitted, auto-detected from --obs-column "
             "(e.g. FLUX -> FLUX_ERR, rate -> rate_err). For proportional "
             "columns like flux_t, errors are inferred from rate_err when possible."
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default=None,
        help="Column name for timestamps (e.g. TIME, t_raw). Auto-detected if omitted."
    )
    parser.add_argument(
        "--n-phase-bins",
        type=int,
        default=None,
        help=(
            "Use fixed-width phase binning with this many bins "
            "(variable counts per bin). Mutually exclusive with --counts-per-bin. "
            "If neither binning option is set, defaults to 50 fixed bins."
        ),
    )
    parser.add_argument(
        "--counts-per-bin",
        type=int,
        default=None,
        help=(
            "Use adaptive phase binning with approximately constant counts "
            "per bin (variable phase width). Mutually exclusive with "
            "--n-phase-bins. Recommended value: 100."
        ),
    )
    parser.add_argument(
        "--no-phase-bin",
        action="store_true",
        help="Use raw 100s data without phase binning. Usually best paired with "
             "--likelihood jitter."
    )
    
    # MCMC options
    parser.add_argument(
        "--sampler",
        type=str,
        choices=['emcee', 'zeus'],
        default='emcee',
        help="MCMC sampler: 'emcee' (stretch-move ensemble) or 'zeus' (slice-sampling ensemble, "
             "better mixing for correlated posteriors). Install zeus with: pip install zeus-mcmc"
    )
    parser.add_argument(
        "--likelihood",
        type=str,
        choices=['chi2', 'jitter'],
        default='chi2',
        help="Likelihood function: 'chi2' (standard Gaussian/chi-squared), "
             "'jitter' (Gaussian with free systematic error term — adds log_f parameter)"
    )
    parser.add_argument(
        "--reparam",
        action="store_true",
        help="Reparameterize (d1, d2) as (a, q) where a = d1+d2 (orbital separation) "
             "and q = d1/(d1+d2) (mass-ratio proxy). Decorrelates the two distance "
             "parameters for faster MCMC convergence."
    )
    parser.add_argument(
        "--kepler",
        action="store_true",
        help="Sample (M_X, M_RH) and derive (a, q) from Kepler's third law and lever-arm. "
             "Mutually exclusive with --reparam."
    )
    parser.add_argument(
        "--orbital-period",
        type=float,
        default=float(ORBITAL_PERIOD),
        help="Fixed orbital period in seconds for --kepler mode."
    )
    parser.add_argument(
        "--freeze",
        type=str,
        default=None,
        metavar="NAME=VAL[,NAME=VAL,...]",
        help="Freeze selected free parameters at fixed values and remove them from sampling. "
             "Supported names: d1,d2,a,q,r,R,i0,M_X,M_RH,Rb,p,beta,fconf,ell. "
             "log_f cannot be frozen."
    )
    parser.add_argument(
        "--n-walkers",
        type=int,
        default=32,
        help="Number of MCMC walkers"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=5000,
        help="Number of MCMC steps"
    )
    parser.add_argument(
        "--no-fit-phase-shift",
        action="store_true",
        help="Disable per-sample phase-shift alignment in likelihood evaluation. "
             "By default, each likelihood call searches for the best phase shift "
             "that minimizes weighted chi-square against the observed light curve."
    )
    parser.add_argument(
        "--phase-shift-grid-size",
        type=int,
        default=DEFAULT_PHASE_SHIFT_GRID_SIZE,
        help="Number of coarse trial phase shifts per likelihood call when "
             "phase-shift alignment is enabled. A local refinement pass is "
             "applied around the best coarse shift, so moderate values are "
             "typically both fast and accurate."
    )
    parser.add_argument(
        "--phase-shift-eval-points",
        type=int,
        default=DEFAULT_PHASE_SHIFT_EVAL_POINTS,
        help="Number of model phase points used internally before per-sample "
             "phase-shift alignment (higher = smoother shift interpolation, slower)."
    )
    parser.add_argument(
        "--n-burn",
        type=int,
        default=1000,
        help="Number of burn-in steps to discard"
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=1,
        help="Number of worker processes for parallel MCMC sampling (1 = serial). "
             "Parallel workers are used with DirectLightCurveModel evaluations."
    )
    parser.add_argument(
        "--numba-threads-per-worker",
        type=int,
        default=None,
        metavar="N",
        help="When using --n-threads>1, set Numba's thread count inside each worker "
             "via numba.set_num_threads(N). Default (omit this "
             "flag): auto = max(1, cpu_count // n_threads) so workers collectively "
             "use about one thread per logical CPU without each worker running a "
             "fully serial Numba kernel. Set explicitly if you tune "
             "--n-threads for your machine."
    )
    parser.add_argument(
        "--compute-bic",
        action="store_true",
        help="Compute Bayesian Information Criterion (BIC) model-comparison metrics. "
             "Works for chi2 and jitter likelihoods."
    )
    
    parser.add_argument(
        "--dth",
        type=float,
        default=5.0,
        help="Model phase resolution in degrees (larger = faster)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mcmc_results",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating diagnostic plots"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bar"
    )
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Also save compact binary outputs (NPZ) for samples/metadata."
    )
    parser.add_argument(
        "--no-csv-output",
        action="store_true",
        help="Skip writing large *_samples.csv tables."
    )
    parser.add_argument(
        "--csv-chunk-size",
        type=int,
        default=50000,
        help="Chunk size used when writing sample CSV files."
    )
    
    # Replot option - regenerate plots from existing MCMC results
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate plots from existing MCMC results (samples CSV files) without re-running MCMC. "
             "Requires samples CSV files to exist in the output directory."
    )
    
    # Chi-square output options
    parser.add_argument(
        "--save-chi2",
        action="store_true",
        help="Compute and save chi-square values for all MCMC samples to a compressed file. "
             "Output: {band}_{wind_model}_chi2.csv.gz"
    )
    parser.add_argument(
        "--chi2-n-samples",
        type=int,
        default=None,
        help="Number of samples to compute chi-square for (default: all samples). "
             "Use a smaller number for faster computation."
    )
    
    # ==========================================================================
    # Simulation parameters (passed to simulate_lightcurve)
    # ==========================================================================
    sim_group = parser.add_argument_group(
        'Simulation Parameters',
        'Parameters passed to the underlying simulate_lightcurve function'
    )
    sim_group.add_argument(
        "--lam",
        type=float,
        default=0.589537,
        help="Target mean nH (in 1e22 cm^-2 units). The raw wind LOS integral "
             "is rescaled so that mean(fl) = lam, which sets the absolute "
             "flux normalization (and is fixed from spectral fits)."
    )
    sim_group.add_argument(
        "--gma0",
        type=float,
        default=-90.0,
        help="Starting phase angle in degrees"
    )
    sim_group.add_argument(
        "--d2h",
        type=float,
        default=6.0,
        help="Angular cell size (degrees) for the polar grid in surface integral"
    )
    sim_group.add_argument(
        "--dz",
        type=float,
        default=0.1,
        help="Step size along line of sight (solar radii)"
    )
    
    # ==========================================================================
    # Prior customization
    # ==========================================================================
    prior_group = parser.add_argument_group(
        'Prior Customization',
        'Override default priors for fitted parameters (d1, d2, r, R, i0)'
    )
    prior_group.add_argument(
        "--prior-d1",
        type=str,
        default=None,
        metavar="MEAN,STD,MIN,MAX",
        help="Prior for d1 (compact object distance from COM). "
             "Format: mean,std,min,max. Default: 11.0,3.0,5.0,20.0"
    )
    prior_group.add_argument(
        "--prior-d2",
        type=str,
        default=None,
        metavar="MEAN,STD,MIN,MAX",
        help="Prior for d2 (companion distance from COM). "
             "Format: mean,std,min,max. Default: 8.0,3.0,3.0,15.0"
    )
    prior_group.add_argument(
        "--prior-r",
        type=str,
        default=None,
        metavar="MEAN,STD,MIN,MAX",
        help="Prior for r (compact object/disk radius). "
             "Format: mean,std,min,max. Default: 0.001,0.001,0.0001,0.01"
    )
    prior_group.add_argument(
        "--prior-R",
        type=str,
        default=None,
        metavar="MEAN,STD,MIN,MAX",
        help="Prior for R (companion star radius). "
             "Format: mean,std,min,max. Default: 2.0,0.5,1.0,5.0"
    )
    prior_group.add_argument(
        "--prior-i0",
        type=str,
        default=None,
        metavar="MEAN,STD,MIN,MAX",
        help="Prior for i0 (orbital inclination in degrees). "
             "Format: mean,std,min,max. Default: 26.0,20.0,10.0,85.0"
    )
    prior_group.add_argument(
        "--prior-a",
        type=str,
        default=None,
        metavar="MEAN,STD,MIN,MAX",
        help="Prior for a = d1+d2 (orbital separation, only with --reparam). "
             "Format: mean,std,min,max. Default: 19.0,4.0,8.0,35.0"
    )
    prior_group.add_argument(
        "--prior-q",
        type=str,
        default=None,
        metavar="MEAN,STD,MIN,MAX",
        help="Prior for q = d1/(d1+d2) (mass-ratio proxy, only with --reparam). "
             "Format: mean,std,min,max. Default: 0.58,0.15,0.01,0.99"
    )
    prior_group.add_argument(
        "--prior-MX",
        dest="prior_M_X",
        type=str,
        default=None,
        metavar="MEAN,STD,MIN,MAX",
        help="Prior for compact-object mass M_X (solar masses, only with --kepler). "
             "Format: mean,std,min,max. Default: 30.0,10.0,1.0,100.0"
    )
    prior_group.add_argument(
        "--prior-MRH",
        dest="prior_M_RH",
        type=str,
        default=None,
        metavar="MEAN,STD,MIN,MAX",
        help="Prior for companion mass M_RH (solar masses, only with --kepler). "
             "Format: mean,std,min,max. Default: 20.0,10.0,1.0,100.0"
    )

    # Wind-shape prior overrides (only used with --fit-wind-shape).
    shape_prior_group = parser.add_argument_group(
        'Wind-Shape Prior Customization',
        'Override default priors for wind-shape parameters '
        '(only active with --fit-wind-shape)',
    )
    for sname in ('Rb', 'p', 'beta', 'fconf', 'ell'):
        prior_def = WIND_SHAPE_PRIORS[sname]
        shape_prior_group.add_argument(
            f"--prior-{sname}",
            type=str,
            default=None,
            metavar="MEAN,STD,MIN,MAX",
            help=(
                f"Prior for wind-shape parameter '{sname}'. "
                f"Format: mean,std,min,max. "
                f"Default: {prior_def['mean']},{prior_def['std']},"
                f"{prior_def['min']},{prior_def['max']}"
            ),
        )

    args = parser.parse_args()

    if args.n_phase_bins is not None and args.counts_per_bin is not None:
        parser.error(
            "Specify either --n-phase-bins (fixed-width) or --counts-per-bin "
            "(constant-SNR), not both."
        )
    if args.n_phase_bins is not None and args.n_phase_bins <= 0:
        parser.error("--n-phase-bins must be > 0.")
    if args.counts_per_bin is not None and args.counts_per_bin <= 0:
        parser.error("--counts-per-bin must be > 0.")

    if args.reparam and getattr(args, 'kepler', False):
        parser.error("--reparam and --kepler are mutually exclusive.")
    if getattr(args, 'orbital_period', 0.0) <= 0:
        parser.error("--orbital-period must be > 0.")
    try:
        args.frozen_params = parse_freeze_map(getattr(args, 'freeze', None))
    except Exception as e:
        parser.error(f"Invalid --freeze: {e}")

    # Build simulation parameters dict
    sim_params = {
        'lam': args.lam,
        'gma0': args.gma0,
        'd2h': args.d2h,
        'dz': args.dz,
    }

    # Build custom geometry priors
    reparam = getattr(args, 'reparam', False)
    kepler = bool(getattr(args, 'kepler', False))
    if kepler:
        priors = copy.deepcopy(KEPLER_PRIORS)
        base_names = KEPLER_PARAM_NAMES
    elif reparam:
        priors = copy.deepcopy(REPARAM_PRIORS)
        base_names = REPARAM_PARAM_NAMES
    else:
        priors = DEFAULT_PRIORS.copy()
        base_names = PARAM_NAMES
    for param in base_names:
        prior_arg = getattr(args, f'prior_{param}', None)
        if prior_arg:
            try:
                parts = [float(x.strip()) for x in prior_arg.split(',')]
                if len(parts) != 4:
                    raise ValueError(f"Expected 4 values for --prior-{param}")
                priors[param] = {
                    'mean': parts[0],
                    'std': parts[1],
                    'min': parts[2],
                    'max': parts[3],
                }
                print(f"Custom prior for {param}: mean={parts[0]}, std={parts[1]}, "
                      f"min={parts[2]}, max={parts[3]}")
            except Exception as e:
                parser.error(f"Invalid format for --prior-{param}: {e}")

    # Build wind-shape prior overrides (always parsed; only applied when
    # --fit-wind-shape and the param is in WIND_SHAPE_FIT[wind_model]).
    shape_prior_overrides: Dict[str, Dict[str, float]] = {}
    for sname in ('Rb', 'p', 'beta', 'fconf', 'ell'):
        prior_arg = getattr(args, f'prior_{sname}', None)
        if prior_arg:
            try:
                parts = [float(x.strip()) for x in prior_arg.split(',')]
                if len(parts) != 4:
                    raise ValueError(f"Expected 4 values for --prior-{sname}")
                shape_prior_overrides[sname] = {
                    'mean': parts[0],
                    'std': parts[1],
                    'min': parts[2],
                    'max': parts[3],
                }
                print(f"Custom prior for shape param {sname}: "
                      f"mean={parts[0]}, std={parts[1]}, min={parts[2]}, max={parts[3]}")
            except Exception as e:
                parser.error(f"Invalid format for --prior-{sname}: {e}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which bands to fit
    if args.band == 'all':
        bands = ['broad', 'soft', 'medium', 'hard']
    else:
        bands = [args.band]

    # Wind models: a single value (no more 'both' loop).
    wind_models = [args.wind_model]
    fit_wind_shape = bool(getattr(args, 'fit_wind_shape', False))
    frozen_params = dict(getattr(args, 'frozen_params', {}) or {})

    # Validate frozen names against run configuration before sampling starts.
    try:
        _spec_check = build_param_spec(
            likelihood=getattr(args, 'likelihood', 'chi2'),
            reparam=reparam,
            kepler=kepler,
            wind_model=args.wind_model,
            fit_wind_shape=fit_wind_shape,
            frozen=frozen_params,
            orbital_period_s=float(getattr(args, 'orbital_period', ORBITAL_PERIOD)),
        )
    except Exception as e:
        parser.error(str(e))

    _check_priors = get_active_priors(
        base_priors=priors,
        wind_model=args.wind_model,
        fit_wind_shape=fit_wind_shape,
        likelihood=getattr(args, 'likelihood', 'chi2'),
        shape_prior_overrides=shape_prior_overrides,
        frozen=None,
    )
    for sname in WIND_SHAPE_FIT.get(args.wind_model, []):
        _check_priors.setdefault(sname, dict(WIND_SHAPE_PRIORS[sname]))
        if sname in shape_prior_overrides:
            _check_priors[sname].update(shape_prior_overrides[sname])
    for fname, fval in frozen_params.items():
        fp = _check_priors.get(fname)
        if fp is None:
            continue
        if (fval <= fp['min']) or (fval >= fp['max']):
            warnings.warn(
                f"Frozen value {fname}={fval} lies outside prior box "
                f"({fp['min']}, {fp['max']}). Continuing because frozen values "
                f"are treated as fixed constants."
            )

    # Run MCMC or replot from existing results
    all_results = {}

    for band in bands:
        try:
            obs_df = load_observed_lightcurves(
                band, args.data_dir,
                flux_column=args.obs_column,
                error_column=args.obs_error_column,
                time_column=args.time_column,
            )

            is_binned = not args.no_phase_bin
            if not args.no_phase_bin:
                if args.counts_per_bin is not None:
                    obs_df = phase_bin_data_snr(
                        obs_df,
                        counts_per_bin=args.counts_per_bin,
                    )
                else:
                    obs_df = phase_bin_data(
                        obs_df,
                        n_bins=(args.n_phase_bins or 50),
                    )

            obs_phase = obs_df['phase'].values
            obs_flux = obs_df['flux'].values
            obs_err = obs_df['flux_err'].values
            obs_phase_width = (
                obs_df['width'].to_numpy(dtype=float)
                if ('width' in obs_df.columns and is_binned)
                else None
            )

            invalid_err = ~np.isfinite(obs_err) | (obs_err <= 0)
            if np.any(invalid_err):
                valid_err = obs_err[np.isfinite(obs_err) & (obs_err > 0)]
                if len(valid_err) > 0:
                    err_floor = float(np.median(valid_err))
                else:
                    err_floor = float(np.finfo(float).eps)
                obs_err[invalid_err] = np.maximum(
                    np.abs(obs_flux[invalid_err]) * 0.1, err_floor
                )
                warnings.warn(
                    f"Replaced {np.sum(invalid_err)} invalid errors with max(10% flux, median valid error)"
                )

            for wind_model in wind_models:
                try:
                    key = (band, wind_model)

                    if args.replot:
                        stats = replot_from_existing(
                            args, band, wind_model,
                            obs_phase, obs_flux, obs_err, obs_phase_width,
                            priors=priors,
                            sim_params=sim_params,
                            reparam=reparam,
                            kepler=kepler,
                            fit_wind_shape=fit_wind_shape,
                            shape_prior_overrides=shape_prior_overrides,
                            is_binned=is_binned,
                        )
                        if stats is not None:
                            all_results[key] = stats
                    else:
                        stats, _ = run_single_fit(
                            band, wind_model, args,
                            obs_phase, obs_flux, obs_err, obs_phase_width,
                            priors=priors,
                            sim_params=sim_params,
                            reparam=reparam,
                            kepler=kepler,
                            fit_wind_shape=fit_wind_shape,
                            shape_prior_overrides=shape_prior_overrides,
                            is_binned=is_binned,
                        )
                        all_results[key] = stats

                except Exception as e:
                    print(f"ERROR {'replotting' if args.replot else 'fitting'} "
                          f"{band} band ({wind_model}): {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        except Exception as e:
            print(f"ERROR loading data for {band} band: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary results
    if all_results:
        summary_path = os.path.join(args.output_dir, "mcmc_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("MCMC Light Curve Fitting Results\n")
            f.write("="*60 + "\n\n")
            bic_values = [
                float(v["bic"])
                for v in all_results.values()
                if isinstance(v, dict) and ("bic" in v) and np.isfinite(v["bic"])
            ]
            bic_best = min(bic_values) if bic_values else np.nan

            likelihood = getattr(args, 'likelihood', 'chi2')
            active_names, _ = get_param_config(
                likelihood, reparam=reparam,
                kepler=kepler,
                wind_model=args.wind_model, fit_wind_shape=fit_wind_shape,
                frozen=getattr(args, 'frozen_params', {}),
                orbital_period_s=float(getattr(args, 'orbital_period', ORBITAL_PERIOD)),
            )
            for key, stats in all_results.items():
                band, wind_model = key
                f.write(f"{band.upper()} Band - {WIND_MODELS[wind_model]}\n")
                f.write("-"*40 + "\n")

                run_meta = stats.get('_run_meta', {}) if isinstance(stats, dict) else {}
                if run_meta:
                    f.write("Run configuration:\n")
                    f.write(
                        f"  sampler={run_meta.get('sampler', 'unknown')}, "
                        f"likelihood={run_meta.get('likelihood', 'unknown')}, "
                        f"walkers={run_meta.get('n_walkers', 'n/a')}, "
                        f"steps={run_meta.get('n_steps', 'n/a')}, "
                        f"burn={run_meta.get('n_burn', 'n/a')}\n"
                    )
                    f.write(
                        f"  fit_phase_shift={run_meta.get('fit_phase_shift', False)}, "
                        f"phase_shift_grid={run_meta.get('phase_shift_grid_size', 'n/a')}, "
                        f"phase_eval_points={run_meta.get('phase_shift_eval_points', 'n/a')}\n"
                    )
                    if np.isfinite(run_meta.get('fit_elapsed_s', np.nan)):
                        f.write(f"  wall_time_s={run_meta['fit_elapsed_s']:.2f}\n")

                f.write("Marginal posterior (median +upper/-lower, 16/84 pct):\n")
                for param in active_names:
                    if param in stats:
                        s = stats[param]
                        f.write(f"  {param}: {s['median']:.6f} "
                                f"(+{s['upper']:.6f}/-{s['lower']:.6f})")
                        if ('mean' in s) and ('std' in s):
                            f.write(f"  [mean={s['mean']:.6f}, std={s['std']:.6f}]")
                        f.write("\n")
                if reparam or kepler:
                    for derived in ('d1', 'd2'):
                        if derived in stats:
                            s = stats[derived]
                            f.write(f"  {derived} (derived): {s['median']:.6f} "
                                    f"(+{s['upper']:.6f}/-{s['lower']:.6f})")
                            if ('mean' in s) and ('std' in s):
                                f.write(f"  [mean={s['mean']:.6f}, std={s['std']:.6f}]")
                            f.write("\n")

                # Self-consistent point estimate (single sample with max log-prob).
                has_map = any(
                    isinstance(stats.get(p), dict) and 'map' in stats[p]
                    for p in active_names
                )
                if has_map:
                    map_meta = stats.get('_map_meta', {})
                    lp = map_meta.get('log_prob')
                    lp_str = f"  (log_prob = {lp:.3f})" if lp is not None else ""
                    f.write(f"Best-fit (MAP, max log-prob){lp_str}:\n")
                    for param in active_names:
                        if param in stats and 'map' in stats[param]:
                            f.write(f"  {param}: {stats[param]['map']:.6f}\n")
                    if reparam or kepler:
                        for derived in ('d1', 'd2'):
                            if derived in stats and 'map' in stats[derived]:
                                f.write(
                                    f"  {derived} (derived): "
                                    f"{stats[derived]['map']:.6f}\n"
                                )
                    if (reparam or kepler) and all(k in stats for k in ('a', 'q', 'd1', 'd2')):
                        a_map = stats['a']['map']
                        d1_map = stats['d1']['map']
                        d2_map = stats['d2']['map']
                        f.write(
                            f"  check: d1+d2 = {d1_map + d2_map:.6f} "
                            f"(== a = {a_map:.6f}); "
                            f"d1/(d1+d2) = "
                            f"{d1_map / (d1_map + d2_map):.6f} "
                            f"(== q = {stats['q']['map']:.6f})\n"
                        )

                if 'reduced_chi2' in stats:
                    f.write(f"Reduced chi-square: {stats['reduced_chi2']:.3f}\n")
                if 'bic' in stats and np.isfinite(stats['bic']):
                    delta_bic = float(stats['bic'] - bic_best) if np.isfinite(bic_best) else np.nan
                    stats['delta_bic'] = delta_bic
                    f.write(
                        f"BIC: {stats['bic']:.3f} "
                        f"(ΔBIC={delta_bic:.3f}, logL_hat={stats.get('logL_hat', np.nan):.3f}, "
                        f"k={int(stats.get('k_params', np.nan)) if np.isfinite(stats.get('k_params', np.nan)) else 'nan'}, "
                        f"n={int(stats.get('n_obs', np.nan)) if np.isfinite(stats.get('n_obs', np.nan)) else 'nan'}, "
                        f"source={stats.get('theta_source', 'unknown')})\n"
                    )

                diag = stats.get('_diagnostics', {}) if isinstance(stats, dict) else {}
                if diag:
                    f.write("Chain diagnostics:\n")
                    if np.isfinite(diag.get('acceptance_fraction_mean', np.nan)):
                        f.write(f"  acceptance_fraction_mean: {diag['acceptance_fraction_mean']:.4f}\n")
                    if diag.get('autocorr_time'):
                        tau_vals = [
                            float(v) for v in diag['autocorr_time'].values()
                            if np.isfinite(v)
                        ]
                        if tau_vals:
                            f.write(
                                f"  autocorr_time_steps: min={np.min(tau_vals):.2f}, "
                                f"median={np.median(tau_vals):.2f}, max={np.max(tau_vals):.2f}\n"
                            )
                    if diag.get('effective_independent_samples') is not None:
                        f.write(
                            f"  effective_independent_samples: "
                            f"{int(diag['effective_independent_samples'])}\n"
                        )
                    if diag.get('converged') is not None:
                        f.write(f"  converged: {bool(diag['converged'])}\n")
                f.write("\n")
        
        print(f"\nSummary saved to: {summary_path}")
    
    if args.replot:
        print("\nReplotting complete!")
    else:
        print("\nMCMC fitting complete!")


if __name__ == "__main__":
    main()
