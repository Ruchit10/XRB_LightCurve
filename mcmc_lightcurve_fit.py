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

The precomputed model grid works for both geometry-only runs and wind-shape
fits.  In shape-fit mode it adds one axis per active wind-shape parameter
(configurable via --shape-grid-points, default 5); building the grid is
more expensive but subsequent likelihood evaluations are still ~O(1)
ND-interpolator calls. Pass --no-grid to use direct simulate_lightcurve
evaluation instead (~60 ms per LC with the Gauss-Legendre mega-kernel),
which is convenient for short debugging chains.

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

    # Use zeus sampler / Student-t / jitter as before
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
import copy
import glob
import multiprocessing as mp
import os
import time
import warnings
import pickle
from typing import Tuple, List, Dict, Optional
from multiprocessing import Pool, cpu_count
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

from scipy.interpolate import RegularGridInterpolator
from scipy.special import gammaln

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
)


def _init_numba_worker(max_numba_threads: int = 1):
    """Pool worker initializer: set Numba thread count inside each worker process.

    In `--no-grid` pooled mode, each worker runs full ``simulate_lightcurve``
    calls that already use ``parallel=True`` / ``prange`` in ``xrb_lightcurve``.

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


def _grid_priors_from_reparam(reparam_priors: Dict) -> Dict:
    """Derive ``(d1, d2, r, R, i0)`` grid bounds from ``(a, q)`` priors.

    The precomputed grid is always built in physical ``(d1, d2)`` space.
    This helper converts the reparameterized prior bounds into the
    corresponding physical-parameter bounds so the grid covers the
    region the MCMC chain will explore.
    """
    a_min = reparam_priors['a']['min']
    a_max = reparam_priors['a']['max']
    q_min = reparam_priors['q']['min']
    q_max = reparam_priors['q']['max']
    d1_min = a_min * q_min
    d1_max = a_max * q_max
    d2_min = a_min * (1.0 - q_max)
    d2_max = a_max * (1.0 - q_min)
    return {
        'd1': {'mean': (d1_min + d1_max) / 2, 'std': (d1_max - d1_min) / 4,
               'min': d1_min, 'max': d1_max},
        'd2': {'mean': (d2_min + d2_max) / 2, 'std': (d2_max - d2_min) / 4,
               'min': d2_min, 'max': d2_max},
        'r':  reparam_priors['r'].copy(),
        'R':  reparam_priors['R'].copy(),
        'i0': reparam_priors['i0'].copy(),
    }

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
    'studentt': 'Student-t (robust to outliers)',
}

JITTER_PRIOR = {'mean': -3.0, 'std': 2.0, 'min': -10.0, 'max': 0.0}

SAMPLER_TYPES = {
    'emcee': 'emcee Ensemble Sampler (stretch moves)',
    'zeus': 'zeus Ensemble Slice Sampler',
}


def get_param_config(
    likelihood: str = 'chi2',
    reparam: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
):
    """Return (param_names, param_labels) for the active MCMC vector.

    Layout:
        geometry (5) -> [log_f if jitter] -> wind-shape params (if requested)

    When *reparam* is True the first two geometric parameters are ``(a, q)``
    instead of ``(d1, d2)``.

    When *fit_wind_shape* is True the model-specific shape params from
    ``WIND_SHAPE_FIT[wind_model]`` are appended (in their listed order).
    """
    if reparam:
        names = list(REPARAM_PARAM_NAMES)
        labels = list(REPARAM_PARAM_LABELS)
    else:
        names = list(PARAM_NAMES)
        labels = list(PARAM_LABELS)
    if likelihood == 'jitter':
        names.append('log_f')
        labels.append(r'$\ln\,f$')
    if fit_wind_shape:
        if wind_model not in WIND_SHAPE_FIT:
            raise ValueError(
                f"--fit-wind-shape is not supported for wind_model "
                f"'{wind_model}'. Choose one of: {list(WIND_SHAPE_FIT)}"
            )
        for name in WIND_SHAPE_FIT[wind_model]:
            names.append(name)
            labels.append(WIND_SHAPE_LABELS.get(name, name))
    return names, labels


def get_active_priors(
    base_priors: Dict,
    wind_model: str,
    fit_wind_shape: bool,
    likelihood: str,
    shape_prior_overrides: Dict[str, Dict] = None,
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
    return out


def _to_wind_params(
    theta: np.ndarray,
    active_names: List[str],
    wind_model: str,
    R_value: float,
    fit_wind_shape: bool = False,
) -> Dict[str, float]:
    """Build the wind_params dict for simulate_lightcurve from a sample.

    Pulls fittable shape values from *theta* using their position in
    *active_names*; fills in fixed shape values from WIND_SHAPE_FIXED; and
    ties R_star to the geometry R for the beta_law / confinement models.
    """
    wp: Dict[str, float] = dict(WIND_SHAPE_FIXED.get(wind_model, {}))

    if fit_wind_shape:
        for name in WIND_SHAPE_FIT.get(wind_model, []):
            try:
                idx = active_names.index(name)
            except ValueError:
                # Shape param expected but not in chain; fall back to default
                wp[name] = float(WIND_SHAPE_PRIORS[name]['mean'])
                continue
            wp[name] = float(theta[idx])
    else:
        # Use prior means as the constant value of the would-be free params
        for name in WIND_SHAPE_FIT.get(wind_model, []):
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
    DataFrame with columns: time, flux, flux_err, phase, obs_id
    """
    band_dir = _resolve_band_directory(band, data_dir)
    print(f"Loading {band} band data from: {band_dir}")

    raw = _load_data_base(
        band_dir,
        obs_column=flux_column,
        obs_error_column=error_column,
        time_column=time_column,
    )

    combined = pd.DataFrame({
        'time': raw['time'].astype(float),
        'flux': raw['rate'].astype(float),
        'flux_err': raw['error'].astype(float) if 'error' in raw.columns else np.nan,
        'obs_id': raw['obs'] if 'obs' in raw.columns else 'data',
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


# =============================================================================
# Pre-computed Model Grid (FAST)
# =============================================================================

def _compute_single_model(args):
    """Worker function to compute a single grid model (for parallel processing).

    Returns the band-flux array sorted by phase, or None on failure.

    Expects ``wind_params`` to be fully resolved by the caller (fixed shape
    params + per-point shape-axis values + R_star tied to R when needed).
    """
    (
        d1, d2, r, R, i0,
        flux_csv_path, band, dth,
        sim_params, wind_model, wind_params,
    ) = args

    flux_column = f"nfl_{band}"

    try:
        results = simulate_lightcurve(
            r=r, R=R, d1=d1, d2=d2,
            gma0=sim_params.get('gma0', -90.0),
            i0=i0,
            dth=dth,
            d2h=sim_params.get('d2h', 6.0),
            dz=sim_params.get('dz', 0.5),
            flux_method="interpolate",
            flux_csv_path=flux_csv_path,
            lam=sim_params.get('lam', 0.589537),
            wind_model=wind_model,
            wind_params=wind_params,
            verbose=False,
        )

        if flux_column not in results.columns:
            return None

        model_phase = results['phase'].values
        flux = results[flux_column].values
        sort_idx = np.argsort(model_phase)
        return flux[sort_idx]

    except Exception:
        return None


class PrecomputedModelGrid:
    """Pre-computed grid of model light curves for fast MCMC evaluation.

    Supports two modes, selected at construction time:

    * **Geometry-only** (default, ``fit_wind_shape=False``) — the grid axes
      are the 5 geometry parameters ``(d1, d2, r, R, i0)``.  Wind-shape
      parameters are held fixed at ``wind_params_template`` (or
      ``default_wind_params`` if not provided).  Fast to build, small on
      disk, cannot represent runs where shape parameters vary.

    * **Geometry + wind-shape** (``fit_wind_shape=True``) — the active
      wind-shape parameters for ``wind_model`` (``WIND_SHAPE_FIT[wind_model]``)
      are added as extra grid axes.  Grid size grows multiplicatively
      (~5^k for k shape axes at ``shape_grid_points=5`` each), but once
      built every likelihood evaluation is still a single ND interpolation.
      Much faster than ``DirectLightCurveModel`` for long chains.
    """

    def __init__(
        self,
        band: str,
        flux_csv_path: str,
        wind_model: str = 'smooth_pl',
        priors: Dict = DEFAULT_PRIORS,
        grid_points: Dict[str, int] = None,
        dth: float = 5.0,
        n_workers: int = None,
        verbose: bool = True,
        load_path: str = None,
        sim_params: Dict = None,
        wind_params_template: Dict[str, float] = None,
        fit_wind_shape: bool = False,
        shape_priors: Dict[str, Dict] = None,
        shape_grid_points: int = 5,
    ):
        """Initialize and pre-compute the model grid (or load from file).

        Parameters
        ----------
        band, flux_csv_path, wind_model, priors, grid_points, dth,
        n_workers, verbose, load_path, sim_params, wind_params_template
            See class docstring / previous signature.
        fit_wind_shape : bool
            If True, include ``WIND_SHAPE_FIT[wind_model]`` as additional
            grid axes so the grid can be used for shape-parameter MCMC.
        shape_priors : dict, optional
            ``{shape_name: {min, max, ...}}``.  Only ``min`` and ``max`` are
            used (to bound each shape grid axis).  Defaults to
            ``WIND_SHAPE_PRIORS``.
        shape_grid_points : int
            Number of grid points per wind-shape axis.  Default 5.  Can be
            overridden per-axis by adding entries to ``grid_points``.
        """
        self.band = band.lower()
        self.flux_csv_path = flux_csv_path
        self.wind_model = wind_model.lower()
        self.priors = priors
        self.dth = dth
        self.verbose = verbose
        self.sim_params = sim_params or {}
        self.fit_wind_shape = bool(fit_wind_shape)
        self.shape_priors = shape_priors if shape_priors is not None else WIND_SHAPE_PRIORS
        self.shape_grid_points = int(shape_grid_points)

        if self.wind_model not in WIND_MODELS:
            raise ValueError(
                f"wind_model must be one of {list(WIND_MODELS)}, "
                f"got '{wind_model}'"
            )

        # Build the wind_params template (fixed shape params) if none provided.
        # This holds shape params that are *constant* across the grid, even
        # when fit_wind_shape=True (e.g. Delta for smooth_pl, H for beta_law).
        if wind_params_template is None:
            wp = default_wind_params(self.wind_model, R=2.0)
            wp.pop('R_star', None)  # R_star is tied to R per-point.
            wind_params_template = wp
        self.wind_params_template = dict(wind_params_template)

        # Shape axes that become grid dimensions.
        if self.fit_wind_shape:
            self.shape_axes: List[str] = list(WIND_SHAPE_FIT.get(self.wind_model, []))
            # Fixed shape params for this model (not on a grid axis) come
            # from WIND_SHAPE_FIXED, overlaid by anything the caller supplied
            # in wind_params_template.
            fixed = dict(WIND_SHAPE_FIXED.get(self.wind_model, {}))
            fixed.update(self.wind_params_template)
            self.wind_params_template = fixed
        else:
            self.shape_axes = []

        self.axis_names: List[str] = list(PARAM_NAMES) + list(self.shape_axes)

        # Default grid resolution
        if grid_points is None:
            grid_points = {'d1': 8, 'd2': 8, 'r': 5, 'R': 8, 'i0': 10}
        grid_points = dict(grid_points)
        for sname in self.shape_axes:
            grid_points.setdefault(sname, self.shape_grid_points)
        self.grid_points = grid_points
        self.n_workers = n_workers if n_workers else max(1, cpu_count() - 1)

        if load_path and os.path.exists(load_path):
            self._load_grid(load_path)
        else:
            self._create_grids()
            self._precompute_models()

        self._setup_interpolators()

    def _create_grids(self):
        """Create 1D parameter grids for geometry + (optional) shape axes."""
        self.param_grids: Dict[str, np.ndarray] = {}

        for param in PARAM_NAMES:
            p_min = self.priors[param]['min']
            p_max = self.priors[param]['max']
            n_pts = self.grid_points.get(param, 8)
            if param == 'r':
                self.param_grids[param] = np.logspace(
                    np.log10(p_min), np.log10(p_max), n_pts
                )
            else:
                self.param_grids[param] = np.linspace(p_min, p_max, n_pts)

        for sname in self.shape_axes:
            sp = self.shape_priors[sname]
            n_pts = self.grid_points.get(sname, self.shape_grid_points)
            self.param_grids[sname] = np.linspace(sp['min'], sp['max'], n_pts)

        # Standard phase grid for output
        n_phase_points = int(360 / self.dth)
        self.phase_grid = np.linspace(0, 1 - 1 / n_phase_points, n_phase_points)

        if self.verbose:
            total_models = int(np.prod([len(g) for g in self.param_grids.values()]))
            print(f"\nGrid configuration: {total_models} models to compute")
            print(f"Wind model: {WIND_MODELS[self.wind_model]} ({self.wind_model})")
            print(f"Axes: {self.axis_names}")
            if self.shape_axes:
                print(f"  (including {len(self.shape_axes)} wind-shape axes: "
                      f"{self.shape_axes})")
            for param in self.axis_names:
                grid = self.param_grids[param]
                print(f"  {param}: {len(grid)} points [{grid[0]:.4f} - {grid[-1]:.4f}]")
    
    def _precompute_models(self):
        """Pre-compute all models on the (geometry + shape) grid."""
        if self.verbose:
            print(f"\nPre-computing model grid using {self.n_workers} workers...")
            start_time = time.time()

        axis_arrays = [self.param_grids[name] for name in self.axis_names]
        axis_lens = [len(a) for a in axis_arrays]

        # Geometry indices used to skip r >= R combinations.
        try:
            idx_r = self.axis_names.index('r')
            idx_R = self.axis_names.index('R')
        except ValueError:
            idx_r = idx_R = None

        param_combos = []
        valid_index_tuples = []
        for index in np.ndindex(*axis_lens):
            values = [axis_arrays[k][i] for k, i in enumerate(index)]
            if idx_r is not None and values[idx_r] >= values[idx_R]:
                continue
            d1, d2, r, R, i0 = values[:5]
            shape_values = values[5:]

            # Build the wind_params for this grid point.
            wp = dict(self.wind_params_template)
            for sname, sval in zip(self.shape_axes, shape_values):
                wp[sname] = float(sval)
            if self.wind_model in ('beta_law', 'confinement'):
                wp['R_star'] = float(R)

            param_combos.append((
                d1, d2, r, R, i0,
                self.flux_csv_path, self.band, self.dth,
                self.sim_params,
                self.wind_model,
                wp,
            ))
            valid_index_tuples.append(index)

        if self.verbose:
            print(f"Computing {len(param_combos)} valid parameter combinations...")

        if self.n_workers > 1:
            with Pool(self.n_workers) as pool:
                if HAS_TQDM:
                    results = list(tqdm(
                        pool.imap(_compute_single_model, param_combos),
                        total=len(param_combos),
                        desc="Building model grid"
                    ))
                else:
                    results = pool.map(_compute_single_model, param_combos)
        else:
            if HAS_TQDM:
                results = [_compute_single_model(args)
                           for args in tqdm(param_combos, desc="Building model grid")]
            else:
                results = [_compute_single_model(args) for args in param_combos]

        shape = tuple(axis_lens) + (len(self.phase_grid),)
        self.flux_grid = np.full(shape, np.nan)

        for index, flux in zip(valid_index_tuples, results):
            if flux is not None:
                self.flux_grid[index + (slice(None),)] = flux

        if self.verbose:
            elapsed = time.time() - start_time
            print(f"Grid pre-computation completed in {elapsed:.1f} seconds")
            mem_mb = self.flux_grid.nbytes / (1024 * 1024)
            print(f"Grid shape: {self.flux_grid.shape} ({mem_mb:.1f} MB in RAM)")
            valid = np.sum(~np.isnan(self.flux_grid)) / self.flux_grid.size
            print(f"Valid grid coverage: {valid*100:.1f}%")

    def _setup_interpolators(self):
        """Set up a single ND interpolator over (axes..., phase)."""
        flux_grid_clean = self.flux_grid.copy()
        if np.any(np.isnan(flux_grid_clean)):
            flux_grid_clean = np.where(
                np.isnan(flux_grid_clean),
                np.nanmean(flux_grid_clean),
                flux_grid_clean,
            )

        interp_axes = tuple(self.param_grids[name] for name in self.axis_names) \
            + (self.phase_grid,)
        self._interp_nd = RegularGridInterpolator(
            interp_axes,
            flux_grid_clean,
            method='linear',
            bounds_error=False,
            fill_value=np.nan,
        )
        # Backwards-compat alias (older code paths reference _interp_6d).
        self._interp_6d = self._interp_nd
    
    def save(self, filepath: str):
        """
        Save pre-computed grid to file for later reuse.

        Parameters
        ----------
        filepath : str
            Path to save the grid (.npz format)
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        sim_param_keys = list(self.sim_params.keys()) if self.sim_params else []
        sim_param_vals = [self.sim_params[k] for k in sim_param_keys] if self.sim_params else []

        wp_keys = list(self.wind_params_template.keys())
        wp_vals = [float(self.wind_params_template[k]) for k in wp_keys]

        save_kwargs = dict(
            flux_grid=self.flux_grid,
            phase_grid=self.phase_grid,
            band=self.band,
            wind_model=self.wind_model,
            dth=self.dth,
            axis_names=np.array(self.axis_names, dtype=str),
            shape_axes=np.array(self.shape_axes, dtype=str),
            fit_wind_shape=np.array(self.fit_wind_shape),
            sim_param_keys=np.array(sim_param_keys, dtype=str),
            sim_param_vals=np.array(sim_param_vals, dtype=float),
            wind_param_keys=np.array(wp_keys, dtype=str),
            wind_param_vals=np.array(wp_vals, dtype=float),
        )
        # Persist every axis as "<name>_grid" so loading can reconstruct the
        # full set regardless of how many shape axes are present.
        for name in self.axis_names:
            save_kwargs[f"{name}_grid"] = self.param_grids[name]
        grid_pts = np.array(
            [self.grid_points.get(p, len(self.param_grids[p])) for p in self.axis_names]
        )
        save_kwargs['grid_points'] = grid_pts

        np.savez_compressed(filepath, **save_kwargs)

        print(f"Grid saved to: {filepath}")
        print(f"  File size: {os.path.getsize(filepath) / 1024 / 1024:.1f} MB")
        if self.sim_params:
            print(f"  Simulation params: {self.sim_params}")
        print(f"  Axes: {self.axis_names}")
        print(f"  Wind model: {self.wind_model} | wind_params: {self.wind_params_template}")

    def _load_grid(self, filepath: str):
        """Load pre-computed grid from file (supports legacy geometry-only grids)."""
        if self.verbose:
            print(f"\nLoading pre-computed grid from: {filepath}")

        data = np.load(filepath, allow_pickle=True)

        self.flux_grid = data['flux_grid']
        self.phase_grid = data['phase_grid']

        # Determine the loaded axis layout.  New grids write an explicit
        # ``axis_names`` entry.  Legacy grids had only the 5 geometry axes.
        if 'axis_names' in data:
            loaded_axis_names = [str(x) for x in data['axis_names']]
        else:
            loaded_axis_names = list(PARAM_NAMES)

        if 'shape_axes' in data:
            loaded_shape_axes = [str(x) for x in data['shape_axes'] if str(x)]
        else:
            loaded_shape_axes = [
                n for n in loaded_axis_names if n not in PARAM_NAMES
            ]
        loaded_fit_wind_shape = bool(loaded_shape_axes)
        if 'fit_wind_shape' in data:
            try:
                loaded_fit_wind_shape = bool(data['fit_wind_shape'])
            except Exception:
                pass

        self.param_grids = {}
        for name in loaded_axis_names:
            key = f"{name}_grid"
            if key in data:
                self.param_grids[name] = data[key]
            else:
                # Legacy fallback: some older grids stored lowercase for R.
                alt = f"{name.lower()}_grid"
                if alt in data:
                    self.param_grids[name] = data[alt]

        loaded_band = str(data['band'])
        loaded_dth = float(data['dth'])
        loaded_wind_model = (
            str(data['wind_model']) if 'wind_model' in data else self.wind_model
        )

        loaded_sim_params = {}
        if 'sim_param_keys' in data and 'sim_param_vals' in data:
            keys = data['sim_param_keys']
            vals = data['sim_param_vals']
            if len(keys) > 0:
                loaded_sim_params = dict(zip(keys, vals))

        loaded_wp = {}
        if 'wind_param_keys' in data and 'wind_param_vals' in data:
            wpk = data['wind_param_keys']
            wpv = data['wind_param_vals']
            if len(wpk) > 0:
                loaded_wp = {str(k): float(v) for k, v in zip(wpk, wpv)}

        if loaded_band != self.band:
            warnings.warn(
                f"Loaded grid band '{loaded_band}' differs from requested '{self.band}'"
            )
        if loaded_wind_model != self.wind_model:
            warnings.warn(
                f"Loaded grid wind_model '{loaded_wind_model}' differs from requested "
                f"'{self.wind_model}'. Using loaded value."
            )
        if self.fit_wind_shape != loaded_fit_wind_shape:
            warnings.warn(
                f"Loaded grid fit_wind_shape={loaded_fit_wind_shape} differs from "
                f"requested fit_wind_shape={self.fit_wind_shape}. Using loaded value."
            )

        if self.sim_params and loaded_sim_params:
            for key in self.sim_params:
                if key in loaded_sim_params and self.sim_params[key] != loaded_sim_params[key]:
                    warnings.warn(
                        f"Loaded grid has {key}={loaded_sim_params[key]}, "
                        f"but requested {key}={self.sim_params[key]}. Using loaded value."
                    )

        self.dth = loaded_dth
        self.sim_params = loaded_sim_params
        self.wind_model = loaded_wind_model
        self.axis_names = loaded_axis_names
        self.shape_axes = loaded_shape_axes
        self.fit_wind_shape = loaded_fit_wind_shape
        if loaded_wp:
            self.wind_params_template = loaded_wp

        if self.verbose:
            print(f"  Band: {loaded_band}, dth: {loaded_dth}, wind_model: {loaded_wind_model}")
            print(f"  Axes: {self.axis_names} (fit_wind_shape={self.fit_wind_shape})")
            print(f"  Grid shape: {self.flux_grid.shape}")
            valid = np.sum(~np.isnan(self.flux_grid)) / self.flux_grid.size
            print(f"  Valid coverage: {valid*100:.1f}%")
            if loaded_sim_params:
                print(f"  Simulation params: {loaded_sim_params}")
            if loaded_wp:
                print(f"  Wind params (template): {loaded_wp}")
    
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
        """Evaluate the model at given parameters using grid interpolation.

        If this grid includes wind-shape axes (``fit_wind_shape=True`` at
        construction time), ``wind_params`` must supply the corresponding
        shape-parameter values.  Otherwise the grid's fixed template values
        are used and ``wind_params`` is ignored.
        """
        n_phase = len(self.phase_grid)
        n_axes = len(self.axis_names)
        points = np.empty((n_phase, n_axes + 1))
        points[:, 0] = d1
        points[:, 1] = d2
        points[:, 2] = r
        points[:, 3] = R
        points[:, 4] = i0

        for k, sname in enumerate(self.shape_axes, start=5):
            if wind_params is not None and sname in wind_params:
                points[:, k] = float(wind_params[sname])
            else:
                # Fall back to the grid's fixed template value if present,
                # otherwise use the axis midpoint as a safe default.
                fallback = self.wind_params_template.get(sname)
                if fallback is None:
                    axis = self.param_grids[sname]
                    fallback = 0.5 * (axis[0] + axis[-1])
                points[:, k] = float(fallback)

        points[:, -1] = self.phase_grid

        grid_flux = self._interp_nd(points)

        if np.any(np.isnan(grid_flux)):
            return np.full_like(obs_phases, np.nan, dtype=float)

        phase_extended = np.concatenate([
            self.phase_grid - 1,
            self.phase_grid,
            self.phase_grid + 1,
        ])
        flux_extended = np.concatenate([grid_flux, grid_flux, grid_flux])

        return np.interp(obs_phases, phase_extended, flux_extended)

    def evaluate_direct(
        self,
        d1: float, d2: float, r: float, R: float, i0: float,
        obs_phases: np.ndarray,
        wind_params: Dict[str, float] = None,
    ) -> np.ndarray:
        """Evaluate the model by running ``simulate_lightcurve`` directly.

        Unlike :meth:`evaluate`, this bypasses the pre-computed grid and
        produces the exact model curve.  Use this for final best-fit
        evaluation and chi-square reporting.
        """
        flux_column = f"nfl_{self.band}"

        if wind_params is None:
            wp = dict(self.wind_params_template)
            if self.wind_model in ('beta_law', 'confinement'):
                wp['R_star'] = float(R)
        else:
            wp = dict(wind_params)

        try:
            results = simulate_lightcurve(
                r=r, R=R, d1=d1, d2=d2,
                gma0=self.sim_params.get('gma0', -90.0),
                i0=i0,
                dth=self.dth,
                d2h=self.sim_params.get('d2h', 6.0),
                dz=self.sim_params.get('dz', 0.5),
                flux_method="interpolate",
                flux_csv_path=self.flux_csv_path,
                lam=self.sim_params.get('lam', 0.589537),
                wind_model=self.wind_model,
                wind_params=wp,
                verbose=False,
            )
        except Exception as e:
            warnings.warn(f"Direct model evaluation failed: {e}")
            return np.full_like(obs_phases, np.nan)

        if flux_column not in results.columns:
            return np.full_like(obs_phases, np.nan)

        model_phase = results['phase'].values
        model_flux = results[flux_column].values
        sort_idx = np.argsort(model_phase)

        phase_ext = np.concatenate([
            model_phase[sort_idx] - 1,
            model_phase[sort_idx],
            model_phase[sort_idx] + 1,
        ])
        flux_ext = np.tile(model_flux[sort_idx], 3)

        return np.interp(obs_phases, phase_ext, flux_ext)


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

        sort_idx = np.argsort(model_phase)
        model_phase_sorted = model_phase[sort_idx]
        model_flux_sorted = model_flux[sort_idx]

        phase_extended = np.concatenate([
            model_phase_sorted - 1,
            model_phase_sorted,
            model_phase_sorted + 1,
        ])
        flux_extended = np.concatenate([
            model_flux_sorted,
            model_flux_sorted,
            model_flux_sorted,
        ])

        return np.interp(obs_phases, phase_extended, flux_extended)


# =============================================================================
# Likelihood Functions
# =============================================================================

def _to_physical(theta, reparam: bool = False):
    """Convert sampling-space parameters to physical ``(d1, d2, r, R, i0)``."""
    if reparam:
        a, q = theta[0], theta[1]
        return a * q, a * (1.0 - q), theta[2], theta[3], theta[4]
    return theta[0], theta[1], theta[2], theta[3], theta[4]


def _evaluate_model(
    theta,
    model,
    obs_phase,
    reparam: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    active_names: List[str] = None,
):
    """Evaluate the physical model. Returns model_flux or None on failure.

    When *fit_wind_shape* is True, the wind-shape parameters are pulled out
    of *theta* (using their position in *active_names*) and passed inside
    *wind_params* to ``model.evaluate``. When False, the model uses its
    constructor-time defaults.
    """
    d1, d2, r, R, i0 = _to_physical(theta, reparam=reparam)

    # Build wind_params only if we need shape-varying behavior. Skipping the
    # dict construction in the geometry-only fast path avoids ~1 us / call.
    if fit_wind_shape and active_names is not None:
        wind_params = _to_wind_params(
            theta, active_names, wind_model, R_value=R,
            fit_wind_shape=True,
        )
    else:
        wind_params = None

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
) -> float:
    """Standard Gaussian log-likelihood (chi-squared)."""
    model_flux = _evaluate_model(
        theta, model, obs_phase, reparam=reparam,
        wind_model=wind_model, fit_wind_shape=fit_wind_shape,
        active_names=active_names,
    )
    if model_flux is None:
        return -np.inf
    chi2 = np.sum(((obs_flux - model_flux) / obs_err) ** 2)
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
) -> float:
    """Gaussian log-likelihood with a free fractional systematic error term.

    The position of the ``log_f`` parameter is looked up from *active_names*.
    The effective variance per point is  sigma_obs^2 + (f * model)^2
    where f = exp(log_f).
    """
    model_flux = _evaluate_model(
        theta, model, obs_phase, reparam=reparam,
        wind_model=wind_model, fit_wind_shape=fit_wind_shape,
        active_names=active_names,
    )
    if model_flux is None:
        return -np.inf
    if active_names is not None and 'log_f' in active_names:
        idx_logf = active_names.index('log_f')
    else:
        idx_logf = 5  # legacy default
    f = np.exp(theta[idx_logf])
    sigma2 = obs_err ** 2 + (f * model_flux) ** 2
    return -0.5 * np.sum((obs_flux - model_flux) ** 2 / sigma2 + np.log(sigma2))


def log_likelihood_studentt(
    theta: np.ndarray,
    model,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    nu: float = 5.0,
    reparam: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    active_names: List[str] = None,
) -> float:
    """Student-t log-likelihood (heavier tails than Gaussian).

    ``nu`` controls tail weight; nu -> inf recovers the Gaussian.
    """
    model_flux = _evaluate_model(
        theta, model, obs_phase, reparam=reparam,
        wind_model=wind_model, fit_wind_shape=fit_wind_shape,
        active_names=active_names,
    )
    if model_flux is None:
        return -np.inf
    resid = (obs_flux - model_flux) / obs_err
    ll = (
        gammaln(0.5 * (nu + 1))
        - gammaln(0.5 * nu)
        - 0.5 * np.log(nu * np.pi)
        - np.log(obs_err)
        - 0.5 * (nu + 1) * np.log(1.0 + resid ** 2 / nu)
    )
    return float(np.sum(ll))


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
    pnames = REPARAM_PARAM_NAMES if reparam else PARAM_NAMES

    if active_names is None:
        active_names = list(pnames)
        if likelihood == 'jitter':
            active_names.append('log_f')

    # Box check on every active parameter (geometry + jitter + shape).
    for name, value in zip(active_names, theta):
        prior = priors.get(name)
        if prior is None:
            # No prior provided for this dim; treat as improper / skip box.
            continue
        if not (prior['min'] < value < prior['max']):
            return -np.inf

    # Physical constraint r < R (indices 2, 3 regardless of reparam).
    if theta[2] >= theta[3]:
        return -np.inf

    log_p = 0.0

    # Gaussian penalty on geometry parameters.
    for i, name in enumerate(pnames):
        prior = priors.get(name)
        if prior is None:
            continue
        log_p += -0.5 * ((theta[i] - prior['mean']) / prior['std']) ** 2

    # Jacobian |d(d1,d2)/d(a,q)| = a.
    if reparam:
        log_p += np.log(theta[0])

    # Gaussian penalty on jitter and shape parameters (anything beyond geometry).
    for i, name in enumerate(active_names):
        if i < len(pnames):
            continue
        prior = priors.get(name)
        if prior is None:
            continue
        log_p += -0.5 * ((theta[i] - prior['mean']) / prior['std']) ** 2

    return log_p


def log_probability(
    theta: np.ndarray,
    model,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    priors: Dict = DEFAULT_PRIORS,
    likelihood: str = 'chi2',
    studentt_nu: float = 5.0,
    reparam: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    active_names: List[str] = None,
) -> float:
    """Log posterior = log prior + log likelihood."""
    lp = log_prior(
        theta, priors, likelihood=likelihood, reparam=reparam,
        active_names=active_names,
    )
    if not np.isfinite(lp):
        return -np.inf

    common_kwargs = dict(
        reparam=reparam,
        wind_model=wind_model,
        fit_wind_shape=fit_wind_shape,
        active_names=active_names,
    )
    if likelihood == 'jitter':
        ll = log_likelihood_jitter(
            theta, model, obs_phase, obs_flux, obs_err, **common_kwargs,
        )
    elif likelihood == 'studentt':
        ll = log_likelihood_studentt(
            theta, model, obs_phase, obs_flux, obs_err,
            nu=studentt_nu, **common_kwargs,
        )
    else:
        ll = log_likelihood_chi2(
            theta, model, obs_phase, obs_flux, obs_err, **common_kwargs,
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
    studentt_nu: float = 5.0,
    reparam: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    numba_threads_per_worker: Optional[int] = None,
) -> Tuple:
    """
    Run MCMC sampling with emcee or zeus.
    
    Parameters
    ----------
    model : PrecomputedModelGrid or DirectLightCurveModel
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
        'chi2', 'jitter', or 'studentt'
    studentt_nu : float
        Degrees-of-freedom for Student-t likelihood
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
    active_names, active_labels = get_param_config(
        likelihood, reparam=reparam,
        wind_model=wind_model, fit_wind_shape=fit_wind_shape,
    )
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

    # Enforce r < R (indices 2, 3 regardless of reparam).
    for j in range(n_walkers):
        if pos[j, 2] >= pos[j, 3]:
            pos[j, 2] = pos[j, 3] * 0.1

    # Auto-disable multiprocessing when using a PrecomputedModelGrid.
    # Each evaluate() call is a microsecond-scale RegularGridInterpolator
    # lookup, while the grid itself can be hundreds of MB to several GB.
    # On macOS (and any system using the 'spawn' start method) multiprocessing
    # would pickle and ship the entire grid to every worker on every step,
    # which dwarfs the actual compute and makes the sampler appear hung.
    if n_threads > 1 and isinstance(model, PrecomputedModelGrid):
        try:
            grid_bytes = model.flux_grid.nbytes
        except Exception:
            grid_bytes = 0
        size_str = (
            f" (~{grid_bytes / 1e6:.0f} MB flux_grid)"
            if grid_bytes else ""
        )
        print(
            f"\n[notice] --n-threads={n_threads} is being ignored because the "
            f"model is a PrecomputedModelGrid{size_str}. Per-step lookups are "
            f"already microsecond-scale, and multiprocessing would have to "
            f"pickle/ship the entire grid to every worker on every step "
            f"(this is what makes the progress bar appear stuck). "
            f"Falling back to serial MCMC. "
            f"For multi-process speedups, use --no-grid (DirectLightCurveModel)."
        )
        n_threads = 1

    parallel_info = f", {n_threads} threads" if n_threads > 1 else " (serial)"
    print(f"\nStarting MCMC ({sampler_type}) with {n_walkers} walkers, "
          f"{n_steps} steps{parallel_info}")
    if reparam:
        print("Reparameterization: (d1, d2) -> (a = d1+d2, q = d1/a)")
    print(f"Likelihood: {LIKELIHOOD_TYPES[likelihood]}")
    print(f"Wind model: {WIND_MODELS.get(wind_model, wind_model)} "
          f"({wind_model})  | fit_wind_shape={fit_wind_shape}")
    print(f"Active params ({n_dim}): {active_names}")
    print(f"Initial parameter values (first walker): {pos[0]}")

    log_prob_args = (
        model, obs_phase, obs_flux, obs_err,
        priors, likelihood, studentt_nu, reparam,
        wind_model, fit_wind_shape, active_names,
    )

    start_time = time.time()

    using_direct_model = isinstance(model, DirectLightCurveModel)
    using_pool = n_threads > 1 and using_direct_model

    if n_threads > 1 and not using_direct_model:
        print(
            f"\n[notice] --n-threads={n_threads} requested with model type "
            f"{type(model).__name__}. Pooling is only enabled for "
            f"DirectLightCurveModel (--no-grid); running serial."
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

    if reparam:
        a_samples = samples[:, 0]
        q_samples = samples[:, 1]
        for derived_name, derived_vals in [
            ('d1', a_samples * q_samples),
            ('d2', a_samples * (1.0 - q_samples)),
        ]:
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
            if reparam:
                a_map = float(map_sample[0])
                q_map = float(map_sample[1])
                stats['d1']['map'] = a_map * q_map
                stats['d2']['map'] = a_map * (1.0 - q_map)
            stats['_map_meta'] = {
                'index': map_idx,
                'log_prob': float(log_prob[map_idx]),
            }

    return stats


def load_existing_results(
    output_dir: str,
    band: str,
    wind_model: str,
    reparam: bool = False,
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
        samples, param_names=loaded_names, reparam=reparam,
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
) -> None:
    """
    Compute chi-square for all (or a subset of) MCMC samples and save to compressed file.
    
    Parameters
    ----------
    model : PrecomputedModelGrid or DirectLightCurveModel
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

    if HAS_TQDM and verbose:
        iterator = tqdm(sample_indices, desc="Computing χ²")
    else:
        iterator = sample_indices

    for idx in iterator:
        sample_params = samples[idx]
        d1, d2, r, R, i0 = _to_physical(sample_params, reparam=reparam)

        if fit_wind_shape and active_names is not None:
            wp = _to_wind_params(
                sample_params, active_names, wind_model, R_value=R,
                fit_wind_shape=True,
            )
        else:
            wp = None

        try:
            try:
                model_flux = model.evaluate(
                    d1, d2, r, R, i0, obs_phase, wind_params=wp,
                )
            except TypeError:
                model_flux = model.evaluate(d1, d2, r, R, i0, obs_phase)

            if np.all(np.isfinite(model_flux)):
                if likelihood == 'jitter' and active_names is not None and 'log_f' in active_names:
                    idx_logf = active_names.index('log_f')
                    f = np.exp(sample_params[idx_logf])
                    sigma2 = obs_err ** 2 + (f * model_flux) ** 2
                    sigma2 = np.maximum(sigma2, np.finfo(float).eps)
                    chi2 = np.sum((obs_flux - model_flux) ** 2 / sigma2)
                else:
                    chi2 = np.sum(((obs_flux - model_flux) / obs_err) ** 2)
                red_chi2 = chi2 / dof if dof > 0 else np.nan
            else:
                chi2 = np.nan
                red_chi2 = np.nan
        except Exception:
            chi2 = np.nan
            red_chi2 = np.nan

        results.append([idx, d1, d2, r, R, i0, chi2, red_chi2])

    columns = ['sample_idx', 'd1', 'd2', 'r', 'R', 'i0', 'chi2', 'reduced_chi2']
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

    if reparam:
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
    stats: Dict,
    band: str,
    wind_model: str,
    output_path: str,
    param_names: List[str] = None,
    reparam: bool = False,
    fit_wind_shape: bool = False,
    is_binned: bool = True,
    likelihood: str = 'chi2',
):
    """
    Plot observed data with best-fit model overlay.

    Parameters
    ----------
    model : PrecomputedModelGrid or DirectLightCurveModel
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

    if reparam:
        best_d1 = stats['d1'][point_key]
        best_d2 = stats['d2'][point_key]
        best_params = [best_d1, best_d2, stats['r'][point_key],
                       stats['R'][point_key], stats['i0'][point_key]]
    else:
        best_params = [stats[p][point_key] for p in PARAM_NAMES]

    # Build best-fit wind_params if shape was fitted.
    best_R = best_params[3]
    if fit_wind_shape and wind_model in WIND_SHAPE_FIT:
        best_wp = dict(WIND_SHAPE_FIXED.get(wind_model, {}))
        for sname in WIND_SHAPE_FIT[wind_model]:
            if sname in stats:
                best_wp[sname] = float(stats[sname][point_key])
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

    try:
        obs_model = eval_fn(*best_params, obs_phase, wind_params=best_wp)
    except TypeError:
        obs_model = eval_fn(*best_params, obs_phase)
    f_best = None
    if likelihood == 'jitter' and 'log_f' in stats and point_key in stats['log_f']:
        f_best = float(np.exp(stats['log_f'][point_key]))
        sigma2 = obs_err ** 2 + (f_best * obs_model) ** 2
        sigma2 = np.maximum(sigma2, np.finfo(float).eps)
        chi2 = np.sum((obs_flux - obs_model) ** 2 / sigma2)
    else:
        chi2 = np.sum(((obs_flux - obs_model) / obs_err) ** 2)
    n_phys = len(param_names) - (1 if 'log_f' in param_names else 0)
    dof = len(obs_flux) - n_phys
    red_chi2 = chi2 / dof if dof > 0 else np.nan

    fig, ax = plt.subplots(figsize=(10, 6))

    if is_binned:
        ax.errorbar(
            obs_phase, obs_flux, yerr=obs_err, fmt='o',
            markersize=4, alpha=0.7, label='Observed (phase-binned)',
            capsize=2, elinewidth=1, color='C0', zorder=5
        )
    else:
        ax.scatter(
            obs_phase, obs_flux, s=10, alpha=0.25, color='C0',
            label='Observed (raw 100s)', zorder=4
        )

    ax.plot(model_phases, model_flux, 'r-', lw=2,
            label=f'Best-fit model ({WIND_MODELS[wind_model]})', zorder=10)

    if (not is_binned) and likelihood == 'jitter' and (f_best is not None):
        phase_mod = np.mod(obs_phase, 1.0)
        edges = np.linspace(0.0, 1.0, 121)
        centers = 0.5 * (edges[:-1] + edges[1:])
        idx = np.digitize(phase_mod, edges) - 1
        idx = np.clip(idx, 0, len(centers) - 1)
        sigma_repr = np.full_like(centers, np.nan, dtype=float)
        for j in range(len(centers)):
            in_bin = idx == j
            if np.any(in_bin):
                sigma_repr[j] = np.nanmedian(obs_err[in_bin])
        global_sigma = np.nanmedian(obs_err[np.isfinite(obs_err) & (obs_err > 0)])
        if not np.isfinite(global_sigma):
            global_sigma = np.finfo(float).eps
        sigma_repr = np.where(np.isfinite(sigma_repr), sigma_repr, global_sigma)
        sigma_obs_model = np.interp(model_phases, centers, sigma_repr)
        sigma_eff_model = np.sqrt(sigma_obs_model ** 2 + (f_best * model_flux) ** 2)
        ax.fill_between(
            model_phases,
            model_flux - sigma_eff_model,
            model_flux + sigma_eff_model,
            alpha=0.18,
            color='C3',
            label='1-sigma effective (jitter)',
            zorder=6,
        )

    ax.set_xlabel('Orbital Phase', fontsize=12)
    ax.set_ylabel('Flux (erg/cm²/s)', fontsize=12)
    ax.set_title(f'{band.upper()} Band - {wind_model.upper()} Wind '
                 f'(χ²/dof = {red_chi2:.2f})', fontsize=14)
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
    if f_best is not None:
        rows.append(f"f: {f_best:.4f}  (from log_f)")
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
    if param_names is None:
        param_names = PARAM_NAMES
    print("\n" + "="*60)
    print(f"MCMC Diagnostics  ({sampler_type})")
    print("="*60)

    # Acceptance fraction (emcee only)
    if sampler_type == 'emcee' and hasattr(sampler, 'acceptance_fraction'):
        acc_frac = np.mean(sampler.acceptance_fraction)
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
        for i, param in enumerate(param_names):
            if i < len(tau):
                print(f"  {param}: {tau[i]:.1f} steps")

        chain = sampler.get_chain()
        n_steps = chain.shape[0]
        n_walkers = chain.shape[1]
        n_independent = n_steps / np.max(tau)
        print(f"\nEffective independent samples: ~{int(n_independent * n_walkers)}")

        if n_steps < 50 * np.max(tau):
            print("  WARNING: Chain may not be converged. Consider running longer.")
        else:
            print("  OK: Chain appears well-converged")
    except Exception:
        print("\nAutocorrelation time: Could not compute (chain too short)")

    print("="*60)


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
    studentt_nu: float = 5.0,
    n_samples: int = 200,
    reparam: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    active_names: List[str] = None,
) -> np.ndarray:
    """Compute per-observation log-likelihood for a subset of posterior samples.

    Returns array of shape (n_samples_used, n_obs).
    """
    n_total = len(samples)
    n_use = min(n_samples, n_total)
    indices = np.random.choice(n_total, size=n_use, replace=False)
    n_obs = len(obs_phase)
    log_lik = np.full((n_use, n_obs), np.nan)

    if active_names is not None and 'log_f' in active_names:
        idx_logf = active_names.index('log_f')
    else:
        idx_logf = 5

    for k, idx in enumerate(indices):
        theta = samples[idx]
        model_flux = _evaluate_model(
            theta, model, obs_phase, reparam=reparam,
            wind_model=wind_model, fit_wind_shape=fit_wind_shape,
            active_names=active_names,
        )
        if model_flux is None:
            continue

        if likelihood == 'jitter':
            f = np.exp(theta[idx_logf])
            sigma2 = obs_err ** 2 + (f * model_flux) ** 2
            log_lik[k] = -0.5 * ((obs_flux - model_flux) ** 2 / sigma2
                                  + np.log(sigma2) + np.log(2 * np.pi))
        elif likelihood == 'studentt':
            nu = studentt_nu
            resid = (obs_flux - model_flux) / obs_err
            log_lik[k] = (
                gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu)
                - 0.5 * np.log(nu * np.pi) - np.log(obs_err)
                - 0.5 * (nu + 1) * np.log(1.0 + resid ** 2 / nu)
            )
        else:
            log_lik[k] = (
                -0.5 * ((obs_flux - model_flux) / obs_err) ** 2
                - np.log(obs_err) - 0.5 * np.log(2 * np.pi)
            )

    valid = ~np.any(np.isnan(log_lik), axis=1)
    return log_lik[valid]


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
    studentt_nu: float = 5.0,
    compute_waic: bool = False,
    n_samples_waic: int = 200,
    output_dir: str = None,
    suffix: str = '',
    chain: np.ndarray = None,
    samples_flat: np.ndarray = None,
    reparam: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
):
    """Run ArviZ convergence diagnostics and optionally WAIC/LOO.

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
    if not HAS_ARVIZ:
        print("arviz not installed. Install with: pip install arviz")
        return None
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
        return None

    # posterior_dict: ArviZ wants (chain=walkers, draw=steps)
    posterior_dict = {
        name: chain[:, :, i].T for i, name in enumerate(param_names)
    }

    log_lik_dict = None
    if compute_waic and model is not None and samples_flat is not None:
        print(f"Computing per-point log-likelihoods for WAIC/LOO ({n_samples_waic} samples)...")
        ll = compute_pointwise_loglik(
            samples_flat, model, obs_phase, obs_flux, obs_err,
            likelihood=likelihood, studentt_nu=studentt_nu,
            n_samples=n_samples_waic,
            reparam=reparam,
            wind_model=wind_model,
            fit_wind_shape=fit_wind_shape,
            active_names=param_names,
        )
        if ll.shape[0] > 10:
            log_lik_dict = {"obs": ll[np.newaxis, :, :]}

    idata = _build_inference_data(posterior_dict, log_lik_dict)

    print("\n--- ArviZ Summary ---")
    summary = az.summary(idata)
    print(summary)

    has_ll = (
        hasattr(idata, "log_likelihood")
        or (hasattr(idata, "children") and "log_likelihood" in idata.children)
    )
    if has_ll:
        if hasattr(az, "waic"):
            try:
                waic = az.waic(idata)
                waic_val = getattr(waic, "elpd_waic", None) or getattr(waic, "waic", None)
                waic_se = getattr(waic, "se", getattr(waic, "waic_se", None))
                print(f"\nWAIC: {waic_val:.2f}" + (f" +/- {waic_se:.2f}" if waic_se else ""))
            except Exception as e:
                print(f"WAIC computation failed: {e}")
        else:
            print("\nWAIC: not available in this ArviZ version (removed in 1.0); using LOO instead.")
        try:
            loo = az.loo(idata)
            loo_val = getattr(loo, "elpd", None) or getattr(loo, "elpd_loo", None) or getattr(loo, "loo", None)
            loo_se = getattr(loo, "se", getattr(loo, "loo_se", None))
            if loo_val is not None:
                msg = f"LOO:  {loo_val:.2f}"
                if loo_se is not None:
                    msg += f" +/- {loo_se:.2f}"
                print(msg)
            else:
                print(f"LOO:  {loo}")
        except Exception as e:
            print(f"LOO computation failed: {e}")

    if output_dir:
        csv_path = os.path.join(output_dir, f"{suffix}_arviz_summary.csv" if suffix else "arviz_summary.csv")
        summary.to_csv(csv_path)
        print(f"ArviZ summary saved to: {csv_path}")

    return idata


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
    model_grid: PrecomputedModelGrid = None,
    priors: Dict = None,
    sim_params: Dict = None,
    reparam: bool = False,
    fit_wind_shape: bool = False,
    shape_prior_overrides: Dict[str, Dict] = None,
    is_binned: bool = True,
) -> Tuple[Dict, object]:
    """Run MCMC fit for a single band/wind_model combination."""

    if priors is None:
        priors = REPARAM_PRIORS.copy() if reparam else DEFAULT_PRIORS.copy()
    if sim_params is None:
        sim_params = {}

    likelihood = getattr(args, 'likelihood', 'chi2')

    # Active priors include geometry + (optional) jitter + (optional) shape.
    active_priors = get_active_priors(
        base_priors=priors,
        wind_model=wind_model,
        fit_wind_shape=fit_wind_shape,
        likelihood=likelihood,
        shape_prior_overrides=shape_prior_overrides,
    )

    # Grid always uses (d1, d2, r, R, i0) bounds
    grid_priors = _grid_priors_from_reparam(priors) if reparam else priors

    print(f"\n{'#'*60}")
    print(f"# Fitting {band.upper()} band - {WIND_MODELS[wind_model]}")
    print('#'*60)

    if sim_params:
        print(f"# Simulation params: {sim_params}")

    if model_grid is not None:
        if model_grid.wind_model != wind_model:
            raise ValueError(
                f"Reused grid was built for wind_model='{model_grid.wind_model}' "
                f"but request is for '{wind_model}'."
            )
        if bool(model_grid.fit_wind_shape) != bool(fit_wind_shape):
            raise ValueError(
                f"Reused grid has fit_wind_shape={model_grid.fit_wind_shape} "
                f"but request is fit_wind_shape={fit_wind_shape}."
            )
        model = model_grid
    elif args.no_grid:
        print("\n[info] Using DirectLightCurveModel (no precomputed grid).")
        model = DirectLightCurveModel(
            band=band,
            flux_csv_path=args.flux_csv,
            wind_model=wind_model,
            dth=args.dth,
            sim_params=sim_params,
        )
    else:
        grid_points = {
            'd1': args.grid_points,
            'd2': args.grid_points,
            'r':  max(5, args.grid_points // 2),
            'R':  args.grid_points,
            'i0': args.grid_points + 2,
        }

        # Shape priors for grid bounds: defaults, overridden per-name by CLI.
        shape_priors_for_grid = {
            name: dict(WIND_SHAPE_PRIORS[name])
            for name in WIND_SHAPE_FIT.get(wind_model, [])
        }
        if shape_prior_overrides:
            for name, override in shape_prior_overrides.items():
                if name in shape_priors_for_grid:
                    shape_priors_for_grid[name].update(override)

        model = PrecomputedModelGrid(
            band=band,
            flux_csv_path=args.flux_csv,
            wind_model=wind_model,
            priors=grid_priors,
            grid_points=grid_points,
            dth=args.dth,
            n_workers=args.n_workers,
            verbose=True,
            load_path=args.load_grid,
            sim_params=sim_params,
            fit_wind_shape=fit_wind_shape,
            shape_priors=shape_priors_for_grid,
            shape_grid_points=getattr(args, 'shape_grid_points', 5),
        )

        if args.save_grid:
            model.save(args.save_grid)

    sampler_type = getattr(args, 'sampler', 'emcee')
    studentt_nu = getattr(args, 'studentt_nu', 5.0)

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
        studentt_nu=studentt_nu,
        reparam=reparam,
        wind_model=wind_model,
        fit_wind_shape=fit_wind_shape,
        numba_threads_per_worker=getattr(
            args, "numba_threads_per_worker", None
        ),
    )

    try:
        chain_log_prob_flat = sampler.get_log_prob(discard=args.n_burn, flat=True)
    except Exception:
        chain_log_prob_flat = None
    stats = compute_statistics(
        samples, param_names=active_names, reparam=reparam,
        log_prob=chain_log_prob_flat,
    )
    print_results(stats, band, wind_model, param_names=active_names, reparam=reparam)
    print_diagnostics(sampler, sampler_type=sampler_type, param_names=active_names)

    compute_waic = getattr(args, 'compute_waic', False)
    run_arviz_diagnostics(
        sampler, active_names, args.n_burn,
        model=model, obs_phase=obs_phase, obs_flux=obs_flux, obs_err=obs_err,
        likelihood=likelihood, studentt_nu=studentt_nu,
        compute_waic=compute_waic,
        output_dir=args.output_dir,
        suffix=f"{band}_{wind_model}",
        reparam=reparam,
        wind_model=wind_model,
        fit_wind_shape=fit_wind_shape,
    )

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
            model, obs_phase, obs_flux, obs_err, stats, band, wind_model,
            os.path.join(args.output_dir, f"{suffix}_bestfit.png"),
            param_names=active_names,
            reparam=reparam,
            fit_wind_shape=fit_wind_shape,
            is_binned=is_binned,
            likelihood=likelihood,
        )
        stats['reduced_chi2'] = red_chi2

    samples_df = pd.DataFrame(samples, columns=active_names)
    try:
        log_prob = sampler.get_log_prob(discard=args.n_burn, flat=True)
        samples_df['log_prob'] = log_prob
    except Exception:
        pass
    samples_df.to_csv(
        os.path.join(args.output_dir, f"{suffix}_samples.csv"),
        index=False
    )
    print(f"Samples saved to: {args.output_dir}/{suffix}_samples.csv")

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
            studentt_nu=studentt_nu,
            reparam=reparam,
            wind_model=wind_model,
            fit_wind_shape=fit_wind_shape,
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
    priors: Dict = None,
    sim_params: Dict = None,
    reparam: bool = False,
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
        priors = REPARAM_PRIORS.copy() if reparam else DEFAULT_PRIORS.copy()
    if sim_params is None:
        sim_params = {}

    grid_priors = _grid_priors_from_reparam(priors) if reparam else priors

    print(f"\n{'#'*60}")
    print(f"# Replotting {band.upper()} band - {WIND_MODELS[wind_model]}")
    print('#'*60)

    samples, stats, loaded_names = load_existing_results(
        args.output_dir, band, wind_model, reparam=reparam,
    )

    if samples is None or stats is None:
        print(f"Could not load existing results for {band}_{wind_model}")
        return None

    print_results(stats, band, wind_model,
                  param_names=loaded_names, reparam=reparam)

    # Auto-detect shape-fit from the saved column names.
    geom_names = REPARAM_PARAM_NAMES if reparam else PARAM_NAMES
    extra_dims = [n for n in loaded_names
                  if n not in geom_names and n != 'log_f']
    saved_fit_wind_shape = bool(extra_dims) or fit_wind_shape

    # Model for best-fit overlay.  Replotting doesn't need fast likelihood
    # evaluation, so we default to the direct evaluator unless the user
    # explicitly asked to reuse a precomputed grid (via --load-grid).
    if args.load_grid:
        print(f"\nLoading precomputed grid from {args.load_grid} for replot...")
        grid_points = {
            'd1': args.grid_points,
            'd2': args.grid_points,
            'r':  max(5, args.grid_points // 2),
            'R':  args.grid_points,
            'i0': args.grid_points + 2,
        }
        model = PrecomputedModelGrid(
            band=band,
            flux_csv_path=args.flux_csv,
            wind_model=wind_model,
            priors=grid_priors,
            grid_points=grid_points,
            dth=args.dth,
            n_workers=args.n_workers,
            verbose=True,
            load_path=args.load_grid,
            sim_params=sim_params,
            fit_wind_shape=saved_fit_wind_shape,
            shape_grid_points=getattr(args, 'shape_grid_points', 5),
        )
    else:
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
        except Exception:
            pass

    if not args.no_plots:
        # Use the loaded column names as the active set.
        active_names = list(loaded_names)
        active_labels = []
        geom_labels = REPARAM_PARAM_LABELS if reparam else PARAM_LABELS
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
            model, obs_phase, obs_flux, obs_err, stats, band, wind_model,
            os.path.join(args.output_dir, f"{suffix}_bestfit.png"),
            param_names=active_names,
            reparam=reparam,
            fit_wind_shape=saved_fit_wind_shape,
            is_binned=is_binned,
            likelihood=saved_likelihood,
        )
        stats['reduced_chi2'] = red_chi2

    compute_waic = getattr(args, 'compute_waic', False)
    if HAS_ARVIZ or compute_waic:
        if os.path.exists(chain_path):
            print(f"\nLoading saved chain from: {chain_path}")
            chain_data = np.load(chain_path, allow_pickle=True)
            saved_chain = chain_data['chain']
            saved_names = list(chain_data['param_names'])
            saved_likelihood = str(chain_data.get('likelihood', 'chi2'))
            saved_nu = float(chain_data.get('studentt_nu', 5.0))
            saved_reparam = bool(chain_data.get('reparam', False))
            saved_wind_model = str(chain_data.get('wind_model', wind_model))
            saved_fws = bool(chain_data.get('fit_wind_shape', saved_fit_wind_shape))
            print(f"  Chain shape: {saved_chain.shape} "
                  f"(likelihood={saved_likelihood})")

            run_arviz_diagnostics(
                param_names=saved_names,
                chain=saved_chain,
                model=model,
                obs_phase=obs_phase,
                obs_flux=obs_flux,
                obs_err=obs_err,
                likelihood=saved_likelihood,
                studentt_nu=saved_nu,
                compute_waic=compute_waic,
                output_dir=args.output_dir,
                suffix=suffix,
                reparam=saved_reparam,
                wind_model=saved_wind_model,
                fit_wind_shape=saved_fws,
            )
        elif compute_waic:
            print(f"Chain file not found: {chain_path}")
            print("  Re-run MCMC to generate it, or skip --compute-waic.")

    if getattr(args, 'save_chi2', False):
        chi2_path = os.path.join(args.output_dir, f"{suffix}_chi2.csv.gz")
        compute_chi2_for_samples(
            model, samples, obs_phase, obs_flux, obs_err,
            output_path=chi2_path,
            n_samples=getattr(args, 'chi2_n_samples', None),
            verbose=True,
            reparam=reparam,
            wind_model=wind_model,
            fit_wind_shape=saved_fit_wind_shape,
            active_names=loaded_names,
            likelihood=saved_likelihood,
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
            "confinement: fconf, ell). Works with the precomputed grid "
            "(grid axes are expanded automatically; tune --shape-grid-points) "
            "or with --no-grid. Override priors via --prior-Rb, --prior-p, "
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
        default=50,
        help="Number of phase bins (ignored if --no-phase-bin)"
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
        choices=['chi2', 'jitter', 'studentt'],
        default='chi2',
        help="Likelihood function: 'chi2' (standard Gaussian/chi-squared), "
             "'jitter' (Gaussian with free systematic error term — adds log_f parameter), "
             "'studentt' (Student-t, robust to outliers)"
    )
    parser.add_argument(
        "--studentt-nu",
        type=float,
        default=5.0,
        help="Degrees of freedom for Student-t likelihood (only used with --likelihood studentt). "
             "Lower values give heavier tails; nu -> inf recovers Gaussian."
    )
    parser.add_argument(
        "--reparam",
        action="store_true",
        help="Reparameterize (d1, d2) as (a, q) where a = d1+d2 (orbital separation) "
             "and q = d1/(d1+d2) (mass-ratio proxy). Decorrelates the two distance "
             "parameters for faster MCMC convergence."
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
             "Only useful with --no-grid (DirectLightCurveModel), where each "
             "log-likelihood call is ~63 ms. With a PrecomputedModelGrid the "
             "per-step lookup is microseconds while the grid can be hundreds of "
             "MB to several GB; multiprocessing would pickle/ship the entire "
             "grid to each worker on every step (causing the progress bar to "
             "hang), so this flag is auto-ignored in that case."
    )
    parser.add_argument(
        "--numba-threads-per-worker",
        type=int,
        default=None,
        metavar="N",
        help="When using --no-grid with --n-threads>1, set Numba's thread count "
             "inside each worker via numba.set_num_threads(N). Default (omit this "
             "flag): auto = max(1, cpu_count // n_threads) so workers collectively "
             "use about one thread per logical CPU without each worker running a "
             "fully serial Numba kernel. Set explicitly if you tune "
             "--n-threads for your machine."
    )
    parser.add_argument(
        "--compute-waic",
        action="store_true",
        help="Compute WAIC and LOO model comparison metrics via ArviZ "
             "(requires arviz: pip install arviz). Slower but useful for comparing "
             "likelihoods and binning strategies."
    )
    
    # Model grid options
    parser.add_argument(
        "--grid-points",
        type=int,
        default=8,
        help="Number of grid points per geometry parameter (d1, d2, r, R, i0) "
             "in the pre-computed grid"
    )
    parser.add_argument(
        "--shape-grid-points",
        type=int,
        default=5,
        help="Number of grid points per wind-shape axis when --fit-wind-shape "
             "is used with the precomputed grid (ignored without --fit-wind-shape "
             "or with --no-grid). Default: 5"
    )
    parser.add_argument(
        "--dth",
        type=float,
        default=5.0,
        help="Model phase resolution in degrees (larger = faster)"
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel workers for grid computation (default: num CPUs - 1)"
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable pre-computed grid (SLOW! Only for debugging)"
    )
    parser.add_argument(
        "--save-grid",
        type=str,
        default=None,
        help="Save pre-computed grid to file (e.g., 'grids/broad_grid.npz')"
    )
    parser.add_argument(
        "--load-grid",
        type=str,
        default=None,
        help="Load pre-computed grid from file (skips grid computation)"
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

    # Warn on very large grids when fitting wind shape + lots of points.
    if getattr(args, 'fit_wind_shape', False) and not args.no_grid:
        wm = getattr(args, 'wind_model', 'smooth_pl')
        shape_dims = len(WIND_SHAPE_FIT.get(wm, []))
        total = (
            args.grid_points ** 4
            * max(5, args.grid_points // 2)
            * (args.shape_grid_points ** shape_dims)
        )
        if shape_dims > 0:
            print(
                f"[notice] --fit-wind-shape with precomputed grid: "
                f"geometry grid × {args.shape_grid_points}^{shape_dims} "
                f"shape axes → ~{total:,} grid points to compute "
                f"(use --no-grid for a quick direct MCMC, or --save-grid to "
                f"cache for future runs)."
            )

    # Build simulation parameters dict
    sim_params = {
        'lam': args.lam,
        'gma0': args.gma0,
        'd2h': args.d2h,
        'dz': args.dz,
    }

    # Build custom geometry priors
    reparam = getattr(args, 'reparam', False)
    if reparam:
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
                obs_df = phase_bin_data(obs_df, n_bins=args.n_phase_bins)

            obs_phase = obs_df['phase'].values
            obs_flux = obs_df['flux'].values
            obs_err = obs_df['flux_err'].values

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
                            obs_phase, obs_flux, obs_err,
                            priors=priors,
                            sim_params=sim_params,
                            reparam=reparam,
                            fit_wind_shape=fit_wind_shape,
                            shape_prior_overrides=shape_prior_overrides,
                            is_binned=is_binned,
                        )
                        if stats is not None:
                            all_results[key] = stats
                    else:
                        stats, _ = run_single_fit(
                            band, wind_model, args,
                            obs_phase, obs_flux, obs_err,
                            model_grid=None,
                            priors=priors,
                            sim_params=sim_params,
                            reparam=reparam,
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

            likelihood = getattr(args, 'likelihood', 'chi2')
            active_names, _ = get_param_config(
                likelihood, reparam=reparam,
                wind_model=args.wind_model, fit_wind_shape=fit_wind_shape,
            )
            for key, stats in all_results.items():
                band, wind_model = key
                f.write(f"{band.upper()} Band - {WIND_MODELS[wind_model]}\n")
                f.write("-"*40 + "\n")

                f.write("Marginal posterior (median +upper/-lower, 16/84 pct):\n")
                for param in active_names:
                    if param in stats:
                        s = stats[param]
                        f.write(f"  {param}: {s['median']:.6f} "
                                f"(+{s['upper']:.6f}/-{s['lower']:.6f})\n")
                if reparam:
                    for derived in ('d1', 'd2'):
                        if derived in stats:
                            s = stats[derived]
                            f.write(f"  {derived} (derived): {s['median']:.6f} "
                                    f"(+{s['upper']:.6f}/-{s['lower']:.6f})\n")

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
                    if reparam:
                        for derived in ('d1', 'd2'):
                            if derived in stats and 'map' in stats[derived]:
                                f.write(
                                    f"  {derived} (derived): "
                                    f"{stats[derived]['map']:.6f}\n"
                                )
                    if reparam:
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
                f.write("\n")

            f.write(
                "Note: marginal medians of nonlinear combinations are not the\n"
                "combinations of medians, so median(d1) + median(d2) need not\n"
                "equal median(a), and median(d1)/(median(d1)+median(d2)) need\n"
                "not equal median(q). The MAP block above is a single sample,\n"
                "so those identities hold exactly.\n"
            )
        
        print(f"\nSummary saved to: {summary_path}")
    
    if args.replot:
        print("\nReplotting complete!")
    else:
        print("\nMCMC fitting complete!")


if __name__ == "__main__":
    main()
