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
import copy
from dataclasses import dataclass, field
import multiprocessing as mp
import os
import time
import warnings
from typing import Tuple, List, Dict, Optional
from multiprocessing import cpu_count
import numpy as np
import pandas as pd

try:
    import emcee
except ImportError:
    raise ImportError("emcee is required. Install with: pip install emcee")

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

# Benign but very frequent (once per model evaluation); simulate_lightcurve is
# already called with verbose=False, so this is a belt-and-suspenders catch.
warnings.filterwarnings(
    "ignore",
    message=r"Some nH values are outside CSV range",
    category=UserWarning,
)

from xrb_lightcurve import (
    simulate_lightcurve,
    WIND_MODEL_PARAM_KEYS,
    default_wind_params,
    evaluate_g_profile,
)
from utils.utils import (
    DEFAULT_PHASE_SHIFT_EVAL_POINTS,
    DEFAULT_PHASE_SHIFT_GRID_SIZE,
    ORBITAL_PERIOD,
    RUN_CONFIG_SUFFIX,
    apply_best_phase_shift as _apply_best_phase_shift,
    apply_saved_run_config,
    build_phase_shift_terms as _build_phase_shift_terms,
    estimate_scattered_flux,
    fmt_val as _fmt_val,
    interp_periodic_phases as _interp_periodic_phases,
    load_observed_lightcurves,
    phase_bin_data,
    phase_bin_data_snr,
    run_config_path,
    save_run_config,
    save_samples_csv_chunked,
    smooth_lightcurve,
)
from utils.plot_utils import (
    plot_corner,
    plot_geometry_vs_phase,
    plot_lightcurve_fit,
    plot_orbit_geometry,
    plot_trace,
    plot_wind_profile,
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


@dataclass
class ParamSpec:
    mode: str = 'phys'  # 'phys' | 'reparam' | 'kepler'
    active_names: List[str] = field(default_factory=list)
    active_labels: List[str] = field(default_factory=list)
    frozen: Dict[str, float] = field(default_factory=dict)
    fit_wind_shape: bool = False
    fit_scatter: bool = False
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
    fit_scatter: bool = False,
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
    if fit_scatter:
        names.append('f_scatter')
        labels.append(r'$f_\mathrm{scat}$')
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
    valid_frozen.add('f_scatter')

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
        fit_scatter=fit_scatter,
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
    fit_scatter: bool = False,
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
        fit_scatter=fit_scatter,
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
    fit_scatter: bool = False,
    scatter_prior: Optional[Dict[str, float]] = None,
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
    if fit_scatter and scatter_prior is not None:
        out.setdefault('f_scatter', dict(scatter_prior))
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


def _resolve_scatter(
    theta: np.ndarray,
    active_names: Optional[List[str]],
    param_spec: Optional[ParamSpec],
) -> float:
    """Resolve additive scattered flux from active or frozen parameters."""
    names = list(active_names or [])
    if param_spec is not None and param_spec.active_names:
        names = list(param_spec.active_names)
    if 'f_scatter' in names:
        idx = names.index('f_scatter')
        return float(theta[idx])
    if param_spec is not None and 'f_scatter' in param_spec.frozen:
        return float(param_spec.frozen['f_scatter'])
    return 0.0


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
    model_flux = np.asarray(model_flux, dtype=float) + _resolve_scatter(
        np.asarray(theta, dtype=float),
        active_names=active_names,
        param_spec=param_spec,
    )
    return model_flux


def _default_priors(reparam: bool = False, kepler: bool = False) -> Dict:
    """The prior dict for a parameterization, as a fresh copy."""
    if kepler:
        return copy.deepcopy(KEPLER_PRIORS)
    if reparam:
        return copy.deepcopy(REPARAM_PRIORS)
    return copy.deepcopy(DEFAULT_PRIORS)


def _aligned_model_flux(
    theta: np.ndarray,
    model,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err2: np.ndarray,
    reparam: bool = False,
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    active_names: List[str] = None,
    phase_shift_terms: Optional[Dict[str, object]] = None,
    param_spec: Optional[ParamSpec] = None,
) -> Optional[np.ndarray]:
    """Model flux at the observed phases, phase-shift-aligned when enabled.

    Shared front half of both likelihoods. Returns None when the model cannot
    be evaluated (out-of-domain geometry), which callers turn into -inf.
    """
    shifting = bool(phase_shift_terms and phase_shift_terms.get("enabled", False))
    eval_phases = (
        np.asarray(phase_shift_terms["phase_eval_grid"], dtype=float)
        if shifting else obs_phase
    )
    model_flux = _evaluate_model(
        theta, model, eval_phases, reparam=reparam,
        wind_model=wind_model, fit_wind_shape=fit_wind_shape,
        active_names=active_names, param_spec=param_spec,
    )
    if model_flux is None or not shifting:
        return model_flux
    model_flux, _ = _apply_best_phase_shift(
        eval_phases, model_flux, obs_phase, obs_flux, obs_err2, phase_shift_terms
    )
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
    if obs_err2 is None:
        obs_err2 = obs_err ** 2
    model_flux = _aligned_model_flux(
        theta, model, obs_phase, obs_flux, obs_err2,
        reparam=reparam, wind_model=wind_model, fit_wind_shape=fit_wind_shape,
        active_names=active_names, phase_shift_terms=phase_shift_terms,
        param_spec=param_spec,
    )
    if model_flux is None:
        return -np.inf
    return -0.5 * np.sum((obs_flux - model_flux) ** 2 / obs_err2)


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
    if obs_err2 is None:
        obs_err2 = obs_err ** 2
    model_flux = _aligned_model_flux(
        theta, model, obs_phase, obs_flux, obs_err2,
        reparam=reparam, wind_model=wind_model, fit_wind_shape=fit_wind_shape,
        active_names=active_names, phase_shift_terms=phase_shift_terms,
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
    sigma2 = obs_err2 + (np.exp(theta[idx_logf]) * model_flux) ** 2
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

    # Clip walkers just inside the box, with a *scale-relative* inset: an
    # absolute epsilon exceeded f_scatter's entire range (~1e-13, prior min 0),
    # collapsing that column and aborting emcee on the condition number.
    for i, name in enumerate(active_names):
        prior = priors[name]
        lo_bound = float(prior['min'])
        hi_bound = float(prior['max'])
        span = hi_bound - lo_bound
        pad = 1e-9 * span if np.isfinite(span) and span > 0 else 0.0
        lo = lo_bound + 0.01 * abs(lo_bound) + pad
        hi = hi_bound - 0.01 * abs(hi_bound) - pad
        if not (lo < hi):
            # Degenerate/inverted box after the inset: fall back to the raw box.
            lo, hi = lo_bound, hi_bound
        pos[:, i] = np.clip(pos[:, i], lo, hi)

    # Enforce r < R only when both are free dimensions.
    idx_r = active_names.index('r') if 'r' in active_names else None
    idx_R = active_names.index('R') if 'R' in active_names else None
    if (idx_r is not None) and (idx_R is not None):
        for j in range(n_walkers):
            if pos[j, idx_r] >= pos[j, idx_R]:
                pos[j, idx_r] = pos[j, idx_R] * 0.1

    # Final guard: emcee/zeus require linearly independent walkers, so any dim
    # that ended up constant (all walkers pinned to a bound) must be re-spread.
    for i, name in enumerate(active_names):
        if np.ptp(pos[:, i]) > 0:
            continue
        prior = priors[name]
        lo_bound, hi_bound = float(prior['min']), float(prior['max'])
        span = hi_bound - lo_bound
        if not (np.isfinite(span) and span > 0):
            continue
        warnings.warn(
            f"Walker initialization for '{name}' collapsed to a single value "
            f"({pos[0, i]:.6g}); re-spreading uniformly inside its prior box. "
            f"Check that the prior for '{name}' is on a sensible scale."
        )
        pos[:, i] = np.random.uniform(
            lo_bound + 0.05 * span, hi_bound - 0.05 * span, size=n_walkers
        )

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
        # Trailing dict holds invariants hoisted out of the likelihood hot loop
        # (formulas unchanged). `jitter_logf_index` is jitter-only.
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




# Options that must never be restored from a saved run. `replot` is the flag
# being used right now — a saved fit always recorded replot=False, so restoring
# it would silently cancel the replot. `output_dir` is defined by where the
# config file was found, not by what the original run typed.














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

    # Geometry block depends on the *saved* run's parameterization. Frozen
    # params are never sampled, so their absence from the CSV is expected.
    if kepler:
        geom_names, saved_mode = KEPLER_PARAM_NAMES, 'kepler'
    elif reparam:
        geom_names, saved_mode = REPARAM_PARAM_NAMES, 'reparam'
    else:
        geom_names, saved_mode = PARAM_NAMES, 'phys'
    frozen_names = set((param_spec.frozen if param_spec is not None else {}) or {})
    missing_geom = [
        p for p in geom_names
        if p not in samples_df.columns and p not in frozen_names
    ]
    if missing_geom:
        present = [c for c in samples_df.columns if c != 'log_prob']
        hint = ""
        for mode, names in (('phys', PARAM_NAMES), ('reparam', REPARAM_PARAM_NAMES),
                            ('kepler', KEPLER_PARAM_NAMES)):
            if mode != saved_mode and all(n in samples_df.columns for n in names):
                flag = {'reparam': '--reparam', 'kepler': '--kepler', 'phys': 'neither --reparam nor --kepler'}[mode]
                hint = (f" The columns look like '{mode}' mode — rerun with {flag} "
                        f"(or let --replot restore it from the saved run config).")
                break
        print(
            f"Error: samples file has no column for geometry parameter(s) "
            f"{missing_geom} expected by '{saved_mode}' mode.\n"
            f"  Columns present: {present}.{hint}"
        )
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
        try:
            # Via _evaluate_model, not model.evaluate: hand-rolling this once
            # omitted f_scatter and biased every per-sample chi2.
            model_flux = _evaluate_model(
                sample_params,
                model,
                phase_eval_grid,
                reparam=reparam,
                wind_model=wind_model,
                fit_wind_shape=fit_wind_shape,
                active_names=active_names,
                param_spec=param_spec,
            )

            if model_flux is not None:
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
        print(f"{param:<15} {_fmt_val(s['median'], 12)} {_fmt_val(s['lower'], 12)} {_fmt_val(s['upper'], 12)}")

    if reparam or ('d1' in stats and 'd2' in stats):
        print('-'*60)
        print("Derived physical parameters:")
        for derived in ('d1', 'd2'):
            if derived in stats:
                s = stats[derived]
                print(f"{derived:<15} {_fmt_val(s['median'], 12)} {_fmt_val(s['lower'], 12)} {_fmt_val(s['upper'], 12)}")

    print('='*60)


def _point_estimate_theta(
    stats: Dict,
    param_names: List[str],
    param_spec: Optional[ParamSpec] = None,
    reparam: bool = False,
) -> Tuple[np.ndarray, str]:
    """Posterior point estimate in active-parameter order, plus its label.

    Prefers the MAP point (one sample) over per-parameter medians: medians of
    nonlinear combinations are not the combinations of medians, so a median
    point would violate d1 + d2 = a and d1/(d1+d2) = q. The MAP point does not.
    Frozen parameters are filled from *param_spec*.
    """
    use_map = all(
        ('map' in stats[p]) for p in
        (PARAM_NAMES if not reparam else ['r', 'R', 'i0', 'd1', 'd2'])
        if p in stats
    )
    point_key = 'map' if use_map else 'median'

    def value(name: str) -> float:
        if name in stats and point_key in stats[name]:
            return float(stats[name][point_key])
        if param_spec is not None and name in param_spec.frozen:
            return float(param_spec.frozen[name])
        raise KeyError(
            f"Missing '{name}' in statistics for best-fit plotting. "
            f"If this parameter is frozen, pass param_spec with frozen values."
        )

    return np.array([value(n) for n in param_names], dtype=float), point_key


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
    smooth_phase: Optional[np.ndarray] = None,
    smooth_flux: Optional[np.ndarray] = None,
    smooth_flux_err: Optional[np.ndarray] = None,
    smooth_sigma: float = 0.01,
):
    """
    Plot observed data with best-fit model overlay.

    Resolves the posterior point estimate, evaluates the physical model (and the
    per-sample phase shift) at the observed phases, then hands the resulting
    arrays to :func:`utils.plot_utils.plot_lightcurve_fit` — the single plotting
    routine shared with ``chandra_phase_analysis``. The figure title carries only
    the energy band and χ²/dof; the parameter values are printed to stdout and
    written to the run summary rather than annotated on the plot.

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

    Returns
    -------
    float
        Reduced χ² of the drawn model at the observed phases.
    """
    if param_names is None:
        param_names = PARAM_NAMES

    # Evaluated through _evaluate_model — the same entry point the likelihood uses.
    theta_best, point_key = _point_estimate_theta(
        stats, param_names, param_spec=param_spec, reparam=reparam)

    def _eval_at(phases: np.ndarray) -> np.ndarray:
        out = _evaluate_model(
            theta_best,
            model,
            np.asarray(phases, dtype=float),
            reparam=reparam,
            wind_model=wind_model,
            fit_wind_shape=fit_wind_shape,
            active_names=param_names,
            param_spec=param_spec,
        )
        if out is None:
            return np.full(np.shape(phases), np.nan, dtype=float)
        return np.asarray(out, dtype=float)

    # Reported separately below; the value itself is already baked into every
    # _eval_at() result by _evaluate_model.
    f_scatter_best = _resolve_scatter(theta_best, param_names, param_spec)

    model_phases = np.linspace(0, 1, 360)
    model_flux = _eval_at(model_phases)

    phase_shift_terms = _build_phase_shift_terms(
        fit_phase_shift,
        obs_phase,
        grid_size=phase_shift_grid_size,
        eval_points=phase_shift_eval_points,
    )
    best_phase_shift = 0.0
    if phase_shift_terms.get("enabled", False):
        phase_eval_grid = np.asarray(phase_shift_terms["phase_eval_grid"], dtype=float)
        obs_model, best_phase_shift = _apply_best_phase_shift(
            phase_eval_grid,
            _eval_at(phase_eval_grid),
            obs_phase,
            obs_flux,
            obs_err ** 2,
            phase_shift_terms,
        )
        if obs_model is None:
            obs_model = np.full_like(obs_phase, np.nan)
            best_phase_shift = 0.0
    else:
        obs_model = _eval_at(obs_phase)
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

    # Reported to stdout rather than annotated on the figure, whose title now
    # carries only the energy band and chi2/dof.
    point_label = 'MAP' if point_key == 'map' else 'median'
    print(f"Best-fit overlay from the {point_label} point estimate: "
          f"chi2/dof = {red_chi2:.6g} (dof = {dof})")
    if fit_phase_shift:
        print(f"  phase_shift = {best_phase_shift:.5f}")
    if f_best is not None:
        print(f"  f = {f_best:.4f} (from log_f)", end='')
        if np.isfinite(red_chi2_eff):
            print(f", chi2_eff/dof = {red_chi2_eff:.6g}", end='')
        print()
    if 'f_scatter' in stats or (param_spec is not None and 'f_scatter' in param_spec.frozen):
        print(f"  f_scatter = {f_scatter_best:.6g}")

    # Drawn by the single shared plotting routine.
    overlay_phase = (
        np.mod(model_phases + best_phase_shift, 1.0) if fit_phase_shift else model_phases
    )
    plot_lightcurve_fit(
        obs_phase,
        obs_flux,
        obs_err,
        model_phase=overlay_phase,
        model_flux=model_flux,
        obs_model=obs_model,
        obs_phase_width=obs_phase_width,
        band=band.upper(),
        red_chi2=red_chi2,
        output_path=output_path,
        is_binned=is_binned,
        obs_label='Observed (phase-binned)' if is_binned else 'Observed (raw 100s)',
        model_label=f'Best-fit model ({WIND_MODELS[wind_model]})',
        smooth_phase=smooth_phase,
        smooth_flux=smooth_flux,
        smooth_flux_err=smooth_flux_err,
        smooth_sigma=smooth_sigma,
        verbose=False,
    )

    print(f"Best-fit plot saved to: {output_path}")

    return red_chi2


def plot_geometry_diagnostics(
    stats: Dict,
    samples: np.ndarray,
    band: str,
    wind_model: str,
    output_dir: str,
    suffix: str,
    param_names: List[str],
    reparam: bool = False,
    fit_wind_shape: bool = False,
    param_spec: Optional[ParamSpec] = None,
    sim_params: Optional[Dict] = None,
    dth: float = 5.0,
    flux_csv_path: Optional[str] = None,
    n_profile_draws: int = 300,
    verbose: bool = True,
) -> Optional[Dict[str, str]]:
    """Geometry figures for the posterior point estimate.

    Three plots, each answering a question the light-curve fit alone does not:

    1. ``*_geometry_orbit.png`` -- projected orbit against the companion disk,
       plus a to-scale top-down view. The eclipse width constrains a
       *combination* of (a, R, i0), so this is where an implausible but
       well-fitting parameter set becomes obvious (e.g. a "companion" larger
       than its own orbit, or a dip that is pure absorption with no geometric
       eclipse at all).
    2. ``*_geometry_phase.png`` -- projected separation against the R+/-r
       eclipse thresholds, sky-plane components, N_H(phase) and the band flux.
       Turns the eclipse from an emergent light-curve feature into a stated
       geometric condition with visible margin.
    3. ``*_wind_profile.png`` -- g(r) with a posterior credible band. The shape
       parameters are only interpretable jointly (Rb and p trade off strongly),
       so the constraint is much clearer on g(r) than in a corner plot. The
       band of radii the line of sight actually probes is shaded: the profile
       inside the minimum impact parameter is unconstrained by these data.

    Returns a dict of the figures written, or None if the point estimate could
    not be resolved / the simulation failed.
    """
    try:
        theta_best, point_key = _point_estimate_theta(
            stats, param_names, param_spec=param_spec, reparam=reparam)
    except KeyError as e:
        warnings.warn(f"Skipping geometry plots: {e}")
        return None

    d1, d2, r, R, i0 = _resolve_geom(
        theta_best, reparam=reparam, active_names=param_names, param_spec=param_spec)
    frozen = dict(param_spec.frozen) if param_spec is not None else {}
    wind_params = _to_wind_params(
        theta_best, param_names, wind_model, R,
        fit_wind_shape=fit_wind_shape, frozen=frozen)
    # Same additive floor the likelihood and plot_best_fit apply, so the flux
    # panel here shows the curve that was actually fitted.
    f_scatter = _resolve_scatter(theta_best, param_names, param_spec)

    sim_params = sim_params or {}
    if verbose:
        shape_txt = ", ".join(f"{k}={v:.4g}" for k, v in sorted(wind_params.items()))
        print(f"\nGeometry diagnostics at the {point_key} point estimate:")
        print(f"  d1={d1:.4f}  d2={d2:.4f}  a={d1 + d2:.4f}  "
              f"r={r:.6g}  R={R:.4f}  i0={i0:.4f} deg")
        print(f"  wind_params: {shape_txt}")
        if f_scatter:
            print(f"  f_scatter:   {f_scatter:.6g} (additive floor)")

    # One simulate_lightcurve call gives every geometry column we need.
    try:
        sim_df = simulate_lightcurve(
            r=r, R=R, d1=d1, d2=d2, i0=i0,
            gma0=sim_params.get('gma0', -90.0),
            dth=dth,
            d2h=sim_params.get('d2h', 6.0),
            dz=sim_params.get('dz', 0.5),
            flux_method="interpolate" if flux_csv_path else "legacy",
            flux_csv_path=flux_csv_path,
            lam=sim_params.get('lam', 0.589537),
            wind_model=wind_model,
            wind_params=wind_params,
            scattered_flux=f_scatter,
            verbose=False,
        )
    except Exception as e:
        warnings.warn(f"Skipping geometry plots: simulate_lightcurve failed: {e}")
        return None

    written: Dict[str, str] = {}
    band_label = band.upper()

    orbit_path = os.path.join(output_dir, f"{suffix}_geometry_orbit.png")
    try:
        plot_orbit_geometry(sim_df, R=R, r=r, d1=d1, d2=d2, i0=i0,
                            output_path=orbit_path, band=band_label,
                            verbose=verbose)
        written['orbit'] = orbit_path
    except Exception as e:
        warnings.warn(f"Orbit-geometry plot failed: {e}")

    phase_path = os.path.join(output_dir, f"{suffix}_geometry_phase.png")
    try:
        plot_geometry_vs_phase(sim_df, R=R, r=r, band=band_label,
                               flux_column=f"nfl_{band.lower()}",
                               output_path=phase_path, verbose=verbose)
        written['phase'] = phase_path
    except Exception as e:
        warnings.warn(f"Geometry-vs-phase plot failed: {e}")

    # --- wind profile with a posterior band ---------------------------------
    # Radii probed: the line of sight starting at the compact object has, by
    # construction, an impact parameter relative to the companion centre equal
    # to the sky-projected separation l3. So the profile inside min(l3) is never
    # sampled, which is exactly the honest statement to put on the figure.
    probed = None
    if 'l3' in sim_df.columns:
        l3 = sim_df['l3'].to_numpy(dtype=float)
        if 'is_eclipsed' in sim_df.columns:
            keep = ~sim_df['is_eclipsed'].to_numpy(dtype=bool)
            l3 = l3[keep] if np.any(keep) else l3
        l3 = l3[np.isfinite(l3)]
        if l3.size:
            probed = (float(np.min(l3)), float(np.max(l3)))

    r_lo = max(1e-3, 0.5 * min(float(R), probed[0] if probed else float(R)))
    r_hi = max(4.0 * float(R), (probed[1] * 3.0 if probed else 10.0 * float(R)))
    if wind_params.get('Rb'):
        r_hi = max(r_hi, 2.0 * float(wind_params['Rb']))
    r_grid = np.logspace(np.log10(r_lo), np.log10(r_hi), 240)

    g_rows: List[np.ndarray] = []
    draws = np.atleast_2d(np.asarray(samples, dtype=float)) if samples is not None else None
    if draws is not None and draws.size and draws.shape[1] == len(param_names):
        n = min(int(n_profile_draws), draws.shape[0])
        idx = (np.random.choice(draws.shape[0], size=n, replace=False)
               if n < draws.shape[0] else np.arange(draws.shape[0]))
        for k in idx:
            th = draws[k]
            try:
                _, _, _, R_k, _ = _resolve_geom(
                    th, reparam=reparam, active_names=param_names,
                    param_spec=param_spec)
                wp_k = _to_wind_params(th, param_names, wind_model, R_k,
                                      fit_wind_shape=fit_wind_shape, frozen=frozen)
                g_rows.append(np.asarray(
                    evaluate_g_profile(r_grid, wind_model, wp_k), dtype=float))
            except Exception:
                continue
    if not g_rows:
        try:
            g_rows.append(np.asarray(
                evaluate_g_profile(r_grid, wind_model, wind_params), dtype=float))
        except Exception as e:
            warnings.warn(f"Wind-profile plot failed: {e}")
            return written or None

    shape_keys = WIND_SHAPE_FIT.get(wind_model, ())
    summary = "\n".join(
        f"{k:>7s} = {wind_params[k]:.4g}" + ("" if k in shape_keys and fit_wind_shape
                                             else "  (fixed)")
        for k in WIND_MODEL_PARAM_KEYS.get(wind_model, ())
        if k in wind_params
    )
    profile_path = os.path.join(output_dir, f"{suffix}_wind_profile.png")
    try:
        plot_wind_profile(
            r_grid, np.vstack(g_rows), R=R, probed_range=probed,
            mark_radii={k: wind_params[k] for k in ('Rb', 'H', 'ell')
                        if k in wind_params},
            wind_model=WIND_MODELS.get(wind_model, wind_model),
            band=band_label, shape_summary=summary or None,
            output_path=profile_path, verbose=verbose)
        written['wind_profile'] = profile_path
    except Exception as e:
        warnings.warn(f"Wind-profile plot failed: {e}")

    return written or None


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

def _phase_shift_opts(args) -> Dict[str, object]:
    """The phase-shift search options, as kwargs for the fit/plot/chi2 helpers."""
    return {
        'fit_phase_shift': not getattr(args, "no_fit_phase_shift", False),
        'phase_shift_grid_size': getattr(args, "phase_shift_grid_size", DEFAULT_PHASE_SHIFT_GRID_SIZE),
        'phase_shift_eval_points': getattr(args, "phase_shift_eval_points", DEFAULT_PHASE_SHIFT_EVAL_POINTS),
    }


def _smooth_plot_kwargs(smoothed, args) -> Dict[str, object]:
    """Smoothed-curve arrays as kwargs for plot_best_fit."""
    if smoothed is None:
        return {'smooth_sigma': float(getattr(args, "smooth_sigma", 0.01))}
    return {
        'smooth_phase': smoothed["phase"].to_numpy(dtype=float),
        'smooth_flux': smoothed["flux_smooth"].to_numpy(dtype=float),
        'smooth_flux_err': smoothed["flux_smooth_err"].to_numpy(dtype=float),
        'smooth_sigma': float(getattr(args, "smooth_sigma", 0.01)),
    }


def _maybe_save_chi2(args, suffix, model, samples, obs_phase, obs_flux, obs_err,
                     *, reparam, wind_model, fit_wind_shape, active_names,
                     likelihood, param_spec) -> None:
    """Write the per-sample chi2 table when --save-chi2 was requested."""
    if not getattr(args, 'save_chi2', False):
        return
    compute_chi2_for_samples(
        model, samples, obs_phase, obs_flux, obs_err,
        output_path=os.path.join(args.output_dir, f"{suffix}_chi2.csv.gz"),
        n_samples=getattr(args, 'chi2_n_samples', None),
        verbose=True,
        reparam=reparam, wind_model=wind_model, fit_wind_shape=fit_wind_shape,
        active_names=active_names, likelihood=likelihood,
        **_phase_shift_opts(args),
        param_spec=param_spec,
    )


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
    smoothed: Optional[pd.DataFrame] = None,
    scatter_prior: Optional[Dict[str, float]] = None,
) -> Tuple[Dict, object]:
    """Run MCMC fit for a single band/wind_model combination."""

    if priors is None:
        priors = _default_priors(reparam, kepler)
    if sim_params is None:
        sim_params = {}

    likelihood = getattr(args, 'likelihood', 'chi2')
    fit_scatter = bool(getattr(args, 'fit_scatter', False))
    frozen_params = dict(getattr(args, 'frozen_params', {}) or {})
    orbital_period_s = float(getattr(args, 'orbital_period', ORBITAL_PERIOD))

    param_spec = build_param_spec(
        likelihood=likelihood,
        reparam=reparam,
        kepler=kepler,
        wind_model=wind_model,
        fit_wind_shape=fit_wind_shape,
        fit_scatter=fit_scatter,
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
        fit_scatter=fit_scatter,
        scatter_prior=scatter_prior,
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
        **_phase_shift_opts(args),
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
        **_phase_shift_opts(args),
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
        **_phase_shift_opts(args),
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
            **_phase_shift_opts(args),
            param_spec=param_spec,
            **_smooth_plot_kwargs(smoothed, args),
        )
        stats['reduced_chi2'] = red_chi2

        if not getattr(args, 'no_geometry_plots', False):
            plot_geometry_diagnostics(
                stats, samples, band, wind_model, args.output_dir, suffix,
                param_names=active_names,
                reparam=reparam,
                fit_wind_shape=fit_wind_shape,
                param_spec=param_spec,
                sim_params=sim_params,
                dth=args.dth,
                flux_csv_path=args.flux_csv,
                verbose=not bool(getattr(args, 'quiet', False)),
            )

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
            # Always recorded (not just with --compute-bic): --replot checks it.
            n_obs=(bic_info.get("n_obs") if bic_info else float(len(obs_flux))),
        )
        print(f"Full chain saved to: {chain_path}")
    except Exception as e:
        warnings.warn(f"Could not save full chain: {e}")

    _maybe_save_chi2(
        args, suffix, model, samples, obs_phase, obs_flux, obs_err,
        reparam=reparam, wind_model=wind_model, fit_wind_shape=fit_wind_shape,
        active_names=active_names, likelihood=likelihood, param_spec=param_spec)

    stats['wind_model'] = wind_model
    return stats, model


def _labels_for_names(names: List[str], mode: str) -> List[str]:
    """Plot labels for an arbitrary set of sampled parameter names."""
    geom_names, geom_labels = get_mode_name_label(mode)
    extra = {'log_f': r'$\ln\,f$', 'f_scatter': r'$f_\mathrm{scat}$'}
    out = []
    for name in names:
        if name in geom_names:
            out.append(geom_labels[geom_names.index(name)])
        else:
            out.append(extra.get(name, WIND_SHAPE_LABELS.get(name, name)))
    return out


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
    smoothed: Optional[pd.DataFrame] = None,
) -> Optional[Dict]:
    """Regenerate plots from existing MCMC results without re-running MCMC.

    The fit_wind_shape flag here matches what *the saved chain* contains; it
    is only used to build the right wind_params for forward model evaluation.
    If the saved chain has more dims than the geometry block, fit_wind_shape
    is auto-detected.
    """
    if priors is None:
        priors = _default_priors(reparam, kepler)
    if sim_params is None:
        sim_params = {}

    print(f"\n{'#'*60}")
    print(f"# Replotting {band.upper()} band - {WIND_MODELS[wind_model]}")
    print('#'*60)

    suffix = f"{band}_{wind_model}"
    saved_mode = 'kepler' if kepler else ('reparam' if reparam else 'phys')
    saved_orbital_period_s = float(getattr(args, 'orbital_period', ORBITAL_PERIOD))
    saved_frozen = dict(getattr(args, 'frozen_params', {}) or {})
    saved_likelihood = getattr(args, 'likelihood', 'chi2')
    chain_path = os.path.join(args.output_dir, f"{suffix}_chain.npz")
    if os.path.exists(chain_path):
        try:
            _meta = np.load(chain_path, allow_pickle=True)
            saved_mode = str(_meta.get('mode', saved_mode))
            saved_likelihood = str(_meta.get('likelihood', saved_likelihood))
            saved_orbital_period_s = float(_meta.get('orbital_period_s', saved_orbital_period_s))
            fn = list(_meta.get('frozen_names', []))
            fv = list(_meta.get('frozen_values', []))
            if len(fn) == len(fv):
                saved_frozen = {str(k): float(v) for k, v in zip(fn, fv)}
            # Cheapest detector of a replot that bins the data differently from
            # the fit, which would report a chi2/dof for unseen data.
            saved_n_obs = float(_meta.get('n_obs', np.nan))
            if np.isfinite(saved_n_obs) and int(saved_n_obs) != len(obs_flux):
                warnings.warn(
                    f"Replot is using {len(obs_flux)} observed points but the saved "
                    f"fit used {int(saved_n_obs)}. The reported chi2/dof will not "
                    f"match the original run. Check --data-dir / --obs-column / "
                    f"--time-column / --counts-per-bin / --n-phase-bins / "
                    f"--no-phase-bin, or rerun --replot in a directory that has a "
                    f"{RUN_CONFIG_SUFFIX.lstrip('_')} file so they are restored "
                    f"automatically."
                )
        except Exception:
            pass

    spec_for_replot = build_param_spec(
        likelihood=getattr(args, 'likelihood', 'chi2'),
        reparam=(saved_mode == 'reparam'),
        kepler=(saved_mode == 'kepler'),
        wind_model=wind_model,
        fit_wind_shape=fit_wind_shape,
        fit_scatter=bool(getattr(args, 'fit_scatter', False)),
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
    geom_names, _ = get_mode_name_label(saved_mode)
    extra_dims = [n for n in loaded_names if n not in geom_names and n not in ('log_f', 'f_scatter')]
    saved_fit_wind_shape = bool(extra_dims) or fit_wind_shape
    saved_fit_scatter = ('f_scatter' in loaded_names) or bool(getattr(args, 'fit_scatter', False))
    spec_for_replot.active_names = list(loaded_names)
    spec_for_replot.fit_wind_shape = bool(saved_fit_wind_shape)
    spec_for_replot.fit_scatter = bool(saved_fit_scatter)

    print("\nUsing DirectLightCurveModel for replot.")
    model = DirectLightCurveModel(
        band=band,
        flux_csv_path=args.flux_csv,
        wind_model=wind_model,
        dth=args.dth,
        sim_params=sim_params,
    )

    spec_for_replot.likelihood = saved_likelihood

    if not args.no_plots:
        # Use the loaded column names as the active set.
        active_names = list(loaded_names)
        active_labels = _labels_for_names(active_names, saved_mode)

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
            **_phase_shift_opts(args),
            param_spec=spec_for_replot,
            **_smooth_plot_kwargs(smoothed, args),
        )
        stats['reduced_chi2'] = red_chi2

        if not getattr(args, 'no_geometry_plots', False):
            plot_geometry_diagnostics(
                stats, samples, band, wind_model, args.output_dir, suffix,
                param_names=active_names,
                reparam=(saved_mode == 'reparam'),
                fit_wind_shape=saved_fit_wind_shape,
                param_spec=spec_for_replot,
                sim_params=sim_params,
                dth=args.dth,
                flux_csv_path=args.flux_csv,
                verbose=not bool(getattr(args, 'quiet', False)),
            )

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
                **_phase_shift_opts(args),
                param_spec=spec_for_replot,
            )
            bic_info = (diag_out or {}).get("bic", {})
            if bic_info:
                stats.update(bic_info)
        elif compute_bic_flag:
            print(f"Chain file not found: {chain_path}")
            print("  Re-run MCMC to generate it, or skip --compute-bic.")

    _maybe_save_chi2(
        args, suffix, model, samples, obs_phase, obs_flux, obs_err,
        reparam=(saved_mode == 'reparam'), wind_model=wind_model,
        fit_wind_shape=saved_fit_wind_shape, active_names=loaded_names,
        likelihood=saved_likelihood, param_spec=spec_for_replot)

    return stats


def _parse_prior_overrides(parser, args, names, kind: str = "") -> Dict[str, Dict[str, float]]:
    """Parse `--prior-NAME MEAN,STD,MIN,MAX` for each of *names* that was given."""
    out: Dict[str, Dict[str, float]] = {}
    for name in names:
        raw = getattr(args, f'prior_{name}', None)
        if not raw:
            continue
        try:
            parts = [float(x.strip()) for x in raw.split(',')]
            if len(parts) != 4:
                raise ValueError(f"Expected 4 values for --prior-{name}")
        except Exception as e:
            parser.error(f"Invalid format for --prior-{name}: {e}")
        out[name] = dict(zip(('mean', 'std', 'min', 'max'), parts))
        print(f"Custom prior for {kind}{name}: mean={parts[0]}, std={parts[1]}, "
              f"min={parts[2]}, max={parts[3]}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="MCMC fitting of XRB light curves to observed Chandra data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--band",
        type=str,
        default=None,
        choices=['broad', 'soft', 'medium', 'hard', 'all'],
        help="Energy band to fit (or 'all' to fit all bands). Required, except "
             "with --replot, where it is restored from the saved run config."
    )
    parser.add_argument(
        "--flux-csv",
        type=str,
        default=None,
        help="Path to flux vs nH CSV file (from compute_flux_vs_nH.py). Required, "
             "except with --replot, where it is restored from the saved run config."
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
             "Supported names: d1,d2,a,q,r,R,i0,M_X,M_RH,f_scatter,Rb,p,beta,fconf,ell. "
             "log_f cannot be frozen."
    )
    parser.add_argument(
        "--fit-scatter",
        action="store_true",
        help="Add a free constant scattered-flux parameter f_scatter to the model.",
    )
    parser.add_argument(
        "--scatter-eclipse-phase",
        nargs=2,
        type=float,
        default=(0.4, 0.6),
        metavar=("PHASE_MIN", "PHASE_MAX"),
        help="Phase window used to center the f_scatter prior on the observed eclipse-floor flux.",
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
        "--no-geometry-plots",
        action="store_true",
        help="Skip the binary-geometry figures (projected orbit / eclipse diagram, "
             "geometry vs phase, and the wind profile with its posterior band). "
             "They cost one extra simulate_lightcurve call (~60 ms) plus a cheap "
             "analytic profile evaluation per draw."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating diagnostic plots"
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Overlay a Gaussian-smoothed observed light curve and MC uncertainty band in best-fit plots.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.01,
        help="Gaussian kernel width in phase for --smooth.",
    )
    parser.add_argument(
        "--smooth-n-mc",
        type=int,
        default=2000,
        help="Number of Monte Carlo perturbations used to estimate the smoothed 1-sigma band (0 disables band).",
    )
    parser.add_argument(
        "--smooth-seed",
        type=int,
        default=None,
        help="RNG seed for smoothing Monte Carlo perturbations.",
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
             "Requires samples CSV files to exist in the output directory. Every option not given "
             "explicitly is restored from that run's <band>_<wind>_run_config.json, so `--replot` "
             "alone reproduces the original band, wind model, data selection, binning and priors; "
             "any flag you do pass overrides the saved value."
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
        'Override default priors for the sampled geometry parameters. '
        'Which set applies depends on --reparam / --kepler.'
    )
    # (flag, dest override, prior dict, description). Defaults are read from the
    # prior dicts, so the help text cannot drift from the values actually used.
    for flag, dest, prior_defs, desc in (
        ("d1", None, DEFAULT_PRIORS, "d1 (compact object distance from COM)"),
        ("d2", None, DEFAULT_PRIORS, "d2 (companion distance from COM)"),
        ("r", None, DEFAULT_PRIORS, "r (compact object/disk radius)"),
        ("R", None, DEFAULT_PRIORS, "R (companion star radius)"),
        ("i0", None, DEFAULT_PRIORS, "i0 (orbital inclination, degrees)"),
        ("a", None, REPARAM_PRIORS, "a = d1+d2 (orbital separation, --reparam only)"),
        ("q", None, REPARAM_PRIORS, "q = d1/(d1+d2) (mass-ratio proxy, --reparam only)"),
        ("MX", "prior_M_X", KEPLER_PRIORS, "compact-object mass M_X (Msun, --kepler only)"),
        ("MRH", "prior_M_RH", KEPLER_PRIORS, "companion mass M_RH (Msun, --kepler only)"),
    ):
        d = prior_defs[dest[len("prior_"):] if dest else flag]
        prior_group.add_argument(
            f"--prior-{flag}",
            type=str,
            default=None,
            metavar="MEAN,STD,MIN,MAX",
            help=(f"Prior for {desc}. Format: mean,std,min,max. "
                  f"Default: {d['mean']},{d['std']},{d['min']},{d['max']}"),
            **({'dest': dest} if dest else {}),
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

    # Restore everything not passed explicitly, before any validation or
    # derived values are computed from args.
    restored_config = None
    if args.replot:
        restored_config = apply_saved_run_config(parser, args)
        if restored_config is None:
            print(
                f"\nNo saved run config found in {args.output_dir} "
                f"(looked for *{RUN_CONFIG_SUFFIX}). Using the command line and "
                f"argparse defaults; data selection, binning and priors may not "
                f"match the original fit."
            )

    # Not required=True, so --replot can supply them from a saved run config.
    for dest, flag in (('band', '--band'), ('flux_csv', '--flux-csv')):
        if getattr(args, dest, None) is None:
            parser.error(
                f"{flag} is required (with --replot it is restored from a saved "
                f"*{RUN_CONFIG_SUFFIX} in --output-dir, if one exists)."
            )

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
    if getattr(args, "smooth_sigma", 0.0) <= 0:
        parser.error("--smooth-sigma must be > 0.")
    if getattr(args, "smooth_n_mc", 0) < 0:
        parser.error("--smooth-n-mc must be >= 0.")
    if len(args.scatter_eclipse_phase) != 2:
        parser.error("--scatter-eclipse-phase requires two values: PHASE_MIN PHASE_MAX")
    scatter_lo, scatter_hi = map(float, args.scatter_eclipse_phase)
    if not (0.0 <= scatter_lo <= scatter_hi <= 1.0):
        parser.error("--scatter-eclipse-phase must satisfy 0 <= PHASE_MIN <= PHASE_MAX <= 1.")
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
    priors = _default_priors(reparam, kepler)
    base_names, _ = get_mode_name_label(
        'kepler' if kepler else ('reparam' if reparam else 'phys'))
    priors.update(_parse_prior_overrides(parser, args, base_names))

    # Wind-shape overrides are always parsed; they are only applied when
    # --fit-wind-shape and the param is in WIND_SHAPE_FIT[wind_model].
    shape_prior_overrides = _parse_prior_overrides(
        parser, args, ('Rb', 'p', 'beta', 'fconf', 'ell'), kind="shape param ")

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
    fit_scatter = bool(getattr(args, 'fit_scatter', False))
    frozen_params = dict(getattr(args, 'frozen_params', {}) or {})

    # Validate frozen names against run configuration before sampling starts.
    try:
        _spec_check = build_param_spec(
            likelihood=getattr(args, 'likelihood', 'chi2'),
            reparam=reparam,
            kepler=kepler,
            wind_model=args.wind_model,
            fit_wind_shape=fit_wind_shape,
            fit_scatter=fit_scatter,
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
        fit_scatter=fit_scatter,
        scatter_prior=(
            {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': np.inf}
            if fit_scatter else None
        ),
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
                bin_cols = dict(rate_column='flux', error_column='flux_err')
                if args.counts_per_bin is not None:
                    obs_df = phase_bin_data_snr(
                        obs_df, counts_per_bin=args.counts_per_bin, **bin_cols)
                else:
                    obs_df = phase_bin_data(
                        obs_df, n_bins=(args.n_phase_bins or 50), **bin_cols)

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

            smoothed = None
            if getattr(args, "smooth", False):
                smoothed = smooth_lightcurve(
                    obs_phase,
                    obs_flux,
                    obs_err,
                    sigma=float(getattr(args, "smooth_sigma", 0.01)),
                    n_mc=int(getattr(args, "smooth_n_mc", 2000)),
                    random_state=getattr(args, "smooth_seed", None),
                    verbose=not bool(getattr(args, "quiet", False)),
                )

            scatter_prior = None
            if fit_scatter:
                scatter_center = estimate_scattered_flux(
                    obs_phase,
                    obs_flux,
                    window=tuple(map(float, args.scatter_eclipse_phase)),
                )
                flux_max = float(np.nanmax(obs_flux)) if np.any(np.isfinite(obs_flux)) else 1.0
                tiny = max(1e-30, abs(float(np.nanmedian(obs_flux))) * 1e-6)
                scatter_prior = {
                    'mean': float(scatter_center),
                    'std': float(max(scatter_center, tiny)),
                    'min': 0.0,
                    'max': float(max(flux_max, scatter_center + tiny)),
                }
                if not getattr(args, "quiet", False):
                    print(
                        "Scatter prior: mean={mean:.4g}, std={std:.4g}, "
                        "min={min:.4g}, max={max:.4g}".format(**scatter_prior)
                    )

            for wind_model in wind_models:
                try:
                    key = (band, wind_model)

                    if args.replot:
                        # Self-healing: results predating run-config saving get
                        # one written, so the next --replot needs no arguments.
                        if restored_config is None and not os.path.exists(
                            run_config_path(args.output_dir, band, wind_model)
                        ):
                            save_run_config(args.output_dir, band, wind_model, args)

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
                            smoothed=smoothed,
                        )
                        if stats is not None:
                            all_results[key] = stats
                    else:
                        # Written before sampling so the configuration survives
                        # an interrupted or crashed run.
                        save_run_config(args.output_dir, band, wind_model, args)
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
                            smoothed=smoothed,
                            scatter_prior=scatter_prior,
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
                fit_scatter=fit_scatter,
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
                        f.write(f"  {param}: {_fmt_val(s['median'])} "
                                f"(+{_fmt_val(s['upper'])}/-{_fmt_val(s['lower'])})")
                        if ('mean' in s) and ('std' in s):
                            f.write(f"  [mean={_fmt_val(s['mean'])}, std={_fmt_val(s['std'])}]")
                        f.write("\n")
                if reparam or kepler:
                    for derived in ('d1', 'd2'):
                        if derived in stats:
                            s = stats[derived]
                            f.write(f"  {derived} (derived): {_fmt_val(s['median'])} "
                                    f"(+{_fmt_val(s['upper'])}/-{_fmt_val(s['lower'])})")
                            if ('mean' in s) and ('std' in s):
                                f.write(f"  [mean={_fmt_val(s['mean'])}, std={_fmt_val(s['std'])}]")
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
                            f.write(f"  {param}: {_fmt_val(stats[param]['map'])}\n")
                    if reparam or kepler:
                        for derived in ('d1', 'd2'):
                            if derived in stats and 'map' in stats[derived]:
                                f.write(
                                    f"  {derived} (derived): "
                                    f"{_fmt_val(stats[derived]['map'])}\n"
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
