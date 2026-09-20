#!/usr/bin/env python3
"""
MCMC Light Curve Fitting for X-ray Binary Systems
--------------------------------------------------
Fits the wind-absorbed, eclipsed light-curve model of ``cloak/kernel.py`` to
phase-folded Chandra data with an ensemble sampler (emcee or zeus), one energy
band and one wind model per run.

WIND MODELS (selected via --wind-model):
- smooth_pl   : Smoothly broken power-law density profile (Rb, p, Delta)
- confinement : Inner-confinement / compression amplification (R_star, fconf, ell)
- beta_law    : Velocity-based n = Mdot/(4 pi r^2 v(r)), CAK beta law with an
                inner acceleration scale (R_star, beta, H)

GEOMETRY parameterizations (mutually exclusive):
- phys (default) : d1, d2, r, R, i0
- --reparam      : a = d1 + d2, q = d1/a, r, R, i0   (d1, d2 derived)
- --kepler       : M_X, M_RH, r, R, i0               (a, q, d1, d2 derived)
- --kepler-mtot  : M_tot, q_m = M_RH/M_tot, r, R, i0 (a, M_X, M_RH, d1, d2 derived)
i0 is the standard astronomical inclination: degrees from the orbital-plane
normal, 90 = edge-on. For a circular orbit the light curve depends on the
separation a alone, so q / q_m are exactly unidentifiable and their posteriors
equal their priors; --kepler-mtot puts that flat direction on its own axis.

WIND-SHAPE parameters (added with --fit-wind-shape):
- smooth_pl   : Rb, p (Delta fixed at 2)
- confinement : fconf, ell (R_star tied to R)
- beta_law    : beta, H (R_star tied to R; freeze H for a one-parameter fit)

The wind column is normalized physically from --mdot / --v-inf, so N_H carries
real units. --fit-fopacity adds log10 of the effective photoelectric-opacity
factor as a free parameter and is strongly recommended. Note that scaling every
length together with f_opacity leaves the light curve invariant, so the absolute
scale (and M_tot) is set by the priors on R and f_opacity, not by the data.

Usage:
    python -m cloak.mcmc_fit --band broad --flux-csv flux_vs_nH_broad.csv \\
        --wind-model smooth_pl --fit-wind-shape --fit-fopacity --reparam \\
        --likelihood jitter --sampler zeus

    python -m cloak.mcmc_fit --band broad --flux-csv flux_vs_nH_broad.csv \\
        --wind-model beta_law --fit-wind-shape --fit-fopacity --kepler-mtot \\
        --freeze q_m=0.6

    # Regenerate every figure from a finished run; all other options are
    # restored from <band>_<wind_model>_run_config.json in --output-dir.
    python -m cloak.mcmc_fit --replot --output-dir mcmc_results/broad
"""

import argparse
import copy
import multiprocessing as mp
import os
import random
import signal
import sys
import time
import warnings
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Tuple

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

if __package__ in (None, ""):   # run as a plain script: python cloak/mcmc_fit.py
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from cloak.kernel import (
    simulate_lightcurve,
    simulate_band_flux,
    SIM_DEFAULTS,
    WIND_MODEL_IDS,
    WIND_MODEL_PARAM_KEYS,
    default_wind_params,
    evaluate_g_profile,
    flux_table_bands,
)
from cloak.utils import (
    ORBITAL_PERIOD,
    RUN_CONFIG_SUFFIX,
    WIND_NORMALIZATION,
    PhaseShiftSearch,
    apply_phase_window,
    apply_saved_run_config,
    best_phase_shift,
    build_phase_shift_search,
    dest_to_flag,
    estimate_scattered_flux,
    eval_periodic,
    explicit_cli_dests,
    fmt_val,
    in_phase_window,
    load_observed_lightcurves,
    model_dump_path,
    periodic_model,
    phase_bin_data,
    phase_bin_data_snr,
    run_config_path,
    sanitize_errors,
    save_run_config,
    save_samples_csv_chunked,
    smooth_lightcurve,
    validate_binning_args,
    validate_phase_window_args,
    write_model_blocks,
)
import matplotlib
matplotlib.use("Agg")   # every figure is saved to a file; no display needed (SSH, clusters)

from cloak.plots import (
    plot_corner,
    plot_geometry_vs_phase,
    plot_lightcurve_fit,
    plot_orbit_geometry,
    plot_trace,
    plot_wind_profile,
)


# =============================================================================
# Registries: parameterizations, priors, wind-shape parameters
# =============================================================================

G_SI = 6.674e-11
R_SUN_M = 6.957e8
M_SUN_KG = 1.989e30

# Geometry priors shared by every parameterization (IC 10 X-1 working values).
# R is the companion *photosphere* -- the eclipse comes from wind opacity, not
# from a geometric cutoff -- and r the emitter disk. i0 is the standard
# inclination (90 = edge-on), so the box must admit near-edge-on solutions.
R_PRIOR = {'mean': 2.0, 'std': 0.5, 'min': 1.0, 'max': 5.0}               # Solar radii
SMALL_R_PRIOR = {'mean': 0.001, 'std': 0.001, 'min': 0.0001, 'max': 0.1}  # Solar radii
I0_PRIOR = {'mean': 78.0, 'std': 8.0, 'min': 40.0, 'max': 89.9}            # Degrees

# One entry per geometry parameterization: the two scale parameters it samples
# (the shared r, R, i0 are appended), the derived quantities it reports, and the
# CLI flag that selects it. The light curve depends on d1 + d2 alone, so q and
# q_m are exactly unidentifiable; the reparameterized modes put that flat
# direction on its own axis instead of diagonally across (d1, d2) or
# (M_X, M_RH).
MODES: Dict[str, Dict[str, object]] = {
    'phys': {
        'flag': None,
        'scale_names': ['d1', 'd2'],
        'scale_priors': {
            'd1': {'mean': 11.0, 'std': 3.0, 'min': 5.0, 'max': 20.0},
            'd2': {'mean': 8.0, 'std': 3.0, 'min': 3.0, 'max': 15.0},
        },
        'derived': ('a', 'q'),
    },
    'reparam': {
        'flag': '--reparam',
        'scale_names': ['a', 'q'],
        'scale_priors': {
            'a': {'mean': 19.0, 'std': 4.0, 'min': 8.0, 'max': 35.0},
            'q': {'mean': 0.58, 'std': 0.15, 'min': 0.01, 'max': 0.99},
        },
        'derived': ('d1', 'd2'),
    },
    'kepler': {
        'flag': '--kepler',
        'scale_names': ['M_X', 'M_RH'],
        'scale_priors': {
            'M_X': {'mean': 30.0, 'std': 10.0, 'min': 1.0, 'max': 100.0},
            'M_RH': {'mean': 20.0, 'std': 10.0, 'min': 1.0, 'max': 100.0},
        },
        'derived': ('a', 'q', 'd1', 'd2'),
    },
    'kepler_mtot': {
        'flag': '--kepler-mtot',
        'scale_names': ['M_tot', 'q_m'],
        'scale_priors': {
            'M_tot': {'mean': 45.0, 'std': 15.0, 'min': 5.0, 'max': 120.0},
            'q_m': {'mean': 0.6, 'std': 0.2, 'min': 0.02, 'max': 0.98},
        },
        'derived': ('a', 'M_X', 'M_RH', 'd1', 'd2'),
    },
}

# Plot labels for every parameter that can appear in a chain.
PARAM_LABELS: Dict[str, str] = {
    'd1': r'$d_1$ (R$_\odot$)',
    'd2': r'$d_2$ (R$_\odot$)',
    'a': r'$a$ (R$_\odot$)',
    'q': r'$q$',
    'M_X': r'$M_X$ (M$_\odot$)',
    'M_RH': r'$M_\mathrm{RH}$ (M$_\odot$)',
    'M_tot': r'$M_\mathrm{tot}$ (M$_\odot$)',
    'q_m': r'$q_m = M_\mathrm{RH}/M_\mathrm{tot}$',
    'r': r'$r$ (R$_\odot$)',
    'R': r'$R$ (R$_\odot$)',
    'i0': r'$i$ (deg)',
    'log_f': r'$\ln\,f$',
    'f_scatter': r'$f_\mathrm{scat}$',
    'log_fopa': r'$\log_{10} f_\mathrm{opa}$',
    'Rb': r'$R_b$ (R$_\odot$)',
    'p': r'$p$',
    'fconf': r'$f_\mathrm{conf}$',
    'ell': r'$\ell$ (R$_\odot$)',
    'beta': r'$\beta$',
    'H': r'$H$ (R$_\odot$)',
}

# Wind model descriptions (matches cloak.kernel.WIND_MODEL_IDS keys).
WIND_MODELS = {
    'smooth_pl':   'Smoothly Broken Power-Law Wind',
    'confinement': 'Inner-Confinement / Compression Wind',
    'beta_law':    'CAK Beta-Law (Velocity-Based) Wind',
}

# Shape parameters that become free MCMC dimensions under --fit-wind-shape.
# R_star (confinement, beta_law) is tied to the geometry parameter R; any other
# shape parameter (smooth_pl's poorly identifiable Delta) keeps the simulator
# default from cloak.kernel.default_wind_params.
WIND_SHAPE_FIT = {
    'smooth_pl':   ['Rb', 'p'],
    'confinement': ['fconf', 'ell'],
    'beta_law':    ['beta', 'H'],
}
assert set(WIND_MODELS) == set(WIND_MODEL_IDS), "WIND_MODELS must describe every cloak.kernel wind model"
for _model, _names in WIND_SHAPE_FIT.items():
    assert set(_names) <= set(WIND_MODEL_PARAM_KEYS[_model]), f"WIND_SHAPE_FIT[{_model!r}] names unknown to cloak.kernel"

# Default priors for wind-shape parameters; override with --prior-<name>.
# beta ~ 0.8-1 is the CAK range for OB/WR winds; H is the acceleration scale
# height, so the effective break R + 3H mirrors Rb (smooth_pl) and ell
# (confinement).
WIND_SHAPE_PRIORS = {
    'Rb':    {'mean': 5.0, 'std': 3.0,  'min': 0.5, 'max': 30.0},
    'p':     {'mean': 4.0, 'std': 1.0,  'min': 2.0, 'max': 8.0},
    'fconf': {'mean': 5.0, 'std': 5.0,  'min': 0.0, 'max': 50.0},
    'ell':   {'mean': 1.0, 'std': 0.7,  'min': 0.1, 'max': 10.0},
    'beta':  {'mean': 0.8, 'std': 0.3,  'min': 0.3, 'max': 2.0},
    'H':     {'mean': 1.0, 'std': 0.7,  'min': 0.1, 'max': 10.0},
}
ALL_WIND_SHAPE_NAMES = tuple(WIND_SHAPE_PRIORS)

LIKELIHOOD_TYPES = {
    'chi2': 'Chi-squared (Gaussian)',
    'jitter': 'Gaussian with systematic jitter',
}
JITTER_PRIOR = {'mean': -3.0, 'std': 2.0, 'min': -10.0, 'max': 0.0}

# log10 of the effective-opacity factor. It rescales the Mdot-derived column to
# the *effective* photoelectric column, absorbing wind ionization, clumping and
# the departure of a He-rich WR wind from the solar abundances of the TBabs
# table. Centred near -1.5 because Clark & Crowther's Mdot predicts
# N_H ~ 20-50e22 out of eclipse against an observed ~0.75e22.
FOPACITY_PRIOR = {'mean': -1.5, 'std': 1.0, 'min': -4.0, 'max': 0.5}

SAMPLER_TYPES = {
    'emcee': 'emcee Ensemble Sampler (stretch moves)',
    'zeus': 'zeus Ensemble Slice Sampler',
}

# --save-chi2 subset when every row needs a model call (jitter likelihood);
# with the chi2 likelihood the table is read from the chain at no cost.
CHI2_TABLE_DEFAULT_SAMPLES = 2000


def _compute_kepler_prefactor(orbital_period_s: float) -> float:
    """Return factor K for a = K * (Mtot/Msun)^(1/3) in solar radii."""
    p = float(orbital_period_s)
    return ((G_SI * M_SUN_KG * p ** 2) / (4.0 * np.pi ** 2)) ** (1.0 / 3.0) / R_SUN_M


def mode_from_flags(reparam: bool = False, kepler: bool = False,
                    kepler_mtot: bool = False) -> str:
    """Map the three mutually exclusive CLI flags to a mode name."""
    on = [m for m, f in (('reparam', reparam), ('kepler', kepler),
                         ('kepler_mtot', kepler_mtot)) if f]
    if len(on) > 1:
        raise ValueError("--reparam, --kepler and --kepler-mtot are mutually exclusive.")
    return on[0] if on else 'phys'


def geometry_names(mode: str) -> List[str]:
    """The five sampled geometry parameters of *mode*, in chain order."""
    return list(MODES[mode]['scale_names']) + ['r', 'R', 'i0']


def default_geometry_priors(mode: str) -> Dict[str, Dict[str, float]]:
    """A fresh copy of the geometry priors for *mode*."""
    priors = copy.deepcopy(MODES[mode]['scale_priors'])
    priors.update({'r': dict(SMALL_R_PRIOR), 'R': dict(R_PRIOR), 'i0': dict(I0_PRIOR)})
    return priors


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
            raise ValueError(f"Invalid --freeze entry '{item}'. Expected NAME=VALUE.")
        name, value = item.split("=", 1)
        key = name.strip()
        if not key:
            raise ValueError(f"Invalid --freeze entry '{item}': missing parameter name.")
        if key in out:
            raise ValueError(f"Duplicate frozen parameter '{key}' in --freeze.")
        out[key] = float(value)
    return out


# =============================================================================
# ParamSpec: the single description of what is sampled, frozen and derived
# =============================================================================

@dataclass
class ParamSpec:
    """Layout of the sampled vector and how to turn a sample into a model.

    ``active_names`` is the MCMC vector, in order: geometry -> log_f (jitter)
    -> f_scatter -> log_fopa -> wind-shape parameters, minus anything frozen.
    Every consumer (prior, likelihood, statistics, plots, replot) resolves
    parameter values through this object, so there is exactly one place where
    "which value does this name have" is answered.
    """
    mode: str = 'phys'
    active_names: List[str] = field(default_factory=list)
    frozen: Dict[str, float] = field(default_factory=dict)
    fit_wind_shape: bool = False
    fit_scatter: bool = False
    fit_fopacity: bool = False
    wind_model: str = 'smooth_pl'
    likelihood: str = 'chi2'
    orbital_period_s: float = float(ORBITAL_PERIOD)
    K_kepler: float = 0.0

    # -- vector bookkeeping --------------------------------------------------
    @property
    def active_labels(self) -> List[str]:
        return [PARAM_LABELS.get(n, n) for n in self.active_names]

    @property
    def n_dim(self) -> int:
        return len(self.active_names)

    def index(self, name: str) -> Optional[int]:
        return self.active_names.index(name) if name in self.active_names else None

    def value(self, theta, name: str) -> Optional[float]:
        """Value of *name* from the sample or the frozen set; None if neither."""
        if name in self.frozen:
            return float(self.frozen[name])
        i = self.index(name)
        return float(theta[i]) if i is not None else None

    # -- physical resolution --------------------------------------------------
    def geometry(self, theta) -> Tuple[float, float, float, float, float]:
        """(d1, d2, r, R, i0) for a sample; NaNs when the sample is unphysical."""
        nan5 = (np.nan,) * 5
        if self.mode == 'phys':
            d1, d2 = self.value(theta, 'd1'), self.value(theta, 'd2')
        elif self.mode == 'reparam':
            a, q = self.value(theta, 'a'), self.value(theta, 'q')
            if a is None or q is None:
                return nan5
            d1, d2 = a * q, a * (1.0 - q)
        elif self.mode == 'kepler':
            mx, mrh = self.value(theta, 'M_X'), self.value(theta, 'M_RH')
            if mx is None or mrh is None or (mx + mrh) <= 0:
                return nan5
            a = self.K_kepler * (mx + mrh) ** (1.0 / 3.0)
            q = mrh / (mx + mrh)
            d1, d2 = a * q, a * (1.0 - q)
        elif self.mode == 'kepler_mtot':
            mtot, q = self.value(theta, 'M_tot'), self.value(theta, 'q_m')
            if mtot is None or q is None or mtot <= 0:
                return nan5
            a = self.K_kepler * mtot ** (1.0 / 3.0)
            d1, d2 = a * q, a * (1.0 - q)
        else:
            raise ValueError(f"Unknown parameter mode '{self.mode}'")
        r, R, i0 = self.value(theta, 'r'), self.value(theta, 'R'), self.value(theta, 'i0')
        if None in (d1, d2, r, R, i0):
            return nan5
        return float(d1), float(d2), float(r), float(R), float(i0)

    def wind_params(self, theta, R_value: float) -> Optional[Dict[str, float]]:
        """Wind-shape dict for the simulator, or None for the model defaults.

        Fitted shape values come from the sample and frozen ones from
        ``frozen``; every other parameter keeps the simulator default
        (``default_wind_params``, which also ties R_star to R), so freezing one
        parameter never changes the value of another and the CLI simulator
        reproduces the fitted curve.
        """
        shape_names = WIND_SHAPE_FIT[self.wind_model]
        if not self.fit_wind_shape and not any(n in self.frozen for n in shape_names):
            return None
        wp = default_wind_params(self.wind_model, R_value)
        for name in shape_names:
            v = self.value(theta, name)
            if v is not None:
                wp[name] = v
        return wp

    def f_scatter(self, theta) -> float:
        v = self.value(theta, 'f_scatter')
        return 0.0 if v is None else v

    def f_opacity(self, theta) -> Optional[float]:
        """Effective-opacity factor, or None to leave the simulator default."""
        v = self.value(theta, 'log_fopa')
        return None if v is None else float(10.0 ** v)

    # -- derived quantities ---------------------------------------------------
    def derived(self, rows: np.ndarray) -> Dict[str, np.ndarray]:
        """Derived geometry/mass columns for an (n, n_dim) array of samples.

        One implementation serves the marginal statistics, the MAP row, the
        model-curve header and the summary file, so they cannot disagree.
        """
        rows = np.atleast_2d(np.asarray(rows, dtype=float))
        n = rows.shape[0]

        def col(name):
            i = self.index(name)
            if i is not None:
                return rows[:, i]
            if name in self.frozen:
                return np.full(n, float(self.frozen[name]))
            raise KeyError(f"Cannot resolve '{name}' in mode '{self.mode}'")

        out: Dict[str, np.ndarray] = {}
        if self.mode == 'phys':
            d1, d2 = col('d1'), col('d2')
            out['a'] = d1 + d2
            out['q'] = d1 / (d1 + d2)
        elif self.mode == 'reparam':
            a, q = col('a'), col('q')
            out['d1'], out['d2'] = a * q, a * (1.0 - q)
        elif self.mode == 'kepler':
            mtot = col('M_X') + col('M_RH')
            out['a'] = self.K_kepler * np.power(mtot, 1.0 / 3.0)
            out['q'] = col('M_RH') / mtot
            out['d1'], out['d2'] = out['a'] * out['q'], out['a'] * (1.0 - out['q'])
        elif self.mode == 'kepler_mtot':
            mtot, q = col('M_tot'), col('q_m')
            out['a'] = self.K_kepler * np.power(mtot, 1.0 / 3.0)
            # Only M_tot is informed by the light curve; the split is the q_m prior.
            out['M_RH'], out['M_X'] = q * mtot, (1.0 - q) * mtot
            out['d1'], out['d2'] = out['a'] * q, out['a'] * (1.0 - q)
        return out

    @property
    def derived_names(self) -> Tuple[str, ...]:
        return tuple(MODES[self.mode]['derived'])


def build_param_spec(
    likelihood: str = 'chi2',
    mode: str = 'phys',
    wind_model: str = 'smooth_pl',
    fit_wind_shape: bool = False,
    fit_scatter: bool = False,
    fit_fopacity: bool = False,
    frozen: Optional[Dict[str, float]] = None,
    orbital_period_s: float = ORBITAL_PERIOD,
) -> ParamSpec:
    """Build the active-parameter layout for a run, validating frozen names."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose one of {list(MODES)}")
    if wind_model not in WIND_MODELS:
        raise ValueError(f"Unknown wind_model '{wind_model}'. Choose one of {list(WIND_MODELS)}")
    frozen = dict(frozen or {})

    names = geometry_names(mode)
    if likelihood == 'jitter':
        names.append('log_f')
    if fit_scatter:
        names.append('f_scatter')
    if fit_fopacity:
        names.append('log_fopa')
    if fit_wind_shape:
        names.extend(WIND_SHAPE_FIT[wind_model])

    # Shape parameters, f_scatter and log_fopa may be frozen even when not fitted.
    valid_frozen = set(names) | set(WIND_SHAPE_FIT[wind_model]) | {'f_scatter', 'log_fopa'}
    if 'log_f' in frozen:
        raise ValueError("Freezing log_f is not supported. Use --likelihood chi2 instead.")
    unknown = [k for k in frozen if k not in valid_frozen]
    if unknown:
        raise ValueError(
            f"Unknown frozen parameter(s): {unknown}. "
            f"Allowed names for this run: {sorted(valid_frozen)}"
        )
    if ('R' in frozen) and ('Rb' in frozen) and (frozen['Rb'] < frozen['R']):
        raise ValueError("Invalid freeze combination: require Rb >= R when both are frozen.")

    return ParamSpec(
        mode=mode,
        active_names=[n for n in names if n not in frozen],
        frozen=frozen,
        fit_wind_shape=fit_wind_shape,
        fit_scatter=fit_scatter,
        fit_fopacity=fit_fopacity,
        wind_model=wind_model,
        likelihood=likelihood,
        orbital_period_s=float(orbital_period_s),
        K_kepler=_compute_kepler_prefactor(orbital_period_s),
    )


def get_active_priors(
    spec: ParamSpec,
    geometry_priors: Dict[str, Dict[str, float]],
    shape_prior_overrides: Optional[Dict[str, Dict[str, float]]] = None,
    scatter_prior: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Merged {name: prior} for every active dimension of *spec*."""
    overrides = shape_prior_overrides or {}
    out = copy.deepcopy(geometry_priors)
    if spec.likelihood == 'jitter':
        out.setdefault('log_f', dict(JITTER_PRIOR))
    if spec.fit_fopacity:
        out['log_fopa'] = {**FOPACITY_PRIOR, **overrides.get('log_fopa', {})}
    if spec.fit_wind_shape:
        for name in WIND_SHAPE_FIT[spec.wind_model]:
            out[name] = {**WIND_SHAPE_PRIORS[name], **overrides.get(name, {})}
    if spec.fit_scatter and scatter_prior is not None:
        out.setdefault('f_scatter', dict(scatter_prior))
    for name in spec.frozen:
        out.pop(name, None)
    return out


# =============================================================================
# Forward model
# =============================================================================

class DirectLightCurveModel:
    """Evaluate the physical light curve for one band by calling the simulator.

    At a few ms per light curve (Gauss-Legendre kernel over half the orbit, the
    other half by phase reflection, per-cell flux conversion compiled) direct
    evaluation is fast enough for MCMC and is the only path that supports
    per-sample wind-shape parameters.
    """

    # Simulation constants a run may fix through sim_params (everything else
    # is either sampled or set by the constructor).
    SIM_CONSTANTS = ('gma0', 'd2h', 'mdot', 'v_inf', 'mu_wind', 'f_opacity')

    def __init__(self, band: str, flux_csv_path: str, wind_model: str = 'smooth_pl',
                 dth: float = 2.0, flux_method: str = "interpolate",
                 sim_params: Optional[Dict] = None):
        self.band = band.lower()
        self.flux_csv_path = flux_csv_path
        self.wind_model = wind_model.lower()
        self.dth = dth
        self.flux_method = flux_method
        self.sim_params = sim_params or {}
        if self.wind_model not in WIND_MODELS:
            raise ValueError(f"wind_model must be one of {list(WIND_MODELS)}, got '{wind_model}'")
        if self.flux_method not in ('interpolate', 'refit'):
            raise ValueError(f"flux_method must be 'interpolate' or 'refit', got '{flux_method}'")
        if not os.path.exists(flux_csv_path):
            raise FileNotFoundError(f"Flux CSV not found: {flux_csv_path}")
        unknown = set(self.sim_params) - set(self.SIM_CONSTANTS)
        if unknown:
            raise ValueError(f"sim_params keys {sorted(unknown)} are not simulation constants; "
                             f"allowed: {self.SIM_CONSTANTS}")
        # Fail here, not once per likelihood call: a band missing from the table
        # would otherwise make every log-probability -inf, and the sampler would
        # still run to completion with frozen walkers.
        bands = flux_table_bands(flux_csv_path)
        if self.band not in bands:
            raise ValueError(f"Band '{band}' is not in {flux_csv_path} (available: {bands}). "
                             "The model is run one band at a time: pass the table for this band.")

    def sim_kwargs(self, d1, d2, r, R, i0, wind_params=None, f_opacity=None,
                   scattered_flux: float = 0.0) -> Dict[str, object]:
        """Keyword arguments for simulate_lightcurve / simulate_band_flux.

        Constants not set in ``sim_params`` take the simulator's own defaults.
        """
        const = {k: self.sim_params.get(k, SIM_DEFAULTS[k]) for k in self.SIM_CONSTANTS}
        if f_opacity is not None:
            const['f_opacity'] = float(f_opacity)
        return dict(
            r=r, R=R, d1=d1, d2=d2, i0=i0, dth=self.dth,
            flux_method=self.flux_method, flux_csv_path=self.flux_csv_path, band=self.band,
            wind_model=self.wind_model, wind_params=wind_params, scattered_flux=scattered_flux,
            **const,
        )

    def curve(self, d1, d2, r, R, i0,
              wind_params: Optional[Dict[str, float]] = None,
              f_opacity: Optional[float] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Native ``(phase, flux)`` arrays on the kernel's phase grid, or None if
        the simulation fails."""
        try:
            return simulate_band_flux(**self.sim_kwargs(d1, d2, r, R, i0, wind_params, f_opacity))
        except (ValueError, ArithmeticError) as e:
            # The simulator's own rejections (r >= R, non-finite geometry) and
            # numerical failures; a TypeError or KeyError is a bug and propagates.
            warnings.warn(f"Model evaluation failed: {e}")
            return None


@dataclass
class FitData:
    """The observed light curve as the likelihood sees it, plus its invariants."""
    phase: np.ndarray
    flux: np.ndarray
    err: np.ndarray
    err2: np.ndarray
    shift_search: Optional[PhaseShiftSearch] = None   # None: phase shift held at fixed_shift
    fixed_shift: float = 0.0
    is_binned: bool = True
    phase_width: Optional[np.ndarray] = None

    @classmethod
    def build(cls, phase, flux, err, fit_phase_shift: bool = True,
              shift_grid_size: Optional[int] = None, n_model: int = 0,
              fixed_shift: float = 0.0,
              is_binned: bool = True, phase_width=None) -> "FitData":
        phase = np.asarray(phase, dtype=float)
        err = np.asarray(err, dtype=float)
        search = (build_phase_shift_search(phase, n_grid=shift_grid_size, n_model=n_model)
                  if fit_phase_shift else None)
        return cls(phase=phase, flux=np.asarray(flux, dtype=float), err=err, err2=err ** 2,
                   shift_search=search, fixed_shift=float(fixed_shift) % 1.0,
                   is_binned=is_binned, phase_width=phase_width)

    @property
    def fit_phase_shift(self) -> bool:
        return self.shift_search is not None


def model_curve(theta, spec: ParamSpec, model: DirectLightCurveModel
                ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Physical model (including the additive f_scatter floor) on the kernel's
    native phase grid, as ``(phase, flux)``.

    Returns None when the sample is unphysical or the simulation fails. Every
    consumer -- likelihood, per-sample chi2, BIC, best-fit overlay -- goes
    through here, so none can silently omit a term.
    """
    d1, d2, r, R, i0 = spec.geometry(theta)
    if not np.all(np.isfinite([d1, d2, r, R, i0])):
        return None
    curve = model.curve(d1, d2, r, R, i0, wind_params=spec.wind_params(theta, R),
                        f_opacity=spec.f_opacity(theta))
    if curve is None or not np.all(np.isfinite(curve[1])):
        return None
    return curve[0], np.asarray(curve[1], dtype=float) + spec.f_scatter(theta)


def aligned_model_flux(theta, spec: ParamSpec, model: DirectLightCurveModel,
                       data: FitData) -> Tuple[Optional[np.ndarray], float]:
    """Model at the observed phases after the per-sample phase-shift search.

    Returns ``(model_at_obs_phases, best_shift)``; the model is None when it
    could not be evaluated. The kernel's native curve is interpolated once,
    directly onto the (shifted) observed phases.
    """
    curve = model_curve(theta, spec, model)
    if curve is None:
        return None, 0.0
    phase_ext, flux_ext = periodic_model(*curve)
    if data.shift_search is None:
        return (eval_periodic(phase_ext, flux_ext, data.phase, shift=data.fixed_shift),
                float(data.fixed_shift))
    # The shift is profiled on the likelihood being sampled: with the jitter
    # likelihood its variance term enters the search objective too.
    jitter = np.exp(theta[spec.index('log_f')]) if spec.likelihood == 'jitter' else None
    model_at_obs, shift, _ = best_phase_shift(phase_ext, flux_ext, data.flux, data.err2,
                                              data.shift_search, jitter_frac=jitter)
    return model_at_obs, shift


# =============================================================================
# Prior, likelihood, posterior
# =============================================================================

def log_prior(theta, priors: Dict[str, Dict[str, float]], spec: ParamSpec) -> float:
    """Box + Gaussian prior per active dimension, plus physical constraints.

    Priors are stated directly on the sampled parameters of every mode (a and
    q under --reparam, the masses under the Kepler modes), so no
    change-of-variables Jacobian is applied. Constraints are applied to
    *resolved* values so they hold under freezing and the Kepler mappings:
    r < R always, Rb >= R for smooth_pl.
    """
    for name, value in zip(spec.active_names, theta):
        prior = priors.get(name)
        if prior is not None and not (prior['min'] < value < prior['max']):
            return -np.inf

    _, _, r_value, R_value, _ = spec.geometry(theta)
    if not (np.isfinite(r_value) and np.isfinite(R_value)) or r_value >= R_value:
        return -np.inf

    log_p = 0.0
    for i, name in enumerate(spec.active_names):
        prior = priors.get(name)
        if prior is not None:
            log_p += -0.5 * ((theta[i] - prior['mean']) / prior['std']) ** 2

    if spec.wind_model == 'smooth_pl':
        rb_val = spec.value(theta, 'Rb')
        if rb_val is not None and rb_val < R_value:
            return -np.inf
    return log_p


def log_likelihood(theta, spec: ParamSpec, model: DirectLightCurveModel,
                   data: FitData) -> float:
    """Gaussian log-likelihood; with ``spec.likelihood == 'jitter'`` the
    variance per point is sigma_obs^2 + (f * model)^2 with f = exp(log_f)."""
    model_flux, _ = aligned_model_flux(theta, spec, model, data)
    if model_flux is None:
        return -np.inf
    resid2 = (data.flux - model_flux) ** 2
    if spec.likelihood == 'jitter':
        sigma2 = data.err2 + (np.exp(theta[spec.index('log_f')]) * model_flux) ** 2
        return -0.5 * np.sum(resid2 / sigma2 + np.log(sigma2))
    return -0.5 * np.sum(resid2 / data.err2)


def log_probability(theta, spec: ParamSpec, priors, model, data: FitData) -> float:
    """Log posterior = log prior + log likelihood."""
    lp = log_prior(theta, priors, spec)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, spec, model, data)


def chi2_terms(theta, spec: ParamSpec, model, data: FitData) -> Dict[str, float]:
    """Classical and effective-variance chi2 of one sample at the observed phases.

    The classical chi2 uses the measurement errors and is comparable across
    likelihood choices; chi2_eff uses the jitter variance actually optimised.
    """
    model_flux, shift = aligned_model_flux(theta, spec, model, data)
    if model_flux is None:
        return {'chi2': np.nan, 'chi2_eff': np.nan, 'shift': 0.0, 'model': None}
    chi2 = float(np.sum((data.flux - model_flux) ** 2 / data.err2))
    chi2_eff = np.nan
    if spec.likelihood == 'jitter' and spec.index('log_f') is not None:
        f = np.exp(theta[spec.index('log_f')])
        # Positivity guard only: an absolute eps floor would swamp a flux
        # variance of order 1e-25.
        sigma2 = np.maximum(data.err2 + (f * model_flux) ** 2, np.finfo(float).tiny)
        chi2_eff = float(np.sum((data.flux - model_flux) ** 2 / sigma2))
    return {'chi2': chi2, 'chi2_eff': chi2_eff, 'shift': float(shift), 'model': model_flux}


def degrees_of_freedom(spec: ParamSpec, n_obs: int, fit_phase_shift: bool = False) -> int:
    """Observations minus fitted model parameters.

    Counts the sampled physical parameters and the profiled phase shift (one
    parameter, the convention ``fit_simulation`` uses too); the jitter term
    log_f describes the errors rather than the model and is not counted.
    """
    n_phys = spec.n_dim - (1 if 'log_f' in spec.active_names else 0)
    return int(n_obs - n_phys - (1 if fit_phase_shift else 0))


# =============================================================================
# Sampling
# =============================================================================

# Fit context of a pool worker, installed once by _init_worker. emcee and zeus
# otherwise pickle every argument of the log-probability function into each
# task, i.e. the whole FitData (hundreds of KB for unbinned data) per sample.
_WORKER: Dict[str, object] = {}


def _init_worker(max_numba_threads: int, spec, priors, model, data) -> None:
    """Pool initializer: cap Numba threads per worker and store the fit context.

    A Pool initializer must never raise: multiprocessing would respawn the
    worker forever while the parent blocks in ``map``. Workers ignore SIGINT so
    that a Ctrl-C reaches only the parent, which terminates the pool.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import numba
    try:
        numba.set_num_threads(max(1, min(int(max_numba_threads), int(numba.config.NUMBA_NUM_THREADS))))
    except ValueError as e:
        warnings.warn(f"numba.set_num_threads failed in a worker: {e}")
    _WORKER.update(spec=spec, priors=priors, model=model, data=data)


def _log_probability_worker(theta) -> float:
    """log_probability with the context installed by _init_worker."""
    return log_probability(theta, _WORKER['spec'], _WORKER['priors'],
                           _WORKER['model'], _WORKER['data'])


def initial_positions(spec: ParamSpec, priors, n_walkers: int) -> np.ndarray:
    """Walkers in a small ball around the prior means, clipped inside the box.

    The clip inset is relative to each prior's span: an absolute epsilon once
    exceeded f_scatter's entire range (~1e-13) and collapsed the column, which
    aborts emcee on the condition number.
    """
    n_dim = spec.n_dim
    pos = np.empty((n_walkers, n_dim))
    for i, name in enumerate(spec.active_names):
        prior = priors[name]
        pos[:, i] = prior['mean'] + 0.1 * prior['std'] * np.random.randn(n_walkers)
        lo_b, hi_b = float(prior['min']), float(prior['max'])
        span = hi_b - lo_b
        pad = 1e-9 * span if np.isfinite(span) and span > 0 else 0.0
        lo = lo_b + 0.01 * abs(lo_b) + pad
        hi = hi_b - 0.01 * abs(hi_b) - pad
        if not lo < hi:
            lo, hi = lo_b, hi_b
        pos[:, i] = np.clip(pos[:, i], lo, hi)

    idx_r, idx_R = spec.index('r'), spec.index('R')
    if idx_r is not None and idx_R is not None:
        bad = pos[:, idx_r] >= pos[:, idx_R]
        pos[bad, idx_r] = 0.1 * pos[bad, idx_R]

    # Ensemble samplers need linearly independent walkers: re-spread any
    # dimension that collapsed to a constant.
    for i, name in enumerate(spec.active_names):
        if np.ptp(pos[:, i]) > 0:
            continue
        lo_b, hi_b = float(priors[name]['min']), float(priors[name]['max'])
        span = hi_b - lo_b
        if np.isfinite(span) and span > 0:
            warnings.warn(
                f"Walker initialization for '{name}' collapsed to a single value; "
                f"re-spreading uniformly inside its prior box. Check the prior scale.")
            pos[:, i] = np.random.uniform(lo_b + 0.05 * span, hi_b - 0.05 * span, n_walkers)
    return pos


def run_mcmc(
    spec: ParamSpec,
    priors: Dict[str, Dict[str, float]],
    model: DirectLightCurveModel,
    data: FitData,
    n_walkers: int = 32,
    n_steps: int = 5000,
    n_burn: int = 1000,
    sampler_type: str = 'emcee',
    progress: bool = True,
    n_threads: int = 1,
    numba_threads_per_worker: Optional[int] = None,
) -> Tuple[object, np.ndarray]:
    """Run the ensemble sampler; returns ``(sampler, flat_samples_after_burn)``."""
    pos = initial_positions(spec, priors, n_walkers)

    # Evaluate the initial ensemble up front: emcee accepts walkers at -inf and
    # would sample a model that can never be evaluated to completion, with
    # acceptance 0 and a "posterior" equal to the initial ball.
    lp0 = np.array([log_probability(p, spec, priors, model, data) for p in pos])
    if not np.any(np.isfinite(lp0)):
        raise RuntimeError(
            "Every initial walker has log-probability -inf: the model cannot be evaluated "
            "near the prior means (check the flux table, the wind model and the priors).")
    # Walkers that start outside the constraints (r < R, Rb >= R, the boxes) are
    # redrawn: emcee would only reject moves from them, zeus refuses to start.
    for _ in range(20):
        bad = ~np.isfinite(lp0)
        if not bad.any():
            break
        fresh = initial_positions(spec, priors, n_walkers)
        pos[bad] = fresh[bad]
        lp0[bad] = [log_probability(p, spec, priors, model, data) for p in pos[bad]]
    n_bad = int(np.sum(~np.isfinite(lp0)))
    if n_bad:
        msg = (f"{n_bad} of {n_walkers} initial walkers still have log-probability -inf after "
               f"redrawing; the prior means or a --freeze value sit at a constraint boundary.")
        if sampler_type == 'zeus':
            raise RuntimeError(msg + " zeus cannot start from -inf walkers.")
        warnings.warn(msg)

    print(f"\nStarting MCMC ({sampler_type}) with {n_walkers} walkers, {n_steps} steps"
          f"{f', {n_threads} threads' if n_threads > 1 else ' (serial)'}")
    print(f"Parameterization: {spec.mode}  | Likelihood: {LIKELIHOOD_TYPES[spec.likelihood]}")
    print(f"Wind model: {WIND_MODELS[spec.wind_model]} ({spec.wind_model})"
          f"  | fit_wind_shape={spec.fit_wind_shape}")
    if data.shift_search is not None:
        s = data.shift_search
        print(f"Per-sample phase-shift search: enabled (coarse grid {s.shift_grid.size}, "
              f"{s.n_levels} x {s.n_fine}-point dense passes, resolution {s.resolution:.2e})")
    else:
        print(f"Per-sample phase-shift search: disabled (shift held at {data.fixed_shift:.5f})")
    print(f"Active params ({spec.n_dim}): {spec.active_names}")
    if spec.frozen:
        print(f"Frozen: {spec.frozen}")

    pool = None
    log_prob_fn, args = log_probability, (spec, priors, model, data)
    if n_threads > 1:
        import numba
        cpus = int(cpu_count() or 1)
        ntb = (max(1, cpus // int(n_threads)) if numba_threads_per_worker is None
               else max(1, int(numba_threads_per_worker)))
        # numba derives its own limit from the process affinity (a pinned
        # cluster job may see fewer cores than cpu_count reports).
        ntb = max(1, min(ntb, int(numba.config.NUMBA_NUM_THREADS)))
        print(f"[info] Pooled MCMC: {n_threads} worker processes, "
              f"numba.set_num_threads({ntb}) per worker (logical CPUs ~ {cpus}).")
        pool = mp.get_context("spawn").Pool(
            processes=n_threads, initializer=_init_worker,
            initargs=(ntb, spec, priors, model, data))
        # Only theta crosses the pipe: the fit context lives in the workers.
        log_prob_fn, args = _log_probability_worker, ()

    start = time.time()
    try:
        if sampler_type == 'zeus':
            if not HAS_ZEUS:
                raise ImportError("zeus not installed. Install with: pip install zeus-mcmc")
            sampler = zeus_sampler.EnsembleSampler(
                n_walkers, spec.n_dim, log_prob_fn, args=args, pool=pool)
            sampler.run_mcmc(pos, n_steps, progress=progress)
        else:
            sampler = emcee.EnsembleSampler(
                n_walkers, spec.n_dim, log_prob_fn, args=args, pool=pool)
            # emcee seeds its private RandomState from OS entropy; take the
            # global state instead so --seed (np.random.seed in main) covers
            # the moves as well as the initial ball. zeus uses np.random itself.
            sampler.random_state = np.random.get_state()
            state = emcee.State(pos, log_prob=lp0)   # already evaluated above
            if progress and HAS_TQDM:
                for _ in tqdm(sampler.sample(state, iterations=n_steps),
                              total=n_steps, desc="MCMC Sampling"):
                    pass
            else:
                sampler.run_mcmc(state, n_steps, progress=progress)
    except BaseException:
        if pool is not None:
            # An interrupted worker leaves its task in flight; close()/join()
            # would then wait forever (Ctrl-C used to hang the run).
            pool.terminate()
            pool.join()
        raise
    if pool is not None:
        pool.close()
        pool.join()

    elapsed = time.time() - start
    # Flatten in (step, walker) order for both samplers: zeus's own flat=True
    # is walker-major, which would put a fresh run's samples CSV in a different
    # order than the chain NPZ a --replot reshapes.
    chain_post = sampler.get_chain(discard=n_burn)
    samples = chain_post.reshape(-1, chain_post.shape[2])
    print(f"\nMCMC completed in {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")
    print(f"Time per step: {elapsed / n_steps * 1000:.1f} ms")
    print(f"Final chain shape: {samples.shape}")
    return sampler, samples


# =============================================================================
# Statistics and diagnostics
# =============================================================================

def _marginal(values: np.ndarray) -> Dict[str, float]:
    p16, p50, p84 = np.percentile(values, [16, 50, 84])
    return {'median': float(p50), 'lower': float(p50 - p16), 'upper': float(p84 - p50),
            'mean': float(np.mean(values)), 'std': float(np.std(values))}


def compute_statistics(samples: np.ndarray, spec: ParamSpec,
                       log_prob: Optional[np.ndarray] = None) -> Dict:
    """Marginal summaries for the active and derived parameters, plus the MAP.

    Medians of nonlinear combinations are not combinations of medians, so the
    ``median`` rows will not satisfy d1 + d2 = a exactly; the MAP (highest
    log-probability sample) does, and is the point estimate used for overlays.
    """
    stats: Dict = {}
    for i, name in enumerate(spec.active_names):
        stats[name] = _marginal(samples[:, i])
    try:
        derived = spec.derived(samples)
    except KeyError as e:
        warnings.warn(f"Derived quantities unavailable: {e}")
        derived = {}
    for name, values in derived.items():
        stats[name] = {**_marginal(values), 'derived': True}

    if log_prob is not None and len(log_prob) == len(samples):
        finite = np.isfinite(log_prob)
        if finite.any():
            map_idx = int(np.argmax(np.where(finite, log_prob, -np.inf)))
            row = samples[map_idx]
            for i, name in enumerate(spec.active_names):
                stats[name]['map'] = float(row[i])
            if derived:
                for name, values in spec.derived(row[None, :]).items():
                    stats[name]['map'] = float(values[0])
            stats['_map_meta'] = {'index': map_idx, 'log_prob': float(log_prob[map_idx])}
    return stats


def point_estimate_theta(stats: Dict, spec: ParamSpec) -> Tuple[np.ndarray, str]:
    """Point estimate in active order: the MAP when available, else medians."""
    key = 'map' if all('map' in stats[n] for n in spec.active_names if n in stats) else 'median'
    theta = np.array([float(stats[n][key]) for n in spec.active_names], dtype=float)
    return theta, key


def print_results(stats: Dict, spec: ParamSpec, band: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"MCMC Results for {band.upper()} band - {WIND_MODELS[spec.wind_model]}")
    print('=' * 60)
    print(f"{'Parameter':<15} {'Median':<12} {'-1 sigma':<12} {'+1 sigma':<12}")
    print('-' * 60)
    for name in spec.active_names:
        s = stats[name]
        print(f"{name:<15} {fmt_val(s['median'], 12)} {fmt_val(s['lower'], 12)} "
              f"{fmt_val(s['upper'], 12)}")
    derived = [n for n in spec.derived_names if n in stats]
    if derived:
        print('-' * 60)
        print("Derived quantities:")
        for name in derived:
            s = stats[name]
            print(f"{name:<15} {fmt_val(s['median'], 12)} {fmt_val(s['lower'], 12)} "
                  f"{fmt_val(s['upper'], 12)}")
    print('=' * 60)


def print_diagnostics(sampler, sampler_type: str, names: List[str], n_burn: int = 0) -> Dict[str, object]:
    """Acceptance fraction, autocorrelation times and a convergence verdict.

    Autocorrelation times and the effective-sample count are computed on the
    post-burn-in chain, the part that is reported and saved.
    """
    diag: Dict[str, object] = {}
    print("\n" + "=" * 60)
    print(f"MCMC Diagnostics  ({sampler_type})")
    print("=" * 60)
    if sampler_type == 'emcee' and hasattr(sampler, 'acceptance_fraction'):
        acc = float(np.mean(sampler.acceptance_fraction))
        diag['acceptance_fraction_mean'] = acc
        verdict = ("WARNING: low acceptance - consider adjusting priors" if acc < 0.2 else
                   "WARNING: high acceptance - chain may not be mixing well" if acc > 0.5 else
                   "OK: acceptance in the optimal range (0.2-0.5)")
        print(f"Mean acceptance fraction: {acc:.3f}\n  {verdict}")
    try:
        chain = sampler.get_chain(discard=n_burn)
        if sampler_type == 'emcee' and hasattr(sampler, 'get_autocorr_time'):
            tau = sampler.get_autocorr_time(discard=n_burn, quiet=True)
        else:
            tau = np.array([emcee.autocorr.integrated_time(chain[:, :, i].mean(axis=1), quiet=True)[0]
                            for i in range(chain.shape[2])])
        print("\nAutocorrelation times:")
        diag['autocorr_time'] = {}
        for i, name in enumerate(names):
            if i < len(tau):
                print(f"  {name}: {tau[i]:.1f} steps")
                diag['autocorr_time'][name] = float(tau[i])
        n_steps, n_walkers = chain.shape[0], chain.shape[1]
        n_indep = n_steps / np.max(tau)
        diag['effective_independent_samples'] = int(n_indep * n_walkers)
        print(f"\nEffective independent samples (post burn-in, {n_steps} steps): ~{int(n_indep * n_walkers)}")
        diag['converged'] = bool(n_steps >= 50 * np.max(tau))
        print("  OK: chain appears well-converged" if diag['converged']
              else "  WARNING: chain may not be converged (fewer than 50 tau post burn-in). Run longer.")
    except Exception:
        diag.update({'autocorr_time': None, 'effective_independent_samples': None, 'converged': None})
        print("\nAutocorrelation time: could not compute (chain too short)")
    print("=" * 60)
    return diag


def compute_chi2_for_samples(model, spec: ParamSpec, samples: np.ndarray, data: FitData,
                             output_path: str, n_samples: Optional[int] = None,
                             verbose: bool = True, log_prob: Optional[np.ndarray] = None,
                             priors: Optional[Dict[str, Dict[str, float]]] = None) -> None:
    """Per-sample chi2 (classical, and effective-variance for jitter runs) to CSV.

    With the ``chi2`` likelihood the classical chi2 is already in the chain:
    ``log_prob = log_prior - chi2 / 2`` exactly, so given *log_prob* and
    *priors* every sample is tabulated without a model call. The jitter
    likelihood needs the model for ``chi2_eff``; there *n_samples* defaults to
    ``CHI2_TABLE_DEFAULT_SAMPLES`` random samples.
    """
    n_total = len(samples)
    from_chain = (spec.likelihood == 'chi2' and log_prob is not None
                  and priors is not None and len(log_prob) == n_total)
    if n_samples is None and not from_chain:
        n_samples = CHI2_TABLE_DEFAULT_SAMPLES
    if n_samples is None or n_samples >= n_total:
        indices = np.arange(n_total)
    else:
        indices = np.sort(np.random.choice(n_total, size=n_samples, replace=False))
    if verbose:
        print(f"Computing chi-square for {len(indices)} samples"
              f"{' from the stored log-probabilities' if from_chain else ''}...")
    dof = degrees_of_freedom(spec, len(data.flux), data.fit_phase_shift)
    use_eff = spec.likelihood == 'jitter'
    iterator = (tqdm(indices, desc="Computing chi2")
                if (HAS_TQDM and verbose and not from_chain) else indices)

    rows = []
    for idx in iterator:
        theta = samples[idx]
        d1, d2, r, R, i0 = spec.geometry(theta)
        if from_chain:
            chi2 = -2.0 * (float(log_prob[idx]) - log_prior(theta, priors, spec))
            chi2_eff = np.nan
        else:
            try:
                terms = chi2_terms(theta, spec, model, data)
                chi2, chi2_eff = terms['chi2'], terms['chi2_eff']
            except Exception:
                chi2, chi2_eff = np.nan, np.nan
        row = [idx, d1, d2, r, R, i0, chi2, chi2 / dof if dof > 0 else np.nan]
        if use_eff:
            row += [chi2_eff, chi2_eff / dof if dof > 0 else np.nan]
        rows.append(row)

    columns = ['sample_idx', 'd1', 'd2', 'r', 'R', 'i0', 'chi2', 'reduced_chi2']
    if use_eff:
        columns += ['chi2_eff', 'reduced_chi2_eff']
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False, compression='gzip' if output_path.endswith('.gz') else None)
    valid = df['reduced_chi2'].dropna()
    if verbose and len(valid):
        print(f"Reduced chi-square over samples: median {np.median(valid):.3f}, "
              f"min {valid.min():.3f}, max {valid.max():.3f}, std {np.std(valid):.3f}")
    if verbose:
        print(f"Chi-square data saved to: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")


def compute_bic_metrics(stats: Dict, spec: ParamSpec, model, data: FitData,
                        samples: Optional[np.ndarray] = None, log_prob: Optional[np.ndarray] = None,
                        priors: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, object]:
    """BIC = k ln n - 2 ln L_hat.

    L_hat is the maximum likelihood over the chain: ``log_prob - log_prior``
    is exactly the log-likelihood of every sample, so the best one is free
    (*samples*, *log_prob* and the *priors* the chain was sampled under). The
    MAP sample maximises the posterior instead and can sit 5 log-units lower
    in likelihood. Without the chain (or with priors that differ from the
    sampled ones) the likelihood at the point estimate of *stats* is used."""
    theta_hat, key = point_estimate_theta(stats, spec)
    source = "map_log_prob" if key == 'map' else "median_fallback"
    logL_hat = log_likelihood(theta_hat, spec, model, data)
    if samples is not None and log_prob is not None and priors is not None and len(samples):
        log_prior_vals = np.fromiter((log_prior(theta, priors, spec) for theta in samples),
                                     dtype=float, count=len(samples))
        logL = np.asarray(log_prob, dtype=float) - log_prior_vals
        logL = np.where(np.isfinite(logL), logL, -np.inf)
        i_best = int(np.argmax(logL))
        if np.isfinite(logL[i_best]) and logL[i_best] > logL_hat:
            logL_hat, source = float(logL[i_best]), "max_likelihood_sample"
    n = int(len(data.flux))
    if n <= 0 or not np.isfinite(logL_hat):
        return {}
    # Fitted parameters: the sampled dimensions plus the per-sample profiled
    # phase shift, the same count degrees_of_freedom uses.
    k = spec.n_dim + (1 if data.fit_phase_shift else 0)
    return {"bic": float(k * np.log(n) - 2.0 * logL_hat), "logL_hat": float(logL_hat),
            "k_params": float(k), "n_obs": float(n), "theta_source": source}


def _build_inference_data(posterior_dict):
    """ArviZ inference object across ArviZ versions (<= 0.17 and 1.0+)."""
    for call in (lambda: az.from_dict({"posterior": posterior_dict}),
                 lambda: az.from_dict(posterior=posterior_dict)):
        try:
            return call()
        except (TypeError, AttributeError):
            continue
    raise RuntimeError(
        f"Could not build inference data with arviz {getattr(az, '__version__', '?')}. "
        "Try: pip install --upgrade arviz")


def run_arviz_diagnostics(chain: np.ndarray, spec: ParamSpec, output_dir: str,
                          suffix: str) -> None:
    """Print and save the ArviZ summary (r_hat, ess, mcse, HDI) for a chain."""
    if not HAS_ARVIZ:
        print("arviz not installed. Install with: pip install arviz")
        return
    posterior = {name: chain[:, :, i].T for i, name in enumerate(spec.active_names)}
    summary = az.summary(_build_inference_data(posterior))
    print("\n--- ArviZ Summary ---")
    print(summary)
    path = os.path.join(output_dir, f"{suffix}_arviz_summary.csv")
    summary.to_csv(path)
    print(f"ArviZ summary saved to: {path}")


# =============================================================================
# Best-fit outputs
# =============================================================================

def plot_best_fit(model, spec: ParamSpec, data: FitData, stats: Dict, band: str,
                  output_path: str, smooth: Optional[pd.DataFrame] = None,
                  smooth_sigma: float = 0.01) -> float:
    """Observed data with the point-estimate model overlay and residual panel.

    Returns the reduced chi2 of the drawn model at the observed phases. Also
    writes the drawn curve as ``<output>_model.txt`` so it is usable outside
    the figure.
    """
    theta, key = point_estimate_theta(stats, spec)
    curve = model_curve(theta, spec, model)
    if curve is None:
        model_phases = np.linspace(0.0, 1.0, 360, endpoint=False)
        dense = np.full_like(model_phases, np.nan)
    else:
        model_phases, dense = curve
    terms = chi2_terms(theta, spec, model, data)
    obs_model = terms['model']
    if obs_model is None:
        obs_model = np.full_like(data.phase, np.nan)
    shift = terms['shift']
    dof = degrees_of_freedom(spec, len(data.flux), data.fit_phase_shift)
    red_chi2 = terms['chi2'] / dof if dof > 0 else np.nan
    red_chi2_eff = terms['chi2_eff'] / dof if (dof > 0 and np.isfinite(terms['chi2_eff'])) else np.nan
    f_best = (float(np.exp(theta[spec.index('log_f')]))
              if spec.index('log_f') is not None else None)
    f_scatter_best = spec.f_scatter(theta)

    print(f"Best-fit overlay from the {'MAP' if key == 'map' else 'median'} point estimate: "
          f"chi2/dof = {red_chi2:.6g} (dof = {dof})")
    print(f"  phase_shift = {shift:.5f}{'' if data.fit_phase_shift else ' (held fixed)'}")
    if f_best is not None:
        print(f"  f = {f_best:.4f} (from log_f)"
              + (f", chi2_eff/dof = {red_chi2_eff:.6g}" if np.isfinite(red_chi2_eff) else ""))
    if spec.fit_scatter or 'f_scatter' in spec.frozen:
        print(f"  f_scatter = {f_scatter_best:.6g}")

    overlay_phase = np.mod(model_phases + shift, 1.0)
    plot_lightcurve_fit(
        data.phase, data.flux, data.err,
        model_phase=overlay_phase, model_flux=dense, obs_model=obs_model,
        obs_phase_width=data.phase_width, band=band.upper(), red_chi2=red_chi2,
        output_path=output_path, is_binned=data.is_binned,
        obs_label='Observed (phase-binned)' if data.is_binned else 'Observed (raw 100s)',
        model_label=f'Best-fit model ({WIND_MODELS[spec.wind_model]})',
        smooth_phase=None if smooth is None else smooth["phase"].to_numpy(dtype=float),
        smooth_flux=None if smooth is None else smooth["flux_smooth"].to_numpy(dtype=float),
        smooth_flux_err=None if smooth is None else smooth["flux_smooth_err"].to_numpy(dtype=float),
        smooth_sigma=smooth_sigma, verbose=False,
    )
    print(f"Best-fit plot saved to: {output_path}")

    model_txt = model_dump_path(output_path)
    try:
        _write_bestfit_model_txt(
            model_txt, spec=spec, data=data, theta=theta, point_key=key,
            overlay_phase=overlay_phase, model_flux=dense, obs_model=obs_model,
            band=band, red_chi2=red_chi2, dof=dof, shift=shift, f_best=f_best,
            red_chi2_eff=red_chi2_eff)
        print(f"Best-fit model light curve saved to: {model_txt}")
    except Exception as e:
        warnings.warn(f"Could not write best-fit model light curve: {e}")
    return red_chi2


def _write_bestfit_model_txt(path: str, *, spec: ParamSpec, data: FitData,
                             theta: np.ndarray, point_key: str, overlay_phase, model_flux,
                             obs_model, band: str, red_chi2: float, dof: int, shift: float,
                             f_best: Optional[float], red_chi2_eff: float) -> None:
    """Best-fit curve as text: a reproducible header, then the two blocks of
    ``utils.write_model_blocks`` (dense curve; observed bins with residuals)."""
    d1, d2, r_val, R_val, i0_val = spec.geometry(theta)
    wind_params = spec.wind_params(theta, R_val)
    f_opacity = spec.f_opacity(theta)
    derived = {k: float(v[0]) for k, v in spec.derived(theta[None, :]).items()}

    with open(path, 'w') as f:
        f.write(f"# Best-fit model light curve -- {band.upper()} band, {WIND_MODELS[spec.wind_model]}\n")
        f.write(f"# point_estimate: {'MAP' if point_key == 'map' else 'median'}\n")
        f.write(f"# parameterization: {spec.mode}\n")
        f.write(f"# chi2/dof: {red_chi2:.6g}  (dof = {dof})\n")
        if f_best is not None:
            f.write(f"# jitter f: {f_best:.6g}"
                    + (f"   chi2_eff/dof: {red_chi2_eff:.6g}" if np.isfinite(red_chi2_eff) else "") + "\n")
        f.write(f"# phase_shift applied to model: {shift:.6f}\n")
        f.write("#\n# Sampled parameters at this point estimate:\n")
        for name, val in zip(spec.active_names, theta):
            f.write(f"#   {name} = {float(val):.8g}\n")
        if spec.frozen:
            f.write("# Frozen parameters:\n")
            for name, val in sorted(spec.frozen.items()):
                f.write(f"#   {name} = {float(val):.8g}\n")
        f.write("# Geometry:\n")
        geom = {'d1': d1, 'd2': d2, 'a': d1 + d2, 'q': d1 / (d1 + d2) if (d1 + d2) else np.nan,
                'r': r_val, 'R': R_val, 'i0_deg': i0_val}
        for name, val in geom.items():
            f.write(f"#   {name} = {float(val):.8g}\n")
        for name in ('M_X', 'M_RH', 'M_tot'):
            if name in derived:
                f.write(f"#   {name} = {derived[name]:.8g}\n")
            elif name in spec.active_names or name in spec.frozen:
                f.write(f"#   {name} = {spec.value(theta, name):.8g}\n")
        if wind_params:
            f.write("# Wind shape parameters:\n")
            for name, val in sorted(wind_params.items()):
                f.write(f"#   {name} = {float(val):.8g}\n")
        if f_opacity is not None:
            f.write(f"#   f_opacity = {float(f_opacity):.8g}\n")
        f.write(f"#   f_scatter = {spec.f_scatter(theta):.8g}\n")
        order = np.argsort(np.asarray(overlay_phase, dtype=float))
        write_model_blocks(f, np.asarray(overlay_phase, dtype=float)[order],
                           np.asarray(model_flux, dtype=float)[order],
                           data.phase, data.flux, data.err, obs_model)


def plot_geometry_diagnostics(spec: ParamSpec, stats: Dict, samples: np.ndarray,
                              model: DirectLightCurveModel, band: str, output_dir: str,
                              suffix: str, n_profile_draws: int = 300,
                              verbose: bool = True) -> Optional[Dict[str, str]]:
    """Projected orbit, geometry-vs-phase, and wind profile at the point estimate.

    One simulate_lightcurve call with the *same* normalization the likelihood
    used provides every geometry column; the wind-profile band propagates up to
    ``n_profile_draws`` posterior samples through ``evaluate_g_profile``.
    """
    try:
        theta, key = point_estimate_theta(stats, spec)
    except KeyError as e:
        warnings.warn(f"Skipping geometry plots: {e}")
        return None
    d1, d2, r, R, i0 = spec.geometry(theta)
    wind_params = spec.wind_params(theta, R) or {
        **default_wind_params(spec.wind_model, R)}
    f_scatter = spec.f_scatter(theta)
    if verbose:
        print(f"\nGeometry diagnostics at the {key} point estimate:")
        print(f"  d1={d1:.4f}  d2={d2:.4f}  a={d1 + d2:.4f}  r={r:.6g}  R={R:.4f}  i0={i0:.4f} deg")
        print("  wind_params: " + ", ".join(f"{k}={v:.4g}" for k, v in sorted(wind_params.items())))
        if f_scatter:
            print(f"  f_scatter:   {f_scatter:.6g} (additive floor)")
    try:
        sim_df = simulate_lightcurve(**model.sim_kwargs(
            d1, d2, r, R, i0, wind_params=wind_params, f_opacity=spec.f_opacity(theta),
            scattered_flux=f_scatter), verbose=False)
    except Exception as e:
        warnings.warn(f"Skipping geometry plots: simulate_lightcurve failed: {e}")
        return None

    written: Dict[str, str] = {}
    band_label = band.upper()
    for key_name, fn in (
        ('orbit', lambda p: plot_orbit_geometry(sim_df, R=R, r=r, d1=d1, d2=d2, i0=i0,
                                                output_path=p, band=band_label, verbose=verbose)),
        ('phase', lambda p: plot_geometry_vs_phase(sim_df, R=R, r=r, band=band_label,
                                                   flux_column=f"nfl_{model.band}",
                                                   output_path=p, verbose=verbose)),
    ):
        path = os.path.join(output_dir, f"{suffix}_geometry_{key_name}.png")
        try:
            fn(path)
            written[key_name] = path
        except Exception as e:
            warnings.warn(f"Geometry plot '{key_name}' failed: {e}")

    # Radii probed by the line of sight: the impact parameter relative to the
    # companion centre equals the projected separation l3.
    l3 = sim_df['l3'].to_numpy(dtype=float)
    keep = ~sim_df['is_eclipsed'].to_numpy(dtype=bool)
    l3 = l3[keep] if keep.any() else l3
    probed = (float(l3.min()), float(l3.max())) if l3.size else None
    r_lo = max(1e-3, 0.5 * min(R, probed[0] if probed else R))
    r_hi = max(4.0 * R, probed[1] * 3.0 if probed else 10.0 * R, 2.0 * wind_params.get('Rb', 0.0))
    r_grid = np.logspace(np.log10(r_lo), np.log10(r_hi), 240)

    g_rows: List[np.ndarray] = []
    draws = np.atleast_2d(np.asarray(samples, dtype=float)) if samples is not None else None
    if draws is not None and draws.size and draws.shape[1] == spec.n_dim:
        n = min(int(n_profile_draws), draws.shape[0])
        idx = (np.random.choice(draws.shape[0], size=n, replace=False)
               if n < draws.shape[0] else np.arange(draws.shape[0]))
        for k in idx:
            try:
                _, _, _, R_k, _ = spec.geometry(draws[k])
                wp_k = spec.wind_params(draws[k], R_k) or default_wind_params(spec.wind_model, R_k)
                g_rows.append(np.asarray(evaluate_g_profile(r_grid, spec.wind_model, wp_k), dtype=float))
            except Exception:
                continue
    if not g_rows:
        g_rows.append(np.asarray(evaluate_g_profile(r_grid, spec.wind_model, wind_params), dtype=float))

    shape_keys = WIND_SHAPE_FIT[spec.wind_model]
    summary = "\n".join(
        f"{k:>7s} = {wind_params[k]:.4g}" + ("" if (k in shape_keys and spec.fit_wind_shape) else "  (fixed)")
        for k in WIND_MODEL_PARAM_KEYS[spec.wind_model] if k in wind_params)
    # Characteristic radii: the break radius (smooth_pl), the compression scale
    # (confinement) and the effective break R + 3H (beta_law).
    mark_radii = {k: wind_params[k] for k in ('Rb', 'ell') if k in wind_params}
    if spec.wind_model == 'beta_law' and 'H' in wind_params:
        mark_radii['R_*+3H'] = float(R) + 3.0 * float(wind_params['H'])
    path = os.path.join(output_dir, f"{suffix}_wind_profile.png")
    try:
        plot_wind_profile(r_grid, np.vstack(g_rows), R=R, probed_range=probed,
                          mark_radii=mark_radii, wind_model=WIND_MODELS[spec.wind_model],
                          band=band_label, shape_summary=summary or None,
                          output_path=path, verbose=verbose)
        written['wind_profile'] = path
    except Exception as e:
        warnings.warn(f"Wind-profile plot failed: {e}")
    return written or None


# =============================================================================
# Run / replot orchestration
# =============================================================================

def postprocess_fit(args, spec: ParamSpec, priors: Dict, model, data: FitData,
                    samples: np.ndarray, stats: Dict, band: str, chain: Optional[np.ndarray],
                    log_prob_flat: Optional[np.ndarray], smoothed: Optional[pd.DataFrame],
                    sampler=None) -> Dict:
    """Everything after sampling that a fresh fit and a --replot share:
    ArviZ/BIC, the figures, the model-curve dump and the chi2 table."""
    suffix = f"{band}_{spec.wind_model}"
    if chain is not None:
        run_arviz_diagnostics(chain, spec, args.output_dir, suffix)
    if args.compute_bic:
        # log_prob - log_prior is the likelihood only under the sampled priors
        # (same guard as the chi2 table below).
        chain_ok = not (sampler is None and (getattr(args, '_prior_typed', False)
                                             or any(n not in priors for n in spec.active_names)))
        bic_info = compute_bic_metrics(stats, spec, model, data,
                                       samples=samples if chain_ok else None,
                                       log_prob=log_prob_flat if chain_ok else None,
                                       priors=priors if chain_ok else None)
        if bic_info:
            stats.update(bic_info)
            print("\nBIC: {bic:.3f}  (logL_hat={logL_hat:.3f}, k={k_params:.0f}, "
                  "n={n_obs:.0f}, source={theta_source})".format(**bic_info))
            pd.DataFrame([bic_info]).to_csv(
                os.path.join(args.output_dir, f"{suffix}_model_metrics.csv"), index=False)

    if not args.no_plots:
        plot_corner(samples, band, spec.wind_model,
                    os.path.join(args.output_dir, f"{suffix}_corner.png"),
                    param_labels=spec.active_labels)
        if sampler is not None:
            plot_trace(sampler, band, spec.wind_model,
                       os.path.join(args.output_dir, f"{suffix}_trace.png"),
                       param_labels=spec.active_labels, n_burn=args.n_burn)
        stats['reduced_chi2'] = plot_best_fit(
            model, spec, data, stats, band,
            os.path.join(args.output_dir, f"{suffix}_bestfit.png"),
            smooth=smoothed, smooth_sigma=float(args.smooth_sigma))
        if not args.no_geometry_plots:
            plot_geometry_diagnostics(spec, stats, samples, model, band, args.output_dir,
                                      suffix, verbose=not args.quiet)

    if args.save_chi2:
        # chi2 = -2 (log_prob - log_prior) only holds with the priors the chain was
        # sampled under: on a replot with a typed --prior-* (or a rebuilt spec
        # whose parameters lack priors) fall back to evaluating the model.
        chain_priors = priors
        if sampler is None and (getattr(args, '_prior_typed', False)
                                or any(n not in priors for n in spec.active_names)):
            print("Note: priors differ from the sampled ones; chi2 is evaluated from the model.")
            chain_priors = None
        compute_chi2_for_samples(
            model, spec, samples, data,
            output_path=os.path.join(args.output_dir, f"{suffix}_chi2.csv.gz"),
            n_samples=args.chi2_n_samples, verbose=True, log_prob=log_prob_flat,
            priors=chain_priors)
    return stats


def run_single_fit(band: str, args, spec: ParamSpec, priors: Dict, model,
                   data: FitData, smoothed: Optional[pd.DataFrame]) -> Dict:
    """Sample, summarize, persist and plot one (band, wind_model) fit."""
    suffix = f"{band}_{spec.wind_model}"
    print(f"\n{'#' * 60}\n# Fitting {band.upper()} band - {WIND_MODELS[spec.wind_model]}\n{'#' * 60}")

    fit_start = time.time()
    sampler, samples = run_mcmc(
        spec, priors, model, data,
        n_walkers=args.n_walkers, n_steps=args.n_steps, n_burn=args.n_burn,
        sampler_type=args.sampler, progress=not args.quiet, n_threads=args.n_threads,
        numba_threads_per_worker=args.numba_threads_per_worker)
    fit_elapsed = float(time.time() - fit_start)
    chain = sampler.get_chain(discard=args.n_burn)
    log_prob_flat = sampler.get_log_prob(discard=args.n_burn).reshape(-1)   # same order as samples

    # Persist the sampling result first: everything below (statistics, ArviZ,
    # figures, the chi2 table) can fail or be interrupted, and --replot needs
    # only these files.
    if not args.no_csv_output:
        path = os.path.join(args.output_dir, f"{suffix}_samples.csv")
        save_samples_csv_chunked(samples=samples, param_names=spec.active_names,
                                 output_path=path, log_prob=log_prob_flat,
                                 chunk_size=args.csv_chunk_size)
        print(f"Samples saved to: {path}")
    # The chain file is what --replot reads: the post-burn chain and log-prob plus
    # the metadata that rebuilds the ParamSpec.
    path = os.path.join(args.output_dir, f"{suffix}_chain.npz")
    np.savez_compressed(
        path, chain=chain, log_prob=sampler.get_log_prob(discard=args.n_burn),
        param_names=np.array(spec.active_names, dtype=str), n_burn=int(args.n_burn),
        likelihood=spec.likelihood, mode=spec.mode, wind_model=spec.wind_model,
        frozen_names=np.array(list(spec.frozen.keys()), dtype=str),
        frozen_values=np.array(list(spec.frozen.values()), dtype=float),
        orbital_period_s=float(spec.orbital_period_s),
        wind_normalization=WIND_NORMALIZATION,
        n_obs=float(len(data.flux)))   # --replot compares this with the data it loads
    print(f"Full chain saved to: {path}")

    stats = compute_statistics(samples, spec, log_prob=log_prob_flat)
    print_results(stats, spec, band)
    stats['_diagnostics'] = print_diagnostics(sampler, args.sampler, spec.active_names, n_burn=args.n_burn)
    stats['_run_meta'] = {
        'sampler': args.sampler, 'likelihood': spec.likelihood,
        'n_walkers': int(args.n_walkers), 'n_steps': int(args.n_steps),
        'n_burn': int(args.n_burn), 'fit_elapsed_s': fit_elapsed, 'seed': args.seed,
        'fit_phase_shift': data.fit_phase_shift,
        'phase_shift_fixed': None if data.fit_phase_shift else float(data.fixed_shift),
        'phase_window': [float(v) for v in args.phase_window],
        'phase_shift_grid_size': (int(data.shift_search.shift_grid.size)
                                  if data.shift_search is not None else None),
        'phase_shift_resolution': (float(data.shift_search.resolution)
                                   if data.shift_search is not None else None),
    }
    return postprocess_fit(args, spec, priors, model, data, samples, stats, band, chain,
                           log_prob_flat, smoothed, sampler=sampler)


def replot_from_existing(band: str, args, spec: ParamSpec, model, data: FitData,
                         smoothed: Optional[pd.DataFrame],
                         priors_for) -> Optional[Tuple[Dict, ParamSpec]]:
    """Regenerate every output from a saved chain without re-sampling.

    The chain file ``*_chain.npz`` (always written, right after sampling) is
    the single source: the post-burn chain and log-probabilities, the sampled
    parameter names in chain order, and the metadata that defines the
    parameterization (mode, likelihood, period, frozen values) and the wind
    normalization the posterior was sampled under. The ``*_samples.csv`` export
    is not needed. *priors_for(spec)* returns the active priors of the rebuilt
    spec.
    """
    suffix = f"{band}_{spec.wind_model}"
    print(f"\n{'#' * 60}\n# Replotting {band.upper()} band - {WIND_MODELS[spec.wind_model]}\n{'#' * 60}")

    chain_path = os.path.join(args.output_dir, f"{suffix}_chain.npz")
    if not os.path.exists(chain_path):
        print(f"Chain file not found: {chain_path} (every fit since Phase 34 writes it; older "
              f"results were sampled under a different model and must be refitted).")
        return None
    meta = np.load(chain_path, allow_pickle=False)   # numeric and string arrays only
    if str(meta.get('wind_normalization', 'missing')) != WIND_NORMALIZATION:
        print(f"Error: {chain_path} was sampled under a different wind normalization "
              f"(stamp: {meta.get('wind_normalization', 'missing')!s}, current: "
              f"{WIND_NORMALIZATION}); its posterior cannot be re-evaluated with the "
              f"current model. Refit instead.")
        return None

    chain = np.asarray(meta['chain'], dtype=float)
    log_prob_chain = np.asarray(meta['log_prob'], dtype=float)
    loaded_names = [str(n) for n in meta['param_names']]
    saved = {'mode': str(meta['mode']), 'likelihood': str(meta['likelihood']),
             'orbital_period_s': float(meta['orbital_period_s'])}
    fn, fv = list(meta['frozen_names']), list(meta['frozen_values'])
    saved['frozen'] = {str(k): float(v) for k, v in zip(fn, fv)}
    # Cheapest detector of a replot that bins the data differently from the
    # fit, which would report a chi2/dof for data the posterior never saw.
    n_obs_saved = float(meta.get('n_obs', np.nan))
    if np.isfinite(n_obs_saved) and int(n_obs_saved) != len(data.flux):
        warnings.warn(
            f"Replot is using {len(data.flux)} observed points but the saved fit used "
            f"{int(n_obs_saved)}. The reported chi2/dof will not match the original run; "
            f"check the data/binning flags or rerun --replot where a *{RUN_CONFIG_SUFFIX} "
            f"file restores them.")
    print(f"Loaded saved chain from: {chain_path}  shape {chain.shape} "
          f"(mode={saved['mode']}, likelihood={saved['likelihood']})")

    shape_names = WIND_SHAPE_FIT[spec.wind_model]
    spec = build_param_spec(
        likelihood=saved['likelihood'], mode=saved['mode'], wind_model=spec.wind_model,
        fit_wind_shape=any(n in loaded_names for n in shape_names),
        fit_scatter='f_scatter' in loaded_names, fit_fopacity='log_fopa' in loaded_names,
        frozen=saved['frozen'], orbital_period_s=saved['orbital_period_s'])
    missing = [n for n in spec.active_names if n not in loaded_names]
    if set(loaded_names) & set(saved['frozen']):
        print(f"Error: chain columns {loaded_names} include frozen parameters "
              f"{sorted(set(loaded_names) & set(saved['frozen']))}; the chain file is inconsistent.")
        return None
    if missing or chain.shape[2] != len(loaded_names):
        print(f"Error: chain columns {loaded_names} do not match the '{spec.mode}' mode "
              f"parameters {spec.active_names} (missing {missing}).")
        return None
    spec.active_names = list(loaded_names)     # the saved chain order is the active order
    samples = chain.reshape(-1, chain.shape[2])
    log_prob_flat = log_prob_chain.reshape(-1)
    print(f"  {len(samples)} post-burn samples; columns: {loaded_names}")

    stats = compute_statistics(samples, spec, log_prob=log_prob_flat)
    print_results(stats, spec, band)
    return postprocess_fit(args, spec, priors_for(spec), model, data, samples, stats, band,
                           chain, log_prob_flat, smoothed, sampler=None), spec


def write_summary(path: str, band: str, spec: ParamSpec, stats: Dict) -> None:
    """Human-readable summary of one fit (``{band}_{wind_model}_summary.txt``)."""
    with open(path, 'w') as f:
        f.write("MCMC Light Curve Fitting Results\n" + "=" * 60 + "\n\n")
        f.write(f"{band.upper()} Band - {WIND_MODELS[spec.wind_model]}\n" + "-" * 40 + "\n")
        f.write(f"Parameterization: {spec.mode}\n")
        if spec.frozen:
            f.write("Frozen: " + ", ".join(f"{k}={fmt_val(v)}" for k, v in sorted(spec.frozen.items())) + "\n")
        meta = stats.get('_run_meta', {})
        if meta:
            f.write("Run configuration:\n")
            f.write(f"  sampler={meta.get('sampler')}, likelihood={meta.get('likelihood')}, "
                    f"walkers={meta.get('n_walkers')}, steps={meta.get('n_steps')}, "
                    f"burn={meta.get('n_burn')}, seed={meta.get('seed')}\n")
            f.write(f"  fit_phase_shift={meta.get('fit_phase_shift')}, "
                    f"phase_shift_fixed={meta.get('phase_shift_fixed')}, "
                    f"phase_window={meta.get('phase_window')}, "
                    f"phase_shift_grid={meta.get('phase_shift_grid_size')}, "
                    f"phase_shift_resolution={meta.get('phase_shift_resolution')}\n")
            if np.isfinite(meta.get('fit_elapsed_s', np.nan)):
                f.write(f"  wall_time_s={meta['fit_elapsed_s']:.2f}\n")

        def row(name, s, tag=""):
            f.write(f"  {name}{tag}: {fmt_val(s['median'])} (+{fmt_val(s['upper'])}/-{fmt_val(s['lower'])})"
                    f"  [mean={fmt_val(s['mean'])}, std={fmt_val(s['std'])}]\n")

        f.write("Marginal posterior (median +upper/-lower, 16/84 pct):\n")
        for name in spec.active_names:
            row(name, stats[name])
        derived = [n for n in spec.derived_names if n in stats]
        for name in derived:
            row(name, stats[name], " (derived)")

        if any('map' in stats[n] for n in spec.active_names):
            lp = stats.get('_map_meta', {}).get('log_prob')
            f.write(f"Best-fit (MAP, max log-prob){f'  (log_prob = {lp:.3f})' if lp is not None else ''}:\n")
            for name in spec.active_names:
                f.write(f"  {name}: {fmt_val(stats[name]['map'])}\n")
            for name in derived:
                if 'map' in stats[name]:
                    f.write(f"  {name} (derived): {fmt_val(stats[name]['map'])}\n")

        if 'reduced_chi2' in stats:
            f.write(f"Reduced chi-square: {stats['reduced_chi2']:.3f}\n")
        if np.isfinite(stats.get('bic', np.nan)):
            f.write(f"BIC: {stats['bic']:.3f} (logL_hat={stats.get('logL_hat', np.nan):.3f}, "
                    f"k={int(stats['k_params'])}, n={int(stats['n_obs'])}, "
                    f"source={stats.get('theta_source', 'unknown')})\n")

        diag = stats.get('_diagnostics') or {}
        if diag:
            f.write("Chain diagnostics:\n")
            if np.isfinite(diag.get('acceptance_fraction_mean', np.nan)):
                f.write(f"  acceptance_fraction_mean: {diag['acceptance_fraction_mean']:.4f}\n")
            taus = diag.get('autocorr_time') or {}
            vals = [float(v) for v in taus.values() if np.isfinite(v)]
            if vals:
                f.write(f"  autocorr_time_steps: min={np.min(vals):.2f}, median={np.median(vals):.2f}, "
                        f"max={np.max(vals):.2f}\n")
                f.write("  autocorr_time_steps per parameter:\n")
                n_steps = meta.get('n_steps')
                for pname, tau in taus.items():
                    tau = float(tau)
                    if not np.isfinite(tau) or tau <= 0:
                        f.write(f"    {pname}: n/a\n")
                        continue
                    note = ""
                    if n_steps:
                        n_tau = float(n_steps) / tau
                        note = f"  ({n_tau:.1f} tau in chain, {'OK' if n_tau >= 50 else '<50 -> unconverged'})"
                    f.write(f"    {pname}: {tau:.2f}{note}\n")
            if diag.get('effective_independent_samples') is not None:
                f.write(f"  effective_independent_samples: {int(diag['effective_independent_samples'])}\n")
            if diag.get('converged') is not None:
                f.write(f"  converged: {bool(diag['converged'])}\n")
        f.write("\n")
    print(f"\nSummary saved to: {path}")


# =============================================================================
# CLI
# =============================================================================

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
                raise ValueError("expected 4 values")
        except Exception as e:
            parser.error(f"Invalid format for --prior-{name}: {e}")
        mean, std, lo, hi = parts
        if std <= 0 or not np.isfinite(std):
            parser.error(f"--prior-{name}: STD must be > 0 (got {std}).")
        if not lo < hi:
            parser.error(f"--prior-{name}: MIN must be < MAX (got {lo}, {hi}).")
        if not lo < mean < hi:
            parser.error(f"--prior-{name}: MEAN {mean} lies outside the box ({lo}, {hi}); every "
                         f"walker would start at the box edge.")
        out[name] = dict(zip(('mean', 'std', 'min', 'max'), parts))
        print(f"Custom prior for {kind}{name}: mean={parts[0]}, std={parts[1]}, "
              f"min={parts[2]}, max={parts[3]}")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MCMC fitting of XRB light curves to observed Chandra data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--band", type=str, default=None,
                        choices=['broad', 'soft', 'medium', 'hard'],
                        help="Energy band to fit; the flux table must contain it (the model is "
                             "run one band at a time). Required, except with --replot, where it "
                             "is restored from the saved run config.")
    parser.add_argument("--flux-csv", type=str, default=None,
                        help="Flux vs nH CSV (from cloak/flux_table.py). Required, except with "
                             "--replot, where it is restored from the saved run config.")
    parser.add_argument("--wind-model", type=str, choices=list(WIND_MODELS), default='smooth_pl',
                        help="Wind density model: " + ", ".join(f"{k} ({v})" for k, v in WIND_MODELS.items()))
    parser.add_argument("--fit-wind-shape", action="store_true",
                        help="Add the wind-shape parameters of the chosen --wind-model as free "
                             "dimensions (smooth_pl: Rb, p; confinement: fconf, ell; beta_law: "
                             "beta, H). Override priors via --prior-<name>.")

    # Data
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Light-curve directory: a direct path to .txt files, or a parent "
                             "with {band}/, {band}/single/ or {Band}_with_flux/ sub-folders. "
                             "Required, except with --replot (restored from the run config).")
    parser.add_argument("--obs-column", type=str, default="FLUX",
                        help="Observable column in the data files (e.g. flux_t, rate, FLUX, NET_RATE)")
    parser.add_argument("--obs-error-column", type=str, default=None,
                        help="Error column. If omitted, auto-detected from --obs-column; for "
                             "proportional columns such as flux_t it is derived from rate_err.")
    parser.add_argument("--time-column", type=str, default=None,
                        help="Timestamp column (e.g. TIME, t_raw). Auto-detected if omitted.")
    parser.add_argument("--n-phase-bins", type=int, default=None,
                        help="Fixed-width phase binning with this many bins. Mutually exclusive "
                             "with --counts-per-bin; if neither is given, 50 fixed bins are used.")
    parser.add_argument("--counts-per-bin", type=int, default=None,
                        help="Adaptive binning with ~constant counts per bin (recommended: 100). "
                             "Mutually exclusive with --n-phase-bins.")
    parser.add_argument("--no-phase-bin", action="store_true",
                        help="Fit the raw 100 s points; pair with --likelihood jitter.")
    parser.add_argument("--keep-zero-flux", action="store_true",
                        help="Keep rows with flux <= 0 (observed zero-count bins) instead of dropping them "
                             "on load. For Poisson count data whose exposure is known (an exposure column, "
                             "or counts and rate) they belong in the exposure-weighted bins; a CIAO-layout "
                             "file cannot tell an observed empty bin from an unobserved GTI gap, which is "
                             "why dropping is the default. Keeping them lowers the mid-eclipse mean that "
                             "centres the f_scatter prior.")
    parser.add_argument("--phase-window", nargs=2, type=float, default=(0.0, 1.0), metavar=("LO", "HI"),
                        help="Fit only the data with phase in [LO, HI) (LO > HI wraps through 0); the "
                             "model is still evaluated over the full orbit. A partial window needs a "
                             "fixed phase shift (--phase-shift or --no-fit-phase-shift): the model is "
                             "symmetric about mid-eclipse, so with one eclipse edge in the data the "
                             "eclipse width is degenerate with a free shift. Take the shift from a "
                             "full-orbit fit.")

    norm = parser.add_argument_group(
        'Wind Normalization',
        "The density is fixed from --mdot / --v-inf, so the column carries real units and the "
        "eclipse emerges from wind opacity rather than a geometric cutoff.")
    norm.add_argument("--mdot", type=float, default=SIM_DEFAULTS['mdot'],
                      help="Mass-loss rate in Msun/yr (default: Clark & Crowther 2004, clumping-corrected)")
    norm.add_argument("--v-inf", type=float, default=SIM_DEFAULTS['v_inf'], help="Wind terminal velocity in km/s")
    norm.add_argument("--mu-wind", type=float, default=SIM_DEFAULTS['mu_wind'],
                      help="Mean mass per hydrogen-equivalent nucleus for the TBabs column")
    norm.add_argument("--fit-fopacity", action="store_true",
                      help="Fit log10(f_opacity), the effective-opacity factor absorbing wind "
                           "ionization, clumping and abundance departures. Strongly recommended.")

    # Sampling
    parser.add_argument("--sampler", type=str, choices=list(SAMPLER_TYPES), default='emcee',
                        help="'emcee' (stretch move) or 'zeus' (slice sampler; better for "
                             "correlated posteriors; pip install zeus-mcmc)")
    parser.add_argument("--likelihood", type=str, choices=list(LIKELIHOOD_TYPES), default='chi2',
                        help="'chi2' (Gaussian) or 'jitter' (Gaussian with a free fractional "
                             "systematic term log_f)")
    parser.add_argument("--reparam", action="store_true",
                        help="Sample (a = d1+d2, q = d1/a) instead of (d1, d2); q is exactly "
                             "unidentifiable, so this puts the flat direction on its own axis.")
    parser.add_argument("--kepler", action="store_true",
                        help="Sample (M_X, M_RH) and derive a, q from Kepler's third law. Prefer "
                             "--kepler-mtot: the flat mass-ratio direction runs diagonally here.")
    parser.add_argument("--kepler-mtot", action="store_true",
                        help="Sample (M_tot, q_m = M_RH/M_tot); only M_tot enters the light curve "
                             "via a = K*M_tot^(1/3), q_m's posterior equals its prior, and M_X / "
                             "M_RH are derived. Freeze q_m to drop the dead dimension.")
    parser.add_argument("--orbital-period", type=float, default=float(ORBITAL_PERIOD),
                        help="Orbital period in seconds for Kepler's third law (--kepler / "
                             "--kepler-mtot only); the light curves are folded with utils.ORBITAL_PERIOD")
    parser.add_argument("--freeze", type=str, default=None, metavar="NAME=VAL[,NAME=VAL,...]",
                        help="Pin parameters and drop them from the chain. Names: d1,d2,a,q,r,R,"
                             "i0,M_X,M_RH,M_tot,q_m,f_scatter,log_fopa,Rb,p,fconf,ell,beta,H. "
                             "log_fopa pins the opacity at 10**VALUE; log_f cannot be frozen.")
    parser.add_argument("--fit-scatter", action="store_true",
                        help="Add a free constant scattered-flux floor f_scatter.")
    parser.add_argument("--scatter-eclipse-phase", nargs=2, type=float, default=(0.4, 0.6),
                        metavar=("PHASE_MIN", "PHASE_MAX"),
                        help="Phase window whose mean flux centres the f_scatter prior.")
    parser.add_argument("--n-walkers", type=int, default=32, help="Number of walkers")
    parser.add_argument("--n-steps", type=int, default=5000, help="Number of steps")
    parser.add_argument("--n-burn", type=int, default=1000, help="Burn-in steps to discard")
    parser.add_argument("--no-fit-phase-shift", action="store_true",
                        help="Disable the per-sample phase-shift alignment and hold the shift at 0 "
                             "(by default every likelihood call minimises chi2 over a phase shift).")
    parser.add_argument("--phase-shift", type=float, default=None, metavar="SHIFT",
                        help="Hold the model phase shift at this value (no search), e.g. the shift "
                             "of a full-orbit fit when fitting a --phase-window.")
    parser.add_argument("--phase-shift-grid-size", type=int, default=None,
                        help="Coarse trial shifts per likelihood call before the dense refinement. "
                             "Default: max(number of data points, model phases), at most 400.")
    parser.add_argument("--dth", type=float, default=2.0,
                        help="Model phase resolution in degrees (360/dth phases, half of them by "
                             "reflection; larger = faster, coarser eclipse edges)")

    # Everything in these two groups controls one invocation, not the fit, and is
    # therefore never restored by --replot (see NEVER_RESTORED_DESTS).
    execution = parser.add_argument_group(
        'Execution', 'How this invocation runs; not part of the fit definition.')
    execution.add_argument("--n-threads", type=int, default=1,
                           help="Worker processes for parallel likelihood evaluation (1 = serial). The "
                                "kernel already uses every core through numba, so pooling only pays off "
                                "on many-core nodes together with --numba-threads-per-worker 1; on a "
                                "laptop it is slower than serial.")
    execution.add_argument("--numba-threads-per-worker", type=int, default=None, metavar="N",
                           help="Numba threads inside each worker when --n-threads > 1 "
                                "(default: cpu_count // n_threads)")
    execution.add_argument("--seed", type=int, default=None,
                           help="Seed for the walker initialisation, the sampler moves and every "
                                "random subset, so a run can be reproduced exactly")
    execution.add_argument("--quiet", action="store_true", help="Suppress the progress bar")
    execution.add_argument("--replot", action="store_true",
                           help="Regenerate outputs from the saved chain without re-sampling. Every "
                                "option not typed is restored from the run's <band>_<wind>_run_config.json.")

    output = parser.add_argument_group('Output', 'What this invocation writes.')
    output.add_argument("--output-dir", type=str, default="mcmc_results", help="Output directory")
    output.add_argument("--no-geometry-plots", action="store_true",
                        help="Skip the projected-orbit, geometry-vs-phase and wind-profile figures")
    output.add_argument("--no-plots", action="store_true", help="Skip all figures")
    output.add_argument("--smooth", action="store_true",
                        help="Overlay a Gaussian-smoothed observed curve with its 1-sigma band")
    output.add_argument("--smooth-sigma", type=float, default=0.01, help="Smoothing kernel width in phase")
    output.add_argument("--compute-bic", action="store_true",
                        help="Report the Bayesian information criterion at the MAP sample")
    output.add_argument("--no-csv-output", action="store_true",
                        help="Skip the *_samples.csv export (the chain NPZ is always written)")
    output.add_argument("--csv-chunk-size", type=int, default=50000, help="Rows per chunk when writing CSV")
    output.add_argument("--save-chi2", action="store_true",
                        help="Write per-sample chi2 to {band}_{wind_model}_chi2.csv.gz")
    output.add_argument("--chi2-n-samples", type=int, default=None,
                        help="Random subset size for --save-chi2. Default: every sample with "
                             "--likelihood chi2 (read from the chain, no model calls), "
                             f"{CHI2_TABLE_DEFAULT_SAMPLES} with jitter (one model call each).")
    parser.set_defaults(_never_restore=frozenset(
        a.dest for group in (execution, output) for a in group._group_actions))

    sim = parser.add_argument_group('Simulation Parameters', 'Passed to simulate_lightcurve')
    sim.add_argument("--gma0", type=float, default=SIM_DEFAULTS['gma0'], help="Starting phase angle in degrees")
    sim.add_argument("--d2h", type=float, default=SIM_DEFAULTS['d2h'],
                     help="Angular cell size of the emitter grid (degrees)")

    prior_group = parser.add_argument_group(
        'Prior Customization',
        'Override the geometry priors of the active parameterization (format: mean,std,min,max).')
    for flag, dest, prior, desc in (
        ("d1", None, MODES['phys']['scale_priors']['d1'], "d1 (compact-object distance from the COM)"),
        ("d2", None, MODES['phys']['scale_priors']['d2'], "d2 (companion distance from the COM)"),
        ("r", None, SMALL_R_PRIOR, "r (compact object / disk radius)"),
        ("R", None, R_PRIOR, "R (companion photospheric radius)"),
        ("i0", None, I0_PRIOR, "i0 (inclination, degrees from the orbital-plane normal; 90 = edge-on)"),
        ("a", None, MODES['reparam']['scale_priors']['a'], "a = d1+d2 (--reparam)"),
        ("q", None, MODES['reparam']['scale_priors']['q'], "q = d1/(d1+d2) (--reparam; unidentifiable)"),
        ("MX", "prior_M_X", MODES['kepler']['scale_priors']['M_X'], "M_X (Msun, --kepler)"),
        ("MRH", "prior_M_RH", MODES['kepler']['scale_priors']['M_RH'], "M_RH (Msun, --kepler)"),
        ("Mtot", "prior_M_tot", MODES['kepler_mtot']['scale_priors']['M_tot'],
         "M_tot (Msun, --kepler-mtot); sets a = K*M_tot^(1/3)"),
        ("qm", "prior_q_m", MODES['kepler_mtot']['scale_priors']['q_m'],
         "q_m = M_RH/M_tot (--kepler-mtot); unidentifiable, so this prior IS the posterior"),
    ):
        prior_group.add_argument(
            f"--prior-{flag}", type=str, default=None, metavar="MEAN,STD,MIN,MAX",
            help=f"Prior for {desc}. Default: {prior['mean']},{prior['std']},{prior['min']},{prior['max']}",
            **({'dest': dest} if dest else {}))

    prior_group.add_argument(
        "--prior-fopa", type=str, default=None, metavar="MEAN,STD,MIN,MAX", dest="prior_log_fopa",
        help=f"Prior for log10 f_opacity (--fit-fopacity). Default: {FOPACITY_PRIOR['mean']},"
             f"{FOPACITY_PRIOR['std']},{FOPACITY_PRIOR['min']},{FOPACITY_PRIOR['max']}")

    shape_group = parser.add_argument_group(
        'Wind-Shape Prior Customization', 'Only active with --fit-wind-shape (format: mean,std,min,max).')
    for sname in ALL_WIND_SHAPE_NAMES:
        p = WIND_SHAPE_PRIORS[sname]
        shape_group.add_argument(
            f"--prior-{sname}", type=str, default=None, metavar="MEAN,STD,MIN,MAX",
            help=f"Prior for wind-shape parameter '{sname}'. Default: {p['mean']},{p['std']},{p['min']},{p['max']}")
    return parser


# Dests of the Execution and Output groups plus the two that define a replot
# itself; apply_saved_run_config never restores them.
NEVER_RESTORED_DESTS = frozenset(build_parser().get_default('_never_restore')) | {'replot', 'output_dir'}


def load_fit_data(args, band: str) -> Tuple[FitData, Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    """Load, filter and bin the observations for *band*; also the smoothed curve
    and the data-driven f_scatter prior when requested."""
    obs_df = load_observed_lightcurves(band, args.data_dir, flux_column=args.obs_column,
                                       error_column=args.obs_error_column,
                                       time_column=args.time_column,
                                       drop_nonpositive_flux=not args.keep_zero_flux)
    obs_df = apply_phase_window(obs_df, *args.phase_window)
    if 'flux_err' in obs_df.columns and not np.isfinite(obs_df['flux_err']).any():
        if args.no_phase_bin:
            raise ValueError("The light curves carry no measurement errors and --no-phase-bin "
                             "needs them; bin the data or add an error column.")
        warnings.warn("The light curves carry no measurement errors; the binned errors come "
                      "from the scatter within each bin (std / sqrt(n)).")
    is_binned = not args.no_phase_bin
    if is_binned:
        cols = dict(rate_column='flux', error_column='flux_err')
        if args.counts_per_bin is not None:
            obs_df = phase_bin_data_snr(obs_df, counts_per_bin=args.counts_per_bin,
                                        phase_origin=args.phase_window[0], **cols)
        else:
            obs_df = phase_bin_data(obs_df, n_bins=(args.n_phase_bins or 50), **cols)

    phase = obs_df['phase'].to_numpy(dtype=float)
    flux = obs_df['flux'].to_numpy(dtype=float)
    # Binned errors are > 0 whenever the inputs were; the one repair rule
    # covers degenerate bins and refuses data without errors altogether.
    err = sanitize_errors(obs_df['flux_err'], context="binned light curve: ")
    width = obs_df['width'].to_numpy(dtype=float) if ('width' in obs_df.columns and is_binned) else None

    fit_shift = shift_is_searched(args)
    data = FitData.build(phase, flux, err, fit_phase_shift=fit_shift,
                         shift_grid_size=args.phase_shift_grid_size,
                         n_model=int(round(360.0 / float(args.dth))),
                         fixed_shift=(args.phase_shift or 0.0),
                         is_binned=is_binned, phase_width=width)

    smoothed = None
    if args.smooth:
        grid = np.linspace(0.0, 1.0, 300, endpoint=False)
        smoothed = smooth_lightcurve(phase, flux, err, sigma=float(args.smooth_sigma),
                                     eval_phase=grid[in_phase_window(grid, *args.phase_window)],
                                     verbose=not args.quiet)

    scatter_prior = None
    if args.fit_scatter:
        centre = estimate_scattered_flux(phase, flux, window=tuple(map(float, args.scatter_eclipse_phase)))
        flux_max = float(np.nanmax(flux)) if np.any(np.isfinite(flux)) else 1.0
        tiny = max(1e-30, abs(float(np.nanmedian(flux))) * 1e-6)
        scatter_prior = {'mean': float(centre), 'std': float(max(centre, tiny)),
                         'min': 0.0, 'max': float(max(flux_max, centre + tiny))}
        if not args.quiet:
            print("Scatter prior: mean={mean:.4g}, std={std:.4g}, min={min:.4g}, max={max:.4g}"
                  .format(**scatter_prior))
    return data, smoothed, scatter_prior


def shift_is_searched(args) -> bool:
    """True when the per-sample phase-shift search is on (no fixed shift given)."""
    return not args.no_fit_phase_shift and args.phase_shift is None


def validate_args(parser: argparse.ArgumentParser, args, spec: ParamSpec, frozen: Dict[str, float],
                  explicit: set) -> None:
    """Reject argument combinations that contradict each other or have no effect.

    *explicit* is the set of dests the user typed (utils.explicit_cli_dests):
    an option is only an error for "no effect" when it was actually given, so
    values restored by --replot never trip these checks.
    """
    err = parser.error
    flag = dest_to_flag(parser)
    mode = spec.mode

    # --- data, binning, phase window (rules shared with cloak.phase_analysis) ---
    validate_binning_args(err, args)
    fit_shift = shift_is_searched(args)
    validate_phase_window_args(
        err, args, fit_shift_enabled=fit_shift,
        fixed_shift_hint="pass --phase-shift SHIFT (the shift of a full-orbit fit) or "
                         "--no-fit-phase-shift to hold it at 0.",
        scatter_window_used=args.fit_scatter)
    if args.no_fit_phase_shift and args.phase_shift is not None:
        err("--no-fit-phase-shift holds the shift at 0 and --phase-shift holds it at SHIFT; use one.")
    if not fit_shift and 'phase_shift_grid_size' in explicit:
        err("--phase-shift-grid-size has no effect when the phase shift is held fixed.")
    if args.phase_shift_grid_size is not None and args.phase_shift_grid_size < 3:
        err("--phase-shift-grid-size must be >= 3.")

    # --- scattered flux --------------------------------------------------------
    if 'scatter_eclipse_phase' in explicit and not args.fit_scatter:
        err("--scatter-eclipse-phase only centres the f_scatter prior; add --fit-scatter.")
    if args.fit_scatter and 'f_scatter' in frozen:
        err("--fit-scatter and --freeze f_scatter contradict each other.")
    if args.fit_fopacity and 'log_fopa' in frozen:
        err("--fit-fopacity and --freeze log_fopa contradict each other.")

    # --- sampling --------------------------------------------------------------
    sampling = {'n_walkers', 'n_steps', 'n_burn', 'sampler', 'n_threads', 'numba_threads_per_worker'}
    if args.replot:
        typed = sorted(flag[d] for d in sampling & explicit)
        if typed:
            err(f"--replot does not sample; {', '.join(typed)} "
                f"{'has' if len(typed) == 1 else 'have'} no effect.")
    else:
        if args.n_steps <= 0:
            err("--n-steps must be > 0.")
        if not (0 <= args.n_burn < args.n_steps):
            err(f"--n-burn must satisfy 0 <= n_burn < n_steps "
                f"(got n_burn={args.n_burn}, n_steps={args.n_steps}).")
        if args.n_walkers < 2 * spec.n_dim or args.n_walkers % 2:
            err(f"--n-walkers must be even and at least 2 x n_dim = {2 * spec.n_dim} "
                f"for the {spec.n_dim} sampled parameters {spec.active_names}.")
        if args.n_threads < 1:
            err("--n-threads must be >= 1.")
        if args.numba_threads_per_worker is not None:
            if args.n_threads <= 1:
                err("--numba-threads-per-worker only applies to pooled runs (--n-threads > 1).")
            import numba
            if not 1 <= args.numba_threads_per_worker <= numba.config.NUMBA_NUM_THREADS:
                err(f"--numba-threads-per-worker must be in [1, {numba.config.NUMBA_NUM_THREADS}] "
                    f"(numba's thread limit on this machine).")

    if args.seed is not None and not (0 <= args.seed < 2 ** 32):
        err("--seed must be in [0, 2^32).")

    # --- model -----------------------------------------------------------------
    for name in ('dth', 'd2h'):
        value = getattr(args, name)
        if value <= 0 or abs(360.0 / value - round(360.0 / value)) > 1e-9:
            err(f"{flag[name]} must be positive and divide 360 evenly (got {value}).")
    for name in ('mdot', 'v_inf', 'mu_wind'):
        if getattr(args, name) <= 0:
            err(f"{flag[name]} must be > 0.")
    if args.orbital_period <= 0:
        err("--orbital-period must be > 0.")
    if 'orbital_period' in explicit and mode not in ('kepler', 'kepler_mtot'):
        err("--orbital-period only enters Kepler's third law: use it with --kepler or "
            "--kepler-mtot (the light curves are folded with utils.ORBITAL_PERIOD regardless).")

    # --- priors ----------------------------------------------------------------
    active_geometry = set(geometry_names(mode))
    all_geometry = {n for m in MODES.values() for n in m['scale_names']} | {'r', 'R', 'i0'}
    for dest in sorted(explicit):
        if not dest.startswith('prior_'):
            continue
        name = dest[len('prior_'):]
        if name in all_geometry:
            if name not in active_geometry:
                owner = next(m for m, cfg in MODES.items() if name in cfg['scale_names'])
                err(f"{flag[dest]} belongs to the '{owner}' parameterization "
                    f"({MODES[owner]['flag'] or 'no mode flag'}); the run uses '{mode}'.")
        elif name in ALL_WIND_SHAPE_NAMES:
            if name not in WIND_SHAPE_FIT[spec.wind_model]:
                err(f"{flag[dest]} is not a shape parameter of --wind-model {spec.wind_model} "
                    f"({WIND_SHAPE_FIT[spec.wind_model]}).")
            if not spec.fit_wind_shape and name not in frozen:
                err(f"{flag[dest]} has no effect without --fit-wind-shape.")

    # --- output ----------------------------------------------------------------
    if 'prior_log_fopa' in explicit and not (args.fit_fopacity or 'log_fopa' in frozen):
        err("--prior-fopa has no effect without --fit-fopacity.")
    if 'chi2_n_samples' in explicit and not args.save_chi2:
        err("--chi2-n-samples has no effect without --save-chi2.")
    if args.chi2_n_samples is not None and args.chi2_n_samples <= 0:
        err("--chi2-n-samples must be > 0.")
    if 'smooth_sigma' in explicit and not args.smooth:
        err("--smooth-sigma has no effect without --smooth.")
    if args.smooth_sigma <= 0:
        err("--smooth-sigma must be > 0.")
    if 'csv_chunk_size' in explicit and args.no_csv_output:
        err("--csv-chunk-size has no effect with --no-csv-output.")
    if args.csv_chunk_size <= 0:
        err("--csv-chunk-size must be > 0.")


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)   # keep stdout and stderr in order in log files
    parser = build_parser()
    args = parser.parse_args()
    explicit = explicit_cli_dests(parser)

    # Restore everything not typed explicitly before validating or deriving.
    restored_config = None
    if args.replot:
        restored_config = apply_saved_run_config(parser, args, explicit=explicit,
                                                 never_restore=NEVER_RESTORED_DESTS)
        if restored_config is None:
            print(f"\nNo saved run config found in {args.output_dir} (looked for *{RUN_CONFIG_SUFFIX}). "
                  f"Using the command line and argparse defaults.")
    for dest, flag in (('band', '--band'), ('flux_csv', '--flux-csv'), ('data_dir', '--data-dir')):
        if getattr(args, dest, None) is None:
            parser.error(f"{flag} is required (with --replot it is restored from a saved "
                         f"*{RUN_CONFIG_SUFFIX} in --output-dir, if one exists).")
    try:
        mode = mode_from_flags(args.reparam, args.kepler, args.kepler_mtot)
        frozen = parse_freeze_map(args.freeze)
        spec = build_param_spec(likelihood=args.likelihood, mode=mode, wind_model=args.wind_model,
                                fit_wind_shape=args.fit_wind_shape, fit_scatter=args.fit_scatter,
                                fit_fopacity=args.fit_fopacity, frozen=frozen,
                                orbital_period_s=float(args.orbital_period))
    except Exception as e:
        parser.error(str(e))
    validate_args(parser, args, spec, frozen, explicit)
    args._prior_typed = any(d.startswith('prior_') for d in explicit)
    if args.seed is not None:
        # Covers initial_positions, the chi2 subsample and the wind-profile
        # draws; run_mcmc hands the same state to emcee. zeus draws its walker
        # pairs with the stdlib random module, hence the second seed.
        np.random.seed(args.seed)
        random.seed(args.seed)

    geometry_priors = default_geometry_priors(mode)
    geometry_priors.update(_parse_prior_overrides(parser, args, geometry_names(mode)))
    shape_prior_overrides = _parse_prior_overrides(parser, args, ALL_WIND_SHAPE_NAMES, kind="shape param ")
    shape_prior_overrides.update(_parse_prior_overrides(parser, args, ('log_fopa',)))
    for fname, fval in frozen.items():
        fp = geometry_priors.get(fname) or ({**WIND_SHAPE_PRIORS[fname], **shape_prior_overrides.get(fname, {})}
                                            if fname in WIND_SHAPE_PRIORS else None)
        if fp is not None and not (fp['min'] < fval < fp['max']):
            warnings.warn(f"Frozen value {fname}={fval} lies outside the prior box "
                          f"({fp['min']}, {fp['max']}); continuing since frozen values are constants.")

    sim_params = {'gma0': args.gma0, 'd2h': args.d2h, 'mdot': args.mdot,
                  'v_inf': args.v_inf, 'mu_wind': args.mu_wind}
    os.makedirs(args.output_dir, exist_ok=True)
    band = args.band

    try:
        data, smoothed, scatter_prior = load_fit_data(args, band)
        model = DirectLightCurveModel(band=band, flux_csv_path=args.flux_csv,
                                      wind_model=args.wind_model, dth=args.dth, sim_params=sim_params)

        def priors_for(s: ParamSpec) -> Dict[str, Dict[str, float]]:
            # A replot rebuilds the spec from the chain; if its mode differs from
            # the command line's, the command-line geometry priors do not apply.
            gp = geometry_priors if s.mode == mode else default_geometry_priors(s.mode)
            if s.mode != mode:
                warnings.warn(f"The saved chain uses the '{s.mode}' parameterization, not "
                              f"'{mode}'; default geometry priors of '{s.mode}' are used.")
            return get_active_priors(s, gp, shape_prior_overrides, scatter_prior)

        if args.replot:
            result = replot_from_existing(band, args, spec, model, data, smoothed, priors_for)
            if result is None:
                print(f"Could not load existing results for {band}_{args.wind_model}")
                sys.exit(1)
            stats, spec = result
            # Self-healing: results predating run-config saving get one written
            # once a replot has succeeded with these options, so the next
            # --replot needs no arguments.
            if restored_config is None and not os.path.exists(run_config_path(args.output_dir, band, args.wind_model)):
                save_run_config(args.output_dir, band, args.wind_model, args)
        else:
            previous = os.path.join(args.output_dir, f"{band}_{args.wind_model}_chain.npz")
            if os.path.exists(previous):
                print(f"Warning: {previous} exists; this run overwrites the previous "
                      f"{band}/{args.wind_model} results in {args.output_dir}.")
            # Written before sampling so the configuration survives a crash.
            save_run_config(args.output_dir, band, args.wind_model, args)
            stats = run_single_fit(band, args, spec, priors_for(spec), model, data, smoothed)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError, pd.errors.EmptyDataError) as e:
        # Expected user-facing failures: one line, no traceback.
        print(f"ERROR {'replotting' if args.replot else 'fitting'} {band} band ({args.wind_model}): {e}",
              file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR {'replotting' if args.replot else 'fitting'} {band} band ({args.wind_model}): {e}",
              file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    write_summary(os.path.join(args.output_dir, f"{band}_{spec.wind_model}_summary.txt"), band, spec, stats)
    print("\nReplotting complete!" if args.replot else "\nMCMC fitting complete!")


if __name__ == "__main__":
    main()
