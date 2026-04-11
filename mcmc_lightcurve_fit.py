#!/usr/bin/env python3
"""
MCMC Light Curve Fitting for X-ray Binary Systems
--------------------------------------------------
This module performs Markov Chain Monte Carlo (MCMC) fitting to find optimal
binary system parameters by fitting model light curves to observed Chandra data.

WIND MODELS:
- av: Accelerated velocity wind (beta-law wind profile)
- cv: Constant velocity wind (uniform outflow)

PERFORMANCE OPTIMIZATION:
This code pre-computes a grid of model light curves and uses N-dimensional
interpolation during MCMC sampling. This provides ~1000x speedup compared to
calling simulate_lightcurve() for each MCMC step.

The grid can be saved to disk and reloaded for subsequent MCMC runs.

Parameters being fit:
- d1: Distance of compact object from center of mass (solar radii)
- d2: Distance of companion star from center of mass (solar radii)  
- r: Radius of compact object/accretion disk (solar radii)
- R: Radius of companion star (solar radii)
- i0: Orbital inclination (degrees)

Simulation parameters (passed to simulate_lightcurve):
- lam/lam2: Scaling parameter for nH conversion (affects flux normalization)
- gma0: Starting phase angle
- d2h: Angular cell size for polar grid
- dz: Step size along line of sight

Usage:
    # Fit accelerated wind model (default)
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv
    
    # Use zeus sampler (better mixing for correlated posteriors)
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv --sampler zeus
    
    # Use Student-t likelihood (robust to outliers)
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv --likelihood studentt --studentt-nu 5
    
    # Use jitter likelihood (adds free systematic error parameter)
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv --likelihood jitter
    
    # Compute WAIC/LOO model comparison metrics
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv --compute-waic
    
    # Save grid for reuse
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv --save-grid grids/broad_grid.npz
    
    # Load pre-computed grid (skip grid computation)
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv --load-grid grids/broad_grid.npz
    
    # Custom simulation parameters (e.g., different nH scaling)
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv --lam 0.511314
    
    # Custom priors (e.g., higher inclination starting point)
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv --prior-i0 65.0,15.0,30.0,85.0
"""

import argparse
import glob
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

from xrb_lightcurve import simulate_lightcurve
from chandra_phase_analysis import (
    REF_EPOCH,
    ORBITAL_PERIOD,
    frac,
    load_data as _load_data_base,
    phase_bin_data as _phase_bin_data_base,
)

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

# Wind model descriptions
WIND_MODELS = {
    'av': 'Accelerated Velocity Wind',
    'cv': 'Constant Velocity Wind'
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


def get_param_config(likelihood: str = 'chi2'):
    """Return (param_names, param_labels) for the given likelihood type."""
    names = list(PARAM_NAMES)
    labels = list(PARAM_LABELS)
    if likelihood == 'jitter':
        names.append('log_f')
        labels.append(r'$\ln\,f$')
    return names, labels


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
    error_column: str = "FLUX_ERR",
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
    error_column : str
        Column name for flux errors in the data files
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

    valid = (combined['flux'] > 0) & np.isfinite(combined['flux']) & np.isfinite(combined['time'])
    combined = combined.loc[valid].reset_index(drop=True)

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
    """Worker function to compute a single model (for parallel processing).
    
    Returns both av and cv flux arrays.
    """
    d1, d2, r, R, i0, flux_csv_path, band, dth, sim_params = args
    
    flux_column_av = f"nfl_{band}_av"
    flux_column_cv = f"nfl_{band}_cv"
    
    try:
        results = simulate_lightcurve(
            r=r, R=R, d1=d1, d2=d2,
            gma0=sim_params.get('gma0', -90.0),
            i0=i0,
            dth=dth,
            d2h=sim_params.get('d2h', 6.0),
            dz=sim_params.get('dz', 0.1),
            flux_method="interpolate",
            flux_csv_path=flux_csv_path,
            lam=sim_params.get('lam', 0.589537),
            lam2=sim_params.get('lam2', 0.589537),
            verbose=False
        )
        
        # Check columns exist
        if flux_column_av not in results.columns or flux_column_cv not in results.columns:
            return None, None
        
        # Return flux values at standard phases
        model_phase = results['phase'].values
        flux_av = results[flux_column_av].values
        flux_cv = results[flux_column_cv].values
        
        # Sort by phase
        sort_idx = np.argsort(model_phase)
        return flux_av[sort_idx], flux_cv[sort_idx]
        
    except Exception:
        return None, None


class PrecomputedModelGrid:
    """
    Pre-computed grid of model light curves for fast MCMC evaluation.
    
    This class pre-computes light curves on a coarse parameter grid and uses
    N-dimensional interpolation for fast evaluation during MCMC sampling.
    
    Supports both accelerated velocity (av) and constant velocity (cv) wind models.
    The grid can be saved/loaded to avoid recomputation.
    """
    
    def __init__(
        self,
        band: str,
        flux_csv_path: str,
        wind_model: str = 'av',
        priors: Dict = DEFAULT_PRIORS,
        grid_points: Dict[str, int] = None,
        dth: float = 5.0,
        n_workers: int = None,
        verbose: bool = True,
        load_path: str = None,
        sim_params: Dict = None
    ):
        """
        Initialize and pre-compute the model grid (or load from file).
        
        Parameters
        ----------
        band : str
            Energy band ('broad', 'soft', 'medium', 'hard')
        flux_csv_path : str
            Path to flux vs nH CSV file
        wind_model : str
            Wind model to use: 'av' (accelerated) or 'cv' (constant velocity)
        priors : dict
            Prior specifications (used to set grid bounds)
        grid_points : dict, optional
            Number of grid points per parameter.
        dth : float
            Phase resolution for model computation (degrees)
        n_workers : int, optional
            Number of parallel workers.
        verbose : bool
            Print progress messages
        load_path : str, optional
            Path to load pre-computed grid from. If provided, skips computation.
        sim_params : dict, optional
            Additional simulation parameters passed to simulate_lightcurve:
            - gma0: Starting phase angle in degrees (default -90.0)
            - d2h: Angular cell size for polar grid (default 6.0)
            - dz: Step size along line of sight (default 0.1)
            - lam: Scaling parameter for nH (default 0.589537)
            - lam2: Scaling parameter for constant velocity wind (default 0.589537)
        """
        self.band = band.lower()
        self.flux_csv_path = flux_csv_path
        self.wind_model = wind_model.lower()
        self.priors = priors
        self.dth = dth
        self.verbose = verbose
        self.sim_params = sim_params or {}
        
        if self.wind_model not in ['av', 'cv']:
            raise ValueError(f"wind_model must be 'av' or 'cv', got '{wind_model}'")
        
        # Default grid resolution
        if grid_points is None:
            grid_points = {'d1': 8, 'd2': 8, 'r': 5, 'R': 8, 'i0': 10}
        
        self.grid_points = grid_points
        self.n_workers = n_workers if n_workers else max(1, cpu_count() - 1)
        
        if load_path and os.path.exists(load_path):
            # Load pre-computed grid
            self._load_grid(load_path)
        else:
            # Create parameter grids and pre-compute
            self._create_grids()
            self._precompute_models()
        
        # Setup interpolators for the selected wind model
        self._setup_interpolators()
    
    def _create_grids(self):
        """Create 1D parameter grids."""
        self.param_grids = {}
        
        for param in PARAM_NAMES:
            p_min = self.priors[param]['min']
            p_max = self.priors[param]['max']
            n_pts = self.grid_points.get(param, 8)
            
            # Use log spacing for r (spans orders of magnitude)
            if param == 'r':
                self.param_grids[param] = np.logspace(
                    np.log10(p_min), np.log10(p_max), n_pts
                )
            else:
                self.param_grids[param] = np.linspace(p_min, p_max, n_pts)
        
        # Standard phase grid for output
        n_phase_points = int(360 / self.dth)
        self.phase_grid = np.linspace(0, 1 - 1/n_phase_points, n_phase_points)
        
        if self.verbose:
            total_models = np.prod([len(g) for g in self.param_grids.values()])
            print(f"\nGrid configuration: {total_models} models to compute")
            print(f"Wind model: {WIND_MODELS[self.wind_model]} ({self.wind_model})")
            for param, grid in self.param_grids.items():
                print(f"  {param}: {len(grid)} points [{grid[0]:.4f} - {grid[-1]:.4f}]")
    
    def _precompute_models(self):
        """Pre-compute all models on the grid."""
        if self.verbose:
            print(f"\nPre-computing model grid using {self.n_workers} workers...")
            start_time = time.time()
        
        # Generate all parameter combinations
        d1_grid = self.param_grids['d1']
        d2_grid = self.param_grids['d2']
        r_grid = self.param_grids['r']
        R_grid = self.param_grids['R']
        i0_grid = self.param_grids['i0']
        
        # Create list of all parameter combinations
        param_combos = []
        for i_d1, d1 in enumerate(d1_grid):
            for i_d2, d2 in enumerate(d2_grid):
                for i_r, r in enumerate(r_grid):
                    for i_R, R in enumerate(R_grid):
                        for i_i0, i0 in enumerate(i0_grid):
                            # Skip unphysical combinations
                            if r >= R:
                                continue
                            param_combos.append((
                                d1, d2, r, R, i0,
                                self.flux_csv_path, self.band, self.dth,
                                self.sim_params
                            ))
        
        if self.verbose:
            print(f"Computing {len(param_combos)} valid parameter combinations...")
        
        # Compute models in parallel with progress bar
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
                results = [_compute_single_model(args) for args in tqdm(param_combos, desc="Building model grid")]
            else:
                results = [_compute_single_model(args) for args in param_combos]
        
        # Build the N-dimensional grid array
        shape = (
            len(d1_grid), len(d2_grid), len(r_grid), len(R_grid), len(i0_grid),
            len(self.phase_grid)
        )
        self.flux_grid_av = np.full(shape, np.nan)
        self.flux_grid_cv = np.full(shape, np.nan)
        
        idx = 0
        for i_d1 in range(len(d1_grid)):
            for i_d2 in range(len(d2_grid)):
                for i_r in range(len(r_grid)):
                    for i_R in range(len(R_grid)):
                        for i_i0 in range(len(i0_grid)):
                            if r_grid[i_r] >= R_grid[i_R]:
                                continue
                            flux_av, flux_cv = results[idx]
                            if flux_av is not None:
                                self.flux_grid_av[i_d1, i_d2, i_r, i_R, i_i0, :] = flux_av
                            if flux_cv is not None:
                                self.flux_grid_cv[i_d1, i_d2, i_r, i_R, i_i0, :] = flux_cv
                            idx += 1
        
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"Grid pre-computation completed in {elapsed:.1f} seconds")
            valid_av = np.sum(~np.isnan(self.flux_grid_av)) / self.flux_grid_av.size
            valid_cv = np.sum(~np.isnan(self.flux_grid_cv)) / self.flux_grid_cv.size
            print(f"Valid grid coverage: AV={valid_av*100:.1f}%, CV={valid_cv*100:.1f}%")
    
    def _setup_interpolators(self):
        """Setup a single 6D interpolator (params + phase) for vectorized evaluation."""
        if self.wind_model == 'av':
            self.flux_grid = self.flux_grid_av
        else:
            self.flux_grid = self.flux_grid_cv

        flux_grid_clean = self.flux_grid.copy()
        if np.any(np.isnan(flux_grid_clean)):
            flux_grid_clean = np.where(
                np.isnan(flux_grid_clean),
                np.nanmean(flux_grid_clean),
                flux_grid_clean,
            )

        self._interp_6d = RegularGridInterpolator(
            (
                self.param_grids['d1'],
                self.param_grids['d2'],
                self.param_grids['r'],
                self.param_grids['R'],
                self.param_grids['i0'],
                self.phase_grid,
            ),
            flux_grid_clean,
            method='linear',
            bounds_error=False,
            fill_value=np.nan,
        )
    
    def switch_wind_model(self, wind_model: str):
        """
        Switch to a different wind model without recomputing the grid.
        
        Parameters
        ----------
        wind_model : str
            New wind model: 'av' or 'cv'
        """
        if wind_model.lower() not in ['av', 'cv']:
            raise ValueError(f"wind_model must be 'av' or 'cv', got '{wind_model}'")
        
        self.wind_model = wind_model.lower()
        self._setup_interpolators()
        
        if self.verbose:
            print(f"Switched to {WIND_MODELS[self.wind_model]} ({self.wind_model})")
    
    def save(self, filepath: str):
        """
        Save pre-computed grid to file for later reuse.
        
        Parameters
        ----------
        filepath : str
            Path to save the grid (.npz format)
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save sim_params as individual arrays (npz doesn't support dicts directly)
        sim_param_keys = list(self.sim_params.keys()) if self.sim_params else []
        sim_param_vals = [self.sim_params[k] for k in sim_param_keys] if self.sim_params else []
        
        # Save using numpy's compressed format
        np.savez_compressed(
            filepath,
            # Grids
            flux_grid_av=self.flux_grid_av,
            flux_grid_cv=self.flux_grid_cv,
            phase_grid=self.phase_grid,
            # Parameter grids
            d1_grid=self.param_grids['d1'],
            d2_grid=self.param_grids['d2'],
            r_grid=self.param_grids['r'],
            R_grid=self.param_grids['R'],
            i0_grid=self.param_grids['i0'],
            # Metadata
            band=self.band,
            dth=self.dth,
            grid_points=np.array([self.grid_points[p] for p in PARAM_NAMES]),
            # Simulation parameters
            sim_param_keys=np.array(sim_param_keys, dtype=str),
            sim_param_vals=np.array(sim_param_vals, dtype=float)
        )
        
        print(f"Grid saved to: {filepath}")
        print(f"  File size: {os.path.getsize(filepath) / 1024 / 1024:.1f} MB")
        if self.sim_params:
            print(f"  Simulation params: {self.sim_params}")
    
    def _load_grid(self, filepath: str):
        """
        Load pre-computed grid from file.
        
        Parameters
        ----------
        filepath : str
            Path to load the grid from (.npz format)
        """
        if self.verbose:
            print(f"\nLoading pre-computed grid from: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        
        # Load grids
        self.flux_grid_av = data['flux_grid_av']
        self.flux_grid_cv = data['flux_grid_cv']
        self.phase_grid = data['phase_grid']
        
        # Load parameter grids
        self.param_grids = {
            'd1': data['d1_grid'],
            'd2': data['d2_grid'],
            'r': data['r_grid'],
            'R': data['R_grid'],
            'i0': data['i0_grid']
        }
        
        # Load metadata
        loaded_band = str(data['band'])
        loaded_dth = float(data['dth'])
        
        # Load simulation parameters if present
        loaded_sim_params = {}
        if 'sim_param_keys' in data and 'sim_param_vals' in data:
            keys = data['sim_param_keys']
            vals = data['sim_param_vals']
            if len(keys) > 0:
                loaded_sim_params = dict(zip(keys, vals))
        
        # Verify band matches
        if loaded_band != self.band:
            warnings.warn(f"Loaded grid band '{loaded_band}' differs from requested '{self.band}'")
        
        # Check if sim_params match (warn if different)
        if self.sim_params and loaded_sim_params:
            for key in self.sim_params:
                if key in loaded_sim_params and self.sim_params[key] != loaded_sim_params[key]:
                    warnings.warn(
                        f"Loaded grid has {key}={loaded_sim_params[key]}, "
                        f"but requested {key}={self.sim_params[key]}. Using loaded value."
                    )
        
        self.dth = loaded_dth
        self.sim_params = loaded_sim_params
        
        if self.verbose:
            print(f"  Band: {loaded_band}, dth: {loaded_dth}")
            print(f"  Grid shape: {self.flux_grid_av.shape}")
            valid_av = np.sum(~np.isnan(self.flux_grid_av)) / self.flux_grid_av.size
            valid_cv = np.sum(~np.isnan(self.flux_grid_cv)) / self.flux_grid_cv.size
            print(f"  Valid coverage: AV={valid_av*100:.1f}%, CV={valid_cv*100:.1f}%")
            if loaded_sim_params:
                print(f"  Simulation params: {loaded_sim_params}")
    
    def evaluate(
        self,
        d1: float,
        d2: float,
        r: float,
        R: float,
        i0: float,
        obs_phases: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the model at given parameters using grid interpolation.
        
        Uses a single vectorized 6D interpolation call (params + phase)
        instead of looping over per-phase interpolators.
        
        Parameters
        ----------
        d1, d2, r, R, i0 : float
            Model parameters
        obs_phases : np.ndarray
            Orbital phases at which to evaluate model (0-1)
            
        Returns
        -------
        np.ndarray
            Interpolated flux values at requested phases
        """
        n_phase = len(self.phase_grid)
        points = np.empty((n_phase, 6))
        points[:, 0] = d1
        points[:, 1] = d2
        points[:, 2] = r
        points[:, 3] = R
        points[:, 4] = i0
        points[:, 5] = self.phase_grid

        grid_flux = self._interp_6d(points)

        if np.any(np.isnan(grid_flux)):
            return np.full_like(obs_phases, np.nan, dtype=float)

        phase_extended = np.concatenate([
            self.phase_grid - 1,
            self.phase_grid,
            self.phase_grid + 1
        ])
        flux_extended = np.concatenate([grid_flux, grid_flux, grid_flux])

        return np.interp(obs_phases, phase_extended, flux_extended)

    def evaluate_direct(
        self,
        d1: float, d2: float, r: float, R: float, i0: float,
        obs_phases: np.ndarray
    ) -> np.ndarray:
        """Evaluate the model by running ``simulate_lightcurve`` directly.

        Unlike :meth:`evaluate`, this bypasses the pre-computed grid and
        produces the exact model curve.  Use this for final best-fit
        evaluation and chi-square reporting.
        """
        flux_column = f"nfl_{self.band}_{self.wind_model}"
        try:
            results = simulate_lightcurve(
                r=r, R=R, d1=d1, d2=d2,
                gma0=self.sim_params.get('gma0', -90.0),
                i0=i0,
                dth=self.dth,
                d2h=self.sim_params.get('d2h', 6.0),
                dz=self.sim_params.get('dz', 0.1),
                flux_method="interpolate",
                flux_csv_path=self.flux_csv_path,
                lam=self.sim_params.get('lam', 0.589537),
                lam2=self.sim_params.get('lam2', 0.589537),
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
    
    WARNING: This is SLOW! Use PrecomputedModelGrid for MCMC.
    """
    
    def __init__(
        self,
        band: str,
        flux_csv_path: str,
        wind_model: str = 'av',
        dth: float = 5.0,
        flux_method: str = "interpolate",
        sim_params: Dict = None
    ):
        self.band = band.lower()
        self.flux_csv_path = flux_csv_path
        self.wind_model = wind_model.lower()
        self.dth = dth
        self.flux_method = flux_method
        self.flux_column = f"nfl_{self.band}_{self.wind_model}"
        self.sim_params = sim_params or {}
        
        if not os.path.exists(flux_csv_path):
            raise FileNotFoundError(f"Flux CSV not found: {flux_csv_path}")
    
    def evaluate(
        self,
        d1: float,
        d2: float,
        r: float,
        R: float,
        i0: float,
        obs_phases: np.ndarray
    ) -> np.ndarray:
        """Evaluate model (slow - calls full simulation)."""
        try:
            results = simulate_lightcurve(
                r=r, R=R, d1=d1, d2=d2,
                gma0=self.sim_params.get('gma0', -90.0),
                i0=i0,
                dth=self.dth,
                d2h=self.sim_params.get('d2h', 6.0),
                dz=self.sim_params.get('dz', 0.1),
                flux_method=self.flux_method,
                flux_csv_path=self.flux_csv_path,
                lam=self.sim_params.get('lam', 0.589537),
                lam2=self.sim_params.get('lam2', 0.589537),
                verbose=False
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
            model_phase_sorted + 1
        ])
        flux_extended = np.concatenate([
            model_flux_sorted,
            model_flux_sorted,
            model_flux_sorted
        ])
        
        return np.interp(obs_phases, phase_extended, flux_extended)


# =============================================================================
# Likelihood Functions
# =============================================================================

def _evaluate_model(theta, model, obs_phase):
    """Evaluate the physical model. Returns model_flux or None on failure."""
    d1, d2, r, R, i0 = theta[:5]
    try:
        model_flux = model.evaluate(d1, d2, r, R, i0, obs_phase)
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
) -> float:
    """Standard Gaussian log-likelihood (chi-squared)."""
    model_flux = _evaluate_model(theta, model, obs_phase)
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
) -> float:
    """Gaussian log-likelihood with a free fractional systematic error term.

    theta must have 6 elements: [d1, d2, r, R, i0, log_f].
    The effective variance per point is  sigma_obs^2 + (f * model)^2
    where f = exp(log_f).  The log(sigma2) normalisation is included
    so that inflating errors is properly penalised.
    """
    model_flux = _evaluate_model(theta, model, obs_phase)
    if model_flux is None:
        return -np.inf
    f = np.exp(theta[5])
    sigma2 = obs_err ** 2 + (f * model_flux) ** 2
    return -0.5 * np.sum((obs_flux - model_flux) ** 2 / sigma2 + np.log(sigma2))


def log_likelihood_studentt(
    theta: np.ndarray,
    model,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    nu: float = 5.0,
) -> float:
    """Student-t log-likelihood (heavier tails than Gaussian).

    ``nu`` controls tail weight; nu -> inf recovers the Gaussian.
    """
    model_flux = _evaluate_model(theta, model, obs_phase)
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
) -> float:
    """Log prior probability for physical + nuisance parameters."""
    d1, d2, r, R, i0 = theta[:5]

    for i, param in enumerate(PARAM_NAMES):
        if not (priors[param]['min'] < theta[i] < priors[param]['max']):
            return -np.inf

    if r >= R:
        return -np.inf

    log_p = 0.0
    for param, value in zip(PARAM_NAMES, theta[:5]):
        mean = priors[param]['mean']
        std = priors[param]['std']
        log_p += -0.5 * ((value - mean) / std) ** 2

    if likelihood == 'jitter':
        log_f = theta[5]
        if not (JITTER_PRIOR['min'] < log_f < JITTER_PRIOR['max']):
            return -np.inf
        log_p += -0.5 * ((log_f - JITTER_PRIOR['mean']) / JITTER_PRIOR['std']) ** 2

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
) -> float:
    """Log posterior = log prior + log likelihood."""
    lp = log_prior(theta, priors, likelihood=likelihood)
    if not np.isfinite(lp):
        return -np.inf

    if likelihood == 'jitter':
        ll = log_likelihood_jitter(theta, model, obs_phase, obs_flux, obs_err)
    elif likelihood == 'studentt':
        ll = log_likelihood_studentt(theta, model, obs_phase, obs_flux, obs_err, nu=studentt_nu)
    else:
        ll = log_likelihood_chi2(theta, model, obs_phase, obs_flux, obs_err)

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
        Number of threads for parallel MCMC (1 = serial)
    sampler_type : str
        'emcee' or 'zeus'
    likelihood : str
        'chi2', 'jitter', or 'studentt'
    studentt_nu : float
        Degrees-of-freedom for Student-t likelihood
        
    Returns
    -------
    sampler : emcee.EnsembleSampler or zeus.EnsembleSampler
    samples : np.ndarray
        Flattened chain samples (after burn-in)
    active_param_names : list of str
    active_param_labels : list of str
    """
    active_names, active_labels = get_param_config(likelihood)
    n_dim = len(active_names)

    # Initial positions for physical parameters
    initial = np.array([priors[p]['mean'] for p in PARAM_NAMES])
    scatter = np.array([priors[p]['std'] * 0.1 for p in PARAM_NAMES])
    if likelihood == 'jitter':
        initial = np.append(initial, JITTER_PRIOR['mean'])
        scatter = np.append(scatter, JITTER_PRIOR['std'] * 0.1)

    pos = initial + scatter * np.random.randn(n_walkers, n_dim)

    for i, param in enumerate(PARAM_NAMES):
        pos[:, i] = np.clip(pos[:, i],
                            priors[param]['min'] * 1.01,
                            priors[param]['max'] * 0.99)
    if likelihood == 'jitter':
        pos[:, 5] = np.clip(pos[:, 5],
                            JITTER_PRIOR['min'] * 0.99,
                            JITTER_PRIOR['max'] * 0.99)
    for j in range(n_walkers):
        if pos[j, 2] >= pos[j, 3]:
            pos[j, 2] = pos[j, 3] * 0.1

    parallel_info = f", {n_threads} threads" if n_threads > 1 else " (serial)"
    print(f"\nStarting MCMC ({sampler_type}) with {n_walkers} walkers, "
          f"{n_steps} steps{parallel_info}")
    print(f"Likelihood: {LIKELIHOOD_TYPES[likelihood]}")
    print(f"Initial parameter values (first walker): {pos[0]}")

    log_prob_args = (model, obs_phase, obs_flux, obs_err, priors, likelihood, studentt_nu)

    start_time = time.time()

    if sampler_type == 'zeus':
        if not HAS_ZEUS:
            raise ImportError("zeus not installed. Install with: pip install zeus-mcmc")
        sampler = zeus_sampler.EnsembleSampler(
            n_walkers, n_dim, log_probability, args=log_prob_args
        )
        sampler.run_mcmc(pos, n_steps, progress=progress)
    else:
        if n_threads > 1:
            from multiprocessing import Pool as MPPool
            with MPPool(n_threads) as pool:
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
        else:
            sampler = emcee.EnsembleSampler(
                n_walkers, n_dim, log_probability,
                args=log_prob_args,
            )
            if progress and HAS_TQDM:
                for _ in tqdm(sampler.sample(pos, iterations=n_steps),
                              total=n_steps, desc="MCMC Sampling"):
                    pass
            else:
                sampler.run_mcmc(pos, n_steps, progress=progress)

    elapsed = time.time() - start_time

    samples = sampler.get_chain(discard=n_burn, flat=True)

    print(f"\nMCMC completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Time per step: {elapsed/n_steps*1000:.1f} ms")
    print(f"Final chain shape: {samples.shape}")

    return sampler, samples, active_names, active_labels


# =============================================================================
# Output and Diagnostics
# =============================================================================

def compute_statistics(samples: np.ndarray, param_names: List[str] = None) -> Dict:
    """Compute summary statistics from MCMC samples."""
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

    return stats


def load_existing_results(
    output_dir: str,
    band: str,
    wind_model: str
) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Load existing MCMC results from saved files.
    
    Parameters
    ----------
    output_dir : str
        Directory containing MCMC output files
    band : str
        Energy band name
    wind_model : str
        Wind model name ('av' or 'cv')
        
    Returns
    -------
    samples : np.ndarray or None
        Loaded samples array, or None if file not found
    stats : Dict or None
        Computed statistics from samples, or None if samples not found
    """
    suffix = f"{band}_{wind_model}"
    samples_path = os.path.join(output_dir, f"{suffix}_samples.csv")
    
    if not os.path.exists(samples_path):
        print(f"Samples file not found: {samples_path}")
        return None, None
    
    print(f"Loading existing samples from: {samples_path}")
    samples_df = pd.read_csv(samples_path)
    
    # Verify columns match expected parameters
    if not all(p in samples_df.columns for p in PARAM_NAMES):
        print(f"Error: Samples file missing required columns. Expected: {PARAM_NAMES}")
        return None, None
    
    samples = samples_df[PARAM_NAMES].values
    stats = compute_statistics(samples)
    
    print(f"  Loaded {len(samples)} samples")
    
    return samples, stats


def compute_chi2_for_samples(
    model,
    samples: np.ndarray,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    output_path: str,
    n_samples: int = None,
    verbose: bool = True
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
    """
    import gzip
    
    n_total = len(samples)
    if n_samples is None or n_samples > n_total:
        n_samples = n_total
        sample_indices = np.arange(n_total)
    else:
        # Randomly select samples
        sample_indices = np.random.choice(n_total, size=n_samples, replace=False)
        sample_indices = np.sort(sample_indices)  # Keep order for reproducibility
    
    if verbose:
        print(f"Computing chi-square for {n_samples} samples...")
    
    dof = len(obs_flux) - len(PARAM_NAMES)
    
    # Prepare output data
    results = []
    
    if HAS_TQDM and verbose:
        iterator = tqdm(sample_indices, desc="Computing χ²")
    else:
        iterator = sample_indices
    
    for idx in iterator:
        sample_params = samples[idx]
        d1, d2, r, R, i0 = sample_params
        
        try:
            model_flux = model.evaluate(d1, d2, r, R, i0, obs_phase)
            
            if np.all(np.isfinite(model_flux)):
                chi2 = np.sum(((obs_flux - model_flux) / obs_err) ** 2)
                red_chi2 = chi2 / dof if dof > 0 else np.nan
            else:
                chi2 = np.nan
                red_chi2 = np.nan
        except Exception:
            chi2 = np.nan
            red_chi2 = np.nan
        
        results.append([idx, d1, d2, r, R, i0, chi2, red_chi2])
    
    # Create DataFrame
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


def print_results(stats: Dict, band: str, wind_model: str, param_names: List[str] = None):
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
    """
    if param_names is None:
        param_names = PARAM_NAMES
    phys_names = PARAM_NAMES

    best_params = [stats[p]['median'] for p in phys_names]

    # Use direct simulation (exact) instead of grid interpolation.
    eval_fn = getattr(model, 'evaluate_direct', model.evaluate)

    model_phases = np.linspace(0, 1, 360)
    model_flux = eval_fn(*best_params, model_phases)

    obs_model = eval_fn(*best_params, obs_phase)
    chi2 = np.sum(((obs_flux - obs_model) / obs_err) ** 2)
    dof = len(obs_flux) - len(phys_names)
    red_chi2 = chi2 / dof if dof > 0 else np.nan

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(obs_phase, obs_flux, yerr=obs_err, fmt='o',
                markersize=4, alpha=0.7, label='Observed (phase-binned)',
                capsize=2, elinewidth=1, color='C0', zorder=5)

    ax.plot(model_phases, model_flux, 'r-', lw=2,
            label=f'Best-fit model ({WIND_MODELS[wind_model]})', zorder=10)

    ax.set_xlabel('Orbital Phase', fontsize=12)
    ax.set_ylabel('Flux (erg/cm²/s)', fontsize=12)
    ax.set_title(f'{band.upper()} Band - {wind_model.upper()} Wind '
                 f'(χ²/dof = {red_chi2:.2f})', fontsize=14)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    param_text = '\n'.join([
        f"{p}: {stats[p]['median']:.4f} +/- {(stats[p]['lower']+stats[p]['upper'])/2:.4f}"
        for p in param_names
    ])
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
) -> np.ndarray:
    """Compute per-observation log-likelihood for a subset of posterior samples.

    Returns array of shape (n_samples_used, n_obs).
    """
    n_total = len(samples)
    n_use = min(n_samples, n_total)
    indices = np.random.choice(n_total, size=n_use, replace=False)
    n_obs = len(obs_phase)
    log_lik = np.full((n_use, n_obs), np.nan)

    for k, idx in enumerate(indices):
        theta = samples[idx]
        model_flux = _evaluate_model(theta, model, obs_phase)
        if model_flux is None:
            continue

        if likelihood == 'jitter':
            f = np.exp(theta[5])
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
    sim_params: Dict = None
) -> Dict:
    """Run MCMC fit for a single band/wind_model combination."""
    
    if priors is None:
        priors = DEFAULT_PRIORS
    if sim_params is None:
        sim_params = {}
    
    print(f"\n{'#'*60}")
    print(f"# Fitting {band.upper()} band - {WIND_MODELS[wind_model]}")
    print('#'*60)
    
    if sim_params:
        print(f"# Simulation params: {sim_params}")
    
    # Initialize or reuse model
    if model_grid is not None:
        # Reuse pre-computed grid, just switch wind model if needed
        if model_grid.wind_model != wind_model:
            model_grid.switch_wind_model(wind_model)
        model = model_grid
    elif args.no_grid:
        print("\n⚠️  Using direct model evaluation (SLOW!)")
        model = DirectLightCurveModel(
            band=band,
            flux_csv_path=args.flux_csv,
            wind_model=wind_model,
            dth=args.dth,
            sim_params=sim_params
        )
    else:
        # Pre-compute model grid
        grid_points = {
            'd1': args.grid_points,
            'd2': args.grid_points,
            'r': max(5, args.grid_points // 2),
            'R': args.grid_points,
            'i0': args.grid_points + 2
        }
        
        model = PrecomputedModelGrid(
            band=band,
            flux_csv_path=args.flux_csv,
            wind_model=wind_model,
            priors=priors,
            grid_points=grid_points,
            dth=args.dth,
            n_workers=args.n_workers,
            verbose=True,
            load_path=args.load_grid,
            sim_params=sim_params
        )
        
        # Save grid if requested
        if args.save_grid:
            model.save(args.save_grid)
    
    # Resolve sampler / likelihood from CLI args
    sampler_type = getattr(args, 'sampler', 'emcee')
    likelihood = getattr(args, 'likelihood', 'chi2')
    studentt_nu = getattr(args, 'studentt_nu', 5.0)

    # Run MCMC
    sampler, samples, active_names, active_labels = run_mcmc(
        model=model,
        obs_phase=obs_phase,
        obs_flux=obs_flux,
        obs_err=obs_err,
        n_walkers=args.n_walkers,
        n_steps=args.n_steps,
        n_burn=args.n_burn,
        priors=priors,
        progress=not args.quiet,
        n_threads=args.n_threads,
        sampler_type=sampler_type,
        likelihood=likelihood,
        studentt_nu=studentt_nu,
    )

    # Compute statistics
    stats = compute_statistics(samples, param_names=active_names)
    print_results(stats, band, wind_model, param_names=active_names)
    print_diagnostics(sampler, sampler_type=sampler_type, param_names=active_names)

    # ArviZ diagnostics
    compute_waic = getattr(args, 'compute_waic', False)
    run_arviz_diagnostics(
        sampler, active_names, args.n_burn,
        model=model, obs_phase=obs_phase, obs_flux=obs_flux, obs_err=obs_err,
        likelihood=likelihood, studentt_nu=studentt_nu,
        compute_waic=compute_waic,
        output_dir=args.output_dir,
        suffix=f"{band}_{wind_model}",
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
        )
        stats['reduced_chi2'] = red_chi2

    # Save samples with log-probability (flat CSV)
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

    # Save full chain (walkers preserved) for post-hoc diagnostics / WAIC
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
        )
        print(f"Full chain saved to: {chain_path}")
    except Exception as e:
        warnings.warn(f"Could not save full chain: {e}")

    # Optionally save chi-square for all samples
    if getattr(args, 'save_chi2', False):
        chi2_path = os.path.join(args.output_dir, f"{suffix}_chi2.csv.gz")
        compute_chi2_for_samples(
            model, samples, obs_phase, obs_flux, obs_err,
            output_path=chi2_path,
            n_samples=getattr(args, 'chi2_n_samples', None),
            verbose=True
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
    sim_params: Dict = None
) -> Optional[Dict]:
    """
    Regenerate plots from existing MCMC results without re-running MCMC.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    band : str
        Energy band name
    wind_model : str
        Wind model name
    obs_phase, obs_flux, obs_err : np.ndarray
        Observed data
    priors : Dict, optional
        Prior specifications
    sim_params : Dict, optional
        Simulation parameters
        
    Returns
    -------
    stats : Dict or None
        Statistics from loaded samples, or None if loading failed
    """
    if priors is None:
        priors = DEFAULT_PRIORS
    if sim_params is None:
        sim_params = {}
    
    print(f"\n{'#'*60}")
    print(f"# Replotting {band.upper()} band - {WIND_MODELS[wind_model]}")
    print('#'*60)
    
    # Load existing samples
    samples, stats = load_existing_results(args.output_dir, band, wind_model)
    
    if samples is None or stats is None:
        print(f"Could not load existing results for {band}_{wind_model}")
        return None
    
    print_results(stats, band, wind_model)
    
    # Initialize model grid for plotting
    print("\nInitializing model grid for plotting...")
    grid_points = {
        'd1': args.grid_points,
        'd2': args.grid_points,
        'r': max(5, args.grid_points // 2),
        'R': args.grid_points,
        'i0': args.grid_points + 2
    }
    
    model = PrecomputedModelGrid(
        band=band,
        flux_csv_path=args.flux_csv,
        wind_model=wind_model,
        priors=priors,
        grid_points=grid_points,
        dth=args.dth,
        n_workers=args.n_workers,
        verbose=True,
        load_path=args.load_grid,
        sim_params=sim_params
    )
    
    suffix = f"{band}_{wind_model}"

    # Generate plots
    if not args.no_plots:
        plot_corner(
            samples, band, wind_model,
            os.path.join(args.output_dir, f"{suffix}_corner.png")
        )

        red_chi2 = plot_best_fit(
            model, obs_phase, obs_flux, obs_err, stats, band, wind_model,
            os.path.join(args.output_dir, f"{suffix}_bestfit.png"),
        )
        stats['reduced_chi2'] = red_chi2

    # Post-hoc ArviZ diagnostics / WAIC from saved chain
    compute_waic = getattr(args, 'compute_waic', False)
    if HAS_ARVIZ or compute_waic:
        chain_path = os.path.join(args.output_dir, f"{suffix}_chain.npz")
        if os.path.exists(chain_path):
            print(f"\nLoading saved chain from: {chain_path}")
            chain_data = np.load(chain_path, allow_pickle=True)
            saved_chain = chain_data['chain']
            saved_names = list(chain_data['param_names'])
            saved_likelihood = str(chain_data.get('likelihood', 'chi2'))
            saved_nu = float(chain_data.get('studentt_nu', 5.0))
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
            )
        elif compute_waic:
            print(f"Chain file not found: {chain_path}")
            print("  Re-run MCMC to generate it, or skip --compute-waic.")

    # Optionally compute and save chi-square for all samples
    if getattr(args, 'save_chi2', False):
        chi2_path = os.path.join(args.output_dir, f"{suffix}_chi2.csv.gz")
        compute_chi2_for_samples(
            model, samples, obs_phase, obs_flux, obs_err,
            output_path=chi2_path,
            n_samples=getattr(args, 'chi2_n_samples', None),
            verbose=True
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
        choices=['av', 'cv', 'both'],
        default='both',
        help="Wind model: 'av' (accelerated velocity), 'cv' (constant velocity), or 'both'"
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
             "(e.g. FLUX -> FLUX_ERR, rate -> rate_err)"
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
        help="Use raw 100s binned data without phase binning"
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
        help="Number of threads for parallel MCMC sampling (1 = serial). "
             "Note: With pre-computed grid, parallelization may not help much."
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
        help="Number of grid points per parameter for pre-computed grid"
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
        help="Scaling parameter to convert column density to nH (in 1e22 cm^-2 units). "
             "Controls the absolute flux normalization."
    )
    sim_group.add_argument(
        "--lam2",
        type=float,
        default=None,
        help="Scaling parameter for constant velocity wind model. "
             "Defaults to same as --lam if not specified."
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
    
    args = parser.parse_args()
    
    # Build simulation parameters dict
    sim_params = {
        'lam': args.lam,
        'lam2': args.lam2 if args.lam2 is not None else args.lam,
        'gma0': args.gma0,
        'd2h': args.d2h,
        'dz': args.dz
    }
    
    # Build custom priors
    priors = DEFAULT_PRIORS.copy()
    for param in PARAM_NAMES:
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
                    'max': parts[3]
                }
                print(f"Custom prior for {param}: mean={parts[0]}, std={parts[1]}, min={parts[2]}, max={parts[3]}")
            except Exception as e:
                parser.error(f"Invalid format for --prior-{param}: {e}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which bands to fit
    if args.band == 'all':
        bands = ['broad', 'soft', 'medium', 'hard']
    else:
        bands = [args.band]
    
    # Determine which wind models to fit
    if args.wind_model == 'both':
        wind_models = ['av', 'cv']
    else:
        wind_models = [args.wind_model]
    
    # Run MCMC or replot from existing results
    all_results = {}
    
    for band in bands:
        # Load data once per band
        try:
            obs_df = load_observed_lightcurves(
                band, args.data_dir,
                flux_column=args.obs_column,
                error_column=args.obs_error_column or args.obs_column + "_ERR",
                time_column=args.time_column,
            )
            
            if not args.no_phase_bin:
                obs_df = phase_bin_data(obs_df, n_bins=args.n_phase_bins)
            
            obs_phase = obs_df['phase'].values
            obs_flux = obs_df['flux'].values
            obs_err = obs_df['flux_err'].values
            
            # Replace invalid errors with flux-based estimate
            invalid_err = ~np.isfinite(obs_err) | (obs_err <= 0)
            if np.any(invalid_err):
                obs_err[invalid_err] = np.abs(obs_flux[invalid_err]) * 0.1
                warnings.warn(f"Replaced {np.sum(invalid_err)} invalid errors with 10% of flux")
            
            # For multiple wind models, reuse the same grid
            model_grid = None
            
            for wind_model in wind_models:
                try:
                    key = f"{band}_{wind_model}"
                    
                    if args.replot:
                        # Regenerate plots from existing results
                        stats = replot_from_existing(
                            args, band, wind_model,
                            obs_phase, obs_flux, obs_err,
                            priors=priors,
                            sim_params=sim_params
                        )
                        if stats is not None:
                            all_results[key] = stats
                    else:
                        # Run full MCMC fitting
                        stats, model_grid = run_single_fit(
                            band, wind_model, args,
                            obs_phase, obs_flux, obs_err,
                            model_grid=model_grid if len(wind_models) > 1 else None,
                            priors=priors,
                            sim_params=sim_params
                        )
                        all_results[key] = stats
                        
                        # Save grid after first wind model if requested
                        if args.save_grid and wind_model == wind_models[0]:
                            model_grid.save(args.save_grid)
                        
                except Exception as e:
                    print(f"ERROR {'replotting' if args.replot else 'fitting'} {band} band ({wind_model}): {e}")
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
            active_names, _ = get_param_config(likelihood)
            for key, stats in all_results.items():
                band, wind_model = key.rsplit('_', 1)
                f.write(f"{band.upper()} Band - {WIND_MODELS[wind_model]}\n")
                f.write("-"*40 + "\n")
                for param in active_names:
                    if param in stats:
                        s = stats[param]
                        f.write(f"{param}: {s['median']:.6f} (+{s['upper']:.6f}/-{s['lower']:.6f})\n")
                if 'reduced_chi2' in stats:
                    f.write(f"Reduced chi-square: {stats['reduced_chi2']:.3f}\n")
                f.write("\n")
        
        print(f"\nSummary saved to: {summary_path}")
    
    if args.replot:
        print("\nReplotting complete!")
    else:
        print("\nMCMC fitting complete!")


if __name__ == "__main__":
    main()
