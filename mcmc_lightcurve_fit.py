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
    
    # Save grid for reuse
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv --save-grid grids/broad_grid.npz
    
    # Load pre-computed grid (skip grid computation)
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv --load-grid grids/broad_grid.npz
    
    # Fit both wind models using same grid
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv --wind-model both --load-grid grids/broad_grid.npz
    
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

from scipy.interpolate import RegularGridInterpolator

from xrb_lightcurve import simulate_lightcurve
from chandra_phase_analysis import phase_bin_data as _phase_bin_data_base


# =============================================================================
# Constants (from chandra_phase_analysis.py)
# =============================================================================
REF_EPOCH: float = 278801348  # Reference time (t0) for phase zero (seconds)
ORBITAL_PERIOD: float = 125431  # Orbital period in seconds (~34.8 hours)

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


# =============================================================================
# Data Loading and Phase Binning
# =============================================================================

def frac(x: np.ndarray) -> np.ndarray:
    """Return the fractional part of x (vectorized)."""
    return np.abs(x - np.floor(x))


def load_observed_lightcurves(
    band: str,
    data_dir: str = "data/IC_10_X1_LC",
    flux_column: str = "FLUX",
    error_column: str = "FLUX_ERR"
) -> pd.DataFrame:
    """
    Load all observed light curve files for a given energy band.
    
    Parameters
    ----------
    band : str
        Energy band: 'broad', 'soft', or 'hard'
    data_dir : str
        Base directory containing IC_10_X1_LC subdirectories
    flux_column : str
        Column name for flux values
    error_column : str
        Column name for flux errors
        
    Returns
    -------
    DataFrame with columns: time, flux, flux_err, phase, obs_id
    """
    band_dir = os.path.join(data_dir, f"{band.capitalize()}_with_flux")
    
    if not os.path.isdir(band_dir):
        raise FileNotFoundError(f"Band directory not found: {band_dir}")
    
    txt_files = sorted(glob.glob(os.path.join(band_dir, "*.txt")))
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {band_dir}")
    
    all_data = []
    
    for filepath in txt_files:
        obs_id = os.path.basename(filepath).split('_')[0]
        
        # Read file with header detection
        try:
            with open(filepath, 'r') as f:
                header_line = None
                for line in f:
                    if line.strip().startswith('#'):
                        # Check for column header line
                        if 'TIME' in line.upper() and 'FLUX' in line.upper():
                            header_line = line.strip().lstrip('#').strip()
                    else:
                        break
            
            # Read data
            df = pd.read_csv(filepath, delim_whitespace=True, comment='#', header=None)
            
            if header_line:
                col_names = header_line.split()
                if len(col_names) == len(df.columns):
                    df.columns = col_names
            
            # Extract required columns
            time_col = None
            for col in df.columns:
                if str(col).upper() == 'TIME':
                    time_col = col
                    break
            
            if time_col is None:
                # Assume first column is time
                time_col = df.columns[0]
            
            # Get flux and error columns
            if flux_column in df.columns:
                flux_col = flux_column
            else:
                # Try case-insensitive match
                flux_col = None
                for col in df.columns:
                    if str(col).upper() == flux_column.upper():
                        flux_col = col
                        break
                if flux_col is None:
                    continue
            
            if error_column in df.columns:
                err_col = error_column
            else:
                err_col = None
                for col in df.columns:
                    if str(col).upper() == error_column.upper():
                        err_col = col
                        break
            
            obs_df = pd.DataFrame({
                'time': df[time_col].astype(float),
                'flux': df[flux_col].astype(float),
                'flux_err': df[err_col].astype(float) if err_col else np.nan,
                'obs_id': obs_id
            })
            
            # Filter out invalid data
            obs_df = obs_df[
                (obs_df['flux'] > 0) & 
                np.isfinite(obs_df['flux']) &
                np.isfinite(obs_df['time'])
            ]
            
            all_data.append(obs_df)
            
        except Exception as e:
            warnings.warn(f"Failed to read {filepath}: {e}")
            continue
    
    if not all_data:
        raise ValueError(f"No valid data loaded for band '{band}'")
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Convert time to orbital phase
    combined['phase'] = frac((combined['time'] - REF_EPOCH) / ORBITAL_PERIOD)
    
    print(f"Loaded {len(combined)} data points from {len(txt_files)} files for {band} band")
    
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
            Energy band ('broad', 'soft', 'hard')
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
        """Setup interpolators for the current wind model."""
        # Select which grid to use based on wind model
        if self.wind_model == 'av':
            self.flux_grid = self.flux_grid_av
        else:
            self.flux_grid = self.flux_grid_cv
        
        # Create interpolator for each phase point
        self.interpolators = []
        
        d1_grid = self.param_grids['d1']
        d2_grid = self.param_grids['d2']
        r_grid = self.param_grids['r']
        R_grid = self.param_grids['R']
        i0_grid = self.param_grids['i0']
        
        for i_phase in range(len(self.phase_grid)):
            flux_slice = self.flux_grid[:, :, :, :, :, i_phase].copy()
            
            # Replace NaN with mean for interpolation stability
            if np.any(np.isnan(flux_slice)):
                flux_slice = np.where(np.isnan(flux_slice), np.nanmean(flux_slice), flux_slice)
            
            interp = RegularGridInterpolator(
                (d1_grid, d2_grid, r_grid, R_grid, i0_grid),
                flux_slice,
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            self.interpolators.append(interp)
    
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
        # Get flux at each grid phase point
        point = np.array([d1, d2, r, R, i0])
        
        grid_flux = np.array([interp(point)[0] for interp in self.interpolators])
        
        if np.any(np.isnan(grid_flux)):
            return np.full_like(obs_phases, np.nan, dtype=float)
        
        # Interpolate to requested phases (with wrap-around)
        phase_extended = np.concatenate([
            self.phase_grid - 1,
            self.phase_grid,
            self.phase_grid + 1
        ])
        flux_extended = np.concatenate([grid_flux, grid_flux, grid_flux])
        
        return np.interp(obs_phases, phase_extended, flux_extended)


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
# MCMC Sampler
# =============================================================================

def log_prior(theta: np.ndarray, priors: Dict = DEFAULT_PRIORS) -> float:
    """Log prior probability for parameters."""
    d1, d2, r, R, i0 = theta
    
    # Check hard bounds
    if not (priors['d1']['min'] < d1 < priors['d1']['max']):
        return -np.inf
    if not (priors['d2']['min'] < d2 < priors['d2']['max']):
        return -np.inf
    if not (priors['r']['min'] < r < priors['r']['max']):
        return -np.inf
    if not (priors['R']['min'] < R < priors['R']['max']):
        return -np.inf
    if not (priors['i0']['min'] < i0 < priors['i0']['max']):
        return -np.inf
    
    # Physical constraint: compact object must be inside the orbit
    if r >= R:
        return -np.inf
    
    # Gaussian priors (log probability)
    log_p = 0.0
    for param, value in zip(PARAM_NAMES, theta):
        mean = priors[param]['mean']
        std = priors[param]['std']
        log_p += -0.5 * ((value - mean) / std) ** 2
    
    return log_p


def log_likelihood(
    theta: np.ndarray,
    model,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray
) -> float:
    """Log likelihood (chi-square statistic)."""
    d1, d2, r, R, i0 = theta
    
    try:
        model_flux = model.evaluate(d1, d2, r, R, i0, obs_phase)
    except Exception:
        return -np.inf
    
    if np.any(~np.isfinite(model_flux)):
        return -np.inf
    
    # Chi-square
    chi2 = np.sum(((obs_flux - model_flux) / obs_err) ** 2)
    
    return -0.5 * chi2


def log_probability(
    theta: np.ndarray,
    model,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: np.ndarray,
    priors: Dict = DEFAULT_PRIORS
) -> float:
    """Log posterior probability = log prior + log likelihood."""
    lp = log_prior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = log_likelihood(theta, model, obs_phase, obs_flux, obs_err)
    
    return lp + ll


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
    n_threads: int = 1
) -> Tuple[emcee.EnsembleSampler, np.ndarray]:
    """
    Run MCMC sampling.
    
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
        Note: With pre-computed grid, parallelization may not help much
        
    Returns
    -------
    sampler : emcee.EnsembleSampler
        The MCMC sampler object
    samples : np.ndarray
        Flattened chain samples (after burn-in)
    """
    n_dim = len(PARAM_NAMES)
    
    # Initialize walkers around prior means with small scatter
    initial = np.array([priors[p]['mean'] for p in PARAM_NAMES])
    scatter = np.array([priors[p]['std'] * 0.1 for p in PARAM_NAMES])
    
    # Generate initial positions
    pos = initial + scatter * np.random.randn(n_walkers, n_dim)
    
    # Ensure initial positions are within bounds
    for i, param in enumerate(PARAM_NAMES):
        pos[:, i] = np.clip(pos[:, i], 
                           priors[param]['min'] * 1.01, 
                           priors[param]['max'] * 0.99)
    
    # Ensure r < R for all walkers
    for j in range(n_walkers):
        if pos[j, 2] >= pos[j, 3]:  # r >= R
            pos[j, 2] = pos[j, 3] * 0.1
    
    parallel_info = f", {n_threads} threads" if n_threads > 1 else " (serial)"
    print(f"\nStarting MCMC with {n_walkers} walkers, {n_steps} steps{parallel_info}...")
    print(f"Initial parameter values (first walker): {pos[0]}")
    
    # Create sampler (optionally with multiprocessing)
    if n_threads > 1:
        from multiprocessing import Pool as MPPool
        with MPPool(n_threads) as pool:
            sampler = emcee.EnsembleSampler(
                n_walkers, n_dim, log_probability,
                args=(model, obs_phase, obs_flux, obs_err, priors),
                pool=pool
            )
            
            # Run MCMC
            start_time = time.time()
            if progress and HAS_TQDM:
                for _ in tqdm(sampler.sample(pos, iterations=n_steps), 
                              total=n_steps, desc="MCMC Sampling"):
                    pass
            else:
                sampler.run_mcmc(pos, n_steps, progress=progress)
            elapsed = time.time() - start_time
    else:
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, log_probability,
            args=(model, obs_phase, obs_flux, obs_err, priors)
        )
        
        # Run MCMC with timing and progress bar
        start_time = time.time()
        
        if progress and HAS_TQDM:
            for _ in tqdm(sampler.sample(pos, iterations=n_steps), 
                          total=n_steps, desc="MCMC Sampling"):
                pass
        else:
            sampler.run_mcmc(pos, n_steps, progress=progress)
        
        elapsed = time.time() - start_time
    
    # Get samples after burn-in
    samples = sampler.get_chain(discard=n_burn, flat=True)
    
    print(f"\nMCMC completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Time per step: {elapsed/n_steps*1000:.1f} ms")
    print(f"Final chain shape: {samples.shape}")
    
    return sampler, samples


# =============================================================================
# Output and Diagnostics
# =============================================================================

def compute_statistics(samples: np.ndarray) -> Dict:
    """Compute summary statistics from MCMC samples."""
    stats = {}
    
    for i, param in enumerate(PARAM_NAMES):
        param_samples = samples[:, i]
        
        # Percentiles
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


def print_results(stats: Dict, band: str, wind_model: str):
    """Print formatted results table."""
    print(f"\n{'='*60}")
    print(f"MCMC Results for {band.upper()} band - {WIND_MODELS[wind_model]}")
    print('='*60)
    print(f"{'Parameter':<15} {'Median':<12} {'Lower σ':<12} {'Upper σ':<12}")
    print('-'*60)
    
    for param in PARAM_NAMES:
        s = stats[param]
        print(f"{param:<15} {s['median']:<12.6f} {s['lower']:<12.6f} {s['upper']:<12.6f}")
    
    print('='*60)


def plot_corner(samples: np.ndarray, band: str, wind_model: str, output_path: str):
    """Generate corner plot of posterior distributions."""
    if corner is None:
        warnings.warn("corner package not installed, skipping corner plot")
        return
    
    fig = corner.corner(
        samples,
        labels=PARAM_LABELS,
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


def plot_trace(sampler: emcee.EnsembleSampler, band: str, wind_model: str, output_path: str):
    """Generate trace plots for convergence diagnostics."""
    chain = sampler.get_chain()
    n_steps, n_walkers, n_dim = chain.shape
    
    fig, axes = plt.subplots(n_dim, 1, figsize=(10, 2*n_dim), sharex=True)
    
    for i, ax in enumerate(axes):
        for j in range(n_walkers):
            ax.plot(chain[:, j, i], alpha=0.3, lw=0.5)
        ax.set_ylabel(PARAM_LABELS[i])
        ax.axvline(x=sampler.iteration // 5, color='r', linestyle='--', 
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
    samples: np.ndarray = None,
    n_samples_for_ci: int = 200,
    ci_percentiles: Tuple[float, float] = (16, 84),
    ci_style: str = 'band',
    ci_method: str = 'bounds'
):
    """
    Plot observed data with best-fit model overlay and confidence intervals.
    
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
    samples : np.ndarray, optional
        MCMC samples array (n_samples, n_params). If provided and ci_method='samples',
        confidence intervals are computed from these samples.
    n_samples_for_ci : int
        Number of random samples to use for computing confidence intervals.
        Default is 200 for balance between accuracy and speed.
    ci_percentiles : tuple
        Lower and upper percentiles for confidence interval (default: 16, 84 = 1-sigma)
    ci_style : str
        Style for displaying confidence interval:
        - 'band': Shaded region between percentiles (default)
        - 'lines': Dashed lines at percentile bounds
        - 'both': Both shaded band and dashed lines
    ci_method : str
        Method for computing confidence intervals:
        - 'bounds': Use +/- parameter ranges from MCMC summary (faster, default)
        - 'samples': Draw from full posterior samples (more accurate for correlated params)
    """
    # Get best-fit parameters
    best_params = [stats[p]['median'] for p in PARAM_NAMES]
    
    # Generate model curve at fine phase resolution
    model_phases = np.linspace(0, 1, 360)
    model_flux = model.evaluate(*best_params, model_phases)
    
    # Compute chi-square
    obs_model = model.evaluate(*best_params, obs_phase)
    chi2 = np.sum(((obs_flux - obs_model) / obs_err) ** 2)
    dof = len(obs_flux) - len(PARAM_NAMES)
    red_chi2 = chi2 / dof if dof > 0 else np.nan
    
    # Compute confidence intervals
    ci_lower = None
    ci_upper = None
    
    if ci_method == 'bounds':
        # Use +/- parameter bounds directly from MCMC summary
        print("Computing confidence intervals from parameter bounds...")
        
        # Lower bound: median - lower_sigma for each parameter
        lower_params = [stats[p]['median'] - stats[p]['lower'] for p in PARAM_NAMES]
        # Upper bound: median + upper_sigma for each parameter
        upper_params = [stats[p]['median'] + stats[p]['upper'] for p in PARAM_NAMES]
        
        ci_lower_flux = model.evaluate(*lower_params, model_phases)
        ci_upper_flux = model.evaluate(*upper_params, model_phases)
        
        # Take the min/max at each phase to form the envelope
        if np.all(np.isfinite(ci_lower_flux)) and np.all(np.isfinite(ci_upper_flux)):
            ci_lower = np.minimum(ci_lower_flux, ci_upper_flux)
            ci_upper = np.maximum(ci_lower_flux, ci_upper_flux)
            print("  Using parameter bounds for confidence envelope")
        else:
            print("  Warning: Could not evaluate bounds, skipping CI")
    
    elif ci_method == 'samples' and samples is not None and len(samples) > 0:
        print(f"Computing confidence intervals from {min(n_samples_for_ci, len(samples))} posterior samples...")
        
        # Randomly select samples for CI computation
        n_use = min(n_samples_for_ci, len(samples))
        sample_indices = np.random.choice(len(samples), size=n_use, replace=False)
        
        # Compute model flux for each sample
        all_model_fluxes = []
        for idx in sample_indices:
            sample_params = samples[idx]
            sample_flux = model.evaluate(*sample_params, model_phases)
            if np.all(np.isfinite(sample_flux)):
                all_model_fluxes.append(sample_flux)
        
        if len(all_model_fluxes) > 10:  # Need enough valid samples
            all_model_fluxes = np.array(all_model_fluxes)
            ci_lower = np.percentile(all_model_fluxes, ci_percentiles[0], axis=0)
            ci_upper = np.percentile(all_model_fluxes, ci_percentiles[1], axis=0)
            print(f"  Used {len(all_model_fluxes)} valid samples for confidence bands")
        else:
            print(f"  Warning: Only {len(all_model_fluxes)} valid samples, skipping CI")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot confidence interval first (so it's behind other elements)
    if ci_lower is not None and ci_upper is not None:
        if ci_style in ['band', 'both']:
            ax.fill_between(
                model_phases, ci_lower, ci_upper,
                alpha=0.3, color='red',
                label=f'{ci_percentiles[0]:.0f}-{ci_percentiles[1]:.0f}% CI'
            )
        if ci_style in ['lines', 'both']:
            ax.plot(model_phases, ci_lower, 'r--', lw=1, alpha=0.7,
                    label=f'{ci_percentiles[0]:.0f}% bound' if ci_style == 'lines' else None)
            ax.plot(model_phases, ci_upper, 'r--', lw=1, alpha=0.7,
                    label=f'{ci_percentiles[1]:.0f}% bound' if ci_style == 'lines' else None)
    
    # Observed data with error bars
    ax.errorbar(obs_phase, obs_flux, yerr=obs_err, fmt='o', 
                markersize=4, alpha=0.7, label='Observed (phase-binned)',
                capsize=2, elinewidth=1, color='C0', zorder=5)
    
    # Best-fit model (solid line on top)
    ax.plot(model_phases, model_flux, 'r-', lw=2, 
            label=f'Best-fit model ({WIND_MODELS[wind_model]})', zorder=10)
    
    ax.set_xlabel('Orbital Phase', fontsize=12)
    ax.set_ylabel('Flux (erg/cm²/s)', fontsize=12)
    ax.set_title(f'{band.upper()} Band - {wind_model.upper()} Wind (χ²/dof = {red_chi2:.2f})', 
                 fontsize=14)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    # Add parameter annotation
    param_text = '\n'.join([
        f"{p}: {stats[p]['median']:.4f} ± {(stats[p]['lower']+stats[p]['upper'])/2:.4f}"
        for p in PARAM_NAMES
    ])
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Best-fit plot saved to: {output_path}")
    
    return red_chi2


def print_diagnostics(sampler: emcee.EnsembleSampler):
    """Print MCMC diagnostics."""
    print("\n" + "="*60)
    print("MCMC Diagnostics")
    print("="*60)
    
    # Acceptance fraction
    acc_frac = np.mean(sampler.acceptance_fraction)
    print(f"Mean acceptance fraction: {acc_frac:.3f}")
    if acc_frac < 0.2:
        print("  ⚠️  Low acceptance fraction - consider adjusting priors")
    elif acc_frac > 0.5:
        print("  ⚠️  High acceptance fraction - chain may not be mixing well")
    else:
        print("  ✓ Acceptance fraction in optimal range (0.2-0.5)")
    
    # Autocorrelation time
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"\nAutocorrelation times:")
        for i, param in enumerate(PARAM_NAMES):
            print(f"  {param}: {tau[i]:.1f} steps")
        
        n_steps = sampler.iteration
        n_independent = n_steps / np.max(tau)
        print(f"\nEffective independent samples: ~{int(n_independent * sampler.nwalkers)}")
        
        if n_steps < 50 * np.max(tau):
            print("  ⚠️  Chain may not be converged. Consider running longer.")
        else:
            print("  ✓ Chain appears well-converged")
    except Exception:
        print("\nAutocorrelation time: Could not compute (chain too short)")
    
    print("="*60)


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
    
    # Run MCMC
    sampler, samples = run_mcmc(
        model=model,
        obs_phase=obs_phase,
        obs_flux=obs_flux,
        obs_err=obs_err,
        n_walkers=args.n_walkers,
        n_steps=args.n_steps,
        n_burn=args.n_burn,
        priors=priors,
        progress=not args.quiet,
        n_threads=args.n_threads
    )
    
    # Compute statistics
    stats = compute_statistics(samples)
    print_results(stats, band, wind_model)
    print_diagnostics(sampler)
    
    # Generate file suffix
    suffix = f"{band}_{wind_model}"
    
    # Generate plots
    if not args.no_plots:
        plot_corner(
            samples, band, wind_model,
            os.path.join(args.output_dir, f"{suffix}_corner.png")
        )
        plot_trace(
            sampler, band, wind_model,
            os.path.join(args.output_dir, f"{suffix}_trace.png")
        )
        # Parse CI percentiles
        ci_percentiles = (16, 84)  # default
        if hasattr(args, 'ci_percentiles') and args.ci_percentiles:
            try:
                parts = [float(x.strip()) for x in args.ci_percentiles.split(',')]
                if len(parts) == 2:
                    ci_percentiles = tuple(parts)
            except (ValueError, AttributeError):
                pass
        
        red_chi2 = plot_best_fit(
            model, obs_phase, obs_flux, obs_err, stats, band, wind_model,
            os.path.join(args.output_dir, f"{suffix}_bestfit.png"),
            samples=samples,
            n_samples_for_ci=getattr(args, 'n_samples_ci', 200),
            ci_percentiles=ci_percentiles,
            ci_style=getattr(args, 'ci_style', 'band'),
            ci_method=getattr(args, 'ci_method', 'bounds')
        )
        stats['reduced_chi2'] = red_chi2
    
    # Save samples
    samples_df = pd.DataFrame(samples, columns=PARAM_NAMES)
    samples_df.to_csv(
        os.path.join(args.output_dir, f"{suffix}_samples.csv"),
        index=False
    )
    print(f"Samples saved to: {args.output_dir}/{suffix}_samples.csv")
    
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
    
    # Parse CI percentiles
    ci_percentiles = (16, 84)  # default
    if hasattr(args, 'ci_percentiles') and args.ci_percentiles:
        try:
            parts = [float(x.strip()) for x in args.ci_percentiles.split(',')]
            if len(parts) == 2:
                ci_percentiles = tuple(parts)
        except (ValueError, AttributeError):
            pass
    
    # Generate plots
    if not args.no_plots:
        # Corner plot
        plot_corner(
            samples, band, wind_model,
            os.path.join(args.output_dir, f"{suffix}_corner.png")
        )
        
        # Best-fit plot with confidence intervals
        red_chi2 = plot_best_fit(
            model, obs_phase, obs_flux, obs_err, stats, band, wind_model,
            os.path.join(args.output_dir, f"{suffix}_bestfit.png"),
            samples=samples,
            n_samples_for_ci=getattr(args, 'n_samples_ci', 200),
            ci_percentiles=ci_percentiles,
            ci_style=getattr(args, 'ci_style', 'band'),
            ci_method=getattr(args, 'ci_method', 'bounds')
        )
        stats['reduced_chi2'] = red_chi2
    
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
        choices=['broad', 'soft', 'hard', 'all'],
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
        help="Directory containing light curve data"
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
    
    # Confidence interval options for best-fit plot
    parser.add_argument(
        "--n-samples-ci",
        type=int,
        default=200,
        help="Number of posterior samples to use for computing confidence intervals"
    )
    parser.add_argument(
        "--ci-style",
        type=str,
        choices=['band', 'lines', 'both'],
        default='band',
        help="Style for confidence interval display: 'band' (shaded region), "
             "'lines' (dashed bounds), or 'both'"
    )
    parser.add_argument(
        "--ci-percentiles",
        type=str,
        default="16,84",
        metavar="LOW,HIGH",
        help="Percentiles for confidence interval (default: 16,84 for 1-sigma)"
    )
    parser.add_argument(
        "--ci-method",
        type=str,
        choices=['bounds', 'samples'],
        default='bounds',
        help="Method for computing confidence intervals: 'bounds' uses the +/- parameter "
             "ranges from MCMC summary (faster, default), 'samples' draws from the full posterior"
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
        bands = ['broad', 'soft', 'hard']
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
            obs_df = load_observed_lightcurves(band, args.data_dir)
            
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
            
            for key, stats in all_results.items():
                band, wind_model = key.rsplit('_', 1)
                f.write(f"{band.upper()} Band - {WIND_MODELS[wind_model]}\n")
                f.write("-"*40 + "\n")
                for param in PARAM_NAMES:
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
