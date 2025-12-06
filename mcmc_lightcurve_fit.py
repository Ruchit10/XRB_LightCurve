#!/usr/bin/env python3
"""
MCMC Light Curve Fitting for X-ray Binary Systems
--------------------------------------------------
This module performs Markov Chain Monte Carlo (MCMC) fitting to find optimal
binary system parameters by fitting model light curves to observed Chandra data.

PERFORMANCE OPTIMIZATION:
This code pre-computes a grid of model light curves and uses N-dimensional
interpolation during MCMC sampling. This provides ~1000x speedup compared to
calling simulate_lightcurve() for each MCMC step.

Features:
1. Loads observed Chandra light curves and phase-bins them (optional)
2. Pre-computes model grid for fast interpolation
3. Uses emcee ensemble sampler for MCMC parameter estimation
4. Fits each energy band (broad, soft, hard) independently
5. Outputs posterior distributions, best-fit parameters, and diagnostic plots

Parameters being fit:
- d1: Distance of compact object from center of mass (solar radii)
- d2: Distance of companion star from center of mass (solar radii)  
- r: Radius of compact object/accretion disk (solar radii)
- R: Radius of companion star (solar radii)
- i0: Orbital inclination (degrees)

Usage:
    # Fast mode with pre-computed grid (recommended)
    python mcmc_lightcurve_fit.py --band broad --flux-csv data_flux_vs_nH.csv
    
    # Fit all bands
    python mcmc_lightcurve_fit.py --band all --flux-csv data_flux_vs_nH.csv
"""

import argparse
import glob
import os
import time
import warnings
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

from scipy.interpolate import RegularGridInterpolator

from xrb_lightcurve import simulate_lightcurve


# =============================================================================
# Constants (from chandra_phase_analysis.py)
# =============================================================================
REF_EPOCH: float = 278801348  # Reference time (t0) for phase zero (seconds)
ORBITAL_PERIOD: float = 125431  # Orbital period in seconds (~34.8 hours)

# Default priors based on IC 10 X-1 parameters
DEFAULT_PRIORS = {
    'd1': {'mean': 11.0, 'std': 3.0, 'min': 5.0, 'max': 20.0},      # Solar radii
    'd2': {'mean': 8.0, 'std': 3.0, 'min': 3.0, 'max': 15.0},       # Solar radii
    'r': {'mean': 0.001, 'std': 0.001, 'min': 0.0001, 'max': 0.01}, # Solar radii
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
        - phase: bin center
        - flux: weighted mean flux in bin
        - flux_err: standard error of the mean
        - n_points: number of data points in bin
    """
    # Create bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Assign each point to a bin
    df = df.copy()
    df['bin'] = np.digitize(df['phase'], bin_edges) - 1
    df['bin'] = df['bin'].clip(0, n_bins - 1)  # Handle edge case at phase=1
    
    binned_data = []
    
    for i in range(n_bins):
        bin_mask = df['bin'] == i
        bin_df = df[bin_mask]
        
        if len(bin_df) >= min_points_per_bin:
            # Compute weighted mean and error
            flux_vals = bin_df['flux'].values
            
            # If errors are available, use weighted mean
            if 'flux_err' in bin_df.columns and not bin_df['flux_err'].isna().all():
                err_vals = bin_df['flux_err'].values
                # Replace zero/nan errors with median
                valid_err = err_vals[(err_vals > 0) & np.isfinite(err_vals)]
                if len(valid_err) > 0:
                    median_err = np.median(valid_err)
                    err_vals = np.where((err_vals <= 0) | ~np.isfinite(err_vals), median_err, err_vals)
                else:
                    err_vals = np.ones_like(flux_vals) * np.std(flux_vals)
                
                weights = 1.0 / err_vals**2
                mean_flux = np.average(flux_vals, weights=weights)
                # Standard error of weighted mean
                mean_err = np.sqrt(1.0 / np.sum(weights))
            else:
                # Simple mean and standard error
                mean_flux = np.mean(flux_vals)
                mean_err = np.std(flux_vals) / np.sqrt(len(flux_vals))
            
            binned_data.append({
                'phase': bin_centers[i],
                'flux': mean_flux,
                'flux_err': mean_err,
                'n_points': len(bin_df)
            })
    
    result = pd.DataFrame(binned_data)
    
    print(f"Phase binning: {len(df)} points -> {len(result)} bins "
          f"(avg {len(df)/n_bins:.1f} points/bin)")
    
    return result


# =============================================================================
# Pre-computed Model Grid (FAST)
# =============================================================================

def _compute_single_model(args):
    """Worker function to compute a single model (for parallel processing)."""
    d1, d2, r, R, i0, flux_csv_path, band, dth = args
    
    flux_column = f"nfl_{band}_av"
    
    try:
        results = simulate_lightcurve(
            r=r, R=R, d1=d1, d2=d2,
            gma0=-90.0, i0=i0, dth=dth,
            flux_method="interpolate",
            flux_csv_path=flux_csv_path,
            verbose=False
        )
        
        if flux_column not in results.columns:
            return None
        
        # Return flux values at standard phases
        model_phase = results['phase'].values
        model_flux = results[flux_column].values
        
        # Sort by phase
        sort_idx = np.argsort(model_phase)
        return model_flux[sort_idx]
        
    except Exception:
        return None


class PrecomputedModelGrid:
    """
    Pre-computed grid of model light curves for fast MCMC evaluation.
    
    This class pre-computes light curves on a coarse parameter grid and uses
    N-dimensional interpolation for fast evaluation during MCMC sampling.
    
    Typical speedup: ~1000x compared to calling simulate_lightcurve() directly.
    """
    
    def __init__(
        self,
        band: str,
        flux_csv_path: str,
        priors: Dict = DEFAULT_PRIORS,
        grid_points: Dict[str, int] = None,
        dth: float = 5.0,
        n_workers: int = None,
        verbose: bool = True
    ):
        """
        Initialize and pre-compute the model grid.
        
        Parameters
        ----------
        band : str
            Energy band ('broad', 'soft', 'hard')
        flux_csv_path : str
            Path to flux vs nH CSV file
        priors : dict
            Prior specifications (used to set grid bounds)
        grid_points : dict, optional
            Number of grid points per parameter. Default: 8 per parameter.
        dth : float
            Phase resolution for model computation (degrees)
        n_workers : int, optional
            Number of parallel workers. Default: number of CPUs.
        verbose : bool
            Print progress messages
        """
        self.band = band.lower()
        self.flux_csv_path = flux_csv_path
        self.priors = priors
        self.dth = dth
        self.verbose = verbose
        
        # Default grid resolution (8 points per parameter = 32,768 models)
        if grid_points is None:
            grid_points = {'d1': 8, 'd2': 8, 'r': 5, 'R': 8, 'i0': 10}
        
        self.grid_points = grid_points
        self.n_workers = n_workers if n_workers else max(1, cpu_count() - 1)
        
        # Create parameter grids
        self._create_grids()
        
        # Pre-compute all models
        self._precompute_models()
    
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
            print(f"Grid configuration: {total_models} models to compute")
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
                                self.flux_csv_path, self.band, self.dth
                            ))
        
        if self.verbose:
            print(f"Computing {len(param_combos)} valid parameter combinations...")
        
        # Compute models in parallel
        if self.n_workers > 1:
            with Pool(self.n_workers) as pool:
                results = pool.map(_compute_single_model, param_combos)
        else:
            results = [_compute_single_model(args) for args in param_combos]
        
        # Build the N-dimensional grid array
        shape = (
            len(d1_grid), len(d2_grid), len(r_grid), len(R_grid), len(i0_grid),
            len(self.phase_grid)
        )
        self.flux_grid = np.full(shape, np.nan)
        
        idx = 0
        for i_d1 in range(len(d1_grid)):
            for i_d2 in range(len(d2_grid)):
                for i_r in range(len(r_grid)):
                    for i_R in range(len(R_grid)):
                        for i_i0 in range(len(i0_grid)):
                            if r_grid[i_r] >= R_grid[i_R]:
                                continue
                            if results[idx] is not None:
                                self.flux_grid[i_d1, i_d2, i_r, i_R, i_i0, :] = results[idx]
                            idx += 1
        
        # Create interpolator for each phase point
        # We'll interpolate in 5D parameter space
        self.interpolators = []
        
        for i_phase in range(len(self.phase_grid)):
            flux_slice = self.flux_grid[:, :, :, :, :, i_phase]
            
            # Replace NaN with nearest valid value for robustness
            if np.any(np.isnan(flux_slice)):
                from scipy.ndimage import distance_transform_edt
                mask = np.isnan(flux_slice)
                # Fill NaN with mean for interpolation stability
                flux_slice = np.where(mask, np.nanmean(flux_slice), flux_slice)
            
            interp = RegularGridInterpolator(
                (d1_grid, d2_grid, r_grid, R_grid, i0_grid),
                flux_slice,
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            self.interpolators.append(interp)
        
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"Grid pre-computation completed in {elapsed:.1f} seconds")
            valid_fraction = np.sum(~np.isnan(self.flux_grid)) / self.flux_grid.size
            print(f"Valid grid coverage: {valid_fraction*100:.1f}%")
    
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
        
        This is ~1000x faster than calling simulate_lightcurve().
        
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
    Only use this for debugging or single evaluations.
    """
    
    def __init__(
        self,
        band: str,
        flux_csv_path: str,
        dth: float = 5.0,
        flux_method: str = "interpolate"
    ):
        self.band = band.lower()
        self.flux_csv_path = flux_csv_path
        self.dth = dth
        self.flux_method = flux_method
        self.flux_column = f"nfl_{self.band}_av"
        
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
                gma0=-90.0, i0=i0, dth=self.dth,
                flux_method=self.flux_method,
                flux_csv_path=self.flux_csv_path,
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
    """
    Log prior probability for parameters.
    
    Uses truncated Gaussian priors with hard bounds.
    """
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
    model,  # Can be PrecomputedModelGrid or DirectLightCurveModel
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
    progress: bool = True
) -> Tuple[emcee.EnsembleSampler, np.ndarray]:
    """
    Run MCMC sampling.
    
    Parameters
    ----------
    model : PrecomputedModelGrid or DirectLightCurveModel
        Model evaluator (use PrecomputedModelGrid for speed!)
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
            pos[j, 2] = pos[j, 3] * 0.1  # Set r to 10% of R
    
    print(f"\nStarting MCMC with {n_walkers} walkers, {n_steps} steps...")
    print(f"Initial parameter values (first walker): {pos[0]}")
    
    # Create sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_probability,
        args=(model, obs_phase, obs_flux, obs_err, priors)
    )
    
    # Run MCMC with timing
    start_time = time.time()
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


def print_results(stats: Dict, band: str):
    """Print formatted results table."""
    print(f"\n{'='*60}")
    print(f"MCMC Results for {band.upper()} band")
    print('='*60)
    print(f"{'Parameter':<15} {'Median':<12} {'Lower σ':<12} {'Upper σ':<12}")
    print('-'*60)
    
    for param in PARAM_NAMES:
        s = stats[param]
        print(f"{param:<15} {s['median']:<12.6f} {s['lower']:<12.6f} {s['upper']:<12.6f}")
    
    print('='*60)


def plot_corner(samples: np.ndarray, band: str, output_path: str):
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
    
    fig.suptitle(f"Posterior Distributions - {band.upper()} band", fontsize=14, y=1.02)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Corner plot saved to: {output_path}")


def plot_trace(sampler: emcee.EnsembleSampler, band: str, output_path: str):
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
    fig.suptitle(f"MCMC Trace Plots - {band.upper()} band", fontsize=14)
    
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
    output_path: str
):
    """Plot observed data with best-fit model overlay."""
    # Get best-fit parameters
    best_params = [stats[p]['median'] for p in PARAM_NAMES]
    
    # Generate model curve
    model_phases = np.linspace(0, 1, 360)
    model_flux = model.evaluate(*best_params, model_phases)
    
    # Compute chi-square
    obs_model = model.evaluate(*best_params, obs_phase)
    chi2 = np.sum(((obs_flux - obs_model) / obs_err) ** 2)
    dof = len(obs_flux) - len(PARAM_NAMES)
    red_chi2 = chi2 / dof if dof > 0 else np.nan
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Observed data with error bars
    ax.errorbar(obs_phase, obs_flux, yerr=obs_err, fmt='o', 
                markersize=4, alpha=0.7, label='Observed (phase-binned)',
                capsize=2, elinewidth=1)
    
    # Best-fit model
    ax.plot(model_phases, model_flux, 'r-', lw=2, label='Best-fit model')
    
    ax.set_xlabel('Orbital Phase', fontsize=12)
    ax.set_ylabel('Flux (erg/cm²/s)', fontsize=12)
    ax.set_title(f'{band.upper()} Band - Best Fit (χ²/dof = {red_chi2:.2f})', fontsize=14)
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
        print("  ⚠️  Low acceptance fraction - consider adjusting priors or step size")
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
        help="Energy band to fit (or 'all' to fit all bands sequentially)"
    )
    parser.add_argument(
        "--flux-csv",
        type=str,
        required=True,
        help="Path to flux vs nH CSV file (from compute_flux_vs_nH.py)"
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
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which bands to fit
    if args.band == 'all':
        bands = ['broad', 'soft', 'hard']
    else:
        bands = [args.band]
    
    # Run MCMC for each band
    all_results = {}
    
    for band in bands:
        print(f"\n{'#'*60}")
        print(f"# Fitting {band.upper()} band")
        print('#'*60)
        
        try:
            # Load and optionally phase-bin data
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
            
            # Initialize model (pre-computed grid or direct)
            if args.no_grid:
                print("\n⚠️  Using direct model evaluation (SLOW!)")
                model = DirectLightCurveModel(
                    band=band,
                    flux_csv_path=args.flux_csv,
                    dth=args.dth
                )
            else:
                # Pre-compute model grid
                grid_points = {
                    'd1': args.grid_points,
                    'd2': args.grid_points,
                    'r': max(5, args.grid_points // 2),  # Fewer points for r
                    'R': args.grid_points,
                    'i0': args.grid_points + 2  # Extra points for inclination
                }
                
                model = PrecomputedModelGrid(
                    band=band,
                    flux_csv_path=args.flux_csv,
                    grid_points=grid_points,
                    dth=args.dth,
                    n_workers=args.n_workers,
                    verbose=True
                )
            
            # Run MCMC
            sampler, samples = run_mcmc(
                model=model,
                obs_phase=obs_phase,
                obs_flux=obs_flux,
                obs_err=obs_err,
                n_walkers=args.n_walkers,
                n_steps=args.n_steps,
                n_burn=args.n_burn,
                progress=not args.quiet
            )
            
            # Compute statistics
            stats = compute_statistics(samples)
            print_results(stats, band)
            print_diagnostics(sampler)
            
            # Generate plots
            if not args.no_plots:
                plot_corner(
                    samples, band,
                    os.path.join(args.output_dir, f"{band}_corner.png")
                )
                plot_trace(
                    sampler, band,
                    os.path.join(args.output_dir, f"{band}_trace.png")
                )
                red_chi2 = plot_best_fit(
                    model, obs_phase, obs_flux, obs_err, stats, band,
                    os.path.join(args.output_dir, f"{band}_bestfit.png")
                )
                stats['reduced_chi2'] = red_chi2
            
            # Save samples
            samples_df = pd.DataFrame(samples, columns=PARAM_NAMES)
            samples_df.to_csv(
                os.path.join(args.output_dir, f"{band}_samples.csv"),
                index=False
            )
            print(f"Samples saved to: {args.output_dir}/{band}_samples.csv")
            
            # Store results
            all_results[band] = stats
            
        except Exception as e:
            print(f"ERROR fitting {band} band: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary results
    if all_results:
        summary_path = os.path.join(args.output_dir, "mcmc_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("MCMC Light Curve Fitting Results\n")
            f.write("="*60 + "\n\n")
            
            for band, stats in all_results.items():
                f.write(f"{band.upper()} Band\n")
                f.write("-"*40 + "\n")
                for param in PARAM_NAMES:
                    s = stats[param]
                    f.write(f"{param}: {s['median']:.6f} (+{s['upper']:.6f}/-{s['lower']:.6f})\n")
                if 'reduced_chi2' in stats:
                    f.write(f"Reduced chi-square: {stats['reduced_chi2']:.3f}\n")
                f.write("\n")
        
        print(f"\nSummary saved to: {summary_path}")
    
    print("\nMCMC fitting complete!")


if __name__ == "__main__":
    main()
