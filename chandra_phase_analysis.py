#!/usr/bin/env python3
"""
X-ray Binary Phase Analysis
----------------------------
This script converts observational X-ray light-curve data to orbital phase and
fits simulation models to the observations.

Features:
1.  Reads all .txt files from a data directory (or a specified master file)
2.  Converts observation times to orbital phase using the reference epoch and
    orbital period
3.  Optionally verifies that individual files are contained in a master file
4.  Produces scatter plots of count-rate versus orbital phase
5.  Fits simulation models to observations via chi-square minimization
6.  Supports multiple energy bands and automatically detects available flux columns

File Format:
  Whitespace-delimited text files with three columns:
    1. time (seconds)
    2. count rate / flux
    3. error (optional)

Examples
~~~~~~~~
# Load all .txt files from a custom directory:
$ python chandra_phase_analysis.py --data-dir my_observations --output phase_plot.png

# Use a specific master file:
$ python chandra_phase_analysis.py --data-dir data --master-file Chandra.txt --output plot.png

# Use specific observation column (e.g., NET_RATE instead of default):
$ python chandra_phase_analysis.py --data-dir data --obs-column NET_RATE --output plot.png

# Use FLUX column from observations with specific error column:
$ python chandra_phase_analysis.py --data-dir data --obs-column FLUX --obs-error-column FLUX_ERR --output plot.png

# Fit simulation to observations (auto-detects all flux columns):
$ python chandra_phase_analysis.py --data-dir data --fit --sim-file simulation.csv --output fit.png

# Fit specific flux columns:
$ python chandra_phase_analysis.py --data-dir data --fit --sim-file sim.csv \\
    --sim-column nfl_broad_av nfl_soft_av --output fit.png

# Fit with specific observation column, without rescaling:
$ python chandra_phase_analysis.py --data-dir data --fit --sim-file sim.csv \\
    --obs-column FLUX --sim-column nfl_broad_av --output fit.png

# Fit with rescaling enabled:
$ python chandra_phase_analysis.py --data-dir data --fit --sim-file sim.csv \\
    --obs-column NET_RATE --sim-column nfl_broad_av --rescale --output fit.png

# Load CIAO format data (time in second column, flux as ECF):
$ python chandra_phase_analysis.py --data-dir data/IC_10_X1_LC_CIAO/broad \\
    --obs-column ECF --output ciao_plot.png

# Fit CIAO data to simulation:
$ python chandra_phase_analysis.py --data-dir data/IC_10_X1_LC_CIAO/broad \\
    --obs-column ECF --fit --sim-file sim.csv --output ciao_fit.png

Dependencies: numpy, pandas, matplotlib, scipy (in requirements.txt).
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -----------------------------------------------------------------------------
# Constants adopted from the R script (seconds)
# -----------------------------------------------------------------------------
REF_EPOCH: float = 278801348  # Reference time (t0) used for phase zero
ORBITAL_PERIOD: float = 125431  # Orbital period of the system


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def frac(x: np.ndarray | float) -> np.ndarray | float:
    """Return the fractional part of *x* (vectorised)."""
    return np.abs(x - np.floor(x))


def detect_flux_columns(df: pd.DataFrame) -> List[str]:
    """Detect available flux columns in simulation DataFrame.
    
    Looks for columns matching common flux patterns:
    - nfl_{band}_av (accretion velocity wind model, scaled by lam)
    - nfl_{band}_cv (constant velocity wind model, scaled by lam)
    - pho_count_{band}_av (photon counts)
    
    Note: fl and fl2 columns are excluded as they are unscaled column density values.
    
    Parameters
    ----------
    df : DataFrame
        Simulation results DataFrame
        
    Returns
    -------
    List of flux column names found in the DataFrame
    """
    flux_columns = []
    
    # Check for nfl_* columns (normalized flux for various bands, scaled by lam)
    for col in df.columns:
        if col.startswith("nfl_") and (col.endswith("_av") or col.endswith("_cv")):
            flux_columns.append(col)
    
    # Check for pho_count_* columns (photon counts)
    for col in df.columns:
        if col.startswith("pho_count_") and col.endswith("_av"):
            flux_columns.append(col)
    
    return sorted(flux_columns)


def validate_sim_columns(df: pd.DataFrame, requested_columns: List[str]) -> List[str]:
    """Validate that requested columns exist in simulation DataFrame.
    
    Parameters
    ----------
    df : DataFrame
        Simulation results DataFrame
    requested_columns : list of str
        Column names requested by user
        
    Returns
    -------
    List of valid column names
    
    Raises
    ------
    ValueError
        If none of the requested columns exist in the DataFrame
    """
    available = detect_flux_columns(df)
    
    # Check which requested columns exist
    valid_columns = [col for col in requested_columns if col in df.columns]
    missing_columns = [col for col in requested_columns if col not in df.columns]
    
    if missing_columns:
        print(f"⚠️  Warning: The following columns were not found in simulation file: {missing_columns}")
        if available:
            print(f"   Available flux columns: {available}")
    
    if not valid_columns:
        raise ValueError(
            f"None of the requested columns exist in simulation file.\n"
            f"Requested: {requested_columns}\n"
            f"Available: {available if available else 'No flux columns found'}"
        )
    
    return valid_columns


def read_observation(file_path: str, label: str, obs_column: str = "rate", obs_error_column: Optional[str] = None, time_column: Optional[str] = None, phase_column: Optional[str] = None) -> pd.DataFrame:
    """Read a single Chandra observation text file.

    The files can be whitespace-delimited with or without headers.
    If a header is present (lines starting with #), column names are extracted.
    Otherwise, assumes three columns: time, count rate/flux, error.
    
    Supports multiple formats:
    1. Standard format: TIME in first column, tab or space delimited
    2. CIAO format: time column (t_raw or time), space delimited, with "# Columns:" or "# #Columns:" header
    3. Pre-computed phase: if phase_column is specified, uses that instead of computing from time
    
    Parameters
    ----------
    file_path : str
        Path to observation file
    label : str
        Label for this observation
    obs_column : str, default "rate"
        Name of column to use for the observable (e.g., "NET_RATE", "FLUX", "COUNT_RATE", "ECF", "flux_t").
        If file has no header, this is ignored and "rate" is used.
    obs_error_column : str, optional
        Name of column to use for errors. If None, will attempt to auto-detect based on obs_column
        (e.g., "ERR_RATE" for "NET_RATE", "FLUX_ERR" for "FLUX", "rate_err" for CIAO).
    time_column : str, optional
        Name of column containing timestamps (e.g., "TIME", "time", "t_raw"). If None, will auto-detect.
    phase_column : str, optional
        Name of column containing pre-computed orbital phase. If specified, uses this instead of
        computing phase from time. Useful for CIAO files that already have phase computed.
    
    Returns
    -------
    DataFrame with columns: time, phase, obs, and the specified observable column renamed to "rate"
    (and optionally "error" column)
    """
    # Try to read with header detection
    try:
        # Read file and check for header format
        with open(file_path, 'r') as f:
            lines = []
            for line in f:
                lines.append(line.strip())
                if not line.strip().startswith('#') or len(lines) > 10:
                    break
        
        # Check for CIAO format with "Columns:" header (handles both "# Columns:" and "# #Columns:")
        ciao_format = False
        header_line = None
        for line in lines:
            # Match both "# Columns:" and "# #Columns:" formats
            if line.startswith('#') and 'Columns:' in line:
                ciao_format = True
                # Extract column names after "Columns:"
                col_part = line.split('Columns:')[1].strip()
                col_names = [c.strip() for c in col_part.split(',')]
                header_line = ' '.join(col_names)
                break
            elif line.startswith('#') and not ciao_format:
                # Standard format: check if this line has column-like content
                # Skip lines with ":" or "=" which indicate metadata
                clean_line = line.lstrip('#').strip()
                if clean_line and ':' not in clean_line and '=' not in clean_line:
                    if any(name in line.upper() for name in ['TIME', 'RATE', 'FLUX']):
                        header_line = clean_line
        
        # Check if file has a header
        has_header = header_line is not None or any(
            any(col_name in line.upper() for col_name in ['TIME', 'RATE', 'FLUX', 'COUNTS', 'ECF', 'PHASE'])
            for line in lines if line.startswith('#')
        )
        
        if has_header:
            # Read with header - skip comment lines
            df = pd.read_csv(file_path, sep='\\s+', comment='#', header=None)
            
            if header_line:
                col_names = header_line.split()
                if len(col_names) == len(df.columns):
                    df.columns = col_names
                    
                    # Find time column (case-insensitive, or use user-specified)
                    time_col = None
                    if time_column:
                        # User specified time column
                        for col in df.columns:
                            if col.upper() == time_column.upper():
                                time_col = col
                                break
                        if not time_col:
                            print(f"⚠️  Warning: Specified time column '{time_column}' not found in {file_path}")
                            print(f"   Available columns: {list(df.columns)}")
                    
                    if not time_col:
                        # Auto-detect time column - check common names
                        time_column_names = ['TIME', 'T_RAW', 'T', 'MJD']
                        for time_name in time_column_names:
                            for col in df.columns:
                                if col.upper() == time_name:
                                    time_col = col
                                    break
                            if time_col:
                                break
                    
                    # Check for phase column (pre-computed)
                    phase_col = None
                    if phase_column:
                        for col in df.columns:
                            if col.upper() == phase_column.upper():
                                phase_col = col
                                break
                        if not phase_col:
                            print(f"⚠️  Warning: Specified phase column '{phase_column}' not found in {file_path}")
                    else:
                        # Auto-detect phase column
                        for col in df.columns:
                            if col.upper() == 'PHASE':
                                phase_col = col
                                break
                    
                    # Need either time or phase column
                    if not time_col and not phase_col:
                        print(f"⚠️  Warning: No time or phase column found in {file_path}")
                        print(f"   Available columns: {list(df.columns)}")
                        raise ValueError("No time or phase column found")
                    
                    # Case-insensitive column matching for obs_column
                    actual_obs_column = None
                    for col in df.columns:
                        if col.upper() == obs_column.upper():
                            actual_obs_column = col
                            break
                    
                    if not actual_obs_column:
                        # Column not found - print available columns and raise error
                        print(f"⚠️  Error: Column '{obs_column}' not found in {file_path}")
                        print(f"   Available columns: {list(df.columns)}")
                        raise ValueError(f"Column '{obs_column}' not found in observation file")
                    
                    # Use specified columns
                    result_df = pd.DataFrame({
                        'rate': df[actual_obs_column],
                    })
                    
                    # Add time column if available
                    if time_col:
                        result_df['time'] = df[time_col]
                    elif phase_col:
                        # Use a dummy time value if only phase is available
                        result_df['time'] = 0.0
                    
                    # Try to find error column
                    error_col = None
                    if obs_error_column:
                        # Check for user-specified error column (case-insensitive)
                        for col in df.columns:
                            if col.upper() == obs_error_column.upper():
                                error_col = col
                                break
                    
                    if not error_col:
                        # Auto-detect error column
                        possible_error_cols = [
                            f"{actual_obs_column}_ERR",
                            f"ERR_{actual_obs_column}",
                            actual_obs_column.replace("RATE", "ERR_RATE").replace("FLUX", "FLUX_ERR"),
                            # For CIAO format, try rate_err and count_rate_err
                            "rate_err",
                            "count_rate_err",
                        ]
                        for err_col in possible_error_cols:
                            for col in df.columns:
                                if col.upper() == err_col.upper():
                                    error_col = col
                                    break
                            if error_col:
                                break
                        
                        # Also try case-insensitive matching for generic error columns
                        if not error_col:
                            for col in df.columns:
                                if 'ERR' in col.upper():
                                    # Prefer error column related to the obs column
                                    obs_base = actual_obs_column.split('_')[0].upper()
                                    if obs_base in col.upper() or 'RATE' in col.upper():
                                        error_col = col
                                        break
                    
                    if error_col:
                        result_df['error'] = df[error_col]
                    
                    # Determine phase: use pre-computed if available, otherwise compute from time
                    if phase_col:
                        result_df['phase'] = df[phase_col]
                    elif time_col:
                        result_df['phase'] = frac((result_df['time'] - REF_EPOCH) / ORBITAL_PERIOD)
                    
                    result_df['obs'] = label
                    
                    return result_df
    except Exception as e:
        pass
    
    # Fallback: read as headerless file with 3 columns
    df = pd.read_csv(
        file_path,
        sep='\\s+',
        comment='#',
        header=None,
        names=["time", "rate", "error"],
    )
    # Convert timestamps to orbital phase (0–1)
    df["phase"] = frac((df["time"] - REF_EPOCH) / ORBITAL_PERIOD)
    df["obs"] = label
    return df


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------

def load_data(data_dir: str, master_file: Optional[str] = None, obs_column: str = "rate", obs_error_column: Optional[str] = None, time_column: Optional[str] = None, phase_column: Optional[str] = None) -> pd.DataFrame:
    """Load observational data from *data_dir*.

    Parameters
    ----------
    data_dir : str
        Directory containing observation text files
    master_file : str, optional
        Name of master file (if it exists). If provided and exists, only this file is loaded.
        If None, all .txt files in the directory are loaded.
    obs_column : str, default "rate"
        Name of column to use for the observable (e.g., "NET_RATE", "FLUX", "COUNT_RATE", "ECF", "flux_t")
    obs_error_column : str, optional
        Name of column to use for errors. If None, will auto-detect based on obs_column.
    time_column : str, optional
        Name of column containing timestamps. If None, will auto-detect (looks for 'time', 't_raw').
    phase_column : str, optional
        Name of column containing pre-computed phase. If None, will auto-detect or compute from time.

    Returns
    -------
    DataFrame with columns: time, rate (containing the specified observable), error (optional), phase, obs
    """
    # Check for master file if specified
    if master_file:
        master_path = os.path.join(data_dir, master_file)
        if os.path.isfile(master_path):
            print(f"Using master file: {master_path}")
            return read_observation(master_path, "master", obs_column, obs_error_column, time_column, phase_column)
        else:
            print(f"Warning: Master file '{master_file}' not found in {data_dir}, loading all files instead.")

    # Load all .txt files from directory
    txt_pattern = os.path.join(data_dir, "*.txt")
    files: List[str] = sorted(glob.glob(txt_pattern))
    
    if not files:
        raise FileNotFoundError(
            f"No .txt files found in {data_dir}"
        )

    print(f"Loading {len(files)} observation file(s) from {data_dir}")
    dfs = [read_observation(fp, os.path.basename(fp), obs_column, obs_error_column, time_column, phase_column) for fp in files]
    return pd.concat(dfs, ignore_index=True)


def phase_bin_data(
    df: pd.DataFrame,
    n_bins: int = 50,
    min_points_per_bin: int = 3,
    rate_column: str = 'rate',
    error_column: str = 'error',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Bin observed data into orbital phase bins.
    
    This function groups data points by orbital phase and computes weighted
    averages within each bin. Useful for reducing scatter in light curves
    and for comparing with phase-folded models.
    
    Parameters
    ----------
    df : DataFrame
        Observed data with columns: phase, and the rate/error columns
    n_bins : int
        Number of phase bins (default 50)
    min_points_per_bin : int
        Minimum number of data points required per bin (default 3)
    rate_column : str
        Name of column containing flux/rate values (default 'rate')
    error_column : str
        Name of column containing error values (default 'error')
    verbose : bool
        Print summary of binning operation (default True)
        
    Returns
    -------
    DataFrame with columns: phase, rate (binned), error (binned), n_points
    
    Notes
    -----
    - Uses weighted mean if errors are available, otherwise simple mean
    - Bins with fewer than min_points_per_bin are excluded
    - Error on weighted mean is computed as sqrt(1/sum(weights))
    """
    # Create bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Assign each point to a bin
    df = df.copy()
    df['_bin'] = np.digitize(df['phase'], bin_edges) - 1
    df['_bin'] = df['_bin'].clip(0, n_bins - 1)  # Handle edge case at phase=1
    
    binned_data = []
    
    for i in range(n_bins):
        bin_mask = df['_bin'] == i
        bin_df = df[bin_mask]
        
        if len(bin_df) >= min_points_per_bin:
            # Get rate values
            rate_vals = bin_df[rate_column].values
            
            # If errors are available, use weighted mean
            has_errors = (error_column in bin_df.columns and 
                         not bin_df[error_column].isna().all())
            
            if has_errors:
                err_vals = bin_df[error_column].values
                # Replace zero/nan errors with median of valid errors
                valid_err = err_vals[(err_vals > 0) & np.isfinite(err_vals)]
                if len(valid_err) > 0:
                    median_err = np.median(valid_err)
                    err_vals = np.where((err_vals <= 0) | ~np.isfinite(err_vals), 
                                       median_err, err_vals)
                else:
                    err_vals = np.ones_like(rate_vals) * np.std(rate_vals)
                
                weights = 1.0 / err_vals**2
                mean_rate = np.average(rate_vals, weights=weights)
                # Standard error of weighted mean
                mean_err = np.sqrt(1.0 / np.sum(weights))
            else:
                # Simple mean and standard error
                mean_rate = np.mean(rate_vals)
                mean_err = np.std(rate_vals) / np.sqrt(len(rate_vals))
            
            binned_data.append({
                'phase': bin_centers[i],
                'rate': mean_rate,
                'error': mean_err,
                'n_points': len(bin_df)
            })
    
    result = pd.DataFrame(binned_data)
    
    # Preserve observation label if present (use 'binned')
    if 'obs' in df.columns:
        result['obs'] = 'binned'
    
    if verbose:
        print(f"Phase binning: {len(df)} points -> {len(result)} bins "
              f"(avg {len(df)/n_bins:.1f} points/bin)")
    
    return result


def verify_master_contains_individual(data_dir: str, master_file: str = "Chandra.txt") -> None:
    """Check that every timestamp in individual .txt files appears in the master file.
    
    Parameters
    ----------
    data_dir : str
        Directory containing observation files
    master_file : str, optional
        Name of the master file to verify against (default: "Chandra.txt")
    """
    master_path = os.path.join(data_dir, master_file)
    if not os.path.isfile(master_path):
        print(f"No master file '{master_file}' found; skipping verification.")
        return

    print(f"Verifying individual files against master file: {master_file}\n")
    master_times = (
        pd.read_csv(master_path, delim_whitespace=True, header=None, usecols=[0])[0]
        .round(6)
        .astype(str)
        .tolist()
    )
    master_set = set(master_times)

    # Get all .txt files except the master file
    all_files = glob.glob(os.path.join(data_dir, "*.txt"))
    individual_files = [fp for fp in all_files if os.path.basename(fp) != master_file]
    
    if not individual_files:
        print("No individual files found to verify.")
        return

    for fp in sorted(individual_files):
        ind_times = (
            pd.read_csv(fp, delim_whitespace=True, header=None, usecols=[0])[0]
            .round(6)
            .astype(str)
            .tolist()
        )
        missing = [t for t in ind_times if t not in master_set]
        if missing:
            print(f"⚠️  {os.path.basename(fp)}: {len(missing)} point(s) NOT in master file.")
        else:
            print(f"✓ {os.path.basename(fp)} fully contained in master file.")
    print()


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def fit_simulation(obs_df: pd.DataFrame, sim_df: pd.DataFrame, sim_column: str = "fl", rescale: bool = False) -> tuple[float, float, float]:
    """Fit simulation light-curve to observations via chi-square minimization.

    Parameters
    ----------
    obs_df : DataFrame
        Observational data with columns ``phase``, ``rate`` and (optionally) ``error``.
    sim_df : DataFrame
        Simulation results. Must contain columns ``phase`` (or ``deg``) and *sim_column*.
    sim_column : str, default ``"fl"``
        Column in *sim_df* to use as the model flux.
    rescale : bool, default False
        If True, optimize phase shift and scale factor to minimize chi-square.
        If False, compute chi-square with no shift (shift=0) and no scaling (scale=1).

    Returns
    -------
    (phase_shift, scale_factor, reduced_chi2)
        Best-fit phase shift (0–1), multiplicative scale factor, and reduced chi-squared value.
        If rescale=False, returns (0.0, 1.0, reduced_chi2).
    """
    # Prepare observation arrays
    phase_obs = obs_df["phase"].to_numpy()
    rate_obs = obs_df["rate"].to_numpy()

    # Use provided statistical errors when available; otherwise adopt sqrt(counts).
    if "error" in obs_df.columns and not obs_df["error"].isnull().all():
        err_obs = obs_df["error"].to_numpy()
    else:
        err_obs = np.sqrt(np.abs(rate_obs))

    # Guard against zero or negative uncertainties (would blow up χ²)
    err_obs = np.where(err_obs <= 0, 1e-3, err_obs)

    # Prepare simulation arrays
    if "phase" not in sim_df.columns:
        if "deg" in sim_df.columns:
            sim_df["phase"] = (sim_df["deg"] % 360) / 360.0
        else:
            raise ValueError("Simulation file must contain 'phase' or 'deg' column.")

    sim_phase = np.mod(sim_df["phase"].to_numpy(), 1.0)
    sim_flux = sim_df[sim_column].to_numpy()

    # Ensure ascending order for interpolation and duplicate first point +1 for wrap-around
    order = np.argsort(sim_phase)
    sim_phase_sorted = sim_phase[order]
    sim_flux_sorted = sim_flux[order]
    sim_phase_wrap = np.concatenate([sim_phase_sorted, sim_phase_sorted + 1])
    sim_flux_wrap  = np.concatenate([sim_flux_sorted,  sim_flux_sorted])

    # Keep only strictly increasing x-values to avoid interp warnings
    uniq_idx = np.concatenate(([True], np.diff(sim_phase_wrap) > 0))
    sim_phase_wrap = sim_phase_wrap[uniq_idx]
    sim_flux_wrap  = sim_flux_wrap[uniq_idx]

    # Chi-square function
    def chi2(params: np.ndarray) -> float:
        shift, scale = params
        model = np.interp(
            (phase_obs - shift) % 1.0,
            sim_phase_wrap,
            sim_flux_wrap,
        ) * scale
        return np.sum(((rate_obs - model) / err_obs) ** 2)

    if rescale:
        # Perform optimization to find best-fit shift and scale
        # Initial guess: no shift, scale = ratio of means
        mean_sim = np.mean(sim_flux_sorted)
        initial_scale = (np.mean(rate_obs) / mean_sim) if mean_sim > 0 else 1.0
        res = minimize(chi2, x0=[0.0, initial_scale], bounds=[(0, 1), (0, None)], method="Nelder-Mead")

        if not res.success:
            print("⚠️  Optimization did not converge; results may be unreliable.")

        best_shift, best_scale = res.x % np.array([1.0, np.inf])
        reduced_chi2 = res.fun / max(len(rate_obs) - 2, 1)
        print(
            f"Best-fit parameters:\n  Phase shift = {best_shift:.5f}\n  Scale factor = {best_scale:.5f}\n  Reduced χ² = {reduced_chi2:.3f}"
        )
        return float(best_shift), float(best_scale), float(reduced_chi2)
    else:
        # No optimization: compute chi-square with no shift and no scaling
        best_shift = 0.0
        best_scale = 1.0
        chi2_value = chi2(np.array([best_shift, best_scale]))
        reduced_chi2 = chi2_value / max(len(rate_obs) - 2, 1)
        print(
            f"Chi-square (no rescaling):\n  Phase shift = {best_shift:.5f} (fixed)\n  Scale factor = {best_scale:.5f} (fixed)\n  Reduced χ² = {reduced_chi2:.3f}"
        )
        return float(best_shift), float(best_scale), float(reduced_chi2)


def plot_phase(
    df: pd.DataFrame,
    output_path: str | None,
    sim_df: pd.DataFrame | None = None,
    shift: float | None = None,
    scale: float | None = None,
    sim_column: str = "fl",
    chi2: float | None = None,
    ax: plt.Axes | None = None,
    rescaled: bool = False,
    obs_column_name: str = "rate",
    is_binned: bool = False,
) -> None:
    """Scatter plot with optional best-fit simulation overlay.
    
    Parameters
    ----------
    df : DataFrame
        Observational data with columns ``phase``, ``rate``, and optionally ``error``.
    output_path : str, optional
        Output filename to save the plot.
    sim_df : DataFrame, optional
        Simulation data.
    shift : float, optional
        Phase shift for simulation overlay.
    scale : float, optional
        Scale factor for simulation overlay.
    sim_column : str, default ``"fl"``
        Column name in simulation to use.
    chi2 : float, optional
        Reduced chi-squared value to annotate on plot.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure.
    rescaled : bool, default False
        Whether the model was rescaled (optimized) or not.
    obs_column_name : str, default "rate"
        Name of the observable column being plotted (for labeling).
    is_binned : bool, default False
        Whether the data has been phase-binned. If True, plots with error bars.
    """
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    # Check if we have error data
    has_errors = 'error' in df.columns and not df['error'].isna().all()
    
    for label, group in df.groupby("obs"):
        if is_binned and has_errors:
            # Plot with error bars for binned data
            ax.errorbar(
                group["phase"], group["rate"], yerr=group["error"],
                fmt='o', markersize=5, alpha=0.8, capsize=2, elinewidth=1,
                label=f"{label} (n={len(group)} bins)"
            )
        else:
            # Scatter plot for unbinned data
            ax.scatter(group["phase"], group["rate"], s=12, alpha=0.7, label=label)

    if sim_df is not None and shift is not None and scale is not None:
        # Prepare simulation curve for overlay
        if "phase" not in sim_df.columns and "deg" in sim_df.columns:
            sim_df = sim_df.copy()
            sim_df["phase"] = (sim_df["deg"] % 360) / 360.0
        sim_phase = np.mod(sim_df["phase"].to_numpy(), 1.0)
        sim_flux = sim_df[sim_column].to_numpy() * scale

        # Sort and shift
        sort_idx = np.argsort(sim_phase)
        phase_sorted = sim_phase[sort_idx]
        flux_sorted  = sim_flux[sort_idx]
        phase_overlay = (phase_sorted + shift) % 1.0

        # Resort after modulo so the line is drawn strictly within 0–1
        re_sort = np.argsort(phase_overlay)
        phase_overlay = phase_overlay[re_sort]
        flux_overlay  = flux_sorted[re_sort]

        ax.plot(phase_overlay, flux_overlay, "k-", linewidth=2, label="Best-fit model")

    ax.set_xlabel("Orbital phase")
    ylabel = obs_column_name.replace("_", " ").title() if obs_column_name != "rate" else "Count rate / Flux"
    ax.set_ylabel(ylabel)
    
    # Set title with chi-squared annotation if provided
    if chi2 is not None:
        rescale_label = " (rescaled)" if rescaled else " (no rescaling)"
        ax.set_title(f"{sim_column}\nReduced χ² = {chi2:.3f}{rescale_label}")
    else:
        ax.set_title("Chandra Light-curve Observations")
    
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")

    if ax is None or output_path:
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_path}")
        else:
            plt.show()


def plot_multi_column_fits(
    df: pd.DataFrame,
    output_path: str | None,
    sim_df: pd.DataFrame,
    sim_columns: List[str],
    fit_results: List[tuple[float, float, float]],
    rescaled: bool = False,
    obs_column_name: str = "rate",
    is_binned: bool = False,
) -> None:
    """Plot multiple fitted simulation columns in a grid layout.
    
    Parameters
    ----------
    df : DataFrame
        Observational data.
    output_path : str, optional
        Output filename to save the plot.
    sim_df : DataFrame
        Simulation data containing all columns to plot.
    sim_columns : list of str
        List of column names to plot.
    fit_results : list of tuples
        List of (shift, scale, chi2) tuples for each column.
    rescaled : bool, default False
        Whether the models were rescaled (optimized) or not.
    obs_column_name : str, default "rate"
        Name of the observable column being plotted (for labeling).
    is_binned : bool, default False
        Whether the data has been phase-binned.
    """
    n_cols = len(sim_columns)
    
    # Determine grid layout (try to make it roughly square)
    n_rows = int(np.ceil(np.sqrt(n_cols)))
    n_plot_cols = int(np.ceil(n_cols / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(6 * n_plot_cols, 5 * n_rows))
    
    # Handle case where axes is not an array
    if n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (col, (shift, scale, chi2)) in enumerate(zip(sim_columns, fit_results)):
        ax = axes[i]
        plot_phase(df, None, sim_df, shift, scale, col, chi2, ax, rescaled, obs_column_name, is_binned)
    
    # Hide unused subplots
    for i in range(n_cols, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Multi-column plot saved to {output_path}")
    else:
        plt.show()


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert X-ray observation times to orbital phase, plot light curves, and fit simulation models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing observation text files (.txt format with time, rate, error columns).",
    )
    parser.add_argument(
        "--master-file",
        type=str,
        default=None,
        help="Name of master file in data directory (e.g., 'Chandra.txt'). If specified and exists, "
             "only this file is loaded. If not specified, all .txt files in the directory are loaded.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for the generated plot. If omitted, the plot is shown interactively.",
    )
    parser.add_argument(
        "--verify-master",
        action="store_true",
        help="Verify that each individual .txt file is contained in the master file (requires --master-file).",
    )
    parser.add_argument(
        "--sim-file",
        type=str,
        default=None,
        help="CSV file containing simulation results to fit.",
    )
    parser.add_argument(
        "--obs-column",
        type=str,
        default=None,
        help="Column name in observation files to use (e.g., 'NET_RATE', 'FLUX', 'COUNT_RATE'). "
             "If the observation files have headers, this column name will be used. "
             "If not specified, uses 'rate' (assumes 3-column headerless format).",
    )
    parser.add_argument(
        "--obs-error-column",
        type=str,
        default=None,
        help="Column name for observation errors (e.g., 'ERR_RATE', 'FLUX_ERR'). "
             "If not specified, will attempt to auto-detect based on --obs-column.",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default=None,
        help="Column name for timestamps (e.g., 'TIME', 'time', 't_raw'). "
             "If not specified, will auto-detect by looking for common time column names. "
             "Useful for CIAO format files where time may be in a different column.",
    )
    parser.add_argument(
        "--phase-column",
        type=str,
        default=None,
        help="Column name for pre-computed orbital phase (e.g., 'phase'). "
             "If specified, uses this column instead of computing phase from time. "
             "Useful for CIAO files that already have phase computed.",
    )
    parser.add_argument(
        "--sim-column",
        type=str,
        nargs='+',
        default=None,
        help="Column name(s) in simulation CSV to use as model flux. Can specify multiple columns separated by spaces. "
             "If not specified, will auto-detect all available scaled flux columns (nfl_*, pho_count_*).",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Perform χ² minimization to fit simulation to observations.",
    )
    parser.add_argument(
        "--rescale",
        action="store_true",
        help="Optimize phase shift and flux scale to minimize χ². "
             "By default, chi-square is computed without rescaling (shift=0, scale=1).",
    )
    
    # Phase binning options
    parser.add_argument(
        "--n-phase-bins",
        type=int,
        default=50,
        help="Number of phase bins for binning the data (default: 50). "
             "Binning reduces scatter by averaging data within each phase bin.",
    )
    parser.add_argument(
        "--no-phase-bin",
        action="store_true",
        help="Disable phase binning and use raw data points instead.",
    )
    parser.add_argument(
        "--min-points-per-bin",
        type=int,
        default=3,
        help="Minimum number of data points required per bin (default: 3). "
             "Bins with fewer points are excluded.",
    )

    args = parser.parse_args()

    if args.verify_master:
        if not args.master_file:
            parser.error("--verify-master requires --master-file to be specified.")
        verify_master_contains_individual(args.data_dir, args.master_file)

    # Determine observation column to use
    obs_column = args.obs_column if args.obs_column else "rate"
    obs_error_column = args.obs_error_column
    time_column = args.time_column
    phase_column = args.phase_column
    
    if args.obs_column:
        print(f"Using observation column: {obs_column}")
        if obs_error_column:
            print(f"Using error column: {obs_error_column}")
        else:
            print(f"Error column will be auto-detected")
    if time_column:
        print(f"Using time column: {time_column}")
    if phase_column:
        print(f"Using pre-computed phase column: {phase_column}")
    
    df = load_data(args.data_dir, args.master_file, obs_column, obs_error_column, time_column, phase_column)
    print(f"Loaded {len(df)} data point(s) from {df['obs'].nunique()} observation(s).")
    
    # Remove observations with zero or NaN flux (gaps in observations)
    n_before = len(df)
    df = df[(df['rate'] != 0) & (df['rate'].notna())].reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"Removed {n_removed} zero/NaN flux data points ({len(df)} remaining)")
    
    # Show which columns are present in the loaded data
    if 'error' in df.columns:
        print(f"Using data column: '{obs_column}' (with error column)")
    else:
        print(f"Using data column: '{obs_column}' (no error column found)")
    
    # Apply phase binning if requested
    is_binned = False
    if not args.no_phase_bin:
        df = phase_bin_data(
            df,
            n_bins=args.n_phase_bins,
            min_points_per_bin=args.min_points_per_bin,
            rate_column='rate',
            error_column='error',
            verbose=True
        )
        is_binned = True

    if args.fit:
        if not args.sim_file:
            parser.error("--fit requires --sim-file to be specified.")
        
        print(f"Loading simulation file: {args.sim_file}")
        sim_df = pd.read_csv(args.sim_file)
        
        # Auto-detect or validate columns
        if args.sim_column is None:
            # Auto-detect all flux columns
            sim_columns = detect_flux_columns(sim_df)
            if not sim_columns:
                parser.error("No scaled flux columns found in simulation file. Expected columns like nfl_* or pho_count_*")
            print(f"Auto-detected {len(sim_columns)} scaled flux column(s): {sim_columns}")
        else:
            # Validate user-specified columns
            requested_columns = args.sim_column if isinstance(args.sim_column, list) else [args.sim_column]
            sim_columns = validate_sim_columns(sim_df, requested_columns)
            print(f"Using {len(sim_columns)} flux column(s): {sim_columns}")
        
        # Fit each column
        fit_results = []
        for col in sim_columns:
            print(f"\n{'='*60}")
            print(f"Fitting column: {col}")
            print('='*60)
            try:
                shift, scale, chi2 = fit_simulation(df, sim_df, col, rescale=args.rescale)
                fit_results.append((shift, scale, chi2))
            except Exception as e:
                print(f"⚠️  Failed to fit column '{col}': {e}")
                # Add dummy values so we can still plot other columns
                fit_results.append((0.0, 1.0, float('nan')))
        
        # Plot based on number of columns
        if len(sim_columns) == 1:
            # Single column: use original plot
            shift, scale, chi2 = fit_results[0]
            plot_phase(df, args.output, sim_df, shift, scale, sim_columns[0], chi2, 
                      rescaled=args.rescale, obs_column_name=obs_column, is_binned=is_binned)
        else:
            # Multiple columns: use grid plot
            plot_multi_column_fits(df, args.output, sim_df, sim_columns, fit_results, 
                                  rescaled=args.rescale, obs_column_name=obs_column, is_binned=is_binned)
    else:
        plot_phase(df, args.output, obs_column_name=obs_column, is_binned=is_binned)


if __name__ == "__main__":
    main() 