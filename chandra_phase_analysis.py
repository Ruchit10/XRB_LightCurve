#!/usr/bin/env python3
"""
X-ray Binary Phase Analysis
----------------------------
This script converts observational X-ray light-curve data to orbital phase and
fits simulation models to the observations.

Features:
1.  Reads all .txt files from a data directory
2.  Converts observation times to orbital phase using the reference epoch and
    orbital period
3.  Produces scatter plots of count-rate versus orbital phase
4.  Fits simulation models to observations via chi-square minimization
5.  Supports multiple energy bands and automatically detects available flux columns

File Format:
  Whitespace-delimited text files with three columns:
    1. time (seconds)
    2. count rate / flux
    3. error (optional)

Examples
~~~~~~~~
# Load all .txt files from a custom directory:
$ python chandra_phase_analysis.py --data-dir my_observations --output phase_plot.png

# Use specific observation column (e.g., NET_RATE instead of default):
$ python chandra_phase_analysis.py --data-dir data --obs-column NET_RATE --output plot.png

# Use FLUX column from observations with specific error column:
$ python chandra_phase_analysis.py --data-dir data --obs-column FLUX --obs-error-column FLUX_ERR --output plot.png

# Fit simulation to observations (auto-detects all flux columns):
$ python chandra_phase_analysis.py --data-dir data --fit --sim-file simulation.csv --output fit.png

# Fit specific flux columns:
$ python chandra_phase_analysis.py --data-dir data --fit --sim-file sim.csv \\
    --sim-column nfl_broad nfl_soft --output fit.png

# Fit with specific observation column, phase shift held at 0:
$ python chandra_phase_analysis.py --data-dir data --fit --sim-file sim.csv \\
    --obs-column FLUX --sim-column nfl_broad --output fit.png

# Fit the phase shift as well (flux normalization is never rescaled):
$ python chandra_phase_analysis.py --data-dir data --fit --sim-file sim.csv \\
    --obs-column NET_RATE --sim-column nfl_broad --fit-phase-shift --output fit.png

# Adaptive constant-counts binning (equal Poisson weight per point):
$ python chandra_phase_analysis.py --data-dir data/IC_10_X1_LC_CIAO/broad \\
    --obs-column flux_t --time-column t_raw --counts-per-bin 100 \\
    --fit --sim-file sim.csv --fit-phase-shift --output fit.png

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
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# -----------------------------------------------------------------------------
# Constants adopted from the R script (seconds)
# -----------------------------------------------------------------------------
REF_EPOCH: float = 278801348  # Reference time (t0) used for phase zero
# REF_EPOCH: float = 278800407.267 # corrected reference epoch from find_reference_epoch.py
ORBITAL_PERIOD: float = 125431  # Orbital period of the system


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def frac(x: np.ndarray | float) -> np.ndarray | float:
    """Return the fractional part of *x* (vectorised)."""
    return np.abs(x - np.floor(x))


def detect_flux_columns(df: pd.DataFrame) -> List[str]:
    """Detect available flux columns in simulation DataFrame.

    Looks for columns matching `nfl_{band}` (normalized flux per band,
    scaled by lam). With the unified wind model there is a single flux
    column per band (no `_av` / `_cv` split).

    Note: the unscaled `flx` and scaled `fl` column-density columns are
    excluded — they are not per-band flux values.

    Parameters
    ----------
    df : DataFrame
        Simulation results DataFrame

    Returns
    -------
    List of flux column names found in the DataFrame
    """
    flux_columns = [col for col in df.columns if col.startswith("nfl_")]
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


def _derive_err_from_rate_err(df: pd.DataFrame, obs_col: str) -> Optional[pd.Series]:
    """Derive observable errors from rate_err for proportional columns.

    This supports CIAO style files where ``flux_t`` exists but ``flux_t_err``
    does not. For rows with finite, positive ``rate`` we use:

        err_obs = rate_err * (obs / rate)

    For rows where that ratio is undefined, fall back to a robust file-level
    conversion factor median(obs/rate) computed from valid rows.
    """
    rate_col = None
    rate_err_col = None

    for col in df.columns:
        if col.upper() == "RATE":
            rate_col = col
            break

    for col in df.columns:
        if col.upper() in {"RATE_ERR", "ERR_RATE", "COUNT_RATE_ERR"}:
            rate_err_col = col
            break

    if rate_col is None or rate_err_col is None:
        return None

    obs_vals = pd.to_numeric(df[obs_col], errors="coerce")
    rate_vals = pd.to_numeric(df[rate_col], errors="coerce")
    rate_err_vals = pd.to_numeric(df[rate_err_col], errors="coerce")

    valid_ratio = (
        np.isfinite(obs_vals.to_numpy())
        & np.isfinite(rate_vals.to_numpy())
        & (rate_vals.to_numpy() > 0.0)
    )
    if not np.any(valid_ratio):
        return None

    ratio = np.full(len(df), np.nan, dtype=float)
    ratio[valid_ratio] = obs_vals.to_numpy()[valid_ratio] / rate_vals.to_numpy()[valid_ratio]
    cf = float(np.nanmedian(ratio[valid_ratio]))
    ratio = np.where(np.isfinite(ratio), ratio, cf)

    derived = rate_err_vals.to_numpy() * ratio
    return pd.Series(derived, index=df.index, dtype=float)


def read_observation(
    file_path: str,
    label: str,
    obs_column: str = "rate",
    obs_error_column: Optional[str] = None,
    time_column: Optional[str] = None,
    counts_column: Optional[str] = "counts",
) -> pd.DataFrame:
    """Read a single Chandra observation text file.

    The files can be whitespace-delimited with or without headers.
    If a header is present (lines starting with #), column names are extracted.
    Otherwise, assumes three columns: time, count rate/flux, error.
    
    Supports multiple formats:
    1. Standard format: TIME in first column, tab or space delimited
    2. CIAO format: time column (t_raw or time), space delimited, with "# Columns:" or "# #Columns:" header
    
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
    counts_column : str, optional
        Name of column containing counts. If present in the file, it is passed through
        to the output as ``counts``.
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
                    
                    # Need a time column to compute phase
                    if not time_col:
                        print(f"⚠️  Warning: No time column found in {file_path}")
                        print(f"   Available columns: {list(df.columns)}")
                        raise ValueError("No time column found")
                    
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
                    
                    # Add time column (required for phase computation)
                    result_df['time'] = df[time_col]
                    
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
                    else:
                        derived = _derive_err_from_rate_err(df, actual_obs_column)
                        if derived is not None:
                            result_df['error'] = derived

                    if counts_column:
                        actual_counts_column = None
                        for col in df.columns:
                            if col.upper() == counts_column.upper():
                                actual_counts_column = col
                                break
                        if actual_counts_column:
                            result_df['counts'] = pd.to_numeric(
                                df[actual_counts_column], errors='coerce'
                            )
                    
                    # Always compute phase from timestamps and current ephemeris.
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

def load_data(
    data_dir: str,
    obs_column: str = "rate",
    obs_error_column: Optional[str] = None,
    time_column: Optional[str] = None,
    counts_column: Optional[str] = "counts",
) -> pd.DataFrame:
    """Load observational data from *data_dir*.

    Parameters
    ----------
    data_dir : str
        Directory containing observation text files
    obs_column : str, default "rate"
        Name of column to use for the observable (e.g., "NET_RATE", "FLUX", "COUNT_RATE", "ECF", "flux_t")
    obs_error_column : str, optional
        Name of column to use for errors. If None, will auto-detect based on obs_column.
    time_column : str, optional
        Name of column containing timestamps. If None, will auto-detect (looks for 'time', 't_raw').
    counts_column : str, optional
        Name of column containing counts. If present, propagated into the
        combined output as ``counts``.
    Returns
    -------
    DataFrame with columns: time, rate (containing the specified observable), error (optional), phase, obs
    """
    # Load all .txt files from directory
    txt_pattern = os.path.join(data_dir, "*.txt")
    files: List[str] = sorted(glob.glob(txt_pattern))
    
    if not files:
        raise FileNotFoundError(
            f"No .txt files found in {data_dir}"
        )

    print(f"Loading {len(files)} observation file(s) from {data_dir}")
    dfs = [
        read_observation(
            fp,
            os.path.basename(fp),
            obs_column,
            obs_error_column,
            time_column,
            counts_column=counts_column,
        )
        for fp in files
    ]
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


def phase_bin_data_snr(
    df: pd.DataFrame,
    counts_per_bin: int = 100,
    counts_column: str = 'counts',
    rate_column: str = 'rate',
    error_column: str = 'error',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Adaptive phase binning with approximately constant counts per bin.

    Points are sorted by phase and grouped greedily until each bin reaches
    ``counts_per_bin`` total counts, yielding variable phase-width bins.

    Parameters
    ----------
    df : DataFrame
        Observed data with at least ``phase``, *rate_column*, and *counts_column*.
    counts_per_bin : int
        Target counts per bin.
    counts_column : str
        Name of column containing counts.
    rate_column : str
        Name of column containing flux/rate values.
    error_column : str
        Name of column containing error values.
    verbose : bool
        Print summary of the binning operation.

    Returns
    -------
    DataFrame
        Columns: phase, rate, error, n_points, total_counts, phase_lo, phase_hi, width
    """
    if counts_per_bin <= 0:
        raise ValueError("counts_per_bin must be > 0")
    if counts_column not in df.columns:
        raise ValueError(f"counts column '{counts_column}' not found in DataFrame")
    if rate_column not in df.columns:
        raise ValueError(f"rate column '{rate_column}' not found in DataFrame")

    work = df.copy()
    work = work[np.isfinite(work['phase']) & np.isfinite(work[rate_column])].copy()
    if work.empty:
        return pd.DataFrame(
            columns=['phase', 'rate', 'error', 'n_points', 'total_counts', 'phase_lo', 'phase_hi', 'width']
        )

    work[counts_column] = pd.to_numeric(work[counts_column], errors='coerce').fillna(0.0)
    work[counts_column] = np.where(work[counts_column] > 0.0, work[counts_column], 0.0)
    work = work.sort_values('phase').reset_index(drop=True)

    target = float(counts_per_bin)
    bins: List[List[int]] = []
    current: List[int] = []
    current_counts = 0.0
    counts_vals = work[counts_column].to_numpy(dtype=float)

    for i, c in enumerate(counts_vals):
        current.append(i)
        current_counts += c
        if current_counts >= target:
            bins.append(current)
            current = []
            current_counts = 0.0

    if current:
        bins.append(current)

    if len(bins) >= 2:
        tail_counts = float(np.sum(counts_vals[bins[-1]]))
        if tail_counts < target:
            bins[-2].extend(bins[-1])
            bins.pop()

    binned_data = []
    for indices in bins:
        bin_df = work.iloc[indices]
        rate_vals = bin_df[rate_column].to_numpy(dtype=float)
        err_vals = (
            bin_df[error_column].to_numpy(dtype=float)
            if (error_column in bin_df.columns)
            else np.full(len(bin_df), np.nan, dtype=float)
        )
        has_errors = np.any(np.isfinite(err_vals))

        if has_errors:
            valid_err = err_vals[(err_vals > 0) & np.isfinite(err_vals)]
            if len(valid_err) > 0:
                median_err = float(np.median(valid_err))
                err_vals = np.where(
                    (err_vals <= 0) | ~np.isfinite(err_vals),
                    median_err,
                    err_vals,
                )
            else:
                fallback = np.std(rate_vals) if len(rate_vals) > 1 else np.abs(rate_vals[0]) * 0.1
                err_vals = np.full_like(rate_vals, max(float(fallback), np.finfo(float).eps))
            weights = 1.0 / err_vals ** 2
            mean_rate = float(np.average(rate_vals, weights=weights))
            mean_err = float(np.sqrt(1.0 / np.sum(weights)))
        else:
            mean_rate = float(np.mean(rate_vals))
            mean_err = float(np.std(rate_vals) / np.sqrt(len(rate_vals))) if len(rate_vals) > 1 else 0.0

        phase_vals = bin_df['phase'].to_numpy(dtype=float)
        bin_counts = np.maximum(bin_df[counts_column].to_numpy(dtype=float), 0.0)
        total_counts = float(np.sum(bin_counts))
        if total_counts > 0:
            phase_center = float(np.average(phase_vals, weights=bin_counts))
        else:
            phase_center = float(np.mean(phase_vals))

        phase_lo = float(np.min(phase_vals))
        phase_hi = float(np.max(phase_vals))
        width = float(max(phase_hi - phase_lo, 0.0))

        binned_data.append(
            {
                'phase': phase_center,
                'rate': mean_rate,
                'error': mean_err,
                'n_points': int(len(bin_df)),
                'total_counts': total_counts,
                'phase_lo': phase_lo,
                'phase_hi': phase_hi,
                'width': width,
            }
        )

    result = pd.DataFrame(binned_data)
    if 'obs' in work.columns:
        result['obs'] = 'binned'

    if verbose:
        avg_counts = float(np.mean(result['total_counts'])) if len(result) > 0 else 0.0
        print(
            f"Adaptive phase binning: {len(work)} points -> {len(result)} bins "
            f"(target {counts_per_bin} counts/bin, avg {avg_counts:.1f})"
        )
    return result


# -----------------------------------------------------------------------------
# Smoothing and residual helpers
# -----------------------------------------------------------------------------

def smooth_lightcurve(
    phase: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    sigma: float = 0.01,
    eval_phase: Optional[np.ndarray] = None,
    n_eval: int = 300,
    n_mc: int = 2000,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Periodic Gaussian-kernel smoothing with optional MC uncertainty band."""
    phase = np.asarray(phase, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if phase.shape != flux.shape:
        raise ValueError("phase and flux must have identical shapes.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if n_eval <= 0:
        raise ValueError("n_eval must be > 0.")
    if n_mc < 0:
        raise ValueError("n_mc must be >= 0.")

    valid = np.isfinite(phase) & np.isfinite(flux)
    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)
        if flux_err.shape != flux.shape:
            raise ValueError("flux_err must match phase/flux shape.")
        valid &= np.isfinite(flux_err)
        flux_err = np.where(flux_err > 0.0, flux_err, np.nan)

    phase = np.mod(phase[valid], 1.0)
    flux = flux[valid]
    if flux_err is not None:
        flux_err = flux_err[valid]

    if phase.size == 0:
        return pd.DataFrame(
            {"phase": np.array([]), "flux_smooth": np.array([]), "flux_smooth_err": np.array([])}
        )

    if eval_phase is None:
        eval_phase = np.linspace(0.0, 1.0, int(n_eval), endpoint=False, dtype=float)
    else:
        eval_phase = np.mod(np.asarray(eval_phase, dtype=float), 1.0)

    d = np.abs(np.mod(phase[None, :] - eval_phase[:, None] + 0.5, 1.0) - 0.5)
    w = np.exp(-0.5 * (d / float(sigma)) ** 2)
    wsum = w.sum(axis=1)
    flux_smooth = np.full(eval_phase.shape, np.nan, dtype=float)
    good = wsum > 0.0
    if np.any(good):
        flux_smooth[good] = (w[good] @ flux) / wsum[good]

    flux_smooth_err = np.full(eval_phase.shape, np.nan, dtype=float)
    if flux_err is not None and n_mc > 0:
        mc_valid = np.isfinite(flux_err)
        if np.any(mc_valid):
            rng = np.random.default_rng(random_state)
            perturbed = (
                flux[mc_valid][None, :]
                + flux_err[mc_valid][None, :] * rng.standard_normal((int(n_mc), int(np.sum(mc_valid))))
            )
            w_mc = w[:, mc_valid]
            wsum_mc = w_mc.sum(axis=1)
            good_mc = wsum_mc > 0.0
            if np.any(good_mc):
                smoothed_mc = (perturbed @ w_mc[good_mc].T) / wsum_mc[good_mc][None, :]
                flux_smooth_err[good_mc] = smoothed_mc.std(axis=0)

    if verbose:
        print(
            f"Smoothing: {phase.size} points, sigma={sigma:.4f}, "
            f"eval={eval_phase.size}, n_mc={int(n_mc)}"
        )

    return pd.DataFrame(
        {
            "phase": eval_phase,
            "flux_smooth": flux_smooth,
            "flux_smooth_err": flux_smooth_err,
        }
    )


def estimate_scattered_flux(
    phase: np.ndarray,
    flux: np.ndarray,
    window: tuple[float, float] = (0.4, 0.6),
) -> float:
    """Estimate eclipse-floor flux using the mean value in a phase window."""
    lo, hi = float(window[0]), float(window[1])
    if not (0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0 and lo <= hi):
        raise ValueError("window must satisfy 0 <= lo <= hi <= 1.")
    phase = np.mod(np.asarray(phase, dtype=float), 1.0)
    flux = np.asarray(flux, dtype=float)
    valid = np.isfinite(phase) & np.isfinite(flux)
    if not np.any(valid):
        return 0.0
    phase = phase[valid]
    flux = flux[valid]
    mask = (phase >= lo) & (phase <= hi)
    if np.any(mask):
        val = float(np.nanmean(flux[mask]))
    else:
        val = float(np.nanmedian(flux)) * 0.1
    return max(val, 0.0) if np.isfinite(val) else 0.0


def add_residual_panel(
    ax_res: plt.Axes,
    phase: np.ndarray,
    obs: np.ndarray,
    model: np.ndarray,
    err: np.ndarray,
    xerr: Optional[np.ndarray] = None,
) -> None:
    """Draw normalized residuals (O-M)/sigma with reference lines."""
    denom = np.where(np.asarray(err, dtype=float) > 0.0, np.asarray(err, dtype=float), np.nan)
    resid = (np.asarray(obs, dtype=float) - np.asarray(model, dtype=float)) / denom
    ax_res.errorbar(
        np.asarray(phase, dtype=float),
        resid,
        xerr=xerr,
        fmt='o',
        markersize=4,
        alpha=0.8,
        color='C3',
        capsize=2,
        elinewidth=1,
    )
    ax_res.axhline(0.0, color='k', linewidth=1.0)
    ax_res.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    ax_res.axhline(-1.0, color='gray', linestyle='--', linewidth=0.8)
    ax_res.set_ylabel(r'$(O-M)/\sigma$')
    ax_res.set_xlabel("Orbital phase")
    ax_res.grid(alpha=0.3)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _prepare_model_interpolator(
    sim_df: pd.DataFrame, sim_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """Build wrap-around interpolation arrays for a simulation light curve.

    Returns ``(phase_wrap, flux_wrap)`` covering [0, 2) in phase with strictly
    increasing x, so ``np.interp`` handles the periodic boundary correctly.
    """
    if "phase" in sim_df.columns:
        sim_phase = np.mod(sim_df["phase"].to_numpy(dtype=float), 1.0)
    elif "deg" in sim_df.columns:
        sim_phase = np.mod(sim_df["deg"].to_numpy(dtype=float) % 360.0, 360.0) / 360.0
    else:
        raise ValueError("Simulation file must contain 'phase' or 'deg' column.")

    if sim_column not in sim_df.columns:
        raise KeyError(f"Column '{sim_column}' not found in simulation DataFrame.")
    sim_flux = sim_df[sim_column].to_numpy(dtype=float)

    order = np.argsort(sim_phase)
    p = sim_phase[order]
    f = sim_flux[order]
    phase_wrap = np.concatenate([p, p + 1.0])
    flux_wrap = np.concatenate([f, f])

    # Strictly increasing x only, to avoid np.interp ambiguity at duplicates.
    keep = np.concatenate(([True], np.diff(phase_wrap) > 0))
    return phase_wrap[keep], flux_wrap[keep]


def _model_from_wrap(
    phase_wrap: np.ndarray,
    flux_wrap: np.ndarray,
    phases,
    shift=0.0,
    scatter: float = 0.0,
) -> np.ndarray:
    """Evaluate a prepared model at *phases* for a given shift and scatter.

    This is the single definition of the model used everywhere: the χ² in
    :func:`fit_simulation`, the overlay curve in :func:`plot_phase`, and the
    residual panel all route through it, so they cannot silently disagree.
    *shift* may be an array (broadcast against *phases*) to evaluate many trial
    shifts at once.
    """
    ph = np.mod(
        np.asarray(phases, dtype=float) - np.asarray(shift, dtype=float), 1.0
    )
    out = np.interp(ph.ravel(), phase_wrap, flux_wrap).reshape(ph.shape)
    return out + float(scatter)


def evaluate_model_at_phases(
    sim_df: pd.DataFrame,
    sim_column: str,
    phases,
    shift: float = 0.0,
    scatter: float = 0.0,
) -> np.ndarray:
    """Model flux at *phases* for a given phase shift and additive scatter.

    Convenience wrapper over :func:`_prepare_model_interpolator` +
    :func:`_model_from_wrap` for callers that only need a single evaluation.
    """
    phase_wrap, flux_wrap = _prepare_model_interpolator(sim_df, sim_column)
    return _model_from_wrap(phase_wrap, flux_wrap, phases, shift, scatter)


def _obs_errors(
    obs_df: pd.DataFrame,
    rate_column: str = "rate",
    error_column: str = "error",
) -> np.ndarray:
    """Observation uncertainties, with the χ²-safety guards applied.

    Uses the provided errors when available, otherwise sqrt(|rate|); zero,
    negative and non-finite values are floored so they cannot blow up χ².
    Shared by :func:`fit_simulation` and :func:`plot_phase` so both weight the
    data identically.
    """
    rate = obs_df[rate_column].to_numpy(dtype=float)
    if error_column in obs_df.columns and not obs_df[error_column].isnull().all():
        err = obs_df[error_column].to_numpy(dtype=float)
    else:
        err = np.sqrt(np.abs(rate))
    return np.where((err <= 0) | ~np.isfinite(err), 1e-3, err)


def fit_simulation(
    obs_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    sim_column: str = "fl",
    fit_phase_shift: bool = False,
    scatter: float = 0.0,
    n_shift_grid: int = 1000,
) -> tuple[float, float]:
    """Fit simulation light-curve to observations via chi-square minimization.

    Only the **phase shift** (x-direction) is fitted. There is deliberately no
    multiplicative flux scale: the model's absolute normalization is already
    fixed by ``lam`` (the orbit-averaged nH from the spectral fit) together with
    the XSPEC ``flux vs nH`` table, so a free y-scale would silently absorb an
    error in that normalization instead of exposing it. The only y-direction
    freedom is the *additive* ``scatter`` floor, which is supplied by the caller
    (measured at mid-eclipse) rather than fitted here. This matches
    ``mcmc_lightcurve_fit.py``, which likewise fits a per-sample phase shift and
    an additive ``f_scatter`` but no multiplicative scale.

    Parameters
    ----------
    obs_df : DataFrame
        Observational data with columns ``phase``, ``rate`` and (optionally) ``error``.
    sim_df : DataFrame
        Simulation results. Must contain columns ``phase`` (or ``deg``) and *sim_column*.
    sim_column : str, default ``"fl"``
        Column in *sim_df* to use as the model flux.
    fit_phase_shift : bool, default False
        If True, scan the phase shift that minimizes chi-square.
        If False, evaluate chi-square at shift = 0.
    scatter : float, default 0.0
        Constant additive scattered-flux floor added to the model. Added *after*
        interpolation and never scaled.
    n_shift_grid : int, default 1000
        Number of trial shifts in the coarse scan over [0, 1). The scan is
        followed by a bounded local refinement, so this only needs to be fine
        enough to land in the correct basin.

    Returns
    -------
    (phase_shift, reduced_chi2)
        Best-fit phase shift (0–1) and reduced chi-squared value.
        If *fit_phase_shift* is False, returns ``(0.0, reduced_chi2)``.
    """
    # Prepare observation arrays
    phase_obs = obs_df["phase"].to_numpy()
    rate_obs = obs_df["rate"].to_numpy(dtype=float)
    err_obs = _obs_errors(obs_df)

    # Prepared once; every model evaluation below goes through _model_from_wrap
    # so the χ² here and the curve drawn by plot_phase are the same function.
    phase_wrap, flux_wrap = _prepare_model_interpolator(sim_df, sim_column)
    scatter = float(scatter)

    def chi2(shift) -> float:
        model = _model_from_wrap(phase_wrap, flux_wrap, phase_obs, shift, scatter)
        return float(np.sum(((rate_obs - model) / err_obs) ** 2))

    if fit_phase_shift:
        # The phase shift is periodic and the eclipse profile makes chi2(shift)
        # strongly multi-modal, so a local optimizer started at shift=0 would
        # routinely settle in the wrong basin. Scan a coarse grid over the full
        # period first, then refine locally around the best node. This mirrors
        # the two-stage search in mcmc_lightcurve_fit._apply_best_phase_shift.
        n_grid = max(3, int(n_shift_grid))
        shift_grid = np.linspace(0.0, 1.0, n_grid, endpoint=False)

        # Vectorized coarse scan: one interp over all (shift, obs_phase) pairs.
        models = _model_from_wrap(
            phase_wrap, flux_wrap, phase_obs[None, :], shift_grid[:, None], scatter
        )
        chi2_grid = np.sum(
            ((rate_obs[None, :] - models) / err_obs[None, :]) ** 2, axis=1
        )
        best_idx = int(np.argmin(chi2_grid))
        best_shift = float(shift_grid[best_idx])
        best_chi2 = float(chi2_grid[best_idx])

        # Bounded local refinement within one coarse step of the best node.
        step = 1.0 / n_grid
        refined = minimize_scalar(
            chi2,
            bounds=(best_shift - step, best_shift + step),
            method="bounded",
        )
        if refined.success and float(refined.fun) < best_chi2:
            best_shift = float(refined.x) % 1.0
            best_chi2 = float(refined.fun)

        n_free = 1
        reduced_chi2 = best_chi2 / max(len(rate_obs) - n_free, 1)
        print(
            f"Best-fit parameters (phase shift only, no flux rescaling):\n"
            f"  Phase shift = {best_shift:.5f}\n"
            f"  Scattered flux = {scatter:.6g} (fixed, additive)\n"
            f"  Reduced χ² = {reduced_chi2:.3f}  (dof = {max(len(rate_obs) - n_free, 1)})"
        )
        return float(best_shift), float(reduced_chi2)

    # No optimization: evaluate chi-square at zero shift.
    best_shift = 0.0
    n_free = 0
    reduced_chi2 = chi2(best_shift) / max(len(rate_obs) - n_free, 1)
    print(
        f"Chi-square (no phase-shift fit, no flux rescaling):\n"
        f"  Phase shift = {best_shift:.5f} (fixed)\n"
        f"  Scattered flux = {scatter:.6g} (fixed, additive)\n"
        f"  Reduced χ² = {reduced_chi2:.3f}  (dof = {max(len(rate_obs) - n_free, 1)})"
    )
    return float(best_shift), float(reduced_chi2)


def plot_phase(
    df: pd.DataFrame,
    output_path: str | None,
    sim_df: pd.DataFrame | None = None,
    shift: float | None = None,
    sim_column: str = "fl",
    chi2: float | None = None,
    ax: plt.Axes | None = None,
    shift_fitted: bool = False,
    obs_column_name: str = "rate",
    is_binned: bool = False,
    smooth_df: Optional[pd.DataFrame] = None,
    scatter: float = 0.0,
) -> None:
    """Scatter plot with optional best-fit simulation overlay.

    The model overlay is drawn at its native flux normalization (set by ``lam``
    and the XSPEC flux-vs-nH table); the only y-direction adjustment is the
    additive *scatter* floor. There is no multiplicative scale factor — see
    :func:`fit_simulation`.

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
    sim_column : str, default ``"fl"``
        Column name in simulation to use.
    chi2 : float, optional
        Reduced chi-squared value to annotate on plot.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure.
    shift_fitted : bool, default False
        Whether the phase shift was optimized (True) or held at 0 (False).
        Only affects the title annotation.
    obs_column_name : str, default "rate"
        Name of the observable column being plotted (for labeling).
    is_binned : bool, default False
        Whether the data has been phase-binned. If True, plots with error bars.
    scatter : float, default 0.0
        Constant additive scattered-flux floor added to the model overlay.
    """
    has_model = sim_df is not None and shift is not None
    owns_figure = ax is None
    ax_res: Optional[plt.Axes] = None
    if ax is None:
        if has_model:
            fig, (ax, ax_res) = plt.subplots(
                2,
                1,
                figsize=(10, 8),
                sharex=True,
                gridspec_kw={'height_ratios': [3, 1]},
            )
        else:
            fig, ax = plt.subplots(figsize=(10, 6))

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

    model_wrap = None
    if has_model:
        # Overlay and residuals both come from the same evaluator used by
        # fit_simulation's χ², so the drawn curve, the residual panel and the
        # displayed χ² are guaranteed to describe the same model.
        model_wrap = _prepare_model_interpolator(sim_df, sim_column)
        phase_overlay = np.linspace(0.0, 1.0, 721)
        flux_overlay = _model_from_wrap(
            *model_wrap, phase_overlay, shift, scatter
        )
        ax.plot(phase_overlay, flux_overlay, "k-", linewidth=2, label="Best-fit model")

    if smooth_df is not None and len(smooth_df) > 0:
        sphase = np.mod(smooth_df["phase"].to_numpy(dtype=float), 1.0)
        sflux = smooth_df["flux_smooth"].to_numpy(dtype=float)
        serr = smooth_df["flux_smooth_err"].to_numpy(dtype=float)
        order = np.argsort(sphase)
        sphase = sphase[order]
        sflux = sflux[order]
        serr = serr[order]
        ax.plot(
            sphase,
            sflux,
            '--',
            color='green',
            linewidth=1.5,
            label='Gaussian-smoothed data',
            zorder=7,
        )
        if np.any(np.isfinite(serr)):
            ax.fill_between(
                sphase,
                sflux - serr,
                sflux + serr,
                color='green',
                alpha=0.2,
                label='Smoothed 1σ (MC)',
                zorder=3,
            )

    ylabel = obs_column_name.replace("_", " ").title() if obs_column_name != "rate" else "Count rate / Flux"
    ax.set_ylabel(ylabel)
    
    # Set title with chi-squared annotation if provided
    if chi2 is not None:
        if shift_fitted:
            shift_label = f" (phase shift = {shift:.4f}, fitted)"
        else:
            shift_label = " (phase shift = 0, fixed)"
        ax.set_title(f"{sim_column}\nReduced χ² = {chi2:.3f}{shift_label}")
    else:
        ax.set_title("Chandra Light-curve Observations")
    
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")

    if has_model and model_wrap is not None:
        obs_phase = np.mod(df["phase"].to_numpy(dtype=float), 1.0)
        obs_rate = df["rate"].to_numpy(dtype=float)
        obs_err = _obs_errors(df)
        # Same shift and scatter as the overlay above, so residuals match the curve.
        obs_model = _model_from_wrap(*model_wrap, obs_phase, shift, scatter)

        # Self-check: the χ² we display must be the χ² of the model we drew.
        # This catches a `scatter` or `shift` that disagrees with the
        # fit_simulation call, which would otherwise show a correct-looking
        # number over the wrong curve.
        if chi2 is not None and np.isfinite(chi2):
            n_free = 1 if shift_fitted else 0
            recomputed = float(
                np.sum(((obs_rate - obs_model) / obs_err) ** 2)
                / max(len(obs_rate) - n_free, 1)
            )
            if np.isfinite(recomputed) and abs(recomputed - chi2) > 0.01 * max(
                abs(chi2), 1e-300
            ):
                warnings.warn(
                    f"plot_phase: displayed reduced chi2 ({chi2:.4g}) does not match "
                    f"the plotted model ({recomputed:.4g}). The `shift`/`scatter` "
                    f"passed here probably differ from the fit_simulation call "
                    f"(scatter={scatter!r}, shift={shift!r}).",
                    stacklevel=2,
                )

        if ax_res is not None and has_errors:
            xerr = None
            if is_binned and "width" in df.columns:
                width = np.asarray(df["width"].to_numpy(dtype=float), dtype=float)
                if width.shape == obs_phase.shape:
                    xerr = 0.5 * np.clip(width, 0.0, np.inf)
            add_residual_panel(ax_res, obs_phase, obs_rate, obs_model, obs_err, xerr=xerr)
        else:
            ax.set_xlabel("Orbital phase")
    else:
        ax.set_xlabel("Orbital phase")

    if owns_figure and output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    elif owns_figure:
        plt.tight_layout()
        plt.show()
    elif output_path:
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_path}")


def plot_multi_column_fits(
    df: pd.DataFrame,
    output_path: str | None,
    sim_df: pd.DataFrame,
    sim_columns: List[str],
    fit_results: List[tuple[float, float]],
    shift_fitted: bool = False,
    obs_column_name: str = "rate",
    is_binned: bool = False,
    smooth_df: Optional[pd.DataFrame] = None,
    scatter: float = 0.0,
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
        List of ``(shift, chi2)`` tuples for each column.
    shift_fitted : bool, default False
        Whether the phase shifts were optimized or held at 0.
    obs_column_name : str, default "rate"
        Name of the observable column being plotted (for labeling).
    is_binned : bool, default False
        Whether the data has been phase-binned.
    scatter : float, default 0.0
        Constant additive scattered-flux floor added to each model overlay.
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
    
    for i, (col, (shift, chi2)) in enumerate(zip(sim_columns, fit_results)):
        ax = axes[i]
        plot_phase(
            df, None, sim_df, shift, col, chi2, ax, shift_fitted,
            obs_column_name, is_binned, smooth_df=smooth_df, scatter=scatter,
        )
    
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
        "--output",
        type=str,
        default=None,
        help="Output filename for the generated plot. If omitted, the plot is shown interactively.",
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
        "--sim-column",
        type=str,
        nargs='+',
        default=None,
        help="Column name(s) in simulation CSV to use as model flux. Can specify multiple columns separated by spaces. "
             "If not specified, will auto-detect all available scaled flux columns (nfl_*).",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Perform χ² minimization to fit simulation to observations.",
    )
    parser.add_argument(
        "--fit-phase-shift",
        "--rescale",
        dest="fit_phase_shift",
        action="store_true",
        help="Optimize the model phase shift to minimize χ². By default the "
             "shift is held at 0. Flux is never rescaled: the model's absolute "
             "normalization comes from --lam and the XSPEC flux-vs-nH table, and "
             "the only y-direction freedom is the additive --scatter floor. "
             "(--rescale is accepted as a deprecated alias.)",
    )
    
    # Phase binning options. As in mcmc_lightcurve_fit.py, the mode is selected
    # by which argument is present rather than by a separate --bin-mode flag.
    parser.add_argument(
        "--n-phase-bins",
        type=int,
        default=None,
        help="Use fixed-width phase binning with this many bins (variable counts "
             "per bin). Mutually exclusive with --counts-per-bin. If neither "
             "binning option is given, defaults to 50 fixed-width bins.",
    )
    parser.add_argument(
        "--counts-per-bin",
        type=int,
        default=None,
        help="Use adaptive phase binning with approximately constant counts per "
             "bin (variable phase width), giving every binned point equal "
             "Poisson weight. Requires a 'counts' column in the data. Mutually "
             "exclusive with --n-phase-bins. Recommended value: 100.",
    )
    parser.add_argument(
        "--no-phase-bin",
        action="store_true",
        help="Disable phase binning and use raw data points instead. Takes "
             "precedence over both binning options.",
    )
    parser.add_argument(
        "--min-points-per-bin",
        type=int,
        default=3,
        help="Minimum number of data points required per bin (default: 3). "
             "Bins with fewer points are excluded. Fixed-width binning only.",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Overlay a Gaussian-smoothed reference curve of the observed data.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.01,
        help="Gaussian kernel width in phase units for smoothing.",
    )
    parser.add_argument(
        "--smooth-n-mc",
        type=int,
        default=2000,
        help="Number of Monte Carlo perturbations for smoothing uncertainty (0 disables band).",
    )
    parser.add_argument(
        "--smooth-seed",
        type=int,
        default=None,
        help="RNG seed for smoothing Monte Carlo perturbations.",
    )
    parser.add_argument(
        "--scatter",
        type=float,
        default=None,
        help="Constant additive scattered flux term. If omitted during --fit, it is estimated from eclipse phase.",
    )
    parser.add_argument(
        "--scatter-eclipse-phase",
        nargs=2,
        type=float,
        default=(0.4, 0.6),
        metavar=("PHASE_MIN", "PHASE_MAX"),
        help="Phase window used to estimate scattered flux when --scatter is not provided.",
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

    # Determine observation column to use
    obs_column = args.obs_column if args.obs_column else "rate"
    obs_error_column = args.obs_error_column
    time_column = args.time_column
    
    if args.obs_column:
        print(f"Using observation column: {obs_column}")
        if obs_error_column:
            print(f"Using error column: {obs_error_column}")
        else:
            print(f"Error column will be auto-detected")
    if time_column:
        print(f"Using time column: {time_column}")
    df = load_data(
        args.data_dir,
        obs_column=obs_column,
        obs_error_column=obs_error_column,
        time_column=time_column,
    )
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
    
    # Apply phase binning if requested. Mode is chosen by argument presence:
    # --no-phase-bin > --counts-per-bin > --n-phase-bins > 50 fixed-width bins.
    is_binned = False
    if not args.no_phase_bin:
        if args.counts_per_bin is not None:
            if 'counts' not in df.columns:
                parser.error(
                    "--counts-per-bin requires a 'counts' column in the input "
                    "files (present in CIAO-format light curves). Use "
                    "--n-phase-bins for fixed-width binning instead."
                )
            df = phase_bin_data_snr(
                df,
                counts_per_bin=args.counts_per_bin,
                counts_column='counts',
                rate_column='rate',
                error_column='error',
                verbose=True,
            )
        else:
            df = phase_bin_data(
                df,
                n_bins=(args.n_phase_bins or 50),
                min_points_per_bin=args.min_points_per_bin,
                rate_column='rate',
                error_column='error',
                verbose=True
            )
        is_binned = True

    if args.fit:
        if not args.sim_file:
            parser.error("--fit requires --sim-file to be specified.")
        
        if args.scatter is not None:
            scatter_value = float(args.scatter)
            print(f"Using fixed scattered flux: {scatter_value:.6g}")
        else:
            scatter_value = estimate_scattered_flux(
                df["phase"].to_numpy(dtype=float),
                df["rate"].to_numpy(dtype=float),
                window=(float(args.scatter_eclipse_phase[0]), float(args.scatter_eclipse_phase[1])),
            )
            print(f"Estimated scattered flux from eclipse window: {scatter_value:.6g}")

        smooth_df = None
        if args.smooth:
            smooth_df = smooth_lightcurve(
                df["phase"].to_numpy(dtype=float),
                df["rate"].to_numpy(dtype=float),
                df["error"].to_numpy(dtype=float) if "error" in df.columns else None,
                sigma=float(args.smooth_sigma),
                n_mc=int(args.smooth_n_mc),
                random_state=args.smooth_seed,
                verbose=True,
            )

        print(f"Loading simulation file: {args.sim_file}")
        sim_df = pd.read_csv(args.sim_file)
        
        # Auto-detect or validate columns
        if args.sim_column is None:
            # Auto-detect all flux columns
            sim_columns = detect_flux_columns(sim_df)
            if not sim_columns:
                parser.error("No scaled flux columns found in simulation file. Expected columns like nfl_*")
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
                shift, chi2 = fit_simulation(
                    df, sim_df, col,
                    fit_phase_shift=args.fit_phase_shift,
                    scatter=scatter_value,
                )
                fit_results.append((shift, chi2))
            except Exception as e:
                print(f"⚠️  Failed to fit column '{col}': {e}")
                # Add dummy values so we can still plot other columns
                fit_results.append((0.0, float('nan')))

        # Plot based on number of columns
        if len(sim_columns) == 1:
            # Single column: use original plot
            shift, chi2 = fit_results[0]
            plot_phase(
                df, args.output, sim_df, shift, sim_columns[0], chi2,
                shift_fitted=args.fit_phase_shift, obs_column_name=obs_column,
                is_binned=is_binned, smooth_df=smooth_df, scatter=scatter_value,
            )
        else:
            # Multiple columns: use grid plot
            plot_multi_column_fits(
                df, args.output, sim_df, sim_columns, fit_results,
                shift_fitted=args.fit_phase_shift, obs_column_name=obs_column,
                is_binned=is_binned, smooth_df=smooth_df, scatter=scatter_value,
            )
    else:
        smooth_df = None
        if args.smooth:
            smooth_df = smooth_lightcurve(
                df["phase"].to_numpy(dtype=float),
                df["rate"].to_numpy(dtype=float),
                df["error"].to_numpy(dtype=float) if "error" in df.columns else None,
                sigma=float(args.smooth_sigma),
                n_mc=int(args.smooth_n_mc),
                random_state=args.smooth_seed,
                verbose=True,
            )
        plot_phase(
            df,
            args.output,
            obs_column_name=obs_column,
            is_binned=is_binned,
            smooth_df=smooth_df,
        )


if __name__ == "__main__":
    main() 