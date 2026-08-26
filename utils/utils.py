#!/usr/bin/env python3
"""
Shared non-plotting helpers for the XRB light-curve codebase.
-------------------------------------------------------------
Every routine here was previously defined (and in some cases duplicated) inside
``chandra_phase_analysis.py`` / ``mcmc_lightcurve_fit.py``. Both scripts now
import from this module, so there is a single implementation of:

* the ephemeris (``REF_EPOCH``, ``ORBITAL_PERIOD``) and :func:`frac`
* observation loading (:func:`read_observation`, :func:`load_data`) and
  simulation-column discovery (:func:`detect_flux_columns`,
  :func:`validate_sim_columns`)
* phase binning -- fixed-width (:func:`phase_bin_data`) and adaptive
  constant-counts (:func:`phase_bin_data_snr`)
* Gaussian phase smoothing (:func:`smooth_lightcurve`) and the eclipse-floor
  estimate (:func:`estimate_scattered_flux`)
* periodic model interpolation (:func:`prepare_model_interpolator`,
  :func:`model_from_wrap`, :func:`evaluate_model_at_phases`,
  :func:`interp_periodic_phases`)
* the tabulated-model χ² fit (:func:`fit_simulation`) and the periodic
  phase-shift search it shares with the MCMC likelihood
  (:func:`build_phase_shift_terms`, :func:`apply_best_phase_shift`)
* band-directory observation loading (:func:`resolve_band_directory`,
  :func:`load_observed_lightcurves`) and :func:`save_samples_csv_chunked`
* CLI run-config persistence (:func:`save_run_config`,
  :func:`apply_saved_run_config`), which lets ``--replot`` reproduce a fit's
  options without retyping them

This module deliberately depends only on the standard library plus numpy /
pandas / scipy: it must stay importable from either analysis script without
creating an import cycle.

Dependencies: numpy, pandas, scipy (in requirements.txt).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shlex
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# -----------------------------------------------------------------------------
# Constants adopted from the R script (seconds)
# -----------------------------------------------------------------------------
REF_EPOCH: float = 278801348  # Reference time (t0) used for phase zero
# REF_EPOCH: float = 278800407.267 # corrected reference epoch from find_reference_epoch.py
ORBITAL_PERIOD: float = 125431  # Orbital period of the system

# Two-stage periodic phase-shift search: coarse grid, then local refinement.
DEFAULT_PHASE_SHIFT_GRID_SIZE = 25
DEFAULT_PHASE_SHIFT_EVAL_POINTS = 240
DEFAULT_PHASE_SHIFT_REFINE_POINTS = 9


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def frac(x: np.ndarray | float) -> np.ndarray | float:
    """Return the fractional part of *x* (vectorised)."""
    return np.abs(x - np.floor(x))


def fmt_val(value: float, width: int = 0) -> str:
    """Format a parameter value without silently rounding it to zero.

    Fixed-point ``%.6f`` is fine for geometry (order 1-100) but destroys
    flux-scale parameters: ``f_scatter`` has a natural size of ~1e-13
    erg/cm^2/s and printed as "0.000000", which reads as "not fitted". Fall
    back to scientific notation for small-magnitude values.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        text = f"{v}"
    elif v != 0.0 and abs(v) < 1e-4:
        text = f"{v:.6e}"
    else:
        text = f"{v:.6f}"
    return f"{text:<{width}}" if width else text


def band_label_from_column(column: str) -> str:
    """Human-readable energy-band label for a simulation flux column.

    ``"nfl_soft" -> "SOFT"``, ``"nfl_broad" -> "BROAD"``; anything that does not
    follow the ``nfl_{band}`` convention is returned unchanged. Used for plot
    titles, which show only the energy band and χ²/dof.
    """
    col = str(column)
    if col.startswith("nfl_"):
        return col[4:].upper()
    return col


# Energy-band display names and ranges (keV), for plot titles/legends.
BAND_INFO: Dict[str, Tuple[str, str]] = {
    "ultrasoft": ("Ultra-soft", "0.2-0.5 keV"),
    "soft": ("Soft", "0.5-2 keV"),
    "medium": ("Medium", "1.2-2.0 keV"),
    "hard": ("Hard", "2.0-7.0 keV"),
    "broad": ("Broad", "0.5-7.0 keV"),
}


def detect_energy_bands(df: pd.DataFrame) -> List[str]:
    """Energy-band names present as ``nfl_{band}`` columns, in physical order."""
    bands = {c[len("nfl_"):] for c in df.columns
             if c.startswith("nfl_") and len(c) > len("nfl_")}
    ordered = [b for b in BAND_INFO if b in bands]
    return ordered + sorted(bands - set(ordered))


def get_band_display_name(band: str) -> Tuple[str, str]:
    """``(display_name, energy_range)`` for a band; range is '' if unknown."""
    if band in BAND_INFO:
        return BAND_INFO[band]
    return (band.replace("_", " ").title(), "")


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


# -----------------------------------------------------------------------------
# Observation reading
# -----------------------------------------------------------------------------

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
    except Exception:
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


def resolve_band_directory(band: str, data_dir: str) -> str:
    """Resolve the directory holding light-curve files for *band*.

    Tries, in order: *data_dir* itself, ``{Band}_with_flux/`` (old converted
    layout), ``{band}/single/`` (CIAO single-obs) and ``{band}/``.
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
        f"No .txt light-curve files found for band '{band}'. Searched:\n  {tried}"
    )


def load_observed_lightcurves(
    band: str,
    data_dir: str = "data/IC_10_X1_LC",
    flux_column: str = "FLUX",
    error_column: Optional[str] = None,
    time_column: Optional[str] = None,
) -> pd.DataFrame:
    """Load every observed light-curve file for one energy band.

    Wraps :func:`load_data` (via :func:`resolve_band_directory`) and remaps the
    columns to the fitting convention ``flux`` / ``flux_err`` / ``obs_id``.
    Zero and non-finite fluxes are dropped.

    Returns
    -------
    DataFrame with columns: time, flux, flux_err, obs_id, counts, phase
    """
    band_dir = resolve_band_directory(band, data_dir)
    print(f"Loading {band} band data from: {band_dir}")

    raw = load_data(
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
    if n_before - len(combined) > 0:
        print(f"Dropped {n_before - len(combined)} zero/non-finite flux rows before fitting")

    n_files = len(glob.glob(os.path.join(band_dir, "*.txt")))
    print(f"Loaded {len(combined)} data points from {n_files} file(s) for {band} band")
    return combined


# -----------------------------------------------------------------------------
# Phase binning
# -----------------------------------------------------------------------------

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
    DataFrame with columns: phase, *rate_column*, *error_column*, n_points.
    The value columns keep the caller's names, so a fit using ``flux`` /
    ``flux_err`` gets those names back and needs no rename wrapper.

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
                rate_column: mean_rate,
                error_column: mean_err,
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
        Columns: phase, *rate_column*, *error_column*, n_points, total_counts,
        phase_lo, phase_hi, width. The value columns keep the caller's names.
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
            columns=['phase', rate_column, error_column, 'n_points',
                     'total_counts', 'phase_lo', 'phase_hi', 'width']
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
                rate_column: mean_rate,
                error_column: mean_err,
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
# Smoothing and eclipse-floor helpers
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


# -----------------------------------------------------------------------------
# Periodic model interpolation
# -----------------------------------------------------------------------------

def prepare_model_interpolator(
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


def model_from_wrap(
    phase_wrap: np.ndarray,
    flux_wrap: np.ndarray,
    phases,
    shift=0.0,
    scatter: float = 0.0,
) -> np.ndarray:
    """Evaluate a prepared model at *phases* for a given shift and scatter.

    This is the single definition of the tabulated model used everywhere: the χ²
    in :func:`fit_simulation`, the overlay curve and the residual panel all route
    through it, so they cannot silently disagree. *shift* may be an array
    (broadcast against *phases*) to evaluate many trial shifts at once.
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

    Convenience wrapper over :func:`prepare_model_interpolator` +
    :func:`model_from_wrap` for callers that only need a single evaluation.
    """
    phase_wrap, flux_wrap = prepare_model_interpolator(sim_df, sim_column)
    return model_from_wrap(phase_wrap, flux_wrap, phases, shift, scatter)


def interp_periodic_phases(
    obs_phases: np.ndarray,
    model_phase: np.ndarray,
    model_flux: np.ndarray,
) -> np.ndarray:
    """Interpolate a periodic model given as (phase, flux) arrays onto *obs_phases*.

    The array-in/array-out counterpart of :func:`model_from_wrap`, for callers
    that hold a freshly evaluated model curve rather than a prepared
    interpolator (the MCMC likelihood, which rebuilds the curve every sample).
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


def obs_errors(
    obs_df: pd.DataFrame,
    rate_column: str = "rate",
    error_column: str = "error",
) -> np.ndarray:
    """Observation uncertainties, with the χ²-safety guards applied.

    Uses the provided errors when available, otherwise sqrt(|rate|); zero,
    negative and non-finite values are floored so they cannot blow up χ².
    Shared by :func:`fit_simulation` and the plotting helpers so both weight the
    data identically.
    """
    rate = obs_df[rate_column].to_numpy(dtype=float)
    if error_column in obs_df.columns and not obs_df[error_column].isnull().all():
        err = obs_df[error_column].to_numpy(dtype=float)
    else:
        err = np.sqrt(np.abs(rate))
    return np.where((err <= 0) | ~np.isfinite(err), 1e-3, err)


# -----------------------------------------------------------------------------
# Chi-square fitting of a tabulated model
# -----------------------------------------------------------------------------

def fit_simulation(
    obs_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    sim_column: str = "fl",
    fit_phase_shift: bool = False,
    scatter: float = 0.0,
    n_shift_grid: int = 1000,
    verbose: bool = True,
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
    verbose : bool, default True
        Print the fitted shift, scatter and reduced chi-square.

    Returns
    -------
    (phase_shift, reduced_chi2)
        Best-fit phase shift (0–1) and reduced chi-squared value.
        If *fit_phase_shift* is False, returns ``(0.0, reduced_chi2)``.
    """
    # Prepare observation arrays
    phase_obs = obs_df["phase"].to_numpy()
    rate_obs = obs_df["rate"].to_numpy(dtype=float)
    err_obs = obs_errors(obs_df)

    # Prepared once; every model evaluation below goes through model_from_wrap
    # so the χ² here and the curve drawn by plot_phase are the same function.
    phase_wrap, flux_wrap = prepare_model_interpolator(sim_df, sim_column)
    scatter = float(scatter)

    def chi2(shift) -> float:
        model = model_from_wrap(phase_wrap, flux_wrap, phase_obs, shift, scatter)
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
        models = model_from_wrap(
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
        if verbose:
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
    if verbose:
        print(
            f"Chi-square (no phase-shift fit, no flux rescaling):\n"
            f"  Phase shift = {best_shift:.5f} (fixed)\n"
            f"  Scattered flux = {scatter:.6g} (fixed, additive)\n"
            f"  Reduced χ² = {reduced_chi2:.3f}  (dof = {max(len(rate_obs) - n_free, 1)})"
        )
    return float(best_shift), float(reduced_chi2)


# -----------------------------------------------------------------------------
# Periodic phase-shift alignment
# -----------------------------------------------------------------------------
# The model's phase zero is not tied to the ephemeris, so every comparison with
# data allows a free shift. chi2(shift) is periodic and strongly multi-modal
# (the eclipse), so a local optimizer started at 0 settles in the wrong basin:
# both searches here scan a coarse grid over the full period first, then refine.
# :func:`fit_simulation` does the same for a tabulated model.

def build_phase_shift_terms(
    enabled: bool,
    obs_phase: np.ndarray,
    *,
    grid_size: int = DEFAULT_PHASE_SHIFT_GRID_SIZE,
    eval_points: int = DEFAULT_PHASE_SHIFT_EVAL_POINTS,
    refine_points: int = DEFAULT_PHASE_SHIFT_REFINE_POINTS,
) -> Dict[str, object]:
    """Precompute the reusable arrays for a per-sample phase-shift search."""
    if not enabled:
        return {"enabled": False}
    n_grid = max(3, int(grid_size))
    shift_grid = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    return {
        "enabled": True,
        "shift_grid": shift_grid,
        "phase_eval_grid": np.linspace(0.0, 1.0, max(16, int(eval_points)), endpoint=False),
        "shifted_obs_phase": np.mod(obs_phase[None, :] - shift_grid[:, None], 1.0),
        "refine_points": max(0, int(refine_points)),
    }


def apply_best_phase_shift(
    model_phase: np.ndarray,
    model_flux: np.ndarray,
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err2: np.ndarray,
    phase_shift_terms: Optional[Dict[str, object]],
) -> Tuple[Optional[np.ndarray], float]:
    """Align a model curve to observations by minimizing weighted chi-square.

    Returns ``(model_at_obs_phases, best_shift)``, or ``(None, 0.0)`` if no
    trial shift produced a finite model. With the search disabled, returns
    *model_flux* unchanged and a shift of 0.
    """
    if not phase_shift_terms or not bool(phase_shift_terms.get("enabled", False)):
        return model_flux, 0.0

    shift_grid = np.asarray(phase_shift_terms.get("shift_grid", []), dtype=float)
    if shift_grid.size == 0:
        return model_flux, 0.0
    shifted_obs_phase = phase_shift_terms.get("shifted_obs_phase")
    if shifted_obs_phase is None or np.shape(shifted_obs_phase) != (shift_grid.size, obs_phase.size):
        shifted_obs_phase = np.mod(obs_phase[None, :] - shift_grid[:, None], 1.0)

    best_model, best_idx, best_shift, best_chi2 = None, -1, 0.0, np.inf

    def _try(shifted_phase, shift, idx=-1):
        nonlocal best_model, best_idx, best_shift, best_chi2
        flux = interp_periodic_phases(shifted_phase, model_phase, model_flux)
        if np.any(~np.isfinite(flux)):
            return
        chi2 = float(np.sum((obs_flux - flux) ** 2 / obs_err2))
        if chi2 < best_chi2:
            best_model, best_idx, best_shift, best_chi2 = flux, idx, float(shift), chi2

    for i, shift in enumerate(shift_grid):
        _try(shifted_obs_phase[i], shift, i)
    if best_model is None:
        return None, 0.0

    # Local refinement within one coarse step, so a dense global grid is not
    # needed for sub-grid accuracy.
    n_refine = int(phase_shift_terms.get("refine_points", 0) or 0)
    if n_refine > 1 and shift_grid.size >= 3 and best_idx >= 0:
        step = 1.0 / float(shift_grid.size)
        for shift in np.mod(np.linspace(best_shift - step, best_shift + step, n_refine), 1.0):
            _try(np.mod(obs_phase - shift, 1.0), shift)

    return best_model, best_shift


# -----------------------------------------------------------------------------
# Chunked CSV output
# -----------------------------------------------------------------------------

def save_samples_csv_chunked(
    samples: np.ndarray,
    param_names: List[str],
    output_path: str,
    log_prob: Optional[np.ndarray] = None,
    chunk_size: int = 50000,
) -> None:
    """Write a sample table to CSV in chunks to limit peak memory."""
    headers = list(param_names) + (["log_prob"] if log_prob is not None else [])
    with open(output_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(headers)
        for start in range(0, len(samples), int(chunk_size)):
            stop = min(start + int(chunk_size), len(samples))
            block = samples[start:stop]
            if log_prob is None:
                writer.writerows(np.asarray(block, dtype=float).tolist())
            else:
                lp = np.asarray(log_prob[start:stop], dtype=float)
                writer.writerows(
                    np.column_stack([np.asarray(block, dtype=float), lp]).tolist()
                )


# -----------------------------------------------------------------------------
# CLI run-config persistence
# -----------------------------------------------------------------------------
# A fit writes its full CLI configuration next to its results; a later --replot
# restores every option the user did not retype. Without this, replotting falls
# back to argparse defaults for the data selection, binning and priors, which
# changes the observed arrays and so the reported chi2/dof.

RUN_CONFIG_SUFFIX = "_run_config.json"

# `replot` must never be restored: a saved fit recorded replot=False, so
# restoring it would cancel the replot. `output_dir` is defined by where the
# config was found, not by what the original run typed.
_RUN_CONFIG_NEVER_RESTORE = frozenset({"replot", "output_dir"})


def _jsonable(value):
    """Best-effort conversion of an argparse value into JSON-representable form."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def run_config_path(output_dir: str, band: str, wind_model: str) -> str:
    """Path of the run-config file for one (band, wind_model) fit."""
    return os.path.join(output_dir, f"{band}_{wind_model}{RUN_CONFIG_SUFFIX}")


def save_run_config(output_dir: str, band: str, wind_model: str, args) -> Optional[str]:
    """Persist the CLI configuration of a fit alongside its results."""
    path = run_config_path(output_dir, band, wind_model)
    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": shlex.join(sys.argv),
        "band": band,
        "wind_model": wind_model,
        "args": {k: _jsonable(v) for k, v in sorted(vars(args).items())},
    }
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
    except Exception as e:
        warnings.warn(f"Could not save run config to {path}: {e}")
        return None
    print(f"Run config saved to: {path}")
    return path


def find_run_configs(
    output_dir: str,
    band: Optional[str] = None,
    wind_model: Optional[str] = None,
) -> List[str]:
    """Saved run-config paths in *output_dir*, optionally filtered.

    ``band='all'`` counts as unspecified: it is a fit-many-bands request, not
    the name of a saved fit.
    """
    b = band if (band and band != 'all') else "*"
    w = wind_model if wind_model else "*"
    return sorted(glob.glob(os.path.join(output_dir, f"{b}_{w}{RUN_CONFIG_SUFFIX}")))


def _explicit_cli_dests(parser: argparse.ArgumentParser, argv: List[str]) -> set:
    """Argparse dests corresponding to options the user actually typed.

    Comparing against ``parser.get_default()`` is not enough: a user who
    explicitly passes the default value should still beat a saved config.
    Unambiguous prefixes are resolved as argparse resolves them; anything
    unrecognized (a negative number used as a value) is ignored.
    """
    opt_to_dest: Dict[str, str] = {
        opt: action.dest
        for action in parser._actions
        for opt in action.option_strings
    }
    seen = set()
    for token in argv:
        if not token.startswith("-") or token in ("-", "--"):
            continue
        name = token.split("=", 1)[0]
        dest = opt_to_dest.get(name)
        if dest is None:
            matches = {d for o, d in opt_to_dest.items() if o.startswith(name)}
            dest = matches.pop() if len(matches) == 1 else None
        if dest is not None:
            seen.add(dest)
    return seen


def apply_saved_run_config(
    parser: argparse.ArgumentParser,
    args,
    argv: Optional[List[str]] = None,
) -> Optional[str]:
    """Fill in options the user did not type from a previous run's config.

    Intended for ``--replot``, so that flag alone reproduces the original run's
    band, wind model, data selection, binning and priors. Explicit command-line
    values always win. Returns the config path used, or None.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    explicit = _explicit_cli_dests(parser, argv)

    candidates = find_run_configs(
        args.output_dir,
        band=(args.band if 'band' in explicit else None),
        wind_model=(args.wind_model if 'wind_model' in explicit else None),
    )
    if not candidates:
        return None

    if len(candidates) > 1:
        # Several fits share this directory. Configs differing only by band (the
        # `--band all` case) restore identically, so take the first; otherwise
        # ask the user to disambiguate.
        loaded = []
        for path in candidates:
            try:
                with open(path) as fh:
                    loaded.append((path, json.load(fh)))
            except Exception:
                continue
        if not loaded:
            return None
        comparable = []
        for _, cfg in loaded:
            rest = dict(cfg.get("args", {}))
            rest.pop("band", None)
            comparable.append(rest)
        if any(c != comparable[0] for c in comparable[1:]):
            names = "\n  ".join(os.path.basename(p) for p, _ in loaded)
            parser.error(
                f"Multiple saved run configs in {args.output_dir} and they differ; "
                f"specify --band and/or --wind-model to choose one:\n  {names}"
            )
        candidates = [loaded[0][0]]

    config_path = candidates[0]
    try:
        with open(config_path) as fh:
            config = json.load(fh)
    except Exception as e:
        warnings.warn(f"Could not read run config {config_path}: {e}")
        return None

    saved_args = config.get("args", {})
    if not isinstance(saved_args, dict):
        warnings.warn(f"Run config {config_path} has no 'args' block; ignoring.")
        return None

    # dest -> the flag the user would type ('prior_M_X' is spelled '--prior-MX').
    dest_to_flag: Dict[str, str] = {}
    for action in parser._actions:
        longs = [o for o in action.option_strings if o.startswith("--")]
        if longs:
            dest_to_flag[action.dest] = longs[0]
    known_dests = {a.dest for a in parser._actions}

    restored: List[Tuple[str, object]] = []
    for dest, value in saved_args.items():
        if dest in _RUN_CONFIG_NEVER_RESTORE or dest in explicit or dest not in known_dests:
            continue
        # Compare in JSON space: a tuple default round-trips as a list, which is
        # not a real change and should not be reported as one.
        if _jsonable(getattr(args, dest, None)) == value:
            continue
        setattr(args, dest, value)
        restored.append((dest, value))

    print(f"\nRestored CLI options from: {config_path}")
    print(f"  (original run: {config.get('created', 'unknown time')})")
    if config.get("command"):
        print(f"  original command: {config['command']}")
    if restored:
        for dest, value in sorted(restored):
            print(f"    {dest_to_flag.get(dest, '--' + dest)} = {value!r}")
    else:
        print("    (nothing to restore — command line already matches)")
    overridden = sorted(
        dest_to_flag.get(d, '--' + d) for d in explicit
        if d not in _RUN_CONFIG_NEVER_RESTORE and d in saved_args
    )
    if overridden:
        print(f"  kept from the command line: {', '.join(overridden)}")
    return config_path
