#!/usr/bin/env python3
"""
Shared non-plotting helpers for the XRB light-curve codebase.
-------------------------------------------------------------
Every routine here was previously defined (and in some cases duplicated) inside
``chandra_phase_analysis.py`` / ``mcmc_lightcurve_fit.py``. Both scripts now
import from this module, so there is a single implementation of:

* the ephemeris (``REF_EPOCH``, ``ORBITAL_PERIOD``) and :func:`frac`
* observation loading (:func:`read_observation`, :func:`load_data`) and
  simulation-column discovery (:func:`detect_flux_columns`)
* phase binning -- fixed-width (:func:`phase_bin_data`) and adaptive
  constant-counts (:func:`phase_bin_data_snr`)
* Gaussian phase smoothing (:func:`smooth_lightcurve`) and the eclipse-floor
  estimate (:func:`estimate_scattered_flux`)
* the single periodic model interpolator (:func:`periodic_model`,
  :func:`eval_periodic`, :func:`prepare_model_interpolator`)
* the periodic phase-shift search (:class:`PhaseShiftSearch`,
  :func:`build_phase_shift_search`, :func:`best_phase_shift`), shared by the
  tabulated-model χ² fit (:func:`fit_simulation`) and the MCMC likelihood
* band-directory observation loading (:func:`resolve_band_directory`,
  :func:`load_observed_lightcurves`) and :func:`save_samples_csv_chunked`
* CLI run-config persistence (:func:`save_run_config`,
  :func:`apply_saved_run_config`), which lets ``--replot`` reproduce a fit's
  options without retyping them

This module deliberately depends only on the standard library plus numpy /
pandas: it must stay importable from either analysis script (and from the
XSPEC environment) without creating an import cycle.

Dependencies: numpy, pandas (in requirements.txt).
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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Constants adopted from the R script (seconds)
# -----------------------------------------------------------------------------
REF_EPOCH: float = 278801348  # Reference time (t0) used for phase zero
# REF_EPOCH: float = 278800407.267 # corrected reference epoch from find_reference_epoch.py
ORBITAL_PERIOD: float = 125431  # Orbital period of the system

# Periodic phase-shift search (build_phase_shift_search / best_phase_shift):
# a coarse scan over the full period whose step is tied to the data and model
# spacing, then PHASE_SHIFT_LEVELS dense passes of PHASE_SHIFT_FINE_POINTS
# points, each spanning +-1 previous step around the best shift.
PHASE_SHIFT_MIN_GRID = 25
PHASE_SHIFT_MAX_GRID = 400
PHASE_SHIFT_FINE_POINTS = 33
PHASE_SHIFT_LEVELS = 2

# Phase grid for the drawn model overlay, shared by plot_utils.plot_phase and
# write_model_lightcurve so the dumped curve is exactly the plotted one.
MODEL_OVERLAY_N_POINTS = 721


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def frac(x: np.ndarray | float) -> np.ndarray | float:
    """Return the fractional part of *x* in [0, 1) (vectorised)."""
    return x - np.floor(x)


def check_phase_window(lo: float, hi: float) -> Tuple[float, float]:
    """Validate a phase window ``[lo, hi)``; returns ``(lo, hi)`` as floats.

    ``(0, 1)`` is the full orbit. Otherwise both bounds must lie in [0, 1] and
    differ; ``lo > hi`` denotes a window wrapping through phase 0.
    """
    lo, hi = float(lo), float(hi)
    if not (0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0):
        raise ValueError(f"phase window bounds must lie in [0, 1], got {lo} and {hi}")
    if (lo, hi) != (0.0, 1.0) and (lo % 1.0) == (hi % 1.0):
        raise ValueError(f"phase window [{lo:g}, {hi:g}) is empty; use 0 1 for the full orbit")
    return lo, hi


def is_full_phase_window(lo: float, hi: float) -> bool:
    """True for the full orbit ``(0, 1)``."""
    return float(lo) == 0.0 and float(hi) == 1.0


def in_phase_window(phase, lo: float, hi: float) -> np.ndarray:
    """Boolean mask of *phase* values inside ``[lo, hi)``, wrapping through 0 when ``lo > hi``."""
    lo, hi = check_phase_window(lo, hi)
    phase = np.mod(np.asarray(phase, dtype=float), 1.0)
    if is_full_phase_window(lo, hi):
        return np.ones(phase.shape, dtype=bool)
    return np.mod(phase - lo, 1.0) < ((hi - lo) % 1.0)


def phase_window_intervals(lo: float, hi: float) -> List[Tuple[float, float]]:
    """The window as non-wrapping intervals on [0, 1): one, or two when it wraps."""
    lo, hi = check_phase_window(lo, hi)
    if is_full_phase_window(lo, hi):
        return [(0.0, 1.0)]
    return [(lo, hi)] if lo < hi else [(lo, 1.0), (0.0, hi)]


def apply_phase_window(df: pd.DataFrame, lo: float, hi: float, column: str = "phase",
                       verbose: bool = True) -> pd.DataFrame:
    """Rows of *df* whose *column* lies inside the phase window.

    The full window returns *df* unchanged; otherwise the kept count is
    printed and an empty result raises ``ValueError``. Shared by both fitters
    so they select the same rows for the same window.
    """
    lo, hi = check_phase_window(lo, hi)
    if is_full_phase_window(lo, hi):
        return df
    out = df[in_phase_window(df[column].to_numpy(dtype=float), lo, hi)].reset_index(drop=True)
    if verbose:
        print(f"Phase window [{lo:g}, {hi:g}): kept {len(out)} of {len(df)} points")
    if out.empty:
        raise ValueError(f"No observed points fall inside the phase window [{lo:g}, {hi:g}).")
    return out


def drop_invalid_flux_rows(df: pd.DataFrame, column: str, drop_nonpositive: bool = True,
                           verbose: bool = True) -> pd.DataFrame:
    """Drop rows whose *column* is non-finite and, by default, ``<= 0``.

    Zero-count bins (and negative background-subtracted rates) are the
    ``<= 0`` rows; ``drop_nonpositive=False`` keeps them, and their zero errors
    are then repaired by :func:`sanitize_errors`. One rule for both fitters.
    """
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(values)
    if drop_nonpositive:
        keep &= values > 0
    out = df[keep].reset_index(drop=True)
    if verbose:
        n_dropped = len(df) - len(out)
        if n_dropped:
            what = "non-positive/non-finite" if drop_nonpositive else "non-finite"
            print(f"Dropped {n_dropped} {what} {column} rows before fitting")
        if not drop_nonpositive:
            n_zero = int(np.sum(values[keep] <= 0))
            if n_zero:
                print(f"Kept {n_zero} rows with {column} <= 0 (zero-count bins); their errors are "
                      f"repaired by sanitize_errors")
    return out


def model_dump_path(plot_path: Optional[str], default: str = "model_lightcurve.txt") -> str:
    """``<plot stem>_model.txt`` next to a figure, or *default* without a figure.

    Used by both fitters so the model dump always sits beside the plot it
    describes under either entry point.
    """
    if not plot_path:
        return default
    stem, _ = os.path.splitext(str(plot_path))
    return f"{stem}_model.txt"


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


# Chandra energy bands (keV), the single definition shared by the flux-table
# generator, the synthetic-data scripts and the plot labels.
CHANDRA_BANDS: Dict[str, Tuple[float, float]] = {
    "broad": (0.5, 7.0),
    "soft": (0.5, 2.0),
    "medium": (1.2, 2.0),
    "hard": (2.0, 7.0),
}

# Chandra time system: MJD = MJDREF + t / 86400 for mission time t in seconds.
MJDREF_CHANDRA: float = 50814.0

# Energy-band display names and ranges, for plot titles/legends (physical order).
BAND_INFO: Dict[str, Tuple[str, str]] = {
    "ultrasoft": ("Ultra-soft", "0.2-0.5 keV"),
    **{band: (band.capitalize(), f"{lo:g}-{hi:g} keV") for band, (lo, hi) in
       sorted(CHANDRA_BANDS.items(), key=lambda kv: (kv[1][0], kv[1][1]))},
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
    """Band-flux columns (``nfl_{band}``) of a simulation DataFrame, in the
    physical band order of :func:`detect_energy_bands`."""
    return [f"nfl_{band}" for band in detect_energy_bands(df)]


def fit_exponential(nh: np.ndarray, flux: np.ndarray) -> Tuple[float, float]:
    """Least-squares fit of ``flux = A exp(-B nh)`` in log space, as ``(A, B)``.

    Fitting ``log flux = log A - B nh`` gives every point equal weight
    regardless of magnitude, appropriate for data spanning many decades. Shared
    by the simulator's ``refit`` flux method and by ``compute_flux_vs_nH.py``'s
    figure annotation, so the law drawn on the table is the law the model uses.
    """
    nh = np.asarray(nh, dtype=float)
    flux = np.asarray(flux, dtype=float)
    valid = np.isfinite(nh) & np.isfinite(flux) & (flux > 0) & (nh > 0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Exponential fit needs at least two valid (nH, flux) points")
    slope, intercept = np.polyfit(nh[valid], np.log(flux[valid]), 1)
    return float(np.exp(intercept)), float(-slope)


# -----------------------------------------------------------------------------
# Observation reading
# -----------------------------------------------------------------------------

def find_column(df: pd.DataFrame, name: Optional[str]) -> Optional[str]:
    """The column of *df* whose name matches *name* case-insensitively, or None."""
    if name is None:
        return None
    target = str(name).upper()
    for col in df.columns:
        if str(col).upper() == target:
            return col
    return None


def _first_column(df: pd.DataFrame, names) -> Optional[str]:
    """First of *names* present in *df* (case-insensitive), or None."""
    for name in names:
        col = find_column(df, name)
        if col is not None:
            return col
    return None


def _derive_err_from_rate_err(df: pd.DataFrame, obs_col: str) -> Optional[pd.Series]:
    """Derive observable errors from rate_err for proportional columns.

    This supports CIAO style files where ``flux_t`` exists but ``flux_t_err``
    does not. For rows with finite, positive ``rate`` we use:

        err_obs = rate_err * (obs / rate)

    For rows where that ratio is undefined, fall back to a robust file-level
    conversion factor median(obs/rate) computed from valid rows.
    """
    rate_col = find_column(df, "RATE")
    rate_err_col = _first_column(df, ("RATE_ERR", "ERR_RATE", "COUNT_RATE_ERR"))
    if rate_col is None or rate_err_col is None:
        return None

    obs_vals = pd.to_numeric(df[obs_col], errors="coerce").to_numpy(dtype=float)
    rate_vals = pd.to_numeric(df[rate_col], errors="coerce").to_numpy(dtype=float)
    rate_err_vals = pd.to_numeric(df[rate_err_col], errors="coerce").to_numpy(dtype=float)

    valid_ratio = np.isfinite(obs_vals) & np.isfinite(rate_vals) & (rate_vals > 0.0)
    if not np.any(valid_ratio):
        return None

    ratio = np.full(len(df), np.nan, dtype=float)
    ratio[valid_ratio] = obs_vals[valid_ratio] / rate_vals[valid_ratio]
    ratio = np.where(np.isfinite(ratio), ratio, float(np.nanmedian(ratio[valid_ratio])))
    return pd.Series(rate_err_vals * ratio, index=df.index, dtype=float)


def _detect_error_column(df: pd.DataFrame, obs_col: str,
                         requested: Optional[str] = None) -> Optional[str]:
    """Error column belonging to *obs_col*, or None.

    An explicitly requested name wins (with a warning if it is absent).
    Otherwise ``{OBS}_ERR`` and ``ERR_{OBS}`` are matched case-insensitively.
    ``rate_err`` measures the rate, so it is only offered when the observable
    *is* the rate; for a proportional column such as ``flux_t`` it has the
    wrong scale and the caller derives the error from it instead
    (:func:`_derive_err_from_rate_err`).
    """
    if requested:
        col = find_column(df, requested)
        if col is not None:
            return col
        warnings.warn(f"Requested error column '{requested}' not found; "
                      f"auto-detecting. Available columns: {list(df.columns)}")

    obs_upper = str(obs_col).upper()
    candidates = [f"{obs_upper}_ERR", f"ERR_{obs_upper}"]
    if obs_upper in {"RATE", "COUNT_RATE", "NET_RATE"}:
        candidates += ["RATE_ERR", "ERR_RATE", "COUNT_RATE_ERR"]
    for name in candidates:
        col = find_column(df, name)
        if col is not None:
            return col

    # Generic fallback: any *_ERR column naming this observable. Deliberately
    # does not accept a bare rate_err for a non-rate observable.
    obs_base = obs_upper.split("_")[0]
    for col in df.columns:
        col_upper = str(col).upper()
        if col_upper != obs_upper and "ERR" in col_upper and obs_base and obs_base in col_upper:
            return col
    return None


def _header_columns(file_path: str) -> Optional[List[str]]:
    """Column names declared in a file's comment header, or None if it has none.

    Recognizes the CIAO ``# Columns: a, b, c`` (or ``# #Columns:``) form and a
    plain commented header line naming TIME/RATE/FLUX-like columns. Only the
    leading comment block (at most 10 lines) is inspected.
    """
    lines: List[str] = []
    with open(file_path, "r") as fh:
        for line in fh:
            lines.append(line.strip())
            if not line.strip().startswith("#") or len(lines) > 10:
                break

    header: Optional[List[str]] = None
    for line in lines:
        if not line.startswith("#"):
            continue
        if "Columns:" in line:
            return [c.strip() for c in line.split("Columns:")[1].split(",") if c.strip()]
        clean = line.lstrip("#").strip()
        if clean and ":" not in clean and "=" not in clean and any(
                key in clean.upper() for key in ("TIME", "RATE", "FLUX")):
            header = clean.split()
    return header


def read_observation(
    file_path: str,
    label: str,
    obs_column: str = "rate",
    obs_error_column: Optional[str] = None,
    time_column: Optional[str] = None,
    counts_column: Optional[str] = "counts",
) -> pd.DataFrame:
    """Read a single Chandra observation text file.

    Two file shapes are supported:

    1. A commented header naming the columns -- CIAO's ``# Columns: dt, t_raw,
       mjd, phase, counts, rate, rate_err, flux_t`` or a plain
       ``# TIME RATE ERROR`` line. Columns are resolved case-insensitively:
       the timestamp from *time_column* or ``TIME``/``T_RAW``/``T``/``MJD``,
       the observable from *obs_column*, the error from *obs_error_column* or
       auto-detection (falling back to a ``rate_err``-derived error for
       proportional columns such as ``flux_t``), and *counts_column* when
       present.
    2. No header: two or three whitespace-separated columns ``time, rate[, error]``.

    Errors in the header path (unknown observable, header/data column-count
    mismatch, no time column) are raised rather than silently falling back to
    the headerless reader.

    Returns
    -------
    DataFrame with columns ``time, rate, phase, obs`` plus ``error`` and
    ``counts`` when available; ``rate`` holds the requested observable.
    """
    header = _header_columns(file_path)
    if header is None:
        df = pd.read_csv(file_path, sep=r"\s+", comment="#", header=None)
        # With `names=` pandas would silently promote surplus leading columns
        # to the index, shifting time/rate/error by one column.
        if df.shape[1] not in (2, 3):
            raise ValueError(
                f"{file_path}: a headerless file must have 2 or 3 columns "
                f"(time, rate[, error]); found {df.shape[1]}. Name the columns with a "
                f"'# Columns: ...' header line instead.")
        df.columns = ["time", "rate", "error"][: df.shape[1]]
        df["phase"] = frac((df["time"] - REF_EPOCH) / ORBITAL_PERIOD)
        df["obs"] = label
        return df

    df = pd.read_csv(file_path, sep=r"\s+", comment="#", header=None)
    if len(header) != len(df.columns):
        raise ValueError(
            f"{file_path}: header names {header} do not match the "
            f"{len(df.columns)} data columns")
    df.columns = header

    time_col = find_column(df, time_column) if time_column else None
    if time_column and time_col is None:
        warnings.warn(f"Time column '{time_column}' not found in {file_path}; "
                      f"auto-detecting. Available columns: {list(df.columns)}")
    if time_col is None:
        time_col = _first_column(df, ("TIME", "T_RAW", "T", "MJD"))
    if time_col is None:
        raise ValueError(f"No time column found in {file_path}. "
                         f"Available columns: {list(df.columns)}")

    obs_col = find_column(df, obs_column)
    if obs_col is None:
        raise ValueError(f"Column '{obs_column}' not found in {file_path}. "
                         f"Available columns: {list(df.columns)}")

    out = pd.DataFrame({"rate": df[obs_col], "time": df[time_col]})
    err_col = _detect_error_column(df, obs_col, obs_error_column)
    if err_col is not None:
        out["error"] = df[err_col]
    else:
        derived = _derive_err_from_rate_err(df, obs_col)
        if derived is not None:
            out["error"] = derived
    counts_col = find_column(df, counts_column) if counts_column else None
    if counts_col is not None:
        out["counts"] = pd.to_numeric(df[counts_col], errors="coerce")

    # Phase is always recomputed from the timestamps and the current ephemeris.
    out["phase"] = frac((out["time"] - REF_EPOCH) / ORBITAL_PERIOD)
    out["obs"] = label
    return out


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
    drop_nonpositive_flux: bool = True,
) -> pd.DataFrame:
    """Load every observed light-curve file for one energy band.

    Wraps :func:`load_data` (via :func:`resolve_band_directory`) and remaps the
    columns to the fitting convention ``flux`` / ``flux_err`` / ``obs_id``.
    Non-finite fluxes are always dropped. Rows with ``flux <= 0`` (zero-count
    bins, whose error is also 0) are dropped unless *drop_nonpositive_flux* is
    False, in which case :func:`sanitize_errors` gives them the median valid
    error.

    Returns
    -------
    DataFrame with columns: time, flux, flux_err, obs_id, phase, and counts
    when the files carry it.
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
        'obs_id': raw['obs'],
        'phase': raw['phase'].astype(float),
    })
    if 'counts' in raw.columns:
        # Only when the files carry counts: an all-NaN column would pass the
        # presence check of the constant-counts binner.
        combined['counts'] = raw['counts'].astype(float)

    combined = combined.loc[np.isfinite(combined['time'])].reset_index(drop=True)
    combined = drop_invalid_flux_rows(combined, 'flux', drop_nonpositive=drop_nonpositive_flux)
    if 'error' in raw.columns:
        combined['flux_err'] = sanitize_errors(combined['flux_err'], context=f"{band} band light curves: ")

    n_files = len(glob.glob(os.path.join(band_dir, "*.txt")))
    print(f"Loaded {len(combined)} data points from {n_files} file(s) for {band} band")
    return combined


# -----------------------------------------------------------------------------
# Phase binning
# -----------------------------------------------------------------------------

def weighted_mean(values: np.ndarray, errors: Optional[np.ndarray]) -> Tuple[float, float]:
    """Inverse-variance weighted mean of *values* and its error ``sqrt(1/Σw)``.

    Errors must already be valid (finite and > 0): the loaders repair them
    once with :func:`sanitize_errors` before binning, so a bad error here is a
    programming error and raises. Without errors (no error column at all) the
    plain mean and the standard error of the mean are returned. Shared by both
    phase binners so they weight points identically.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if errors is None or not np.any(np.isfinite(errors)):
        if n > 1:
            return float(np.mean(values)), float(np.std(values) / np.sqrt(n))
        return float(values[0]), 0.0
    errors = np.asarray(errors, dtype=float)
    if np.any(~np.isfinite(errors) | (errors <= 0)):
        raise ValueError("weighted_mean: errors must be finite and > 0 (run sanitize_errors first)")
    weights = 1.0 / errors ** 2
    return float(np.average(values, weights=weights)), float(np.sqrt(1.0 / np.sum(weights)))


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

    # Assign each point to a bin. Rows without a finite phase or value are
    # dropped first: np.digitize would file a NaN phase in the last bin.
    df = df[np.isfinite(df['phase']) & np.isfinite(df[rate_column])].copy()
    df['_bin'] = np.digitize(df['phase'], bin_edges) - 1
    df['_bin'] = df['_bin'].clip(0, n_bins - 1)  # Handle edge case at phase=1

    binned_data = []

    for i in range(n_bins):
        bin_mask = df['_bin'] == i
        bin_df = df[bin_mask]

        if len(bin_df) >= min_points_per_bin:
            rate_vals = bin_df[rate_column].to_numpy(dtype=float)
            err_vals = (bin_df[error_column].to_numpy(dtype=float)
                        if error_column in bin_df.columns else None)
            mean_rate, mean_err = weighted_mean(rate_vals, err_vals)

            binned_data.append({
                'phase': bin_centers[i],
                rate_column: mean_rate,
                error_column: mean_err,
                'n_points': len(bin_df)
            })

    if not binned_data:
        raise ValueError(
            f"phase_bin_data: no bin reached min_points_per_bin={min_points_per_bin} "
            f"({len(df)} finite points into {n_bins} bins); use fewer bins or no binning.")
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
    phase_origin: float = 0.0,
) -> pd.DataFrame:
    """
    Adaptive phase binning with approximately constant counts per bin.

    Points are sorted by phase and grouped greedily until each bin reaches
    ``counts_per_bin`` total counts, yielding variable phase-width bins. The
    ordering (and the bin centres and edges) use the phase measured from
    *phase_origin*, so a phase window that wraps through 0 (pass its lower
    bound) never merges the points on either side of its seam into one bin
    whose centre would lie in the excluded gap.

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

    work = df[np.isfinite(df['phase']) & np.isfinite(df[rate_column])].copy()
    if work.empty:
        raise ValueError("phase_bin_data_snr: no rows with a finite phase and value to bin.")

    # The counts must be real: a NaN or all-zero column would never reach the
    # target and the whole light curve would silently become a single bin.
    counts = pd.to_numeric(work[counts_column], errors='coerce').to_numpy(dtype=float)
    if not np.all(np.isfinite(counts)) or np.any(counts < 0) or not np.any(counts > 0):
        raise ValueError(
            f"phase_bin_data_snr: counts column '{counts_column}' must be finite, non-negative "
            f"and not all zero (non-finite: {int(np.sum(~np.isfinite(counts)))}, "
            f"negative: {int(np.sum(counts < 0))}, positive: {int(np.sum(counts > 0))}).")
    work[counts_column] = counts
    origin = float(phase_origin) % 1.0
    work['_u'] = np.mod(work['phase'].to_numpy(dtype=float) - origin, 1.0)
    work = work.sort_values('_u').reset_index(drop=True)

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
        err_vals = (bin_df[error_column].to_numpy(dtype=float)
                    if error_column in bin_df.columns else None)
        mean_rate, mean_err = weighted_mean(rate_vals, err_vals)

        u_vals = bin_df['_u'].to_numpy(dtype=float)
        bin_counts = bin_df[counts_column].to_numpy(dtype=float)
        total_counts = float(np.sum(bin_counts))
        u_center = float(np.average(u_vals, weights=bin_counts)) if total_counts > 0 else float(np.mean(u_vals))
        u_lo, u_hi = float(np.min(u_vals)), float(np.max(u_vals))
        phase_center = (u_center + origin) % 1.0
        phase_lo = (u_lo + origin) % 1.0
        phase_hi = (u_hi + origin) % 1.0
        width = float(max(u_hi - u_lo, 0.0))

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
    verbose: bool = True,
) -> pd.DataFrame:
    """Periodic Gaussian-kernel smoothing with its propagated 1σ band.

    The smoother is linear in the data, so the uncertainty of the smoothed
    value is exact: ``Var[Σ w_i f_i / Σ w_i] = Σ w_i² σ_i² / (Σ w_i)²``. Points
    without a valid error contribute to the smoothed curve but not to its
    band.
    """
    phase = np.asarray(phase, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if phase.shape != flux.shape:
        raise ValueError("phase and flux must have identical shapes.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if n_eval <= 0:
        raise ValueError("n_eval must be > 0.")

    valid = np.isfinite(phase) & np.isfinite(flux)
    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)
        if flux_err.shape != flux.shape:
            raise ValueError("flux_err must match phase/flux shape.")
        flux_err = np.where(np.isfinite(flux_err) & (flux_err > 0.0), flux_err, np.nan)

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
    if flux_err is not None:
        ok = np.isfinite(flux_err)
        if np.any(ok):
            w_ok = w[:, ok]
            wsum_ok = w_ok.sum(axis=1)
            good_ok = wsum_ok > 0.0
            if np.any(good_ok):
                flux_smooth_err[good_ok] = (
                    np.sqrt((w_ok[good_ok] ** 2) @ (flux_err[ok] ** 2)) / wsum_ok[good_ok]
                )

    if verbose:
        print(f"Smoothing: {phase.size} points, sigma={sigma:.4f}, eval={eval_phase.size}")

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

def periodic_model(phase, flux) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare a periodic ``(phase, flux)`` curve for ``np.interp``.

    Returns ``(phase_ext, flux_ext)``: the curve folded into [0, 1), sorted,
    duplicate abscissae removed, with the last point repeated at ``phase - 1``
    and the first at ``phase + 1`` so that every query in [0, 1) is bracketed.
    This is the single periodic interpolator of the codebase: the tabulated χ²
    fit, the plot overlays, the model dumps and the MCMC likelihood all
    evaluate a model through it, so they cannot disagree.
    """
    p = np.mod(np.asarray(phase, dtype=float), 1.0)
    f = np.asarray(flux, dtype=float)
    if p.size == 0:
        raise ValueError("periodic_model needs at least one point.")
    order = np.argsort(p, kind="stable")
    p, f = p[order], f[order]
    keep = np.concatenate(([True], np.diff(p) > 0))
    p, f = p[keep], f[keep]
    return (np.concatenate(([p[-1] - 1.0], p, [p[0] + 1.0])),
            np.concatenate(([f[-1]], f, [f[0]])))


def eval_periodic(
    phase_ext: np.ndarray,
    flux_ext: np.ndarray,
    phases,
    shift=0.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Model from :func:`periodic_model` at ``phases - shift``, plus *offset*.

    *shift* may be an array broadcast against *phases* to evaluate many trial
    shifts at once; *offset* is the additive scattered-flux floor.
    """
    ph = np.mod(np.asarray(phases, dtype=float) - np.asarray(shift, dtype=float), 1.0)
    return np.interp(ph.ravel(), phase_ext, flux_ext).reshape(ph.shape) + float(offset)


def prepare_model_interpolator(
    sim_df: pd.DataFrame, sim_column: str
) -> Tuple[np.ndarray, np.ndarray]:
    """:func:`periodic_model` of a simulation CSV column (``phase`` or ``deg`` x-axis)."""
    if "phase" in sim_df.columns:
        sim_phase = sim_df["phase"].to_numpy(dtype=float)
    elif "deg" in sim_df.columns:
        sim_phase = sim_df["deg"].to_numpy(dtype=float) / 360.0
    else:
        raise ValueError("Simulation file must contain 'phase' or 'deg' column.")
    if sim_column not in sim_df.columns:
        raise KeyError(f"Column '{sim_column}' not found in simulation DataFrame.")
    return periodic_model(sim_phase, sim_df[sim_column].to_numpy(dtype=float))


def sanitize_errors(errors, context: str = "") -> np.ndarray:
    """Measurement errors with non-finite or non-positive entries patched.

    Bad entries are replaced by the median of the valid errors and a warning
    says how many were patched. There is deliberately no absolute floor and no
    Poisson ``sqrt(rate)`` fallback: both depend on the units of the data (a
    ``1e-3`` floor zero-weights a flux point of ``1e-13``, and ``sqrt`` of a
    flux is not a count error). If no error is valid a ``ValueError`` is
    raised: a χ² fit without measurement errors is not meaningful. This is the
    one repair rule, applied at the load boundary (:func:`load_observed_lightcurves`,
    ``chandra_phase_analysis.main``, ``mcmc_lightcurve_fit.load_fit_data``) and
    by :func:`obs_errors`; the binners receive already-valid errors.
    """
    err = np.array(errors, dtype=float, copy=True)
    bad = ~np.isfinite(err) | (err <= 0)
    if bad.any():
        valid = err[~bad]
        if valid.size == 0:
            raise ValueError(f"{context}no valid measurement errors (all non-finite or <= 0).")
        fill = float(np.median(valid))
        err[bad] = fill
        warnings.warn(f"{context}patched {int(bad.sum())} of {err.size} non-finite/non-positive "
                      f"errors with the median valid error {fill:.4g}.")
    return err


def obs_errors(
    obs_df: pd.DataFrame,
    rate_column: str = "rate",
    error_column: str = "error",
) -> np.ndarray:
    """Observation uncertainties for the χ² fit and the residual panel.

    Requires an error column (see :func:`sanitize_errors` for the repair
    rule) and raises ``ValueError`` when the observations carry none.
    """
    if error_column not in obs_df.columns or obs_df[error_column].isnull().all():
        raise ValueError(
            "The observations carry no measurement errors; a chi2 fit needs them "
            "(check --obs-error-column and the file header).")
    return sanitize_errors(obs_df[error_column])


# -----------------------------------------------------------------------------
# Chi-square fitting of a tabulated model
# -----------------------------------------------------------------------------

def fit_simulation(
    obs_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    sim_column: str = "fl",
    fit_phase_shift: bool = False,
    scatter: float = 0.0,
    n_shift_grid: Optional[int] = None,
    fixed_shift: float = 0.0,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Fit a tabulated simulation light curve to observations by χ².

    Only the **phase shift** (x-direction) is fitted. There is deliberately no
    multiplicative flux scale: the model's absolute normalization is already
    fixed by the wind mass-loss rate (via the physical column-density
    normalization) together with the XSPEC ``flux vs nH`` table, so a free
    y-scale would silently absorb an error in that normalization instead of
    exposing it. The only y-direction freedom is the *additive* ``scatter``
    floor, which is supplied by the caller (measured at mid-eclipse) rather
    than fitted here. The MCMC likelihood uses the same
    :func:`best_phase_shift` search and the same additive ``f_scatter``.

    Parameters
    ----------
    obs_df : DataFrame
        Observational data with columns ``phase``, ``rate`` and ``error``.
    sim_df : DataFrame
        Simulation results with ``phase`` (or ``deg``) and *sim_column*.
    sim_column : str, default ``"fl"``
        Column in *sim_df* to use as the model flux.
    fit_phase_shift : bool, default False
        If True, search the phase shift that minimizes χ²; otherwise evaluate
        χ² at shift = 0.
    scatter : float, default 0.0
        Constant additive scattered-flux floor added to the model, never scaled.
    n_shift_grid : int, optional
        Coarse trial shifts over [0, 1); see :func:`build_phase_shift_search`
        for the default.
    fixed_shift : float, default 0.0
        Phase shift applied when *fit_phase_shift* is False (e.g. the shift of
        a full-orbit fit when fitting a phase window).
    verbose : bool, default True
        Print the fitted shift, scatter and reduced χ².

    Returns
    -------
    (phase_shift, reduced_chi2)
        Best-fit phase shift in [0, 1) (0.0 when not fitted) and χ²/dof.
    """
    phase_obs = np.mod(obs_df["phase"].to_numpy(dtype=float), 1.0)
    rate_obs = obs_df["rate"].to_numpy(dtype=float)
    err_obs = obs_errors(obs_df)

    # Prepared once; the χ² here, the overlay drawn by plot_phase and the model
    # dump all evaluate this same periodic model.
    phase_ext, flux_ext = prepare_model_interpolator(sim_df, sim_column)
    flux_ext = flux_ext + float(scatter)

    if fit_phase_shift:
        search = build_phase_shift_search(phase_obs, n_grid=n_shift_grid, n_model=len(sim_df))
        _, best_shift, chi2 = best_phase_shift(phase_ext, flux_ext, rate_obs, err_obs ** 2, search)
    else:
        best_shift = float(fixed_shift) % 1.0
        model = eval_periodic(phase_ext, flux_ext, phase_obs, shift=best_shift)
        chi2 = float(np.sum(((rate_obs - model) / err_obs) ** 2))

    n_free = int(fit_phase_shift)
    dof = max(len(rate_obs) - n_free, 1)
    reduced_chi2 = chi2 / dof
    if verbose:
        print(
            f"{'Best-fit phase shift' if fit_phase_shift else 'Chi-square at zero shift'} "
            f"(no flux rescaling):\n"
            f"  Phase shift = {best_shift:.5f}{'' if fit_phase_shift else ' (held fixed)'}\n"
            f"  Scattered flux = {float(scatter):.6g} (fixed, additive)\n"
            f"  Reduced χ² = {reduced_chi2:.3f}  (dof = {dof})"
        )
    return float(best_shift), float(reduced_chi2)


def write_model_blocks(f, model_phase, model_flux, obs_phase, obs_flux, obs_err, obs_model) -> None:
    """Write the two data blocks of a model dump to an open text file.

    BLOCK 1 is the dense model curve (phase already shifted to the observed
    frame, any additive floor included), BLOCK 2 the observed points with the
    model at their phases and the normalized residual. Both are plain
    whitespace-delimited tables under ``#`` comments, so
    ``np.genfromtxt(..., names=True)`` reads either after selecting its rows.
    Shared by :func:`write_model_lightcurve` (tabulated fit) and
    ``mcmc_lightcurve_fit._write_bestfit_model_txt``.
    """
    model_phase = np.asarray(model_phase, dtype=float)
    model_flux = np.asarray(model_flux, dtype=float)
    obs_phase = np.asarray(obs_phase, dtype=float)
    obs_flux = np.asarray(obs_flux, dtype=float)
    obs_err = np.asarray(obs_err, dtype=float)
    obs_model = np.asarray(obs_model, dtype=float)

    f.write("#\n# --- BLOCK 1: dense model curve (phase already shifted to the "
            "observed frame, scatter added) ---\n")
    f.write("phase model_flux\n")
    for p_val, flux_val in zip(model_phase, model_flux):
        f.write(f"{p_val:.8f} {flux_val:.8e}\n")

    with np.errstate(divide="ignore", invalid="ignore"):
        resid = (obs_flux - obs_model) / obs_err
    f.write("#\n# --- BLOCK 2: observed bins vs model ---\n")
    f.write("phase obs_flux obs_err model_flux resid_sigma\n")
    for idx in np.argsort(obs_phase):
        f.write(f"{obs_phase[idx]:.8f} {obs_flux[idx]:.8e} {obs_err[idx]:.8e} "
                f"{obs_model[idx]:.8e} {resid[idx]:.6f}\n")


def tabulated_model_arrays(
    obs_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    sim_column: str,
    shift: float,
    scatter: float = 0.0,
    n_model_points: int = MODEL_OVERLAY_N_POINTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The tabulated model as drawn and dumped: ``(model_phase, model_flux,
    obs_phase, obs_model)`` for the given shift and additive floor.

    One evaluation shared by ``plot_utils.plot_phase`` and
    :func:`write_model_lightcurve`, so the plotted overlay and the dumped curve
    are the same arrays.
    """
    phase_ext, flux_ext = prepare_model_interpolator(sim_df, sim_column)
    obs_phase = np.mod(obs_df["phase"].to_numpy(dtype=float), 1.0)
    model_phase = np.linspace(0.0, 1.0, int(n_model_points))
    model_flux = eval_periodic(phase_ext, flux_ext, model_phase, shift, scatter)
    obs_model = eval_periodic(phase_ext, flux_ext, obs_phase, shift, scatter)
    return model_phase, model_flux, obs_phase, obs_model


def write_model_lightcurve(
    path: str,
    obs_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    sim_column: str,
    shift: float,
    scatter: float = 0.0,
    *,
    shift_fitted: bool = False,
    obs_column: str = "rate",
    sim_file: Optional[str] = None,
    n_model_points: int = MODEL_OVERLAY_N_POINTS,
    verbose: bool = True,
) -> str:
    """Write the fitted tabulated model light curve to a text file.

    The point of this file is that the *fitted* model is not the ``--sim-file``
    contents: :func:`fit_simulation` slides the model in phase and adds the
    ``scatter`` floor, so reproducing the drawn curve from the simulation CSV
    alone means re-applying both by hand. Here they are already applied, and the
    header records them so the transformation stays auditable. Every model
    value routes through :func:`eval_periodic`, the same evaluator used by the
    χ² and the plot overlay; the blocks come from :func:`write_model_blocks`.

    Returns the path written.
    """
    shift = float(shift)
    scatter = float(scatter)
    model_phase, model_flux, obs_phase, obs_model = tabulated_model_arrays(
        obs_df, sim_df, sim_column, shift, scatter, n_model_points)
    obs_flux = obs_df["rate"].to_numpy(dtype=float)
    obs_err = obs_errors(obs_df)

    n_free = 1 if shift_fitted else 0
    dof = max(len(obs_flux) - n_free, 1)
    chi2_total = float(np.sum(((obs_flux - obs_model) / obs_err) ** 2))

    parent = os.path.dirname(str(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w") as f:
        f.write(f"# Fitted model light curve -- {band_label_from_column(sim_column)} "
                f"band, column '{sim_column}'\n")
        if sim_file:
            f.write(f"# sim_file: {sim_file}\n")
        f.write(f"# obs_column: {obs_column}\n")
        f.write(f"# phase_shift applied to model: {shift:.6f}"
                f"  ({'fitted' if shift_fitted else 'held fixed'})\n")
        f.write(f"# scattered flux added to model: {scatter:.8g}  (constant, additive)\n")
        f.write("# No multiplicative flux rescaling is applied -- the model keeps its "
                "native normalization.\n")
        f.write(f"# chi2/dof: {chi2_total / dof:.6g}  (chi2 = {chi2_total:.6g}, "
                f"dof = {dof} = {len(obs_flux)} bins - {n_free} free)\n")
        write_model_blocks(f, model_phase, model_flux, obs_phase, obs_flux, obs_err, obs_model)

    if verbose:
        print(f"Model light curve written to: {path}")
    return str(path)


# -----------------------------------------------------------------------------
# Periodic phase-shift search
# -----------------------------------------------------------------------------
# The model's phase zero is not tied to the ephemeris, so every comparison with
# data allows a free shift. chi2(shift) is periodic and strongly multi-modal
# (the eclipse), so a local optimizer started at 0 settles in the wrong basin:
# the search scans the full period on a grid fine enough to resolve every basin
# first, then refines densely around the best node. One implementation serves
# fit_simulation (tabulated model) and the MCMC likelihood (kernel model).

@dataclass(frozen=True)
class PhaseShiftSearch:
    """Trial shifts precomputed once for an observed phase array."""
    obs_phase: np.ndarray           # observed phases folded into [0, 1)
    shift_grid: np.ndarray          # coarse trial shifts in [0, 1)
    shifted_obs_phase: np.ndarray   # (n_grid, n_obs): mod(obs_phase - shift, 1)
    n_fine: int = PHASE_SHIFT_FINE_POINTS
    n_levels: int = PHASE_SHIFT_LEVELS

    @property
    def resolution(self) -> float:
        """Shift spacing of the final dense pass."""
        step = 1.0 / self.shift_grid.size
        for _ in range(self.n_levels):
            step = 2.0 * step / (self.n_fine - 1)
        return step


def build_phase_shift_search(
    obs_phase: np.ndarray,
    n_grid: Optional[int] = None,
    n_model: int = 0,
    n_fine: int = PHASE_SHIFT_FINE_POINTS,
    n_levels: int = PHASE_SHIFT_LEVELS,
) -> PhaseShiftSearch:
    """Precompute the coarse trial shifts for :func:`best_phase_shift`.

    The coarse step must not exceed the narrowest feature of χ²(shift), which
    is set by the data spacing (a bin crossing the eclipse edge) and by the
    model spacing (one ``dth``), so by default ``n_grid = max(n_obs, n_model)``
    clipped to ``[PHASE_SHIFT_MIN_GRID, PHASE_SHIFT_MAX_GRID]``. The cap keeps
    the scan affordable for unbinned data, whose χ²(shift) is smooth anyway.
    """
    obs_phase = np.mod(np.asarray(obs_phase, dtype=float), 1.0)
    if n_grid is None:
        n_grid = min(max(PHASE_SHIFT_MIN_GRID, obs_phase.size, int(n_model)), PHASE_SHIFT_MAX_GRID)
    n_grid = max(3, int(n_grid))
    shift_grid = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    return PhaseShiftSearch(
        obs_phase=obs_phase,
        shift_grid=shift_grid,
        shifted_obs_phase=np.mod(obs_phase[None, :] - shift_grid[:, None], 1.0),
        n_fine=max(3, int(n_fine)),
        n_levels=max(0, int(n_levels)),
    )


def best_phase_shift(
    phase_ext: np.ndarray,
    flux_ext: np.ndarray,
    obs_flux: np.ndarray,
    obs_err2: np.ndarray,
    search: PhaseShiftSearch,
) -> Tuple[np.ndarray, float, float]:
    """Shift of a periodic model that minimizes χ² against the observations.

    One ``np.interp`` over the precomputed ``(n_grid, n_obs)`` trial-phase
    matrix gives χ² at every coarse shift; ``search.n_levels`` dense passes of
    ``search.n_fine`` points, each spanning ±1 previous step around the best
    shift, then bring the resolution to ``search.resolution``. The model is
    ``(phase_ext, flux_ext)`` from :func:`periodic_model`, with any additive
    floor already included in ``flux_ext``.

    Returns ``(model_at_obs_phases, shift, chi2)``.
    """
    obs_flux = np.asarray(obs_flux, dtype=float)
    obs_err2 = np.asarray(obs_err2, dtype=float)

    def scan(shifted):
        model = np.interp(shifted.ravel(), phase_ext, flux_ext).reshape(shifted.shape)
        chi2 = np.sum((obs_flux - model) ** 2 / obs_err2, axis=1)
        j = int(np.argmin(chi2))
        return float(chi2[j]), j, model[j]

    best_chi2, j, best_model = scan(search.shifted_obs_phase)
    if not np.isfinite(best_chi2):
        raise ValueError("best_phase_shift: the model or the data contain non-finite values.")
    best_shift = float(search.shift_grid[j])
    step = 1.0 / search.shift_grid.size
    for _ in range(search.n_levels):
        cand = best_shift + np.linspace(-step, step, search.n_fine)
        chi2, j, model = scan(np.mod(search.obs_phase[None, :] - cand[:, None], 1.0))
        if chi2 < best_chi2:
            best_chi2, best_shift, best_model = chi2, float(cand[j] % 1.0), model
        step = 2.0 * step / (search.n_fine - 1)
    return best_model, best_shift, best_chi2


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

# Options that define the *fit* are restored by --replot; options that only
# control one invocation (the caller passes them as `never_restore`, derived
# from its Execution/Output argument groups) are not. `replot` itself must never
# be restored (a saved fit recorded replot=False, which would cancel the
# replot) and `output_dir` is defined by where the config was found.
# store_true flags cannot be negated on the command line, so restoring
# no_plots/save_chi2/... would make a fit run with --no-plots impossible to
# replot with figures.
_RUN_CONFIG_NEVER_RESTORE = frozenset({"replot", "output_dir"})

# Mutually exclusive option groups: typing one member on --replot must not
# restore a saved sibling, which would trip the exclusivity check.
_RUN_CONFIG_EXCLUSIVE_GROUPS = (
    frozenset({"n_phase_bins", "counts_per_bin", "no_phase_bin"}),
    frozenset({"reparam", "kepler", "kepler_mtot"}),
    frozenset({"no_fit_phase_shift", "phase_shift"}),
)

# Stamped into every new run config. i0 was formerly measured from the line of
# sight, so a chain written before the switch stores the complement of what the
# priors and plots now mean; --replot cannot detect that from the numbers alone.
INCLINATION_CONVENTION = "i0-from-orbital-normal"

# Stamped into every run config and chain file. Results sampled under the
# retired `lam` normalization cannot be re-evaluated with the physical
# Mdot / v_inf normalization: chi2, overlays and BIC would describe a different
# model than the posterior, so --replot refuses them.
WIND_NORMALIZATION = "physical-mdot-vinf"


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
        "inclination_convention": INCLINATION_CONVENTION,
        "wind_normalization": WIND_NORMALIZATION,
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
    """Saved run-config paths in *output_dir*, optionally filtered."""
    b = band if band else "*"
    w = wind_model if wind_model else "*"
    return sorted(glob.glob(os.path.join(output_dir, f"{b}_{w}{RUN_CONFIG_SUFFIX}")))


def dest_to_flag(parser: argparse.ArgumentParser) -> Dict[str, str]:
    """dest -> the long option a user would type (``prior_M_X`` is spelled ``--prior-MX``)."""
    out: Dict[str, str] = {}
    for action in parser._actions:
        longs = [o for o in action.option_strings if o.startswith("--")]
        if longs:
            out[action.dest] = longs[0]
    return out


def validate_binning_args(err, args) -> None:
    """Binning options shared by both fitters: exclusivity and positivity."""
    if args.no_phase_bin and (args.n_phase_bins is not None or args.counts_per_bin is not None):
        err("--no-phase-bin excludes --n-phase-bins and --counts-per-bin.")
    if args.n_phase_bins is not None and args.counts_per_bin is not None:
        err("Specify either --n-phase-bins (fixed-width) or --counts-per-bin (constant counts), not both.")
    if args.n_phase_bins is not None and args.n_phase_bins <= 0:
        err("--n-phase-bins must be > 0.")
    if args.counts_per_bin is not None and args.counts_per_bin <= 0:
        err("--counts-per-bin must be > 0.")


def validate_phase_window_args(err, args, fit_shift_enabled: bool, fixed_shift_hint: str,
                               scatter_window_used: bool) -> Tuple[float, float, bool]:
    """Phase-window rules shared by both fitters; returns ``(lo, hi, partial)``.

    A partial window with the shift search enabled is rejected (the symmetric
    model makes the eclipse width degenerate with a free shift when only one
    edge is in the data), and the scattered-flux window must overlap the data
    window whenever the floor is estimated from the data.
    """
    try:
        lo, hi = check_phase_window(*args.phase_window)
    except ValueError as e:
        err(f"--phase-window: {e}")
    partial = not is_full_phase_window(lo, hi)
    if partial and fit_shift_enabled:
        err("A partial --phase-window needs a fixed phase shift. The model is symmetric about "
            "mid-eclipse, so with only one eclipse edge in the data the eclipse width is "
            f"degenerate with a free shift: {fixed_shift_hint}")
    s_lo, s_hi = map(float, args.scatter_eclipse_phase)
    if not (0.0 <= s_lo <= s_hi <= 1.0):
        err("--scatter-eclipse-phase must satisfy 0 <= PHASE_MIN <= PHASE_MAX <= 1.")
    if partial and scatter_window_used:
        # Positive-length overlap: a window that only touches the open end [lo, hi)
        # of the data window holds no data points.
        if not any(max(a, s_lo) < min(b, s_hi) for a, b in phase_window_intervals(lo, hi)):
            err(f"--scatter-eclipse-phase {s_lo:g} {s_hi:g} lies outside --phase-window "
                f"{lo:g} {hi:g}, so the scattered flux cannot be estimated from the data.")
    return lo, hi, partial


def explicit_cli_dests(parser: argparse.ArgumentParser, argv: Optional[List[str]] = None) -> set:
    """Argparse dests corresponding to options the user actually typed.

    Comparing against ``parser.get_default()`` is not enough: a user who
    explicitly passes the default value should still beat a saved config, and
    an option that has no effect in the chosen configuration should be
    rejected only when it was typed. Unambiguous prefixes are resolved as
    argparse resolves them; anything unrecognized (a negative number used as a
    value) is ignored.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
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
    explicit: Optional[set] = None,
    never_restore=(),
) -> Optional[str]:
    """Fill in options the user did not type from a previous run's config.

    Intended for ``--replot``, so that flag alone reproduces the original run's
    band, wind model, data selection, binning and priors. Explicit command-line
    values always win (*explicit* is the typed set from
    :func:`explicit_cli_dests`, computed here when not given), and the dests
    in *never_restore* (invocation-only options) plus ``replot``/``output_dir``
    are never restored. Returns the config path used, or None.
    """
    explicit = explicit_cli_dests(parser, argv) if explicit is None else set(explicit)
    never = _RUN_CONFIG_NEVER_RESTORE | set(never_restore)

    candidates = find_run_configs(
        args.output_dir,
        band=(args.band if 'band' in explicit else None),
        wind_model=(args.wind_model if 'wind_model' in explicit else None),
    )
    if not candidates:
        return None

    if len(candidates) > 1:
        # Several fits share this directory. Configs differing only by band
        # restore identically, so take the first; otherwise ask the user to
        # disambiguate.
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

    if config.get("inclination_convention") != INCLINATION_CONVENTION:
        warnings.warn(
            f"{os.path.basename(config_path)} predates the inclination convention "
            f"change: its i0 samples and --prior-i0 are measured from the line of "
            f"sight, whereas i0 is now measured from the orbital-plane normal "
            f"(90 deg = edge-on). Its i0 values mean the complement of what the "
            f"model now expects, so any chi2 reported from this chain is "
            f"meaningless. Refit before trusting the output."
        )
    if config.get("wind_normalization") != WIND_NORMALIZATION:
        parser.error(
            f"{os.path.basename(config_path)} predates the physical wind normalization "
            f"(no 'wind_normalization: {WIND_NORMALIZATION}' stamp). Its chain was sampled "
            f"under a different model, so --replot would report chi2, overlays and BIC for a "
            f"model the posterior never saw. Refit instead."
        )

    flag_of = dest_to_flag(parser)
    known_dests = {a.dest for a in parser._actions}

    blocked = set(never)
    for group in _RUN_CONFIG_EXCLUSIVE_GROUPS:
        if explicit & group:
            blocked |= group

    restored: List[Tuple[str, object]] = []
    for dest, value in saved_args.items():
        if dest in blocked or dest in explicit or dest not in known_dests:
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
            print(f"    {flag_of.get(dest, '--' + dest)} = {value!r}")
    else:
        print("    (nothing to restore — command line already matches)")
    overridden = sorted(
        flag_of.get(d, '--' + d) for d in explicit
        if d not in never and d in saved_args
    )
    if overridden:
        print(f"  kept from the command line: {', '.join(overridden)}")
    return config_path
