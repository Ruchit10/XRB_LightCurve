#!/usr/bin/env python3
"""
Chandra Phase Analysis
----------------------
This script replicates and extends the phase–conversion logic from the original
R prototype (`light_curve_model_opt_bw.R`).  It

1.  Reads either the master `Chandra.txt` file or all `chandra*.txt` files in a
    data directory.
2.  Converts observation times to orbital phase using the reference epoch and
    orbital period found in the R script.
3.  Optionally verifies that every point in the individual `chandra*.txt` files
    is present in the master file.
4.  Produces a scatter plot of count-rate versus orbital phase, colour-coded by
    observation.

Example
~~~~~~~
$ python chandra_phase_analysis.py --data-dir data --output chandra_phase_plot.png --verify-master

Dependencies: numpy, pandas, matplotlib (already listed in requirements.txt).
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import List

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


def read_observation(file_path: str, label: str) -> pd.DataFrame:
    """Read a single Chandra observation text file.

    The files are whitespace-delimited with three columns:
        1. time (seconds)
        2. count rate / flux
        3. formal error (ignored here)
    """
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
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

def load_data(data_dir: str) -> pd.DataFrame:
    """Load Chandra data from *data_dir*.

    Preference order:
        1. Use master `Chandra.txt` if it exists.
        2. Otherwise, concatenate all `chandra*.txt` files.
    """
    master_file = os.path.join(data_dir, "Chandra.txt")
    individual_pattern = os.path.join(data_dir, "chandra*.txt")

    if os.path.isfile(master_file):
        print(f"Using master file: {master_file}")
        return read_observation(master_file, "master")

    # Fall back to individual files
    files: List[str] = sorted(glob.glob(individual_pattern))
    if not files:
        raise FileNotFoundError(
            f"No Chandra data found in {data_dir} (expected 'Chandra.txt' or 'chandra*.txt')"
        )

    dfs = [read_observation(fp, os.path.basename(fp)) for fp in files]
    print(f"Loaded {len(files)} individual observation file(s).")
    return pd.concat(dfs, ignore_index=True)


def verify_master_contains_individual(data_dir: str) -> None:
    """Check that every timestamp in `chandra*.txt` appears in `Chandra.txt`."""
    master_file = os.path.join(data_dir, "Chandra.txt")
    if not os.path.isfile(master_file):
        print("No master 'Chandra.txt' found; skipping verification.")
        return

    print("Verifying individual files against master...\n")
    master_times = (
        pd.read_csv(master_file, delim_whitespace=True, header=None, usecols=[0])[0]
        .round(6)
        .astype(str)
        .tolist()
    )
    master_set = set(master_times)

    for fp in glob.glob(os.path.join(data_dir, "chandra*.txt")):
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

def fit_simulation(obs_df: pd.DataFrame, sim_df: pd.DataFrame, sim_column: str = "fl") -> tuple[float, float, float]:
    """Fit simulation light-curve to observations via chi-square minimization.

    Parameters
    ----------
    obs_df : DataFrame
        Observational data with columns ``phase``, ``rate`` and (optionally) ``error``.
    sim_df : DataFrame
        Simulation results. Must contain columns ``phase`` (or ``deg``) and *sim_column*.
    sim_column : str, default ``"fl"``
        Column in *sim_df* to use as the model flux.

    Returns
    -------
    (phase_shift, scale_factor, reduced_chi2)
        Best-fit phase shift (0–1), multiplicative scale factor, and reduced chi-squared value.
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


def plot_phase(
    df: pd.DataFrame,
    output_path: str | None,
    sim_df: pd.DataFrame | None = None,
    shift: float | None = None,
    scale: float | None = None,
    sim_column: str = "fl",
    chi2: float | None = None,
    ax: plt.Axes | None = None,
) -> None:
    """Scatter plot with optional best-fit simulation overlay.
    
    Parameters
    ----------
    df : DataFrame
        Observational data with columns ``phase``, ``rate``.
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
    """
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    for label, group in df.groupby("obs"):
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
    ax.set_ylabel("Count rate / Flux")
    
    # Set title with chi-squared annotation if provided
    if chi2 is not None:
        ax.set_title(f"{sim_column}\nReduced χ² = {chi2:.3f}")
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
        plot_phase(df, None, sim_df, shift, scale, col, chi2, ax)
    
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
        description="Convert Chandra observation times to orbital phase and plot the data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing Chandra observation text files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="chandra_phase_plot.png",
        help="Output filename for the generated plot. If omitted, the plot is shown interactively.",
    )
    parser.add_argument(
        "--verify-master",
        action="store_true",
        help="Verify that each individual 'chandra*.txt' file is contained in 'Chandra.txt'.",
    )
    parser.add_argument(
        "--sim-file",
        type=str,
        default=None,
        help="CSV file containing simulation results to fit.",
    )
    parser.add_argument(
        "--sim-column",
        type=str,
        nargs='+',
        default=["nfl_soft_av", "nfl_hard_av", "nfl_soft_cv", "nfl_hard_cv"],
        help="Column name(s) in simulation CSV to use as model flux. Can specify multiple columns separated by spaces.",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Perform χ² minimization to fit simulation to observations.",
    )

    args = parser.parse_args()

    if args.verify_master:
        verify_master_contains_individual(args.data_dir)

    df = load_data(args.data_dir)
    print(f"Loaded {len(df)} data point(s) from {df['obs'].nunique()} observation(s).")

    if args.fit:
        if not args.sim_file:
            parser.error("--fit requires --sim-file to be specified.")
        sim_df = pd.read_csv(args.sim_file)
        
        # Handle multiple columns
        sim_columns = args.sim_column if isinstance(args.sim_column, list) else [args.sim_column]
        
        # Fit each column
        fit_results = []
        for col in sim_columns:
            print(f"\n{'='*60}")
            print(f"Fitting column: {col}")
            print('='*60)
            shift, scale, chi2 = fit_simulation(df, sim_df, col)
            fit_results.append((shift, scale, chi2))
        
        # Plot based on number of columns
        if len(sim_columns) == 1:
            # Single column: use original plot
            shift, scale, chi2 = fit_results[0]
            plot_phase(df, args.output, sim_df, shift, scale, sim_columns[0], chi2)
        else:
            # Multiple columns: use grid plot
            plot_multi_column_fits(df, args.output, sim_df, sim_columns, fit_results)
    else:
        plot_phase(df, args.output)


if __name__ == "__main__":
    main() 