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

Implementation note
~~~~~~~~~~~~~~~~~~~
This file is now only the command-line front end. The analysis routines live in
``utils/utils.py`` (loading, phase binning, smoothing, periodic model
interpolation, the χ² fit) and every plot is drawn by
``utils/plot_utils.plot_lightcurve_fit`` — the same function
``mcmc_lightcurve_fit.py`` uses — so the two scripts share one implementation.
All of those names are re-exported here, so ``from chandra_phase_analysis import
*`` still works.

Dependencies: numpy, pandas, matplotlib, scipy (in requirements.txt).
"""
from __future__ import annotations

import argparse

import pandas as pd

# Every analysis helper lives in utils/ so that this script and
# mcmc_lightcurve_fit.py share one implementation instead of importing from each
# other. The names are re-exported below, so `from chandra_phase_analysis import
# *` (used by the notebooks) keeps working unchanged.
from utils.utils import (
    ORBITAL_PERIOD,
    REF_EPOCH,
    band_label_from_column,
    detect_flux_columns,
    estimate_scattered_flux,
    evaluate_model_at_phases,
    fit_simulation,
    frac,
    interp_periodic_phases,
    load_data,
    model_from_wrap,
    obs_errors,
    phase_bin_data,
    phase_bin_data_snr,
    prepare_model_interpolator,
    read_observation,
    smooth_lightcurve,
    validate_sim_columns,
)
from utils.plot_utils import (
    add_residual_panel,
    plot_lightcurve_fit,
    plot_multi_column_fits,
    plot_phase,
)

__all__ = [
    "ORBITAL_PERIOD",
    "REF_EPOCH",
    "add_residual_panel",
    "band_label_from_column",
    "detect_flux_columns",
    "estimate_scattered_flux",
    "evaluate_model_at_phases",
    "fit_simulation",
    "frac",
    "interp_periodic_phases",
    "load_data",
    "main",
    "model_from_wrap",
    "obs_errors",
    "phase_bin_data",
    "phase_bin_data_snr",
    "plot_lightcurve_fit",
    "plot_multi_column_fits",
    "plot_phase",
    "prepare_model_interpolator",
    "read_observation",
    "smooth_lightcurve",
    "validate_sim_columns",
]


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