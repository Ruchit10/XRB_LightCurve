#!/usr/bin/env python3
"""
Simulation of column densities for eclipsing binary systems.
This module simulates the column densities obtained as the compact object 
eclipses companion in a Binary System orbiting a common Center of Mass.

The Compact object and the Accretion disk is referred to as Star B
The Companion Star is referred to as Star A
All Distance units are in Solar Radii
All Angle units are converted into radians for trigonometric functions
"""

import argparse
import numpy as np
import pandas as pd
import math
import os
import sys
import warnings
from typing import Tuple, List, Optional, Dict
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# Cache for fast converged LOS integration of the accelerating-wind integrand:
#   ∫ (1 + u^2)^(-5/4) du  for u in [0, U_MAX]
_ACCEL_U_GRID: Optional[np.ndarray] = None
_ACCEL_F_GRID: Optional[np.ndarray] = None
_ACCEL_U_MAX: float = 1e5


def create_grid(
    r: float, l: float, R: float, gma: float, d2h: float = 6.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create grid for wind integral of eclipsing binaries.

    Args:
        r: Radius of smaller star B (compact object)
        l: Separation along viewing plane
        R: Radius of larger star A (companion)
        gma: Phase angle in radians
        d2h: Angle size for polar grid cell (degrees)

    Returns:
        Tuple of (av_x, av_th, av_db, A) arrays
    """
    # Create polar grid
    g1_r = np.linspace(r / 10, r, 10)
    g1_th = np.linspace(0, 2 * np.pi, int(360 / d2h) + 1)

    # Expand grid
    g1_r_mesh, g1_th_mesh = np.meshgrid(g1_r, g1_th)
    g1_r_flat = g1_r_mesh.flatten()
    g1_th_flat = g1_th_mesh.flatten()

    # Filter points based on conditions
    # Only apply eclipse filtering when emitter is BEHIND companion (sin(gma) > 0)
    # When emitter is in front (sin(gma) <= 0), no occultation is possible
    if np.sin(gma) > 0:
        # Calculate distance from center - filter out points blocked by companion
        nn = np.sqrt(g1_r_flat**2 + l**2 - 2 * g1_r_flat * l * np.cos(g1_th_flat))
        mask = nn >= R
        g1_s_r = g1_r_flat[mask]
        g1_s_th = g1_th_flat[mask]
    else:
        # Emitter is in front of companion - all points visible
        g1_s_r = g1_r_flat
        g1_s_th = g1_th_flat

    if g1_s_r.size < 2:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    # Vectorized segment construction (adjacent pairs in flattened order)
    x1 = g1_s_r[:-1]
    x2 = g1_s_r[1:]
    th1 = g1_s_th[:-1]
    dx = x2 - x1
    valid = dx > 0

    if not np.any(valid):
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    av_x = 0.5 * (x1[valid] + x2[valid])
    av_th = 0.5 * (th1[valid] + (th1[valid] + (d2h * np.pi / 180.0)))
    av_db = np.sqrt(av_x**2 + l**2 - 2.0 * av_x * l * np.cos(av_th))
    A = 0.5 * (d2h * np.pi / 180.0) * ((x2[valid] ** 2) - (x1[valid] ** 2))

    return av_x.astype(float), av_th.astype(float), av_db.astype(float), A.astype(float)


def density_function(
    d: float, l: float, gma: float, i: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate wind density along the line of sight.

    Args:
        d: Separation between stars
        l: Separation along viewing plane
        gma: Phase angle in radians
        i: Inclination angle in radians

    Returns:
        Tuple of (z5, colm) arrays
    """
    dz = 0.1
    d4 = d
    z4 = d * np.sin(gma) * np.cos(i)
    t4 = np.sqrt((2 * d) ** 2 - l**2)

    z5 = []
    colm = []

    while abs(z4) <= t4:
        z5.append(z4)
        colm.append(d4 ** (-2))
        z4 = z4 - 0.1
        d4 = np.sqrt(l**2 + z4**2)

    return np.array(z5), np.array(colm)


def wind_los_integral(
    d: float,
    d1: float,
    d2: float,
    gma: float,
    i: float,
    av_x: np.ndarray,
    av_th: np.ndarray,
    av_db: np.ndarray,
    A: np.ndarray,
    dz: float = 0.1,
    Rmax: Optional[float] = None,
    converge_rmax: bool = False,
    conv_eps_rel: float = 1e-6,
    conv_eps_abs: float = 1e-12,
    conv_min_steps: int = 50,
    conv_r_cap_mult: float = 1000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate column integral along the line of sight.

    Uses vectorized summation over the line-of-sight grid to avoid Python loops.

    Args:
        d: Separation between stars
        d1: Distance of star B from COM
        d2: Distance of star A from COM
        gma: Phase angle in radians
        i: Inclination angle in radians
        av_x, av_th, av_db: Grid arrays
        A: Area array
        dz: Step along line of sight (solar radii)
        Rmax: Maximum radius (solar radii) for LOS integration cutoff. If None, defaults to 2*d.
        converge_rmax: If True, ignore Rmax and integrate adaptively until tail contributions become negligible.
        conv_eps_rel: Relative convergence tolerance for adaptive stopping (both wind models).
        conv_eps_abs: Absolute convergence tolerance for adaptive stopping (both wind models).
        conv_min_steps: Minimum number of dz-steps before convergence checks start.
        conv_r_cap_mult: Safety cap for adaptive integration, as multiple of d (stop when r >= cap).

    Returns:
        Tuple of (lw, lw2, icd, A2) arrays
    """
    if av_db is None or av_db.size == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    # z start and bounds (identical for each cell within a phase)
    z1 = d1 * np.sin(gma) * np.cos(i)
    z2 = d2 * np.sin(gma) * np.cos(i)
    z_start = z1 + z2

    # If requested (or if Rmax is None), integrate to -infinity (observer) using fast closed-forms / cached quadrature.
    #
    # This is MUCH faster than stepping by dz to large radii, and is effectively "fully converged".
    # - Constant-velocity wind term uses exact primitive: ∫ dz / (b^2 + z^2) = (1/b) arctan(z/b)
    # - Accelerating wind term integrates (b^2 + z^2)^(-5/4). With z=b*u, dz=b du:
    #     ∫ (b^2 + z^2)^(-5/4) dz = b^(-3/2) ∫ (1 + u^2)^(-5/4) du
    if converge_rmax or (Rmax is None):
        global _ACCEL_U_GRID, _ACCEL_F_GRID, _ACCEL_U_MAX

        b = av_db.astype(float)
        b_safe = np.maximum(b, 1e-8)
        u0 = float(z_start) / b_safe  # vector
        u_abs = np.abs(u0)

        # Exact constant-velocity integral from -∞ to z_start
        # I2 = (1/b) * (arctan(z/b) + π/2)
        los2 = (np.arctan(u0) + (np.pi / 2.0)) / b_safe

        # Build / reuse a cached table for F(u)=∫_0^u (1+t^2)^(-5/4) dt for u∈[0,U_MAX]
        if _ACCEL_U_GRID is None or _ACCEL_F_GRID is None:
            # Dense near 0, log-spaced tail for good accuracy across many decades
            u_lin = np.linspace(0.0, 50.0, 20000)
            u_log = np.logspace(np.log10(50.0), np.log10(_ACCEL_U_MAX), 20000)
            u = np.unique(np.concatenate([u_lin, u_log]))
            f = (1.0 + u * u) ** (-5.0 / 4.0)
            du = np.diff(u)
            # cumulative trapezoid integral
            F = np.empty_like(u)
            F[0] = 0.0
            F[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * du)
            _ACCEL_U_GRID = u
            _ACCEL_F_GRID = F

        # Analytic value of F(∞) = ∫_0^∞ (1+u^2)^(-5/4) du
        # = (sqrt(pi) * Gamma(3/4)) / (2 * Gamma(5/4))
        F_inf = (math.sqrt(math.pi) * math.gamma(0.75)) / (2.0 * math.gamma(1.25))

        # F_abs(|u0|) = ∫_0^{|u0|} (1+u^2)^(-5/4) du
        F_abs = np.interp(
            np.minimum(u_abs, _ACCEL_U_MAX),
            _ACCEL_U_GRID,
            _ACCEL_F_GRID,
        )
        # For very large u, use asymptotic tail: F_inf - F(u) ≈ (2/3) u^(-3/2)
        large = u_abs > _ACCEL_U_MAX
        if np.any(large):
            F_abs = F_abs.copy()
            F_abs[large] = F_inf - ((2.0 / 3.0) / (u_abs[large] ** (3.0 / 2.0)))

        # Integral from -∞ to u0:
        #  u0>=0: F_inf + F_abs(u0)
        #  u0< 0: F_inf - F_abs(|u0|)
        I_u = F_inf + (np.sign(u0) * F_abs)

        # Accelerating-wind LOS integral
        los = I_u / (b_safe ** (3.0 / 2.0))

        lw = los * A
        lw2 = los2 * A

        return lw.astype(float), lw2.astype(float), los.astype(float), A.astype(float)

    # Fixed physical cutoff at radius Rmax
    # NOTE: if Rmax is None we return early via adaptive integration above.
    Rmax_use = float(Rmax)

    # Per-cell LOS limit in z such that r = sqrt(b^2 + z^2) <= Rmax_use
    # (integrating from z_start toward decreasing z until r hits Rmax_use)
    t = np.sqrt(np.maximum((Rmax_use ** 2) - (av_db ** 2), 0.0))

    # Determine per-cell validity: original algorithm integrates only if |z_start| <= t
    valid_cells = (t > 0) & (np.abs(z_start) <= t)

    # Per-cell end step index; invalid cells get -1 so they contribute zero
    end_k = np.floor((z_start + t) / dz).astype(int)
    end_k = np.where(valid_cells, end_k, -1)

    max_steps = int(end_k.max())
    if max_steps < 0:
        return (
            np.zeros_like(av_db),
            np.zeros_like(av_db),
            np.zeros_like(av_db),
            np.zeros_like(av_db),
        )

    # Broadcasted z values across steps
    k = np.arange(max_steps + 1, dtype=float)  # shape (K+1,)
    z_vals = z_start - dz * k  # shape (K+1,)

    # Broadcast to (N, K+1)
    cl2 = (av_db**2)[:, None]  # (N,1)
    z2_vals = z_vals[None, :] ** 2  # (1, K+1)

    denom = cl2 + z2_vals  # (N, K+1)

    # Masks to include only valid steps per cell (k from 0..end_k inclusive)
    step_mask = (k[None, :] <= end_k[:, None]) & valid_cells[:, None]

    # Compute sums
    con_sum = np.sum((denom ** (-5.0 / 4.0)) * step_mask, axis=1)
    con2_sum = np.sum((denom ** (-1.0)) * step_mask, axis=1)

    los = dz * con_sum
    los2 = dz * con2_sum

    lw = los * A
    lw2 = los2 * A

    # icd is los for each cell, A2 is A
    return lw.astype(float), lw2.astype(float), los.astype(float), A.astype(float)


def get_available_bands_from_csv(df: pd.DataFrame) -> List[str]:
    """
    Detect available energy bands from CSV column names.
    
    Looks for columns matching pattern: flux_{band}_ph
    
    Args:
        df: DataFrame from flux vs nH CSV
        
    Returns:
        List of band names (e.g., ['broad', 'soft', 'medium', 'hard'])
    """
    bands = []
    for col in df.columns:
        if col.startswith("flux_") and col.endswith("_ph"):
            # Extract band name from flux_{band}_ph
            band = col[5:-3]  # Remove "flux_" prefix and "_ph" suffix
            bands.append(band)
    return sorted(bands)


def load_flux_vs_nh_csv(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load flux vs nH CSV file generated by compute_flux_vs_nH.py.
    Automatically detects available energy bands from column names.
    
    Args:
        csv_path: Path to CSV file with columns like nH_1e22, flux_{band}_ph, flux_{band}_erg
        
    Returns:
        Tuple of (DataFrame with flux vs nH data, list of available band names)
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV is missing required columns or has no valid bands
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Flux vs nH CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check for nH column
    if "nH_1e22" not in df.columns:
        raise ValueError("CSV missing required column: nH_1e22")
    
    # Detect available bands
    bands = get_available_bands_from_csv(df)
    if not bands:
        raise ValueError("No flux columns found in CSV. Expected columns like flux_{band}_ph")
    
    print(f"Detected energy bands in CSV: {', '.join(bands)}")
    
    # Filter out rows with invalid nH
    df = df[df["nH_1e22"].notna() & (df["nH_1e22"] > 0)]
    
    # Filter out rows where all flux columns are NaN or negative
    valid_mask = df["nH_1e22"].notna()
    for band in bands:
        flux_col = f"flux_{band}_ph"
        if flux_col in df.columns:
            # Keep row if at least one band has valid data
            valid_mask = valid_mask & df[flux_col].notna()
    
    df = df[valid_mask]
    
    if len(df) == 0:
        raise ValueError("No valid data points in CSV after filtering")
    
    return df, bands


def interpolate_flux_from_nh(
    nh_1e22: np.ndarray, df: pd.DataFrame, band: str
) -> np.ndarray:
    """
    Interpolate flux values for given nH array using CSV data.
    
    Args:
        nh_1e22: Array of nH values in 1e22 cm^-2 units
        df: DataFrame from load_flux_vs_nh_csv
        band: Band name (e.g., "soft", "hard", "broad", "medium")
        
    Returns:
        Array of interpolated flux values (photons/cm^2/s)
        
    Raises:
        ValueError: If band column is not found in DataFrame
    """
    flux_col = f"flux_{band}_ph"
    
    if flux_col not in df.columns:
        available = get_available_bands_from_csv(df)
        raise ValueError(
            f"Band '{band}' not found in CSV. Available bands: {available}"
        )
    
    # Sort by nH for interpolation
    df_sorted = df.sort_values("nH_1e22")
    nh_csv = df_sorted["nH_1e22"].values
    flux_csv = df_sorted[flux_col].values
    
    # Filter out NaN/invalid flux values
    valid = np.isfinite(flux_csv) & (flux_csv > 0)
    if not np.any(valid):
        raise ValueError(f"No valid flux data for band '{band}'")
    
    nh_csv = nh_csv[valid]
    flux_csv = flux_csv[valid]
    
    # Check range coverage
    nh_min, nh_max = nh_csv.min(), nh_csv.max()
    if np.any(nh_1e22 < nh_min) or np.any(nh_1e22 > nh_max):
        warnings.warn(
            f"Some nH values are outside CSV range [{nh_min:.3f}, {nh_max:.3f}] 1e22 cm^-2 for band '{band}'. "
            f"Extrapolation will be used (fill_value='extrapolate')."
        )
    
    # Create interpolator (log-log space for better behavior)
    interp_func = interp1d(
        np.log10(nh_csv),
        np.log10(flux_csv),
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    
    # Interpolate (handle edge cases)
    nh_1e22 = np.asarray(nh_1e22)
    nh_1e22_safe = np.clip(nh_1e22, 1e-6, 1e6)  # Avoid log10(0)
    log_flux = interp_func(np.log10(nh_1e22_safe))
    flux = 10 ** log_flux
    
    return flux


def fit_exponential_to_csv(df: pd.DataFrame, band: str) -> Tuple[float, float]:
    """
    Fit exponential function A * exp(-B * nH) to CSV flux data in LOG SPACE.
    
    Fitting in log space: log(flux) = log(A) - B * nH
    This gives equal weight to all data points regardless of magnitude,
    appropriate for data spanning many orders of magnitude.
    
    Args:
        df: DataFrame from load_flux_vs_nh_csv
        band: Band name (e.g., "soft", "hard", "broad", "medium")
        
    Returns:
        Tuple of (A, B) coefficients for flux = A * exp(-B * nH_1e22)
        where flux is in photons/cm²/s (raw units from CSV)
        
    Raises:
        ValueError: If band column is not found or fit fails without fallback
    """
    flux_col = f"flux_{band}_ph"
    
    if flux_col not in df.columns:
        available = get_available_bands_from_csv(df)
        raise ValueError(
            f"Band '{band}' not found in CSV. Available bands: {available}"
        )
    
    # Get data
    df_sorted = df.sort_values("nH_1e22")
    nh = df_sorted["nH_1e22"].values
    flux = df_sorted[flux_col].values
    
    # Filter out NaN/invalid values
    valid = np.isfinite(nh) & np.isfinite(flux) & (flux > 0) & (nh > 0)
    if not np.any(valid):
        raise ValueError(f"No valid flux data for band '{band}'")
    
    nh = nh[valid]
    flux = flux[valid]
    
    # Take logarithm for fitting in log space
    log_flux = np.log(flux)
    
    # Fit linear function in log space: log(flux) = log(A) - B * nH
    def linear_func(x, log_A, B):
        return log_A - B * x
    
    try:
        # Initial guess for log(A) and B from endpoints
        log_A_guess = np.log(flux[0]) + 0.1 * nh[0]
        B_guess = -(log_flux[-1] - log_flux[0]) / (nh[-1] - nh[0])
        
        popt, _ = curve_fit(
            linear_func,
            nh,
            log_flux,
            p0=[log_A_guess, max(B_guess, 0.01)],
            maxfev=10000,
        )
        log_A, B = popt
        A = np.exp(log_A)  # Convert back from log space
        
        print(f"Fitted exponential for {band} band: A={A:.6e}, B={B:.6f}")
        return float(A), float(B)
        
    except Exception as e:
        # Try fallback for legacy bands (with proper scaling)
        if band == "hard":
            warnings.warn(f"Exponential fit failed for {band} band: {e}. Using legacy values.")
            return 9.524e-13, 0.057  # Legacy hard band coefficients (converted to raw units)
        elif band == "soft":
            warnings.warn(f"Exponential fit failed for {band} band: {e}. Using legacy values.")
            return 9.3923e-13, 2.5062  # Legacy soft band coefficients (converted to raw units)
        else:
            raise ValueError(f"Exponential fit failed for {band} band: {e}") from e


def simulate_lightcurve(
    r: float = 0.001,
    R: float = 2.0,
    d1: float = 11.0,
    d2: float = 8.0,
    gma0: float = -90.0,
    i0: float = 26.0,
    dth: float = 1.0,
    d2h: float = 6.0,
    dz: float = 0.1,
    verbose: bool = False,
    n_jobs: int = 1,
    flux_method: str = "legacy",
    flux_csv_path: Optional[str] = None,
    lam: float = 0.589537,
    lam2: float = 0.589537,
    Rmax: Optional[float] = None,
    converge_rmax: bool = False,
) -> pd.DataFrame:
    """
    Main simulation function for lightcurve calculation.

    Args:
        r: Radius of smaller star B (compact object) in solar radii
        R: Radius of larger star A (companion) in solar radii
        d1: Distance of star B from COM in solar radii
        d2: Distance of star A from COM in solar radii
        gma0: Starting phase angle in degrees
        i0: Orbital inclination in degrees
        dth: Orbital increment in degrees
        d2h: Angular cell size (degrees) for the polar grid used in the surface integral
        dz: Step size along the line of sight (solar radii)
        verbose: If True, prints per-phase progress
        n_jobs: Number of parallel workers across phases (1 = serial)
        flux_method: Method for converting nH to flux. Options:
            - "legacy": Use hardcoded exponential coefficients (default)
            - "interpolate": Interpolate from CSV flux vs nH data
            - "refit": Fit new exponentials to CSV data
        flux_csv_path: Path to CSV file from compute_flux_vs_nH.py (required if flux_method != "legacy")
        lam: Scaling parameter to convert flx to nH (in 1e22 cm^-2 units).
            Default 0.589537 means mean(fl) = 0.589537 (i.e., mean nH ≈ 5.9e21 cm^-2)
        lam2: Scaling parameter for constant velocity wind model (flx2)
        Rmax: Maximum radius (solar radii) used as a hard cutoff for LOS integration.
            If None, the LOS integration uses adaptive convergence stopping (see converge_rmax).
            Note: the CLI sets the default to 2*(d1+d2), reproducing the legacy cutoff.
        converge_rmax: If True, ignore fixed Rmax and integrate adaptively until tail contributions are negligible.

    Returns:
        DataFrame with simulation results. Key columns:
            - fl, fl2: Scaled nH values (in 1e22 cm^-2 units)
            - nfl_hard_av, nfl_soft_av: Photon fluxes (photons/cm^2/s) for hard/soft bands
            - pho_count_hard_av, pho_count_soft_av: Photon counts (legacy mode only)
            
    Notes:
        - The column density integral (flx) has units of (atoms/solar_radius^4)
        - Scaling by lam converts this to nH in units of 1e22 cm^-2
        - Example: fl=1.0 means nH = 1.0e22 cm^-2
        - Photon count columns are only included when flux_method="legacy"
    """
    # Convert angles to radians
    gma = gma0 * np.pi / 180
    i = i0 * np.pi / 180
    d = d1 + d2

    # Integration cutoff handling
    # - If converge_rmax is enabled OR Rmax is None: use adaptive stopping
    # - Otherwise: use fixed cutoff at Rmax
    converge_rmax_use = bool(converge_rmax) or (Rmax is None)
    Rmax_use: Optional[float] = None if Rmax is None else float(Rmax)

    # Prepare phase values
    n_iterations = int(360 / dth)
    gma_values = gma + (np.arange(n_iterations) * (dth * np.pi / 180.0))

    # Worker to compute one phase
    def _compute_one_phase(cur_gma: float):
        h1 = d1 * np.sin(cur_gma) * np.sin(i)
        h2 = d2 * np.sin(cur_gma) * np.sin(i)
        L1 = d1 * np.cos(cur_gma)
        L2 = d2 * np.cos(cur_gma)
        l1 = np.sqrt(h1**2 + L1**2)
        l2 = np.sqrt(h2**2 + L2**2)
        h = h1 + h2
        L = L1 + L2
        l = l1 + l2

        a = l**2 + r**2 - R**2
        b = 2 * abs(l) * r
        n = l / (R + r)

        # Track if emitter is fully eclipsed (blocked by companion)
        is_eclipsed = False

        # Only check for eclipse when emitter is BEHIND companion (sin(gma) > 0)
        # When emitter is in front (sin(gma) <= 0), no occultation possible
        if np.sin(cur_gma) > 0:
            if n >= 1:
                # Emitter is behind but not overlapping with companion disk
                av_x, av_th, av_db, A_cells = create_grid(r, l, R, cur_gma, d2h=d2h)
                lw, lw2, icd_val, A2_val = wind_los_integral(
                    d,
                    d1,
                    d2,
                    cur_gma,
                    i,
                    av_x,
                    av_th,
                    av_db,
                    A_cells,
                    dz=dz,
                    Rmax=Rmax_use,
                    converge_rmax=converge_rmax_use,
                )
                if lw.size > 0:
                    flx_i = float(np.sum(lw) / np.sum(A_cells))
                    flx2_i = float(np.sum(lw2) / np.sum(A_cells))
                    icd_i = float(np.sum(lw))
                    A2_i = float(np.sum(A_cells))
                else:
                    flx_i = 0.0
                    flx2_i = 0.0
                    icd_i = 0.0
                    A2_i = 0.0
            else:
                n2 = a / b
                n3 = l / (R - r)
                if abs(n3) <= 1:
                    # Total eclipse: emitter is completely behind companion
                    is_eclipsed = True
                    flx_i = 0.0
                    flx2_i = 0.0
                    icd_i = 0.0
                    A2_i = 0.0
                else:
                    # Partial overlap - compute visible portion
                    av_x, av_th, av_db, A_cells = create_grid(r, l, R, cur_gma, d2h=d2h)
                    lw, lw2, icd_val, A2_val = wind_los_integral(
                        d,
                        d1,
                        d2,
                        cur_gma,
                        i,
                        av_x,
                        av_th,
                        av_db,
                        A_cells,
                        dz=dz,
                        Rmax=Rmax_use,
                        converge_rmax=converge_rmax_use,
                    )
                    if lw.size > 0:
                        flx_i = float(np.sum(lw) / np.sum(A_cells))
                        flx2_i = float(np.sum(lw2) / np.sum(A_cells))
                        icd_i = float(np.sum(lw))
                        A2_i = float(np.sum(A_cells))
                    else:
                        flx_i = 0.0
                        flx2_i = 0.0
                        icd_i = 0.0
                        A2_i = 0.0
        else:
            # Emitter is in front of companion - fully visible, no eclipse possible
            av_x, av_th, av_db, A_cells = create_grid(r, l, R, cur_gma, d2h=d2h)
            lw, lw2, icd_val, A2_val = wind_los_integral(
                d,
                d1,
                d2,
                cur_gma,
                i,
                av_x,
                av_th,
                av_db,
                A_cells,
                dz=dz,
                Rmax=Rmax_use,
                converge_rmax=converge_rmax_use,
            )
            if lw.size > 0:
                flx_i = float(np.sum(lw) / np.sum(A_cells))
                flx2_i = float(np.sum(lw2) / np.sum(A_cells))
                icd_i = float(np.sum(lw))
                A2_i = float(np.sum(A_cells))
            else:
                flx_i = 0.0
                flx2_i = 0.0
                icd_i = 0.0
                A2_i = 0.0

        deg_i = cur_gma * 180.0 / np.pi
        time_i = deg_i * 348.42
        phase_i = (cur_gma - (gma0 * np.pi / 180.0)) / (2.0 * np.pi)
        return (
            flx_i,
            flx2_i,
            icd_i,
            A2_i,
            cur_gma,
            deg_i,
            phase_i,
            time_i,
            l,
            L,
            h,
            is_eclipsed,
        )

    # Compute phases, optionally in parallel
    if n_jobs == 1:
        results_list = []
        for idx, cur_gma in enumerate(gma_values):
            out = _compute_one_phase(cur_gma)
            results_list.append(out)
            if verbose:
                print(f"Phase: {out[5]:.2f} degrees")
    else:
        try:
            from joblib import Parallel, delayed

            results_list = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_compute_one_phase)(float(cur_gma)) for cur_gma in gma_values
            )
            if verbose:
                for out in results_list:
                    print(f"Phase: {out[5]:.2f} degrees")
        except Exception:
            # Fallback to serial if joblib missing or errors
            results_list = []
            for idx, cur_gma in enumerate(gma_values):
                out = _compute_one_phase(cur_gma)
                results_list.append(out)
                if verbose:
                    print(f"Phase: {out[5]:.2f} degrees")

    # Unpack (now includes is_eclipsed flag)
    flx, flx2, icd_vals, A2_vals, ph, deg, phase, time, l3, L3, h3, is_eclipsed = map(
        list, zip(*results_list)
    )

    # Create results DataFrame
    results = pd.DataFrame(
        {
            "deg": deg,
            "ph": ph,
            "phase": phase,
            "A2": A2_vals,
            "flx": flx,
            "flx2": flx2,
            "icd": icd_vals,
            "time": time,
            "l3": l3,
            "L3": L3,
            "h3": h3,
            "is_eclipsed": is_eclipsed,
        }
    )

    # Calculate additional flux parameters
    # Compute scaling factors (use auto-scaling if lam not explicitly set)
    lam_computed = 0.589537 / np.mean(flx) if np.mean(flx) > 0 else 1
    lam2_computed = 0.589537 / np.mean(flx2) if np.mean(flx2) > 0 else 1
    
    # Use provided lam if different from default, otherwise use computed
    lam_use = lam if lam != 0.589537 else lam_computed
    lam2_use = lam2 if lam2 != 0.589537 else lam2_computed

    fl = np.array(flx) * lam_use
    fl2 = np.array(flx2) * lam2_use

    # Calculate scaled fluxes based on method
    if flux_method == "legacy":
        # Legacy hardcoded exponential coefficients
        nfl_hard_av = 9.524 * np.exp(-fl * 0.057)
        nfl_hard_cv = 9.524 * np.exp(-fl2 * 0.057)
        nfl_soft_av = 9.3923 * np.exp(-fl * 2.5062)
        nfl_soft_cv = 9.3923 * np.exp(-fl2 * 2.5062)
        pho_count_hard_av = 0.0001464 * np.exp(-fl * 0.1066818)
        pho_count_soft_av = 0.0005275 * np.exp(-fl * 2.7556631)
        
        # Add flux columns to results (legacy mode)
        results["fl"] = fl
        results["fl2"] = fl2
        results["nfl_hard_av"] = nfl_hard_av
        results["nfl_hard_cv"] = nfl_hard_cv
        results["nfl_soft_av"] = nfl_soft_av
        results["nfl_soft_cv"] = nfl_soft_cv
        results["pho_count_hard_av"] = pho_count_hard_av
        results["pho_count_soft_av"] = pho_count_soft_av
        
    elif flux_method == "interpolate":
        # Interpolate from CSV data
        if flux_csv_path is None:
            raise ValueError("flux_csv_path required when flux_method='interpolate'")
        
        df_flux, available_bands = load_flux_vs_nh_csv(flux_csv_path)
        
        # Add base columns
        results["fl"] = fl
        results["fl2"] = fl2
        
        # Dynamically compute flux for all available bands
        for band in available_bands:
            try:
                results[f"nfl_{band}_av"] = interpolate_flux_from_nh(fl, df_flux, band)
                results[f"nfl_{band}_cv"] = interpolate_flux_from_nh(fl2, df_flux, band)
            except Exception as e:
                warnings.warn(f"Failed to interpolate flux for band '{band}': {e}")
        
    elif flux_method == "refit":
        # Fit new exponentials to CSV data
        if flux_csv_path is None:
            raise ValueError("flux_csv_path required when flux_method='refit'")
        
        df_flux, available_bands = load_flux_vs_nh_csv(flux_csv_path)
        
        # Add base columns
        results["fl"] = fl
        results["fl2"] = fl2
        
        # Dynamically fit and compute flux for all available bands
        for band in available_bands:
            try:
                A, B = fit_exponential_to_csv(df_flux, band)
                results[f"nfl_{band}_av"] = A * np.exp(-B * fl)
                results[f"nfl_{band}_cv"] = A * np.exp(-B * fl2)
            except Exception as e:
                warnings.warn(f"Failed to fit exponential for band '{band}': {e}")
        
    else:
        raise ValueError(f"Invalid flux_method: {flux_method}. Must be 'legacy', 'interpolate', or 'refit'")

    # Set all scaled flux columns to 0 when eclipsed
    # During eclipse, the emitter is physically blocked - flux should be zero,
    # not computed from absorption formula (which would give max flux when nH=0)
    eclipse_mask = results["is_eclipsed"].values
    if np.any(eclipse_mask):
        # Find all flux columns (nfl_* and pho_count_*)
        flux_cols = [col for col in results.columns 
                     if col.startswith("nfl_") or col.startswith("pho_count_")]
        for col in flux_cols:
            results.loc[eclipse_mask, col] = 0.0

    return results


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Simulation of column densities for eclipsing binary systems",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--r",
        type=float,
        default=0.001,
        help="Radius of smaller star B (compact object) in solar radii",
    )
    parser.add_argument(
        "--R",
        type=float,
        default=2.0,
        help="Radius of larger star A (companion) in solar radii",
    )
    parser.add_argument(
        "--d1",
        type=float,
        default=11.0,
        help="Distance of star B from COM in solar radii",
    )
    parser.add_argument(
        "--d2",
        type=float,
        default=8.0,
        help="Distance of star A from COM in solar radii",
    )
    parser.add_argument(
        "--gma0", type=float, default=-90.0, help="Starting phase angle in degrees"
    )
    parser.add_argument(
        "--i0", type=float, default=26.0, help="Orbital inclination in degrees"
    )
    parser.add_argument(
        "--dth", type=float, default=1.0, help="Orbital increment in degrees"
    )
    parser.add_argument(
        "--d2h",
        type=float,
        default=6.0,
        help="Angular cell size (degrees) for the polar grid used in the surface integral",
    )
    parser.add_argument(
        "--dz",
        type=float,
        default=0.1,
        help="Step size along the line of sight (solar radii)",
    )
    parser.add_argument(
        "--Rmax",
        type=float,
        default=None,
        help="Maximum radius (solar radii) for LOS integration cutoff. "
        "If not provided, defaults to 2*(d1+d2). Ignored if --converge-rmax is set.",
    )
    parser.add_argument(
        "--converge-rmax",
        action="store_true",
        help="Override fixed Rmax cutoff and integrate adaptively until LOS tail contributions "
        "become negligible (both wind models).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-phase progress during simulation",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel workers across phases (1 = serial)",
    )
    parser.add_argument(
        "--flux_method",
        type=str,
        choices=["legacy", "interpolate", "refit"],
        default="legacy",
        help="Method for converting nH to flux: 'legacy' (hardcoded exponentials), "
        "'interpolate' (from CSV), or 'refit' (fit new exponentials to CSV)",
    )
    parser.add_argument(
        "--flux_csv",
        type=str,
        default=None,
        help="Path to flux vs nH CSV file from compute_flux_vs_nH.py "
        "(required if flux_method is not 'legacy')",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.589537,
        help="Scaling parameter to convert flx to nH (in 1e22 cm^-2 units). "
        "Default: 0.589537 (legacy). Use XSPEC fitted value (0.572385) for better accuracy. "
        "Get XSPEC value with: python get_xspec_nH.py",
    )
    parser.add_argument(
        "--lam2",
        type=float,
        default=0.589537,
        help="Scaling parameter for constant velocity wind model (flx2). "
        "Default: 0.589537 (legacy). Typically set to same value as --lam",
    )
    parser.add_argument(
        "--use_xspec_nH",
        action="store_true",
        help="Automatically use nH from XSPEC fit (reads from xspec_nH.txt). "
        "Overrides --lam if specified. Run 'python get_xspec_nH.py' first.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="xrb_lightcurve_output.csv",
        help="Output file name for results",
    )

    args = parser.parse_args()

    # Default physical cutoff reproduces legacy behavior
    if args.Rmax is None:
        args.Rmax = 2.0 * (args.d1 + args.d2)
    
    # Check if XSPEC nH should be used
    if args.use_xspec_nH:
        try:
            with open('xspec_nH.txt', 'r') as f:
                xspec_nH = float(f.read().strip())
            print(f"Using XSPEC fitted nH: {xspec_nH} × 10²² cm⁻²")
            args.lam = xspec_nH
            args.lam2 = xspec_nH
        except FileNotFoundError:
            print("Error: xspec_nH.txt not found!")
            print("Run 'python get_xspec_nH.py' first to extract nH from XSPEC fit.")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading xspec_nH.txt: {e}")
            sys.exit(1)
    
    # Validate arguments
    if args.flux_method in ["interpolate", "refit"] and args.flux_csv is None:
        parser.error(f"--flux_csv is required when flux_method='{args.flux_method}'")

    print("Starting XRB Lightcurve Simulation...")
    print(f"Parameters:")
    print(f"  r (emitter radius): {args.r} solar radii")
    print(f"  R (companion radius): {args.R} solar radii")
    print(f"  d1 (emitter separation): {args.d1} solar radii")
    print(f"  d2 (companion separation): {args.d2} solar radii")
    print(f"  gma0 (starting phase): {args.gma0} degrees")
    print(f"  i0 (inclination): {args.i0} degrees")
    print(f"  dth (orbital increment): {args.dth} degrees")
    print(f"  d2h (polar cell size): {args.d2h} degrees")
    print(f"  dz (LOS step size): {args.dz}")
    print(f"  Rmax (LOS cutoff): {args.Rmax}")
    print(f"  converge_rmax: {args.converge_rmax}")
    print(f"  n_jobs (parallel workers): {args.n_jobs}")
    print(f"  flux_method: {args.flux_method}")
    if args.flux_csv:
        print(f"  flux_csv: {args.flux_csv}")
    print(f"  lam: {args.lam}")
    print(f"  lam2: {args.lam2}")
    print(f"  Output file: {args.output}")
    print()

    # Run simulation
    results = simulate_lightcurve(
        r=args.r,
        R=args.R,
        d1=args.d1,
        d2=args.d2,
        gma0=args.gma0,
        i0=args.i0,
        dth=args.dth,
        d2h=args.d2h,
        dz=args.dz,
        verbose=args.verbose,
        n_jobs=args.n_jobs,
        flux_method=args.flux_method,
        flux_csv_path=args.flux_csv,
        lam=args.lam,
        lam2=args.lam2,
        Rmax=args.Rmax,
        converge_rmax=args.converge_rmax,
    )

    # Save results
    results.to_csv(args.output, index=False)
    print(f"\nSimulation completed! Results saved to {args.output}")
    print(f"Total data points: {len(results)}")
    print(
        f"Phase range: {results['deg'].min():.2f} to {results['deg'].max():.2f} degrees"
    )

    return results


if __name__ == "__main__":
    main()
