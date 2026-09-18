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
import warnings
from typing import Tuple, List, Optional, Dict, Callable
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

try:
    from numba import njit, prange
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "numba is required: the Gauss-Legendre mega-kernel is the only LOS "
        "integrator, and it is also the only path that returns the per-cell "
        "columns needed for the nonlinear N_H -> flux conversion. "
        "Install with: pip install numba"
    ) from exc

# =============================================================================
# Wind density profiles
# =============================================================================
#
# Each wind model defines a dimensionless density profile g(r) with r in solar
# radii. The absolute scale of g is arbitrary; the simulation fixes it from
# Mdot / v_inf (see wind_density_norm_from_mdot), so the column density carries
# real units and the light curve is sensitive to the absolute size of the
# system rather than only to ratios such as R/a.
#
# Supported models (Wind_Density.pdf):
#   0 smooth_pl   — smoothly broken power law; params (Rb, p, Delta)
#   1 confinement — inner confinement / compression; params (R_star, fconf, ell)
#   2 beta_law    — velocity-based, n = Mdot / (4 pi r^2 v(r)) with a CAK
#                   beta law plus inner acceleration scale H; params
#                   (R_star, beta, H). g = 1 / (r^2 v_hat), v_hat = v / v_inf.

# Physical constants. Defined here rather than beside the column-density
# helpers below because they are used as default argument values.
R_SUN_CM = 6.957e10  # 1 solar radius in cm
M_H_G = 1.6726e-24  # hydrogen atom mass in g
M_SUN_G = 1.989e33  # solar mass in g
KM_TO_CM = 1.0e5  # 1 km in cm
YEAR_S = 3.1558e7  # 1 Julian year in s

# Mean mass per hydrogen-equivalent nucleus. TBabs columns are quoted as
# equivalent hydrogen columns at solar abundance, so this converts a wind mass
# column into the N_H that the flux_vs_nH table expects.
MU_WIND_DEFAULT = 1.4

WIND_MODEL_IDS: Dict[str, int] = {
    "smooth_pl": 0,
    "confinement": 1,
    "beta_law": 2,
}

WIND_MODEL_PARAM_KEYS: Dict[str, Tuple[str, ...]] = {
    "smooth_pl": ("Rb", "p", "Delta"),
    "confinement": ("R_star", "fconf", "ell"),
    "beta_law": ("R_star", "beta", "H"),
}

# Profiles whose R_star is the companion photosphere. Callers may omit R_star
# for these and it is filled from the geometry parameter R.
R_STAR_TIED_MODELS: Tuple[str, ...] = ("beta_law", "confinement")

# Cache flux-vs-NH interpolation/refit inputs keyed by (csv_path, flux_type)
# to avoid repeated CSV read/sort/interpolator creation during MCMC.
_FLUX_CACHE: Dict[Tuple[str, str], Dict[str, object]] = {}


def pack_wind_params(
    wind_model: str,
    wind_params: Dict[str, float],
) -> Tuple[int, float, float, float, float]:
    """
    Convert a (wind_model, params dict) pair into a flat tuple
    (model_id, p1, p2, p3, p4) suitable for passing to the numba kernel.

    Keys expected per model are listed in WIND_MODEL_PARAM_KEYS. Missing keys
    raise ValueError. Unused slots are filled with 0.
    """
    if wind_model not in WIND_MODEL_IDS:
        raise ValueError(
            f"Unknown wind_model '{wind_model}'. "
            f"Choose one of: {list(WIND_MODEL_IDS.keys())}"
        )
    model_id = WIND_MODEL_IDS[wind_model]
    keys = WIND_MODEL_PARAM_KEYS[wind_model]
    missing = [k for k in keys if k not in wind_params]
    if missing:
        raise ValueError(
            f"wind_model '{wind_model}' requires parameters {keys}; "
            f"missing: {missing}"
        )
    p = [float(wind_params[k]) for k in keys]
    while len(p) < 4:
        p.append(0.0)
    return model_id, p[0], p[1], p[2], p[3]


@njit(cache=True, inline="always")
def _g_profile(r, model_id, p1, p2, p3, p4):
    """
    Dimensionless density profile g(r) with r in solar radii.

    model_id encodes the profile; p1..p4 are model-specific parameters
    (see pack_wind_params and WIND_MODEL_PARAM_KEYS).
    """
    if r <= 0.0:
        return 0.0

    if model_id == 0:
        # smooth_pl: Rb=p1, p=p2, Delta=p3
        Rb = p1
        p_slope = p2
        Delta = p3
        x = r / Rb
        if Delta <= 0.0:
            return x ** (-2.0)
        base = x ** (-2.0)
        bracket = 1.0 + (1.0 / x) ** Delta
        exponent = (p_slope - 2.0) / Delta
        return base * (bracket ** exponent)

    if model_id == 1:
        # confinement: R_star=p1, fconf=p2, ell=p3
        R_star = p1
        fconf = p2
        ell = p3
        factor = 1.0 + fconf * math.exp(-(r - R_star) / ell)
        return factor / (r * r)

    if model_id == 2:
        # beta_law: R_star=p1, beta=p2, H=p3
        # Mass continuity n = Mdot / (4 pi r^2 v) with the dimensionless
        # velocity v_hat = (1 - exp(-(r-R*)/H)) * (1 - R*/r)^beta -> 1 at
        # infinity, so g -> 1/r^2 and C = 1 in wind_asymptotic_coefficient.
        # Inside the photosphere there is no wind; v_hat -> 0 at the surface,
        # so g diverges there and rays grazing the limb are effectively opaque.
        R_star = p1
        beta = p2
        H = p3
        if r <= R_star:
            return 0.0
        v_hat = (1.0 - math.exp(-(r - R_star) / H)) * (1.0 - R_star / r) ** beta
        if v_hat <= 0.0:
            return 0.0
        return 1.0 / (r * r * v_hat)

    return 0.0


def evaluate_g_profile(
    r,
    wind_model: str,
    wind_params: Dict[str, float],
):
    """
    Pure-Python (vectorized) wind density profile for use in helpers.

    Returns the dimensionless g(r) matching `_g_profile` for arrays or scalars.
    """
    model_id, p1, p2, p3, p4 = pack_wind_params(wind_model, wind_params)
    r = np.asarray(r, dtype=float)

    if model_id == 0:
        Rb = p1
        p_slope = p2
        Delta = p3
        x = np.where(r > 0.0, r / Rb, np.inf)
        if Delta <= 0.0:
            return x ** (-2.0)
        base = x ** (-2.0)
        bracket = 1.0 + (1.0 / x) ** Delta
        exponent = (p_slope - 2.0) / Delta
        return base * (bracket ** exponent)

    if model_id == 1:
        R_star = p1
        fconf = p2
        ell = p3
        safe_r = np.where(r > 0.0, r, np.inf)
        factor = 1.0 + fconf * np.exp(-(safe_r - R_star) / ell)
        return factor / (safe_r * safe_r)

    if model_id == 2:
        R_star = p1
        beta = p2
        H = p3
        # r <= R_star maps to inf so that v_hat -> 1 and g -> 0 there, matching
        # the scalar kernel without evaluating a negative base.
        rr = np.where(r > R_star, r, np.inf)
        v_hat = (1.0 - np.exp(-(rr - R_star) / H)) * (1.0 - R_star / rr) ** beta
        return np.where(v_hat > 0.0, 1.0 / (rr * rr * v_hat), 0.0)

    return np.zeros_like(r)


# =============================================================================
# Numba-accelerated LOS integration kernel
# =============================================================================
#
# `_simulate_phases_numba` is a mega-kernel that, for each phase, builds the
# polar emitter grid inline and integrates every cell's LOS using fixed-node
# Gauss-Legendre quadrature with the substitution u = arctan(z/b). This
# collapses the slowly-decaying r^{-2} tail to a bounded smooth integrand on a
# finite interval, so 16 GL nodes per cell give >10 digits of accuracy for any
# wind profile and any impact parameter (no special-casing of b vs Rb), and the
# full z-tail is always integrated — there is no cutoff radius to choose. The
# whole 360-phase loop runs under one numba @njit(parallel=True) call with
# prange over phases, eliminating per-phase Python overhead and per-call thread
# launches.

# Pre-computed 16-point Gauss-Legendre nodes/weights on [-1, 1].
# Generated once via numpy.polynomial.legendre.leggauss(16).
_GL16_X = np.array([
    -0.9894009349916499, -0.9445750230732326, -0.8656312023878318,
    -0.7554044083550030, -0.6178762444026438, -0.4580167776572274,
    -0.2816035507792589, -0.0950125098376374,  0.0950125098376374,
     0.2816035507792589,  0.4580167776572274,  0.6178762444026438,
     0.7554044083550030,  0.8656312023878318,  0.9445750230732326,
     0.9894009349916499,
], dtype=np.float64)
_GL16_W = np.array([
    0.0271524594117540, 0.0622535239386477, 0.0951585116824928,
    0.1246289712555340, 0.1495959888165768, 0.1691565193950026,
    0.1826034150449236, 0.1894506104550686, 0.1894506104550686,
    0.1826034150449236, 0.1691565193950026, 0.1495959888165768,
    0.1246289712555340, 0.0951585116824928, 0.0622535239386477,
    0.0271524594117540,
], dtype=np.float64)


@njit(cache=True, inline="always")
def _los_gl_quadrature(b, z_start, model_id, p1, p2, p3, p4, gl_x, gl_w):
    """
    LOS integral ∫_{-∞}^{z_start} g(r=sqrt(b²+z²)) dz via Gauss-Legendre
    quadrature in u = arctan(z/b).

    The substitution gives:
        ∫ g(r) dz = b · ∫_{-π/2}^{u_start} g(b/cos u) · sec²(u) du
    The integrand is bounded and smooth on the finite interval [-π/2, u_start]
    for any profile that falls at least as fast as r^{-1} at infinity.
    """
    if b < 1e-8:
        b = 1e-8
    u_start = math.atan(z_start / b)
    u_lo = -1.5707963267948966  # -pi/2
    u_hi = u_start
    half_range = 0.5 * (u_hi - u_lo)
    mid = 0.5 * (u_hi + u_lo)
    if half_range <= 0.0:
        return 0.0
    n_gl = gl_x.shape[0]
    integral = 0.0
    for k in range(n_gl):
        u_k = mid + half_range * gl_x[k]
        cos_uk = math.cos(u_k)
        if cos_uk <= 1e-15:
            continue
        r_at_u = b / cos_uk
        g_val = _g_profile(r_at_u, model_id, p1, p2, p3, p4)
        # integrand = g(r) * sec²(u) * b ; jacobian for [-1,1] -> [u_lo, u_hi] is half_range
        sec2 = 1.0 / (cos_uk * cos_uk)
        integral += gl_w[k] * g_val * sec2
    return integral * b * half_range


@njit(cache=True, parallel=True)
def _simulate_phases_numba(
    gma_values,
    r,
    R,
    d1,
    d2,
    incl,
    d2h_deg,
    model_id,
    p1, p2, p3, p4,
    gl_x, gl_w,
):
    """
    Mega-kernel: compute (flx, icd, A2, l, L, h, is_eclipsed) for ALL phases.

    For each phase (parallelized via prange):
      - Compute orbital geometry (l, h, eclipse test).
      - If eclipsed, return zeros and is_eclipsed=1.
      - Otherwise iterate the polar (theta, r) grid inline: for each
        consecutive valid (i.e. unmasked) cell pair within the same theta
        ring, build the segment (av_x, av_th, av_db, A_seg) and integrate
        its LOS column with `_los_gl_quadrature`.
      - Reduce per-phase to the area-weighted mean column mean(lw)/sum(A),
        alongside the raw sums and the per-cell arrays.
    """
    n_phases = gma_values.shape[0]
    flx_out = np.zeros(n_phases)
    icd_out = np.zeros(n_phases)
    A2_out = np.zeros(n_phases)
    l_out = np.zeros(n_phases)
    L_out = np.zeros(n_phases)
    h_out = np.zeros(n_phases)
    eclipse_out = np.zeros(n_phases, dtype=np.uint8)

    n_th = int(360.0 / d2h_deg) + 1
    n_r_ring = 10

    # Per-cell LOS columns and areas. The nH -> flux conversion is nonlinear,
    # so <F(N)> != F(<N>): when the column varies steeply across the emitter
    # disk (near the occulter limb, or anywhere in physical-normalization mode)
    # the flux must be converted per cell and only then area-averaged. Callers
    # that only need the mean column can ignore these.
    n_cells_max = n_th * n_r_ring
    cell_col_out = np.zeros((n_phases, n_cells_max))
    cell_area_out = np.zeros((n_phases, n_cells_max))
    cell_count_out = np.zeros(n_phases, dtype=np.int64)
    d2h_rad = d2h_deg * math.pi / 180.0
    th_step_rad = 2.0 * math.pi / (n_th - 1)
    th_vals = np.empty(n_th, dtype=np.float64)
    cos_th_vals = np.empty(n_th, dtype=np.float64)
    for i_th in range(n_th):
        th_val = i_th * th_step_rad
        th_vals[i_th] = th_val
        cos_th_vals[i_th] = math.cos(th_val)

    # Pre-compute r-grid (shared across phases, no shared writes)
    r_min = r / 10.0
    r_step = (r - r_min) / (n_r_ring - 1)

    sin_i = math.sin(incl)
    cos_i = math.cos(incl)
    R2 = R * R

    for ip in prange(n_phases):
        cur_gma = gma_values[ip]
        sin_g = math.sin(cur_gma)
        cos_g = math.cos(cur_gma)

        h1 = d1 * sin_g * sin_i
        h2 = d2 * sin_g * sin_i
        L1 = d1 * cos_g
        L2 = d2 * cos_g
        l1 = math.sqrt(h1 * h1 + L1 * L1)
        l2 = math.sqrt(h2 * h2 + L2 * L2)
        h = h1 + h2
        L = L1 + L2
        l = l1 + l2

        z_start = (d1 + d2) * sin_g * cos_i

        # Eclipse test (only when emitter is BEHIND companion: sin_g > 0)
        is_eclipsed_phase = False
        if sin_g > 0.0:
            n_outer = l / (R + r) if (R + r) > 0.0 else 1e30
            if n_outer < 1.0:
                # Compact object disk overlaps companion projected disk
                if (R - r) > 0.0:
                    n_inner = l / (R - r)
                    if abs(n_inner) <= 1.0:
                        is_eclipsed_phase = True
                else:
                    is_eclipsed_phase = True

        l_out[ip] = l
        L_out[ip] = L
        h_out[ip] = h

        if is_eclipsed_phase:
            eclipse_out[ip] = 1
            continue

        # Walk the polar grid in (i_th, i_r) flat order, tracking the previous
        # unmasked cell so that consecutive unmasked cells within the same
        # theta ring (dx > 0) form an annular-sector segment.
        prev_is_set = False
        prev_r = 0.0
        prev_th = 0.0

        sum_lw = 0.0
        sum_A = 0.0

        for i_th in range(n_th):
            th_val = th_vals[i_th]
            cos_th = cos_th_vals[i_th]
            for i_r in range(n_r_ring):
                r_val = r_min + i_r * r_step

                # Eclipse mask (cells of compact object surface blocked)
                if sin_g > 0.0:
                    nn2 = r_val * r_val + l * l - 2.0 * r_val * l * cos_th
                    if nn2 < R2:
                        continue

                if prev_is_set and r_val > prev_r:
                    x1 = prev_r
                    x2 = r_val
                    th1 = prev_th
                    av_x = 0.5 * (x1 + x2)
                    av_th = th1 + 0.5 * d2h_rad
                    cos_avth = math.cos(av_th)
                    bv2 = av_x * av_x + l * l - 2.0 * av_x * l * cos_avth
                    if bv2 < 0.0:
                        bv2 = 0.0
                    bv = math.sqrt(bv2)
                    A_seg = 0.5 * d2h_rad * (x2 * x2 - x1 * x1)

                    los_val = _los_gl_quadrature(
                        bv, z_start, model_id, p1, p2, p3, p4, gl_x, gl_w
                    )
                    sum_lw += los_val * A_seg
                    sum_A += A_seg

                    k_cell = cell_count_out[ip]
                    cell_col_out[ip, k_cell] = los_val
                    cell_area_out[ip, k_cell] = A_seg
                    cell_count_out[ip] = k_cell + 1

                prev_r = r_val
                prev_th = th_val
                prev_is_set = True

        if sum_A > 0.0:
            flx_out[ip] = sum_lw / sum_A
        icd_out[ip] = sum_lw
        A2_out[ip] = sum_A

    return (
        flx_out, icd_out, A2_out, l_out, L_out, h_out, eclipse_out,
        cell_col_out, cell_area_out, cell_count_out,
    )

# =============================================================================
# Flux conversion (XSPEC flux-vs-nH table)
# =============================================================================


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


def load_flux_vs_nh_csv(
    csv_path: str, verbose: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
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
    
    if verbose:
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


def _build_flux_context(csv_path: str, flux_type: str) -> Dict[str, object]:
    """Build/load cached interpolation and refit context for flux conversion."""
    key = (os.path.abspath(csv_path), str(flux_type))
    cached = _FLUX_CACHE.get(key)
    if cached is not None:
        return cached

    df, bands = load_flux_vs_nh_csv(csv_path, verbose=False)
    band_data: Dict[str, Dict[str, object]] = {}

    # Sort once and reuse for all bands.
    df_sorted = df.sort_values("nH_1e22")
    nh_base = df_sorted["nH_1e22"].values
    valid_nh = np.isfinite(nh_base) & (nh_base > 0)
    nh_base = nh_base[valid_nh]

    for band in bands:
        flux_col = f"flux_{band}_{flux_type}"
        if flux_col not in df_sorted.columns:
            continue
        flux_vals = df_sorted[flux_col].values
        flux_vals = flux_vals[valid_nh]
        valid = np.isfinite(flux_vals) & (flux_vals > 0)
        if not np.any(valid):
            continue

        nh_csv = nh_base[valid]
        flux_csv = flux_vals[valid]
        interp_func = interp1d(
            np.log10(nh_csv),
            np.log10(flux_csv),
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )
        band_data[band] = {
            "nh": nh_csv,
            "flux": flux_csv,
            "interp_loglog": interp_func,
        }

    ctx: Dict[str, object] = {
        "csv_path": key[0],
        "flux_type": key[1],
        "bands": sorted(list(band_data.keys())),
        "band_data": band_data,
        "exp_fit": {},
    }
    _FLUX_CACHE[key] = ctx
    return ctx


def _interpolate_flux_from_context(
    nh_1e22: np.ndarray,
    ctx: Dict[str, object],
    band: str,
    warn_extrapolation: bool = True,
) -> np.ndarray:
    """Interpolate using prebuilt flux context."""
    band_data = ctx["band_data"]  # type: ignore[index]
    if band not in band_data:
        available = sorted(list(band_data.keys()))
        raise ValueError(
            f"Band '{band}' not present in flux cache for flux_type='{ctx['flux_type']}'. "
            f"Available bands: {available}"
        )

    info = band_data[band]
    nh_csv = info["nh"]
    interp_func = info["interp_loglog"]
    nh_min, nh_max = float(np.min(nh_csv)), float(np.max(nh_csv))
    nh_1e22 = np.asarray(nh_1e22)

    if warn_extrapolation and (
        np.any(nh_1e22 < nh_min) or np.any(nh_1e22 > nh_max)
    ):
        warnings.warn(
            f"Some nH values are outside CSV range [{nh_min:.3f}, {nh_max:.3f}] 1e22 cm^-2 for band '{band}'. "
            f"Extrapolation will be used (fill_value='extrapolate')."
        )

    nh_1e22_safe = np.clip(nh_1e22, 1e-6, 1e6)
    log_flux = interp_func(np.log10(nh_1e22_safe))
    return 10 ** log_flux



def fit_exponential_to_csv(
    df: pd.DataFrame, band: str, flux_type: str = "erg"
) -> Tuple[float, float]:
    """
    Fit exponential function A * exp(-B * nH) to CSV flux data in LOG SPACE.
    
    Fitting in log space: log(flux) = log(A) - B * nH
    This gives equal weight to all data points regardless of magnitude,
    appropriate for data spanning many orders of magnitude.
    
    Args:
        df: DataFrame from load_flux_vs_nh_csv
        band: Band name (e.g., "soft", "hard", "broad", "medium")
        flux_type: Which flux column to use — "erg" (erg/cm^2/s, default) or
                   "ph" (photons/cm^2/s)
        
    Returns:
        Tuple of (A, B) coefficients for flux = A * exp(-B * nH_1e22)
        in units determined by flux_type
        
    Raises:
        ValueError: If band/flux_type column is not found or fit fails without fallback
    """
    flux_col = f"flux_{band}_{flux_type}"

    if flux_col not in df.columns:
        available = get_available_bands_from_csv(df)
        raise ValueError(
            f"Column '{flux_col}' not found in CSV. "
            f"Available bands: {available}. "
            f"flux_type must be 'ph' or 'erg'."
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
        raise ValueError(
            f"Exponential fit failed for {band} band (flux_type='{flux_type}'): {e}"
        ) from e


def default_wind_params(wind_model: str, R: float) -> Dict[str, float]:
    """
    Return sensible default parameters for a given wind model.

    `R` is the companion radius in solar radii, used as `R_star` for the
    confinement and beta_law models.
    """
    if wind_model == "smooth_pl":
        return {"Rb": 5.0, "p": 4.0, "Delta": 2.0}
    if wind_model == "confinement":
        return {"R_star": float(R), "fconf": 10.0, "ell": 0.5}
    if wind_model == "beta_law":
        # Effective break at R_star + 3H, so H = 1 puts it near the smooth_pl
        # default Rb = 5 for a 2 Rsun companion.
        return {"R_star": float(R), "beta": 1.0, "H": 1.0}
    raise ValueError(f"Unknown wind_model '{wind_model}'")


def inclination_to_internal_rad(i0_deg: float) -> float:
    """Convert a conventional inclination into the kernels' internal angle.

    ``i0`` at the public API is the standard astronomical inclination: the angle
    between the orbital-plane normal and the line of sight, so ``i0 = 90 deg``
    is edge-on (eclipses possible) and ``i0 = 0 deg`` is face-on (the orbit lies
    in the plane of the sky and never eclipses).

    The geometry kernel (``_simulate_phases_numba``) instead measures ``incl``
    from the *line of sight*, so that
    ``h = a sin(gma) sin(incl)`` is the sky-plane offset and
    ``z = a sin(gma) cos(incl)`` the offset along the line of sight. The two
    differ by the 90 deg complement applied here; nothing downstream changes.
    """
    return (90.0 - float(i0_deg)) * np.pi / 180.0


def simulate_lightcurve(
    r: float = 0.001,
    R: float = 2.0,
    d1: float = 11.0,
    d2: float = 8.0,
    gma0: float = -90.0,
    i0: float = 64.0,
    dth: float = 1.0,
    d2h: float = 6.0,
    verbose: bool = False,
    flux_method: str = "interpolate",
    flux_csv_path: Optional[str] = None,
    flux_type: str = "erg",
    wind_model: str = "smooth_pl",
    wind_params: Optional[Dict[str, float]] = None,
    scattered_flux: float = 0.0,
    mdot: float = 4.0e-6,
    v_inf: float = 1750.0,
    mu_wind: float = MU_WIND_DEFAULT,
    f_opacity: float = 1.0,
) -> pd.DataFrame:
    """
    Main simulation function for lightcurve calculation.

    Args:
        r: Radius of smaller star B (compact object) in solar radii
        R: Radius of larger star A (companion) in solar radii
        d1: Distance of star B from COM in solar radii
        d2: Distance of star A from COM in solar radii
        gma0: Starting phase angle in degrees
        i0: Orbital inclination in degrees, standard astronomical convention:
            measured from the orbital-plane normal, so 90 deg is edge-on and
            0 deg is face-on. Converted internally by
            inclination_to_internal_rad(); the geometry itself is unchanged.
        dth: Orbital increment in degrees
        d2h: Angular cell size (degrees) for the polar grid used in the surface integral
        verbose: If True, prints a one-line summary of the kernel call
        flux_method: Method for converting nH to flux. Options:
            - "interpolate": log-log interpolation of the CSV flux vs nH table
              (default)
            - "refit": fit exponentials A*exp(-B*nH) to the same CSV table
        flux_csv_path: Path to CSV file from compute_flux_vs_nH.py (required)
        flux_type: Which flux column from the CSV to use — "erg" (erg/cm^2/s,
            default) or "ph" (photons/cm^2/s).
        wind_model: Name of the dimensionless wind density profile, one of
            "smooth_pl", "confinement" or "beta_law". Default "smooth_pl".
        wind_params: Dict of profile parameters (see WIND_MODEL_PARAM_KEYS).
            If None, uses defaults from default_wind_params(wind_model, R).
        scattered_flux: Constant additive flux offset applied to all ``nfl_*``
            columns after eclipse handling. Useful for modeling phase-invariant
            scattered flux floors.
        mdot: WR mass-loss rate in Msun/yr, setting the absolute wind density.
        v_inf: Wind terminal velocity in km/s.
        mu_wind: Mean mass per hydrogen-equivalent nucleus, converting the wind
            mass column into the N_H the solar-abundance TBabs table expects.
        f_opacity: Effective-opacity factor applied to the Mdot-derived column,
            absorbing wind photoionization, clumping and WR abundance
            departures.

    Returns:
        DataFrame with simulation results. Key columns:
            - flx: Raw dimensionless mean wind LOS integral per phase
            - fl: Absolute column density N_H in 1e22 cm^-2
            - nfl_{band}: Band flux, area-averaged over the emitter disk

    Notes:
        - The density normalization n_0 is fixed from mdot / v_inf, so ``fl``
          carries real units. The eclipse therefore emerges from wind opacity
          rather than from a geometric cutoff, and ``R`` means the true
          photosphere rather than an effective opaque radius.
        - The nH -> flux conversion is nonlinear, so the band flux is computed
          per emitter cell and only then area-averaged: <F(N)> != F(<N>) when
          the column varies steeply across the disk, which it does during
          ingress/egress and throughout the eclipse core.
    """
    # Convert angles to radians
    gma = gma0 * np.pi / 180
    # Only the input convention changes here: `i` is the internal angle from the
    # line of sight that every geometry expression below already assumes.
    i = inclination_to_internal_rad(i0)

    # Wind profile parameter packing (done once per call)
    if wind_params is None:
        wind_params = default_wind_params(wind_model, R)
    # For the profiles anchored at the photosphere, auto-fill R_star from R
    # if the caller omitted it.
    if wind_model in R_STAR_TIED_MODELS and "R_star" not in wind_params:
        wind_params = dict(wind_params)
        wind_params["R_star"] = float(R)
    model_id, p1, p2, p3, p4 = pack_wind_params(wind_model, wind_params)

    # Prepare phase values
    n_iterations = int(360 / dth)
    gma_values = gma + (np.arange(n_iterations) * (dth * np.pi / 180.0))

    # One numba parallel call covers every phase; the Gauss-Legendre quadrature
    # integrates the full z-tail, so there is no cutoff radius to choose.
    (flx_arr, icd_arr, A2_arr, l_arr, L_arr, h_arr, eclipsed_arr,
     cell_col_arr, cell_area_arr, cell_count_arr) = _simulate_phases_numba(
        gma_values.astype(np.float64),
        float(r),
        float(R),
        float(d1),
        float(d2),
        float(i),
        float(d2h),
        int(model_id),
        float(p1), float(p2), float(p3), float(p4),
        _GL16_X, _GL16_W,
    )
    if verbose:
        print(f"Computed {n_iterations} phases via mega-kernel "
              f"(GL quadrature, parallel over phases)")

    deg = gma_values * (180.0 / np.pi)
    results = pd.DataFrame(
        {
            "deg": deg,
            "ph": np.asarray(gma_values, dtype=float),
            "phase": (gma_values - (gma0 * np.pi / 180.0)) / (2.0 * np.pi),
            "A2": np.asarray(A2_arr, dtype=float),
            "flx": np.asarray(flx_arr, dtype=float),
            "icd": np.asarray(icd_arr, dtype=float),
            "time": deg * 348.42,
            "l3": np.asarray(l_arr, dtype=float),
            "L3": np.asarray(L_arr, dtype=float),
            "h3": np.asarray(h_arr, dtype=float),
            "is_eclipsed": np.asarray(eclipsed_arr, dtype=bool),
        }
    )

    # ------------------------------------------------------------------
    # Column-density normalization.
    #
    # n_0 is set from Mdot / v_inf, so fl carries real units. This breaks the
    # scale degeneracy that an orbit-averaged rescaling would leave behind and
    # lets the eclipse emerge from wind opacity instead of from a geometric
    # cutoff.
    # ------------------------------------------------------------------
    n0 = wind_density_norm_from_mdot(
        mdot, v_inf, wind_model, wind_params, mu=mu_wind
    )
    # fl is in units of 1e22 cm^-2; flx is the LOS integral of g in R_sun.
    # f_opacity is an effective-opacity factor absorbing wind ionization,
    # clumping and abundance departures from the solar-abundance TBabs table
    # (a hyper-ionized wind has far less photoelectric opacity than its mass
    # column implies).
    col_scale = float(f_opacity) * n0 * R_SUN_CM / 1.0e22
    results["fl"] = results["flx"].to_numpy(dtype=float) * col_scale

    # Build one nH -> flux mapping per band.
    if flux_csv_path is None:
        raise ValueError(f"flux_csv_path is required for flux_method='{flux_method}'")

    band_maps: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    ctx = _build_flux_context(flux_csv_path, flux_type=flux_type)
    available_bands = ctx["bands"]  # type: ignore[index]
    if verbose:
        print(f"Detected energy bands in CSV: {', '.join(available_bands)}")

    if flux_method == "interpolate":
        for band in available_bands:
            band_maps[band] = (
                lambda n, _b=band: _interpolate_flux_from_context(
                    n, ctx, _b, warn_extrapolation=False,
                )
            )
    elif flux_method == "refit":
        exp_fit_cache = ctx["exp_fit"]  # type: ignore[index]
        # Keep behavior identical to a direct fit_exponential_to_csv call by
        # constructing the same validated DataFrame once.
        df_flux, _ = load_flux_vs_nh_csv(flux_csv_path, verbose=False)
        for band in available_bands:
            if band in exp_fit_cache:
                A, B = exp_fit_cache[band]
            else:
                A, B = fit_exponential_to_csv(df_flux, band, flux_type=flux_type)
                exp_fit_cache[band] = (A, B)
            band_maps[band] = lambda n, _A=A, _B=B: _A * np.exp(-_B * n)
    else:
        raise ValueError(
            f"Invalid flux_method: {flux_method}. "
            "Must be 'interpolate' or 'refit'"
        )

    # Per-cell columns, so the nonlinear nH -> flux map is applied before the
    # area average rather than after it.
    cell_nh = np.asarray(cell_col_arr, dtype=float) * col_scale
    cell_A = np.asarray(cell_area_arr, dtype=float)
    counts = np.asarray(cell_count_arr, dtype=np.int64)
    valid = np.arange(cell_A.shape[1])[None, :] < counts[:, None]
    cell_A = np.where(valid, cell_A, 0.0)
    area_tot = cell_A.sum(axis=1)

    for band, fmap in band_maps.items():
        try:
            # <F(N)> over the emitter disk, NOT F(<N>): during ingress and in
            # the eclipse core the column varies by orders of magnitude across
            # the disk, and the surviving flux is dominated by the
            # least-absorbed cells.
            per_cell = fmap(cell_nh.reshape(-1)).reshape(cell_nh.shape)
            num = np.einsum("ij,ij->i", np.nan_to_num(per_cell), cell_A)
            results[f"nfl_{band}"] = np.divide(
                num, area_tot,
                out=np.zeros_like(num), where=area_tot > 0,
            )
        except Exception as e:
            warnings.warn(f"Failed to compute flux for band '{band}': {e}")

    # Set all scaled flux columns to 0 when eclipsed.
    # During eclipse the emitter is physically blocked - flux should be zero,
    # not computed from the absorption formula (which would give max flux at nH=0).
    eclipse_mask = results["is_eclipsed"].values
    if np.any(eclipse_mask):
        flux_cols = [col for col in results.columns if col.startswith("nfl_")]
        for col in flux_cols:
            results.loc[eclipse_mask, col] = 0.0

    if float(scattered_flux) != 0.0:
        flux_cols = [col for col in results.columns if col.startswith("nfl_")]
        for col in flux_cols:
            results[col] = results[col].astype(float) + float(scattered_flux)

    return results


# =============================================================================
# Wind density normalization
# =============================================================================
#
# Units note: the LOS integrator returns a dimensionless integral
#   flx_code = <∫ g(r) dz>_cells
# where r and z are in solar radii (R_sun = 6.957e10 cm) and g(r) is
# dimensionless. The physical LOS column density at phase phi is
#   N_H(phi) = n_0 * R_sun * ∫ g(r(phi, z)) dz
#           = n_0 * R_sun * flx_code(phi)
# where n_0 is the "reference" number density such that the physical number
# density at a point with dimensionless g value g(r) is n(r) = n_0 * g(r).
#
# n_0 is fixed from the mass-loss rate by matching the asymptotic r^-2 limit of
# g to a spherical constant-velocity wind; see wind_density_norm_from_mdot.
# The number density at any radius is then n(r) = n_0 * g(r; params), so the
# companion-surface density is n(R_star) = n_0 * g(R_star; params).

def wind_asymptotic_coefficient(
    wind_model: str, wind_params: Dict[str, float]
) -> float:
    """Return C with ``g(r) -> C / r^2`` as ``r -> inf`` (r in solar radii).

    Needed to tie the dimensionless profile to a physical mass-loss rate: far
    from the star every supported profile relaxes to a constant-velocity
    ``r^-2`` wind, and C is whatever prefactor that limit carries.
    """
    if wind_model == "smooth_pl":
        Rb = float(wind_params["Rb"])
        return Rb * Rb
    if wind_model in ("confinement", "beta_law"):
        # Both are written directly as Mdot / (4 pi r^2 v_inf) times a factor
        # that tends to 1, so n_0 is exactly the terminal-velocity density
        # normalization.
        return 1.0
    raise ValueError(f"Unknown wind_model '{wind_model}'")


def wind_density_norm_from_mdot(
    mdot_msun_yr: float,
    v_inf_kms: float,
    wind_model: str,
    wind_params: Dict[str, float],
    mu: float = MU_WIND_DEFAULT,
) -> float:
    """Absolute density normalization ``n_0`` [cm^-3] from Mdot and v_inf.

    The profile is used as ``n(r) = n_0 * g(r)``. Matching the asymptotic
    ``r^-2`` limit to a spherical constant-velocity wind,
    ``n(r) = Mdot / (4 pi (r R_sun)^2 v_inf mu m_H)``, gives

        n_0 = Mdot / (4 pi R_sun^2 v_inf mu m_H C)

    with C from :func:`wind_asymptotic_coefficient`.

    This carries real units, which is what makes the light curve sensitive to
    the *absolute* size of the system rather than only to ratios such as R/a.
    """
    mdot_cgs = float(mdot_msun_yr) * M_SUN_G / YEAR_S
    v_cgs = float(v_inf_kms) * KM_TO_CM
    C = wind_asymptotic_coefficient(wind_model, wind_params)
    denom = 4.0 * np.pi * (R_SUN_CM ** 2) * v_cgs * float(mu) * M_H_G * C
    if denom <= 0.0:
        raise ValueError("Non-positive denominator in wind density normalization.")
    return mdot_cgs / denom



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
        "--i0", type=float, default=64.0,
        help="Orbital inclination in degrees from the orbital-plane normal "
             "(90 = edge-on, 0 = face-on)",
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
        "--verbose",
        action="store_true",
        help="Print a one-line kernel summary during simulation",
    )
    parser.add_argument(
        "--flux_method",
        type=str,
        choices=["interpolate", "refit"],
        default="interpolate",
        help="Method for converting nH to flux: 'interpolate' (log-log "
        "interpolation of the CSV table, default) or 'refit' (fit new "
        "exponentials to the CSV table)",
    )
    parser.add_argument(
        "--flux_csv",
        type=str,
        required=True,
        help="Path to flux vs nH CSV file from compute_flux_vs_nH.py",
    )
    parser.add_argument(
        "--flux_type",
        type=str,
        choices=["erg", "ph"],
        default="erg",
        help="Which flux column from the CSV to use: "
        "'erg' (erg/cm^2/s, default) or 'ph' (photons/cm^2/s).",
    )
    parser.add_argument(
        "--mdot",
        type=float,
        default=4.0e-6,
        help="WR mass-loss rate in Msun/yr, setting the absolute wind density. "
        "Default 4e-6 (Clark & Crowther 2004, clumping-corrected).",
    )
    parser.add_argument(
        "--v-inf",
        type=float,
        default=1750.0,
        help="Wind terminal velocity in km/s. "
        "Default 1750 (Clark & Crowther 2004).",
    )
    parser.add_argument(
        "--mu-wind",
        type=float,
        default=MU_WIND_DEFAULT,
        help="Mean mass per hydrogen-equivalent nucleus, converting the wind "
        "mass column into the N_H that the solar-abundance TBabs flux_vs_nH "
        f"table expects. Default {MU_WIND_DEFAULT}.",
    )
    parser.add_argument(
        "--f-opacity",
        type=float,
        default=1.0,
        help="Effective-opacity factor applied to the Mdot-derived column. "
        "Absorbs wind photoionization, clumping and WR abundance departures. "
        "Clark & Crowther's Mdot overpredicts the observed N_H by ~1.5-2 dex, "
        "so values around 0.01-0.03 reproduce IC 10 X-1. "
        "Default 1.0 (no correction).",
    )
    parser.add_argument(
        "--wind-model",
        type=str,
        choices=list(WIND_MODEL_IDS.keys()),
        default="smooth_pl",
        help="Dimensionless wind density profile to use. One of: "
        "smooth_pl, confinement, beta_law. Default: smooth_pl.",
    )
    parser.add_argument(
        "--Rb",
        type=float,
        default=5.0,
        help="Break radius (solar radii) for smooth_pl. Default: 5.0.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=4.0,
        help="Inner-region power-law slope for smooth_pl. Default: 4.0.",
    )
    parser.add_argument(
        "--Delta",
        type=float,
        default=2.0,
        # Must match default_wind_params() and the MCMC's WIND_SHAPE_FIXED, or a
        # CLI-generated model would use a different break sharpness than the one
        # the MCMC fitted.
        help="Smoothness parameter for smooth_pl. Larger -> sharper break. Default: 2.0.",
    )
    parser.add_argument(
        "--fconf",
        type=float,
        default=10.0,
        help="Confinement overdensity amplitude for confinement model. Default: 10.0.",
    )
    parser.add_argument(
        "--ell",
        type=float,
        default=0.5,
        help="Confinement scale length (solar radii) for confinement model. Default: 0.5.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="CAK velocity-law exponent for beta_law. Default: 1.0.",
    )
    parser.add_argument(
        "--H",
        type=float,
        default=1.0,
        help="Inner acceleration scale height (solar radii) for beta_law; the "
        "effective break radius is R + 3H. Default: 1.0.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="xrb_lightcurve_output.csv",
        help="Output file name for results",
    )

    args = parser.parse_args()

    # Collect wind-model parameters
    if args.wind_model == "smooth_pl":
        wind_params = {"Rb": args.Rb, "p": args.p, "Delta": args.Delta}
    elif args.wind_model == "confinement":
        wind_params = {"R_star": args.R, "fconf": args.fconf, "ell": args.ell}
    elif args.wind_model == "beta_law":
        wind_params = {"R_star": args.R, "beta": args.beta, "H": args.H}
    else:
        parser.error(f"Unsupported wind_model: {args.wind_model}")

    print("Starting XRB Lightcurve Simulation...")
    print("Parameters:")
    print(f"  r (emitter radius): {args.r} solar radii")
    print(f"  R (companion radius): {args.R} solar radii")
    print(f"  d1 (emitter separation): {args.d1} solar radii")
    print(f"  d2 (companion separation): {args.d2} solar radii")
    print(f"  gma0 (starting phase): {args.gma0} degrees")
    print(f"  i0 (inclination): {args.i0} degrees from the orbital-plane normal "
          f"(90 = edge-on)")
    print(f"  dth (orbital increment): {args.dth} degrees")
    print(f"  d2h (polar cell size): {args.d2h} degrees")
    print(f"  flux_method: {args.flux_method}")
    print(f"  flux_csv: {args.flux_csv}")
    print(f"  flux_type: {args.flux_type}")
    print(f"  mdot: {args.mdot} Msun/yr")
    print(f"  v_inf: {args.v_inf} km/s")
    print(f"  mu_wind: {args.mu_wind}")
    print(f"  f_opacity: {args.f_opacity}")
    print(f"  wind_model: {args.wind_model}")
    print(f"  wind_params: {wind_params}")
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
        verbose=args.verbose,
        flux_method=args.flux_method,
        flux_csv_path=args.flux_csv,
        flux_type=args.flux_type,
        wind_model=args.wind_model,
        wind_params=wind_params,
        mdot=args.mdot,
        v_inf=args.v_inf,
        mu_wind=args.mu_wind,
        f_opacity=args.f_opacity,
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
