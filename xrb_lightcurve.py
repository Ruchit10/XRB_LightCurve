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

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):                    # noqa: E303
        """No-op decorator when numba is not installed."""
        def _decorator(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return _decorator

    prange = range  # type: ignore[assignment]

# =============================================================================
# Wind density profiles
# =============================================================================
#
# Each wind model defines a dimensionless density profile g(r) with r in solar
# radii. The absolute scale of g is arbitrary — the simulation normalizes the
# wind LOS integral so that mean(fl) = lam (the target mean nH from a fit).
#
# Supported models (Wind_Density.pdf):
#   0 broken_pl   — piecewise power law; params (Rb, p)
#   1 smooth_pl   — smoothly broken power law; params (Rb, p, Delta)
#   2 beta_law    — velocity-based (exponential + CAK beta-law); params (R_star, beta, H)
#   3 confinement — inner confinement / compression; params (R_star, fconf, ell)

WIND_MODEL_IDS: Dict[str, int] = {
    "broken_pl": 0,
    "smooth_pl": 1,
    "beta_law": 2,
    "confinement": 3,
}

WIND_MODEL_PARAM_KEYS: Dict[str, Tuple[str, ...]] = {
    "broken_pl": ("Rb", "p"),
    "smooth_pl": ("Rb", "p", "Delta"),
    "beta_law": ("R_star", "beta", "H"),
    "confinement": ("R_star", "fconf", "ell"),
}


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
        # broken_pl: Rb=p1, p=p2
        Rb = p1
        p_slope = p2
        x = r / Rb
        if x <= 1.0:
            return x ** (-p_slope)
        return x ** (-2.0)

    if model_id == 1:
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

    if model_id == 2:
        # beta_law: R_star=p1, beta=p2, H=p3
        R_star = p1
        beta = p2
        H = p3
        if r <= R_star:
            return 0.0
        u1 = 1.0 - math.exp(-(r - R_star) / H)
        u2 = (1.0 - R_star / r) ** beta
        v = u1 * u2
        if v <= 0.0:
            return 0.0
        return 1.0 / (r * r * v)

    if model_id == 3:
        # confinement: R_star=p1, fconf=p2, ell=p3
        R_star = p1
        fconf = p2
        ell = p3
        factor = 1.0 + fconf * math.exp(-(r - R_star) / ell)
        return factor / (r * r)

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
        x = np.where(r > 0.0, r / Rb, np.inf)
        return np.where(x <= 1.0, x ** (-p_slope), x ** (-2.0))

    if model_id == 1:
        Rb = p1
        p_slope = p2
        Delta = p3
        if Delta <= 0.0:
            x = np.where(r > 0.0, r / Rb, np.inf)
            return x ** (-2.0)
        x = np.where(r > 0.0, r / Rb, np.inf)
        base = x ** (-2.0)
        bracket = 1.0 + (1.0 / x) ** Delta
        exponent = (p_slope - 2.0) / Delta
        return base * (bracket ** exponent)

    if model_id == 2:
        R_star = p1
        beta = p2
        H = p3
        out = np.zeros_like(r)
        mask = r > R_star
        if np.any(mask):
            rr = r[mask] if r.ndim > 0 else np.array([float(r)])
            u1 = 1.0 - np.exp(-(rr - R_star) / H)
            u2 = np.where(rr > R_star, (1.0 - R_star / rr) ** beta, 0.0)
            v = u1 * u2
            inner = np.where(v > 0.0, 1.0 / (rr * rr * v), 0.0)
            if r.ndim > 0:
                out[mask] = inner
                return out
            return inner[0]
        return out

    if model_id == 3:
        R_star = p1
        fconf = p2
        ell = p3
        safe_r = np.where(r > 0.0, r, np.inf)
        factor = 1.0 + fconf * np.exp(-(safe_r - R_star) / ell)
        return factor / (safe_r * safe_r)

    return np.zeros_like(r)


# =============================================================================
# Numba-accelerated LOS integration kernels
# =============================================================================
#
# Two LOS integration paths exist:
#
#  1) `_wind_los_profile_numba`  — original adaptive trapezoid in z. Used by
#     the standalone `wind_los_integral` for backwards compatibility / debug.
#
#  2) `_simulate_phases_numba`   — fast mega-kernel that, for each phase,
#     builds the polar emitter grid inline and integrates every cell's LOS
#     using fixed-node Gauss-Legendre quadrature with the substitution
#     u = arctan(z/b). This collapses the slowly-decaying r^{-2} tail to a
#     bounded smooth integrand on a finite interval, so 16 GL nodes per cell
#     give >10 digits of accuracy for any wind profile and any impact
#     parameter (no special-casing of b vs Rb). The whole 360-phase loop
#     runs under one numba @njit(parallel=True) call with prange over phases,
#     eliminating per-phase Python overhead and per-call thread launches.

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
      - Reduce per-phase to mean(lw)/sum(A) etc., matching the legacy
        `_compute_one_phase` outputs.
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
    d2h_rad = d2h_deg * math.pi / 180.0
    th_step_rad = 2.0 * math.pi / (n_th - 1)

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
        # theta ring (dx > 0) form a segment — matches the legacy create_grid.
        prev_is_set = False
        prev_r = 0.0
        prev_th = 0.0

        sum_lw = 0.0
        sum_A = 0.0

        for i_th in range(n_th):
            th_val = i_th * th_step_rad
            cos_th = math.cos(th_val)
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

                prev_r = r_val
                prev_th = th_val
                prev_is_set = True

        if sum_A > 0.0:
            flx_out[ip] = sum_lw / sum_A
        icd_out[ip] = sum_lw
        A2_out[ip] = sum_A

    return flx_out, icd_out, A2_out, l_out, L_out, h_out, eclipse_out


@njit(cache=True, parallel=True)
def _wind_los_profile_numba(
    av_db,
    A,
    z_start,
    dz,
    model_id,
    p1,
    p2,
    p3,
    p4,
    Rmax,
    converge_rmax,
    eps_rel,
    min_steps,
    r_cap,
):
    """
    Generic LOS integral of a user-selected dimensionless density profile g(r).

    For each cell i, integrates g(sqrt(b^2 + z^2)) along z using a fixed-step
    trapezoidal rule starting at z = z_start and stepping by -dz.

    - converge_rmax: adaptive stopping once the per-step contribution falls
      below `eps_rel * integral_so_far` (after `min_steps` steps and once a
      positive running integral is accumulated). The max-contribution test is
      also applied so that the integrator does not terminate before it has
      crossed the peak of the integrand (important when the LOS starts far in
      front of or behind the wind center). `r_cap` is a hard safety cap.
    - otherwise: integrate from z = z_start down to z = -sqrt(Rmax^2 - b^2).

    Returns (lw, los_arr) where lw = los * A, los_arr = unweighted integral.
    """
    N = len(av_db)
    lw = np.zeros(N)
    los_arr = np.zeros(N)

    # Cell loop is embarrassingly parallel: each iteration only writes to its
    # own (lw[i], los_arr[i]) slot. Numba parallel=True + prange dispatches
    # cells across CPU threads.
    for i in prange(N):
        b = av_db[i]
        if b < 1e-8:
            b = 1e-8
        b2 = b * b

        integral = 0.0

        if converge_rmax:
            z = z_start
            r0 = math.sqrt(b2 + z * z)
            g_prev = _g_profile(r0, model_id, p1, p2, p3, p4)
            max_contrib = 0.0
            # Track integral as of when max_contrib was last updated; this lets
            # the termination check compare the current step to how much
            # integrand mass accumulated after the peak was seen.
            integral_at_peak = 0.0
            k = 0
            while True:
                k += 1
                z -= dz
                r_cur = math.sqrt(b2 + z * z)
                g_cur = _g_profile(r_cur, model_id, p1, p2, p3, p4)
                contrib = 0.5 * (g_prev + g_cur) * dz
                integral += contrib
                abs_contrib = abs(contrib)
                if abs_contrib > max_contrib:
                    max_contrib = abs_contrib
                    integral_at_peak = integral
                g_prev = g_cur

                if r_cur > r_cap:
                    break
                if k > min_steps and integral > 0.0:
                    # Terminate once the step contribution is a small fraction
                    # of the integral accumulated since the peak. This handles
                    # both (a) monotonic-tail starts (peak at step 1, integral
                    # grows fast) and (b) peak-crossing starts (integral_at_peak
                    # set when peak is passed, then tail fraction shrinks).
                    denom = integral - integral_at_peak
                    if denom <= 0.0:
                        denom = integral
                    if abs_contrib < eps_rel * denom:
                        break
        else:
            t2 = Rmax * Rmax - b2
            if t2 <= 0.0:
                continue
            t = math.sqrt(t2)
            if abs(z_start) > t:
                continue
            end_k = int(math.floor((z_start + t) / dz))
            r0 = math.sqrt(b2 + z_start * z_start)
            g_prev = _g_profile(r0, model_id, p1, p2, p3, p4)
            for k in range(1, end_k + 1):
                z = z_start - dz * k
                r_cur = math.sqrt(b2 + z * z)
                g_cur = _g_profile(r_cur, model_id, p1, p2, p3, p4)
                integral += 0.5 * (g_prev + g_cur) * dz
                g_prev = g_cur

        lw[i] = integral * A[i]
        los_arr[i] = integral

    return lw, los_arr


# =============================================================================
# Grid construction
# =============================================================================

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
    model_id: int,
    p1: float,
    p2: float,
    p3: float,
    p4: float,
    dz: float = 0.5,
    Rmax: Optional[float] = None,
    converge_rmax: bool = False,
    conv_eps_rel: float = 1e-4,
    conv_min_steps: int = 50,
    conv_r_cap_mult: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate wind column integral along the line of sight for a chosen density profile.

    For each emitter cell, integrates the dimensionless profile g(r) along the
    LOS using a fixed-step trapezoidal rule.

    Args:
        d: Separation between stars
        d1: Distance of star B from COM
        d2: Distance of star A from COM
        gma: Phase angle in radians
        i: Inclination angle in radians
        av_x, av_th, av_db: Grid arrays
        A: Area array
        model_id, p1..p4: Packed wind profile parameters (see pack_wind_params).
        dz: Step along line of sight (solar radii)
        Rmax: Maximum radius (solar radii) for fixed-cutoff LOS integration.
        converge_rmax: If True, ignore Rmax and integrate adaptively until tail
            contributions become negligible.
        conv_eps_rel: Relative convergence tolerance for adaptive stopping
            (compared to max step contribution seen so far).
        conv_min_steps: Minimum number of dz-steps before convergence check starts.
        conv_r_cap_mult: Safety cap for adaptive integration, as multiple of d
            (stop when r >= cap).

    Returns:
        Tuple of (lw, icd, A2) arrays where lw[i] = los[i] * A[i],
        icd = los_values without area weight, A2 = A passed through.
    """
    if av_db is None or av_db.size == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    # z start and bounds (identical for each cell within a phase)
    z1 = d1 * np.sin(gma) * np.cos(i)
    z2 = d2 * np.sin(gma) * np.cos(i)
    z_start = z1 + z2

    r_cap = float(conv_r_cap_mult) * float(d)
    converge_flag = bool(converge_rmax) or (Rmax is None)
    Rmax_use = float(Rmax) if Rmax is not None else 0.0

    if HAS_NUMBA:
        lw, los = _wind_los_profile_numba(
            av_db.astype(np.float64),
            A.astype(np.float64),
            float(z_start),
            float(dz),
            int(model_id),
            float(p1),
            float(p2),
            float(p3),
            float(p4),
            Rmax_use,
            converge_flag,
            float(conv_eps_rel),
            int(conv_min_steps),
            r_cap,
        )
        return lw, los, A.astype(np.float64)

    # --- numpy fallback (vectorized fixed-step trapezoidal) ---
    b = av_db.astype(float)
    b_safe = np.maximum(b, 1e-8)
    g_name = _id_to_name(int(model_id))
    g_params = _unpack_params(int(model_id), float(p1), float(p2), float(p3), float(p4))

    if converge_flag:
        # Vectorized approximation of the adaptive path: integrate out to a
        # fixed cap on every cell. Profiles in this code fall off at least as
        # r^-2 so 200 solar radii already captures >99% of the LOS integral
        # for realistic geometries; the `r_cap` passed in is only used as an
        # upper bound here to cap memory usage in the numpy fallback.
        z_max_extent = float(min(r_cap, 200.0))
        max_steps = int(math.ceil((float(z_start) + z_max_extent) / float(dz)))
        if max_steps < 2:
            return (
                np.zeros_like(b),
                np.zeros_like(b),
                A.astype(float),
            )
        k_arr = np.arange(max_steps + 1, dtype=float)
        z_vals = float(z_start) - float(dz) * k_arr
        r_grid = np.sqrt((b_safe ** 2)[:, None] + (z_vals[None, :]) ** 2)
        g_vals = evaluate_g_profile(r_grid, g_name, g_params)
        # Zero-out contributions past r_cap to tighten the tail
        g_vals = np.where(r_grid <= r_cap, g_vals, 0.0)
        inner = 0.5 * (g_vals[:, :-1] + g_vals[:, 1:])
        los = float(dz) * np.sum(inner, axis=1)
        lw = los * A.astype(float)
        return lw.astype(float), los.astype(float), A.astype(float)

    # Fixed-Rmax numpy fallback
    t = np.sqrt(np.maximum((Rmax_use ** 2) - (b ** 2), 0.0))
    valid_cells = (t > 0) & (np.abs(z_start) <= t)
    end_k = np.floor((z_start + t) / dz).astype(int)
    end_k = np.where(valid_cells, end_k, -1)
    max_steps = int(end_k.max()) if end_k.size else -1
    if max_steps < 0:
        return (
            np.zeros_like(b),
            np.zeros_like(b),
            A.astype(float),
        )
    k_arr = np.arange(max_steps + 1, dtype=float)
    z_vals = z_start - dz * k_arr
    r_grid = np.sqrt((b_safe ** 2)[:, None] + (z_vals[None, :]) ** 2)
    step_mask = (k_arr[None, :] <= end_k[:, None]) & valid_cells[:, None]
    g_vals = evaluate_g_profile(r_grid, g_name, g_params)
    g_vals = np.where(step_mask, g_vals, 0.0)
    inner = 0.5 * (g_vals[:, :-1] + g_vals[:, 1:])
    los = float(dz) * np.sum(inner, axis=1)
    lw = los * A.astype(float)
    return lw.astype(float), los.astype(float), A.astype(float)


def _id_to_name(model_id: int) -> str:
    for name, mid in WIND_MODEL_IDS.items():
        if mid == model_id:
            return name
    raise ValueError(f"Unknown wind model id: {model_id}")


def _unpack_params(model_id: int, p1: float, p2: float, p3: float, p4: float) -> Dict[str, float]:
    name = _id_to_name(model_id)
    keys = WIND_MODEL_PARAM_KEYS[name]
    vals = (p1, p2, p3, p4)[: len(keys)]
    return dict(zip(keys, vals))


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
    nh_1e22: np.ndarray, df: pd.DataFrame, band: str, flux_type: str = "erg"
) -> np.ndarray:
    """
    Interpolate flux values for given nH array using CSV data.
    
    Args:
        nh_1e22: Array of nH values in 1e22 cm^-2 units
        df: DataFrame from load_flux_vs_nh_csv
        band: Band name (e.g., "soft", "hard", "broad", "medium")
        flux_type: Which flux column to use — "erg" (erg/cm^2/s, default) or
                   "ph" (photons/cm^2/s)
        
    Returns:
        Array of interpolated flux values in units determined by flux_type
        
    Raises:
        ValueError: If band/flux_type column is not found in DataFrame
    """
    flux_col = f"flux_{band}_{flux_type}"

    if flux_col not in df.columns:
        available = get_available_bands_from_csv(df)
        raise ValueError(
            f"Column '{flux_col}' not found in CSV. "
            f"Available bands: {available}. "
            f"flux_type must be 'ph' or 'erg'."
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
        # Fallback legacy values are photon-flux-based; only apply for flux_type="ph"
        if flux_type == "ph":
            if band == "hard":
                warnings.warn(f"Exponential fit failed for {band} band: {e}. Using legacy ph values.")
                return 9.524e-13, 0.057
            elif band == "soft":
                warnings.warn(f"Exponential fit failed for {band} band: {e}. Using legacy ph values.")
                return 9.3923e-13, 2.5062
        raise ValueError(
            f"Exponential fit failed for {band} band (flux_type='{flux_type}'): {e}"
        ) from e


def default_wind_params(wind_model: str, R: float) -> Dict[str, float]:
    """
    Return sensible default parameters for a given wind model.

    `R` is the companion radius in solar radii, used as `R_star` for the
    velocity-based and confinement models.
    """
    if wind_model == "broken_pl":
        return {"Rb": 5.0, "p": 4.0}
    if wind_model == "smooth_pl":
        return {"Rb": 5.0, "p": 4.0, "Delta": 2.0}
    if wind_model == "beta_law":
        return {"R_star": float(R), "beta": 1.0, "H": 1.0}
    if wind_model == "confinement":
        return {"R_star": float(R), "fconf": 10.0, "ell": 0.5}
    raise ValueError(f"Unknown wind_model '{wind_model}'")


def simulate_lightcurve(
    r: float = 0.001,
    R: float = 2.0,
    d1: float = 11.0,
    d2: float = 8.0,
    gma0: float = -90.0,
    i0: float = 26.0,
    dth: float = 1.0,
    d2h: float = 6.0,
    dz: float = 0.5,
    verbose: bool = False,
    n_jobs: int = 1,
    flux_method: str = "legacy",
    flux_csv_path: Optional[str] = None,
    flux_type: str = "erg",
    lam: float = 0.589537,
    Rmax: Optional[float] = None,
    converge_rmax: bool = False,
    wind_model: str = "smooth_pl",
    wind_params: Optional[Dict[str, float]] = None,
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
        flux_type: Which flux column from the CSV to use — "erg" (erg/cm^2/s, default)
            or "ph" (photons/cm^2/s). Only applies when flux_method is "interpolate" or "refit".
        lam: Target mean nH in 1e22 cm^-2 units. The raw wind integral (flx) is
            scaled so that mean(fl) = lam. Default 0.589537 (i.e., mean nH ≈ 5.9e21 cm^-2).
        Rmax: Maximum radius (solar radii) used as a hard cutoff for LOS integration.
            If None, the LOS integration uses adaptive convergence stopping (see converge_rmax).
            Note: the CLI sets the default to 2*(d1+d2), reproducing the legacy cutoff.
        converge_rmax: If True, ignore fixed Rmax and integrate adaptively until tail contributions are negligible.
        wind_model: Name of the dimensionless wind density profile. One of
            "broken_pl", "smooth_pl", "beta_law", "confinement". Default "smooth_pl".
        wind_params: Dict of profile parameters (see WIND_MODEL_PARAM_KEYS).
            If None, uses defaults from default_wind_params(wind_model, R).

    Returns:
        DataFrame with simulation results. Key columns:
            - flx: Raw (unscaled) mean wind LOS integral per phase
            - fl: Scaled nH values (1e22 cm^-2 units), mean(fl) = lam
            - nfl_{band}: Photon or energy flux per band (depending on flux_method)

    Notes:
        - The column density integral (flx) has arbitrary units until scaled.
        - The scaling factor is computed as lam / mean(flx), so mean(fl) = lam.
        - fl values are in units of 1e22 cm^-2 (e.g., fl=1.0 means nH = 1.0e22 cm^-2).
    """
    # Convert angles to radians
    gma = gma0 * np.pi / 180
    i = i0 * np.pi / 180
    d = d1 + d2

    # Wind profile parameter packing (done once per call)
    if wind_params is None:
        wind_params = default_wind_params(wind_model, R)
    # For velocity-based and confinement profiles, auto-fill R_star from R
    # if the caller omitted it.
    if wind_model in ("beta_law", "confinement") and "R_star" not in wind_params:
        wind_params = dict(wind_params)
        wind_params["R_star"] = float(R)
    model_id, p1, p2, p3, p4 = pack_wind_params(wind_model, wind_params)

    # Integration cutoff handling
    # - If converge_rmax is enabled OR Rmax is None: use adaptive stopping
    # - Otherwise: use fixed cutoff at Rmax
    converge_rmax_use = bool(converge_rmax) or (Rmax is None)
    Rmax_use: Optional[float] = None if Rmax is None else float(Rmax)

    # Prepare phase values
    n_iterations = int(360 / dth)
    gma_values = gma + (np.arange(n_iterations) * (dth * np.pi / 180.0))

    # ------------------------------------------------------------------
    # Fast path: mega-kernel that processes ALL phases inside one numba
    # parallel call using Gauss-Legendre quadrature for the LOS integral.
    # Requirements:
    #   - numba available
    #   - using adaptive integration limits (Rmax_use is None or
    #     converge_rmax_use is True) — the GL quadrature integrates the
    #     full z-tail, equivalent to converge_rmax=True. For legacy
    #     fixed-Rmax behavior we fall back to the per-phase Python path.
    # ------------------------------------------------------------------
    use_mega_kernel = (
        HAS_NUMBA
        and (Rmax_use is None or converge_rmax_use)
    )

    if use_mega_kernel:
        flx_arr, icd_arr, A2_arr, l_arr, L_arr, h_arr, eclipsed_arr = (
            _simulate_phases_numba(
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
        )
        deg_arr = gma_values * (180.0 / np.pi)
        time_arr = deg_arr * 348.42
        phase_arr = (gma_values - (gma0 * np.pi / 180.0)) / (2.0 * np.pi)
        if verbose:
            print(f"Computed {n_iterations} phases via mega-kernel "
                  f"(GL quadrature, parallel over phases)")

        results_list = list(zip(
            flx_arr.tolist(),
            icd_arr.tolist(),
            A2_arr.tolist(),
            gma_values.tolist(),
            deg_arr.tolist(),
            phase_arr.tolist(),
            time_arr.tolist(),
            l_arr.tolist(),
            L_arr.tolist(),
            h_arr.tolist(),
            [bool(x) for x in eclipsed_arr.tolist()],
        ))
        # Skip the per-phase python loop below
        _skip_python_loop = True
    else:
        _skip_python_loop = False

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

        def _integrate(cur_gma_inner):
            av_x, av_th, av_db, A_cells = create_grid(r, l, R, cur_gma_inner, d2h=d2h)
            lw, icd_val, A2_val = wind_los_integral(
                d,
                d1,
                d2,
                cur_gma_inner,
                i,
                av_x,
                av_th,
                av_db,
                A_cells,
                model_id,
                p1,
                p2,
                p3,
                p4,
                dz=dz,
                Rmax=Rmax_use,
                converge_rmax=converge_rmax_use,
            )
            if lw.size > 0:
                flx_val = float(np.sum(lw) / np.sum(A_cells))
                icd_val_sum = float(np.sum(lw))
                A2_val_sum = float(np.sum(A_cells))
            else:
                flx_val = 0.0
                icd_val_sum = 0.0
                A2_val_sum = 0.0
            return flx_val, icd_val_sum, A2_val_sum

        # Only check for eclipse when emitter is BEHIND companion (sin(gma) > 0)
        # When emitter is in front (sin(gma) <= 0), no occultation possible
        if np.sin(cur_gma) > 0:
            if n >= 1:
                flx_i, icd_i, A2_i = _integrate(cur_gma)
            else:
                n2 = a / b
                n3 = l / (R - r)
                if abs(n3) <= 1:
                    is_eclipsed = True
                    flx_i = 0.0
                    icd_i = 0.0
                    A2_i = 0.0
                else:
                    flx_i, icd_i, A2_i = _integrate(cur_gma)
        else:
            flx_i, icd_i, A2_i = _integrate(cur_gma)

        deg_i = cur_gma * 180.0 / np.pi
        time_i = deg_i * 348.42
        phase_i = (cur_gma - (gma0 * np.pi / 180.0)) / (2.0 * np.pi)
        return (
            flx_i,
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

    # Compute phases, optionally in parallel (skipped if mega-kernel was used)
    if not _skip_python_loop:
        if n_jobs == 1:
            results_list = []
            for idx, cur_gma in enumerate(gma_values):
                out = _compute_one_phase(cur_gma)
                results_list.append(out)
                if verbose:
                    print(f"Phase: {out[4]:.2f} degrees")
        else:
            try:
                from joblib import Parallel, delayed

                results_list = Parallel(n_jobs=n_jobs, prefer="processes")(
                    delayed(_compute_one_phase)(float(cur_gma)) for cur_gma in gma_values
                )
                if verbose:
                    for out in results_list:
                        print(f"Phase: {out[4]:.2f} degrees")
            except Exception:
                # Fallback to serial if joblib missing or errors
                results_list = []
                for idx, cur_gma in enumerate(gma_values):
                    out = _compute_one_phase(cur_gma)
                    results_list.append(out)
                    if verbose:
                        print(f"Phase: {out[4]:.2f} degrees")

    # Unpack (now includes is_eclipsed flag)
    flx, icd_vals, A2_vals, ph, deg, phase, time, l3, L3, h3, is_eclipsed = map(
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
            "icd": icd_vals,
            "time": time,
            "l3": l3,
            "L3": L3,
            "h3": h3,
            "is_eclipsed": is_eclipsed,
        }
    )

    # Scale raw wind integrals so that mean(fl) = lam (target mean nH)
    mean_flx = float(np.mean(flx))
    lam_scale = lam / mean_flx if mean_flx > 0 else 1.0
    fl = np.array(flx) * lam_scale
    results["fl"] = fl

    # Calculate scaled fluxes based on method
    if flux_method == "legacy":
        # Legacy hardcoded exponential coefficients (single wind model only)
        results["nfl_hard"] = 9.524 * np.exp(-fl * 0.057)
        results["nfl_soft"] = 9.3923 * np.exp(-fl * 2.5062)

    elif flux_method == "interpolate":
        # Interpolate from CSV data
        if flux_csv_path is None:
            raise ValueError("flux_csv_path required when flux_method='interpolate'")
        df_flux, available_bands = load_flux_vs_nh_csv(flux_csv_path)
        for band in available_bands:
            try:
                results[f"nfl_{band}"] = interpolate_flux_from_nh(
                    fl, df_flux, band, flux_type=flux_type
                )
            except Exception as e:
                warnings.warn(f"Failed to interpolate flux for band '{band}': {e}")

    elif flux_method == "refit":
        # Fit new exponentials to CSV data
        if flux_csv_path is None:
            raise ValueError("flux_csv_path required when flux_method='refit'")
        df_flux, available_bands = load_flux_vs_nh_csv(flux_csv_path)
        for band in available_bands:
            try:
                A, B = fit_exponential_to_csv(df_flux, band, flux_type=flux_type)
                results[f"nfl_{band}"] = A * np.exp(-B * fl)
            except Exception as e:
                warnings.warn(f"Failed to fit exponential for band '{band}': {e}")

    else:
        raise ValueError(
            f"Invalid flux_method: {flux_method}. "
            "Must be 'legacy', 'interpolate', or 'refit'"
        )

    # Set all scaled flux columns to 0 when eclipsed.
    # During eclipse the emitter is physically blocked - flux should be zero,
    # not computed from the absorption formula (which would give max flux at nH=0).
    eclipse_mask = results["is_eclipsed"].values
    if np.any(eclipse_mask):
        flux_cols = [col for col in results.columns if col.startswith("nfl_")]
        for col in flux_cols:
            results.loc[eclipse_mask, col] = 0.0

    return results


# =============================================================================
# Surface number density helpers
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
# The simulation normalizes so that mean(fl) = lam, i.e.
#   mean(N_H) = lam * 1e22  (cm^-2)
# Therefore
#   n_0 = (lam * 1e22) / (R_sun * mean(flx_code))  (cm^-3)
# This n_0 is the companion-surface number density when g(R_star) = 1
# (true for the broken_pl / smooth_pl models at r = Rb, or for beta_law and
# confinement when r just exceeds R_star and the normalization is chosen such
# that g(r_reference) = 1). For a general profile, the surface density is
#   n(R_star) = n_0 * g(R_star; params)
#
# compute_surface_density returns n(R_star) directly so callers need not know
# about g's internal normalization.

R_SUN_CM = 6.957e10  # 1 solar radius in cm


def compute_surface_density(
    sim_df: pd.DataFrame,
    lam: float,
    R_star: float,
    wind_model: str,
    wind_params: Dict[str, float],
) -> float:
    """
    Estimate the wind number density at the companion surface r = R_star.

    Derivation:
      The simulation scales the raw wind integral so that
          mean(N_H) = lam * 1e22 cm^-2.
      The physical LOS column is
          N_H(phi) = n_0 * R_sun * flx_code(phi)
      with flx_code the per-phase mean of the dimensionless LOS integral
      returned by `wind_los_integral`. Taking the orbital mean:
          lam * 1e22 = n_0 * R_sun * <flx_code>
          n_0        = lam * 1e22 / (R_sun * <flx_code>)
      The number density at any radius is n(r) = n_0 * g(r; params),
      so at the companion surface:
          n(R_star) = n_0 * g(R_star; params)

    Args:
        sim_df: DataFrame returned by `simulate_lightcurve` (needs the `flx` column).
        lam: Target mean nH (1e22 cm^-2 units) used in that simulation.
        R_star: Radius at which to evaluate the surface density (solar radii).
        wind_model: Name of the wind profile (same one used in simulation).
        wind_params: Parameter dict (same one used in simulation).

    Returns:
        n(R_star) in cm^-3.
    """
    if "flx" not in sim_df.columns:
        raise KeyError("sim_df must contain a 'flx' column from simulate_lightcurve().")
    flx_mean = float(np.mean(sim_df["flx"].to_numpy()))
    if flx_mean <= 0.0:
        raise ValueError("mean(flx) is non-positive; cannot compute surface density.")
    g_surface = float(evaluate_g_profile(np.array([R_star]), wind_model, wind_params)[0])
    if g_surface <= 0.0:
        raise ValueError(
            f"g(R_star={R_star}) = {g_surface} is non-positive for wind_model="
            f"'{wind_model}'. Surface density is ill-defined at this radius."
        )
    n0 = (float(lam) * 1e22) / (R_SUN_CM * flx_mean)
    return n0 * g_surface


def wind_density_posterior(
    flx_mean,
    lam_samples: np.ndarray,
    R_star_samples,
    wind_model: str,
    wind_params_samples,
) -> Dict[str, object]:
    """
    Posterior estimate of the wind surface number density n(R_star) from MCMC samples.

    Two usage patterns:

    (a) Fixed geometry (most common): run `simulate_lightcurve` once with the
        best-fit geometry, take `flx_mean = mean(sim_df['flx'])`, then feed an
        array of `lam` posterior samples here. `R_star_samples` and
        `wind_params_samples` may be scalars / single dicts.

    (b) Per-sample geometry: pass arrays/lists for each of `flx_mean`,
        `R_star_samples`, `wind_params_samples` (one entry per posterior sample).
        Callers are responsible for re-running `simulate_lightcurve` to build
        those per-sample `flx_mean` values.

    Args:
        flx_mean: Scalar or 1-D array of mean(flx_code) per sample.
        lam_samples: 1-D array of lam posterior samples (1e22 cm^-2 units).
        R_star_samples: Scalar or 1-D array of R_star per sample (solar radii).
        wind_model: Wind profile name.
        wind_params_samples: Either a single dict (reused for all samples) or
            an iterable of per-sample dicts with the same keys.

    Returns:
        Dict with keys:
            - 'samples': per-sample n(R_star) array (cm^-3)
            - 'median', 'p16', 'p84': summary statistics
    """
    lam_arr = np.asarray(lam_samples, dtype=float)
    n_samples = lam_arr.size

    flx_arr = np.asarray(flx_mean, dtype=float)
    if flx_arr.ndim == 0:
        flx_arr = np.full(n_samples, float(flx_arr))
    if flx_arr.size != n_samples:
        raise ValueError(
            f"flx_mean must be scalar or have length {n_samples}, got {flx_arr.size}"
        )

    R_arr = np.asarray(R_star_samples, dtype=float)
    if R_arr.ndim == 0:
        R_arr = np.full(n_samples, float(R_arr))
    if R_arr.size != n_samples:
        raise ValueError(
            f"R_star_samples must be scalar or have length {n_samples}, got {R_arr.size}"
        )

    if isinstance(wind_params_samples, dict):
        params_iter = [wind_params_samples] * n_samples
    else:
        params_iter = list(wind_params_samples)
        if len(params_iter) != n_samples:
            raise ValueError(
                f"wind_params_samples length {len(params_iter)} != n_samples {n_samples}"
            )

    out = np.empty(n_samples, dtype=float)
    for idx in range(n_samples):
        flx_i = flx_arr[idx]
        if flx_i <= 0.0:
            out[idx] = np.nan
            continue
        params_i = params_iter[idx]
        g_surf = float(evaluate_g_profile(np.array([R_arr[idx]]), wind_model, params_i)[0])
        if g_surf <= 0.0:
            out[idx] = np.nan
            continue
        n0 = (lam_arr[idx] * 1e22) / (R_SUN_CM * flx_i)
        out[idx] = n0 * g_surf

    good = np.isfinite(out)
    if not np.any(good):
        return {"samples": out, "median": np.nan, "p16": np.nan, "p84": np.nan}
    q16, q50, q84 = np.percentile(out[good], [16.0, 50.0, 84.0])
    return {
        "samples": out,
        "median": float(q50),
        "p16": float(q16),
        "p84": float(q84),
    }


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
        default=0.5,
        help="Step size along the line of sight (solar radii). Default 0.5 gives "
             "<0.1% truncation error for typical r^-2-like wind profiles with "
             "impact parameter b~d. Use smaller (e.g. 0.1) for very compact winds.",
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
        "--flux_type",
        type=str,
        choices=["erg", "ph"],
        default="erg",
        help="Which flux column from the CSV to use: "
        "'erg' (erg/cm^2/s, default) or 'ph' (photons/cm^2/s). "
        "Only applies when --flux_method is 'interpolate' or 'refit'.",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.589537,
        help="Target mean nH in 1e22 cm^-2 units. The raw wind integral is "
        "scaled so that mean(fl) = lam. Default: 0.589537.",
    )
    parser.add_argument(
        "--wind-model",
        type=str,
        choices=list(WIND_MODEL_IDS.keys()),
        default="smooth_pl",
        help="Dimensionless wind density profile to use. One of: "
        "broken_pl, smooth_pl, beta_law, confinement. Default: smooth_pl.",
    )
    parser.add_argument(
        "--Rb",
        type=float,
        default=5.0,
        help="Break radius (solar radii) for broken_pl / smooth_pl. Default: 5.0.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=4.0,
        help="Inner-region power-law slope for broken_pl / smooth_pl. Default: 4.0.",
    )
    parser.add_argument(
        "--Delta",
        type=float,
        default=2.0,
        help="Smoothness parameter for smooth_pl. Larger -> sharper break. Default: 2.0.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="CAK beta-law exponent for beta_law. Default: 1.0.",
    )
    parser.add_argument(
        "--H",
        type=float,
        default=1.0,
        help="Acceleration scale height (solar radii) for beta_law. Default: 1.0.",
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
        "--output",
        type=str,
        default="xrb_lightcurve_output.csv",
        help="Output file name for results",
    )

    args = parser.parse_args()

    # Default physical cutoff reproduces legacy behavior
    if args.Rmax is None:
        args.Rmax = 2.0 * (args.d1 + args.d2)

    # Validate arguments
    if args.flux_method in ["interpolate", "refit"] and args.flux_csv is None:
        parser.error(f"--flux_csv is required when flux_method='{args.flux_method}'")

    # Collect wind-model parameters
    if args.wind_model == "broken_pl":
        wind_params = {"Rb": args.Rb, "p": args.p}
    elif args.wind_model == "smooth_pl":
        wind_params = {"Rb": args.Rb, "p": args.p, "Delta": args.Delta}
    elif args.wind_model == "beta_law":
        wind_params = {"R_star": args.R, "beta": args.beta, "H": args.H}
    elif args.wind_model == "confinement":
        wind_params = {"R_star": args.R, "fconf": args.fconf, "ell": args.ell}
    else:
        parser.error(f"Unsupported wind_model: {args.wind_model}")

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
        print(f"  flux_type: {args.flux_type}")
    print(f"  lam: {args.lam}")
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
        dz=args.dz,
        verbose=args.verbose,
        n_jobs=args.n_jobs,
        flux_method=args.flux_method,
        flux_csv_path=args.flux_csv,
        flux_type=args.flux_type,
        lam=args.lam,
        Rmax=args.Rmax,
        converge_rmax=args.converge_rmax,
        wind_model=args.wind_model,
        wind_params=wind_params,
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
