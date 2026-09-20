#!/usr/bin/env python3
"""
Forward model for the X-ray light curve of an eclipsing, wind-fed binary.

For every orbital phase the compact object (Star B, an emitting disk of radius
``r``) is decomposed into a polar grid; each visible cell's line of sight is
integrated through the companion's (Star A, radius ``R``) spherically symmetric
wind to obtain a column density, the column is converted to a band flux with an
XSPEC-derived ``flux vs nH`` table, and the per-cell fluxes are area-averaged.

All distances are in solar radii; all angles are converted to radians for the
trigonometric functions.
"""

import argparse
import inspect
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):   # run as a plain script: python cloak/kernel.py
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from cloak.utils import fit_exponential

try:
    from numba import njit, prange
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "numba is required: the Gauss-Legendre kernel is the only LOS "
        "integrator and the per-cell N_H -> flux conversion runs inside it. "
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

# Radial cells of the emitter grid (9 annular segments per angular sector).
N_RADIAL_CELLS = 10

# Cache of flux-vs-nH tables keyed by (abs csv path, flux_type), so an MCMC run
# reads and prepares the CSV once rather than once per likelihood call.
_FLUX_CACHE: Dict[Tuple[str, str], Dict[str, object]] = {}


def pack_wind_params(
    wind_model: str,
    wind_params: Dict[str, float],
) -> Tuple[int, float, float, float]:
    """
    Convert a (wind_model, params dict) pair into a flat tuple
    (model_id, p1, p2, p3) suitable for passing to the numba kernel.

    Keys expected per model are listed in WIND_MODEL_PARAM_KEYS (every profile
    takes exactly three). Missing keys raise ValueError.
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
    p1, p2, p3 = (float(wind_params[k]) for k in keys)
    return model_id, p1, p2, p3


@njit(cache=True, inline="always")
def _g_profile(r, model_id, p1, p2, p3):
    """
    Dimensionless density profile g(r) with r in solar radii.

    model_id encodes the profile; p1..p3 are model-specific parameters
    (see pack_wind_params and WIND_MODEL_PARAM_KEYS).
    """
    if r <= 0.0:
        return 0.0

    if model_id == 0:
        # smooth_pl: Rb=p1, p=p2, Delta=p3
        #   g = x^-2 (1 + x^-Delta)^((p-2)/Delta),   x = r / Rb
        # x^-2 as a division and a single pow for x^-Delta: pow dominates the
        # per-node cost of the kernel, and 3 -> 2 calls is x0.75 on the whole
        # light curve.
        x = r / p1
        inv_x2 = 1.0 / (x * x)
        if p3 <= 0.0:
            return inv_x2
        bracket = 1.0 + x ** (-p3)
        return inv_x2 * bracket ** ((p2 - 2.0) / p3)

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


@njit(cache=True)
def _g_profile_array(r_flat, model_id, p1, p2, p3):
    """Elementwise ``_g_profile`` over a 1-D array (one implementation, no mirror)."""
    out = np.empty(r_flat.shape[0])
    for i in range(r_flat.shape[0]):
        out[i] = _g_profile(r_flat[i], model_id, p1, p2, p3)
    return out


def evaluate_g_profile(r, wind_model: str, wind_params: Dict[str, float]):
    """
    Wind density profile g(r) for arrays or scalars, for helpers and plots.

    Evaluates the same compiled ``_g_profile`` the kernel uses, so there is no
    second implementation to keep in step.
    """
    model_id, p1, p2, p3 = pack_wind_params(wind_model, wind_params)
    r_arr = np.asarray(r, dtype=float)
    flat = np.ascontiguousarray(r_arr.reshape(-1))
    out = _g_profile_array(flat, model_id, p1, p2, p3).reshape(r_arr.shape)
    return float(out) if r_arr.ndim == 0 else out


# =============================================================================
# Numba-accelerated LOS integration kernel
# =============================================================================
#
# `_simulate_phases_numba` computes every orbital phase in one call (prange over
# phases). Per phase it builds the polar emitter grid inline and integrates each
# visible cell's LOS with fixed-node Gauss-Legendre quadrature under the
# substitution u = arctan(z/b), which maps the slowly decaying r^-2 tail onto a
# bounded smooth integrand on a finite interval: 16 nodes per cell give many
# digits of accuracy for any profile and impact parameter, and the full z-tail
# is always integrated (no cutoff radius).
#
# The angular sectors are mirror-symmetric about the line joining the two stars
# (the impact parameter depends on the sector angle only through its cosine), so
# only half of them are integrated and each result is used twice.

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
def _gl_piece(b, u_lo, u_hi, model_id, p1, p2, p3, gl_x, gl_w):
    """Gauss-Legendre estimate of b * int_{u_lo}^{u_hi} g(b / cos u) sec^2(u) du."""
    half_range = 0.5 * (u_hi - u_lo)
    if half_range <= 0.0:
        return 0.0
    mid = 0.5 * (u_hi + u_lo)
    integral = 0.0
    for k in range(gl_x.shape[0]):
        u_k = mid + half_range * gl_x[k]
        cos_uk = math.cos(u_k)
        if cos_uk <= 1e-15:
            continue
        g_val = _g_profile(b / cos_uk, model_id, p1, p2, p3)
        # integrand = g(r) * sec^2(u) * b ; jacobian for [-1,1] -> [u_lo, u_hi] is half_range
        integral += gl_w[k] * g_val / (cos_uk * cos_uk)
    return integral * b * half_range


# beta_law rays closer than this to the photosphere get the split quadrature.
_LIMB_SPLIT_EXCESS = 0.3


@njit(cache=True, inline="always")
def _los_gl_quadrature(b, z_start, model_id, p1, p2, p3, gl_x, gl_w):
    """
    LOS integral int_{-inf}^{z_start} g(r=sqrt(b^2+z^2)) dz via Gauss-Legendre
    quadrature in u = arctan(z/b).

    The substitution gives:
        int g(r) dz = b * int_{-pi/2}^{u_start} g(b/cos u) * sec^2(u) du
    The integrand is bounded and smooth on the finite interval [-pi/2, u_start]
    for any profile that falls at least as fast as r^{-1} at infinity, and 16
    nodes resolve it to <1e-5 for smooth_pl and confinement at any impact
    parameter. The beta_law profile diverges at the photosphere, so a ray
    grazing the limb (b - R_star small) has an integrand peaked at closest
    approach (u = 0) with a width ~ sqrt(2 (b - R_star) / b) that 16 nodes over
    the whole interval under-resolve (-30 % at b - R_star = 0.01, -90 % at
    0.001). Such rays are integrated piecewise: [-pi/2, -w], [-w, 0], [0, w],
    [w, u_start] with the two central pieces halved again, which brings the
    error below 1e-7 everywhere (checked against adaptive quadrature).
    """
    if b < 1e-8:
        b = 1e-8
    u_hi = math.atan(z_start / b)
    u_lo = -1.5707963267948966  # -pi/2
    if u_hi <= u_lo:
        return 0.0
    if model_id == 2 and u_hi > 0.0 and 0.0 < b - p1 < _LIMB_SPLIT_EXCESS:
        w = 4.0 * math.sqrt(2.0 * (b - p1) / b)
        if w > 0.6:
            w = 0.6
        total = 0.0
        # Outer pieces: one GL rule each (empty when the interval is short).
        a = -w if -w > u_lo else u_lo
        total += _gl_piece(b, u_lo, a, model_id, p1, p2, p3, gl_x, gl_w)
        c = w if w < u_hi else u_hi
        total += _gl_piece(b, c, u_hi, model_id, p1, p2, p3, gl_x, gl_w)
        # Central pieces around the peak: two GL rules per side.
        total += _gl_piece(b, a, 0.5 * a, model_id, p1, p2, p3, gl_x, gl_w)
        total += _gl_piece(b, 0.5 * a, 0.0, model_id, p1, p2, p3, gl_x, gl_w)
        total += _gl_piece(b, 0.0, 0.5 * c, model_id, p1, p2, p3, gl_x, gl_w)
        total += _gl_piece(b, 0.5 * c, c, model_id, p1, p2, p3, gl_x, gl_w)
        return total
    return _gl_piece(b, u_lo, u_hi, model_id, p1, p2, p3, gl_x, gl_w)


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
    p1, p2, p3,
    gl_x, gl_w,
):
    """
    Kernel: per-phase geometry, eclipse test and per-cell LOS columns.

    For each phase (parallelized via prange):
      - Compute the projected separation l, its sky-plane components (L, h)
        and the LOS offset z_start of the emitter behind the companion.
      - If the emitter disk lies entirely behind the companion disk, flag the
        phase as eclipsed and skip the grid.
      - Otherwise walk the polar emitter grid. Each radial segment of an
        angular sector is tested for occultation at its centre (the point its
        impact parameter is evaluated at): a visible segment gets one column
        integral (`_los_gl_quadrature`) there and its annular-sector area.
        Testing the centre rather than both bounding radii keeps the visible
        area and mean column of a partially eclipsed extended emitter within
        ~1 % of a Monte Carlo of the disk (dropping every segment that
        straddles the limb was 40-60 % low at r ~ R). Sectors mirrored about
        the star-star line are geometrically identical, so each integral is
        computed once and recorded twice.
      - Reduce to the area-weighted mean column and total visible area, and
        return the per-cell columns/areas the nonlinear nH -> flux conversion
        needs (<F(N)> != F(<N>) wherever the column varies across the disk).

    Returns
    -------
    flx, A2, l, L, h : per-phase arrays
    eclipsed         : uint8 per-phase flag
    cell_col, cell_area : (n_phases, n_cells_max) per-cell columns and areas
    cell_count       : number of valid cells per phase
    """
    n_phases = gma_values.shape[0]
    flx_out = np.zeros(n_phases)
    A2_out = np.zeros(n_phases)
    l_out = np.zeros(n_phases)
    L_out = np.zeros(n_phases)
    h_out = np.zeros(n_phases)
    eclipse_out = np.zeros(n_phases, dtype=np.uint8)

    # n_th equal angular sectors of width 2*pi/n_th (== d2h when it divides
    # 360). Sector i spans [i, i+1) * step; its mask and impact parameter are
    # both evaluated at the sector centre, and the cosine table is made exactly
    # mirror-symmetric so sectors i and n_th-1-i share the same geometry.
    n_th = int(360.0 / d2h_deg)
    if n_th < 2:
        n_th = 2
    n_half = (n_th + 1) // 2
    th_step = 2.0 * math.pi / n_th
    cos_c = np.empty(n_th)
    for i_th in range(n_th):
        cos_c[i_th] = math.cos((i_th + 0.5) * th_step)
    for i_th in range(n_th // 2):
        cos_c[n_th - 1 - i_th] = cos_c[i_th]

    # Radial grid of the emitter disk and the segment tables derived from it.
    n_r = N_RADIAL_CELLS
    r_min = r / 10.0
    r_step = (r - r_min) / (n_r - 1)
    r_vals = np.empty(n_r)
    av_x_tab = np.zeros(n_r)   # centre radius of segment (i_r-1, i_r)
    A_seg_tab = np.zeros(n_r)  # annular-sector area of that segment
    for i_r in range(n_r):
        r_vals[i_r] = r_min + i_r * r_step
    for i_r in range(1, n_r):
        av_x_tab[i_r] = 0.5 * (r_vals[i_r - 1] + r_vals[i_r])
        A_seg_tab[i_r] = 0.5 * th_step * (
            r_vals[i_r] * r_vals[i_r] - r_vals[i_r - 1] * r_vals[i_r - 1]
        )

    n_cells_max = n_th * n_r
    cell_col_out = np.zeros((n_phases, n_cells_max))
    cell_area_out = np.zeros((n_phases, n_cells_max))
    cell_count_out = np.zeros(n_phases, dtype=np.int64)

    sin_i = math.sin(incl)
    cos_i = math.cos(incl)
    R2 = R * R
    a = d1 + d2

    for ip in prange(n_phases):
        cur_gma = gma_values[ip]
        sin_g = math.sin(cur_gma)
        cos_g = math.cos(cur_gma)

        # Geometry depends on the separation a = d1 + d2 alone.
        h = a * sin_g * sin_i
        L = a * cos_g
        l = math.sqrt(h * h + L * L)
        z_start = a * sin_g * cos_i

        # Total eclipse (only when the emitter is BEHIND the companion: sin_g > 0)
        is_eclipsed_phase = False
        if sin_g > 0.0 and l < (R + r):
            if (R - r) > 0.0:
                if l <= (R - r):
                    is_eclipsed_phase = True
            else:
                is_eclipsed_phase = True

        l_out[ip] = l
        L_out[ip] = L
        h_out[ip] = h

        if is_eclipsed_phase:
            eclipse_out[ip] = 1
            continue

        sum_lw = 0.0
        sum_A = 0.0
        k_cell = 0
        for i_th in range(n_half):
            cos_th = cos_c[i_th]
            reps = 2 if (n_th - 1 - i_th) != i_th else 1
            for i_r in range(1, n_r):
                # Impact parameter of the segment centre; the same point decides
                # whether the segment is hidden behind the companion disk (only
                # possible when the emitter is behind it: sin_g > 0).
                av_x = av_x_tab[i_r]
                bv2 = av_x * av_x + l * l - 2.0 * av_x * l * cos_th
                if sin_g > 0.0 and bv2 < R2:
                    continue
                bv = math.sqrt(bv2) if bv2 > 0.0 else 0.0
                A_seg = A_seg_tab[i_r]
                los_val = _los_gl_quadrature(
                    bv, z_start, model_id, p1, p2, p3, gl_x, gl_w
                )
                for _rep in range(reps):
                    sum_lw += los_val * A_seg
                    sum_A += A_seg
                    cell_col_out[ip, k_cell] = los_val
                    cell_area_out[ip, k_cell] = A_seg
                    k_cell += 1

        cell_count_out[ip] = k_cell
        if sum_A > 0.0:
            flx_out[ip] = sum_lw / sum_A
        A2_out[ip] = sum_A

    return (
        flx_out, A2_out, l_out, L_out, h_out, eclipse_out,
        cell_col_out, cell_area_out, cell_count_out,
    )


# =============================================================================
# Per-cell N_H -> flux conversion
# =============================================================================
#
# The band flux is <F(N)> over the visible emitter disk, NOT F(<N>): the
# attenuation law is convex, so during ingress/egress and in the eclipse core
# (where the column varies by orders of magnitude across the disk) the
# surviving flux is carried by the least-absorbed cells. Both converters below
# apply the mapping per cell and then area-average, in one compiled pass.

@njit(cache=True, parallel=True)
def _cell_flux_loglog(cell_col, cell_area, cell_count, col_scale, log_nh, log_flux):
    """Area-averaged flux per phase from a log-log table (linear in log space).

    Reproduces ``scipy.interpolate.interp1d(kind='linear',
    fill_value='extrapolate')`` on ``(log10 nH, log10 flux)``: linear
    extrapolation from the upper end segment, with the column clipped to
    [1e-6, 1e6] x 1e22 cm^-2 first. Below the table the flux is held at the
    first tabulated value: absorption cannot exceed 1, and extrapolating the
    first segment in log-log returned up to 28 % more than the table's own
    low-nH plateau for tables starting at 0.01 x 1e22 cm^-2.
    """
    n_phases = cell_col.shape[0]
    n = log_nh.shape[0]
    out = np.zeros(n_phases)
    for ip in prange(n_phases):
        num = 0.0
        den = 0.0
        for k in range(cell_count[ip]):
            N = cell_col[ip, k] * col_scale
            if N < 1e-6:
                N = 1e-6
            elif N > 1e6:
                N = 1e6
            lx = math.log10(N)
            if lx <= log_nh[0]:
                F = 10.0 ** log_flux[0]
            else:
                if lx >= log_nh[n - 1]:
                    j = n - 2
                else:
                    lo = 0
                    hi = n - 1
                    while hi - lo > 1:
                        mid = (lo + hi) >> 1
                        if log_nh[mid] <= lx:
                            lo = mid
                        else:
                            hi = mid
                    j = lo
                t = (lx - log_nh[j]) / (log_nh[j + 1] - log_nh[j])
                F = 10.0 ** (log_flux[j] + t * (log_flux[j + 1] - log_flux[j]))
            A = cell_area[ip, k]
            num += F * A
            den += A
        out[ip] = num / den if den > 0.0 else 0.0
    return out


@njit(cache=True, parallel=True)
def _cell_flux_exp(cell_col, cell_area, cell_count, col_scale, A_coef, B_coef):
    """Area-averaged flux per phase for the analytic law F = A exp(-B N)."""
    n_phases = cell_col.shape[0]
    out = np.zeros(n_phases)
    for ip in prange(n_phases):
        num = 0.0
        den = 0.0
        for k in range(cell_count[ip]):
            N = cell_col[ip, k] * col_scale
            A = cell_area[ip, k]
            num += A_coef * math.exp(-B_coef * N) * A
            den += A
        out[ip] = num / den if den > 0.0 else 0.0
    return out


# =============================================================================
# Flux-vs-nH table (cloak/flux_table.py output)
# =============================================================================

def get_available_bands_from_csv(df: pd.DataFrame, flux_type: str = "erg") -> list:
    """Band names present as ``flux_{band}_{flux_type}`` columns, sorted."""
    suffix = f"_{flux_type}"
    return sorted(
        col[len("flux_"):-len(suffix)]
        for col in df.columns
        if col.startswith("flux_") and col.endswith(suffix)
    )


def load_flux_vs_nh_csv(csv_path: str, flux_type: str = "erg") -> Tuple[pd.DataFrame, list]:
    """
    Load a flux vs nH CSV from cloak/flux_table.py.

    Returns ``(df, bands)`` with rows restricted to a valid, positive
    ``nH_1e22`` and at least one finite flux value among the detected bands.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Flux vs nH CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "nH_1e22" not in df.columns:
        raise ValueError("CSV missing required column: nH_1e22")

    bands = get_available_bands_from_csv(df, flux_type)
    if not bands:
        raise ValueError(
            f"No flux columns found in CSV for flux_type='{flux_type}'. "
            f"Expected columns like flux_{{band}}_{flux_type}"
        )
    nh = pd.to_numeric(df["nH_1e22"], errors="coerce")
    keep = nh.notna() & (nh > 0)
    any_flux = np.zeros(len(df), dtype=bool)
    for band in bands:
        any_flux |= pd.to_numeric(df[f"flux_{band}_{flux_type}"], errors="coerce").notna().to_numpy()
    # Keep the numeric column, not the raw one: a stray non-numeric token makes
    # the whole column object dtype, which would later sort lexicographically.
    df = df.assign(nH_1e22=nh)[keep.to_numpy() & any_flux]
    if len(df) == 0:
        raise ValueError("No valid data points in CSV after filtering")
    return df, bands


def _build_flux_context(csv_path: str, flux_type: str) -> Dict[str, object]:
    """Build (once) the per-band interpolation and refit data for a CSV."""
    key = (os.path.abspath(csv_path), str(flux_type))
    cached = _FLUX_CACHE.get(key)
    if cached is not None:
        return cached

    df, bands = load_flux_vs_nh_csv(csv_path, flux_type=flux_type)
    df_sorted = df.sort_values("nH_1e22")
    nh_base = pd.to_numeric(df_sorted["nH_1e22"], errors="coerce").to_numpy(dtype=float)

    band_data: Dict[str, Dict[str, object]] = {}
    for band in bands:
        flux_vals = pd.to_numeric(
            df_sorted[f"flux_{band}_{flux_type}"], errors="coerce"
        ).to_numpy(dtype=float)
        valid = np.isfinite(nh_base) & (nh_base > 0) & np.isfinite(flux_vals) & (flux_vals > 0)
        if np.count_nonzero(valid) < 2:
            raise ValueError(
                f"{csv_path}: band '{band}' has {int(np.count_nonzero(valid))} usable rows "
                f"(finite nH > 0 and flux > 0); the interpolation needs at least two.")
        nh_csv = nh_base[valid]
        flux_csv = flux_vals[valid]
        if np.any(np.diff(nh_csv) <= 0):
            raise ValueError(
                f"{csv_path}: nH values for band '{band}' must be unique (repeated rows break "
                f"the log-log interpolation); found {int(np.sum(np.diff(nh_csv) <= 0))} repeats.")
        band_data[band] = {
            "nh": nh_csv,
            "flux": flux_csv,
            "log_nh": np.ascontiguousarray(np.log10(nh_csv)),
            "log_flux": np.ascontiguousarray(np.log10(flux_csv)),
            "exp_fit": None,  # (A, B), fitted on first 'refit' use
        }

    ctx: Dict[str, object] = {
        "csv_path": key[0],
        "flux_type": key[1],
        "bands": sorted(band_data),
        "band_data": band_data,
    }
    _FLUX_CACHE[key] = ctx
    return ctx


def _select_band(ctx: Dict[str, object], band: Optional[str]) -> str:
    """Resolve the band to simulate: the requested one, or the CSV's only one."""
    bands = ctx["bands"]  # type: ignore[index]
    if band is None:
        if len(bands) == 1:
            return bands[0]
        raise ValueError(
            f"The flux table {ctx['csv_path']} contains {len(bands)} bands "
            f"{bands}; pass band=... to choose one (the model is run one band "
            f"at a time)."
        )
    if band not in bands:
        raise ValueError(
            f"Band '{band}' not in flux table {ctx['csv_path']} "
            f"(flux_type='{ctx['flux_type']}'). Available: {bands}"
        )
    return band


def flux_table_bands(csv_path: str, flux_type: str = "erg") -> list:
    """Bands available in a flux-vs-nH table (loads and caches the table)."""
    return list(_build_flux_context(csv_path, flux_type=flux_type)["bands"])


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
    """Convert a conventional inclination into the kernel's internal angle.

    ``i0`` at the public API is the standard astronomical inclination: the angle
    between the orbital-plane normal and the line of sight, so ``i0 = 90 deg``
    is edge-on (eclipses possible) and ``i0 = 0 deg`` is face-on (the orbit lies
    in the plane of the sky and never eclipses).

    The geometry kernel (``_simulate_phases_numba``) instead measures ``incl``
    from the *line of sight*, so that ``h = a sin(gma) sin(incl)`` is the
    sky-plane offset and ``z = a sin(gma) cos(incl)`` the offset along the line
    of sight. The two differ by the 90 deg complement applied here.
    """
    return (90.0 - float(i0_deg)) * np.pi / 180.0


# =============================================================================
# Simulation
# =============================================================================

def _mirror_indices(gma0_deg: float, dth_deg: float, n_phases: int):
    """Phase indices to compute, and every index's reflection partner.

    The kernel sees the orbital phase only through ``sin(gma)`` and
    ``|cos(gma)|`` (``l``, ``z_start`` and the occultation test), so ``gma`` and
    ``pi - gma`` give identical columns and an ``L`` of opposite sign. On the
    uniform grid ``gma_k = gma0 + k dth`` that reflection maps index ``k`` to
    ``(m - k) mod n`` with ``m = (180 - 2 gma0) / dth``. When ``m`` is an
    integer -- the default ``gma0 = -90`` with any ``dth`` dividing 360 -- only
    one member of each pair is run through the kernel and the other is copied,
    which halves the kernel and flux-conversion work. The two halves of a full
    computation already agree only to trig round-off (~1e-15), so the copy is
    exact.

    Returns ``(run, partner)``: the indices to compute and, for every index,
    its partner; ``partner`` is None when the grid has no such symmetry.
    """
    m = (180.0 - 2.0 * float(gma0_deg)) / float(dth_deg)
    if abs(m - round(m)) > 1e-9:
        return np.arange(n_phases), None
    idx = np.arange(n_phases)
    partner = (int(round(m)) - idx) % n_phases
    return idx[idx <= partner], partner


def _unmirror(values: np.ndarray, run: np.ndarray, partner, negate: bool = False) -> np.ndarray:
    """Scatter per-phase results of the computed indices onto the full grid."""
    if partner is None:
        return values
    full = np.empty(partner.shape[0], dtype=values.dtype)
    full[partner[run]] = -values if negate else values
    full[run] = values
    return full


def _simulate_core(
    *,
    r: float = 0.001,
    R: float = 2.0,
    d1: float = 11.0,
    d2: float = 8.0,
    gma0: float = -90.0,
    i0: float = 64.0,
    dth: float = 1.0,
    d2h: float = 6.0,
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
    band: Optional[str] = None,
) -> Dict[str, object]:
    """Shared implementation of simulate_lightcurve / simulate_band_flux.

    This signature is the single definition of the simulation defaults
    (exported as ``SIM_DEFAULTS`` for the CLIs); keyword-only, so a misspelled
    argument raises ``TypeError`` instead of silently running the default.
    """
    if flux_csv_path is None:
        raise ValueError("flux_csv_path is required (table from cloak/flux_table.py)")
    if flux_method not in ("interpolate", "refit"):
        raise ValueError(
            f"Invalid flux_method: {flux_method}. Must be 'interpolate' or 'refit'"
        )
    if not (0.0 < r < R):
        raise ValueError(
            f"Need 0 < r < R (got r={r}, R={R}): the eclipse test assumes the emitter disk "
            f"is smaller than the companion.")
    if d1 + d2 <= 0.0:
        raise ValueError(f"Need d1 + d2 > 0 (got d1={d1}, d2={d2}).")
    if not (0.0 <= i0 <= 180.0):
        # The kernel decides "emitter behind the companion" from sin(gma) alone,
        # which assumes cos(incl) >= 0; a negative i0 would invert that test.
        raise ValueError(f"i0 must lie in [0, 180] degrees (got {i0}).")
    if mdot <= 0.0 or v_inf <= 0.0 or f_opacity < 0.0:
        raise ValueError(f"Need mdot > 0, v_inf > 0 and f_opacity >= 0 "
                         f"(got mdot={mdot}, v_inf={v_inf}, f_opacity={f_opacity}).")
    for name, step in (("dth", dth), ("d2h", d2h)):
        # The phase grid and the sector grid must close: int(360 / step) rings
        # would otherwise leave a gap, and the phase reflection assumes a
        # closed grid. At most two phases / two sectors (step 180).
        if not (0.0 < step <= 180.0) or abs(360.0 / step - round(360.0 / step)) > 1e-9:
            raise ValueError(f"{name} must lie in (0, 180] and divide 360 evenly (got {step}).")

    # Only the input convention changes here: `incl` is the internal angle from
    # the line of sight that the kernel's geometry assumes.
    incl = inclination_to_internal_rad(i0)

    if wind_params is None:
        wind_params = default_wind_params(wind_model, R)
    # Profiles anchored at the photosphere take R_star from R when omitted.
    if wind_model in R_STAR_TIED_MODELS and "R_star" not in wind_params:
        wind_params = dict(wind_params)
        wind_params["R_star"] = float(R)
    model_id, p1, p2, p3 = pack_wind_params(wind_model, wind_params)

    gma0_rad = gma0 * np.pi / 180.0
    n_phases = int(round(360.0 / dth))     # dth divides 360 (checked above)
    gma_values = gma0_rad + np.arange(n_phases) * (dth * np.pi / 180.0)

    # Phases gma and pi - gma are geometrically identical (see
    # _mirror_indices): only one of each pair goes through the kernel.
    run, partner = _mirror_indices(gma0, dth, n_phases)

    (flx, A2, l_arr, L_arr, h_arr, eclipsed,
     cell_col, cell_area, cell_count) = _simulate_phases_numba(
        np.ascontiguousarray(gma_values[run], dtype=np.float64),
        float(r), float(R), float(d1), float(d2), float(incl), float(d2h),
        int(model_id), float(p1), float(p2), float(p3),
        _GL16_X, _GL16_W,
    )

    # Column-density normalization: n_0 from Mdot / v_inf, so fl carries real
    # units (1e22 cm^-2); flx is the LOS integral of g in R_sun. f_opacity is an
    # effective-opacity factor absorbing wind ionization, clumping and abundance
    # departures from the solar-abundance TBabs table.
    n0 = wind_density_norm_from_mdot(mdot, v_inf, wind_model, wind_params, mu=mu_wind)
    col_scale = float(f_opacity) * n0 * R_SUN_CM / 1.0e22

    ctx = _build_flux_context(flux_csv_path, flux_type=flux_type)
    band = _select_band(ctx, band)
    info = ctx["band_data"][band]  # type: ignore[index]

    if flux_method == "interpolate":
        nfl = _cell_flux_loglog(
            cell_col, cell_area, cell_count, col_scale,
            info["log_nh"], info["log_flux"],
        )
    else:
        if info["exp_fit"] is None:
            info["exp_fit"] = fit_exponential(info["nh"], info["flux"])
        A_coef, B_coef = info["exp_fit"]
        nfl = _cell_flux_exp(cell_col, cell_area, cell_count, col_scale, A_coef, B_coef)

    # Copy the computed phases onto their reflection partners (L flips sign).
    flx, A2, l_arr, h_arr, eclipsed, nfl = (
        _unmirror(x, run, partner) for x in (flx, A2, l_arr, h_arr, eclipsed, nfl))
    L_arr = _unmirror(L_arr, run, partner, negate=True)

    # Eclipsed phases have no visible cells, so nfl is already 0 there; the
    # scattered-light floor is a constant, phase-independent addition.
    if float(scattered_flux) != 0.0:
        nfl = nfl + float(scattered_flux)

    # Column order of the DataFrame simulate_lightcurve builds from this.
    return {
        "deg": gma_values * (180.0 / np.pi),
        "phase": (gma_values - gma0_rad) / (2.0 * np.pi),
        "l3": l_arr,
        "L3": L_arr,
        "h3": h_arr,
        "A2": A2,
        "is_eclipsed": eclipsed.astype(bool),
        "flx": flx,
        "fl": flx * col_scale,
        f"nfl_{band}": nfl,
        "band": band,
        "n_computed": int(run.size),
    }


# Simulation defaults, taken from the one place they are defined.
SIM_DEFAULTS: Dict[str, object] = {
    name: param.default for name, param in inspect.signature(_simulate_core).parameters.items()
}


def simulate_lightcurve(verbose: bool = False, **kwargs) -> pd.DataFrame:
    """
    Simulate one orbit and return a per-phase DataFrame.

    All simulation arguments are keywords with the defaults of ``SIM_DEFAULTS``
    (an unknown keyword raises ``TypeError``):
        r: Radius of smaller star B (compact object) in solar radii
        R: Radius of larger star A (companion) in solar radii
        d1: Distance of star B from COM in solar radii
        d2: Distance of star A from COM in solar radii
        gma0: Starting phase angle in degrees
        i0: Orbital inclination in degrees, standard astronomical convention:
            measured from the orbital-plane normal, so 90 deg is edge-on and
            0 deg is face-on.
        dth: Orbital increment in degrees
        d2h: Angular cell size (degrees) of the polar emitter grid
        flux_method: nH -> flux conversion: "interpolate" (log-log
            interpolation of the CSV table, default) or "refit" (analytic
            A*exp(-B*nH) fitted to the same table)
        flux_csv_path: Path to a CSV from cloak/flux_table.py (required)
        flux_type: Which flux column to use — "erg" (erg/cm^2/s, default) or
            "ph" (photons/cm^2/s).
        wind_model: Dimensionless wind density profile, one of "smooth_pl",
            "confinement" or "beta_law". Default "smooth_pl".
        wind_params: Dict of profile parameters (see WIND_MODEL_PARAM_KEYS).
            If None, uses defaults from default_wind_params(wind_model, R).
        scattered_flux: Constant additive flux offset applied to the band flux
            after eclipse handling (phase-invariant scattered-light floor).
        mdot: Mass-loss rate in Msun/yr, setting the absolute wind density.
        v_inf: Wind terminal velocity in km/s.
        mu_wind: Mean mass per hydrogen-equivalent nucleus, converting the wind
            mass column into the N_H the solar-abundance TBabs table expects.
        f_opacity: Effective-opacity factor applied to the Mdot-derived column,
            absorbing wind photoionization, clumping and abundance departures.
        band: Energy band to simulate. May be omitted when the CSV holds a
            single band (the model is run one band at a time).
    ``verbose`` prints a one-line summary of the kernel call.

    Returns:
        DataFrame with one row per phase and columns
            deg, phase        orbital phase in degrees / [0, 1)
            l3, L3, h3        projected separation and its sky-plane components
            A2                visible emitter area (grid units)
            is_eclipsed       geometric total eclipse flag
            flx               dimensionless mean LOS integral of g
            fl                absolute mean column density N_H in 1e22 cm^-2
            nfl_{band}        band flux, area-averaged over the emitter disk

    Notes:
        - The density normalization n_0 is fixed from mdot / v_inf, so ``fl``
          carries real units and the eclipse emerges from wind opacity rather
          than from a geometric cutoff; ``R`` is the true photosphere.
        - The nH -> flux conversion is applied per emitter cell and only then
          area-averaged, because <F(N)> != F(<N>) when the column varies across
          the disk (ingress/egress, eclipse core).
    """
    res = _simulate_core(**kwargs)
    if verbose:
        print(f"Computed {res['n_computed']} of {res['deg'].size} phases via the GL "
              f"kernel (parallel over phases; the rest by phase reflection); "
              f"band '{res['band']}'")
    return pd.DataFrame({k: v for k, v in res.items() if k not in ("band", "n_computed")})


def simulate_band_flux(**kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """``(phase, band_flux)`` arrays for the same keywords as simulate_lightcurve.

    The lightweight entry point for likelihood evaluation: no DataFrame is
    built and only the two arrays the fit needs are returned.
    """
    res = _simulate_core(**kwargs)
    return res["phase"], res[f"nfl_{res['band']}"]


# =============================================================================
# Wind density normalization
# =============================================================================
#
# Units note: the kernel returns the dimensionless integral
#   flx_code = <∫ g(r) dz>_cells
# with r and z in solar radii (R_sun = 6.957e10 cm). The physical column at
# phase phi is N_H(phi) = n_0 * R_sun * flx_code(phi), where n_0 is the
# reference number density such that n(r) = n_0 * g(r).

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
    """
    mdot_cgs = float(mdot_msun_yr) * M_SUN_G / YEAR_S
    v_cgs = float(v_inf_kms) * KM_TO_CM
    C = wind_asymptotic_coefficient(wind_model, wind_params)
    denom = 4.0 * np.pi * (R_SUN_CM ** 2) * v_cgs * float(mu) * M_H_G * C
    if denom <= 0.0:
        raise ValueError("Non-positive denominator in wind density normalization.")
    return mdot_cgs / denom


# =============================================================================
# CLI
# =============================================================================

def main():
    """Simulate one light curve from the command line and write it to CSV."""
    parser = argparse.ArgumentParser(
        description="Simulate the wind-absorbed, eclipsed light curve of a binary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Every default comes from _simulate_core (SIM_DEFAULTS) or from
    # default_wind_params, so the CLI, the Python API and the MCMC agree.
    D = SIM_DEFAULTS
    spl = default_wind_params("smooth_pl", D["R"])
    conf = default_wind_params("confinement", D["R"])
    beta = default_wind_params("beta_law", D["R"])
    parser.add_argument("--r", type=float, default=D["r"],
                        help="Radius of star B (compact object / disk) in solar radii")
    parser.add_argument("--R", type=float, default=D["R"],
                        help="Radius of star A (companion) in solar radii")
    parser.add_argument("--d1", type=float, default=D["d1"],
                        help="Distance of star B from COM in solar radii")
    parser.add_argument("--d2", type=float, default=D["d2"],
                        help="Distance of star A from COM in solar radii")
    parser.add_argument("--gma0", type=float, default=D["gma0"],
                        help="Starting phase angle in degrees")
    parser.add_argument("--i0", type=float, default=D["i0"],
                        help="Orbital inclination in degrees from the orbital-plane "
                             "normal (90 = edge-on, 0 = face-on)")
    parser.add_argument("--dth", type=float, default=D["dth"],
                        help="Orbital increment in degrees")
    parser.add_argument("--d2h", type=float, default=D["d2h"],
                        help="Angular cell size (degrees) of the polar emitter grid")
    parser.add_argument("--verbose", action="store_true",
                        help="Print a one-line kernel summary")
    parser.add_argument("--flux_method", type=str, choices=["interpolate", "refit"],
                        default=D["flux_method"],
                        help="nH -> flux conversion: log-log interpolation of the CSV "
                             "table, or an exponential refit to it")
    parser.add_argument("--flux_csv", type=str, required=True,
                        help="Flux vs nH CSV from cloak/flux_table.py")
    parser.add_argument("--flux_type", type=str, choices=["erg", "ph"], default=D["flux_type"],
                        help="Flux column to use: erg (erg/cm^2/s) or ph (photons/cm^2/s)")
    parser.add_argument("--band", type=str, default=None,
                        help="Energy band to simulate; optional when the CSV holds one band")
    parser.add_argument("--mdot", type=float, default=D["mdot"],
                        help="Mass-loss rate in Msun/yr, setting the absolute wind density "
                             "(default: Clark & Crowther 2004, clumping-corrected)")
    parser.add_argument("--v-inf", type=float, default=D["v_inf"],
                        help="Wind terminal velocity in km/s")
    parser.add_argument("--mu-wind", type=float, default=D["mu_wind"],
                        help="Mean mass per hydrogen-equivalent nucleus")
    parser.add_argument("--f-opacity", type=float, default=D["f_opacity"],
                        help="Effective-opacity factor on the Mdot-derived column "
                             "(ionization, clumping, abundances); ~0.01-0.03 for IC 10 X-1")
    parser.add_argument("--wind-model", type=str, choices=list(WIND_MODEL_IDS),
                        default=D["wind_model"], help="Dimensionless wind density profile")
    parser.add_argument("--Rb", type=float, default=spl["Rb"],
                        help="Break radius (solar radii) for smooth_pl")
    parser.add_argument("--p", type=float, default=spl["p"],
                        help="Inner-region power-law slope for smooth_pl")
    parser.add_argument("--Delta", type=float, default=spl["Delta"],
                        help="Break sharpness for smooth_pl (larger = sharper; the MCMC "
                             "holds it at this value)")
    parser.add_argument("--fconf", type=float, default=conf["fconf"],
                        help="Overdensity amplitude for confinement")
    parser.add_argument("--ell", type=float, default=conf["ell"],
                        help="Compression scale length (solar radii) for confinement")
    parser.add_argument("--beta", type=float, default=beta["beta"],
                        help="CAK velocity-law exponent for beta_law")
    parser.add_argument("--H", type=float, default=beta["H"],
                        help="Inner acceleration scale height (solar radii) for beta_law; "
                             "the effective break radius is R + 3H")
    parser.add_argument("--output", type=str, default="xrb_lightcurve_output.csv",
                        help="Output CSV file")

    args = parser.parse_args()

    if args.wind_model == "smooth_pl":
        wind_params = {"Rb": args.Rb, "p": args.p, "Delta": args.Delta}
    elif args.wind_model == "confinement":
        wind_params = {"R_star": args.R, "fconf": args.fconf, "ell": args.ell}
    else:
        wind_params = {"R_star": args.R, "beta": args.beta, "H": args.H}

    print("Starting XRB light-curve simulation with parameters:")
    for name in ("r", "R", "d1", "d2", "gma0", "i0", "dth", "d2h", "flux_method",
                 "flux_csv", "flux_type", "band", "mdot", "v_inf", "mu_wind",
                 "f_opacity", "wind_model"):
        print(f"  {name}: {getattr(args, name)}")
    print(f"  wind_params: {wind_params}")
    print(f"  output: {args.output}\n")

    try:
        results = simulate_lightcurve(
            r=args.r, R=args.R, d1=args.d1, d2=args.d2, gma0=args.gma0, i0=args.i0,
            dth=args.dth, d2h=args.d2h, verbose=args.verbose,
            flux_method=args.flux_method, flux_csv_path=args.flux_csv,
            flux_type=args.flux_type, band=args.band,
            wind_model=args.wind_model, wind_params=wind_params,
            mdot=args.mdot, v_inf=args.v_inf, mu_wind=args.mu_wind,
            f_opacity=args.f_opacity,
        )
    except (FileNotFoundError, ValueError, KeyError) as e:
        raise SystemExit(f"ERROR: {e}")
    results.to_csv(args.output, index=False)
    print(f"Simulation completed: {len(results)} phases written to {args.output}")
    return results


if __name__ == "__main__":
    main()
