#!/usr/bin/env python3
"""
Plotting helpers for the XRB light-curve codebase.
--------------------------------------------------
:func:`plot_lightcurve_fit` is the **single** light-curve drawing routine. It is
the one that used to live in ``mcmc_lightcurve_fit.plot_best_fit``, generalized
so every caller can use it:

* ``mcmc_lightcurve_fit.plot_best_fit`` resolves the posterior point estimate,
  evaluates the physical model and the per-sample phase shift, then hands the
  resulting arrays here.
* ``chandra_phase_analysis`` (via the :func:`plot_phase` wrapper below)
  interpolates a tabulated simulation light curve and hands the arrays here.

Because both routes end in the same function, the observed data, the model
overlay, the smoothed curve, the residual panel and the χ²/dof in the title are
drawn by one implementation and cannot drift apart.

Plot titles show only the **energy band** and the **χ²/dof** value; per-parameter
values are intentionally not annotated on the figure (the legend identifies the
curves, and the numbers live in the printed results table / summary file).

Dependencies: numpy, pandas, matplotlib; corner (optional, for corner plots).
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.utils import (
    band_label_from_column,
    detect_energy_bands,
    get_band_display_name,
    model_from_wrap,
    obs_errors,
    prepare_model_interpolator,
)

try:
    import corner
except ImportError:  # pragma: no cover - optional dependency
    warnings.warn(
        "corner not installed. Corner plots will be disabled. "
        "Install with: pip install corner"
    )
    corner = None


# -----------------------------------------------------------------------------
# Shared pieces
# -----------------------------------------------------------------------------

def format_reduced_chi2(value: float) -> Optional[str]:
    """Format a reduced χ² for a plot title, or return None if unusable."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return f"{v:.2e}" if abs(v) < 1e-2 else f"{v:.3f}"


def build_fit_title(band: Optional[str] = None, red_chi2: Optional[float] = None) -> str:
    """Compose the plot title from the energy band and χ²/dof only.

    Deliberately minimal: best-fit parameter values are reported in the printed
    results table and the summary file, not on the figure.
    """
    bits: List[str] = []
    if band:
        bits.append(f"{band} band")
    chi2_text = None if red_chi2 is None else format_reduced_chi2(red_chi2)
    if chi2_text is not None:
        bits.append(f"$\\chi^2$/dof = {chi2_text}")
    return "  —  ".join(bits)


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


def half_widths(phase_width: Optional[np.ndarray], shape: tuple) -> Optional[np.ndarray]:
    """Convert bin widths to matplotlib ``xerr`` half-widths, or None."""
    if phase_width is None:
        return None
    width = np.asarray(phase_width, dtype=float)
    if width.shape != shape:
        return None
    return 0.5 * np.clip(width, 0.0, np.inf)


# -----------------------------------------------------------------------------
# The single light-curve plotting function
# -----------------------------------------------------------------------------

def plot_lightcurve_fit(
    obs_phase: np.ndarray,
    obs_flux: np.ndarray,
    obs_err: Optional[np.ndarray] = None,
    *,
    model_phase: Optional[np.ndarray] = None,
    model_flux: Optional[np.ndarray] = None,
    obs_model: Optional[np.ndarray] = None,
    obs_phase_width: Optional[np.ndarray] = None,
    obs_group: Optional[Sequence] = None,
    band: Optional[str] = None,
    red_chi2: Optional[float] = None,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    ax_res: Optional[plt.Axes] = None,
    is_binned: bool = True,
    obs_label: Optional[str] = None,
    obs_color: Optional[str] = None,
    model_label: str = 'Best-fit model',
    smooth_phase: Optional[np.ndarray] = None,
    smooth_flux: Optional[np.ndarray] = None,
    smooth_flux_err: Optional[np.ndarray] = None,
    smooth_sigma: Optional[float] = None,
    ylabel: str = 'Flux (erg/cm²/s)',
    xlabel: str = 'Orbital phase',
    residual_ylim: Optional[tuple[float, float]] = (-5.0, 5.0),
    figsize: Optional[tuple[float, float]] = None,
    dpi: int = 150,
    legend_loc: str = 'best',
    legend_fontsize: Optional[str] = None,
    show: bool = False,
    verbose: bool = True,
) -> Optional[plt.Figure]:
    """Draw an observed light curve with an optional model overlay and residuals.

    All model evaluation is done by the caller: this function only draws what it
    is given, which is what makes it usable from both the MCMC path (physical
    model + posterior point estimate) and the tabulated-simulation path.

    Parameters
    ----------
    obs_phase, obs_flux : array
        Observed orbital phase and flux/count rate.
    obs_err : array, optional
        Observed 1σ uncertainties. Required for error bars and for the residual
        panel.
    model_phase, model_flux : array, optional
        Model curve to overlay, already phase-shifted. Sorted internally, so an
        overlay wrapped through phase 1 may be passed unsorted.
    obs_model : array, optional
        Model evaluated *at the observed phases*, used for the residual panel.
        Must correspond to the same shift/scatter as *model_flux*.
    obs_phase_width : array, optional
        Bin widths in phase (adaptive binning); drawn as horizontal error bars.
    obs_group : sequence, optional
        Per-point group label (e.g. observation id). When given, each group is
        drawn as its own series with its own legend entry.
    band : str, optional
        Energy-band label for the title (e.g. ``"BROAD"``).
    red_chi2 : float, optional
        Reduced χ² for the title.
    title : str, optional
        Explicit title, overriding the ``band`` / ``red_chi2`` composition.
    output_path : str, optional
        If given, save the figure here.
    ax, ax_res : Axes, optional
        Draw into existing axes. When *ax* is given and *ax_res* is not, the
        residual panel is skipped (used for grid layouts).
    is_binned : bool
        Binned data is drawn with error bars; unbinned data as a light scatter.
    smooth_phase, smooth_flux, smooth_flux_err : array, optional
        Gaussian-smoothed data curve and its MC 1σ band.
    smooth_sigma : float, optional
        Smoothing kernel width, shown in the legend entry when given.
    residual_ylim : tuple, optional
        y-limits for the residual panel; None leaves them automatic.

    Returns
    -------
    Figure or None
        The figure containing *ax*, or None if the figure was created here and
        already closed after saving.
    """
    obs_phase = np.asarray(obs_phase, dtype=float)
    obs_flux = np.asarray(obs_flux, dtype=float)
    if obs_err is not None:
        obs_err = np.asarray(obs_err, dtype=float)

    has_model = model_phase is not None and model_flux is not None
    xerr = half_widths(obs_phase_width, obs_phase.shape) if is_binned else None

    owns_figure = ax is None
    want_residuals = obs_model is not None and obs_err is not None
    if owns_figure:
        if want_residuals:
            fig, (ax, ax_res) = plt.subplots(
                2,
                1,
                figsize=figsize or (10, 8),
                sharex=True,
                gridspec_kw={'height_ratios': [3, 1]},
            )
        else:
            fig, ax = plt.subplots(figsize=figsize or (10, 6))
            ax_res = None
    else:
        fig = ax.get_figure()
    draw_residuals = want_residuals and ax_res is not None

    # --- observed data --------------------------------------------------------
    def _draw_obs(mask: Optional[np.ndarray], label: Optional[str], color: Optional[str]):
        sel = slice(None) if mask is None else mask
        if is_binned and obs_err is not None:
            ax.errorbar(
                obs_phase[sel], obs_flux[sel],
                yerr=obs_err[sel],
                xerr=None if xerr is None else xerr[sel],
                fmt='o', markersize=4, alpha=0.7, capsize=2, elinewidth=1,
                label=label, color=color, zorder=5,
            )
        else:
            ax.scatter(
                obs_phase[sel], obs_flux[sel], s=10, alpha=0.25,
                label=label, color=color, zorder=4,
            )

    if obs_group is not None:
        groups = np.asarray(obs_group)
        # Stable order of first appearance, so colors are reproducible.
        seen: List = []
        for g in groups:
            if g not in seen:
                seen.append(g)
        for g in seen:
            mask = groups == g
            _draw_obs(mask, f"{g} (n={int(np.sum(mask))})", None)
    else:
        if obs_label is None:
            obs_label = 'Observed (phase-binned)' if is_binned else 'Observed (raw)'
        _draw_obs(None, obs_label, obs_color or 'C0')

    # --- model overlay --------------------------------------------------------
    if has_model:
        mphase = np.mod(np.asarray(model_phase, dtype=float), 1.0)
        mflux = np.asarray(model_flux, dtype=float)
        order = np.argsort(mphase)
        ax.plot(mphase[order], mflux[order], 'r-', lw=2, label=model_label, zorder=10)

    # --- smoothed data overlay ------------------------------------------------
    if smooth_phase is not None and smooth_flux is not None:
        sphase = np.mod(np.asarray(smooth_phase, dtype=float), 1.0)
        sflux = np.asarray(smooth_flux, dtype=float)
        sorder = np.argsort(sphase)
        sphase = sphase[sorder]
        sflux = sflux[sorder]
        smooth_label = 'Gaussian-smoothed data'
        if smooth_sigma is not None:
            smooth_label += f' ($\\sigma$={float(smooth_sigma):.3f})'
        ax.plot(sphase, sflux, '--', color='green', lw=1.5, label=smooth_label, zorder=8)
        if smooth_flux_err is not None:
            serr = np.asarray(smooth_flux_err, dtype=float)[sorder]
            if np.any(np.isfinite(serr)):
                ax.fill_between(
                    sphase, sflux - serr, sflux + serr,
                    color='green', alpha=0.2, label='Smoothed 1$\\sigma$ (MC)', zorder=3,
                )

    # --- labels, title, legend ------------------------------------------------
    ax.set_ylabel(ylabel, fontsize=12)
    plot_title = title if title is not None else build_fit_title(band, red_chi2)
    if plot_title:
        ax.set_title(plot_title, fontsize=14)
    legend_kwargs = {'loc': legend_loc}
    if legend_fontsize is not None:
        legend_kwargs['fontsize'] = legend_fontsize
    ax.legend(**legend_kwargs)
    ax.grid(alpha=0.3)

    # --- residuals ------------------------------------------------------------
    if draw_residuals:
        add_residual_panel(
            ax_res, obs_phase, obs_flux,
            np.asarray(obs_model, dtype=float), obs_err, xerr=xerr,
        )
        if residual_ylim is not None:
            ax_res.set_ylim(*residual_ylim)
        ax_res.set_xlabel(xlabel, fontsize=12)
    else:
        ax.set_xlabel(xlabel, fontsize=12)

    # --- output ---------------------------------------------------------------
    if output_path:
        if owns_figure:
            fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        if verbose:
            print(f"Plot saved to: {output_path}")

    if owns_figure:
        if output_path:
            plt.close(fig)
            return None
        fig.tight_layout()
        if show:
            plt.show()
    return fig


# -----------------------------------------------------------------------------
# Tabulated-simulation convenience wrappers
# -----------------------------------------------------------------------------

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
    """Observed light curve with an optional tabulated-simulation overlay.

    A thin adapter over :func:`plot_lightcurve_fit`: it interpolates *sim_df* at
    the requested *shift* and additive *scatter*, then delegates all drawing.
    The model overlay is shown at its native flux normalization (set by ``lam``
    and the XSPEC flux-vs-nH table); the only y-direction adjustment is the
    additive *scatter* floor. There is no multiplicative scale factor — see
    :func:`utils.utils.fit_simulation`.

    Parameters
    ----------
    df : DataFrame
        Observational data with columns ``phase``, ``rate``, and optionally
        ``error``, ``obs`` and ``width``.
    output_path : str, optional
        Output filename to save the plot.
    sim_df : DataFrame, optional
        Simulation data.
    shift : float, optional
        Phase shift for the simulation overlay.
    sim_column : str, default ``"fl"``
        Column name in the simulation to use. Its ``nfl_{band}`` suffix supplies
        the energy-band label in the title.
    chi2 : float, optional
        Reduced chi-squared value, shown in the title.
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure (with a
        residual panel when a model and errors are available).
    shift_fitted : bool, default False
        Whether the phase shift was optimized (True) or held at 0 (False). Only
        affects the degrees of freedom used by the displayed-χ² self-check.
    obs_column_name : str, default "rate"
        Name of the observable column being plotted (for the y-axis label).
    is_binned : bool, default False
        Whether the data has been phase-binned. If True, plots with error bars.
    smooth_df : DataFrame, optional
        Output of :func:`utils.utils.smooth_lightcurve`.
    scatter : float, default 0.0
        Constant additive scattered-flux floor added to the model overlay.
    """
    has_model = sim_df is not None and shift is not None
    has_errors = 'error' in df.columns and not df['error'].isna().all()

    obs_phase = np.mod(df["phase"].to_numpy(dtype=float), 1.0)
    obs_rate = df["rate"].to_numpy(dtype=float)
    obs_err = obs_errors(df) if has_errors else None

    model_phase = model_flux = obs_model = None
    if has_model:
        # Overlay and residuals both come from the same evaluator used by
        # fit_simulation's χ², so the drawn curve, the residual panel and the
        # displayed χ² are guaranteed to describe the same model.
        model_wrap = prepare_model_interpolator(sim_df, sim_column)
        model_phase = np.linspace(0.0, 1.0, 721)
        model_flux = model_from_wrap(*model_wrap, model_phase, shift, scatter)
        obs_model = model_from_wrap(*model_wrap, obs_phase, shift, scatter)

        # Self-check: the χ² we display must be the χ² of the model we drew.
        # This catches a `scatter` or `shift` that disagrees with the
        # fit_simulation call, which would otherwise show a correct-looking
        # number over the wrong curve.
        if chi2 is not None and np.isfinite(chi2):
            check_err = obs_err if obs_err is not None else obs_errors(df)
            n_free = 1 if shift_fitted else 0
            recomputed = float(
                np.sum(((obs_rate - obs_model) / check_err) ** 2)
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

    ylabel = (
        obs_column_name.replace("_", " ").title()
        if obs_column_name != "rate"
        else "Count rate / Flux"
    )
    phase_width = df["width"].to_numpy(dtype=float) if "width" in df.columns else None

    plot_lightcurve_fit(
        obs_phase,
        obs_rate,
        obs_err,
        model_phase=model_phase,
        model_flux=model_flux,
        obs_model=obs_model,
        obs_phase_width=phase_width,
        obs_group=df["obs"].to_numpy() if "obs" in df.columns else None,
        band=band_label_from_column(sim_column) if has_model else None,
        red_chi2=chi2,
        title=None if has_model else "Chandra Light-curve Observations",
        output_path=output_path,
        ax=ax,
        is_binned=is_binned,
        ylabel=ylabel,
        smooth_phase=None if smooth_df is None or len(smooth_df) == 0 else smooth_df["phase"].to_numpy(dtype=float),
        smooth_flux=None if smooth_df is None or len(smooth_df) == 0 else smooth_df["flux_smooth"].to_numpy(dtype=float),
        smooth_flux_err=None if smooth_df is None or len(smooth_df) == 0 else smooth_df["flux_smooth_err"].to_numpy(dtype=float),
        legend_loc="upper right",
        legend_fontsize="small",
        dpi=300,
        show=(ax is None and not output_path),
    )


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

    Each panel is drawn by :func:`plot_phase` (and therefore by
    :func:`plot_lightcurve_fit`). Panels share the observed data and differ only
    in the simulation column, so the title's energy-band label identifies them.

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
    smooth_df : DataFrame, optional
        Output of :func:`utils.utils.smooth_lightcurve`.
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
    axes = np.asarray(axes).flatten()

    for i, (col, (shift, chi2)) in enumerate(zip(sim_columns, fit_results)):
        plot_phase(
            df, None, sim_df, shift, col, chi2, axes[i], shift_fitted,
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
# MCMC diagnostic plots
# -----------------------------------------------------------------------------

def plot_corner(samples: np.ndarray, band: str, wind_model: str, output_path: str,
                param_labels: List[str] = None):
    """Generate corner plot of posterior distributions."""
    if corner is None:
        warnings.warn("corner package not installed, skipping corner plot")
        return
    if param_labels is None:
        param_labels = [f"param {i}" for i in range(np.shape(samples)[1])]

    fig = corner.corner(
        samples,
        labels=param_labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        title_fmt=".4f"
    )

    fig.suptitle(f"Posterior Distributions - {band.upper()} band ({wind_model.upper()})",
                 fontsize=14, y=1.02)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Corner plot saved to: {output_path}")


def plot_trace(sampler, band: str, wind_model: str, output_path: str,
               param_labels: List[str] = None, n_burn: int = None):
    """Generate trace plots for convergence diagnostics."""
    chain = sampler.get_chain()
    n_steps, n_walkers, n_dim = chain.shape
    if param_labels is None:
        param_labels = [f"param {i}" for i in range(n_dim)]
    if n_burn is None:
        n_burn = n_steps // 5

    fig, axes = plt.subplots(n_dim, 1, figsize=(10, 2*n_dim), sharex=True)
    if n_dim == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        for j in range(n_walkers):
            ax.plot(chain[:, j, i], alpha=0.3, lw=0.5)
        ax.set_ylabel(param_labels[i] if i < len(param_labels) else f"param {i}")
        ax.axvline(x=n_burn, color='r', linestyle='--',
                   alpha=0.5, label='Burn-in' if i == 0 else None)

    axes[-1].set_xlabel("Step")
    axes[0].legend(loc='upper right')
    fig.suptitle(f"MCMC Trace Plots - {band.upper()} band ({wind_model.upper()})", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Trace plot saved to: {output_path}")


# -----------------------------------------------------------------------------
# Simulation / geometry plots
# -----------------------------------------------------------------------------
# These read the geometry columns that simulate_lightcurve already returns:
#   l3, L3, h3   sky-plane position of the compact object relative to the
#                companion centre. (L3, h3) are Cartesian components and
#                l3 = sqrt(L3^2 + h3^2) is the projected separation -- exactly
#                the quantity the eclipse test compares against R +/- r.
#   fl           lam-normalized column density N_H (1e22 cm^-2)
#   is_eclipsed  per-phase geometric eclipse flag
# Two panels of the old plot_results.plot_geometric_parameters are deliberately
# not reproduced: "Time vs Phase" is linear by construction, and A2 is the
# polar-grid cell area, an artifact of the integration mesh rather than physics.

def _shade_eclipse(ax: plt.Axes, phase: np.ndarray, eclipsed: np.ndarray,
                   label: Optional[str] = "Geometric eclipse") -> bool:
    """Shade the eclipsed phase interval(s). Returns True if anything was drawn."""
    mask = np.asarray(eclipsed, dtype=bool)
    if not np.any(mask):
        return False
    phase = np.asarray(phase, dtype=float)
    order = np.argsort(phase)
    p, m = phase[order], mask[order]
    edges = np.flatnonzero(np.diff(m.astype(int)) != 0) + 1
    for seg in np.split(np.arange(p.size), edges):
        if seg.size and m[seg[0]]:
            ax.axvspan(p[seg[0]], p[seg[-1]], color='0.85', zorder=0,
                       label=label)
            label = None  # legend entry only once
    return True


def plot_orbit_geometry(
    sim_df: pd.DataFrame,
    R: float,
    r: float,
    d1: float,
    d2: float,
    i0: float,
    output_path: Optional[str] = None,
    band: Optional[str] = None,
    dpi: int = 150,
    verbose: bool = True,
) -> Optional[plt.Figure]:
    """Sky-projected eclipse geometry and a top-down view of the orbit.

    The most diagnostic geometry plot for a posterior: the eclipse width
    constrains a combination of ``(a, R, i0)``, so very different parameter sets
    can fit the same light curve. Seeing the projected track against the
    companion disk makes that degeneracy concrete, and shows at a glance whether
    the fitted parameters produce a *geometric* eclipse at all or whether the
    dip is pure wind absorption.

    Parameters
    ----------
    sim_df : DataFrame
        Output of ``simulate_lightcurve`` at the parameters of interest; needs
        ``phase``, ``L3``, ``h3`` and (optionally) ``is_eclipsed``.
    R, r, d1, d2, i0 : float
        Companion radius, compact-object/disk radius, the two distances from the
        centre of mass (solar radii), and inclination (degrees).
    """
    for col in ("L3", "h3"):
        if col not in sim_df.columns:
            raise KeyError(f"sim_df must contain '{col}' (from simulate_lightcurve).")
    L = sim_df["L3"].to_numpy(dtype=float)
    h = sim_df["h3"].to_numpy(dtype=float)
    phase = sim_df["phase"].to_numpy(dtype=float) if "phase" in sim_df.columns else None
    eclipsed = (sim_df["is_eclipsed"].to_numpy(dtype=bool)
                if "is_eclipsed" in sim_df.columns else np.zeros(L.size, dtype=bool))
    a = float(d1) + float(d2)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # --- panel 1: sky projection (what the observer sees) --------------------
    ax.add_patch(plt.Circle((0, 0), float(R), color='darkorange', alpha=0.45,
                            zorder=2, label=f'Companion (R = {float(R):.2f} R$_\\odot$)'))
    ax.plot(L, h, '-', color='0.4', lw=1.2, zorder=3,
            label='Compact-object track (projected)')
    behind = h > 0  # emitter behind the companion: only these can be occulted
    ax.plot(L[behind], h[behind], '.', color='C0', ms=3, zorder=4,
            label='Behind companion')
    if np.any(eclipsed):
        ax.plot(L[eclipsed], h[eclipsed], 'o', color='crimson', ms=4, zorder=6,
                label=f'Eclipsed ({100.0 * eclipsed.mean():.1f}% of orbit)')
    else:
        ax.plot([], [], ' ', label='No geometric eclipse')

    # Mark phase 0 and the deepest-projection phase.
    if phase is not None:
        i_zero = int(np.argmin(np.abs(np.mod(phase, 1.0))))
        ax.plot(L[i_zero], h[i_zero], '*', color='k', ms=13, zorder=7,
                label='Phase 0')
    l_proj = np.hypot(L, h)
    i_min = int(np.argmin(np.where(h > 0, l_proj, np.inf)))
    if np.isfinite(l_proj[i_min]):
        ax.plot(L[i_min], h[i_min], 'v', color='crimson', ms=8, zorder=7,
                label=f'Min projected sep. = {l_proj[i_min]:.2f} R$_\\odot$')

    ax.axhline(0, color='0.8', lw=0.8, zorder=1)
    ax.axvline(0, color='0.8', lw=0.8, zorder=1)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('Sky-plane offset $L$ (R$_\\odot$)', fontsize=11)
    ax.set_ylabel('Sky-plane offset $h$ (R$_\\odot$)', fontsize=11)
    ax.set_title('Projected geometry (observer view)', fontsize=12)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.25)

    # --- panel 2: orbital plane, top-down ------------------------------------
    th = np.linspace(0, 2 * np.pi, 361)
    ax2.plot(a * np.cos(th), a * np.sin(th), '--', color='0.5', lw=1.0,
             label=f'Relative orbit ($a$ = {a:.2f} R$_\\odot$)')
    ax2.add_patch(plt.Circle((0, 0), float(R), color='darkorange', alpha=0.45,
                             zorder=3, label='Companion'))
    ax2.plot([0], [0], '+', color='k', ms=10, zorder=4)
    ax2.plot([a], [0], 'o', color='C0', ms=7, zorder=4, label='Compact object')
    # Line of sight enters the orbital plane at angle i0 from the normal; the
    # observer sits in the +y direction of this projection.
    span = 1.25 * a
    ax2.annotate('', xy=(0, -0.95 * span), xytext=(0, -0.55 * span),
                 arrowprops=dict(arrowstyle='-|>', color='C3', lw=1.8))
    ax2.text(0.03 * span, -0.78 * span,
             f'to observer\n$i_0$ = {float(i0):.2f}$^\\circ$',
             color='C3', fontsize=9, va='center')
    ax2.plot([0, a], [0, 0], ':', color='0.3', lw=1.0)
    ax2.text(0.5 * a, 0.04 * span,
             f'$d_1$={float(d1):.2f}, $d_2$={float(d2):.2f} R$_\\odot$',
             fontsize=8, ha='center', color='0.3')
    ax2.set_xlim(-span, span)
    ax2.set_ylim(-span, span)
    ax2.set_aspect('equal')
    ax2.set_xlabel('Orbital plane $x$ (R$_\\odot$)', fontsize=11)
    ax2.set_ylabel('Orbital plane $y$ (R$_\\odot$)', fontsize=11)
    ax2.set_title('Orbit, to scale', fontsize=12)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(alpha=0.25)

    # Eclipse condition, stated numerically so the figure is self-contained.
    if np.isfinite(l_proj[i_min]):
        verdict = ('total eclipse' if l_proj[i_min] <= float(R) - float(r)
                   else 'partial eclipse' if l_proj[i_min] <= float(R) + float(r)
                   else 'no eclipse')
        fig.text(0.5, 0.005,
                 f'min projected separation {l_proj[i_min]:.3f} vs '
                 f'R - r = {float(R) - float(r):.3f} and '
                 f'R + r = {float(R) + float(r):.3f} R$_\\odot$  ->  {verdict}',
                 ha='center', fontsize=9, color='0.25')

    if band:
        fig.suptitle(f'{band} band - binary geometry', fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        if verbose:
            print(f"Geometry plot saved to: {output_path}")
        plt.close(fig)
        return None
    return fig


def plot_geometry_vs_phase(
    sim_df: pd.DataFrame,
    R: float,
    r: float,
    band: Optional[str] = None,
    flux_column: Optional[str] = None,
    output_path: Optional[str] = None,
    dpi: int = 150,
    verbose: bool = True,
) -> Optional[plt.Figure]:
    """Projected separation, sky-plane components, N_H and band flux vs phase.

    Panel 1 is the diagnostic one: ``l3(phase)`` against the ``R - r`` / ``R + r``
    thresholds shows directly which phases are geometrically occulted and how
    much margin the fit has, rather than leaving the eclipse width as an
    emergent property of the light curve.
    """
    need = ("phase", "l3", "L3", "h3")
    for col in need:
        if col not in sim_df.columns:
            raise KeyError(f"sim_df must contain '{col}' (from simulate_lightcurve).")
    phase = sim_df["phase"].to_numpy(dtype=float)
    order = np.argsort(phase)
    phase = phase[order]
    l3 = sim_df["l3"].to_numpy(dtype=float)[order]
    L3 = sim_df["L3"].to_numpy(dtype=float)[order]
    h3 = sim_df["h3"].to_numpy(dtype=float)[order]
    eclipsed = (sim_df["is_eclipsed"].to_numpy(dtype=bool)[order]
                if "is_eclipsed" in sim_df.columns else np.zeros(phase.size, bool))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    ax_sep, ax_comp, ax_nh, ax_flux = axes.flatten()

    # 1. projected separation vs the eclipse thresholds
    _shade_eclipse(ax_sep, phase, eclipsed)
    ax_sep.plot(phase, l3, '-', color='C0', lw=2, label='$l$ (projected sep.)')
    ax_sep.axhline(float(R) + float(r), color='C3', ls='--', lw=1.2,
                   label=f'$R+r$ = {float(R) + float(r):.2f}')
    ax_sep.axhline(max(float(R) - float(r), 0.0), color='C3', ls=':', lw=1.2,
                   label=f'$R-r$ = {float(R) - float(r):.2f}')
    ax_sep.set_ylabel('Distance (R$_\\odot$)')
    ax_sep.set_title('Projected separation vs eclipse thresholds', fontsize=11)
    ax_sep.legend(fontsize=8, loc='best')
    ax_sep.grid(alpha=0.3)

    # 2. sky-plane components
    _shade_eclipse(ax_comp, phase, eclipsed, label=None)
    ax_comp.plot(phase, L3, '-', color='C0', lw=1.8, label='$L$ (in-plane)')
    ax_comp.plot(phase, h3, '--', color='C2', lw=1.8, label='$h$ (out-of-plane)')
    ax_comp.axhline(0, color='0.7', lw=0.8)
    ax_comp.set_ylabel('Offset (R$_\\odot$)')
    ax_comp.set_title('Sky-plane components ($h>0$: emitter behind)', fontsize=11)
    ax_comp.legend(fontsize=8, loc='best')
    ax_comp.grid(alpha=0.3)

    # 3. column density
    if 'fl' in sim_df.columns:
        fl = sim_df['fl'].to_numpy(dtype=float)[order]
        _shade_eclipse(ax_nh, phase, eclipsed, label=None)
        ax_nh.plot(phase, fl, '-', color='C4', lw=2)
        ax_nh.axhline(float(np.nanmean(fl)), color='0.4', ls='--', lw=1.0,
                      label=f'orbit mean = {np.nanmean(fl):.4g} (= $\\lambda$)')
        ax_nh.set_ylabel('$N_H$ ($10^{22}$ cm$^{-2}$)')
        ax_nh.set_title('Wind column density along the line of sight', fontsize=11)
        ax_nh.legend(fontsize=8, loc='best')
    else:
        ax_nh.text(0.5, 0.5, "no 'fl' column", ha='center', transform=ax_nh.transAxes)
    ax_nh.set_xlabel('Orbital phase')
    ax_nh.grid(alpha=0.3)

    # 4. resulting band flux
    if flux_column is None and band:
        flux_column = f"nfl_{band.lower()}"
    if flux_column and flux_column in sim_df.columns:
        _shade_eclipse(ax_flux, phase, eclipsed, label=None)
        ax_flux.plot(phase, sim_df[flux_column].to_numpy(dtype=float)[order],
                     '-', color='C3', lw=2)
        ax_flux.set_ylabel('Flux (erg/cm$^2$/s)')
        ax_flux.set_title(f'Model band flux ({flux_column})', fontsize=11)
    else:
        ax_flux.text(0.5, 0.5, 'no band flux column', ha='center',
                     transform=ax_flux.transAxes)
    ax_flux.set_xlabel('Orbital phase')
    ax_flux.grid(alpha=0.3)

    if band:
        fig.suptitle(f'{band} band - geometry and absorption vs phase', fontsize=13)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        if verbose:
            print(f"Geometry-vs-phase plot saved to: {output_path}")
        plt.close(fig)
        return None
    return fig


def plot_wind_profile(
    radii: np.ndarray,
    g_samples: np.ndarray,
    R: Optional[float] = None,
    probed_range: Optional[tuple[float, float]] = None,
    mark_radii: Optional[Dict[str, float]] = None,
    wind_model: str = "",
    band: Optional[str] = None,
    shape_summary: Optional[str] = None,
    output_path: Optional[str] = None,
    dpi: int = 150,
    verbose: bool = True,
) -> Optional[plt.Figure]:
    """Dimensionless wind density profile g(r) with a posterior credible band.

    The wind-shape parameters are what the fit is ultimately constraining, but
    they are only interpretable jointly (``Rb`` and ``p`` trade off strongly).
    Propagating the posterior into g(r) shows the constraint on the quantity
    that actually enters the model, and marking the radii the line of sight
    truly probes shows which part of the profile the data can speak to at all.

    Parameters
    ----------
    radii : array, shape (n_r,)
        Radii (solar radii) at which the profile was evaluated.
    g_samples : array, shape (n_samples, n_r)
        g(r) for each posterior draw. A single row is fine (fixed shape).
    R : float, optional
        Companion radius, drawn as the stellar surface.
    probed_range : (float, float), optional
        Min/max radius actually sampled along the line of sight.
    mark_radii : dict, optional
        ``{label: radius}`` characteristic radii to mark (e.g. the break radius
        ``Rb``), so a feature of the profile can be read against the radii the
        data constrain.
    """
    radii = np.asarray(radii, dtype=float)
    g = np.atleast_2d(np.asarray(g_samples, dtype=float))
    finite = np.all(np.isfinite(g), axis=1)
    g = g[finite] if np.any(finite) else g

    fig, ax = plt.subplots(figsize=(8, 6))
    med = np.nanmedian(g, axis=0)
    if g.shape[0] >= 20:
        lo95, lo68, hi68, hi95 = np.nanpercentile(g, [2.5, 16, 84, 97.5], axis=0)
        ax.fill_between(radii, lo95, hi95, color='C0', alpha=0.15, label='95% credible')
        ax.fill_between(radii, lo68, hi68, color='C0', alpha=0.30, label='68% credible')
        ax.plot(radii, med, '-', color='C0', lw=2,
                label=f'posterior median ({g.shape[0]} draws)')
    else:
        ax.plot(radii, med, '-', color='C0', lw=2, label='g(r)')

    # r^-2 reference: every profile is asymptotically a free-streaming wind.
    ref_at = radii[-1]
    ref_val = med[-1] if np.isfinite(med[-1]) else np.nanmedian(med)
    if np.isfinite(ref_val) and ref_val > 0:
        ax.plot(radii, ref_val * (radii / ref_at) ** -2.0, ':', color='0.5', lw=1.4,
                label=r'$r^{-2}$ (free-streaming)')

    if R is not None and np.isfinite(R):
        ax.axvline(float(R), color='darkorange', ls='--', lw=1.5,
                   label=f'companion surface $R$ = {float(R):.2f} R$_\\odot$')
    if probed_range is not None:
        lo, hi = float(probed_range[0]), float(probed_range[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.axvspan(lo, hi, color='C2', alpha=0.10, zorder=0,
                       label=f'radii probed by the LOS ({lo:.1f}-{hi:.1f} R$_\\odot$)')
    for label, rad in (mark_radii or {}).items():
        if rad is not None and np.isfinite(rad) and rad > 0:
            ax.axvline(float(rad), color='C4', ls='-.', lw=1.3,
                       label=f'{label} = {float(rad):.2f} R$_\\odot$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Radius from companion centre $r$ (R$_\\odot$)', fontsize=11)
    ax.set_ylabel('Dimensionless wind density $g(r)$', fontsize=11)
    title = 'Wind density profile'
    if wind_model:
        title += f' - {wind_model}'
    if band:
        title = f'{band} band - ' + title
    ax.set_title(title, fontsize=12)
    if shape_summary:
        ax.text(0.02, 0.02, shape_summary, transform=ax.transAxes, fontsize=8,
                va='bottom', ha='left', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        if verbose:
            print(f"Wind-profile plot saved to: {output_path}")
        plt.close(fig)
        return None
    return fig


def plot_simulation_bands(
    sim_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "XRB Lightcurve Simulation Results",
    dpi: int = 300,
) -> None:
    """Grid of per-band model light curves from a simulation DataFrame.

    One panel for the raw wind LOS integral and the lam-scaled column density
    (when present), then one per detected ``nfl_{band}`` column.
    """
    bands = detect_energy_bands(sim_df)
    has_base = "flx" in sim_df.columns and "fl" in sim_df.columns
    panels: List[tuple] = []
    if has_base:
        panels.append(("flx", "Flux", "Wind LOS integral", "b-"))
        panels.append(("fl", "Scaled $N_H$", "Scaled nH", "g-"))
    for b in bands:
        name, erange = get_band_display_name(b)
        panels.append((f"nfl_{b}", f"{name} Band Flux",
                       name + (f" ({erange})" if erange else ""), "b-"))
    if not panels:
        print("No plottable columns found in simulation DataFrame.")
        return

    x = sim_df["deg"] if "deg" in sim_df.columns else sim_df["phase"]
    xlabel = "Phase (degrees)" if "deg" in sim_df.columns else "Orbital phase"
    n_cols = min(2, len(panels))
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows),
                            squeeze=False)
    axes = axes.flatten()
    for ax, (col, ylabel, label, style) in zip(axes, panels):
        ax.plot(x, sim_df[col], style, lw=2, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    for ax in axes[len(panels):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Plots saved to {output_path}")
        plt.close(fig)
    else:
        plt.show()
