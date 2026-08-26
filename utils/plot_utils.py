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
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.utils import (
    band_label_from_column,
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
