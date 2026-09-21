"""
Helpers behind ``figures/paper_figures.ipynb``: the experiments, the caching
and the drawing for every results figure and table of the CLOAK methods
paper. Everything runs on synthetic data from ``cloak.synthetic.fiducial``.

The notebook is the user interface; this module keeps it short. Expensive
steps (MCMC fits, the SBC batch) run as ``python -m cloak.mcmc_fit`` /
``figures/run_sbc.py`` subprocesses and are skipped when their outputs exist
under ``figures/cache/`` (not tracked); a fit's cache directory is keyed by a
digest of its inputs (flux table, light-curve files, fit options), so a
regenerated table or data set never reuses a stale chain. Small summaries the figures need go to
``figures/results/`` (tracked) and the figures themselves to ``figures/``.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cloak import kernel as K  # noqa: E402
from cloak import mcmc_fit as M  # noqa: E402
from cloak import utils as U  # noqa: E402
from cloak.plots import plot_lightcurve_fit  # noqa: E402
from cloak.synthetic import fiducial as F  # noqa: E402

PY = sys.executable
FIG_DIR = os.path.join(ROOT, "figures")
CACHE = os.path.join(FIG_DIR, "cache")
RESULTS = os.path.join(FIG_DIR, "results")
DATA = os.path.join(ROOT, "synthetic_data")
TABLES = os.path.join(DATA, "tables")
BANDS = dict(U.CHANDRA_BANDS)                       # name -> (emin, emax) keV
WIDTH = 13.5 / 2.54                                 # MDPI figure width in inches
PROFILE_LABELS = {"smooth_pl": "smoothly broken power law", "confinement": "confinement", "beta_law": r"$\beta$-law"}
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)


# ----------------------------------------------------------------------------
# Style, paths, small utilities
# ----------------------------------------------------------------------------

def style() -> None:
    """Paper style: 9 pt text, thin lines, embedded TrueType fonts."""
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.2, "axes.linewidth": 0.7,
        "figure.dpi": 110, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": False, "axes.formatter.use_mathtext": True,
    })


def save_fig(fig: plt.Figure, name: str) -> str:
    """Write ``figures/<name>.pdf`` (vector, for the paper)."""
    path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"saved {os.path.relpath(path, ROOT)}")
    return path


def write_table(name: str, rows: Sequence[Sequence[str]], header: Optional[Sequence[str]] = None,
                caption_note: str = "") -> str:
    """LaTeX tabular body ``figures/results/<name>.tex`` to paste or \\input into the paper."""
    lines = []
    if caption_note:
        lines.append(f"% {caption_note}")
    if header:
        lines.append(" & ".join(header) + r" \\")
        lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + r" \\")
    path = os.path.join(RESULTS, f"{name}.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"saved {os.path.relpath(path, ROOT)}")
    return path


def save_json(name: str, obj) -> str:
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
    return path


def load_json(name: str):
    path = os.path.join(RESULTS, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def label_panels(axes, letters: str = "abcdefgh") -> None:
    """Panel letters as left-aligned titles: above the axes, never over a label or the data."""
    for letter, ax in zip(letters, np.ravel(axes)):
        ax.set_title(f"({letter})", loc="left", fontweight="bold", fontsize=9)


LEGEND_ROW_IN = 0.17      # height of one legend row at 7 pt, inches


def panels(n: int, height: float, legend_rows: int, width: float = WIDTH, width_ratios=None, span_legend: bool = False):
    """*n* side-by-side panels with a legend strip beneath them, laid out by the constrained-layout engine.

    Returns ``(fig, axes, legend_axes)``; the legend axes are invisible frames of the strip (one per
    panel, or a single one spanning the width when *span_legend*). Legends drawn into them never
    overlap data, tick labels or each other."""
    lh = LEGEND_ROW_IN * max(1, legend_rows) + 0.12
    fig = plt.figure(figsize=(width, height + lh), constrained_layout=True)
    gs = fig.add_gridspec(2, n, height_ratios=[height, lh], width_ratios=width_ratios)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    if span_legend:
        laxes = [fig.add_subplot(gs[1, :])]
    else:
        laxes = [fig.add_subplot(gs[1, i]) for i in range(n)]
    for la in laxes:
        la.axis("off")
    return fig, axes, laxes


def put_legend(lax, src_ax=None, ncol: int = 1, fontsize: float = 7, handles=None, labels=None):
    """Draw the legend of *src_ax* (or the given handles) into the legend strip *lax*."""
    if handles is None:
        handles, labels = src_ax.get_legend_handles_labels()
    if not handles:
        return None
    return lax.legend(handles, labels, loc="upper center", ncol=ncol, fontsize=fontsize, frameon=False,
                      borderaxespad=0.0, columnspacing=1.0, handlelength=1.6, handletextpad=0.5)


def merged_handles(axes):
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    return handles, labels


def fmt(x: float, digits: int = 3) -> str:
    """Compact number for tables: 3 significant digits, scientific when needed."""
    if x == 0 or not np.isfinite(x):
        return f"{x:g}"
    if 1e-3 <= abs(x) < 1e4:
        return f"{x:.{digits}g}"
    return f"{x:.{digits - 1}e}"


# ----------------------------------------------------------------------------
# Tables (flux vs nH) and fiducial systems
# ----------------------------------------------------------------------------

def available_tables() -> Dict[str, str]:
    """Band -> flux-vs-nH CSV. The generic tables written by the data notebook
    (``synthetic_data/tables/flux_vs_nH_<band>.csv``) are used exclusively as soon as any exists;
    only without them do the tracked example table(s) ``synthetic_data/flux_vs_nH_tbabs_<band>.csv``
    serve, so tables from two different spectra are never mixed in one figure."""
    generic = {os.path.basename(p)[len("flux_vs_nH_"):-4]: p
               for p in sorted(glob.glob(os.path.join(TABLES, "flux_vs_nH_*.csv")))}
    generic = {b: p for b, p in generic.items() if b in BANDS}
    if generic:
        return generic
    return {b: os.path.join(DATA, f"flux_vs_nH_tbabs_{b}.csv") for b in BANDS
            if os.path.exists(os.path.join(DATA, f"flux_vs_nH_tbabs_{b}.csv"))}


def require_table(band: str) -> str:
    tables = available_tables()
    if band not in tables:
        raise FileNotFoundError(
            f"no flux-vs-nH table for band '{band}' (have {sorted(tables)}); run the data notebook "
            f"synthetic_data/generate_synthetic_data.ipynb under HEASoft to make synthetic_data/tables/.")
    return tables[band]


def system_kwargs(name: str, band: str = "broad", dth: float = 1.0, **overrides) -> Dict:
    """Keywords for ``cloak.kernel.simulate_band_flux`` for fiducial system *name*."""
    kw = F.simulation_kwargs(F.SYSTEMS[name])
    kw.update(flux_csv_path=require_table(band), band=band, dth=dth)
    kw.update(overrides)
    return kw


def curve(name: str, band: str = "broad", dth: float = 1.0, **overrides) -> Tuple[np.ndarray, np.ndarray]:
    return K.simulate_band_flux(**system_kwargs(name, band, dth, **overrides))


def data_dir(name: str, band: str) -> str:
    return os.path.join(DATA, f"system{name}", band)


def truth_file(name: str, band: str) -> str:
    return os.path.join(data_dir(name, band), f"system{name}_{band}_truth.json")


def load_truth(name: str, band: str = "broad") -> Optional[dict]:
    path = truth_file(name, band)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def fiducial_table() -> str:
    """Rows of Table 'Fiducial synthetic systems'."""
    rows = []
    A, B = F.SYSTEMS["A"], F.SYSTEMS["B"]
    dA, dB = F.describe(A), F.describe(B)
    gA, gB = F.geometry(A), F.geometry(B)

    def shape(s):
        return ", ".join(f"${k}={v:g}$" for k, v in s["wind_params"].items())
    rows.append(["$P$ (d)", f"{dA['period_d']:g}", f"{dB['period_d']:g}"])
    rows.append(["$a$ ($R_\\odot$); $M_{\\rm tot}$ ($M_\\odot$)", f"{dA['a']:g}; {dA['M_tot']:.1f}", f"{dB['a']:g}; {dB['M_tot']:.1f}"])
    rows.append(["$R$, $r$ ($R_\\odot$)", f"{gA['R']:g}, {gA['r']:g}", f"{gB['R']:g}, {gB['r']:g}"])
    rows.append(["$i$ (deg)", f"{gA['i0']:g}", f"{gB['i0']:g}"])
    rows.append(["profile; shape", f"{PROFILE_LABELS[A['wind_model']]}; {shape(A)}", f"{PROFILE_LABELS[B['wind_model']]}; {shape(B)}"])
    rows.append(["$\\dot M$ ($M_\\odot$\\,yr$^{-1}$), $v_\\infty$ (km\\,s$^{-1}$)", f"{A['mdot']:.0e}, {A['v_inf']:g}", f"{B['mdot']:.0e}, {B['v_inf']:g}"])
    rows.append(["$f_{\\rm opa}$, $f_{\\rm sc}/F_{\\rm out}$", f"{A['f_opacity']:g}, {A['observation']['scatter_fraction']:g}",
                 f"{B['f_opacity']:g}, {B['observation']['scatter_fraction']:g}"])
    rows.append(["bands (keV)", ", ".join(f"{lo:g}--{hi:g}" for lo, hi in BANDS.values()), "same"])
    return write_table("tab_fiducial", rows, caption_note="Table: fiducial synthetic systems (generated by figures/figlib.py)")


def priors_table() -> str:
    """Rows of Table 'Priors used in the synthetic fits' (the fitter's defaults)."""
    def cell(p):
        return f"$\\mathcal{{N}}({p['mean']:g},\\,{p['std']:g})$ on $[{p['min']:g},\\,{p['max']:g}]$"
    sp = M.MODES["kepler_mtot"]["scale_priors"]
    rows = [
        ["$M_{\\rm tot}$ ($M_\\odot$)", cell(sp["M_tot"]), "broad; see Identifiability"],
        ["$q_m$", cell(sp["q_m"]), "posterior $\\equiv$ prior (Prop.~1)"],
        ["$R$ ($R_\\odot$)", cell(M.R_PRIOR), "external anchor of the scale"],
        ["$r$ ($R_\\odot$)", cell(M.SMALL_R_PRIOR), ""],
        ["$i$ (deg)", cell(M.I0_PRIOR), ""],
        ["$R_{\\rm b}$ ($R_\\odot$), $p$", cell(M.WIND_SHAPE_PRIORS["Rb"]) + "; " + cell(M.WIND_SHAPE_PRIORS["p"]), "$R_{\\rm b}\\ge R$ enforced"],
        ["$f_{\\rm c}$, $\\ell$", cell(M.WIND_SHAPE_PRIORS["fconf"]) + "; " + cell(M.WIND_SHAPE_PRIORS["ell"]), "confinement profile"],
        ["$\\beta$, $H$", cell(M.WIND_SHAPE_PRIORS["beta"]) + "; " + cell(M.WIND_SHAPE_PRIORS["H"]), "$\\beta$-law profile"],
        ["$\\log_{10} f_{\\rm opa}$", cell(M.FOPACITY_PRIOR), ""],
        ["$f_{\\rm sc}$", "$\\mathcal{N}(\\bar F_{0.4\\text{--}0.6}, \\bar F_{0.4\\text{--}0.6})$ on $[0, F_{\\max}]$",
         "data-driven: $\\bar F$ the mean flux in phase 0.4--0.6, $F_{\\max}$ the brightest bin"],
        ["$\\ln f$", cell(M.JITTER_PRIOR), "jitter likelihood only"],
    ]
    return write_table("tab_priors", rows, caption_note="Table: priors (the fitter's defaults, from cloak/mcmc_fit.py)")


# ----------------------------------------------------------------------------
# Figure 2: wind profiles
# ----------------------------------------------------------------------------

def fig_wind_profiles(R: float = 2.5) -> plt.Figure:
    """(a) g(r) of the three profiles at their defaults (R_star = R), with the r^-2 asymptote;
    (b) the dimensionless column of a ray behind the star versus impact parameter."""
    r = np.logspace(np.log10(R * 1.001), np.log10(60.0), 600)
    fig, (ax, bx), (la, lb) = panels(2, WIDTH * 0.4, legend_rows=2)
    for model in K.WIND_MODEL_IDS:
        params = K.default_wind_params(model, R)
        g = K.evaluate_g_profile(r, model, params)
        # Normalize each profile to the same r^-2 asymptote so shapes compare.
        C = K.wind_asymptotic_coefficient(model, params) if hasattr(K, "wind_asymptotic_coefficient") else 1.0
        ax.loglog(r, g / C, label=PROFILE_LABELS[model])
        model_id, p1, p2, p3 = K.pack_wind_params(model, params)
        b = np.linspace(R * 1.002, 40.0, 300)
        col = np.array([K._los_gl_quadrature(float(bb), 1e6, model_id, p1, p2, p3, K._GL16_X, K._GL16_W) for bb in b])
        bx.semilogy(b, col / C, label=PROFILE_LABELS[model])
    ax.loglog(r, r ** -2.0, "k:", lw=0.8, label=r"$r^{-2}$")
    ax.set(xlabel=r"$r$ ($R_\odot$)", ylabel=r"$g(r)\,/\,C$")
    ax.axvline(R, color="0.6", lw=0.7); ax.text(R * 1.05, ax.get_ylim()[0] * 3, r"$R_\star$", color="0.4")
    bx.set(xlabel=r"impact parameter $b$ ($R_\odot$)", ylabel=r"$\int g\,{\rm d}z\,/\,C$")
    bx.set_title("emitter behind the star", loc="right", fontsize=7, color="0.35")
    label_panels([ax, bx])
    put_legend(la, ax, ncol=2); put_legend(lb, bx, ncol=2)
    return fig


# ----------------------------------------------------------------------------
# Figure 3: quadrature
# ----------------------------------------------------------------------------

def integrand_u(u: np.ndarray, b: float, model: str, params: Dict[str, float]) -> np.ndarray:
    """b g(b / cos u) sec^2 u, the transformed LOS integrand."""
    model_id, p1, p2, p3 = K.pack_wind_params(model, params)
    out = np.zeros_like(u)
    for i, uu in enumerate(u):
        c = math.cos(uu)
        out[i] = b * K._g_profile(b / c, model_id, p1, p2, p3) / (c * c) if c > 1e-15 else 0.0
    return out


def reference_column(b: float, z_start: float, model: str, params: Dict[str, float]) -> float:
    """Adaptive quadrature of the untransformed integral int_{-inf}^{z_start} g(sqrt(b^2+z^2)) dz."""
    model_id, p1, p2, p3 = K.pack_wind_params(model, params)

    def f(z):
        return K._g_profile(math.sqrt(b * b + z * z), model_id, p1, p2, p3)
    total = 0.0
    edges = [-np.inf, -50.0, 0.0, 50.0, z_start] if z_start > 50 else [-np.inf, -50.0, min(0.0, z_start), z_start]
    edges = sorted(set(e for e in edges if e <= z_start))
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        val, _ = quad(f, lo, hi, epsabs=0.0, epsrel=1e-12, limit=800)
        total += val
    return total


def gl_column(b: float, z_start: float, model: str, params: Dict[str, float], n: int) -> float:
    """The n-point Gauss-Legendre rule in u over the whole interval (no limb split)."""
    model_id, p1, p2, p3 = K.pack_wind_params(model, params)
    x, w = np.polynomial.legendre.leggauss(n)
    u_lo, u_hi = -0.5 * math.pi, math.atan(z_start / b)
    half, mid = 0.5 * (u_hi - u_lo), 0.5 * (u_hi + u_lo)
    total = 0.0
    for xk, wk in zip(x, w):
        c = math.cos(mid + half * xk)
        if c > 1e-15:
            total += wk * K._g_profile(b / c, model_id, p1, p2, p3) / (c * c)
    return total * b * half


def kernel_column(b: float, z_start: float, model: str, params: Dict[str, float]) -> float:
    """What the kernel computes (16 nodes, with the limb split for beta_law)."""
    model_id, p1, p2, p3 = K.pack_wind_params(model, params)
    return float(K._los_gl_quadrature(float(b), float(z_start), model_id, p1, p2, p3, K._GL16_X, K._GL16_W))


def fig_quadrature(R: float = 2.0, z_start: float = 17.0) -> Tuple[plt.Figure, dict]:
    params = {m: K.default_wind_params(m, R) for m in K.WIND_MODEL_IDS}
    b_list = [2.1, 3.0, 5.0, 20.0]                       # all beyond R_star = 2, so every profile is defined
    b_sets = {m: b_list for m in K.WIND_MODEL_IDS}
    fig, (ax, bx), (la, lb) = panels(2, WIDTH * 0.42, legend_rows=4)
    axes = [ax, bx]
    u = np.linspace(-0.5 * math.pi + 1e-4, math.atan(z_start / b_list[0]), 800)
    styles = {"smooth_pl": "-", "confinement": "--", "beta_law": ":"}
    short = {"smooth_pl": "broken PL", "confinement": "confinement", "beta_law": r"$\beta$-law"}
    for model, bs in b_sets.items():
        for j, b in enumerate(bs):
            uu = u[u < math.atan(z_start / b)]
            ax.plot(uu, integrand_u(uu, b, model, params[model]), styles[model], color=f"C{j}", lw=1.0)
    ax.set(xlabel=r"$u = \arctan(z/b)$", ylabel=r"$b\,g(b/\cos u)\,\sec^2 u$", yscale="log")
    ax.set_xlim(-0.5 * math.pi, 1.6)
    from matplotlib.lines import Line2D
    handles_a = [Line2D([], [], color=f"C{j}", label=f"$b = {b:g}\\,R_\\odot$") for j, b in enumerate(b_list)]
    handles_a += [Line2D([], [], color="0.3", ls=styles[m], label=short[m]) for m in K.WIND_MODEL_IDS]
    put_legend(la, handles=handles_a, labels=[h.get_label() for h in handles_a], ncol=2, fontsize=6.5)

    cases = [("smooth_pl", 2.1), ("smooth_pl", 5.0), ("confinement", 2.1), ("beta_law", 2.5), ("beta_law", 2.1), ("beta_law", 2.02)]
    ns = np.arange(4, 34, 2)
    summary = {}
    for model, b in cases:
        ref = reference_column(b, z_start, model, params[model])
        err = np.array([abs(gl_column(b, z_start, model, params[model], int(n)) / ref - 1.0) for n in ns])
        lab = f"{short[model]}, $b={b:g}$" + (f" ($b-R_\\star={b - R:g}$)" if model == "beta_law" else "")
        line, = bx.semilogy(ns, np.maximum(err, 1e-17), marker="o", ms=3, lw=0.9, label=lab)
        kern = abs(kernel_column(b, z_start, model, params[model]) / ref - 1.0)
        bx.plot([16], [max(kern, 1e-17)], marker="*", ms=9, color=line.get_color(), ls="none")
        summary[f"{model} b={b:g}"] = {"gl16_plain": float(err[ns == 16][0]), "kernel": float(kern), "reference": ref}
    bx.axvline(16, color="0.5", lw=0.8)
    bx.set(xlabel=r"nodes $n$", ylabel="relative error", ylim=(1e-17, 3))
    bx.plot([], [], "k*", ms=8, ls="none", label="kernel at $n=16$ (limb split)")
    label_panels(axes)
    put_legend(lb, bx, ncol=2, fontsize=6.5)
    save_json("quadrature", summary)
    return fig, summary


# ----------------------------------------------------------------------------
# Figure 4: grid convergence
# ----------------------------------------------------------------------------

def _timed(fn, repeat: int = 3):
    fn()                                   # warm-up (numba compile, table cache)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter(); out = fn(); times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


def convergence_study(systems: Sequence[str] = ("A", "B"), dths=(4.0, 2.0, 1.0, 0.5, 0.25), d2hs=(12.0, 6.0, 3.0, 2.0, 1.0),
                      band: str = "broad", extended_r: float = 1.0, force: bool = False) -> dict:
    """Max relative change of the band flux against the finest grid, for the phase step (at
    d2h = 6; the coarser curve interpolated onto the finest grid, as the fitters interpolate the
    model onto the data phases) and the sector size (at dth = 1), with wall time per light curve.
    Cached in results/."""
    cached = load_json("convergence")
    if cached is not None and not force:
        return cached
    out = {"band": band, "dth": {}, "d2h": {}, "dths": list(dths), "d2hs": list(d2hs)}

    def series(kwargs_fn, values, key):
        curves, times = {}, {}
        for v in values:
            (ph, fl), dt = _timed(lambda v=v: K.simulate_band_flux(**kwargs_fn(v)))
            curves[v], times[v] = (ph, fl), dt
        ref_ph, ref_fl = curves[values[-1]]
        rows = []
        for v in values:
            ph, fl = curves[v]
            # The model at a phase does not depend on the grid it belongs to, so
            # the coarse curve is compared as the fitters use it: interpolated
            # (periodically, linearly) onto the reference grid's phases.
            on_ref = U.eval_periodic(*U.periodic_model(ph, fl), ref_ph)
            # Normalized to the out-of-eclipse flux: near totality the flux falls by
            # orders of magnitude within one step, so a per-point relative error
            # would be dominated by the bins that carry no signal.
            rel = np.abs(on_ref - ref_fl) / ref_fl.max()
            last = v == values[-1]
            rows.append({"value": v, "max_rel_change": 0.0 if last else float(np.max(rel)),
                         "mean_rel_change": 0.0 if last else float(np.mean(rel)), "wall_ms": 1e3 * times[v]})
        return rows
    for name in systems:
        out["dth"][name] = series(lambda v, n=name: system_kwargs(n, band, dth=v), list(dths), "dth")
        out["d2h"][name] = series(lambda v, n=name: system_kwargs(n, band, dth=1.0, d2h=v), list(d2hs), "d2h")
    # An extended emitter (r = extended_r) on System A's geometry: the sector grid matters only here.
    out["d2h"]["A_extended"] = series(lambda v: system_kwargs("A", band, dth=1.0, d2h=v, r=extended_r), list(d2hs), "d2h")
    out["extended_r"] = extended_r
    save_json("convergence", out)
    return out


def _step_axis(ax, values):
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xticks(values); ax.set_xticklabels([f"{v:g}" for v in values]); ax.minorticks_off()


def fig_convergence(study: dict) -> plt.Figure:
    fig, (ax, bx), (la, lb) = panels(2, WIDTH * 0.42, legend_rows=3)
    axes = [ax, bx]
    dths = [v for v in study["dths"]]
    for k, (name, rows) in enumerate(study["dth"].items()):
        x = [r["value"] for r in rows][:-1]
        mean = [max(r["mean_rel_change"], 1e-16) for r in rows][:-1]
        mx = [max(r["max_rel_change"], 1e-16) for r in rows][:-1]
        ax.plot(x, mean, marker="o", color=f"C{k}", label=f"{name}: mean over phase")
        ax.plot(x, mx, ls="--", marker="o", mfc="none", color=f"C{k}", lw=0.8, label=f"{name}: maximum (contacts)")
    ax.set_yscale("log"); _step_axis(ax, dths)
    ax.axvline(2.0, color="0.5", lw=0.8); ax.axhline(1e-3, color="0.7", lw=0.7, ls=":")
    ax.set(xlabel=r"phase step $\Delta\gamma$ (deg)", ylabel=r"$|\Delta F_b|\,/\,F_{b,\rm out}$ vs $\Delta\gamma=%g^\circ$" % dths[-1])
    tx = ax.twinx()
    rows_a = study["dth"]["A"]
    tx.plot([r["value"] for r in rows_a], [r["wall_ms"] for r in rows_a], color="0.45", ls=":", marker=".", lw=0.9)
    tx.set_ylabel("wall time, System A (ms)", color="0.45", fontsize=7)
    tx.tick_params(axis="y", colors="0.45", labelsize=7); tx.set_yscale("log")
    h, l = ax.get_legend_handles_labels()
    h.append(tx.get_lines()[0]); l.append("wall time (right axis)")
    put_legend(la, handles=h, labels=l, ncol=2, fontsize=6.3)
    labels = {"A": "System A", "B": "System B", "A_extended": f"System A, $r={study['extended_r']:g}\\,R_\\odot$ emitter"}
    for name, rows in study["d2h"].items():
        x = [r["value"] for r in rows][:-1]; y = [max(r["max_rel_change"], 1e-16) for r in rows][:-1]
        bx.plot(x, y, marker="s", label=labels.get(name, name))
    bx.set_yscale("log"); _step_axis(bx, [v for v in study["d2hs"]])
    bx.axvline(6.0, color="0.5", lw=0.8); bx.axhline(1e-3, color="0.7", lw=0.7, ls=":")
    bx.set(xlabel=r"sector size $\Delta\theta$ (deg)", ylabel=r"max $|\Delta F_b|\,/\,F_{b,\rm out}$ vs $\Delta\theta=%g^\circ$" % study["d2hs"][-1])
    label_panels(axes)
    put_legend(lb, bx, ncol=2, fontsize=6.5)
    return fig


# ----------------------------------------------------------------------------
# Figure 5: per-cell versus mean-column attenuation
# ----------------------------------------------------------------------------

def table_flux_of_column(band: str, column_1e22: np.ndarray) -> np.ndarray:
    """F(N) from the band's table, log-log interpolation with the kernel's clamping."""
    ctx = K._build_flux_context(require_table(band), flux_type="erg")
    info = ctx["band_data"][band]
    lx = np.log10(np.clip(column_1e22, 1e-6, 1e6))
    lf = np.interp(lx, info["log_nh"], info["log_flux"], left=info["log_flux"][0])
    return 10.0 ** lf


def fig_percell(band: str = "broad", r: float = 6.0, name: str = "B") -> Tuple[plt.Figure, dict]:
    """Extended emitter (default: a 6 R_sun disk on System B's geometry, where the effect is a
    factor ~16; on System A it never exceeds 8 %): column maps, <F(N)> vs F(<N>), their ratio."""
    kw = system_kwargs(name, band, dth=1.0, r=r, d2h=3.0)      # finer sectors: a 6 R_sun disk spans many cells
    df = K.simulate_lightcurve(**kw)
    deg, phase = df["deg"].to_numpy(), df["phase"].to_numpy()
    F_cell = df[f"nfl_{band}"].to_numpy()
    F_mean = table_flux_of_column(band, df["fl"].to_numpy())
    F_mean[df["is_eclipsed"].to_numpy()] = 0.0
    # Pick three phases: out of eclipse, and two partial phases (visible fraction ~0.6 and ~0.15).
    vis_frac = df["A2"].to_numpy() / df["A2"].max()
    partial = np.where((vis_frac > 0.02) & (vis_frac < 0.98) & (deg > 0) & (deg < 90))[0]
    picks = [int(np.argmin(np.abs(deg - 0.0)))]
    for target in (0.6, 0.15):
        picks.append(int(partial[np.argmin(np.abs(vis_frac[partial] - target))]) if partial.size else picks[0])

    fig = plt.figure(figsize=(WIDTH, WIDTH * 1.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.25, 0.16])
    geo = {k: kw[k] for k in ("r", "R", "d1", "d2", "i0")}
    cells_kw = dict(geo, d2h=kw["d2h"], wind_model=kw["wind_model"], wind_params=kw["wind_params"], mdot=kw["mdot"],
                    v_inf=kw["v_inf"], mu_wind=kw["mu_wind"], f_opacity=kw["f_opacity"])
    all_cols = []
    maps = []
    for i, k in enumerate(picks):
        cells = K.emitter_cell_columns(deg[k], **cells_kw)
        maps.append(cells)
        all_cols.append(cells["column"][cells["visible"]])
    vmin, vmax = np.log10(np.concatenate(all_cols).min()), np.log10(np.concatenate(all_cols).max())
    for i, (k, cells) in enumerate(zip(picks, maps)):
        ax = fig.add_subplot(gs[0, i], projection="polar")
        vis = cells["visible"]
        sc = ax.scatter(cells["theta"][vis], cells["rho"][vis], c=np.log10(cells["column"][vis]), s=4,
                        cmap="viridis", vmin=vmin, vmax=vmax, lw=0)
        ax.scatter(cells["theta"][~vis], cells["rho"][~vis], c="0.8", s=4, lw=0)
        ax.set_yticks([]); ax.set_xticks([]); ax.set_ylim(0, r)
        ax.set_title(("(a)  " if i == 0 else "") + f"$\\phi={phase[k]:.3f}$\n{vis_frac[k]:.0%} visible", fontsize=7.5, pad=3)
    cbar = fig.colorbar(sc, ax=fig.axes[:3], orientation="horizontal", fraction=0.05, pad=0.08)
    cbar.set_label(r"$\log_{10} N_{\rm H}$ (10$^{22}$ cm$^{-2}$)")
    bx = fig.add_subplot(gs[1, :2])
    bx.plot(phase, F_cell, label=r"per cell: $\langle F(N_k)\rangle$")
    bx.plot(phase, F_mean, "--", label=r"mean column: $F(\langle N\rangle)$")
    for k in picks:
        bx.axvline(phase[k], color="0.7", lw=0.7)
    bx.set(xlabel="orbital phase", ylabel=r"$F_b$ (erg cm$^{-2}$ s$^{-1}$)", yscale="log", xlim=(0.25, 0.75),
           ylim=(1e-4 * F_cell.max(), 1.6 * F_cell.max()))
    lax = fig.add_subplot(gs[2, :2]); lax.axis("off"); put_legend(lax, bx, ncol=2)
    cx = fig.add_subplot(gs[1, 2])
    # The ratio is only meaningful where the mean-column flux is not itself
    # negligible: below 1e-3 of the out-of-eclipse flux it diverges as the mean
    # column crosses the opaque limit.
    floor = 1e-3 * F_cell.max()
    ok = (F_mean > floor) & (F_cell > 0)
    ratio = np.full(phase.size, np.nan); ratio[ok] = F_cell[ok] / F_mean[ok]
    cx.semilogy(phase, ratio)
    cx.set(xlabel="orbital phase", ylabel=r"$\langle F(N_k)\rangle / F(\langle N\rangle)$", xlim=(0.25, 0.75))
    cx.set_title(r"where $F(\langle N\rangle) > 10^{-3} F_{\rm out}$", loc="right", fontsize=6, color="0.35")
    label_panels([bx, cx], letters="bc")
    summary = {"system": name, "max_ratio": float(np.nanmax(ratio)), "phase_of_max": float(phase[int(np.nanargmax(ratio))]),
               "r": r, "d2h": kw["d2h"], "phases_shown": [float(phase[k]) for k in picks],
               "note": "ratio evaluated where F(<N>) > 1e-3 F_out"}
    save_json("percell", summary)
    return fig, summary


# ----------------------------------------------------------------------------
# Figure 6: energy dependence
# ----------------------------------------------------------------------------

def eclipse_width_half_depth(phase: np.ndarray, flux: np.ndarray) -> Tuple[float, float]:
    """(width in phase, depth) of the dip at half its depth: the contiguous run of points below
    the half-depth level that contains the minimum, measured on the curve rolled so that the
    minimum sits at phase 0.5 (a dip through phase 0 is therefore handled)."""
    phase = np.asarray(phase, dtype=float); flux = np.asarray(flux, dtype=float)
    f_out = float(np.max(flux)); f_min = float(np.min(flux))
    depth = 1.0 - f_min / f_out
    level = 0.5 * (f_out + f_min)
    k = int(np.argmin(flux))
    u = np.mod(phase - phase[k] + 0.5, 1.0)
    order = np.argsort(u); u, fl = u[order], flux[order]
    below = fl < level
    kk = int(np.argmin(fl))
    lo = kk
    while lo > 0 and below[lo - 1]:
        lo -= 1
    hi = kk
    while hi < fl.size - 1 and below[hi + 1]:
        hi += 1
    return float(u[hi] - u[lo]), depth


def fig_energy_dependence(dth: float = 1.0) -> Tuple[plt.Figure, dict]:
    tables = available_tables()
    bands = [b for b in ("soft", "medium", "hard", "broad") if b in tables]
    fig, (ax, bx), (la, lb) = panels(2, WIDTH * 0.4, legend_rows=2)
    axes = [ax, bx]
    summary = {}
    for name, marker in (("A", "o"), ("B", "s")):
        for band in bands:
            ph, fl = curve(name, band, dth)
            width, depth = eclipse_width_half_depth(ph, fl)
            lo, hi = BANDS[band]
            summary[f"{name}_{band}"] = {"width": width, "depth": depth, "emin": lo, "emax": hi}
            if name == "A":
                ax.plot(ph, fl / fl.max(), label=f"{band} ({lo:g}--{hi:g} keV)")
            bx.plot([math.sqrt(lo * hi)], [width], marker=marker, color=f"C{bands.index(band)}", ls="none")
    for name, marker in (("A", "o"), ("B", "s")):
        bx.plot([], [], marker=marker, color="k", ls="none", label=f"System {name}")
    ax.set(xlabel="orbital phase", ylabel=r"$F_b / F_{b,\rm out}$", xlim=(0.3, 0.7))
    bx.set(xlabel=r"band energy $\sqrt{E_{\min}E_{\max}}$ (keV)", ylabel="half-depth eclipse width (phase)", xscale="log")
    label_panels(axes)
    put_legend(la, ax, ncol=2); put_legend(lb, bx, ncol=2)
    if len(bands) < 2:
        bx.set_title("only one band table available", loc="right", fontsize=7, color="crimson")
    save_json("energy_dependence", summary)
    return fig, summary


# ----------------------------------------------------------------------------
# Figure 7 and Table: invariances
# ----------------------------------------------------------------------------

def _scaled_kwargs(kw: Dict, lam: float, scale_fopa: bool) -> Dict:
    """Lengths x lam (a via d1, d2; R; r; the profile's length scales), f_opa x lam when asked."""
    out = dict(kw)
    for key in ("d1", "d2", "R", "r"):
        out[key] = kw[key] * lam
    params = dict(kw["wind_params"])
    for key in ("Rb", "R_star", "H", "ell"):
        if key in params:
            params[key] = params[key] * lam
    out["wind_params"] = params
    if scale_fopa:
        out["f_opacity"] = kw["f_opacity"] * lam
    return out


def invariance_study(band: str = "broad", dth: float = 1.0, lams=(0.8, 2.0)) -> dict:
    """Max relative change of F_b(phi) under (i) q at fixed a, (ii) T_lambda, (iii) lengths x lambda
    with f_opa fixed (control), for the three profiles on System A's geometry."""
    out = {}
    base_sys = F.SYSTEMS["A"]
    for model in K.WIND_MODEL_IDS:
        params = K.default_wind_params(model, base_sys["R"])
        kw = system_kwargs("A", band, dth, wind_model=model, wind_params=params)
        ph, ref = K.simulate_band_flux(**kw)
        ok = ref > 1e-3 * ref.max()
        rel = lambda fl: float(np.max(np.abs(fl[ok] / ref[ok] - 1.0)))
        a = kw["d1"] + kw["d2"]
        row = {}
        q_lo = dict(kw, d1=a * 0.20, d2=a * 0.80); q_hi = dict(kw, d1=a * 0.95, d2=a * 0.05)
        row["q"] = max(rel(K.simulate_band_flux(**q_lo)[1]), rel(K.simulate_band_flux(**q_hi)[1]))
        for lam in lams:
            row[f"T_{lam:g}"] = rel(K.simulate_band_flux(**_scaled_kwargs(kw, lam, True))[1])
            row[f"control_{lam:g}"] = rel(K.simulate_band_flux(**_scaled_kwargs(kw, lam, False))[1])
        out[model] = row
        out[f"curves_{model}"] = {"phase": ph.tolist(), "ref": ref.tolist(),
                                  **{f"T_{lam:g}": K.simulate_band_flux(**_scaled_kwargs(kw, lam, True))[1].tolist() for lam in lams},
                                  **{f"control_{lam:g}": K.simulate_band_flux(**_scaled_kwargs(kw, lam, False))[1].tolist() for lam in lams}}
    save_json("invariance", {m: out[m] for m in K.WIND_MODEL_IDS})
    return out


def invariance_table(study: dict, lams=(0.8, 2.0)) -> str:
    rows = []
    short = {"smooth_pl": "PL", "confinement": "conf.", "beta_law": r"$\beta$"}
    def cells(key):
        return " / ".join(fmt(study[m][key], 2) for m in K.WIND_MODEL_IDS)
    rows.append([r"$q$: 0.20 $\to$ 0.95 at fixed $a$", " / ".join(short.values()), cells("q"), "0 (round-off)"])
    for lam in lams:
        rows.append([rf"$T_\lambda$, $\lambda={lam:g}$", " / ".join(short.values()), cells(f"T_{lam:g}"), "0 (round-off)"])
    for lam in lams:
        rows.append([rf"lengths $\times\lambda$, $f_{{\rm opa}}$ fixed (control), $\lambda={lam:g}$", "PL",
                     fmt(study["smooth_pl"][f"control_{lam:g}"], 2), r"$\mathcal{O}(1-\lambda)$"])
    return write_table("tab_invariance", rows, caption_note="Table: numerical verification of Propositions 1 and 2")


def fig_invariance(study: dict, model: str = "smooth_pl", lams=(0.8, 2.0)) -> plt.Figure:
    c = study[f"curves_{model}"]
    ph, ref = np.array(c["phase"]), np.array(c["ref"])
    lh = LEGEND_ROW_IN * 3 + 0.12
    fig = plt.figure(figsize=(WIDTH * 0.75, WIDTH * 0.62 + lh), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.0, lh / (WIDTH * 0.62) * 3.0])
    ax = fig.add_subplot(gs[0]); rx = fig.add_subplot(gs[1], sharex=ax); lax = fig.add_subplot(gs[2]); lax.axis("off")
    axes = [ax, rx]
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.plot(ph, ref, "k", lw=1.6, label=r"$\lambda=1$")
    for j, lam in enumerate(lams):
        t = np.array(c[f"T_{lam:g}"]); ctrl = np.array(c[f"control_{lam:g}"])
        ax.plot(ph, t, "--", color=f"C{j}", label=rf"$T_\lambda$ ($\lambda={lam:g}$): lengths and $f_{{\rm opa}}$ scaled")
        ax.plot(ph, ctrl, ":", color=f"C{j}", label=rf"control: lengths $\times{lam:g}$, $f_{{\rm opa}}$ fixed")
        ok = ref > 1e-3 * ref.max()
        rx.plot(ph[ok], (t[ok] / ref[ok] - 1.0) * 1e15, color=f"C{j}", lw=0.9)
    ax.set(ylabel=r"$F_b$ (erg cm$^{-2}$ s$^{-1}$)", yscale="log")
    rx.set(xlabel="orbital phase", ylabel=r"$T_\lambda$ residual ($\times 10^{15}$)")
    label_panels(axes)
    put_legend(lax, ax, ncol=2, fontsize=6.5)
    return fig


# ----------------------------------------------------------------------------
# MCMC fits: launching, loading, predictive curves
# ----------------------------------------------------------------------------

FIT_NAMES = ("A_fiducial", "ridge_broad", "ridge_tightR", "ridge_fopa_frozen")


def _inputs_digest(files: Sequence[str], settings: Sequence[str]) -> str:
    """Short digest of the input files' contents and the fit settings (paths made relative)."""
    h = hashlib.sha1()
    for path in files:
        h.update(os.path.basename(path).encode())
        with open(path, "rb") as fh:
            h.update(fh.read())
    for item in settings:
        h.update((os.path.relpath(item, ROOT) if os.path.exists(item) else item).encode())
        h.update(b"\0")
    return h.hexdigest()[:10]


def fit_config(name: str) -> dict:
    """Command line and cache directory of a named fit of System A's broad light curve.

    The cache directory name carries a digest of the flux table, the light-curve files and every
    fit option, so a regenerated table or data set, or a changed option, gets a fresh fit."""
    if name not in FIT_NAMES:
        raise ValueError(f"unknown fit {name!r}; choose from {FIT_NAMES}")
    sysA = F.SYSTEMS["A"]
    band = "broad"
    dth = 2.0
    table = require_table(band)
    ddir = data_dir("A", band)
    steps = ("32", "3000", "1000") if name.startswith("ridge") else ("32", "5000", "1000")
    opts = ["--band", band, "--flux-csv", table, "--data-dir", ddir, "--obs-column", "flux_t",
            "--time-column", "t_raw", "--counts-per-bin", "100", "--keep-zero-flux", "--kepler-mtot",
            "--orbital-period", f"{sysA['period_s']:g}", "--mdot", f"{sysA['mdot']:g}",
            "--v-inf", f"{sysA['v_inf']:g}", "--mu-wind", f"{sysA['mu_wind']:g}",
            "--fit-wind-shape", "--fit-scatter", "--likelihood", "jitter", "--dth", f"{dth:g}",
            "--n-walkers", steps[0], "--n-steps", steps[1], "--n-burn", steps[2], "--seed", "1",
            "--compute-bic", "--save-chi2", "--quiet", "--no-geometry-plots"]
    truth_fopa = math.log10(sysA["f_opacity"])
    if name == "A_fiducial":
        opts += ["--fit-fopacity"]
    elif name == "ridge_broad":
        opts += ["--fit-fopacity", "--prior-R", "2.5,2.0,1.0,8.0", f"--prior-fopa={truth_fopa:.4f},1.5,-4.0,0.5"]
    elif name == "ridge_tightR":
        opts += ["--fit-fopacity", "--prior-R", "2.5,0.1,1.0,8.0", f"--prior-fopa={truth_fopa:.4f},1.5,-4.0,0.5"]
    elif name == "ridge_fopa_frozen":
        opts += ["--prior-R", "2.5,2.0,1.0,8.0", "--freeze", f"log_fopa={truth_fopa:.6f}"]
    files = [table] + sorted(glob.glob(os.path.join(ddir, "*.txt"))) + sorted(glob.glob(os.path.join(ddir, "*.json")))
    out_dir = os.path.join(CACHE, f"{name}_{_inputs_digest(files, opts)}")
    cmd = [PY, "-m", "cloak.mcmc_fit"] + opts + ["--output-dir", out_dir]
    return {"name": name, "cmd": cmd, "out_dir": out_dir, "band": band, "wind_model": sysA["wind_model"],
            "dth": dth, "sim_params": {"mdot": sysA["mdot"], "v_inf": sysA["v_inf"], "mu_wind": sysA["mu_wind"]},
            "period_s": sysA["period_s"], "chain": os.path.join(out_dir, f"{band}_{sysA['wind_model']}_chain.npz")}


def ensure_fit(name: str, verbose: bool = True) -> dict:
    """Run the named fit unless a chain for exactly these inputs exists; returns its config."""
    cfg = fit_config(name)
    if os.path.exists(cfg["chain"]):
        if verbose:
            print(f"{name}: using cached {os.path.relpath(cfg['chain'], ROOT)}")
        return cfg
    os.makedirs(cfg["out_dir"], exist_ok=True)
    log = os.path.join(cfg["out_dir"], "fit.log")
    print(f"{name}: running; log {os.path.relpath(log, ROOT)} (about 20-30 min)")
    with open(log, "w") as fh:
        res = subprocess.run(cfg["cmd"], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"fit {name} failed (rc={res.returncode}); see {log}")
    return cfg


def load_fit(cfg: dict) -> dict:
    """Chain, flat samples, MAP, the ParamSpec, the model wrapper and the binned data of a fit."""
    meta = np.load(cfg["chain"], allow_pickle=False)
    names = [str(n) for n in meta["param_names"]]
    frozen = {str(k): float(v) for k, v in zip(meta["frozen_names"], meta["frozen_values"])}
    chain, log_prob = meta["chain"], meta["log_prob"]
    samples, lp = chain.reshape(-1, len(names)), log_prob.reshape(-1)
    shape_names = set(M.WIND_SHAPE_FIT[str(meta["wind_model"])])
    spec = M.build_param_spec(likelihood=str(meta["likelihood"]), mode=str(meta["mode"]), wind_model=str(meta["wind_model"]),
                              fit_wind_shape=bool(shape_names & (set(names) | set(frozen))),
                              fit_scatter=("f_scatter" in names) or ("f_scatter" in frozen),
                              fit_fopacity=("log_fopa" in names) or ("log_fopa" in frozen),
                              frozen=frozen, orbital_period_s=float(meta["orbital_period_s"]))
    model = M.DirectLightCurveModel(band=cfg["band"], flux_csv_path=require_table(cfg["band"]),
                                    wind_model=cfg["wind_model"], dth=cfg["dth"], sim_params=cfg["sim_params"])
    obs = U.load_observed_lightcurves(cfg["band"], data_dir("A", cfg["band"]), flux_column="flux_t",
                                      time_column="t_raw", drop_nonpositive_flux=False, period=cfg["period_s"])
    binned = U.phase_bin_data_snr(obs, counts_per_bin=100, counts_column="counts", rate_column="flux",
                                  error_column="flux_err", verbose=False)
    err = U.sanitize_errors(binned["flux_err"].to_numpy(), context="binned: ")
    data = M.FitData.build(binned["phase"].to_numpy(), binned["flux"].to_numpy(), err, fit_phase_shift=True,
                           n_model=int(round(360.0 / cfg["dth"])), is_binned=True,
                           phase_width=binned["width"].to_numpy())
    i_map = int(np.argmax(lp))
    return {"cfg": cfg, "names": names, "labels": spec.active_labels, "frozen": frozen, "chain": chain,
            "samples": samples, "log_prob": lp, "theta_map": samples[i_map], "spec": spec, "model": model,
            "data": data, "binned": binned}


def predictive_curves(fit: dict, n_draws: int = 200, seed: int = 0, band: Optional[str] = None,
                      table: Optional[str] = None) -> dict:
    """MAP curve and the 16/84 % band of posterior model curves on a fine phase grid, in the data's
    phase (each draw is shifted by the MAP's profiled phase offset). *band*/*table* switch the model
    to another energy band for cross-band prediction; the fitted scattered floor, a flux of the
    fitted band, is then rescaled by the ratio of the two bands' out-of-eclipse fluxes (a
    band-independent scattered fraction, which is how the generator defines it)."""
    spec, data = fit["spec"], fit["data"]
    cross = band is not None and band != fit["cfg"]["band"]
    model = fit["model"] if not cross else M.DirectLightCurveModel(
        band=band, flux_csv_path=table or require_table(band), wind_model=fit["cfg"]["wind_model"],
        dth=fit["cfg"]["dth"], sim_params=fit["cfg"]["sim_params"])
    _, shift_map = M.aligned_model_flux(fit["theta_map"], spec, fit["model"], data)

    def curve_for(theta):
        c = M.model_curve(theta, spec, model)
        if c is None:
            return None
        phase, flux = c
        if cross:
            f_sc = spec.f_scatter(theta)
            c_fit = M.model_curve(theta, spec, fit["model"])
            if c_fit is None:
                return None
            fit_out, band_out = float(np.max(c_fit[1])) - f_sc, float(np.max(flux)) - f_sc
            ratio = band_out / fit_out if fit_out > 0 else 1.0
            flux = flux - f_sc + f_sc * ratio
        return phase, flux
    grid = np.linspace(0.0, 1.0, 721)
    rng = np.random.default_rng(seed)
    idx = rng.choice(fit["samples"].shape[0], size=min(n_draws, fit["samples"].shape[0]), replace=False)
    curves = []
    for i in idx:
        c = curve_for(fit["samples"][i])
        if c is not None:
            curves.append(U.eval_periodic(*U.periodic_model(*c), grid, shift=shift_map))
    curves = np.array(curves)
    c_map = curve_for(fit["theta_map"])
    map_curve = U.eval_periodic(*U.periodic_model(*c_map), grid, shift=shift_map)
    map_at_obs = U.eval_periodic(*U.periodic_model(*c_map), data.phase, shift=shift_map)
    return {"grid": grid, "map": map_curve, "map_at_obs": map_at_obs, "lo": np.percentile(curves, 16, axis=0),
            "hi": np.percentile(curves, 84, axis=0), "shift": shift_map, "n_draws": int(curves.shape[0])}


def truth_for_names(names: Sequence[str], system: str = "A", band: str = "broad") -> Dict[str, Optional[float]]:
    """Injected value of every sampled parameter (None where the truth is not defined)."""
    s = F.SYSTEMS[system]
    g = F.geometry(s)
    truth_json = load_truth(system, band) or {}
    obs = s["observation"]
    a = g["d1"] + g["d2"]
    values = {"M_tot": F.total_mass(a, s["period_s"]), "q_m": s["q_m"], "R": g["R"], "r": g["r"], "i0": g["i0"],
              "a": a, "d1": g["d1"], "d2": g["d2"], "log_fopa": math.log10(s["f_opacity"]),
              "f_scatter": truth_json.get("scatter"),
              "log_f": math.log(obs["intrinsic_scatter"]) if obs.get("intrinsic_scatter", 0) > 0 else None}
    values.update({k: float(v) for k, v in s["wind_params"].items()})
    return {n: values.get(n) for n in names}


def fig_injection(fit: dict, pred: dict) -> Tuple[plt.Figure, dict]:
    data, binned = fit["data"], fit["binned"]
    lh = LEGEND_ROW_IN + 0.12
    fig = plt.figure(figsize=(WIDTH, WIDTH * 0.62 + lh), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.0, 4.0 * lh / (WIDTH * 0.62)])
    ax = fig.add_subplot(gs[0]); rx = fig.add_subplot(gs[1], sharex=ax); lax = fig.add_subplot(gs[2]); lax.axis("off")
    resid2 = (data.flux - pred["map_at_obs"]) ** 2
    chi2 = float(np.sum(resid2 / data.err2))
    dof = M.degrees_of_freedom(fit["spec"], data.flux.size, True)
    # Under the jitter likelihood the fitted intrinsic scatter is part of the model's variance:
    # bars, residuals and the quoted chi2 use sigma_eff^2 = sigma^2 + (f m)^2 at the MAP.
    j_f = fit["spec"].index("log_f")
    if j_f is not None:
        f_map = float(np.exp(fit["theta_map"][j_f]))
        sigma = np.sqrt(data.err2 + (f_map * pred["map_at_obs"]) ** 2)
        chi2_used = float(np.sum(resid2 / sigma ** 2))
        obs_label = "synthetic bins (bars include the fitted intrinsic scatter)"
    else:
        f_map, sigma, chi2_used, obs_label = None, data.err, chi2, "synthetic bins"
    plot_lightcurve_fit(data.phase, data.flux, sigma, model_phase=pred["grid"], model_flux=pred["map"],
                        obs_model=pred["map_at_obs"], obs_phase_width=binned["width"].to_numpy(), band=fit["cfg"]["band"],
                        red_chi2=chi2_used / dof, ax=ax, ax_res=rx, model_label="MAP model", obs_label=obs_label)
    ax.fill_between(pred["grid"], pred["lo"], pred["hi"], color="C1", alpha=0.25, lw=0, label="68% band of posterior model curves")
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    put_legend(lax, ax, ncol=3, fontsize=6.5)
    summary = {"chi2_measurement_errors": chi2, "chi2_effective": chi2_used, "dof": int(dof), "f_map": f_map,
               "shift_map": pred["shift"], "n_bins": int(data.flux.size)}
    return fig, summary


def fig_corner(fit: dict, truths: Dict[str, Optional[float]]) -> plt.Figure:
    import corner
    t = [truths.get(n) for n in fit["names"]]
    fig = corner.corner(fit["samples"], labels=fit["labels"], truths=[np.nan if v is None else v for v in t],
                        truth_color="crimson", quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3g",
                        title_kwargs={"fontsize": 6.5}, label_kwargs={"fontsize": 7}, max_n_ticks=3,
                        use_math_text=True, labelpad=0.08)
    fig.set_size_inches(WIDTH * 1.2, WIDTH * 1.2)
    for ax in fig.axes:
        ax.tick_params(labelsize=5.5)
    if "log_f" in fit["names"] and truths.get("log_f") is not None:
        fig.text(0.62, 0.97, r"red lines: injected values; for $\ln f$ the reference is $\ln\epsilon$ of the"
                 "\ninjected variability, an approximate correspondence", fontsize=7, ha="left", va="top")
    return fig


def recovery_table(fit: dict, truths: Dict[str, Optional[float]]) -> Tuple[str, List[dict]]:
    rows, records = [], []
    for j, (name, label) in enumerate(zip(fit["names"], fit["labels"])):
        s = fit["samples"][:, j]
        lo, med, hi = np.percentile(s, [16, 50, 84])
        tv = truths.get(name)
        inside = None if tv is None else bool(lo <= tv <= hi)
        note = "" if tv is not None else "no injected value"
        if name == "q_m":
            note = r"posterior $\equiv$ prior"
        if name == "M_tot":
            note = "prior-anchored via $R$"
        rows.append([label, "--" if tv is None else fmt(tv), f"{fmt(med)}$^{{+{fmt(hi - med, 2)}}}_{{-{fmt(med - lo, 2)}}}$",
                     ("yes" if inside else "no") if inside is not None else "--", note])
        records.append({"param": name, "truth": tv, "median": float(med), "p16": float(lo), "p84": float(hi), "inside68": inside})
    path = write_table("tab_recovery", rows, header=["Parameter", "Injected", "Median (68\\%)", "Truth in 68\\%?", "Note"],
                       caption_note="Table: injection-recovery summary for System A")
    save_json("recovery", records)
    return path, records


# ----------------------------------------------------------------------------
# Figure 10: simulation-based calibration
# ----------------------------------------------------------------------------

def fig_sbc(alpha: float = 0.05) -> Tuple[plt.Figure, dict]:
    path = os.path.join(RESULTS, "sbc_ranks.csv")
    labels = {"M_tot": r"$M_{\rm tot}$", "R": "$R$", "r": "$r$", "i0": "$i$", "Rb": r"$R_{\rm b}$", "p": "$p$",
              "log_fopa": r"$\log_{10} f_{\rm opa}$"}
    if not os.path.exists(path):
        fig, ax = plt.subplots(figsize=(WIDTH, WIDTH * 0.3)); ax.axis("off")
        ax.text(0.5, 0.5, "no SBC ranks yet: run  python figures/run_sbc.py --n-draws 100", ha="center", va="center")
        return fig, {"n_draws": 0}
    ranks = pd.read_csv(path)
    params = [p for p in labels if p in set(ranks["param"])]
    n_draws = ranks.groupby("param")["draw"].nunique().min()
    eps = math.sqrt(math.log(2.0 / alpha) / (2.0 * n_draws))            # DKW band
    ncol = 4
    nrow = int(math.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(WIDTH, WIDTH * 0.3 * nrow), sharex=True, sharey=True)
    axes = np.ravel(axes)
    summary = {"n_draws": int(n_draws), "dkw_eps": eps}
    for ax, p in zip(axes, params):
        sub = ranks[ranks["param"] == p]
        u = np.sort((sub["rank"].to_numpy() + 0.5) / (sub["n_post"].to_numpy() + 1.0))
        ecdf = np.arange(1, u.size + 1) / u.size
        grid = np.linspace(0, 1, 201)
        diff = np.interp(grid, u, ecdf, left=0.0, right=1.0) - grid
        ax.fill_between(grid, -eps, eps, color="0.85", lw=0)
        ax.plot(grid, diff, color="C0")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(labels[p], fontsize=8)
        summary[p] = {"max_abs_ecdf_diff": float(np.max(np.abs(diff))), "outside_band": bool(np.max(np.abs(diff)) > eps)}
    for ax in axes[len(params):]:
        ax.axis("off")
    for ax in axes[-ncol:]:
        ax.set_xlabel("rank / $L$")
    axes[0].set_ylabel("ECDF $-$ uniform")
    fig.suptitle(f"SBC, {n_draws} prior draws; grey: {100 * (1 - alpha):.0f}% DKW band", fontsize=8)
    fig.tight_layout()
    save_json("sbc_summary", summary)
    return fig, summary


# ----------------------------------------------------------------------------
# Figure 11: the scale ridge
# ----------------------------------------------------------------------------

def fig_ridge(fits: Dict[str, dict], truths: Dict[str, float]) -> plt.Figure:
    import corner
    titles = {"ridge_broad": r"(a) broad priors on $R$ and $f_{\rm opa}$", "ridge_tightR": r"(b) tight prior on $R$",
              "ridge_fopa_frozen": r"(c) $f_{\rm opa}$ frozen"}
    fig, axes, (lax,) = panels(3, WIDTH * 0.34, legend_rows=1, span_legend=True)
    for ax in axes[1:]:
        ax.sharey(axes[0]); plt.setp(ax.get_yticklabels(), visible=False)
    m_true, f_true = truths["M_tot"], truths["log_fopa"]
    for ax, name in zip(axes, ("ridge_broad", "ridge_tightR", "ridge_fopa_frozen")):
        fit = fits.get(name)
        ax.set_title(titles[name], fontsize=8)
        if fit is None:
            ax.text(0.5, 0.5, "fit not available", ha="center", va="center", transform=ax.transAxes)
            continue
        jm = fit["names"].index("M_tot")
        m = fit["samples"][:, jm]
        if "log_fopa" in fit["names"]:
            f = fit["samples"][:, fit["names"].index("log_fopa")]
            corner.hist2d(m, f, ax=ax, bins=40, levels=(0.393, 0.865), plot_datapoints=False, smooth=1.0,
                          color="C0", fill_contours=True)
        else:
            f0 = fit["frozen"]["log_fopa"]
            lo68, hi68 = np.percentile(m, [16, 84]); lo95, hi95 = np.percentile(m, [2.5, 97.5])
            ax.plot([lo95, hi95], [f0, f0], color="C0", lw=2, alpha=0.5)
            ax.plot([lo68, hi68], [f0, f0], color="C0", lw=5, label=r"$M_{\rm tot}$ 68 % (thick) / 95 % (thin)")
        mm = np.linspace(max(1.0, 0.3 * m_true), 3.5 * m_true, 200)
        ax.plot(mm, f_true + (1.0 / 3.0) * np.log10(mm / m_true), "k--", lw=1.0, label=r"flat direction $f_{\rm opa}\propto M_{\rm tot}^{1/3}$")
        ax.plot([m_true], [f_true], marker="*", ms=10, color="crimson", ls="none", label="injected")
        ax.set_xlabel(r"$M_{\rm tot}$ ($M_\odot$)")
    axes[0].set_ylabel(r"$\log_{10} f_{\rm opa}$")
    handles, labels = merged_handles(axes)
    put_legend(lax, handles=handles, labels=labels, ncol=3, fontsize=6.5)
    return fig


# ----------------------------------------------------------------------------
# Figure 12: the profiled phase shift
# ----------------------------------------------------------------------------

def fig_shift_profile(fit: Optional[dict] = None, seed: int = 3) -> Tuple[plt.Figure, dict]:
    """chi2 as a function of the trial shift (brute force) with the coarse grid and the two dense
    passes of best_phase_shift marked. Uses the fiducial fit's binned data and MAP model when
    available, else a noisy realization of the System A model."""
    if fit is not None:
        data = fit["data"]
        c = M.model_curve(fit["theta_map"], fit["spec"], fit["model"])
        pe, fe = U.periodic_model(*c)
        obs_phase, obs_flux, err2 = data.phase, data.flux, data.err2
        n_model = int(round(360.0 / fit["cfg"]["dth"]))
    else:
        ph, fl = curve("A", "broad", 2.0)
        pe, fe = U.periodic_model(ph, fl)
        rng = np.random.default_rng(seed)
        obs_phase = np.sort(rng.uniform(0, 1, 150))
        true_shift = 0.02
        sig = 0.05 * fl.max()
        obs_flux = U.eval_periodic(pe, fe, obs_phase, shift=true_shift) + rng.normal(0, sig, obs_phase.size)
        err2 = np.full(obs_phase.size, sig ** 2)
        n_model = 180
    search = U.build_phase_shift_search(obs_phase, n_model=n_model)
    _, best_shift, best_chi2 = U.best_phase_shift(pe, fe, obs_flux, err2, search)
    grid = np.linspace(0, 1, 20001)[:-1]
    chi2 = np.array([np.sum((obs_flux - U.eval_periodic(pe, fe, obs_phase, shift=s)) ** 2 / err2) for s in grid])
    coarse = np.array([np.sum((obs_flux - U.eval_periodic(pe, fe, obs_phase, shift=s)) ** 2 / err2) for s in search.shift_grid])
    fig, (ax, bx), (la, lb) = panels(2, WIDTH * 0.38, legend_rows=2)
    ax.plot(grid, chi2, color="0.3", lw=0.8, label="brute force (20 000 shifts)")
    ax.plot(search.shift_grid, coarse, "o", ms=2.5, color="C0", label=f"coarse grid ({search.shift_grid.size})")
    ax.plot([best_shift], [best_chi2], "*", ms=10, color="crimson", label="search result")
    ax.set(xlabel="trial phase shift", ylabel=r"$\chi^2$", yscale="log")
    j = int(np.argmin(chi2)); half = 1.0 / search.shift_grid.size
    sel = np.abs(((grid - grid[j] + 0.5) % 1.0) - 0.5) < 1.5 * half
    bx.plot(grid[sel], chi2[sel], color="0.3", lw=0.8)
    near = np.abs(((search.shift_grid - grid[j] + 0.5) % 1.0) - 0.5) < 1.5 * half
    bx.plot(search.shift_grid[near], coarse[near], "o", ms=3, color="C0")
    bx.plot([best_shift], [best_chi2], "*", ms=10, color="crimson")
    bx.axvline(grid[j], color="0.6", lw=0.7)
    bx.set(xlabel="trial phase shift (zoom)", ylabel=r"$\chi^2$")
    label_panels([ax, bx])
    put_legend(la, ax, ncol=2, fontsize=6.5)
    summary = {"search_shift": best_shift, "search_chi2": best_chi2, "brute_shift": float(grid[j]),
               "brute_chi2": float(chi2[j]), "search_minus_brute_chi2": float(best_chi2 - chi2[j]),
               "brute_grid_step": float(grid[1] - grid[0]), "search_resolution": search.resolution,
               "note": "a negative difference means the search's finer final step found a lower chi2 than the brute-force grid"}
    save_json("shift_profile", summary)
    return fig, summary


# ----------------------------------------------------------------------------
# Figure 13: binning estimators
# ----------------------------------------------------------------------------

def binning_bias_study(lams=(0.5, 1, 2, 3, 5, 10, 20, 50), n_rows: int = 45, trials: int = 3000, seed: int = 1) -> dict:
    """Bias of three bin estimators on Poisson rows with a third of the rows partially exposed."""
    rng = np.random.default_rng(seed)
    dt, c = 100.0, 1e-13
    out = {"lam": list(lams), "inv_var": [], "exposure": [], "drop_zero_mean": [], "inv_var_err_ratio": [], "exposure_err_ratio": []}
    for lam in lams:
        iv, ex, dz, iv_e, ex_e = [], [], [], [], []
        for _ in range(trials):
            t = np.full(n_rows, dt); t[: n_rows // 3] = dt * rng.uniform(0.3, 1.0, n_rows // 3)
            N = rng.poisson(lam * t / dt)
            v = N / t * c; e = np.sqrt(N) / t * c
            e_fix = e.copy(); e_fix[e_fix <= 0] = np.median(e_fix[e_fix > 0]) if np.any(e_fix > 0) else 1.0
            w = 1 / e_fix ** 2
            iv.append(np.sum(w * v) / np.sum(w)); iv_e.append(np.sqrt(1 / np.sum(w)))
            val, err = U.bin_estimate(v, e, t)
            ex.append(val); ex_e.append(err)
            keep = N > 0; dz.append(np.mean(v[keep]) if keep.any() else np.nan)
        truth = lam / dt * c
        out["inv_var"].append(float(np.mean(iv) / truth - 1)); out["exposure"].append(float(np.mean(ex) / truth - 1))
        out["drop_zero_mean"].append(float(np.nanmean(dz) / truth - 1))
        out["inv_var_err_ratio"].append(float(np.mean(iv_e) / np.std(iv))); out["exposure_err_ratio"].append(float(np.mean(ex_e) / np.std(ex)))
    save_json("binning_bias", out)
    return out


def fig_binning_bias(study: dict) -> plt.Figure:
    lam = np.array(study["lam"])
    fig, (ax, bx), (la, lb) = panels(2, WIDTH * 0.38, legend_rows=3)
    ax.semilogx(lam, 100 * np.array(study["inv_var"]), "o-", label=r"inverse-variance, $\sigma_n = \sqrt{N_n}$")
    ax.semilogx(lam, 100 * np.array(study["drop_zero_mean"]), "s-", label="plain mean, zero-count rows dropped")
    ax.semilogx(lam, 100 * np.array(study["exposure"]), "^-", label="exposure-weighted (adopted)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel=r"mean counts per row $\lambda$", ylabel="bias of the bin mean (%)", ylim=(-35, 60))
    dz = 100 * np.array(study["drop_zero_mean"])
    off = dz > 60
    if off.any():
        ax.annotate(f"+{dz[off].max():.0f} % at $\\lambda={lam[off][0]:g}$ (off scale)", xy=(lam[off][-1], 58),
                    xytext=(lam[off][-1] * 1.8, 48), fontsize=6, arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4"))
    bx.semilogx(lam, study["inv_var_err_ratio"], "o-", label="inverse-variance")
    bx.semilogx(lam, study["exposure_err_ratio"], "^-", label="exposure-weighted")
    bx.axhline(1, color="k", lw=0.6)
    bx.set(xlabel=r"mean counts per row $\lambda$", ylabel="reported error / actual scatter")
    label_panels([ax, bx])
    put_legend(la, ax, ncol=1, fontsize=6.5); put_legend(lb, bx, ncol=1, fontsize=6.5)
    return fig


# ----------------------------------------------------------------------------
# Figure 14: cross-band prediction
# ----------------------------------------------------------------------------

def fig_crossband(fit: dict, bands: Sequence[str] = ("soft", "medium", "hard"), n_draws: int = 100) -> Tuple[plt.Figure, dict]:
    """Predict the other bands' light curves from the broad-band posterior; overlay their synthetic bins."""
    tables = available_tables()
    bands = [b for b in bands if b in tables and os.path.isdir(data_dir("A", b))]
    fig, axes_list, laxes = panels(max(1, len(bands)), WIDTH * 0.4, legend_rows=2)
    axes = np.array([axes_list])
    summary = {}
    if not bands:
        axes[0, 0].text(0.5, 0.5, "no other-band tables / synthetic data available", ha="center", va="center")
        axes[0, 0].axis("off")
        return fig, summary
    for ax, lax, band in zip(axes[0], laxes, bands):
        pred = predictive_curves(fit, n_draws=n_draws, band=band)
        obs = U.load_observed_lightcurves(band, data_dir("A", band), flux_column="flux_t", time_column="t_raw",
                                          drop_nonpositive_flux=False, period=fit["cfg"]["period_s"])
        binned = U.phase_bin_data_snr(obs, counts_per_bin=100, counts_column="counts", rate_column="flux",
                                      error_column="flux_err", verbose=False)
        err = U.sanitize_errors(binned["flux_err"].to_numpy(), context="binned: ")
        ax.errorbar(binned["phase"], binned["flux"], yerr=err, fmt=".", ms=3, color="0.4", lw=0.6, label=f"{band} synthetic bins")
        ax.fill_between(pred["grid"], pred["lo"], pred["hi"], color="C1", alpha=0.3, lw=0, label="68% predicted from broad fit")
        ax.plot(pred["grid"], pred["map"], color="C1", lw=1.0)
        model_at = U.eval_periodic(pred["grid"], pred["map"], binned["phase"].to_numpy())
        chi2 = float(np.sum((binned["flux"].to_numpy() - model_at) ** 2 / err ** 2))
        summary[band] = {"chi2": chi2, "n_bins": int(binned.shape[0])}
        ax.set(xlabel="orbital phase", ylabel=r"$F_b$ (erg cm$^{-2}$ s$^{-1}$)")
        ax.set_title(f"{band} band, $\\chi^2/n = {chi2 / binned.shape[0]:.2f}$", loc="right", fontsize=7, color="0.35")
        put_legend(lax, ax, ncol=1, fontsize=6.5)
    label_panels(axes[0])
    save_json("crossband", summary)
    return fig, summary


# ----------------------------------------------------------------------------
# Performance table
# ----------------------------------------------------------------------------

def performance_table(fit: Optional[dict] = None, repeat: int = 20) -> Tuple[str, dict]:
    import numba
    kw1 = system_kwargs("A", "broad", dth=1.0); kw2 = system_kwargs("A", "broad", dth=2.0)

    def bench(fn, n=repeat):
        fn()
        t = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
        return 1e3 * float(np.median(t))
    n_threads = numba.get_num_threads()
    res = {"threads": n_threads, "cpu": platform.processor() or platform.machine(), "python": platform.python_version(),
           "numba": numba.__version__, "numpy": np.__version__, "machine": platform.platform()}
    res["forward_dth1_threads"] = bench(lambda: K.simulate_band_flux(**kw1))
    res["forward_dth2_threads"] = bench(lambda: K.simulate_band_flux(**kw2))
    numba.set_num_threads(1)
    res["forward_dth1_single"] = bench(lambda: K.simulate_band_flux(**kw1))
    numba.set_num_threads(n_threads)
    ph, fl = K.simulate_band_flux(**kw2)
    pe, fe = U.periodic_model(ph, fl)
    if fit is not None:
        data = fit["data"]
        res["shift_search"] = bench(lambda: U.best_phase_shift(pe, fe, data.flux, data.err2, data.shift_search))
        theta = fit["theta_map"]
        priors = M.get_active_priors(fit["spec"], M.default_geometry_priors(fit["spec"].mode), {}, None)
        res["log_posterior_dth2"] = bench(lambda: M.log_probability(theta, fit["spec"], priors, fit["model"], data), n=10)
        res["n_bins"] = int(data.flux.size)
    rows = [["forward model, single thread", f"{res['forward_dth1_single']:.1f} ms", r"$\Delta\gamma=1^\circ$, $\Delta\theta=6^\circ$, $n_\rho=10$"],
            [f"forward model, {n_threads} threads", f"{res['forward_dth1_threads']:.1f} ms / {res['forward_dth2_threads']:.1f} ms",
             r"$\Delta\gamma=1^\circ$ / $2^\circ$"]]
    if fit is not None:
        rows.append(["phase-offset search alone", f"{res['shift_search']:.2f} ms", f"{res['n_bins']} bins, automatic $n_c$, two 33-point refinements"])
        rows.append(["log-posterior evaluation (incl.\\ phase profiling)", f"{res['log_posterior_dth2']:.1f} ms", r"$\Delta\gamma=2^\circ$"])
    path = write_table("tab_performance", rows, caption_note=f"Table: wall-clock performance on {res['machine']}, Python {res['python']}, numba {res['numba']}")
    save_json("performance", res)
    return path, res
