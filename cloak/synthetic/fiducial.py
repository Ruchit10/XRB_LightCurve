"""Fiducial synthetic systems of the CLOAK methods paper.

Two generic configurations (Table "Fiducial synthetic systems" of the paper),
deliberately not the IC 10 X-1 values: a compact WR-like binary and an
OB-supergiant-like binary, both inclined enough for a total eclipse. The data
notebook (``synthetic_data/generate_synthetic_data.ipynb``) turns them into
synthetic light curves; the figure notebook (``figures/paper_figures.ipynb``)
reads the same definitions, so the two can never disagree.

Every entry maps directly onto ``cloak.kernel.simulate_band_flux`` keywords
except ``period_s`` (the fold and Kepler period), ``q_m`` (companion mass
fraction, which fixes how ``a`` is split into ``d1``/``d2``), and the
observation settings under ``observation``.
"""
from __future__ import annotations

import glob
import math
import os
from typing import Dict

G_SI = 6.674e-11
M_SUN_KG = 1.989e30
R_SUN_M = 6.957e8


def kepler_prefactor(period_s: float) -> float:
    """K in a = K * (M_tot / M_sun)^(1/3), a in solar radii (same law as the fitter)."""
    return ((G_SI * M_SUN_KG * float(period_s) ** 2) / (4.0 * math.pi ** 2)) ** (1.0 / 3.0) / R_SUN_M


def total_mass(a_rsun: float, period_s: float) -> float:
    """M_tot (solar masses) of a binary with separation *a_rsun* and period *period_s*."""
    return (float(a_rsun) / kepler_prefactor(period_s)) ** 3


SYSTEMS: Dict[str, Dict] = {
    "A": {
        "label": "System A (WR-like)",
        "period_s": 1.6 * 86400.0,
        "a": 17.0, "q_m": 0.6,            # d1 = a q_m (compact object), d2 = a (1 - q_m)
        "R": 2.5, "r": 0.001, "i0": 84.0,
        "wind_model": "smooth_pl", "wind_params": {"Rb": 6.0, "p": 4.0, "Delta": 2.0},
        "mdot": 1.0e-5, "v_inf": 1750.0, "mu_wind": 1.4, "f_opacity": 0.05,
        "observation": {
            "scatter_fraction": 0.03,       # floor as a fraction of the out-of-eclipse flux
            "target_rate": 0.5,             # out-of-eclipse count rate (cts/s) -> flux_per_rate
            "dt": 100.0,
            "visit_starts_orbits": [0.0, 3.3, 7.6, 12.9, 19.2], "visit_duration_s": 40e3,
            "gap_fraction": 0.10, "gap_duration": 3000.0,
            "phase_shift": 0.02, "intrinsic_scatter": 0.10, "seed": 11,
        },
    },
    "B": {
        "label": "System B (OB-like)",
        "period_s": 9.0 * 86400.0,
        "a": 60.0, "q_m": 0.7,
        "R": 18.0, "r": 0.001, "i0": 78.0,
        "wind_model": "beta_law", "wind_params": {"beta": 0.8, "H": 2.0},   # R_star tied to R
        "mdot": 1.0e-6, "v_inf": 1500.0, "mu_wind": 1.4, "f_opacity": 1.0,
        "observation": {
            "scatter_fraction": 0.03,
            "target_rate": 2.0,
            "dt": 100.0,
            "visit_starts_orbits": [0.0, 2.4, 5.1, 8.7, 13.3, 17.6], "visit_duration_s": 60e3,
            "gap_fraction": 0.10, "gap_duration": 3000.0,
            "phase_shift": 0.0, "intrinsic_scatter": 0.0, "seed": 22,
        },
    },
}


def available_tables(data_dir: str, bands) -> Dict[str, str]:
    """Band -> flux-vs-nH CSV under *data_dir*. The generic tables written by the data notebook
    (``tables/flux_vs_nH_<band>.csv``) are used exclusively as soon as any exists; only without them
    do the tracked example table(s) ``flux_vs_nH_tbabs_<band>.csv`` serve, so tables from two
    different spectra are never mixed. Shared by the data notebook and the figure code."""
    generic = {os.path.basename(p)[len("flux_vs_nH_"):-4]: p
               for p in sorted(glob.glob(os.path.join(data_dir, "tables", "flux_vs_nH_*.csv")))}
    generic = {b: p for b, p in generic.items() if b in bands}
    if generic:
        return generic
    return {b: os.path.join(data_dir, f"flux_vs_nH_tbabs_{b}.csv") for b in bands
            if os.path.exists(os.path.join(data_dir, f"flux_vs_nH_tbabs_{b}.csv"))}


def visits(system: Dict):
    """Observing visits as (start, duration) in seconds after the reference epoch: the starts are
    given in orbits so they follow the system's period."""
    obs = system["observation"]
    return [(float(n) * float(system["period_s"]), float(obs["visit_duration_s"])) for n in obs["visit_starts_orbits"]]


def visits_arg(system: Dict) -> str:
    """The same visits as the generator's ``--visits start:duration,...`` argument."""
    return ",".join(f"{s:.0f}:{d:.0f}" for s, d in visits(system))


def geometry(system: Dict) -> Dict[str, float]:
    """d1, d2, r, R, i0 of a fiducial system (a split by the companion mass fraction)."""
    a, q = float(system["a"]), float(system["q_m"])
    return {"d1": a * q, "d2": a * (1.0 - q), "r": float(system["r"]), "R": float(system["R"]),
            "i0": float(system["i0"])}


def simulation_kwargs(system: Dict) -> Dict:
    """Keywords for ``cloak.kernel.simulate_band_flux`` / ``simulate_lightcurve``
    (add ``flux_csv_path`` and ``band``)."""
    params = dict(system["wind_params"])
    if system["wind_model"] in ("beta_law", "confinement"):
        params.setdefault("R_star", float(system["R"]))
    return {**geometry(system), "wind_model": system["wind_model"], "wind_params": params,
            "mdot": float(system["mdot"]), "v_inf": float(system["v_inf"]),
            "mu_wind": float(system["mu_wind"]), "f_opacity": float(system["f_opacity"])}


def describe(system: Dict) -> Dict[str, float]:
    """Derived quantities for the fiducial table: M_tot, period in days, eclipse test."""
    g = geometry(system)
    a = g["d1"] + g["d2"]
    min_sep = a * math.cos(math.radians(g["i0"]))       # projected separation at conjunction
    return {"a": a, "M_tot": total_mass(a, system["period_s"]), "period_d": system["period_s"] / 86400.0,
            "min_projected_separation": min_sep, "total_eclipse": min_sep < g["R"] - g["r"]}
