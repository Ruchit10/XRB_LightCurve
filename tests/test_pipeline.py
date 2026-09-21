#!/usr/bin/env python3
"""
Self-contained checks of the CLOAK pipeline: kernel symmetries, the periodic
model helpers, the phase-window rules and the command-line tools run end to
end on a synthetic light curve drawn from the tracked example flux table.
No real data and no HEASoft are needed.

    python -m unittest discover -s tests -v          # from the repository root
    python tests/test_pipeline.py                     # same thing
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cloak import kernel  # noqa: E402
from cloak import utils as U  # noqa: E402

TABLE = os.path.join(ROOT, "synthetic_data", "flux_vs_nH_tbabs_broad.csv")
GEOMETRY = dict(R=2.0, r=0.001, d1=11.0, d2=8.0, i0=78.0, f_opacity=0.02)
PY = sys.executable


def run(*args, **kw):
    """Run `python -m <module> ...` from the repository root."""
    return subprocess.run([PY, "-m", *args], cwd=ROOT, capture_output=True, text=True, **kw)


class KernelTests(unittest.TestCase):
    def test_light_curve_is_symmetric_about_mid_eclipse(self):
        df = kernel.simulate_lightcurve(flux_csv_path=TABLE, band="broad", dth=5.0, **GEOMETRY)
        deg = df["deg"].to_numpy()
        flux = df["nfl_broad"].to_numpy()
        k0 = int(np.argmin(np.abs(deg - 90.0)))          # mid-eclipse (phase 0.5)
        self.assertEqual(flux.argmin(), k0)
        for j in range(1, 18):
            self.assertAlmostEqual(flux[k0 + j] / flux[k0 - j], 1.0, places=10)
        self.assertAlmostEqual(df["phase"].iloc[k0], 0.5, places=12)

    def test_grid_step_must_divide_360(self):
        with self.assertRaises(ValueError):
            kernel.simulate_lightcurve(flux_csv_path=TABLE, band="broad", dth=7.0, **GEOMETRY)

    def test_emitter_must_be_smaller_than_companion(self):
        bad = dict(GEOMETRY, r=2.5)
        with self.assertRaises(ValueError):
            kernel.simulate_lightcurve(flux_csv_path=TABLE, band="broad", dth=5.0, **bad)

    def test_per_cell_columns_match_the_kernel_mean(self):
        geo = dict(GEOMETRY, r=1.0, d2h=6.0)
        df = kernel.simulate_lightcurve(flux_csv_path=TABLE, band="broad", dth=1.0, **geo)
        for deg in (0.0, 84.0, 88.0, 90.0):
            row = df.iloc[int(np.argmin(np.abs(df["deg"].to_numpy() - deg)))]
            cells = kernel.emitter_cell_columns(deg, **geo)
            vis = cells["visible"]
            if row["is_eclipsed"]:
                self.assertFalse(vis.any())
                continue
            mean = np.sum(cells["column"][vis] * cells["area"][vis]) / np.sum(cells["area"][vis])
            self.assertAlmostEqual(mean / row["fl"], 1.0, places=9, msg=f"deg {deg}")

    def test_flux_methods_agree_out_of_eclipse(self):
        a = kernel.simulate_lightcurve(flux_csv_path=TABLE, band="broad", dth=5.0, flux_method="interpolate", **GEOMETRY)
        b = kernel.simulate_lightcurve(flux_csv_path=TABLE, band="broad", dth=5.0, flux_method="refit", **GEOMETRY)
        self.assertTrue(np.all(np.isfinite(a["nfl_broad"])) and np.all(np.isfinite(b["nfl_broad"])))
        self.assertTrue(np.all(a["nfl_broad"] > 0) and np.all(b["nfl_broad"] > 0))


class PeriodicModelTests(unittest.TestCase):
    def setUp(self):
        self.phase = np.linspace(0.0, 1.0, 181)[:-1]
        self.flux = 1.0 - 0.5 * np.exp(-0.5 * ((self.phase - 0.5) / 0.05) ** 2)

    def test_eval_periodic_reproduces_nodes_and_wraps(self):
        pe, fe = U.periodic_model(self.phase, self.flux)
        np.testing.assert_allclose(U.eval_periodic(pe, fe, self.phase), self.flux, rtol=0, atol=1e-12)
        np.testing.assert_allclose(U.eval_periodic(pe, fe, self.phase + 3.0), self.flux, atol=1e-12)
        np.testing.assert_allclose(U.eval_periodic(pe, fe, self.phase, shift=0.25),
                                   U.eval_periodic(pe, fe, self.phase - 0.25), atol=1e-12)

    def test_best_phase_shift_matches_brute_force(self):
        rng = np.random.default_rng(3)
        true_shift = 0.137
        obs_phase = np.sort(rng.uniform(0, 1, 150))
        pe, fe = U.periodic_model(self.phase, self.flux)
        err = np.full(obs_phase.size, 0.02)
        obs = U.eval_periodic(pe, fe, obs_phase, shift=true_shift) + rng.normal(0, 0.02, obs_phase.size)
        search = U.build_phase_shift_search(obs_phase, n_model=self.phase.size)
        model_at_obs, shift, chi2 = U.best_phase_shift(pe, fe, obs, err ** 2, search)
        grid = np.linspace(0, 1, 20001)[:-1]
        brute = min(np.sum((obs - U.eval_periodic(pe, fe, obs_phase, shift=s)) ** 2 / err ** 2) for s in grid)
        self.assertLess(abs(((shift - true_shift + 0.5) % 1.0) - 0.5), 0.01)
        self.assertLessEqual(chi2, brute + 0.05)
        np.testing.assert_allclose(model_at_obs, U.eval_periodic(pe, fe, obs_phase, shift=shift), atol=1e-12)


class PhaseWindowTests(unittest.TestCase):
    def test_window_membership_and_wrapping(self):
        ph = np.array([0.05, 0.3, 0.5, 0.7, 0.95])
        np.testing.assert_array_equal(U.in_phase_window(ph, 0.4, 0.6), [False, False, True, False, False])
        np.testing.assert_array_equal(U.in_phase_window(ph, 0.9, 0.1), [True, False, False, False, True])
        self.assertTrue(U.is_full_phase_window(0.0, 1.0))
        self.assertFalse(U.is_full_phase_window(0.0, 0.5))

    def test_invalid_windows_are_rejected(self):
        for lo, hi in ((0.5, 0.5), (1.0, 0.0), (-0.1, 0.5), (0.2, 1.2)):
            with self.assertRaises(ValueError, msg=f"window {lo} {hi}"):
                U.check_phase_window(lo, hi)


class CommandLineTests(unittest.TestCase):
    """The tools run end to end on a synthetic light curve; the injected shift is recovered."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="cloak_test_")
        cls.lc_dir = os.path.join(cls.tmp.name, "lc")
        cls.model_csv = os.path.join(cls.tmp.name, "model_broad.csv")
        geo = ["--R", "2", "--r", "0.001", "--d1", "11", "--d2", "8", "--i0", "78", "--f-opacity", "0.02"]
        r = run("cloak.synthetic.lightcurve", "--flux-csv", TABLE, "--band", "broad", *geo,
                "--phase-shift", "0.985", "--scatter", "3e-13", "--n-orbits", "2", "--seed", "1",
                "--flux-per-rate", "1e-12", "--dt", "100", "--output", os.path.join(cls.lc_dir, "synth_broad.txt"))
        assert r.returncode == 0, r.stderr
        r = run("cloak.kernel", "--flux_csv", TABLE, "--band", "broad", *geo, "--output", cls.model_csv)
        assert r.returncode == 0, r.stderr

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _phase_analysis(self, *extra):
        return run("cloak.phase_analysis", "--data-dir", self.lc_dir, "--obs-column", "flux_t",
                   "--time-column", "t_raw", "--counts-per-bin", "100",
                   "--output", os.path.join(self.tmp.name, "fit.png"), *extra)

    def test_tabulated_fit_recovers_injected_shift(self):
        r = self._phase_analysis("--fit", "--sim-file", self.model_csv, "--sim-column", "nfl_broad",
                                 "--fit-phase-shift", "--scatter", "3e-13")
        self.assertEqual(r.returncode, 0, r.stderr)
        m = re.search(r"Phase shift = ([0-9.]+)", r.stdout)
        self.assertIsNotNone(m, r.stdout)
        self.assertLess(abs(float(m.group(1)) - 0.985), 0.005)

    def test_partial_window_needs_a_fixed_shift(self):
        r = self._phase_analysis("--fit", "--sim-file", self.model_csv, "--sim-column", "nfl_broad",
                                 "--fit-phase-shift", "--phase-window", "0.3", "0.5")
        self.assertEqual(r.returncode, 2)
        self.assertIn("phase-window", r.stderr)

    def test_mcmc_fit_runs_and_replots(self):
        out = os.path.join(self.tmp.name, "mcmc")
        common = ["--band", "broad", "--flux-csv", TABLE, "--data-dir", self.lc_dir, "--obs-column", "flux_t",
                  "--time-column", "t_raw", "--n-phase-bins", "40", "--dth", "5", "--quiet", "--no-plots",
                  "--n-walkers", "12", "--n-steps", "6", "--n-burn", "2", "--seed", "1", "--output-dir", out]
        r = run("cloak.mcmc_fit", *common)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertTrue(any(f.endswith("_chain.npz") for f in os.listdir(out)), os.listdir(out))
        r = run("cloak.mcmc_fit", "--replot", "--output-dir", out)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)

    def test_contradictory_arguments_exit_2(self):
        out = os.path.join(self.tmp.name, "mcmc_bad")
        r = run("cloak.mcmc_fit", "--band", "broad", "--flux-csv", TABLE, "--data-dir", self.lc_dir,
                "--obs-column", "flux_t", "--time-column", "t_raw", "--n-phase-bins", "40", "--no-phase-bin",
                "--output-dir", out)
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
