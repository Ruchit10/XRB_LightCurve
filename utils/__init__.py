"""Shared helpers for the XRB light-curve analysis codebase.

Two modules hold everything that used to be duplicated between the
top-level analysis scripts:

- :mod:`utils.utils`      -- ephemeris constants, data loading, phase binning,
                             smoothing, model interpolation and χ² fitting.
- :mod:`utils.plot_utils` -- every plotting routine, built on the single
                             :func:`utils.plot_utils.plot_lightcurve_fit`
                             drawing function.

``chandra_phase_analysis.py`` and ``mcmc_lightcurve_fit.py`` import from here
rather than from each other, so there is exactly one implementation of each
routine. The other files in this directory are standalone data-prep scripts and
are not part of the package API.
"""
