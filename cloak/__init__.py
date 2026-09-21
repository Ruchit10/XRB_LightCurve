"""CLOAK: Column-density and Line-of-sight Occultation & Absorption Kernel.

A forward model for the orbital X-ray light curve of a wind-fed high-mass
X-ray binary (photoelectric absorption in the companion's wind plus
occultation by the companion), with an MCMC fitter, a tabulated phase-shift
fitter, plotting and synthetic-data generators.

Modules
-------
- :mod:`cloak.kernel`         -- the forward model (geometry, wind column,
                                 flux from a flux-vs-nH table); also a CLI
                                 that writes one model light curve.
- :mod:`cloak.flux_table`     -- builds the flux-vs-nH table with XSPEC
                                 (needs HEASoft / PyXspec).
- :mod:`cloak.mcmc_fit`       -- MCMC fit of the kernel to a light curve
                                 (emcee or zeus), replot from a saved chain.
- :mod:`cloak.phase_analysis` -- phase-binned plots and the tabulated
                                 phase-shift fit of a model light curve.
- :mod:`cloak.plot_results`   -- standalone figures from a model CSV.
- :mod:`cloak.plots`          -- every plotting routine.
- :mod:`cloak.utils`          -- ephemeris constants, loading, binning,
                                 smoothing, periodic interpolation, the
                                 phase-shift search, run-config handling.
- :mod:`cloak.synthetic`      -- fake spectra (PyXspec ``fakeit``) and
                                 synthetic light curves for injection tests.

Every CLI module runs both as ``python -m cloak.<module>`` and as
``python cloak/<module>.py`` from the repository root.
"""

__version__ = "0.1.0"
