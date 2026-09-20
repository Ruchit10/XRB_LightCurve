"""Synthetic data for injection-recovery tests of the light-curve model.

Two generators, both writing files the existing pipeline reads unchanged:

- :mod:`synthetic_data.make_spectrum` -- a fake absorbed power-law spectrum
  through PyXspec ``fakeit`` (needs HEASoft), from which
  ``compute_flux_vs_nH.py`` builds the ``flux vs nH`` table, plus the
  flux-per-count-rate factor of every Chandra band.
- :mod:`synthetic_data.make_lightcurve` -- a CIAO-layout light curve
  (``dt, t_raw, mjd, phase, counts, rate, rate_err, flux_t``) drawn from the
  forward model at known parameters with Poisson counts, visits and gaps, and
  a ``*_truth.json`` recording everything that was injected.

Energy bands come from ``utils.utils.CHANDRA_BANDS``, the same definition the
flux-table generator and the fitters use.
"""
