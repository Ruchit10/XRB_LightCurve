"""Synthetic data for injection-recovery tests of the CLOAK model.

Two generators, both writing files the pipeline reads unchanged:

- :mod:`cloak.synthetic.spectrum`   -- a fake absorbed power-law spectrum
  through PyXspec ``fakeit`` (needs HEASoft), from which
  :mod:`cloak.flux_table` builds the ``flux vs nH`` table, plus the
  flux-per-count-rate factor of every Chandra band.
- :mod:`cloak.synthetic.lightcurve` -- a CIAO-layout light curve
  (``dt, t_raw, mjd, phase, counts, rate, rate_err, flux_t``) drawn from the
  forward model at known parameters with Poisson counts, visits and gaps, and
  a ``*_truth.json`` recording everything that was injected.

Energy bands come from ``cloak.utils.CHANDRA_BANDS``, the same definition the
flux-table generator and the fitters use. Generated products belong in the
repository's ``synthetic_data/`` directory, which is tracked.
"""
