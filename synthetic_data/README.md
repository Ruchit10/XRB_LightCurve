# Synthetic data

Generators for injection–recovery tests. Everything they write is read by the
existing pipeline unchanged, and the energy bands are `utils.utils.CHANDRA_BANDS`,
the same edges `compute_flux_vs_nH.py` and the fitters use.

```
make_spectrum.py  ──fakeit──▶  fake PHA  ──compute_flux_vs_nH.py──▶  flux vs nH table
                                   │                                        │
                                   └── band_factors.json (flux per rate) ──┐│
                                                                           ▼▼
make_lightcurve.py ──forward model + Poisson──▶  CIAO-layout light curve + *_truth.json
                                                                           │
                                            chandra_phase_analysis.py / mcmc_lightcurve_fit.py
```

## 1. Spectrum (needs PyXspec)

```bash
python synthetic_data/make_spectrum.py --out-dir synthetic_data/out/spec \
    --model tbabs --nH 0.75 --PhoIndex 1.86 --norm 1e-4 --exposure 100000 --seed 1
python compute_flux_vs_nH.py --specdir synthetic_data/out/spec --band broad \
    --out_csv synthetic_data/out/flux_vs_nH_broad.csv --out_png synthetic_data/out/flux_vs_nH_broad.png
```

`make_spectrum.py` fakes an absorbed power law through the IC 10 X-1 combined
ACIS response (override with `--rmf/--arf`, optional `--bkg`), copies the RMF
and ARF next to the fake PHA so the output directory is self-contained, and
prints, per band, the model flux, the fake count rate and their ratio. That
ratio is the `--flux-per-rate` the light-curve generator needs. Without HEASoft, use an
existing table (`temp/flux_vs_nH_tbabs_broad.csv`) and the real-data factor
`1.13e-11` erg cm⁻² s⁻¹ per count s⁻¹ (the default).

## 2. Light curve

```bash
python synthetic_data/make_lightcurve.py --flux-csv synthetic_data/out/flux_vs_nH_broad.csv \
    --band broad --R 2 --r 0.001 --d1 11 --d2 8 --i0 78 --f-opacity 0.02 \
    --phase-shift 0.985 --scatter 3e-13 --n-orbits 2 --gap-fraction 0.1 --seed 1 \
    --output synthetic_data/out/broad/synth_broad.txt
```

All forward-model keywords of `xrb_lightcurve.py` are accepted with the same
defaults. The model flux at each bin is shifted by `--phase-shift` (mid-eclipse
lands at data phase `0.5 + shift`), lifted by `--scatter`, converted to counts
with `--flux-per-rate` and `--dt`, and Poisson-sampled (`--noiseless` to skip).
Visits are `--visits start:duration,...` in seconds after `REF_EPOCH`, or one
visit of `--n-orbits`; `--gap-fraction`/`--gap-duration` remove random blocks.
`--bkg-rate` adds a background that is subtracted from the net rate, which
produces the zero and negative bins real data have. The truth file records
every injected value plus `mid_eclipse_data_phase`, the number of bins and
zero-count bins.

## 3. Recover

```bash
python xrb_lightcurve.py --flux_csv synthetic_data/out/flux_vs_nH_broad.csv --band broad \
    --R 2 --r 0.001 --d1 11 --d2 8 --i0 78 --f-opacity 0.02 --output synthetic_data/out/model_broad.csv
python chandra_phase_analysis.py --data-dir synthetic_data/out/broad --obs-column flux_t \
    --time-column t_raw --counts-per-bin 100 --fit --sim-file synthetic_data/out/model_broad.csv \
    --sim-column nfl_broad --fit-phase-shift --scatter 3e-13

python mcmc_lightcurve_fit.py --band broad --flux-csv synthetic_data/out/flux_vs_nH_broad.csv \
    --data-dir synthetic_data/out/broad --obs-column flux_t --time-column t_raw \
    --n-phase-bins 150 --reparam --fit-wind-shape --fit-fopacity --likelihood jitter \
    --seed 1 --compute-bic --output-dir synthetic_data/out/mcmc_broad
```

Pass the injected floor as `--scatter`: the tabulated fitter's default
estimate takes the mean flux inside phase 0.4–0.6 as the floor, which assumes
a total eclipse and, for a partial dip, overestimates it and biases the shift
(0.960 instead of 0.985 in the example above; 0.9848 with `--scatter 3e-13`).
Compare the posterior (`mcmc_summary.txt`, the MAP in `*_bestfit_model.txt`)
with `synth_broad_truth.json`. The exact degeneracies of the model apply to
synthetic data too: `q` is unidentifiable and the overall length scale is tied
to `f_opacity`, so compare `a`, `R`, `i0` and `f_opacity` jointly, not
individually.
