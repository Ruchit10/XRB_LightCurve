#! /bin/bash
#
# Command reference for the IC 10 X-1 light-curve pipeline.
#
# This is a scrapbook, NOT a pipeline to execute top to bottom: steps 1 and 2
# are one-time setup that overwrite generated inputs (step 1 in particular
# rebuilds the XSPEC table that everything downstream depends on, and needs
# HEASoft/PyXspec). Copy the block you want and run it from the repository root
# with the `henv` environment active.
#
# Pipeline order:
#   1. Build the XSPEC flux-vs-nH table (once per spectral model / band set).
#   2. Attach a calibrated FLUX column to the Chandra light curves (once).
#   3. Generate a model light curve, or fit one with chi-square / MCMC.
#
# The wind column density is normalized physically from --mdot / --v-inf, so
# the model flux is absolute: no multiplicative rescaling is ever applied.

CSV=analyses/flux_vs_nH_tbabs_600bin_15803_broad.csv

# -----------------------------------------------------------------------------
# 1. Flux vs nH table (upstream of everything else). ONE-TIME — requires XSPEC,
#    and overwrites "$CSV", which is gitignored and cannot be rebuilt without
#    HEASoft. Uncomment deliberately.
# -----------------------------------------------------------------------------
# python compute_flux_vs_nH.py \
#     --specdir data/IC10X1_spec \
#     --out_csv "$CSV"

# -----------------------------------------------------------------------------
# 2. Add flux columns to the converted light curves. ONE-TIME — overwrites the
#    data/IC_10_X1_LC/*_with_flux/ trees.
# -----------------------------------------------------------------------------
# python utils/add_flux_simple.py \
#     data/IC_10_X1_LC/Broad_converted/ \
#     data/IC_10_X1_LC/Broad_with_flux/ \
#     1.500509e-11
#
# python utils/add_flux_simple.py \
#     data/IC_10_X1_LC/Soft_converted/ \
#     data/IC_10_X1_LC/Soft_with_flux/ \
#     5.920967e-12
#
# python utils/add_flux_simple.py \
#     data/IC_10_X1_LC/Hard_converted/ \
#     data/IC_10_X1_LC/Hard_with_flux/ \
#     2.807102e-11

# -----------------------------------------------------------------------------
# 3a. Single model light curve (e.g. broad band)
# -----------------------------------------------------------------------------
python xrb_lightcurve.py \
    --flux_csv "$CSV" \
    --flux_method interpolate \
    --wind-model smooth_pl \
    --R 2.0 --r 0.001 --d1 11.0 --d2 8.0 --i0 78.0 \
    --f-opacity 0.02 \
    --output sim_broad.csv

# -----------------------------------------------------------------------------
# 3b. Single chi-square fit of that model to the data
# -----------------------------------------------------------------------------
python chandra_phase_analysis.py \
    --data-dir data/IC_10_X1_LC/Broad_with_flux/ \
    --sim-file sim_broad.csv \
    --obs-column FLUX \
    --fit --fit-phase-shift \
    --output x1_fit_broad.png --write-model

# -----------------------------------------------------------------------------
# 3c. MCMC fit (broad band, single ObsID 15803, geometry + wind shape)
# -----------------------------------------------------------------------------
python mcmc_lightcurve_fit.py \
    --band broad --flux-csv "$CSV" \
    --data-dir data/IC_10_X1_LC_CIAO/broad/single/ \
    --obs-column flux_t --time-column t_raw \
    --n-phase-bins 150 \
    --wind-model smooth_pl --fit-wind-shape --fit-fopacity \
    --reparam --sampler zeus --likelihood jitter \
    --n-walkers 32 --n-steps 5000 --n-burn 500 --dth 4.0 \
    --n-threads 4 --compute-bic \
    --prior-i0 78.0,8.0,63.0,89.5 \
    --prior-r 0.01,1.0,0.001,10.0 \
    --prior-R 2.0,1.0,1.0,20.0 \
    --prior-a 19.0,5.0,2.0,80.0 \
    --prior-q 0.5,0.2,0.01,0.99 \
    --prior-Rb 6.0,3.0,3.0,80.0 \
    --prior-p 4.0,2.0,2.0,8.0

# -----------------------------------------------------------------------------
# 3d. Regenerate every plot from a finished run (no resampling).
#     All fit-defining options are restored from <band>_<wind>_run_config.json;
#     output options (--smooth, --compute-bic, --save-chi2, ...) are typed. Only
#     fits made since Phase 34 carry the wind_normalization stamp; the older
#     lam-mode results in mcmc_results/ are refused and must be refitted.
# -----------------------------------------------------------------------------
python mcmc_lightcurve_fit.py --replot --output-dir mcmc_results/broad --compute-bic
