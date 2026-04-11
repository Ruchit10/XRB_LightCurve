#! /bin/bash

# Add flux columns to the light curve files
python utils/add_flux_simple.py \
    data/IC_10_X1_LC/Broad_converted/ \                                       
    data/IC_10_X1_LC/Broad_with_flux/ \
1.500509e-11

python utils/add_flux_simple.py \
    data/IC_10_X1_LC/Soft_converted/ \                                       
    data/IC_10_X1_LC/Soft_with_flux/ \ 
5.920967e-12

python utils/add_flux_simple.py \
    data/IC_10_X1_LC/Hard_converted/ \                                       
    data/IC_10_X1_LC/Hard_with_flux/ \ 
2.807102e-11

# Single run LC generation (Eg: broad band)
python xrb_lightcurve.py \
--flux_method interpolate \
--flux_csv flux_vs_nH_broad.csv \
--output sim_flux_interp6_broad.csv \
--i0 12.0 --lam 0.572385 --lam2 0.572385 

# Single chisq fit (Eg: broad band)
python chandra_phase_analysis.py \
--data-dir data/IC_10_X1_LC/Broad_with_flux/ \
--sim-file sim_flux_interp6_broad.csv \
--fit --output x1_fit_flux_interp6_broad.png \
--obs-column FLUX

# Run the MCMC fitting for the Soft band on single LC 15803
python mcmc_lightcurve_fit.py \
--band soft --wind-model av \
--load-grid soft_15803_grid.npz --flux-csv ./analyses/flux_vs_nH_tbabs_15803_soft.csv \
--data-dir data/IC_10_X1_LC_CIAO/soft/single/ \
--lam 0.533551 --lam2 0.533551 --obs-column flux_t --time-column t_raw \
--no-phase-bin --sampler zeus --dth 2.0 --n-workers 7 --likelihood chi2 \
--prior-i0 10.0,5.0,2.0,40.0 \
--prior-r 0.01,0.1,0.001,5.0 \
--prior-R 2.0,0.5,1.0,20.0 \
--prior-d1 11.0,3.0,1.5,40.0 \
--prior-d2 11.0,3.0,1.5,40.0

python mcmc_lightcurve_fit.py \
--band soft --wind-model cv --replot --compute-waic \
--load-grid soft_cv_15803_grid.npz --grid-points 9 --n-steps 10000 --flux-csv ./analyses/flux_vs_nH_tbabs_600bin_15803_soft.csv \
--data-dir data/IC_10_X1_LC_CIAO/soft/single/ \
--lam 0.533545 --lam2 0.533545 --obs-column flux_t --time-column t_raw \
--n-phase-bins 150 --sampler zeus --dth 2.0 --n-workers 7 --likelihood chi2 \
--prior-i0 10.0,5.0,2.0,40.0 \
--prior-r 0.01,0.1,0.001,5.0 \
--prior-R 2.0,0.5,1.0,20.0 \
--prior-d1 11.0,3.0,1.5,40.0 \
--prior-d2 11.0,3.0,1.5,40.0
