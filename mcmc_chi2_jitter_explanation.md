# MCMC Log-Likelihood, Jitter Term, and Sampler Comparison — Explanation

This document explains how `log_probability` drives a single MCMC walker step in
`mcmc_lightcurve_fit.py`, how the optional jitter term modifies the standard
chi-squared likelihood, and how the emcee and zeus samplers differ in how they
propose and accept new positions.

---

## 1. The Core Formula

Every emcee/zeus step calls `log_probability`, which implements Bayes' theorem in
log space:

```
log P(θ | data) = log P(θ) + log P(data | θ)
                = log_prior(θ) + log_likelihood(θ)
```

The parameter vector `θ` contains, in order:

```
geometry (5)  →  [log_f if jitter]  →  wind-shape params (if --fit-wind-shape)
 d1, d2, r, R, i0
```

---

## 2. `log_prior(θ)` — Checking Plausibility of the Proposed Position

The prior enforces two layers of constraint.

### 2.1 Hard Box Rejection

If any parameter falls outside its `[min, max]` range, or if the physical
constraint `r ≥ R` is violated, the prior returns `-inf` immediately. The
proposed step is then always rejected with zero probability.

```python
for name, value in zip(active_names, theta):
    if not (prior['min'] < value < prior['max']):
        return -np.inf

if theta[2] >= theta[3]:   # r >= R is unphysical
    return -np.inf
```

### 2.2 Soft Gaussian Penalty

For every parameter inside the box, a Gaussian log-prior is accumulated:

```
log_p += -0.5 * ((θ_i - mean_i) / std_i)²
```

This gently pulls parameters toward their prior means without hard-rejecting
anything. The `std` values are intentionally wide (see `DEFAULT_PRIORS`) so the
data dominates the posterior.

---

## 3. `log_likelihood(θ)` — How Well Does the Model Fit the Data?

### 3.1 Standard chi² (Gaussian, `--likelihood chi2`)

```python
chi2 = np.sum(((obs_flux - model_flux) / obs_err) ** 2)
return -0.5 * chi2
```

This is the log of a product of independent Gaussians:

$$\ln \mathcal{L} = -\frac{1}{2} \sum_i \left(\frac{F_{\text{obs},i} - F_{\text{model},i}}{\sigma_i}\right)^2$$

Normalization constants are dropped because they are independent of `θ` and
cancel exactly in the Metropolis acceptance ratio.

### 3.2 Jitter Likelihood (`--likelihood jitter`)

Standard chi² assumes the reported error bars `obs_err` perfectly describe all
noise. In practice there is often **extra scatter** — intrinsic source
variability, calibration systematics, or model misfit — that inflates residuals
beyond `obs_err`. Ignoring this produces an overconfident posterior.

The jitter model adds a **free fractional systematic error** `f = exp(log_f)` as
an extra MCMC dimension, inflating the effective per-point variance:

$$\sigma^2_{\text{eff},i} = \sigma^2_{\text{obs},i} + \left(f \cdot F_{\text{model},i}\right)^2$$

The log-likelihood then becomes:

$$\ln \mathcal{L} = -\frac{1}{2} \sum_i \left[ \frac{(F_{\text{obs},i} - F_{\text{model},i})^2}{\sigma^2_{\text{eff},i}} + \ln \sigma^2_{\text{eff},i} \right]$$

```python
f      = np.exp(theta[idx_logf])
sigma2 = obs_err**2 + (f * model_flux)**2
return -0.5 * np.sum((obs_flux - model_flux)**2 / sigma2 + np.log(sigma2))
```

#### Why the `+ log(σ²_eff)` term?

Without it, you could drive the residual term to zero by making `f` arbitrarily
large. The `log(σ²_eff)` term is the normalization of the Gaussian that *does*
depend on `f` — including it prevents unconstrained jitter inflation.

#### Why multiplicative (proportional to model flux)?

`f` scales with the predicted flux, not as a flat additive offset. For an X-ray
binary, intrinsic variability is expected to scale with luminosity, so this is
physically motivated.

#### The jitter prior

```python
JITTER_PRIOR = {'mean': -3.0, 'std': 2.0, 'min': -10.0, 'max': 0.0}
```

- `log_f ∈ (-10, 0)` → `f ∈ (e⁻¹⁰, 1)`, i.e. at most ~100% fractional jitter.
- Prior mean `log_f = -3` → `f ≈ 0.05` (5% fractional jitter as a starting
  expectation).
- A Gaussian penalty on `log_f` is applied inside `log_prior`, discouraging
  runaway error inflation unless the data genuinely demands it.

---

## 4. Sampler Comparison: emcee vs zeus

Both samplers share the same `log_probability` function, the same `n_walkers`
and `n_steps` interface, and both return chains via `get_chain(discard=n_burn,
flat=True)`. The difference is entirely in **how each sampler proposes and
accepts new positions**.

### 4.1 emcee — Ensemble Stretch Move (Metropolis–Hastings)

emcee uses the **affine-invariant stretch move** (Goodman & Weare 2010). For
walker `k` at position `θ_k`:

1. Pick a random "complementary" walker `θ_j` from the rest of the ensemble.
2. Draw a stretch factor `z` from the distribution `g(z) ∝ 1/√z` on `[1/a, a]`
   (default `a = 2`).
3. Propose `θ_new = θ_j + z · (θ_k − θ_j)` — a point along the line between
   the two walkers, scaled by `z`.
4. **Accept/reject via Metropolis–Hastings**:

$$\alpha = \min\!\left(1,\ z^{n_\text{dim}-1} \cdot e^{\ln P_\text{new} - \ln P_\text{old}}\right)$$

The `z^{n_dim - 1}` factor is the Jacobian correction for the stretch
transformation. If `α < 1`, draw `u ~ Uniform(0,1)` and accept only if `u < α`.
**The acceptance rate is always < 1** and varies with the geometry of the
posterior.

**Parallelism**: emcee splits the ensemble into two halves and updates each half
in parallel using the other half as the complementary set. This enables
multi-threaded runs (`--n-threads N`), which is supported in this codebase.

#### emcee weakness

The stretch move performs poorly on highly correlated or non-elliptical
posteriors. If `θ_k` and `θ_j` lie near a narrow banana-shaped ridge, most
proposed steps land off the ridge and are rejected, producing a high
autocorrelation time.

---

### 4.2 zeus — Ensemble Slice Sampler

zeus uses **Ensemble Slice Sampling** (Karamanis & Beutler 2020). Rather than
a Metropolis accept/reject, it uses the **slice sampling** principle: for a
given direction, bracket the current log-posterior value with a "slice", then
sample uniformly within that slice.

**Per-step procedure:**

1. Pick a complementary walker `θ_j` from the ensemble and define a direction
   `d = θ_j − θ_k` (the **Differential Move**, default).
2. Define the 1D slice along that direction: find the interval `[L, R]` such
   that `log P(θ_k + t·d) > log P(θ_k)` for `t ∈ [L, R]` using an
   **expand-and-contract** (stepping-out) procedure.
3. Sample `t*` uniformly from `[L, R]` and set `θ_new = θ_k + t*·d`.  
   If `log P(θ_new) < log P(θ_k)` (outside the slice), shrink the bracket and
   resample until a valid point is found.
4. **Acceptance rate is identically 1** — every step produces a new accepted
   position because the shrinking guarantees the final sample is within the
   slice.

**Key properties:**
- No hand-tuning of a proposal scale — the slice width adapts automatically.
- The stepping-out phase requires **extra `log_probability` evaluations** per
  step compared to emcee's single evaluation. In practice this means each zeus
  step is more expensive in wall-clock terms, but the walker moves farther and
  the autocorrelation time is shorter, so fewer total steps are needed.
- `light_mode=True` can be set to skip expansions after tuning, roughly halving
  the cost per step for approximately Gaussian posteriors.

**Parallelism**: zeus does not currently support multi-process parallelism
through the same `pool=` interface as emcee. In this codebase, `--n-threads` is
therefore only wired up for the emcee path.

---

### 4.3 Side-by-side Comparison

| Property | emcee | zeus |
|----------|-------|------|
| Algorithm | Stretch move (affine-invariant M–H) | Ensemble Slice Sampling |
| Acceptance rate | Variable (< 1), must tune `a` | Always 1 (no rejection) |
| `log_prob` calls per step | 1 per walker | Multiple per walker (expand + shrink) |
| Correlated posteriors | Struggles (high autocorr time) | Handles well |
| Hand-tuning needed | Minimal (`a` parameter) | None |
| Multi-thread support | Yes (`--n-threads N`) | Not wired up in this codebase |
| Recommended use | Well-conditioned, low-correlation posteriors | Correlated, curved, or multi-modal posteriors |

**Rule of thumb**: if the geometry parameters `(d1, d2, r, R, i0)` are strongly
correlated (e.g. d1 and d2 are anti-correlated along the total-separation axis),
zeus will mix much faster and is the better choice despite the higher per-step
cost.

---

## 5. The Metropolis–Hastings Accept/Reject (emcee only)

Given the current walker position `θ_old` with score `log P_old`, emcee proposes
a new `θ_new` (via its stretch move) and computes:

```
log P_new = log_prior(θ_new) + log_likelihood(θ_new)
```

The acceptance probability is:

$$\alpha = \min\!\left(1,\ z^{n_\text{dim}-1} \cdot e^{\ln P_\text{new} - \ln P_\text{old}}\right)$$

| Condition | Result |
|-----------|--------|
| `log P_new > log P_old` | `α = 1` — always accepted (better posterior) |
| `log P_new < log P_old` | accepted with probability `exp(ΔlogP) < 1` — occasional downhill moves prevent trapping |
| `log P_new = -inf` (prior rejection) | `α = 0` — always rejected |

zeus bypasses this step entirely: its slice-sampling shrinkage guarantees
`log P_new ≥ log P_old` before the step is committed.

---

## 6. Full Step Flow for a Single Walker

```
         ┌──────────────────────────────────────────────────────┐
         │  emcee (stretch move)    │  zeus (slice sampler)     │
         │                          │                            │
propose  │  θ_new = θ_j + z·(θ_k−θ_j)  │  expand slice along d=θ_j−θ_k │
         │  (single log_prob call)  │  (multiple log_prob calls)│
         └──────────────────────────────────────────────────────┘
                           │
                           ▼
             log_prior(θ_new)
               ├─ hard reject: out-of-bounds or r ≥ R  → -inf
               └─ soft: Σ -0.5 * ((θ_i - mean_i)/std_i)²
                           │
                           ▼
             log_likelihood(θ_new)
               [chi2]   → -0.5 · Σ [(obs - model) / σ_obs]²
               [jitter] → -0.5 · Σ [(obs - model)² / σ²_eff + log σ²_eff]
                          where  σ²_eff = σ²_obs + (f·model)²,  f = exp(log_f)
                           │
                           ▼
             log P_new = log_prior + log_likelihood
                           │
               ┌───────────┴─────────────┐
               │ emcee                   │ zeus
               │ accept with prob        │ shrink slice until
               │ min(1, z^(d-1)·eΔlogP) │ log P_new ≥ log P_old
               │ (can reject)            │ (always accepted)
               └─────────────────────────┘
```

---

## 7. Parameter Vector Layout Summary

| Index | `chi2` mode | `jitter` mode |
|-------|-------------|---------------|
| 0 | `d1` (or `a` if reparam) | same |
| 1 | `d2` (or `q` if reparam) | same |
| 2 | `r` | same |
| 3 | `R` | same |
| 4 | `i0` | same |
| 5 | *(wind-shape if active)* | `log_f` |
| 6+ | — | wind-shape params (if active) |
