"""
Rare-event probability P(Σ X_i ≥ a·n) via multilevel splitting.
Level routine:
  1. keep the upper 50 % (median threshold u_k);
  2. resample with replacement to restore N samples;
  3. run a single vectorised Gibbs pass (one coordinate update per particle);
  4. record acceptance rate and variance of the median.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ------------------------------------------------------------------
# helper functions
# ------------------------------------------------------------------
def normal_tail_exact(a: float, n: int):
    """Exact P(Σ ≥ a·n) with Σ~N(0,n)."""
    z = a * np.sqrt(n)
    return norm.sf(z)                  # 1 − Φ(z)

def normal_tail_clt(a: float, n: int):
    """CLT tail approximation."""
    return (1 / (a * np.sqrt(2 * np.pi * n))) * np.exp(-0.5 * a**2 * n)

def pdf_sum(u: float, n: int):
    """PDF of Σ~N(0,n) evaluated at u."""
    return (1 / np.sqrt(2 * np.pi * n)) * np.exp(-u**2 / (2 * n))

# ------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------
n        = 10          # dimension
a        = 2.0         # target threshold in mean units
N        = 2**18       # samples per level
rng      = np.random.default_rng()

# ------------------------------------------------------------------
# Initialize
# ------------------------------------------------------------------
samples       = rng.normal(size=(N, n))
thresholds    = []
vars_median   = []
accept_rates  = []
iteration     = 0

while True:
    sums = samples.sum(axis=1)
    u_k  = np.median(sums)                 # median threshold
    thresholds.append(u_k)

    # variance of the sample median
    f_u   = pdf_sum(u_k, n)
    var_u = 1.0 / (4 * N * f_u**2) if f_u > 0 else np.inf
    vars_median.append(var_u)

    print(f"iter {iteration:2d}  u_k = {u_k: .5f}  Var(median) = {var_u:.2e}")

    # stop if the target threshold is reached
    if u_k >= a * n:
        break

    mask       = sums >= u_k
    survivors  = samples[mask]             # shape (N/2, n)
    parent_idx = rng.integers(0, survivors.shape[0], size=N)
    samples    = survivors[parent_idx].copy()
    total_prop   = N * n                 
    total_accept = 0
    current_sums = samples.sum(axis=1)

  # single constrained Gibbs pass
    for j in range(n):
        prop = rng.normal(size=N)
        new_sums = current_sums - samples[:, j] + prop
        accept_mask = new_sums >= u_k
        total_accept += accept_mask.sum()
        samples[accept_mask, j] = prop[accept_mask]
        current_sums[accept_mask] = new_sums[accept_mask]

    accept_rate = total_accept / total_prop
    accept_rates.append(accept_rate)
    print(f"           acceptance rate = {accept_rate:.4f}")

    iteration += 1


prob_est = 0.5 ** iteration
prob_true = normal_tail_exact(a, n)
prob_clt = normal_tail_clt(a, n)

print("\n================== summary ==================")
print(f"levels                  : {iteration}")
print(f"estimated probability   : {prob_est: .6e}")
print(f"exact probability       : {prob_true: .6e}")
print(f"CLT approximation       : {prob_clt: .6e}")

# ------------------------------------------------------------------
# plot
# ------------------------------------------------------------------
iters = range(len(thresholds))
log_vars = [np.log10(v) if np.isfinite(v) else -np.inf for v in vars_median]

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(iters, thresholds, marker='o')
ax[0].set(title="Median threshold evolution",
          xlabel="level", ylabel="uₖ")
ax[0].grid(True)

ax[1].plot(iters, log_vars, marker='s')
ax[1].set(title="log₁₀(Variance of median)",
          xlabel="level", ylabel="log₁₀ Var(uₖ)")
ax[1].grid(True)

plt.tight_layout()
plt.show()
