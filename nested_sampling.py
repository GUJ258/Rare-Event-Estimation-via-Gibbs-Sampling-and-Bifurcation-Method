"""
Core nested sampling algorithm implementation.
"""

import numpy as np


def nested_sampling(n, a, N=2**18, verbose=True):
    """
    Estimate P(mean(X_1,...,X_n) >= a) where X_i ~ N(0,1).
    
    Parameters
    ----------
    n : int
        Dimension
    a : float
        Target threshold for sample mean
    N : int
        Number of samples per level (default: 2^18)
    verbose : bool
        Print iteration details
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - probability : float
            Estimated probability (1/2)^K
        - K : int
            Number of iterations
        - thresholds : list
            Median threshold at each iteration
        - vars_median : list
            Variance of median at each iteration
        - accept_rates : list
            Gibbs acceptance rate at each iteration
        - final_samples : ndarray
            Final sample array (N x n)
    """
    
    target = n * a  # Target sum
    
    # Storage
    thresholds = []
    vars_median = []
    accept_rates = []
    
    # Initialize
    X = np.random.randn(N, n)
    
    if verbose:
        print("="*70)
        print(f"Nested Sampling: n={n}, a={a}, N={N}")
        print(f"Target sum: {target:.2f}")
        print("="*70)
    
    iteration = 0
    
    while True:
        # Compute sums
        sums = X.sum(axis=1)
        
        # Median threshold
        u_next = np.median(sums)
        thresholds.append(u_next)
        
        # Variance of median: Var(median) ≈ 1/(4*N*f(u)^2)
        # Estimate f(u) using IQR: f(u) ≈ (N/2)/(Q75 - Q50)
        q75, q50, q25 = np.percentile(sums, [75, 50, 25])
        if q75 > q50:
            f_u = (N / 2) / (q75 - q50)
            var_median = 1.0 / (4 * N * f_u**2)
        else:
            var_median = np.inf
        vars_median.append(var_median)
        
        if verbose:
            print(f"\nIteration {iteration}:")
            print(f"  u_{iteration+1} = {u_next:.4f}")
            print(f"  Var(median) = {var_median:.4e}")
        
        # Check stopping condition
        if u_next >= target:
            if verbose:
                print(f"\nReached target! {u_next:.4f} >= {target:.2f}")
            break
        
        # Selection: keep survivors
        survivors = X[sums >= u_next]
        n_survivors = len(survivors)
        
        if verbose:
            print(f"  Survivors: {n_survivors}/{N}")
        
        # Resampling: restore population size
        indices = np.random.choice(n_survivors, size=N, replace=True)
        X = survivors[indices].copy()
        
        # Gibbs pass: one full sweep
        total_accepts = 0
        for j in range(n):
            current_sums = X.sum(axis=1)
            proposals = np.random.randn(N)
            new_sums = current_sums - X[:, j] + proposals
            accept = new_sums >= u_next
            X[accept, j] = proposals[accept]
            total_accepts += accept.sum()
        
        accept_rate = total_accepts / (N * n)
        accept_rates.append(accept_rate)
        
        if verbose:
            print(f"  Accept rate: {accept_rate:.4f}")
        
        iteration += 1
    
    # Final result
    K = iteration
    prob = 0.5 ** K
    
    if verbose:
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"K = {K}")
        print(f"P = (1/2)^{K} = {prob:.6e}")
        print("="*70)
    
    return {
        'probability': prob,
        'K': K,
        'thresholds': thresholds,
        'vars_median': vars_median,
        'accept_rates': accept_rates,
        'final_samples': X,
    }
