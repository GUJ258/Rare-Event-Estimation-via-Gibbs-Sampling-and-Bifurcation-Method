"""
1D Example: Simple Gaussian Tail Estimation

Estimate P(X >= a) where X ~ N(0,1).

Usage:
    python run_1d_example.py
"""

import sys
import os

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(os.path.dirname(script_dir), 'src')
sys.path.insert(0, src_path)

import utils

import numpy as np
import matplotlib.pyplot as plt


def nested_sampling_1d(a, N=2**12, verbose=True):
    """
    1D nested sampling for P(X >= a) where X ~ N(0,1).
    
    Parameters
    ----------
    a : float
        Target threshold
    N : int
        Number of samples per level
    verbose : bool
        Print details
    
    Returns
    -------
    results : dict
    """
    
    # Storage
    thresholds = []
    accept_rates = []
    
    # Initialize
    X = np.random.randn(N, 1)
    
    if verbose:
        print("="*70)
        print(f"1D Nested Sampling: a={a}, N={N}")
        print("="*70)
    
    iteration = 0
    
    while True:
        # Median threshold
        u_next = np.median(X[:, 0])
        thresholds.append(u_next)
        
        if verbose:
            print(f"\nIteration {iteration}:")
            print(f"  u_{iteration+1} = {u_next:.4f}")
        
        # Check stopping
        if u_next >= a:
            if verbose:
                print(f"\nReached target! {u_next:.4f} >= {a:.2f}")
            break
        
        # Selection
        survivors = X[X[:, 0] >= u_next]
        n_survivors = len(survivors)
        
        if verbose:
            print(f"  Survivors: {n_survivors}/{N}")
        
        # Resampling
        indices = np.random.choice(n_survivors, size=N, replace=True)
        X = survivors[indices].copy()
        
        # Gibbs pass
        proposals = np.random.randn(N)
        accept = proposals >= u_next
        X[accept, 0] = proposals[accept]
        
        accept_rate = accept.sum() / N
        accept_rates.append(accept_rate)
        
        if verbose:
            print(f"  Accept rate: {accept_rate:.4f}")
        
        iteration += 1
    
    # Final result
    K = iteration
    prob = 0.5 ** K
    
    if verbose:
        print("\n" + "="*70)
        print(f"K = {K}")
        print(f"P = (1/2)^{K} = {prob:.6e}")
        print("="*70)
    
    return {
        'probability': prob,
        'K': K,
        'thresholds': thresholds,
        'accept_rates': accept_rates,
    }


def main():
    """Run 1D example."""
    
    a = 3.0
    N = 2**12
    
    print("\n" + "="*70)
    print("1D GAUSSIAN EXAMPLE")
    print("="*70)
    print(f"Problem: P(X >= {a}) where X ~ N(0,1)")
    print()
    
    # Theoretical
    prob_exact = utils.gaussian_tail_exact(a, 1)
    print(f"Exact: {prob_exact:.6e}")
    print()
    
    # Run algorithm
    results = nested_sampling_1d(a, N, verbose=True)
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Exact:     {prob_exact:.6e}")
    print(f"Estimated: {results['probability']:.6e}")
    print(f"Difference: {abs(results['probability'] - prob_exact):.6e}")
    print()
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    ax[0].plot(results['thresholds'], 'o-')
    ax[0].axhline(a, color='r', linestyle='--', label=f'Target={a}')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Threshold')
    ax[0].set_title('Threshold Evolution')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    ax[1].plot(results['accept_rates'], 's-')
    ax[1].axhline(0.5, color='r', linestyle='--', label='Expected=0.5')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Acceptance Rate')
    ax[1].set_title('Accept Rate')
    ax[1].set_ylim([0, 1])
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('1d_results.png', dpi=150)
    print("Saved: 1d_results.png")
    plt.show()


if __name__ == "__main__":
    main()
