"""
Main Example: N-Dimensional Gaussian Rare Event Estimation

Reproduces the paper results for n=10, a=2.0.

Usage:
    python run_main_example.py
"""

import sys
import os

# Add src directory to path (robust way)
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(os.path.dirname(script_dir), 'src')
sys.path.insert(0, src_path)

# Now import directly
import nested_sampling
import utils
import plotting

import matplotlib.pyplot as plt


def main():
    """Run main example: n=10, a=2.0"""
    
    # Parameters
    n = 10
    a = 2.0
    N = 2**18
    
    print("\n" + "="*70)
    print("GAUSSIAN RARE EVENT ESTIMATION")
    print("="*70)
    print(f"Problem: P(mean(X_1,...,X_{n}) >= {a}) where X_i ~ N(0,1)")
    print()
    
    # Theoretical values
    prob_exact = utils.gaussian_tail_exact(a, n)
    prob_mills = utils.mills_approximation(a, n)
    
    print("Theoretical values:")
    print(f"  Exact: {prob_exact:.6e}")
    print(f"  Mills: {prob_mills:.6e}")
    print()
    
    # Run algorithm
    results = nested_sampling.nested_sampling(n, a, N, verbose=True)
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    prob_est = results['probability']
    print(f"Exact:     {prob_exact:.6e}")
    print(f"Mills:     {prob_mills:.6e}")
    print(f"Estimated: {prob_est:.6e}")
    print()
    print("Absolute differences:")
    print(f"  |Est - Exact| = {abs(prob_est - prob_exact):.6e}")
    print(f"  |Est - Mills| = {abs(prob_est - prob_mills):.6e}")
    print()
    
    # Create diagnostic plots
    plotting.plot_diagnostics(results, n, a, filename='diagnostics.png')
    plt.show()


if __name__ == "__main__":
    main()
