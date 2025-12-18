"""
Plotting functions for diagnostic visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_diagnostics(results, n, a, filename='diagnostics.png'):
    """
    Create 4-panel diagnostic plot.
    
    Parameters
    ----------
    results : dict
        Output from nested_sampling()
    n : int
        Dimension
    a : float
        Target threshold
    filename : str
        Output filename (default: 'diagnostics.png')
    """
    
    thresholds = results['thresholds']
    vars_median = results['vars_median']
    accept_rates = results['accept_rates']
    
    levels = np.arange(len(thresholds))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Panel 1: Threshold evolution
    ax = axes[0, 0]
    ax.plot(levels, thresholds, 'o-', markersize=3)
    ax.axhline(n * a, color='r', linestyle='--', label=f'Target={n*a}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Threshold')
    ax.set_title('Threshold Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Threshold spacing
    ax = axes[0, 1]
    increments = np.diff([0] + thresholds)
    ax.plot(levels, increments, 's-', markersize=3, color='green')
    expected = 0.67 * np.sqrt(n)
    ax.axhline(expected, color='r', linestyle='--', label=f'Expected≈{expected:.2f}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Increment')
    ax.set_title('Threshold Spacing')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Variance of median
    ax = axes[1, 0]
    log_vars = [np.log10(v) for v in vars_median if np.isfinite(v)]
    ax.plot(range(len(log_vars)), log_vars, '^-', markersize=3, color='purple')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('log₁₀ Var(median)')
    ax.set_title('Variance of Median')
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Acceptance rates
    ax = axes[1, 1]
    ax.plot(range(len(accept_rates)), accept_rates, 'o-', markersize=3, color='orange')
    ax.axhline(0.5, color='r', linestyle='--', label='Expected=0.5')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('Gibbs Sampler Acceptance Rate')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    return fig
