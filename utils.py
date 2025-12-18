"""
Utility functions for theoretical benchmarks.
"""

import numpy as np
from scipy.stats import norm


def gaussian_tail_exact(a, n):
    """
    Exact probability P(mean(X_1,...,X_n) >= a) where X_i ~ N(0,1).
    
    Parameters
    ----------
    a : float
        Threshold for sample mean
    n : int
        Number of samples
    
    Returns
    -------
    prob : float
        Exact probability using CDF
    """
    z = a * np.sqrt(n)
    return norm.sf(z)


def mills_approximation(a, n):
    """
    Mills ratio approximation for Gaussian tail probability.
    
    For X ~ N(0,1), approximates P(X > z) ≈ (1/z√(2π)) * exp(-z²/2)
    
    Parameters
    ----------
    a : float
        Threshold
    n : int
        Dimension
    
    Returns
    -------
    prob : float
        Approximate probability
    """
    z = a * np.sqrt(n)
    return (1.0 / (z * np.sqrt(2 * np.pi))) * np.exp(-z**2 / 2)
