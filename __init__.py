"""
Rare Event Simulation Package

Core implementation of median-based nested sampling algorithm.
"""

from .nested_sampling import nested_sampling
from .utils import gaussian_tail_exact, mills_approximation
from .plotting import plot_diagnostics

__version__ = "1.0.0"
__all__ = ["nested_sampling", "gaussian_tail_exact", "mills_approximation", "plot_diagnostics"]
