# Rare Event Estimation via Gibbs Sampling and Bifurcation Method

Clean, professional implementation of the nested sampling algorithm for rare event probability estimation.

## Structure

```
rare-event-simulation/
├── src/                          # Core package
│   ├── __init__.py
│   ├── nested_sampling.py        # Main algorithm
│   ├── plotting.py               # Diagnostic plots
│   └── utils.py                  # Theoretical benchmarks
├── examples/                     # Example scripts
│   ├── run_main_example.py       # Main n-dimensional example
│   ├── run_1d_example.py         # Simple 1D case
│   └── run_median_validation.py  # Median CLT validation
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running Examples

```bash
# Main example (n=10, a=2.0)
cd examples
python run_main_example.py

# 1D simple example
python run_1d_example.py

# Median CLT validation
python run_median_validation.py
```

### Using as a Package

```python
import sys
sys.path.insert(0, 'path/to/src')

from nested_sampling import nested_sampling
from utils import gaussian_tail_exact
from plotting import plot_diagnostics

# Run algorithm
results = nested_sampling(n=10, a=2.0, N=2**18)

# Get results
prob = results['probability']          # Estimated probability (1/2)^K
K = results['K']                       # Number of iterations
thresholds = results['thresholds']     # Threshold at each level
vars_median = results['vars_median']   # Variance of median at each level
accept_rates = results['accept_rates'] # Gibbs acceptance rate at each level

print(f"P = {prob:.6e}")

# Create diagnostic plots
plot_diagnostics(results, n=10, a=2.0)
```

## File Descriptions

### Core Package (`src/`)

- **`nested_sampling.py`**: Main algorithm implementation
  - `nested_sampling(n, a, N, verbose)`: Estimates P(mean >= a) for n-dimensional Gaussian
  
- **`utils.py`**: Theoretical benchmark calculations
  - `gaussian_tail_exact(a, n)`: Exact probability using CDF
  - `mills_approximation(a, n)`: Mills ratio approximation
  
- **`plotting.py`**: Visualization functions
  - `plot_diagnostics(results, n, a)`: Creates 4-panel diagnostic plot

### Examples (`examples/`)

- **`run_main_example.py`**: Full n-dimensional example (n=10, a=2.0)
  - Reproduces paper results
  - Compares with theoretical values
  - Generates diagnostic plots
  
- **`run_1d_example.py`**: Simple 1D case
  - Estimates P(X >= 3.0) where X ~ N(0,1)
  - Good for understanding the algorithm
  
- **`run_median_validation.py`**: Validates median CLT
  - Demonstrates asymptotic normality of sample median
  - Includes Q-Q plot and KS test

## Algorithm Output

Each iteration prints:
```
Iteration k:
  u_k = ...           # Median threshold
  Var(median) = ...   # Variance of median estimator
  Survivors: .../...  # Number of survivors (should be ~50%)
  Accept rate: ...    # Gibbs sampler acceptance rate
```

Final output:
```
K = ...
P = (1/2)^K = ...
```

## Modifying Parameters

Edit the `main()` function in each example script:

```python
n = 10      # Dimension
a = 2.0     # Threshold for sample mean
N = 2**18   # Samples per level
```

## Requirements

- Python 3.7+
- numpy
- scipy
- matplotlib

This implementation corresponds to:
- **Algorithm**: Main algorithm in `nested_sampling.py`
- **Section 7 (Gaussian case)**: `run_main_example.py`
- **Median CLT**: `run_median_validation.py`
