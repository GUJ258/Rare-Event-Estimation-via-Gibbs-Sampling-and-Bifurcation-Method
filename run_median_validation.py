"""
Median CLT Validation

Validates the asymptotic normality of the sample median:
    sqrt(N) * (median - true_median) ~ N(0, 1/(4*f(median)^2))

Usage:
    python run_median_validation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def validate_median_clt():
    """Validate asymptotic normality of sample median."""
    
    num_reps = 10000
    sample_size = 10000
    
    print("="*70)
    print("MEDIAN CLT VALIDATION")
    print("="*70)
    print(f"Replications: {num_reps:,}")
    print(f"Sample size: {sample_size:,}")
    print()
    
    # Generate data and compute medians
    print("Generating data...")
    data = np.random.randn(num_reps, sample_size)
    
    print("Computing medians...")
    medians = np.median(data, axis=1)
    
    # Theoretical values for N(0,1)
    true_median = 0.0
    f_at_median = 1.0 / np.sqrt(2 * np.pi)  # Density at 0
    theoretical_std = np.sqrt(1.0 / (4 * sample_size * f_at_median**2))
    
    # Empirical values
    empirical_mean = np.mean(medians)
    empirical_std = np.std(medians)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Mean:")
    print(f"  Theoretical: {true_median:.8f}")
    print(f"  Empirical:   {empirical_mean:.8f}")
    print()
    print(f"Std Dev:")
    print(f"  Theoretical: {theoretical_std:.8f}")
    print(f"  Empirical:   {empirical_std:.8f}")
    print()
    
    # Kolmogorov-Smirnov test
    standardized = (medians - true_median) / theoretical_std
    ks_stat, ks_pval = stats.kstest(standardized, 'norm')
    print(f"KS test:")
    print(f"  Statistic: {ks_stat:.6f}")
    print(f"  P-value:   {ks_pval:.6f}")
    if ks_pval > 0.05:
        print("  ✓ Cannot reject normality")
    else:
        print("  ✗ Reject normality")
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram with theoretical PDF
    ax = axes[0]
    x = np.linspace(medians.min(), medians.max(), 200)
    pdf = stats.norm.pdf(x, loc=true_median, scale=theoretical_std)
    ax.hist(medians, bins=50, density=True, alpha=0.7, edgecolor='black', label='Empirical')
    ax.plot(x, pdf, 'r-', linewidth=2, label='Theoretical')
    ax.axvline(true_median, color='g', linestyle='--', linewidth=1.5, label='True median')
    ax.set_xlabel('Sample Median')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Sample Medians')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax = axes[1]
    stats.probplot(standardized, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normal)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('median_validation.png', dpi=150)
    print("\nSaved: median_validation.png")
    plt.show()


if __name__ == "__main__":
    validate_median_clt()
