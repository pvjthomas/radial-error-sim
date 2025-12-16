# Radial Error Simulation

A Python toolkit for Monte Carlo simulation and statistical analysis of radial error distributions arising from 2D Gaussian positional errors.

## Overview

When independent Gaussian errors occur in two dimensions (x and y), the resulting radial error r = √(x² + y²) follows a non-Gaussian distribution (typically Rayleigh-like). This project provides tools to:

- **Simulate** radial error samples from 2D Gaussian inputs
- **Analyze** the resulting distributions with robust statistical methods
- **Visualize** results with comprehensive plotting utilities

## Installation

```bash
pip install numpy scipy matplotlib
```

## Quick Start

```python
from radial_error.simulation import simulate_radial_error
from radial_error.stats import analyze_distribution
from radial_error.plots import Fitplot

# Simulate radial error from 2D Gaussian errors
r = simulate_radial_error(
    mu_x=0.0, sigma_x=10.0,
    mu_y=0.0, sigma_y=20.0,
    n_samples=10_000,
    seed=42
)

# Analyze the distribution
summary = analyze_distribution(r, distribution="normal", verbose=True)

# Visualize with histogram and fitted PDF
Fitplot(r, distribution="normal", title="Radial Error Distribution")
```

## Core Modules

### `simulation.py` - Monte Carlo Simulation

Core functions for generating radial error samples:

- `sample_xy()` - Sample independent x, y from Gaussian distributions
- `compute_radial_error()` - Compute r = √(x² + y²) from x, y arrays
- `simulate_radial_error()` - End-to-end simulation returning radial error array

**Example:**
```python
from radial_error.simulation import sample_xy, compute_radial_error

x, y = sample_xy(mu_x=0, sigma_x=1, mu_y=0, sigma_y=1, n_samples=1000, seed=42)
r = compute_radial_error(x, y)
```

### `stats.py` - Statistical Analysis

Robust statistical analysis tools with edge case handling:

- `safe_fit()` - Robust distribution fitting (normal/exponential) with fallbacks
- `test_normality()` - Shapiro-Wilk normality test with optional QQ plots
- `analyze_distribution()` - Comprehensive descriptive analysis with intervals
- `bootstrap_summary_stats()` - Bootstrap confidence intervals for summary statistics
- `predict_intervals()` - Prediction and tolerance intervals
- `percentile_table()` - Percentile summaries

**Example:**
```python
from radial_error.stats import analyze_distribution, bootstrap_summary_stats

# Comprehensive analysis
summary = analyze_distribution(r, distribution="normal", verbose=True)

# Bootstrap confidence intervals
bootstrap_results = bootstrap_summary_stats(r, n_resamples=1000, ci_level=0.95)
```

### `plots.py` - Visualization

Comprehensive plotting utilities for exploratory analysis:

- `auto_hist()` - Histograms with optional KDE, CDF, and rug plots
- `Fitplot()` / `Fitplot_r()` - Histogram with fitted PDF overlay
- `QQplot()` - Q-Q plots for normality assessment
- `Boxplot()` - Boxplots with mean CI and shortest half intervals
- `plot_spc()` - Statistical Process Control charts with moving range

**Example:**
```python
from radial_error.plots import auto_hist, Fitplot, Boxplot

# Histogram with KDE and CDF
auto_hist(r, kde=True, cdf=True, rug=True)

# Fit plot with normal distribution
Fitplot(r, distribution="normal", title="Radial Error")

# Boxplot with statistics
Boxplot(r, labels=["Radial Error"], title="Radial Error Distribution")
```

## Use Cases

- **Measurement System Accuracy**: Analyze positioning errors in 2D measurement systems
- **Quality Control**: Characterize radial deviations in manufacturing processes
- **Statistical Analysis**: Study the distribution of radial errors from independent Gaussian components
- **Process Monitoring**: Use SPC charts to monitor radial error over time

## Features

- **Robust Error Handling**: Functions gracefully handle edge cases (empty data, NaNs, small samples)
- **Flexible RNG Control**: Support for seeds or external `numpy.random.Generator` instances
- **Small Sample Support**: Bootstrap methods and warnings for small sample sizes
- **Comprehensive Statistics**: Confidence intervals, prediction intervals, tolerance intervals
- **Rich Visualizations**: Multiple plot types with customizable overlays

## Project Structure

```
radial-error-sim/
├── radial_error/
│   ├── __init__.py
│   ├── simulation.py    # Monte Carlo simulation
│   ├── stats.py         # Statistical analysis
│   └── plots.py         # Visualization utilities
├── tests/
│   └── test_simulation.py
├── notebooks/
│   └── monte_carlo_visualization.ipynb
└── README.md
```

## Dependencies

- `numpy` - Numerical computations
- `scipy` - Statistical functions and distributions
- `matplotlib` - Plotting and visualization

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

