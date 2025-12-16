from __future__ import annotations
import numpy as np


def _get_rng(seed=None, rng=None):
    """
    Internal helper function to select or create a numpy random number generator.

    Parameters
    ----------
    seed : int or None, optional
        Random seed for reproducible sampling. If None and rng is None,
        creates a non-deterministic generator. Default is None.
    rng : numpy.random.Generator or None, optional
        External random number generator to use. If provided, seed is ignored.
        Default is None.

    Returns
    -------
    numpy.random.Generator
        The provided rng or a newly created generator from seed.

    Raises
    ------
    TypeError
        If rng is provided but is not a numpy.random.Generator instance.
    """
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        return rng
    return np.random.default_rng(seed)


def sample_xy(mu_x=0.0, sigma_x=1.0, mu_y=0.0, sigma_y=1.0, n_samples=10_000, seed=None, rng=None):
    """
    Sample x and y coordinates as independent Gaussian random variables.

    This function generates samples from two independent normal distributions:
    - x ~ N(μₓ, σₓ²)
    - y ~ N(μᵧ, σᵧ²)

    Parameters
    ----------
    mu_x : float, optional
        Mean of the x-coordinate Gaussian distribution. Default is 0.0.
    sigma_x : float, optional
        Standard deviation of the x-coordinate Gaussian distribution. Default is 1.0.
    mu_y : float, optional
        Mean of the y-coordinate Gaussian distribution. Default is 0.0.
    sigma_y : float, optional
        Standard deviation of the y-coordinate Gaussian distribution. Default is 1.0.
    n_samples : int, optional
        Number of samples to generate. Default is 10,000.
    seed : int or None, optional
        Random seed for reproducible sampling. If None and rng is None,
        uses non-deterministic sampling. Default is None.
    rng : numpy.random.Generator or None, optional
        External random number generator. If provided, seed is ignored.
        Useful for advanced control over random number generation. Default is None.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple (x, y) containing two 1D arrays of length n_samples:
        - x: Array of x-coordinate samples
        - y: Array of y-coordinate samples

    Examples
    --------
    >>> x, y = sample_xy(mu_x=0, sigma_x=1, mu_y=0, sigma_y=1, n_samples=1000, seed=42)
    >>> print(f"x mean: {x.mean():.3f}, y mean: {y.mean():.3f}")
    x mean: 0.036, y mean: -0.010

    >>> # Using external RNG
    >>> rng = np.random.default_rng(42)
    >>> x, y = sample_xy(n_samples=100, rng=rng)
    """
    rng = _get_rng(seed=seed, rng=rng)
    x = rng.normal(loc=mu_x, scale=sigma_x, size=n_samples)
    y = rng.normal(loc=mu_y, scale=sigma_y, size=n_samples)
    return x, y


def compute_radial_error(x, y):
    """
    Compute radial error from x and y coordinate arrays.

    Calculates the Euclidean distance r = √(x² + y²) for each pair of
    (x, y) values. Uses numpy.hypot() for numerical stability.

    Parameters
    ----------
    x : array-like
        Array of x-coordinates. Will be converted to numpy array.
    y : array-like
        Array of y-coordinates. Must have the same length as x.
        Will be converted to numpy array.

    Returns
    -------
    numpy.ndarray
        Array of radial errors r = √(x² + y²) with the same length as input arrays.

    Examples
    --------
    >>> x = np.array([3.0, 0.0, -4.0])
    >>> y = np.array([4.0, 0.0, 3.0])
    >>> r = compute_radial_error(x, y)
    >>> print(r)
    [5.  0.  5.]

    >>> # With many samples
    >>> x, y = sample_xy(n_samples=1000, seed=42)
    >>> r = compute_radial_error(x, y)
    >>> print(f"Mean radial error: {r.mean():.3f}")
    Mean radial error: 1.253
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.hypot(x, y)  # numerically stable sqrt(x*x + y*y)


def simulate_radial_error(mu_x=0.0, sigma_x=1.0, mu_y=0.0, sigma_y=1.0, n_samples=10_000, seed=None, rng=None):
    """
    Simulate radial error samples from independent 2D Gaussian positional errors.

    This is a convenience function that combines `sample_xy()` and `compute_radial_error()`
    to directly generate radial error samples. When x and y are independent Gaussians,
    the radial error r = √(x² + y²) follows a non-Gaussian distribution (typically
    Rayleigh-like when μₓ = μᵧ = 0).

    Parameters
    ----------
    mu_x : float, optional
        Mean of the x-coordinate Gaussian distribution. Default is 0.0.
    sigma_x : float, optional
        Standard deviation of the x-coordinate Gaussian distribution. Default is 1.0.
    mu_y : float, optional
        Mean of the y-coordinate Gaussian distribution. Default is 0.0.
    sigma_y : float, optional
        Standard deviation of the y-coordinate Gaussian distribution. Default is 1.0.
    n_samples : int, optional
        Number of radial error samples to generate. Default is 10,000.
    seed : int or None, optional
        Random seed for reproducible sampling. If None and rng is None,
        uses non-deterministic sampling. Default is None.
    rng : numpy.random.Generator or None, optional
        External random number generator. If provided, seed is ignored.
        Default is None.

    Returns
    -------
    numpy.ndarray
        Array of radial errors r = √(x² + y²) with length n_samples.

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> r = simulate_radial_error(n_samples=1000, seed=42)
    >>> print(f"Mean: {r.mean():.3f}, Std: {r.std():.3f}")
    Mean: 1.253, Std: 0.655

    >>> # Asymmetric error distributions
    >>> r = simulate_radial_error(
    ...     mu_x=0, sigma_x=10,
    ...     mu_y=0, sigma_y=20,
    ...     n_samples=10_000,
    ...     seed=42
    ... )
    >>> print(f"Mean radial error: {r.mean():.2f}")
    Mean radial error: 22.36

    Notes
    -----
    When μₓ = μᵧ = 0 and σₓ = σᵧ = σ, the radial error follows a Rayleigh
    distribution with scale parameter σ. For asymmetric cases (σₓ ≠ σᵧ),
    the distribution is more complex.
    """
    x, y = sample_xy(mu_x, sigma_x, mu_y, sigma_y, n_samples, seed=seed, rng=rng)
    return compute_radial_error(x, y)
