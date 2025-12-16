import numpy as np
import pytest

from radial_error.stats import safe_fit


def test_fit_with_good_data():
    rng = np.random.default_rng(0)
    data = rng.normal(5, 2, size=1000)
    mu, sigma = safe_fit(data, distribution="normal")
    assert abs(mu - 5) < 0.2
    assert abs(sigma - 2) < 0.2


def test_fit_with_empty_data_returns_fallback():
    mu, sigma = safe_fit([], distribution="normal")
    assert (mu, sigma) == (0.0, 1.0)


def test_fit_with_all_nan_returns_fallback():
    mu, sigma = safe_fit([np.nan, np.nan], distribution="normal")
    assert (mu, sigma) == (0.0, 1.0)


def test_fit_with_infinite_values_returns_fallback():
    mu, sigma = safe_fit([1.0, 2.0, np.inf], distribution="normal")
    assert (mu, sigma) == (0.0, 1.0)


def test_unsupported_distribution_raises():
    with pytest.raises(ValueError):
        safe_fit([1, 2, 3], distribution="nope")
