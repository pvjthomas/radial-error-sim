import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from radial_error.stats import safe_fit
import numpy as np
rng = np.random.default_rng(0)

def test_fit_with_good_data():
    data = rng.normal(5, 2, size=1000)
    mu, sigma = safe_fit(data, distribution='normal')
    assert abs(mu - 5) < 0.2
    assert abs(sigma - 2) < 0.2

def test_fit_with_empty_data():
    mu, sigma = safe_fit([], distribution='normal')
    assert mu == 0.0 and sigma == 1.0

def test_fit_with_all_nan():
    mu, sigma = safe_fit([np.nan, np.nan], distribution='normal')
    assert mu == 0.0 and sigma == 1.0

def test_fit_with_infinite_values():
    data = [1.0, 2.0, np.inf]
    mu, sigma = safe_fit(data, distribution='normal')
    assert mu == 0.0 and sigma == 1.0
