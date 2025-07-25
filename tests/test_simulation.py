import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from radial_error.stats import safe_norm_fit
import numpy as np

def test_fit_with_good_data():
    data = np.random.normal(5, 2, size=1000)
    mu, sigma = safe_norm_fit(data)
    assert abs(mu - 5) < 0.2
    assert abs(sigma - 2) < 0.2

def test_fit_with_empty_data():
    mu, sigma = safe_norm_fit([])
    assert mu == 0.0 and sigma == 1.0

def test_fit_with_all_nan():
    mu, sigma = safe_norm_fit([np.nan, np.nan])
    assert mu == 0.0 and sigma == 1.0

def test_fit_with_infinite_values():
    data = [1.0, 2.0, np.inf]
    mu, sigma = safe_norm_fit(data)
    assert mu == 0.0 and sigma == 1.0
