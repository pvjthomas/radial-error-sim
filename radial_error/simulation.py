import numpy as np

def sample_xy(mu_x=0, sigma_x=1, mu_y=0, sigma_y=1, n_samples=10000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.random.normal(mu_x, sigma_x, n_samples)
    y = np.random.normal(mu_y, sigma_y, n_samples)
    return x, y

def compute_radial_error(x, y):
    return np.sqrt(x**2 + y**2)

def simulate_radial_error(mu_x=0, sigma_x=1, mu_y=0, sigma_y=1, n_samples=10000, seed=None):
    x, y = sample_xy(mu_x, sigma_x, mu_y, sigma_y, n_samples, seed)
    r = compute_radial_error(x, y)
    return r