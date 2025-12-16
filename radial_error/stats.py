from __future__ import annotations

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import (
    skew,
    kurtosis,
    bootstrap,
    norm,
    expon,
    t,
    chi2,
    shapiro,
    probplot,
)


def safe_fit(data, distribution="normal", fallback=(0.0, 1.0), warn=False):
    """
    Safely fit a statistical distribution to data with robust error handling.

    This function fits either a normal or exponential distribution to the data,
    with fallback behavior for edge cases (empty data, NaNs, infinite values).
    For small sample sizes, uses sample standard deviation instead of MLE
    for normal distribution fitting.

    Parameters
    ----------
    data : array-like
        One-dimensional array of data to fit. Will be converted to numpy array.
    distribution : {'normal', 'exponential'}, optional
        Type of distribution to fit. Default is 'normal'.
    fallback : tuple, optional
        Tuple of parameters to return if fitting fails. For normal distribution,
        should be (mu, sigma). For exponential, should be (loc, scale).
        Default is (0.0, 1.0).
    warn : bool, optional
        If True, print warnings about small sample sizes or fitting failures.
        Default is False (for test-friendliness).

    Returns
    -------
    tuple
        Distribution parameters:
        - For 'normal': (mu, sigma) where mu is mean and sigma is standard deviation
        - For 'exponential': (loc, scale) where loc is location and scale is mean

    Raises
    ------
    ValueError
        If distribution is not 'normal' or 'exponential'.

    Examples
    --------
    >>> data = np.random.normal(5, 2, size=1000)
    >>> mu, sigma = safe_fit(data, distribution="normal")
    >>> print(f"μ={mu:.2f}, σ={sigma:.2f}")
    μ=5.00, σ=2.00

    >>> # Handles edge cases gracefully
    >>> mu, sigma = safe_fit([], distribution="normal")
    >>> print(f"Fallback: μ={mu}, σ={sigma}")
    Fallback: μ=0.0, σ=1.0

    Notes
    -----
    - For normal distribution with n < 50, uses sample standard deviation
      (ddof=1) instead of MLE estimate for better small-sample behavior.
    - Returns fallback values for empty data, all-NaN data, or fitting failures.
    """
    data = np.asarray(data)

    if data.size == 0 or np.all(np.isnan(data)):
        if warn:
            print("⚠️ Fit failed: empty or all NaNs — using fallback.")
        return fallback

    if distribution == "normal":
        if warn:
            if data.size < 30:
                print("⚠️ Mean may be unreliable — recommended n > 30")
            if data.size < 50:
                print("⚠️ Sigma may be unreliable — recommended n > 50")
        try:
            mu, sigma = norm.fit(data)
            if not np.isfinite(mu) or not np.isfinite(sigma):
                raise ValueError("Non-finite result from norm.fit()")
            # Your original behavior: use sample std for small n
            if data.size < 50:
                sigma = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
            return float(mu), float(sigma)
        except Exception as e:
            if warn:
                print(f"⚠️ norm.fit() error: {e} — using fallback.")
            return fallback

    if distribution == "exponential":
        try:
            loc, scale = expon.fit(data)
            if not np.isfinite(loc) or not np.isfinite(scale):
                raise ValueError("Non-finite result from expon.fit()")
            return float(loc), float(scale)
        except Exception as e:
            if warn:
                print(f"⚠️ expon.fit() error: {e} — using fallback.")
            return fallback

    raise ValueError(f"Unsupported distribution: {distribution!r}")


def test_normality(data, alpha=0.05, plot=True, verbose=True, name="data", ax=None, show=True):
    """
    Perform Shapiro-Wilk normality test with optional Q-Q plot visualization.

    The Shapiro-Wilk test is a statistical test of the null hypothesis that
    data comes from a normal distribution. This function performs the test and
    optionally creates a Q-Q plot for visual assessment.

    Parameters
    ----------
    data : array-like
        One-dimensional array of data to test. Will be converted to numpy array.
    alpha : float, optional
        Significance level for the test. Default is 0.05.
    plot : bool, optional
        If True, create a Q-Q plot. Default is True.
    verbose : bool, optional
        If True, print test results. Default is True.
    name : str, optional
        Name/label for the data (used in plots and output). Default is "data".
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, creates a new figure. Default is None.
    show : bool, optional
        If True, display the plot immediately. Default is True.

    Returns
    -------
    dict
        Dictionary containing:
        - 'statistic' (float): Shapiro-Wilk test statistic W
        - 'p_value' (float): p-value of the test
        - 'is_normal' (bool): True if p > alpha (fail to reject normality)

    Examples
    --------
    >>> data = np.random.normal(0, 1, size=100)
    >>> result = test_normality(data, name="sample", verbose=True)
    --- Shapiro–Wilk Normality Test on 'sample' ---
    W statistic: 0.9912
    p-value:     0.5678
    ✅ Fail to reject H₀ (α = 0.05)
    >>> print(f"Normal: {result['is_normal']}")
    Normal: True

    Notes
    -----
    - The Shapiro-Wilk test works best for sample sizes between 3 and 5000.
    - For very large samples, even small deviations from normality may be detected.
    """
    data = np.asarray(data)
    stat, p = shapiro(data)
    is_normal = bool(p > alpha)

    result = {"statistic": float(stat), "p_value": float(p), "is_normal": is_normal}

    if verbose:
        print(f"--- Shapiro–Wilk Normality Test on '{name}' ---")
        print(f"W statistic: {stat:.4f}")
        print(f"p-value:     {p:.4f}")
        print(("✅ Fail to reject H₀" if is_normal else "🚫 Reject H₀") + f" (α = {alpha})")

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = ax.figure
        probplot(data, dist="norm", plot=ax)
        ax.set_title(f"QQ Plot: {name}")
        ax.grid(True)
        fig.tight_layout()
        if show:
            plt.show()

    return result


def bootstrap_summary_stats(
    data,
    n_resamples=1000,
    ci_level=0.95,
    method="percentile",
    random_state=None,
    verbose=True,
):
    """
    Compute bootstrap confidence intervals for common summary statistics.

    Uses the bootstrap resampling method to estimate confidence intervals for
    mean, median, variance, skewness, and kurtosis. Bootstrap is particularly
    useful for small samples or when the distribution is unknown.

    Parameters
    ----------
    data : array-like
        One-dimensional array of data. Will be converted to numpy array.
    n_resamples : int, optional
        Number of bootstrap resamples to perform. Default is 1000.
    ci_level : float, optional
        Confidence level for intervals (between 0 and 1). Default is 0.95.
    method : str, optional
        Bootstrap method. Currently only 'percentile' is supported.
        Default is 'percentile'.
    random_state : int or None, optional
        Random seed for reproducible bootstrap resampling. Default is None.
    verbose : bool, optional
        If True, print warnings about small sample sizes. Default is True.

    Returns
    -------
    dict
        Dictionary with keys 'mean', 'median', 'variance', 'skewness', 'kurtosis'.
        Each value is a dict with:
        - 'value' (float): Observed statistic value
        - 'ci_lower' (float): Lower bound of confidence interval
        - 'ci_upper' (float): Upper bound of confidence interval

    Raises
    ------
    ValueError
        If data is not 1D or is empty.

    Examples
    --------
    >>> data = np.random.normal(5, 2, size=30)
    >>> results = bootstrap_summary_stats(data, n_resamples=1000, ci_level=0.95)
    >>> print(f"Mean: {results['mean']['value']:.2f} "
    ...       f"[{results['mean']['ci_lower']:.2f}, {results['mean']['ci_upper']:.2f}]")
    Mean: 5.12 [4.45, 5.79]

    Notes
    -----
    - Bootstrap results may be unstable for very small samples (n < 10).
    - Skewness and kurtosis estimates are often unreliable for n < 30.
    """
    data = np.asarray(data)

    if data.ndim != 1:
        raise ValueError("Data must be 1D array-like")

    n = data.size
    if n == 0:
        raise ValueError("Data must be non-empty")

    if verbose:
        if n < 10:
            print("⚠️ Sample size < 10. Bootstrap results may be unstable.")
        elif n < 20:
            print("ℹ️ Mean/median bootstrap may be usable, interpret with caution.")
        elif n < 30:
            print("ℹ️ Skew/kurtosis estimates often unstable below n=30.")

    ci_kwargs = dict(
        confidence_level=ci_level,
        n_resamples=n_resamples,
        method=method,
        random_state=random_state,
    )

    def stat_ci(stat_func):
        res = bootstrap((data,), stat_func, **ci_kwargs)
        return {
            "value": float(stat_func(data)),
            "ci_lower": float(res.confidence_interval.low),
            "ci_upper": float(res.confidence_interval.high),
        }

    return {
        "mean": stat_ci(np.mean),
        "median": stat_ci(np.median),
        "variance": stat_ci(np.var),
        "skewness": stat_ci(lambda x: skew(x, bias=False)),
        "kurtosis": stat_ci(lambda x: kurtosis(x, fisher=True)),
    }


def analyze_distribution(x, distribution=None, verbose=True, return_dict=True):
    """
    Comprehensive statistical analysis of a univariate distribution.

    Computes descriptive statistics (mean, median, std, skewness, kurtosis),
    assesses distribution shape, and optionally calculates confidence intervals,
    prediction intervals, and tolerance intervals. Automatically suggests
    bootstrap methods for small samples when appropriate.

    Parameters
    ----------
    x : array-like
        One-dimensional array of data to analyze. Will be converted to numpy array.
    distribution : {'normal', 'exponential'} or None, optional
        Assumed distribution type for interval calculations. If None, only
        descriptive statistics are computed. Default is None.
    verbose : bool, optional
        If True, print detailed analysis results. Default is True.
    return_dict : bool, optional
        If True, return summary dictionary. If False, return None (useful when
        only printing is desired). Default is True.

    Returns
    -------
    dict or None
        Summary dictionary containing:
        - 'n' (int): Sample size
        - 'mean', 'median', 'stdev', 'sem' (float): Basic statistics
        - 'cv_percent' (float): Coefficient of variation
        - 'skewness', 'skew_desc' (float, str): Skewness and description
        - 'kurtosis_excess', 'kurt_desc' (float, str): Excess kurtosis and description
        - 'conclusion' (str): Distribution shape assessment
        - 'likely_normal' (bool): Whether data appears Gaussian-like
        - 'intervals' (dict): Confidence/prediction/tolerance intervals (if distribution provided)
        - 'bootstrap' (dict or None): Bootstrap results for small samples (if applicable)
        Returns None if return_dict=False.

    Raises
    ------
    ValueError
        If x is empty.

    Examples
    --------
    >>> data = np.random.normal(5, 2, size=100)
    >>> summary = analyze_distribution(data, distribution="normal", verbose=True)
    --- Analysis of normal ---
    N: 100
    Mean: 5.023
    Median: 5.012
    Stdev: 1.987
    ...
    >>> print(f"Skewness: {summary['skewness']:.3f}")
    Skewness: 0.123

    Notes
    -----
    - For n < 15, automatically computes bootstrap summary statistics.
    - For n < 30, suggests comparing results to bootstrap.
    - Skewness requires n > 2, kurtosis requires n > 3.
    - Interval calculations use t-distribution for normal, chi-squared for exponential.
    """
    x = np.asarray(x)
    n = x.size
    if n == 0:
        raise ValueError("x must be non-empty")

    x_mean = float(np.mean(x))
    x_med = float(np.median(x))
    x_std = float(np.std(x, ddof=1)) if n > 1 else 0.0

    s = float(skew(x, bias=False)) if n > 2 else float("nan")
    k = float(kurtosis(x, fisher=True)) if n > 3 else float("nan")

    likely_normal = True
    if np.isfinite(s) and abs(s) >= 1:
        likely_normal = False
    if np.isfinite(k) and k >= 2:
        likely_normal = False

    # CV can be unstable if mean ~ 0
    cv = float("inf") if abs(x_mean) < 1e-12 else float(100 * x_std / x_mean)

    # Determine descriptions
    if not np.isfinite(s):
        skew_desc = "insufficient n for skewness"
    elif abs(s) < 0.5:
        skew_desc = "approximately symmetric"
    elif 0.5 <= abs(s) < 1:
        skew_desc = "moderately " + ("right" if s > 0 else "left") + " skewed"
    else:
        skew_desc = "highly " + ("right" if s > 0 else "left") + " skewed"

    if not np.isfinite(k):
        kurt_desc = "insufficient n for kurtosis"
    elif k < -1:
        kurt_desc = "very light-tailed"
    elif -1 <= k < 0.5:
        kurt_desc = "normal-tailed"
    elif 0.5 <= k < 2:
        kurt_desc = "moderately heavy-tailed"
    else:
        kurt_desc = "heavy-tailed"

    if np.isfinite(s) and np.isfinite(k) and abs(s) < 0.5 and -1 <= k < 0.5:
        conclusion = "Gaussian-like. Gaussian fit likely appropriate."
    elif (np.isfinite(s) and abs(s) >= 1) or (np.isfinite(k) and k >= 2):
        conclusion = "Not Gaussian. Consider Laplace or t-distribution."
    else:
        conclusion = "Some deviation from Gaussian. Consider testing Laplace or logistic."

    summary = {
        "n": int(n),
        "mean": x_mean,
        "median": x_med,
        "stdev": x_std,
        "sem": float(x_std / np.sqrt(n)) if n > 0 else float("nan"),
        "cv_percent": cv,
        "skewness": s,
        "skew_desc": skew_desc,
        "kurtosis_excess": k,
        "kurt_desc": kurt_desc,
        "conclusion": conclusion,
        "likely_normal": bool(likely_normal),
        "intervals": {},
        "bootstrap": None,
    }

    if verbose:
        print(f"--- Analysis of {distribution or 'distribution'} ---")
        print(f"N: {n}")
        print(f"Mean: {x_mean:.3f}")
        print(f"Median: {x_med:.3f}")
        print(f"Stdev: {x_std:.3f}")
        print(f"Std Error Mean: {summary['sem']:.3f}")
        print(f"CV: {cv:.1f}" if np.isfinite(cv) else "CV: inf (mean ~ 0)")
        print(f"Skewness: {s:.3f} ({skew_desc})" if np.isfinite(s) else f"Skewness: {skew_desc}")
        print(f"Kurtosis: {k:.3f} ({kurt_desc})" if np.isfinite(k) else f"Kurtosis: {kurt_desc}")
        print(f"Conclusion: {conclusion}")

    # Your original behavior: suggest bootstrap at small n
    if likely_normal:
        if n < 15:
            if verbose:
                print("< 15 samples, try bootstrap")
            summary["bootstrap"] = bootstrap_summary_stats(x, verbose=False)
        elif n < 30:
            if verbose:
                print("< 30 samples, compare results to bootstrap")
            summary["bootstrap"] = bootstrap_summary_stats(x, verbose=False)

    # Interval calculations if distribution is provided
    if distribution is not None:
        alpha = 0.05
        coverage = 0.99

        if distribution == "normal":
            if n < 2:
                intervals = {}
            else:
                t_crit = float(t.ppf(1 - alpha / 2, df=n - 1))
                ci_low = x_mean - t_crit * x_std / sqrt(n)
                ci_high = x_mean + t_crit * x_std / sqrt(n)

                df = n - 1
                chi2_low = float(chi2.ppf(alpha / 2, df))
                chi2_high = float(chi2.ppf(1 - alpha / 2, df))
                ci_sigma_low = sqrt((df * x_std**2) / chi2_high)
                ci_sigma_high = sqrt((df * x_std**2) / chi2_low)

                pred_low = x_mean - t_crit * x_std * sqrt(1 + 1 / n)
                pred_high = x_mean + t_crit * x_std * sqrt(1 + 1 / n)

                z = float(norm.ppf((1 + coverage) / 2))
                tol_low = x_mean - z * x_std
                tol_high = x_mean + z * x_std

                intervals = {
                    "mu_ci_95": (ci_low, ci_high),
                    "sigma_ci_95": (ci_sigma_low, ci_sigma_high),
                    "pred_interval_95": (pred_low, pred_high),
                    "tol_interval_99_approx": (tol_low, tol_high),
                }

        elif distribution == "exponential":
            # CI for mean of exponential, plus rough PI/TI per your original approach
            df = 2 * n
            chi2_low = float(chi2.ppf(alpha / 2, df))
            chi2_high = float(chi2.ppf(1 - alpha / 2, df))

            ci_low = (2 * n * x_mean) / chi2_high
            ci_high = (2 * n * x_mean) / chi2_low

            pred_low = 0.0
            pred_high = -x_mean * np.log(alpha)

            z = -np.log(1 - coverage)
            tol_low = 0.0
            tol_high = (2 * n * x_mean * z) / float(chi2.ppf(1 - alpha, 2 * n))

            intervals = {
                "mu_ci_95": (float(ci_low), float(ci_high)),
                "pred_interval_95": (float(pred_low), float(pred_high)),
                "tol_interval_99_approx": (float(tol_low), float(tol_high)),
            }
        else:
            intervals = {}

        summary["intervals"] = intervals

        if verbose and intervals:
            if "sigma_ci_95" in intervals:
                lo, hi = intervals["sigma_ci_95"]
                print(f"95% Confidence Interval for σ: [{lo:.3f}, {hi:.3f}]")
            lo, hi = intervals["mu_ci_95"]
            print(f"95% Confidence Interval for μ: [{lo:.3f}, {hi:.3f}]")
            lo, hi = intervals["pred_interval_95"]
            print(f"95% Prediction Interval: [{lo:.3f}, {hi:.3f}]")
            lo, hi = intervals["tol_interval_99_approx"]
            print(f"Approximate 99% Tolerance Interval: [{lo:.3f}, {hi:.3f}]")

    return summary if return_dict else None


def percentile_table(data, percentiles=None):
    """
    Print a formatted table of percentiles for the data.

    Computes and displays percentiles in a formatted table, useful for
    understanding the distribution of values at various quantiles.

    Parameters
    ----------
    data : array-like
        One-dimensional array of data. Will be converted to numpy array.
    percentiles : list of float or None, optional
        List of percentiles to compute (values between 0 and 100).
        If None, uses default: [0, 0.5, 1, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 99.5, 100].
        Default is None.

    Examples
    --------
    >>> data = np.random.normal(5, 2, size=1000)
    >>> percentile_table(data)
    Percentile |  Raw Score
    -------------------------
         0.000 |     -1.234
         0.500 |      0.567
    ...

    >>> # Custom percentiles
    >>> percentile_table(data, percentiles=[25, 50, 75])
    Percentile |  Raw Score
    -------------------------
        25.000 |      3.456
        50.000 |      5.012
        75.000 |      6.789
    """
    data = np.asarray(data)
    if percentiles is None:
        percentiles = [0, 0.5, 1, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 99.5, 100]

    scores = np.percentile(data, percentiles)

    print(f"{'Percentile':>10} | {'Raw Score':>10}")
    print("-" * 25)
    for p, s in zip(percentiles, scores):
        print(f"{p:10.3f} | {s:10.3f}")


def predict_intervals(mu, sigma, n_sample, n_future=1, alpha=0.05, coverage=0.95):
    """
    Compute prediction intervals and tolerance intervals under normal distribution assumption.

    Calculates three types of intervals:
    1. Prediction interval for a single future observation
    2. Bonferroni-corrected prediction interval for multiple future observations
    3. Approximate tolerance interval for a specified coverage proportion

    Parameters
    ----------
    mu : float
        Sample mean (estimate of population mean μ).
    sigma : float
        Sample standard deviation (estimate of population standard deviation σ).
    n_sample : int
        Sample size used to estimate mu and sigma. Must be >= 2.
    n_future : int, optional
        Number of future observations for Bonferroni correction. Default is 1.
    alpha : float, optional
        Significance level (1 - confidence level). Default is 0.05 (95% confidence).
    coverage : float, optional
        Coverage proportion for tolerance interval (between 0 and 1). Default is 0.95.

    Returns
    -------
    dict
        Dictionary with three keys:
        - 'Prediction (1 value)': (lower, upper) tuple for single observation
        - 'Prediction (n_future values, Bonferroni)': (lower, upper) tuple for multiple observations
        - 'Tolerance Interval (coverage%, approx)': (lower, upper) tuple for tolerance interval

    Raises
    ------
    ValueError
        If n_sample < 2.

    Examples
    --------
    >>> # From sample statistics
    >>> mu, sigma = 5.0, 2.0
    >>> intervals = predict_intervals(mu, sigma, n_sample=30, n_future=5, alpha=0.05)
    >>> print(intervals['Prediction (1 value)'])
    (0.876, 9.124)

    Notes
    -----
    - Prediction intervals account for uncertainty in both the mean and the new observation.
    - Bonferroni correction adjusts alpha for multiple future observations.
    - Tolerance intervals are approximate and assume normal distribution.
    - Uses t-distribution for prediction intervals (accounts for estimated parameters).
    """
    mu = float(mu)
    sigma = float(sigma)

    if n_sample < 2:
        raise ValueError("n_sample must be >= 2")

    t_crit_1 = float(t.ppf(1 - alpha / 2, df=n_sample - 1))
    half_width_1 = t_crit_1 * sigma * sqrt(1 + 1 / n_sample)

    adj_alpha = alpha / max(int(n_future), 1)
    t_crit_bonf = float(t.ppf(1 - adj_alpha / 2, df=n_sample - 1))
    half_width_bonf = t_crit_bonf * sigma * sqrt(1 + 1 / n_sample)

    z = float(norm.ppf((1 + coverage) / 2))
    half_width_tol = z * sigma

    return {
        "Prediction (1 value)": (mu - half_width_1, mu + half_width_1),
        f"Prediction ({n_future} values, Bonferroni)": (mu - half_width_bonf, mu + half_width_bonf),
        f"Tolerance Interval ({int(coverage * 100)}%, approx)": (mu - half_width_tol, mu + half_width_tol),
    }
