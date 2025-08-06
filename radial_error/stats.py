from scipy.stats import skew, kurtosis, bootstrap, norm, expon, t, chi2, shapiro, probplot
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import norm, expon

def safe_fit(data, distribution='normal', fallback=(0.0, 1.0)):
    """
    Safely fits a statistical distribution to the data.

    Parameters:
        data         : array-like
            The input data to fit.
        distribution : str
            Distribution name: 'normal' or 'exponential'.
        fallback     : tuple
            Values to return if fitting fails.

    Returns:
        tuple
            Parameters of the fitted distribution (or fallback).
    """
    data = np.asarray(data)

    if len(data) == 0 or np.all(np.isnan(data)):
        print("⚠️ Fit failed: empty or all NaNs — using fallback.")
        return fallback

    if distribution == 'normal':
        if len(data) < 30:
            print("⚠️ Mean may be unreliable — recommended n > 30")
        if len(data) < 50:
            print("⚠️ Sigma may be unreliable — recommended n > 50")
        try:
            mu, sigma = norm.fit(data)
            if not np.isfinite(mu) or not np.isfinite(sigma):
                raise ValueError("Non-finite result from norm.fit()")
            if len(data) < 50:
                sigma = np.std(data,ddof=1)
            return mu, sigma
        except Exception as e:
            print(f"⚠️ norm.fit() error: {e} — using fallback.")
            return fallback

    elif distribution == 'exponential':
        try:
            loc, scale = expon.fit(data)
            if not np.isfinite(loc) or not np.isfinite(scale):
                raise ValueError("Non-finite result from expon.fit()")
            return loc, scale
        except Exception as e:
            print(f"⚠️ expon.fit() error: {e} — using fallback.")
            return fallback

    else:
        raise ValueError(f"❌ Unsupported distribution: {distribution}")


def test_normality(data, alpha=0.05, plot=True, verbose=True, name="data"):
    """
    Run Shapiro–Wilk test and display a QQ plot to assess normality.

    Parameters:
        data    : array-like
        alpha   : significance level (default 0.05)
        plot    : whether to show QQ plot
        verbose : whether to print interpretation
        name    : name of the dataset (for labeling)

    Returns:
        result  : dict with test statistic, p-value, and normality conclusion
    """
    data = np.asarray(data)
    stat, p = shapiro(data)
    is_normal = p > alpha

    result = {
        "statistic": stat,
        "p_value": p,
        "is_normal": is_normal
    }

    if verbose:
        print(f"--- Shapiro–Wilk Normality Test on '{name}' ---")
        print(f"W statistic: {stat:.4f}")
        print(f"p-value:     {p:.4f}")
        if is_normal:
            print(f"✅ Fail to reject H₀ — data looks normal (α = {alpha})")
        else:
            print(f"🚫 Reject H₀ — data is not normal (α = {alpha})")

    if plot:
        plt.figure(figsize=(5, 5))
        probplot(data, dist="norm", plot=plt)
        plt.title(f"QQ Plot: {name}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return result

def bootstrap_summary_stats(data, n_resamples=1000, ci_level=0.95, method='percentile', random_state=None, verbose=True):
    """
    Bootstrap confidence intervals for common summary statistics, with guidance.

    Parameters:
        data         : array-like, 1D data
        n_resamples  : number of bootstrap resamples
        ci_level     : confidence level (e.g., 0.95)
        method       : bootstrap method: 'percentile', 'basic', etc.
        random_state : int or None, for reproducibility
        verbose      : bool, print sample size warnings

    Returns:
        dict with point estimates and confidence intervals
    """
    data = np.asarray(data)

    if data.ndim != 1:
        raise ValueError("Data must be 1D array-like")

    n = len(data)

    if verbose:
        if n < 10:
            print("⚠️ Warning: Sample size < 10. Bootstrap results may be unstable for any statistic.")
        elif n < 20:
            print("ℹ️ Note: Mean/median bootstrap may be usable, but interpret with caution.")
        elif n < 30:
            print("ℹ️ Note: Skew/kurtosis estimates are likely unstable below n=30.")

    ci_kwargs = dict(
        confidence_level=ci_level,
        n_resamples=n_resamples,
        method=method,
        random_state=random_state
    )

    def stat_ci(stat_func):
        res = bootstrap((data,), stat_func, **ci_kwargs)
        return {
            "value": stat_func(data),
            "ci_lower": res.confidence_interval.low,
            "ci_upper": res.confidence_interval.high
        }

    results = {
        "mean": stat_ci(np.mean),
        "median": stat_ci(np.median),
        "variance": stat_ci(np.var),
        "skewness": stat_ci(skew),
        "kurtosis": stat_ci(kurtosis)
    }

    return results

def analyze_distribution(x, name=""):
    likely_normal = True
    x_med = np.median(x)
    s = skew(x, bias=False)
    k = kurtosis(x, fisher=True)  # excess kurtosis

    if len(x) < 200:
        print("⚠️ Skewness may be unreliable — recommended n > 200")
    if len(x) < 500:
        print("⚠️ Kurtosis may be unreliable — recommended n > 500")

    # Interpret skew
    if abs(s) < 0.5:
        skew_desc = "approximately symmetric"
    elif 0.5 <= abs(s) < 1:
        skew_desc = "moderately " + ("right" if s > 0 else "left") + " skewed"
    else:
        skew_desc = "highly " + ("right" if s > 0 else "left") + " skewed"
        likely_normal = np.logical_and(likely_normal,False)

    # Interpret kurtosis
    if k < -1:
        kurt_desc = "very light-tailed"
    elif -1 <= k < 0.5:
        kurt_desc = "normal-tailed"
    elif 0.5 <= k < 2:
        kurt_desc = "moderately heavy-tailed"
    else:
        kurt_desc = "heavy-tailed"
        likely_normal = np.logical_and(likely_normal,False)


    # Conclusion
    if abs(s) < 0.5 and -1 <= k < 0.5:
        conclusion = "Gaussian-like. Gaussian fit likely appropriate."
    elif abs(s) >= 1 or k >= 2:
        conclusion = "Not Gaussian. Consider Laplace or t-distribution."
    else:
        conclusion = "Some deviation from Gaussian. Consider testing Laplace or logistic."

    # Print summary
    print(f"--- Analysis of {name or 'distribution'} ---")
    print(f"Median: {x_med:.3f}")
    print(f"Skewness: {s:.3f} ({skew_desc})")
    print(f"Kurtosis: {k:.3f} ({kurt_desc})")
    print(f"Kurtosis Conclusion: {conclusion}")

    if likely_normal is True:
        if len(x) < 15:
            print('< 15 samples, try bootstrap')
            bootstrap_summary_stats(x, verbose=False)
        else:
            if len(x) < 30:
                print('< 30 samples, compare results to bootstrap')
                bootstrap_summary_stats(x, verbose=False)
            mean_x = np.average(x)
            sigma_x = np.std(x,ddof=1)
            n = len(x)
            alpha = 0.05  # for 95% intervals
            confidence = 1 - alpha
            t_crit = t.ppf(1 - alpha/2, df=n - 1)
            ci_low = mean_x - t_crit * sigma_x / sqrt(n)
            ci_high = mean_x + t_crit * sigma_x / sqrt(n)

            # chi-squared critical values
            df = n - 1
            chi2_low = chi2.ppf(alpha / 2, df)
            chi2_high = chi2.ppf(1 - alpha / 2, df)

            # Confidence interval for σ
            ci_sigma_low = sqrt((df * sigma_x**2) / chi2_high)
            ci_sigma_high = sqrt((df * sigma_x**2) / chi2_low)

            print(f"95% Confidence Interval for μ (x): [{ci_low:.3f}, {ci_high:.3f}]")
            print(f"95% Confidence Interval for σ (x): [{ci_sigma_low:.3f}, {ci_sigma_high:.3f}]")

            pred_low = mean_x - t_crit * sigma_x * sqrt(1 + 1/n)
            pred_high = mean_x + t_crit * sigma_x * sqrt(1 + 1/n)

            print(f"95% Prediction Interval for new x: [{pred_low:.3f}, {pred_high:.3f}]")
            confidence_level = 0.95
            coverage = 0.99  # 99% of the population
            z = norm.ppf((1 + coverage) / 2)  # e.g., z = 2.576 for 99% coverage
            tol_low = mean_x - z * sigma_x
            tol_high = mean_x + z * sigma_x

            print(f"Approximate 99% Tolerance Interval: [{tol_low:.3f}, {tol_high:.3f}]")

def predict_intervals(mu, sigma, n_sample, n_future=1, alpha=0.05, coverage=0.95):
    # Individual prediction interval (standard)
    t_crit_1 = t.ppf(1 - alpha / 2, df=n_sample - 1)
    half_width_1 = t_crit_1 * sigma * sqrt(1 + 1/n_sample)

    # Simultaneous prediction interval (Bonferroni correction)
    adj_alpha = alpha / n_future
    t_crit_bonf = t.ppf(1 - adj_alpha / 2, df=n_sample - 1)
    half_width_bonf = t_crit_bonf * sigma * sqrt(1 + 1/n_sample)

    # Approximate tolerance interval (normal assumption)
    z = norm.ppf((1 + coverage) / 2)
    half_width_tol = z * sigma

    return {
        "Prediction (1 value)": (mu - half_width_1, mu + half_width_1),
        f"Prediction ({n_future} values, Bonferroni)": (mu - half_width_bonf, mu + half_width_bonf),
        f"Tolerance Interval ({int(coverage*100)}%, approx)": (mu - half_width_tol, mu + half_width_tol)
    }
