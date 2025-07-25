from scipy.stats import skew, kurtosis, norm
import numpy as np

def safe_norm_fit(data, fallback=(0.0, 1.0)):
    data = np.asarray(data)
    if len(data) == 0 or np.all(np.isnan(data)):
        print("⚠️ norm.fit() failed: empty or all NaNs — using fallback.")
        return fallback
    try:
        mu, sigma = norm.fit(data)
        if not np.isfinite(mu) or not np.isfinite(sigma):
            raise ValueError("Non-finite result from norm.fit()s")
        return mu, sigma
    except Exception as e:
        print(f"⚠️ norm.fit() error: {e} — using fallback.")
        return fallback

def analyze_distribution(data, name=""):
    s = skew(data)
    k = kurtosis(data, fisher=True)  # excess kurtosis

    # Interpret skew
    if abs(s) < 0.5:
        skew_desc = "approximately symmetric"
    elif 0.5 <= abs(s) < 1:
        skew_desc = "moderately " + ("right" if s > 0 else "left") + " skewed"
    else:
        skew_desc = "highly " + ("right" if s > 0 else "left") + " skewed"

    # Interpret kurtosis
    if k < -1:
        kurt_desc = "very light-tailed"
    elif -1 <= k < 0.5:
        kurt_desc = "normal-tailed"
    elif 0.5 <= k < 2:
        kurt_desc = "moderately heavy-tailed"
    else:
        kurt_desc = "heavy-tailed"

    # Conclusion
    if abs(s) < 0.5 and -1 <= k < 0.5:
        conclusion = "Gaussian-like. Gaussian fit likely appropriate."
    elif abs(s) >= 1 or k >= 2:
        conclusion = "Not Gaussian. Consider Laplace or t-distribution."
    else:
        conclusion = "Some deviation from Gaussian. Consider testing Laplace or logistic."

    # Print summary
    print(f"--- Analysis of {name or 'distribution'} ---")
    print(f"Skewness: {s:.3f} ({skew_desc})")
    print(f"Kurtosis: {k:.3f} ({kurt_desc})")
    print(f"Conclusion: {conclusion}")
