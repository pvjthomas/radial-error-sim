from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde

from radial_error.stats import safe_fit


def plt_title_x_y(title=None, xtitle=None, ytitle=None, ax=None):
    """
    Convenience function to set plot title and axis labels.

    Helper function for setting matplotlib plot titles and labels. If no axes
    is provided, uses the current axes (plt.gca()).

    Parameters
    ----------
    title : str or None, optional
        Plot title. If None, title is not changed. Default is None.
    xtitle : str or None, optional
        X-axis label. If None, x-axis label is not changed. Default is None.
    ytitle : str or None, optional
        Y-axis label. If None, y-axis label is not changed. Default is None.
    ax : matplotlib.axes.Axes or None, optional
        Axes to modify. If None, uses current axes (plt.gca()). Default is None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object (either provided or current).

    Examples
    --------
    >>> ax = plt.subplots()[1]
    >>> plt_title_x_y(title="My Plot", xtitle="X", ytitle="Y", ax=ax)
    """
    if ax is None:
        ax = plt.gca()
    if title is not None:
        ax.set_title(title)
    if xtitle is not None:
        ax.set_xlabel(xtitle)
    if ytitle is not None:
        ax.set_ylabel(ytitle)
    return ax


def auto_hist(
    data,
    bins="auto",
    density=True,
    ax=None,
    cdf=True,
    rug=True,
    kde=True,
    kde_bandwidth=None,
    log_transform=False,
    log_scale=False,
    return_cdf_ax=False,
    **kwargs,
):
    """
    Create an enhanced histogram with optional KDE, CDF, and rug plot overlays.

    Generates a histogram with multiple optional enhancements:
    - Kernel Density Estimation (KDE) overlay
    - Empirical Cumulative Distribution Function (CDF) on twin y-axis
    - Rug plot showing individual data points
    - Log transformation or log scale support

    Parameters
    ----------
    data : array-like
        One-dimensional array of data to plot. Will be converted to numpy array.
    bins : int, str, or array-like, optional
        Histogram binning method. Can be an integer (number of bins), a string
        like 'auto', 'fd', 'scott', etc., or an array of bin edges.
        Default is "auto".
    density : bool, optional
        If True, normalize histogram to form a probability density. If False,
        shows counts. Default is True.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, creates a new figure. Default is None.
    cdf : bool, optional
        If True, overlay empirical CDF on a twin y-axis. Default is True.
    rug : bool, optional
        If True, show rug plot (vertical lines at each data point). Default is True.
    kde : bool, optional
        If True, overlay Kernel Density Estimation line. Default is True.
    kde_bandwidth : float, str, or None, optional
        Bandwidth method for KDE. Can be a float, or string like 'scott', 'silverman'.
        If None, uses 'scott'. Default is None.
    log_transform : bool, optional
        If True, apply log transformation to data before plotting. Data must be
        positive. If True, log_scale is ignored. Default is False.
    log_scale : bool, optional
        If True, set x-axis to logarithmic scale (visual only, no data transformation).
        Ignored if log_transform=True. Default is False.
    return_cdf_ax : bool, optional
        If True and cdf=True, return tuple (ax, bin_edges, cdf_ax). Otherwise
        return (ax, bin_edges). Default is False.
    **kwargs
        Additional keyword arguments passed to ax.hist().

    Returns
    -------
    tuple
        By default: (ax, bin_edges)
        If return_cdf_ax=True and cdf=True: (ax, bin_edges, cdf_ax)
        - ax: matplotlib.axes.Axes object
        - bin_edges: numpy.ndarray of bin edges
        - cdf_ax: matplotlib.axes.Axes object for CDF (if returned)

    Raises
    ------
    ValueError
        If log_transform=True and data contains non-positive values.

    Examples
    --------
    >>> data = np.random.normal(5, 2, size=1000)
    >>> ax, bins = auto_hist(data, kde=True, cdf=True, rug=True)
    >>> plt.show()

    >>> # Log-transformed data
    >>> positive_data = np.random.exponential(2, size=1000)
    >>> ax, bins = auto_hist(positive_data, log_transform=True)
    """
    data = np.asarray(data)

    if log_transform:
        if np.any(data <= 0):
            raise ValueError("Data must be positive to apply log transform.")
        data = np.log(data)
        # log_scale would be misleading after transforming
        log_scale = False

    bin_edges = np.histogram_bin_edges(data, bins=bins)

    if ax is None:
        _, ax = plt.subplots()

    # Histogram
    ax.hist(data, bins=bin_edges, density=density, alpha=0.6, **kwargs)

    cdf_ax = None
    if cdf:
        cdf_ax = ax.twinx()
        sorted_data = np.sort(data)
        cdf_vals = np.arange(1, len(data) + 1) / len(data)
        cdf_ax.plot(sorted_data, cdf_vals, color="black", linewidth=1.2, label="Empirical CDF")
        cdf_ax.set_ylabel("Cumulative Probability")

    if rug:
        # Rug lines from y=0 to a small fraction of current ylim
        ymax = ax.get_ylim()[1]
        rug_y = ymax * 0.05
        ax.vlines(data, 0, rug_y, color="k", alpha=0.3, linewidth=0.5)

    if kde:
        bw = "scott" if kde_bandwidth is None else kde_bandwidth
        kde_obj = gaussian_kde(data, bw_method=bw)

        x_eval = np.linspace(data.min(), data.max(), 1000)
        y_eval = kde_obj(x_eval)

        # If histogram is counts (density=False), scale KDE to counts
        if not density:
            bin_width = float(np.diff(bin_edges).mean())
            y_eval *= len(data) * bin_width

        ax.plot(x_eval, y_eval, linestyle="-", linewidth=2)

    if log_scale:
        ax.set_xscale("log")

    ax.grid(True)

    if return_cdf_ax and cdf:
        return ax, bin_edges, cdf_ax
    return ax, bin_edges


def QQplot(x, mu_x, sigma_x, y, mu_y, sigma_y, show=True):
    """
    Create side-by-side Q-Q plots for x and y against specified normal distributions.

    Generates two Q-Q (quantile-quantile) plots to assess whether x and y
    follow their specified normal distributions. Useful for checking
    normality assumptions in 2D error analysis.

    Parameters
    ----------
    x : array-like
        First dataset to plot. Will be converted to numpy array.
    mu_x : float
        Mean of the normal distribution to compare x against.
    sigma_x : float
        Standard deviation of the normal distribution to compare x against.
    y : array-like
        Second dataset to plot. Will be converted to numpy array.
    mu_y : float
        Mean of the normal distribution to compare y against.
    sigma_y : float
        Standard deviation of the normal distribution to compare y against.
    show : bool, optional
        If True, display the plot immediately. Default is True.

    Returns
    -------
    tuple
        (fig, (ax1, ax2)) where:
        - fig: matplotlib.figure.Figure object
        - ax1: matplotlib.axes.Axes for x Q-Q plot
        - ax2: matplotlib.axes.Axes for y Q-Q plot

    Examples
    --------
    >>> x, y = sample_xy(mu_x=0, sigma_x=1, mu_y=0, sigma_y=1, n_samples=100)
    >>> fig, (ax1, ax2) = QQplot(x, 0, 1, y, 0, 1)
    >>> plt.show()

    Notes
    -----
    Points falling close to the diagonal line indicate good agreement with
    the normal distribution. Systematic deviations suggest non-normality.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    stats.probplot(x, dist="norm", sparams=(mu_x, sigma_x), plot=ax1)
    ax1.set_title("QQ Plot for x")
    ax1.grid(True)

    stats.probplot(y, dist="norm", sparams=(mu_y, sigma_y), plot=ax2)
    ax2.set_title("QQ Plot for y")
    ax2.grid(True)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, (ax1, ax2)


def Boxplot(*datasets, labels=None, title="Boxplot", xlabel="Value", show=True):
    """
    Create horizontal boxplots with enhanced statistical overlays.

    Generates horizontal boxplots for one or more datasets with additional
    statistical information overlaid:
    - Mean 95% confidence interval (diamond shape)
    - Shortest Half Interval (SHI) - the narrowest interval containing half the data

    Parameters
    ----------
    *datasets : array-like
        One or more datasets to plot. Each dataset will be converted to numpy array.
    labels : list of str or None, optional
        Labels for each dataset. If None, generates default labels (x1, x2, ...).
        Length must match number of datasets. Default is None.
    title : str, optional
        Plot title. Default is "Boxplot".
    xlabel : str, optional
        X-axis label. Default is "Value".
    show : bool, optional
        If True, display the plot immediately. Default is True.

    Returns
    -------
    tuple
        (fig, ax) where:
        - fig: matplotlib.figure.Figure object
        - ax: matplotlib.axes.Axes object

    Raises
    ------
    ValueError
        If no datasets provided or if labels length doesn't match number of datasets.

    Examples
    --------
    >>> x = np.random.normal(5, 2, size=100)
    >>> y = np.random.normal(7, 3, size=100)
    >>> fig, ax = Boxplot(x, y, labels=['X', 'Y'], title="Comparison")
    >>> plt.show()

    Notes
    -----
    - Boxplots show quartiles, median, and outliers
    - Purple diamond shows mean ± 95% CI
    - Orange line shows shortest half interval (robust measure of spread)
    """
    def overlay_stat(data, y_pos, ax):
        data = np.asarray(data)
        mean = float(np.mean(data))
        sem = stats.sem(data)
        ci_low, ci_high = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=sem)

        # Diamond-ish CI marker
        ax.plot(
            [ci_low, mean, ci_high, mean, ci_low],
            [y_pos, y_pos + 0.1, y_pos, y_pos - 0.1, y_pos],
            color="purple",
            label="Mean 95% CI" if y_pos == 1 else None,
        )

        # Shortest Half Interval (SHI)
        sorted_data = np.sort(data)
        n = len(sorted_data)
        width = n // 2
        if width <= 0:
            return

        min_range = np.inf
        start = 0
        for i in range(n - width):
            r = sorted_data[i + width] - sorted_data[i]
            if r < min_range:
                min_range = r
                start = i

        shi_low, shi_high = sorted_data[start], sorted_data[start + width]
        ax.plot(
            [shi_low, shi_high],
            [y_pos, y_pos],
            color="orange",
            linewidth=2,
            label="Shortest Half" if y_pos == 1 else None,
        )

    n = len(datasets)
    if n == 0:
        raise ValueError("Boxplot() requires at least one dataset.")

    if labels is None:
        labels = [f"x{i+1}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels length must match number of datasets.")

    fig, ax = plt.subplots(figsize=(10, 0.8 * n + 2))
    ax.boxplot(datasets, vert=False, patch_artist=True, labels=labels)

    for i, data in enumerate(datasets, start=1):
        overlay_stat(data, y_pos=i, ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


def Fitplot(x, distribution="normal", title=None, ax=None, show=True, bins="auto", **hist_kwargs):
    """
    Plot histogram with fitted probability density function overlay.

    Creates a histogram of the data and overlays the fitted PDF of the specified
    distribution. Useful for visually assessing how well a distribution fits the data.

    Parameters
    ----------
    x : array-like
        One-dimensional array of data to plot. Will be converted to numpy array.
    distribution : {'normal', 'exponential'}, optional
        Distribution type to fit and overlay. Default is 'normal'.
    title : str or None, optional
        Plot title. If None, no title is set. Default is None.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, creates a new figure. Default is None.
    show : bool, optional
        If True, display the plot immediately. Default is True.
    bins : int, str, or array-like, optional
        Histogram binning method. Passed to auto_hist(). Default is "auto".
    **hist_kwargs
        Additional keyword arguments passed to auto_hist().

    Returns
    -------
    tuple
        For 'normal': (mu, sigma, fig, ax)
        For 'exponential': (loc, scale, fig, ax)
        - mu/loc: Fitted distribution parameter (mean for normal, location for exponential)
        - sigma/scale: Fitted distribution parameter (std for normal, scale for exponential)
        - fig: matplotlib.figure.Figure object
        - ax: matplotlib.axes.Axes object

    Raises
    ------
    ValueError
        If distribution is not 'normal' or 'exponential'.

    Examples
    --------
    >>> data = np.random.normal(5, 2, size=1000)
    >>> mu, sigma, fig, ax = Fitplot(data, distribution="normal", title="Normal Fit")
    >>> print(f"Fitted: μ={mu:.2f}, σ={sigma:.2f}")
    Fitted: μ=5.01, σ=2.03
    """
    x = np.asarray(x)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    # Use density=True for PDF overlay to make sense
    ax_hist, bin_edges = auto_hist(
        x,
        bins=bins,
        density=True,
        ax=ax,
        cdf=False,
        rug=False,
        kde=False,
        **hist_kwargs,
    )

    x_line = np.linspace(x.min(), x.max(), 500)

    if distribution == "normal":
        mu, sigma = safe_fit(x, distribution="normal")
        ax_hist.plot(
            x_line,
            stats.norm.pdf(x_line, mu, sigma),
            label=f"Gaussian fit\nμ={mu:.2f}, σ={sigma:.2f}",
        )
        params = (mu, sigma)
    elif distribution == "exponential":
        loc, scale = safe_fit(x, distribution="exponential")
        ax_hist.plot(
            x_line,
            stats.expon.pdf(x_line, loc, scale),
            label=f"Exponential fit\nloc={loc:.2f}, μ={scale:.2f}",
        )
        params = (loc, scale)
    else:
        raise ValueError("distribution must be 'normal' or 'exponential'.")

    if title is not None:
        ax_hist.set_title(title)

    ax_hist.grid(True)
    ax_hist.legend()
    fig.tight_layout()

    if show:
        plt.show()

    return (*params, fig, ax_hist)


def Fitplot_r(x, y, show=True, bins="auto", **hist_kwargs):
    """
    Create side-by-side histograms with normal distribution fits for x and y.

    Convenience function for visualizing and fitting both x and y datasets
    simultaneously. Useful for comparing the distributions of 2D error components.

    Parameters
    ----------
    x : array-like
        First dataset to plot. Will be converted to numpy array.
    y : array-like
        Second dataset to plot. Will be converted to numpy array.
    show : bool, optional
        If True, display the plot immediately. Default is True.
    bins : int, str, or array-like, optional
        Histogram binning method. Passed to auto_hist(). Default is "auto".
    **hist_kwargs
        Additional keyword arguments passed to auto_hist().

    Returns
    -------
    tuple
        (mu_x, sigma_x, mu_y, sigma_y, fig, (ax1, ax2)) where:
        - mu_x: Fitted mean for x
        - sigma_x: Fitted standard deviation for x
        - mu_y: Fitted mean for y
        - sigma_y: Fitted standard deviation for y
        - fig: matplotlib.figure.Figure object
        - ax1: matplotlib.axes.Axes for x plot
        - ax2: matplotlib.axes.Axes for y plot

    Examples
    --------
    >>> x, y = sample_xy(mu_x=0, sigma_x=1, mu_y=0, sigma_y=2, n_samples=1000)
    >>> mu_x, sigma_x, mu_y, sigma_y, fig, (ax1, ax2) = Fitplot_r(x, y)
    >>> print(f"x: μ={mu_x:.2f}, σ={sigma_x:.2f}")
    >>> print(f"y: μ={mu_y:.2f}, σ={sigma_y:.2f}")
    """
    x = np.asarray(x)
    y = np.asarray(y)

    mu_x, sigma_x = safe_fit(x, distribution="normal")
    mu_y, sigma_y = safe_fit(y, distribution="normal")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    auto_hist(x, bins=bins, density=True, ax=ax1, cdf=False, rug=False, kde=False, **hist_kwargs)
    x_line = np.linspace(x.min(), x.max(), 500)
    ax1.plot(x_line, stats.norm.pdf(x_line, mu_x, sigma_x), label=f"Gaussian fit\nμ={mu_x:.2f}, σ={sigma_x:.2f}")
    ax1.set_title("Fit to x")
    ax1.grid(True)
    ax1.legend()

    auto_hist(y, bins=bins, density=True, ax=ax2, cdf=False, rug=False, kde=False, **hist_kwargs)
    y_line = np.linspace(y.min(), y.max(), 500)
    ax2.plot(y_line, stats.norm.pdf(y_line, mu_y, sigma_y), label=f"Gaussian fit\nμ={mu_y:.2f}, σ={sigma_y:.2f}")
    ax2.set_title("Fit to y")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    if show:
        plt.show()

    return mu_x, sigma_x, mu_y, sigma_y, fig, (ax1, ax2)


def plot_spc(
    data,
    title="SPC Chart with Moving Range",
    xlabel="Sample",
    ylabel="Value",
    highlight=True,
    show=True,
):
    """
    Create a Statistical Process Control (SPC) chart with Moving Range subplot.

    Generates a two-panel SPC chart:
    - Top panel: Individual values with control limits (mean ± 3σ)
    - Bottom panel: Moving Range chart for detecting process variability changes

    Parameters
    ----------
    data : array-like
        One-dimensional time series data to plot. Will be converted to numpy array.
    title : str, optional
        Overall plot title. Default is "SPC Chart with Moving Range".
    xlabel : str, optional
        X-axis label for both subplots. Default is "Sample".
    ylabel : str, optional
        Y-axis label for the top (individuals) chart. Default is "Value".
    highlight : bool, optional
        If True, highlight out-of-control points (beyond ±3σ limits). Default is True.
    show : bool, optional
        If True, display the plot immediately. Default is True.

    Returns
    -------
    tuple
        (fig, (ax1, ax2)) where:
        - fig: matplotlib.figure.Figure object
        - ax1: matplotlib.axes.Axes for individuals chart (top)
        - ax2: matplotlib.axes.Axes for moving range chart (bottom)

    Examples
    --------
    >>> data = np.random.normal(5, 1, size=50)
    >>> fig, (ax1, ax2) = plot_spc(data, title="Process Control Chart")
    >>> plt.show()

    Notes
    -----
    - Control limits use ±3σ (standard 3-sigma limits)
    - Moving Range uses constant 3.267 for n=2 (standard MR chart constant)
    - Out-of-control points are highlighted in red if highlight=True
    - Useful for monitoring process stability over time
    """
    data = np.asarray(data)
    n = len(data)
    x = np.arange(n)

    # Individuals-style limits (simple approach using sample stddev)
    mean = float(np.mean(data))
    sigma = float(np.std(data, ddof=1)) if n > 1 else 0.0
    ucl = mean + 3 * sigma
    lcl = mean - 3 * sigma

    # Moving range
    mr = np.abs(np.diff(data))
    mr_x = np.arange(1, n)
    mr_mean = float(np.mean(mr)) if len(mr) else 0.0
    mr_ucl = mr_mean * 3.267  # MR chart constant for n=2

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, height_ratios=[2, 1]
    )
    fig.suptitle(title, fontsize=14)

    # Top chart
    ax1.plot(x, data, marker="o", linestyle="-", label="Data")
    ax1.axhline(mean, linestyle="--", label="Mean")
    ax1.axhline(ucl, linestyle="--", label="UCL (μ+3σ)")
    ax1.axhline(lcl, linestyle="--", label="LCL (μ−3σ)")

    if highlight and n > 0:
        out_of_bounds = (data > ucl) | (data < lcl)
        ax1.plot(x[out_of_bounds], data[out_of_bounds], "o", label="Out-of-control")

    ax1.set_ylabel(ylabel)
    ax1.grid(True)
    ax1.legend()

    # MR chart
    ax2.plot(mr_x, mr, marker="o", linestyle="-", label="Moving Range")
    ax2.axhline(mr_mean, linestyle="--", label="MR Mean")
    ax2.axhline(mr_ucl, linestyle="--", label="MR UCL (3.267·MR̄)")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Moving Range")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()

    return fig, (ax1, ax2)
