import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from radial_error.stats import safe_fit



def plt_title_x_y(title=None, xtitle=None, ytitle=None):
    if title is not None:
        plt.title(title)
    if xtitle is not None:
        plt.xlabel(xtitle)
    if ytitle is not None:
        plt.ylabel(ytitle)
    return

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def auto_hist(
    data, bins='auto', density=True, ax=None, cdf=True, rug=True,
    kde=True, kde_bandwidth=None, log_transform=False, log_scale=False,
    **kwargs
):
    """
    Histogram with optional KDE, rug plot, and log transformation/scale.

    Parameters:
        data           : array-like
        bins           : str or int, histogram binning method
        density        : bool, normalize histogram
        ax             : matplotlib axis
        cdf            : bool, show CDF
        rug            : bool, show rug plot
        kde            : bool, overlay KDE line
        kde_bandwidth  : float or str, bandwidth for KDE
        log_transform  : bool, apply log transform to data before plotting
        log_scale      : bool, set x-axis to log scale (visual only)
        kwargs         : additional args to pass to ax.hist

    Returns:
        ax : the axis used
        bin_edges : the bin edges used (in transformed space if log_transform=True)
    """
    data = np.asarray(data)
    if log_transform:
        if np.any(data <= 0):
            raise ValueError("Data must be positive to apply log transform.")
        data = np.log(data)

    bin_edges = np.histogram_bin_edges(data, bins=bins)

    if ax is None:
        fig, ax = plt.subplots()

    # Plot histogram
    ax.hist(data, bins=bin_edges, density=density, alpha=0.6, **kwargs)

    # CDF plot
    if cdf:
        ax2 = ax.twinx()
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(data)+1) / len(data)
        ax2.plot(sorted_data, cdf, color='black', linewidth=1.2, label="Empirical CDF")
        ax2.set_ylabel("Cumulative Probability")

    # Rug plot
    if rug:
        rug_height = 0.05
        ylim = ax.get_ylim()
        rug_y = np.full_like(data, ylim[1] * rug_height)
        ax.vlines(data, 0, rug_y, color='k', alpha=0.3, linewidth=0.5)

    # KDE plot
    if kde:
        kde_bandwidth = 'scott'
        kde_obj = gaussian_kde(data, bw_method=kde_bandwidth)
        x_eval = np.linspace(data.min(), data.max(), 1000)
        y_eval = kde_obj(x_eval)
        # Scale KDE to match histogram counts if density=False
        if not density:
            bin_width = np.diff(bin_edges).mean()
            y_eval *= len(data) * bin_width
        ax.plot(x_eval, y_eval, color='C1',
                linestyle='-', linewidth=2)

    # Apply x-axis log scale (visual only)
    if log_scale:
        ax.set_xscale('log')

    ax.grid(True)

    return ax, bin_edges

def QQplot(x, mu_x, sigma_x, y, mu_y, sigma_y):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    stats.probplot(x, dist="norm", sparams=(mu_x, sigma_x), plot=plt)
    plt.title("QQ Plot for x")
    plt.subplot(1, 2, 2)
    stats.probplot(y, dist="norm", sparams=(mu_y, sigma_y), plot=plt)
    plt.title("QQ Plot for y")
    plt.tight_layout(), plt.show()
    return
'''
def Boxplot(x,y):
    def overlay_stat(data, y_pos,ax):
        # Mean and CI
        mean = np.mean(data)
        sem = stats.sem(data)
        ci_low, ci_high = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
        #print(ci_low)
        #print(ci_high)
        # Diamond for CI
        ax.plot([ci_low, mean, ci_high, mean, ci_low], [y_pos, y_pos+0.1, y_pos, y_pos-0.1, y_pos],
                color='purple', label='Mean 95% CI' if y_pos == 1 else None)

        # Mean point
        #ax.plot(mean, y_pos, 'ro', label='Mean' if y_pos == 1 else None)

        # Shortest half interval
        sorted_data = np.sort(data)
        n = len(sorted_data)
        width = n // 2
        print(width)
        min_range = np.inf
        start = 0
        #print(sorted_data)
        for i in range(n - width):
            r = sorted_data[i + width] - sorted_data[i]
            #print(str(i)+": "+str(sorted_data[i + width])+'-'+str(sorted_data[i])+'='+str(r))
            if r < min_range:
                #print('short found')
                min_range = r
                start = i
        shi_low, shi_high = sorted_data[start], sorted_data[start + width]
        ax.plot([shi_low, shi_high], [y_pos, y_pos], color='orange', linewidth=2, label='Shortest Half' if y_pos == 1 else None)


    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot([x, y], tick_labels=["x", "y"], vert=False, patch_artist=True)
    # Apply to both datasets
    overlay_stat(x, y_pos=1, ax=ax)
    overlay_stat(y, y_pos=2, ax=ax)

    plt_title_x_y("Box Plot of x and y","Value",None)
    plt.grid(True), plt.legend(), plt.show()
    return
'''
def Boxplot(*datasets, labels=None, title="Boxplot", xlabel="Value"):
    """
    Draw horizontal boxplots with overlaid statistics (CI and SHI) for N datasets.
    """
    def overlay_stat(data, y_pos, ax):
        mean = np.mean(data)
        sem = stats.sem(data)
        ci_low, ci_high = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
        ax.plot([ci_low, mean, ci_high, mean, ci_low],
                [y_pos, y_pos+0.1, y_pos, y_pos-0.1, y_pos],
                color='purple', label='Mean 95% CI' if y_pos == 1 else None)

        # Shortest Half Interval (SHI)
        sorted_data = np.sort(data)
        n = len(sorted_data)
        width = n // 2
        min_range = np.inf
        start = 0
        for i in range(n - width):
            r = sorted_data[i + width] - sorted_data[i]
            if r < min_range:
                min_range = r
                start = i
        shi_low, shi_high = sorted_data[start], sorted_data[start + width]
        ax.plot([shi_low, shi_high], [y_pos, y_pos],
                color='orange', linewidth=2, label='Shortest Half' if y_pos == 1 else None)

    n = len(datasets)
    if labels is None:
        labels = [f"x{i+1}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 0.8 * n + 2))  # Dynamically adjust height
    ax.boxplot(datasets, vert=False, patch_artist=True, labels=labels)

    for i, data in enumerate(datasets, start=1):
        overlay_stat(data, y_pos=i, ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def Fitplot(x, y):
    # Fit Gaussian
    mu_x, sigma_x = safe_fit(x, distribution='normal')
    mu_y, sigma_y = safe_fit(y, distribution='normal')
    
    # Plot histogram + fit for x
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    _, bins = auto_hist(x, label="x samples", ax = axs[0])

    x_line = np.linspace(min(x), max(x), len(bins))
    
    axs[0].plot(x_line, stats.norm.pdf(x_line, mu_x, sigma_x), 'r-', label=f'Gaussian fit\nμ={mu_x:.2f}, σ={sigma_x:.2f}')
    axs[0].set_title("Fit to x")
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot histogram + fit for y
    _, bins = auto_hist(y, label="y samples", ax=axs[1])
    y_line = np.linspace(min(y), max(y), len(bins))
    axs[1].plot(y_line, stats.norm.pdf(y_line, mu_y, sigma_y), 'r-', label=f'Gaussian fit\nμ={mu_y:.2f}, σ={sigma_y:.2f}')
    axs[1].set_title("Fit to y")
    axs[1].legend()
    axs[1].grid(True)
    fig.tight_layout()
    plt.show()
    
    return mu_x, sigma_x, mu_y, sigma_y

def plot_spc(data, title="SPC Chart", xlabel="Sample", ylabel="Value", ax=None, highlight=True):
    """
    Plot a basic Statistical Process Control (SPC) chart.

    Parameters:
        data      : array-like sequence of measurements
        title     : plot title
        xlabel    : x-axis label
        ylabel    : y-axis label
        ax        : optional matplotlib axis to plot on
        highlight : if True, mark out-of-control points
    """
    data = np.asarray(data)
    n = len(data)
    x = np.arange(n)

    mean = np.mean(data)
    sigma = np.std(data, ddof=1)
    ucl = mean + 3 * sigma
    lcl = mean - 3 * sigma

    # Create axis if not given
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    # Plot line + points
    ax.plot(x, data, marker='o', linestyle='-', label='Data')
    ax.axhline(mean, color='green', linestyle='--', label='Mean')
    ax.axhline(ucl, color='red', linestyle='--', label='UCL (μ+3σ)')
    ax.axhline(lcl, color='red', linestyle='--', label='LCL (μ−3σ)')

    # Highlight out-of-control points
    if highlight:
        out_of_bounds = (data > ucl) | (data < lcl)
        ax.plot(x[out_of_bounds], data[out_of_bounds], 'ro', label='Out-of-control')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

    return ax

