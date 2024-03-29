import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt


def fill_outliers_nan(array, threshold=3):
    """Fill outliers in 1D array with nan"""
    # Get index of outliers
    idx_outlier = np.where(np.abs(zscore(array, nan_policy='omit')) > threshold)[0]
    # Fill each outlier with mean of closest non outlier
    array[idx_outlier] = np.NAN
    return array


def norm_perc(array):
    """Normalize feature to stimulation block start and return as percentage"""
    mean_start = np.nanmean(array[..., :5], axis=-1)[..., np.newaxis]
    array_norm_perc = ((array - mean_start) / mean_start) * 100
    return array_norm_perc

def norm_all(array):
    """Normalize feature to mean of both conditions (over time)"""
    mean = np.nanmean(array[..., :, :], axis=(-2), keepdims=True)
    array_norm = array - mean
    return array_norm


def despine():
    axes = plt.gca()
    axes.spines[['right', 'top']].set_visible(False)


def smooth_moving_average(array, window_size=5, axis=2):
    """Return the smoothed array where values are averaged in a moving window"""
    box = np.ones(window_size) / window_size
    array_smooth = np.apply_along_axis(lambda m: np.convolve(m, box, mode='same'), axis=axis, arr=array)
    return array_smooth


def plot_conds(array, var=None, color_slow="#00863b", color_fast="#3b0086"):
    """array = (conds x trials)
    Plot data divided into two conditions, if given add the variance as shaded area"""
    # Plot without the first 5 movements
    plt.plot(array[0, :], label="Slow", color=color_slow, linewidth=3)
    plt.plot(array[1, :], label="Fast", color=color_fast, linewidth=3)
    # Add line at 0
    plt.axhline(0, linewidth=2, color="black", linestyle="dashed")
    x = np.arange(array.shape[1])
    # Add variance as shaded area
    if var is not None:
        plt.fill_between(x, array[0, :] - var[0, :], array[0, :] + var[0, :], color=color_slow, alpha=0.2)
        plt.fill_between(x, array[1, :] - var[1, :], array[1, :] + var[1, :], color=color_fast, alpha=0.2)


def plot_bar_points_connect(matrix, colors, labels, alpha_bar=0.5, line_width=0.7):
    """Matrix: Samples x Conditions (2)
    Plot the mean as a bar and add points for each sample connected by a line"""

    plt.bar(1, np.mean(matrix, axis=0)[0], color=colors[0], label=labels[0], width=0.5, alpha=alpha_bar)
    plt.bar(2, np.mean(matrix, axis=0)[1], color=colors[1], label=labels[1], width=0.5, alpha=alpha_bar)

    # Add points and connecting lines
    for dat in matrix:
        plt.plot(1, dat[0], marker='o', markersize=3, color=colors[0])
        plt.plot(2, dat[1], marker='o', markersize=3, color=colors[1])
        # Add line connecting the points
        plt.plot([1, 2], dat, color="black", linewidth=line_width, alpha=0.5)
