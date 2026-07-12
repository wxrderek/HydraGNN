import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TITLE_FONTSIZE = 9
AXIS_LABEL_FONTSIZE = 8


def _save_figure(fig, output_path: str, dpi: int = 300):
    '''save a matplotlib figure as both PNG and PDF'''

    # output_path may be passed with or without an extension
    base_path, extension = os.path.splitext(output_path)
    if extension.lower() in [".png", ".pdf"]:
        output_base = base_path
    else:
        output_base = output_path

    png_path = output_base + ".png"
    pdf_path = output_base + ".pdf"

    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return [png_path, pdf_path]


def plot_metric_by_feature(
    metrics_frame,
    metric_name: str,
    feature_name: str,
    output_path: str,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    marker: str = "o",
    figure_width: float = 6.0,
    figure_height: float = 4.0,
    max_feature_value=None,
    feature_bin_size=None,
    dpi: int = 300,
):
    '''plot one scalar metric grouped by one scalar molecule feature'''

    if not isinstance(metrics_frame, pd.DataFrame):
        metrics_frame = pd.DataFrame(metrics_frame)

    if metric_name not in metrics_frame.columns:
        raise KeyError(f"metric_name {metric_name} is not a column")
    if feature_name not in metrics_frame.columns:
        raise KeyError(f"feature_name {feature_name} is not a column")

    # non-scalar values are converted to NaN and removed before grouping
    frame = metrics_frame[[feature_name, metric_name]].copy()
    frame[feature_name] = pd.to_numeric(frame[feature_name], errors="coerce")
    frame[metric_name] = pd.to_numeric(frame[metric_name], errors="coerce")
    frame = frame.dropna()
    if max_feature_value is not None:
        frame = frame[frame[feature_name] <= float(max_feature_value)]
    if len(frame) == 0:
        raise ValueError(f"No finite values found for {metric_name} over {feature_name}")

    if feature_bin_size is not None:
        feature_bin_size = float(feature_bin_size)
        if feature_bin_size <= 0.0:
            raise ValueError("feature_bin_size must be positive")
        # bin by lower edge so grouped means are stable and easy to interpret
        frame[feature_name] = np.floor(frame[feature_name] / feature_bin_size) * feature_bin_size

    grouped = frame.groupby(feature_name)[metric_name].agg(["mean", "std", "count"])
    grouped = grouped.sort_index()
    grouped["std"] = grouped["std"].fillna(0.0)

    # one point is plotted for each exact atom count or exact matrix dimension
    x_values = grouped.index.to_numpy(dtype=np.float64) # (n_groups,)
    y_values = grouped["mean"].to_numpy(dtype=np.float64) # (n_groups,)
    y_errors = grouped["std"].to_numpy(dtype=np.float64) # (n_groups,)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    ax.errorbar(
        x_values,
        y_values,
        yerr=y_errors,
        fmt=marker + "-",
        color="blue",
        ecolor="black",
        capsize=3,
        capthick=0.8,
        elinewidth=0.8,
        linewidth=1.8,
        markersize=7,
    )

    ax.set_xlabel(xlabel if xlabel is not None else feature_name, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel if ylabel is not None else metric_name, fontsize=AXIS_LABEL_FONTSIZE)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    return _save_figure(fig=fig, output_path=output_path, dpi=dpi)


def _flatten_matrix_values(values):
    '''flatten one matrix or a list of matrices into one vector'''

    if isinstance(values, (list, tuple)):
        arrays = [np.asarray(value).reshape(-1) for value in values]
        if len(arrays) == 0:
            raise ValueError("At least one matrix is required")
        return np.concatenate(arrays, axis=0)

    return np.asarray(values).reshape(-1)


def reconstruction_scatter(
    true_matrices,
    predicted_matrices,
    output_path: str,
    title: str = None,
    zscore: bool = True,
    point_alpha: float = 0.12,
    point_size: float = 2.0,
    figure_size: float = 5.5,
    dpi: int = 300,
):
    '''plot predicted versus true density-matrix entries for one or more matrices'''

    true_values = _flatten_matrix_values(true_matrices).astype(np.float64) # (n_entries,)
    predicted_values = _flatten_matrix_values(predicted_matrices).astype(np.float64) # (n_entries,)
    if true_values.shape != predicted_values.shape:
        raise ValueError(
            f"true and predicted entries must have the same shape, "
            f"got {true_values.shape} and {predicted_values.shape}"
        )

    finite_mask = np.isfinite(true_values) & np.isfinite(predicted_values) # (n_entries,)
    true_values = true_values[finite_mask] # (n_finite,)
    predicted_values = predicted_values[finite_mask] # (n_finite,)
    if true_values.size == 0:
        raise ValueError("No finite entries available for reconstruction scatter")

    if zscore:
        # use the true entries to preserve absolute scale and bias errors
        true_mean = float(np.mean(true_values))
        true_std = float(np.std(true_values))
        if true_std <= 0.0:
            true_std = 1.0
        true_values = (true_values - true_mean) / true_std # (n_finite,)
        predicted_values = (predicted_values - true_mean) / true_std # (n_finite,)
        axis_label = "z-score normalized entry"
    else:
        axis_label = "density matrix entry"

    min_value = float(min(np.min(true_values), np.min(predicted_values)))
    max_value = float(max(np.max(true_values), np.max(predicted_values)))

    # rasterized points keep large matrix-entry scatter PDFs reasonably small
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    ax.scatter(
        true_values,
        predicted_values,
        s=point_size,
        alpha=point_alpha,
        rasterized=True,
        linewidths=0,
    )
    ax.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1.0)
    ax.set_xlabel("True " + axis_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Predicted " + axis_label, fontsize=AXIS_LABEL_FONTSIZE)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    return _save_figure(fig=fig, output_path=output_path, dpi=dpi)
