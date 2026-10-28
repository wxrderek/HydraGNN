import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figs.density_matrix_visualizers import _save_figure, _validate_square_matrix


TITLE_FONTSIZE = 9
AXIS_LABEL_FONTSIZE = 8


def leading_block_relative_errors(predicted_matrix: np.ndarray, true_matrix: np.ndarray, eps: float = 1e-12):
    '''relative error ||D_hat[:k,:k] - D[:k,:k]||_F / ||D[:k,:k]||_F for every k = 1..n'''

    predicted_matrix = _validate_square_matrix(predicted_matrix, "predicted_matrix").astype(np.float64) # (n_basis, n_basis)
    true_matrix = _validate_square_matrix(true_matrix, "true_matrix").astype(np.float64) # (n_basis, n_basis)

    # the squared Frobenius norm of the leading k x k block is the (k-1, k-1) entry of the
    # 2-D cumulative sum of the squared entries, so all n blocks cost O(n^2)
    error_sums = np.cumsum(np.cumsum((predicted_matrix - true_matrix) ** 2, axis=0), axis=1) # (n_basis, n_basis)
    true_sums = np.cumsum(np.cumsum(true_matrix ** 2, axis=0), axis=1) # (n_basis, n_basis)
    return np.sqrt(np.diag(error_sums)) / np.maximum(np.sqrt(np.diag(true_sums)), eps) # (n_basis,)


def block_error_curve_plot(
    curves,
    output_path: str,
    title: str = None,
    figure_size: float = 6.0,
    dpi: int = 300,
):
    '''plot leading-block relative errors against k/n for one or more conformers

    curves is a list of (label, free_running_errors, teacher_forced_errors), each array (n_basis,)
    '''

    fig, ax = plt.subplots(figsize=(figure_size, 0.7 * figure_size))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i_curve, (label, free_running, teacher_forced) in enumerate(curves):
        color = colors[i_curve % len(colors)]
        fraction = np.arange(1, free_running.size + 1) / free_running.size # (n_basis,), k/n
        ax.plot(fraction, free_running, color=color, lw=1.4, label=f"{label} free running")
        if teacher_forced is not None:
            ax.plot(fraction, teacher_forced, color=color, lw=1.0, ls="--", label=f"{label} teacher forced")

    ax.set_xlabel("k / n  (leading block of the recursion)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\|\hat D_k - D_k\|_F \,/\, \|D_k\|_F$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.tick_params(labelsize=AXIS_LABEL_FONTSIZE)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=AXIS_LABEL_FONTSIZE - 1, frameon=False)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)

    fig.tight_layout()
    return _save_figure(fig=fig, output_path=output_path, dpi=dpi)


def pivot_latent_plot(
    delta: np.ndarray,
    z_norms: np.ndarray,
    output_path: str,
    title: str = None,
    figure_size: float = 6.0,
    dpi: int = 300,
):
    '''plot the predicted pivots delta_k and latent norms ||z_k|| against k/n for one conformer'''

    delta = np.asarray(delta, dtype=np.float64).reshape(-1) # (n_basis,)
    z_norms = np.asarray(z_norms, dtype=np.float64).reshape(-1) # (n_basis,)
    if delta.shape != z_norms.shape:
        raise ValueError(f"delta has shape {delta.shape} but z_norms has shape {z_norms.shape}")
    fraction = np.arange(1, delta.size + 1) / delta.size # (n_basis,), k/n

    fig, (ax_delta, ax_z) = plt.subplots(2, 1, sharex=True, figsize=(figure_size, 0.9 * figure_size))
    ax_delta.plot(fraction, delta, color="#1f77b4", lw=0.8, marker=".", ms=2)
    ax_delta.set_yscale("log")
    ax_delta.set_ylabel(r"pivot $\delta_k = f(\alpha_k)$", fontsize=AXIS_LABEL_FONTSIZE)
    ax_delta.grid(alpha=0.3, lw=0.5)

    # z_1 is empty, so its norm is 0 and is left off the log axis
    ax_z.plot(fraction[1:], z_norms[1:], color="#d62728", lw=0.8, marker=".", ms=2)
    ax_z.set_yscale("log")
    ax_z.set_ylabel(r"$\|z_k\|_2$", fontsize=AXIS_LABEL_FONTSIZE)
    ax_z.set_xlabel("k / n", fontsize=AXIS_LABEL_FONTSIZE)
    ax_z.set_xlim(0.0, 1.0)
    ax_z.grid(alpha=0.3, lw=0.5)
    for axis in (ax_delta, ax_z):
        axis.tick_params(labelsize=AXIS_LABEL_FONTSIZE)
    if title is not None:
        ax_delta.set_title(title, fontsize=TITLE_FONTSIZE)

    fig.tight_layout()
    return _save_figure(fig=fig, output_path=output_path, dpi=dpi)
