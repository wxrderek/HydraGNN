import os

import numpy as np

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


def _validate_square_matrix(matrix: np.ndarray, name: str):
    '''validate that a plotting input is a square matrix'''

    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {matrix.shape}")

    return matrix


def _draw_density_matrix_heatmap(
    matrix: np.ndarray,
    title: str = None,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
    colorbar_label: str = "Density matrix entry",
    figure_size: float = 8.0,
    origin: str = "upper",
):
    '''draw a density matrix heatmap and return the figure and axes'''

    matrix = _validate_square_matrix(matrix, "matrix") # (n_basis, n_basis)

    # interpolation is disabled so individual AO-index entries remain visible
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    image = ax.imshow(
        matrix,
        cmap=cmap,
        origin=origin,
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label, fontsize=AXIS_LABEL_FONTSIZE)

    ax.set_xlabel("AO index", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("AO index", fontsize=AXIS_LABEL_FONTSIZE)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)

    fig.tight_layout()
    return fig, ax


def density_matrix_heatmap(
    matrix: np.ndarray,
    output_path: str,
    title: str = None,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
    colorbar_label: str = "Density matrix entry",
    figure_size: float = 8.0,
    origin: str = "upper",
    dpi: int = 300,
):
    '''plot and save a heatmap of one preprocessed density matrix'''

    fig, _ = _draw_density_matrix_heatmap(
        matrix=matrix,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar_label=colorbar_label,
        figure_size=figure_size,
        origin=origin,
    )

    return _save_figure(fig=fig, output_path=output_path, dpi=dpi)


def density_matrix_heatmap_with_mask(
    matrix: np.ndarray,
    mask: np.ndarray,
    output_path: str,
    title: str = None,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
    colorbar_label: str = "Density matrix entry",
    mask_alpha: float = 0.25,
    mask_cmap: str = "gray_r",
    figure_size: float = 8.0,
    origin: str = "upper",
    dpi: int = 300,
):
    '''plot and save a density matrix heatmap with withheld entries overlaid'''

    matrix = _validate_square_matrix(matrix, "matrix") # (n_basis, n_basis)
    mask = _validate_square_matrix(mask, "mask").astype(bool) # (n_basis, n_basis)
    if mask.shape != matrix.shape:
        raise ValueError(f"mask shape {mask.shape} does not match matrix shape {matrix.shape}")

    # draw the matrix first, then overlay withheld mask entries
    fig, ax = _draw_density_matrix_heatmap(
        matrix=matrix,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar_label=colorbar_label,
        figure_size=figure_size,
        origin=origin,
    )

    # mask == False indicates density matrix entries not seen by the loss
    overlay = np.where(mask, np.nan, 1.0) # (n_basis, n_basis)
    ax.imshow(
        overlay,
        cmap=mask_cmap,
        origin=origin,
        interpolation="none",
        alpha=mask_alpha,
    )

    fig.tight_layout()
    return _save_figure(fig=fig, output_path=output_path, dpi=dpi)


def eigenvalue_spectrum_plot(
    true_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    output_path: str,
    title: str = None,
    negative_threshold: float = -1.0e-6,
    figure_size: float = 6.0,
    dpi: int = 300,
):
    '''plot and save the sorted eigenvalues of a true and a predicted density matrix'''

    true_matrix = _validate_square_matrix(true_matrix, "true_matrix") # (n_basis, n_basis)
    predicted_matrix = _validate_square_matrix(predicted_matrix, "predicted_matrix") # (n_basis, n_basis)
    if true_matrix.shape != predicted_matrix.shape:
        raise ValueError(f"shapes differ: {true_matrix.shape} and {predicted_matrix.shape}")

    # eigvalsh reads the lower triangle only, so the matrices are symmetrized first;
    # descending order puts the occupied part on the left
    true_eigenvalues = np.linalg.eigvalsh(0.5 * (true_matrix + true_matrix.T).astype(np.float64))[::-1] # (n_basis,)
    predicted_eigenvalues = np.linalg.eigvalsh(0.5 * (predicted_matrix + predicted_matrix.T).astype(np.float64))[::-1] # (n_basis,)

    # a stored float32 D has eigenvalues down to about -1e-7 from round-off alone, so a count
    # with threshold 0 would call it non-PSD; the threshold is stated in the legend instead
    legend = []
    for name, eigenvalues in (("true", true_eigenvalues), ("predicted", predicted_eigenvalues)):
        n_negative = int(np.sum(eigenvalues < negative_threshold))
        legend.append(f"{name}: min {eigenvalues.min():.2e}, {n_negative} below {negative_threshold:.0e}")

    fig, ax = plt.subplots(figsize=(figure_size, 0.7 * figure_size))
    index = np.arange(1, true_eigenvalues.size + 1)
    ax.plot(index, true_eigenvalues, color="#1f77b4", lw=1.5, label=legend[0])
    ax.plot(index, predicted_eigenvalues, color="#d62728", lw=1.2, ls="--", label=legend[1])
    ax.axhline(0.0, color="#999999", lw=0.8)
    # linear near zero so negative eigenvalues stay visible next to the O(1) occupied ones
    ax.set_yscale("symlog", linthresh=1.0e-4)
    ax.set_xlabel("eigenvalue index (descending)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("eigenvalue of D (AO basis)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.legend(fontsize=AXIS_LABEL_FONTSIZE, frameon=False)
    ax.tick_params(labelsize=AXIS_LABEL_FONTSIZE)
    ax.grid(alpha=0.3, lw=0.5)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)

    fig.tight_layout()
    return _save_figure(fig=fig, output_path=output_path, dpi=dpi)
