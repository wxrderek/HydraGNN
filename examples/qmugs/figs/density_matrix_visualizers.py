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
