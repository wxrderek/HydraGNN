import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rdkit import Chem


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


def _validate_molecule_arrays(positions_angstrom: np.ndarray, atomic_numbers: np.ndarray):
    '''validate molecule arrays used by the 3D plotting helpers'''

    positions_angstrom = np.asarray(positions_angstrom, dtype=np.float64) # (n_atoms, 3)
    atomic_numbers = np.asarray(atomic_numbers).reshape(-1).astype(int) # (n_atoms,)

    if positions_angstrom.ndim != 2 or positions_angstrom.shape[1] != 3:
        raise ValueError(f"positions_angstrom must have shape (n_atoms, 3), got {positions_angstrom.shape}")
    if atomic_numbers.shape[0] != positions_angstrom.shape[0]:
        raise ValueError(
            f"atomic_numbers has length {atomic_numbers.shape[0]}, "
            f"but positions has {positions_angstrom.shape[0]} atoms"
        )

    return positions_angstrom, atomic_numbers


def _atom_color(atomic_number: int):
    '''return a conventional plotting color for one element'''

    colors = {
        1: "#f2f2f2",
        6: "#4a4a4a",
        7: "#3050f8",
        8: "#ff0d0d",
        9: "#90e050",
        15: "#ff8000",
        16: "#ffff30",
        17: "#1ff01f",
        35: "#a62929",
        53: "#940094",
    }

    return colors.get(int(atomic_number), "#b0b0b0")


def _atom_size(atomic_number: int, n_atoms: int):
    '''return a marker size for one atomic number'''

    if int(n_atoms) < 20:
        size_scale = 3.5
    elif int(n_atoms) < 30:
        size_scale = 2.5
    else:
        size_scale = 1.0
    if int(atomic_number) == 1:
        return 35.0 * size_scale
    return 80.0 * size_scale


def _draw_molecule(ax, positions_angstrom: np.ndarray, atomic_numbers: np.ndarray, bonds=None):
    '''draw atoms and optional bonds on an existing 3D axis'''

    positions_angstrom, atomic_numbers = _validate_molecule_arrays(
        positions_angstrom=positions_angstrom,
        atomic_numbers=atomic_numbers,
    )
    n_atoms = int(atomic_numbers.shape[0])

    # RDKit SDF bonds are optional because positions and species are sufficient
    if bonds is not None:
        for bond in bonds:
            i_atom, j_atom = int(bond[0]), int(bond[1])
            if i_atom < 0 or j_atom < 0:
                continue
            if i_atom >= positions_angstrom.shape[0] or j_atom >= positions_angstrom.shape[0]:
                continue
            bond_positions = positions_angstrom[[i_atom, j_atom], :] # (2, 3)
            ax.plot(
                bond_positions[:, 0],
                bond_positions[:, 1],
                bond_positions[:, 2],
                color="#8a8a8a",
                linewidth=1.4,
                alpha=0.8,
            )

    for atomic_number in sorted(set(atomic_numbers.tolist())):
        atom_mask = atomic_numbers == atomic_number # (n_atoms,)
        atom_positions = positions_angstrom[atom_mask, :] # (n_element_atoms, 3)
        ax.scatter(
            atom_positions[:, 0],
            atom_positions[:, 1],
            atom_positions[:, 2],
            s=_atom_size(atomic_number, n_atoms=n_atoms),
            color=_atom_color(atomic_number),
            edgecolors="black",
            linewidths=0.35,
            depthshade=True,
            label=Chem.GetPeriodicTable().GetElementSymbol(int(atomic_number)),
        )


def _set_axes_equal(ax, points_angstrom: np.ndarray):
    '''set equal 3D axis scaling around a collection of points'''

    points_angstrom = np.asarray(points_angstrom, dtype=np.float64) # (n_points, 3)
    min_xyz = np.min(points_angstrom, axis=0) # (3,)
    max_xyz = np.max(points_angstrom, axis=0) # (3,)
    center = 0.5 * (min_xyz + max_xyz) # (3,)
    radius = 0.5 * float(np.max(max_xyz - min_xyz))
    if radius <= 0.0:
        radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _scaled_dipole_vectors(
    dipole_vectors_au: np.ndarray,
    positions_angstrom: np.ndarray,
    normalize_vectors: bool = False,
    arrow_fraction: float = 0.45,
):
    '''scale dipole vectors into molecular plotting coordinates'''

    dipole_vectors_au = np.asarray(dipole_vectors_au, dtype=np.float64) # (n_vectors, 3)
    if dipole_vectors_au.ndim == 1:
        dipole_vectors_au = dipole_vectors_au.reshape(1, 3)
    if dipole_vectors_au.ndim != 2 or dipole_vectors_au.shape[1] != 3:
        raise ValueError(f"dipole_vectors_au must have shape (n_vectors, 3), got {dipole_vectors_au.shape}")

    positions_angstrom = np.asarray(positions_angstrom, dtype=np.float64) # (n_atoms, 3)
    molecule_span = float(np.max(np.ptp(positions_angstrom, axis=0)))
    if molecule_span <= 0.0:
        molecule_span = 1.0

    norms = np.linalg.norm(dipole_vectors_au, axis=1) # (n_vectors,)
    safe_norms = np.where(norms > 0.0, norms, 1.0) # (n_vectors,)
    unit_vectors = dipole_vectors_au / safe_norms[:, None] # (n_vectors, 3)

    # vectors remain in AU; only the displayed arrow lengths are rescaled
    if normalize_vectors:
        scaled_vectors = unit_vectors * (0.70 * molecule_span) # (n_vectors, 3)
    else:
        max_norm = float(np.max(safe_norms))
        scaled_vectors = unit_vectors * (norms[:, None] / max_norm) * (arrow_fraction * molecule_span) # (n_vectors, 3)

    return scaled_vectors


def _perpendicular_basis(unit_vector: np.ndarray):
    '''return two unit vectors perpendicular to one direction'''

    unit_vector = np.asarray(unit_vector, dtype=np.float64) # (3,)
    if abs(unit_vector[0]) < 0.9:
        trial_vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        trial_vector = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

    first_vector = np.cross(unit_vector, trial_vector)
    first_norm = np.linalg.norm(first_vector)
    if first_norm <= 0.0:
        first_vector = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        first_vector = first_vector / first_norm
    second_vector = np.cross(unit_vector, first_vector)

    return first_vector, second_vector


def _draw_filled_arrow(
    ax,
    arrow_start: np.ndarray,
    vector: np.ndarray,
    color: str,
    label: str,
    linewidth: float = 4.4,
    head_length_ratio: float = 0.108,
    head_radius_ratio: float = 0.048,
):
    '''draw one opaque 3D arrow with a filled triangular head'''

    arrow_start = np.asarray(arrow_start, dtype=np.float64) # (3,)
    vector = np.asarray(vector, dtype=np.float64) # (3,)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 0.0:
        return

    unit_vector = vector / vector_norm # (3,)
    arrow_end = arrow_start + vector # (3,)
    head_length = head_length_ratio * vector_norm
    head_radius = head_radius_ratio * vector_norm
    head_base = arrow_end - head_length * unit_vector # (3,)

    # draw the shaft separately so the arrowhead can be filled
    ax.plot(
        [arrow_start[0], head_base[0]],
        [arrow_start[1], head_base[1]],
        [arrow_start[2], head_base[2]],
        color=color,
        linewidth=linewidth,
        alpha=1.0,
        label=label,
    )

    first_vector, second_vector = _perpendicular_basis(unit_vector=unit_vector)
    angles = np.linspace(0.0, 2.0 * np.pi, 9)[:-1]
    base_points = np.asarray([
        head_base
        + head_radius * np.cos(angle) * first_vector
        + head_radius * np.sin(angle) * second_vector
        for angle in angles
    ]) # (n_head_vertices, 3)

    faces = []
    for i_point in range(base_points.shape[0]):
        j_point = (i_point + 1) % base_points.shape[0]
        faces.append([arrow_end, base_points[i_point], base_points[j_point]])

    head = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors=color,
        linewidths=0.2,
        alpha=1.0,
    )
    ax.add_collection3d(head)


def multiple_dipole_moment_visual(
    positions_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
    dipole_vectors_au,
    output_path: str,
    labels=None,
    colors=None,
    bonds=None,
    normalize_vectors: bool = False,
    title: str = None,
    figure_size: float = 7.0,
    elev: float = 20.0,
    azim: float = 35.0,
    show_axes: bool = False,
    show_legend: bool = True,
    dpi: int = 300,
):
    '''visualize one molecule with multiple dipole moment arrows in atomic units'''

    positions_angstrom, atomic_numbers = _validate_molecule_arrays(
        positions_angstrom=positions_angstrom,
        atomic_numbers=atomic_numbers,
    )
    dipole_vectors_au = np.asarray(dipole_vectors_au, dtype=np.float64) # (n_vectors, 3)
    if dipole_vectors_au.ndim == 1:
        dipole_vectors_au = dipole_vectors_au.reshape(1, 3)
    if dipole_vectors_au.ndim != 2 or dipole_vectors_au.shape[1] != 3:
        raise ValueError(f"dipole_vectors_au must have shape (n_vectors, 3), got {dipole_vectors_au.shape}")

    n_vectors = dipole_vectors_au.shape[0]
    if labels is None:
        labels = [f"dipole {i}" for i in range(n_vectors)]
    if colors is None:
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    scaled_vectors = _scaled_dipole_vectors(
        dipole_vectors_au=dipole_vectors_au,
        positions_angstrom=positions_angstrom,
        normalize_vectors=normalize_vectors,
    ) # (n_vectors, 3)
    # arrows are centered at the geometric center of the molecule
    center = np.mean(positions_angstrom, axis=0) # (3,)

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111, projection="3d")
    # draw atoms and bonds before arrows so the dipole passes through the structure
    _draw_molecule(
        ax=ax,
        positions_angstrom=positions_angstrom,
        atomic_numbers=atomic_numbers,
        bonds=bonds,
    )

    arrow_points = [positions_angstrom]
    for i_vector in range(n_vectors):
        vector = scaled_vectors[i_vector, :] # (3,)
        # use the molecule center as the midpoint of the displayed arrow
        arrow_start = center - 0.5 * vector # (3,)
        arrow_end = center + 0.5 * vector # (3,)
        color = colors[i_vector % len(colors)]
        label = labels[i_vector]
        _draw_filled_arrow(
            ax=ax,
            arrow_start=arrow_start,
            vector=vector,
            color=color,
            label=label,
        )
        arrow_points.append(np.vstack((arrow_start[None, :], arrow_end[None, :]))) # (2, 3)

    all_points = np.vstack(arrow_points) # (n_atoms + 2 * n_vectors, 3)
    _set_axes_equal(ax=ax, points_angstrom=all_points)

    ax.view_init(elev=elev, azim=azim)
    if show_axes:
        ax.set_xlabel("x (Angstrom)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("y (Angstrom)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_zlabel("z (Angstrom)", fontsize=AXIS_LABEL_FONTSIZE)
    else:
        ax.set_axis_off()
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
    if show_legend:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    return _save_figure(fig=fig, output_path=output_path, dpi=dpi)


def dipole_moment_visual(
    positions_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
    dipole_vector_au: np.ndarray,
    output_path: str,
    label: str = "dipole",
    color: str = "#1f77b4",
    bonds=None,
    normalize_vector: bool = False,
    title: str = None,
    figure_size: float = 7.0,
    elev: float = 20.0,
    azim: float = 35.0,
    show_axes: bool = False,
    show_legend: bool = True,
    dpi: int = 300,
):
    '''visualize one molecule with one dipole moment arrow in atomic units'''

    return multiple_dipole_moment_visual(
        positions_angstrom=positions_angstrom,
        atomic_numbers=atomic_numbers,
        dipole_vectors_au=np.asarray(dipole_vector_au, dtype=np.float64).reshape(1, 3),
        output_path=output_path,
        labels=[label],
        colors=[color],
        bonds=bonds,
        normalize_vectors=normalize_vector,
        title=title,
        figure_size=figure_size,
        elev=elev,
        azim=azim,
        show_axes=show_axes,
        show_legend=show_legend,
        dpi=dpi,
    )
