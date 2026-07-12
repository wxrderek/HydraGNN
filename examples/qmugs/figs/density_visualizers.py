import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rdkit import Chem
from skimage import measure


BOHR_TO_ANGSTROM = 0.52917721092
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
        1: "#f2f2f2",   # H
        6: "#4a4a4a",   # C
        7: "#3050f8",   # N
        8: "#ff0d0d",   # O
        9: "#90e050",   # F
        15: "#ff8000",  # P
        16: "#ffff30",  # S
        17: "#1ff01f",  # Cl
        35: "#a62929",  # Br
        53: "#940094",  # I
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
        return 50.0 * size_scale
    return 100.0 * size_scale


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
                linewidth=1.5,
                alpha=1.0,
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
            linewidths=0.5,
            depthshade=True,
            label=Chem.GetPeriodicTable().GetElementSymbol(int(atomic_number)),
        )


def _set_axes_equal(ax, points_angstrom: np.ndarray):
    '''set equal 3D axis scaling around a collection of points'''

    points_angstrom = np.asarray(points_angstrom, dtype=np.float64) # (n_points, 3)
    if points_angstrom.ndim != 2 or points_angstrom.shape[1] != 3:
        raise ValueError(f"points_angstrom must have shape (n_points, 3), got {points_angstrom.shape}")

    min_xyz = np.min(points_angstrom, axis=0) # (3,)
    max_xyz = np.max(points_angstrom, axis=0) # (3,)
    center = 0.5 * (min_xyz + max_xyz) # (3,)
    radius = 0.5 * float(np.max(max_xyz - min_xyz))
    if radius <= 0.0:
        radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _select_density_points(
    density_values: np.ndarray,
    point_filter: str = "top_percentile",
    top_percentile: float = 99.5,
    absolute_cutoff: float = 1.0e-4,
    max_points: int = 5000,
):
    '''select density grid points for a readable point-cloud plot'''

    density_values = np.asarray(density_values, dtype=np.float64).reshape(-1) # (n_grid,)
    finite_mask = np.isfinite(density_values) # (n_grid,)
    finite_indices = np.where(finite_mask)[0] # (n_finite,)
    if finite_indices.size == 0:
        raise ValueError("No finite density values are available for plotting")

    # use absolute values so negative error-like fields can be filtered sensibly
    scores = np.abs(density_values) # (n_grid,)
    point_filter = str(point_filter).strip().lower()

    if point_filter == "top_percentile":
        threshold = np.percentile(scores[finite_indices], float(top_percentile))
        selected_indices = finite_indices[scores[finite_indices] >= threshold] # (n_selected,)
    elif point_filter == "absolute_cutoff":
        selected_indices = finite_indices[scores[finite_indices] >= float(absolute_cutoff)] # (n_selected,)
    elif point_filter == "max_points":
        selected_indices = finite_indices # (n_finite,)
    else:
        raise ValueError(f"Unknown density point filter: {point_filter}")

    if selected_indices.size == 0:
        # keep the largest finite value so the plot records that the calculation succeeded
        selected_indices = finite_indices[np.argsort(scores[finite_indices])[-1:]]

    # cap plotted points after thresholding to keep static figures readable
    if max_points is not None and int(max_points) > 0 and selected_indices.size > int(max_points):
        order = np.argsort(scores[selected_indices])
        selected_indices = selected_indices[order[-int(max_points):]] # (max_points,)

    return selected_indices


def _grid_points_from_metadata(selected_indices: np.ndarray, grid_shape, metadata):
    '''recover selected grid point coordinates from cube-style grid metadata'''

    selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1) # (n_selected,)
    grid_shape = tuple(int(value) for value in grid_shape)
    origin = np.asarray(metadata["origin"], dtype=np.float64) # (3,)
    axis_vectors = np.asarray(metadata["axis_vectors"], dtype=np.float64) # (3, 3)

    if origin.shape != (3,):
        raise ValueError(f"metadata origin must have shape (3,), got {origin.shape}")
    if axis_vectors.shape != (3, 3):
        raise ValueError(f"metadata axis_vectors must have shape (3, 3), got {axis_vectors.shape}")

    # cube metadata stores origin and voxel vectors in bohr
    grid_indices = np.asarray(np.unravel_index(selected_indices, grid_shape)).T # (n_selected, 3)
    points_bohr = origin[None, :] + grid_indices @ axis_vectors # (n_selected, 3)
    points_angstrom = points_bohr * BOHR_TO_ANGSTROM # (n_selected, 3)

    return points_angstrom


def _isosurface_level(
    density_values: np.ndarray,
    percentile: float = 99.0,
    absolute_level=None,
):
    '''choose one isosurface level inside the density range'''

    density_values = np.asarray(density_values, dtype=np.float64) # (n_x, n_y, n_z)
    finite_values = density_values[np.isfinite(density_values)] # (n_finite,)
    if finite_values.size == 0:
        raise ValueError("No finite density values are available for isosurface plotting")

    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    if max_value <= min_value:
        raise ValueError("Density values are constant; no isosurface can be extracted")

    if absolute_level is None:
        level = float(np.percentile(finite_values, float(percentile)))
    else:
        level = float(absolute_level)

    # marching cubes requires a level strictly inside the scalar range
    if level <= min_value or level >= max_value:
        level = 0.5 * (min_value + max_value)

    return level


def _isosurface_vertices_angstrom(density_values: np.ndarray, density_metadata, level: float):
    '''extract an isosurface mesh and convert vertices to Angstrom'''

    grid_shape = tuple(int(value) for value in density_metadata["grid_shape"])
    if density_values.shape != grid_shape:
        raise ValueError(f"density_values shape {density_values.shape} does not match grid_shape {grid_shape}")

    origin = np.asarray(density_metadata["origin"], dtype=np.float64) # (3,)
    axis_vectors = np.asarray(density_metadata["axis_vectors"], dtype=np.float64) # (3, 3)
    if origin.shape != (3,):
        raise ValueError(f"metadata origin must have shape (3,), got {origin.shape}")
    if axis_vectors.shape != (3, 3):
        raise ValueError(f"metadata axis_vectors must have shape (3, 3), got {axis_vectors.shape}")

    # marching cubes vertices are fractional grid indices in x, y, z array order
    vertices, faces, _, _ = measure.marching_cubes(density_values, level=level)
    vertices_bohr = origin[None, :] + vertices @ axis_vectors # (n_vertices, 3)
    vertices_angstrom = vertices_bohr * BOHR_TO_ANGSTROM # (n_vertices, 3)

    return vertices_angstrom, faces


def molecule_visual(
    positions_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
    output_path: str,
    bonds=None,
    title: str = None,
    figure_size: float = 7.0,
    elev: float = 20.0,
    azim: float = 35.0,
    show_axes: bool = False,
    show_legend: bool = True,
    dpi: int = 300,
):
    '''visualize a molecule in 3D using RDKit atom metadata'''

    positions_angstrom, atomic_numbers = _validate_molecule_arrays(
        positions_angstrom=positions_angstrom,
        atomic_numbers=atomic_numbers,
    )

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111, projection="3d")
    _draw_molecule(
        ax=ax,
        positions_angstrom=positions_angstrom,
        atomic_numbers=atomic_numbers,
        bonds=bonds,
    )
    _set_axes_equal(ax=ax, points_angstrom=positions_angstrom)

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


def density_visual(
    positions_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
    density_values: np.ndarray,
    density_metadata,
    output_path: str,
    bonds=None,
    title: str = None,
    point_filter: str = "top_percentile",
    top_percentile: float = 99.5,
    absolute_cutoff: float = 1.0e-4,
    max_points: int = 5000,
    density_label: str = "Electron density (a.u.)",
    density_cmap: str = "viridis",
    figure_size: float = 7.0,
    elev: float = 20.0,
    azim: float = 35.0,
    show_axes: bool = False,
    show_legend: bool = True,
    dpi: int = 300,
):
    '''visualize a molecule surrounded by a scalar density point cloud'''

    positions_angstrom, atomic_numbers = _validate_molecule_arrays(
        positions_angstrom=positions_angstrom,
        atomic_numbers=atomic_numbers,
    )
    density_values = np.asarray(density_values, dtype=np.float64) # (n_x, n_y, n_z)
    grid_shape = tuple(int(value) for value in density_metadata["grid_shape"])
    if density_values.shape != grid_shape:
        raise ValueError(f"density_values shape {density_values.shape} does not match grid_shape {grid_shape}")

    # select a readable subset before converting grid indices to coordinates
    selected_indices = _select_density_points(
        density_values=density_values.reshape(-1),
        point_filter=point_filter,
        top_percentile=top_percentile,
        absolute_cutoff=absolute_cutoff,
        max_points=max_points,
    ) # (n_selected,)
    selected_points = _grid_points_from_metadata(
        selected_indices=selected_indices,
        grid_shape=grid_shape,
        metadata=density_metadata,
    ) # (n_selected, 3)
    selected_values = density_values.reshape(-1)[selected_indices] # (n_selected,)

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111, projection="3d")

    # draw the density cloud first so atoms and bonds remain visible on top
    scatter = ax.scatter(
        selected_points[:, 0],
        selected_points[:, 1],
        selected_points[:, 2],
        c=selected_values,
        cmap=density_cmap,
        s=4.0,
        alpha=0.4,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(density_label, fontsize=AXIS_LABEL_FONTSIZE)

    _draw_molecule(
        ax=ax,
        positions_angstrom=positions_angstrom,
        atomic_numbers=atomic_numbers,
        bonds=bonds,
    )

    all_points = np.vstack((positions_angstrom, selected_points)) # (n_atoms + n_selected, 3)
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


def density_isosurface_visual(
    positions_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
    density_values: np.ndarray,
    density_metadata,
    output_path: str,
    bonds=None,
    title: str = None,
    isosurface_percentile: float = 99.0,
    isosurface_absolute_level=None,
    density_cmap: str = "viridis",
    density_alpha: float = 0.35,
    figure_size: float = 7.0,
    elev: float = 20.0,
    azim: float = 35.0,
    show_axes: bool = False,
    show_legend: bool = True,
    dpi: int = 300,
):
    '''visualize one scalar density isosurface around a molecule'''

    positions_angstrom, atomic_numbers = _validate_molecule_arrays(
        positions_angstrom=positions_angstrom,
        atomic_numbers=atomic_numbers,
    )
    density_values = np.asarray(density_values, dtype=np.float64) # (n_x, n_y, n_z)

    level = _isosurface_level(
        density_values=density_values,
        percentile=isosurface_percentile,
        absolute_level=isosurface_absolute_level,
    )
    vertices_angstrom, faces = _isosurface_vertices_angstrom(
        density_values=density_values,
        density_metadata=density_metadata,
        level=level,
    )

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111, projection="3d")

    # draw the continuous density shell first so atoms and bonds remain visible on top
    face_color = plt.get_cmap(density_cmap)(0.70)
    surface = Poly3DCollection(
        vertices_angstrom[faces],
        facecolors=face_color,
        edgecolors="none",
        alpha=float(density_alpha),
    )
    ax.add_collection3d(surface)

    _draw_molecule(
        ax=ax,
        positions_angstrom=positions_angstrom,
        atomic_numbers=atomic_numbers,
        bonds=bonds,
    )

    all_points = np.vstack((positions_angstrom, vertices_angstrom)) # (n_atoms + n_vertices, 3)
    _set_axes_equal(ax=ax, points_angstrom=all_points)

    ax.view_init(elev=elev, azim=azim)
    if show_axes:
        ax.set_xlabel("x (Angstrom)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("y (Angstrom)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_zlabel("z (Angstrom)", fontsize=AXIS_LABEL_FONTSIZE)
    else:
        ax.set_axis_off()
    if title is not None:
        ax.set_title(f"{title}\nlevel = {level:.4e}", fontsize=TITLE_FONTSIZE)
    if show_legend:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    return _save_figure(fig=fig, output_path=output_path, dpi=dpi)


def density_signed_isosurface_visual(
    positions_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
    density_values: np.ndarray,
    density_metadata,
    output_path: str,
    bonds=None,
    title: str = None,
    isosurface_percentile: float = 99.0,
    isosurface_absolute_level=None,
    positive_color: str = "#d62728",
    negative_color: str = "#1f77b4",
    density_alpha: float = 0.35,
    figure_size: float = 7.0,
    elev: float = 20.0,
    azim: float = 35.0,
    show_axes: bool = False,
    show_legend: bool = True,
    dpi: int = 300,
):
    '''visualize positive and negative isosurfaces of one signed density field'''

    positions_angstrom, atomic_numbers = _validate_molecule_arrays(
        positions_angstrom=positions_angstrom,
        atomic_numbers=atomic_numbers,
    )
    density_values = np.asarray(density_values, dtype=np.float64) # (n_x, n_y, n_z)
    finite_values = density_values[np.isfinite(density_values)] # (n_finite,)
    if finite_values.size == 0:
        raise ValueError("No finite density values are available for signed isosurface plotting")

    abs_values = np.abs(finite_values) # (n_finite,)
    if isosurface_absolute_level is None:
        level = float(np.percentile(abs_values, float(isosurface_percentile)))
    else:
        level = float(isosurface_absolute_level)
    max_abs_value = float(np.max(abs_values))
    if level <= 0.0 or level >= max_abs_value:
        level = 0.5 * max_abs_value
    if level <= 0.0:
        raise ValueError("Signed density values are zero; no isosurface can be extracted")

    fig = plt.figure(figsize=(figure_size, figure_size))
    ax = fig.add_subplot(111, projection="3d")
    all_points = [positions_angstrom]

    # positive surface indicates overpredicted density
    if float(np.max(finite_values)) > level:
        vertices_angstrom, faces = _isosurface_vertices_angstrom(
            density_values=density_values,
            density_metadata=density_metadata,
            level=level,
        )
        surface = Poly3DCollection(
            vertices_angstrom[faces],
            facecolors=positive_color,
            edgecolors="none",
            alpha=float(density_alpha),
        )
        ax.add_collection3d(surface)
        all_points.append(vertices_angstrom)

    # negative surface indicates underpredicted density
    if float(np.min(finite_values)) < -level:
        vertices_angstrom, faces = _isosurface_vertices_angstrom(
            density_values=density_values,
            density_metadata=density_metadata,
            level=-level,
        )
        surface = Poly3DCollection(
            vertices_angstrom[faces],
            facecolors=negative_color,
            edgecolors="none",
            alpha=float(density_alpha),
        )
        ax.add_collection3d(surface)
        all_points.append(vertices_angstrom)

    _draw_molecule(
        ax=ax,
        positions_angstrom=positions_angstrom,
        atomic_numbers=atomic_numbers,
        bonds=bonds,
    )

    all_points = np.vstack(all_points)
    _set_axes_equal(ax=ax, points_angstrom=all_points)

    ax.view_init(elev=elev, azim=azim)
    if show_axes:
        ax.set_xlabel("x (Angstrom)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("y (Angstrom)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_zlabel("z (Angstrom)", fontsize=AXIS_LABEL_FONTSIZE)
    else:
        ax.set_axis_off()
    if title is not None:
        ax.set_title(f"{title}\nlevels = +/- {level:.4e}", fontsize=TITLE_FONTSIZE)
    if show_legend:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    return _save_figure(fig=fig, output_path=output_path, dpi=dpi)
