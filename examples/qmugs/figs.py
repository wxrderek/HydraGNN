import os
import sys
import json
import glob
import argparse
import logging

from mpi4py import MPI

import numpy as np
import pandas as pd

import torch
from torch_geometric.loader import DataLoader

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import hydragnn
import hydragnn.utils.model
import hydragnn.utils.print
from hydragnn.utils.print.print_utils import log

# allow figs.py to be run as a standalone script from shell scripts
QMUGS_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
if QMUGS_EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, QMUGS_EXAMPLE_DIR)

from figs.density_matrix_visualizers import (
    density_matrix_heatmap,
    density_matrix_heatmap_with_mask,
)
from figs.density_visualizers import (
    density_isosurface_visual,
    density_signed_isosurface_visual,
    density_visual,
    molecule_visual,
)
from figs.dipole_moment_visualizers import multiple_dipole_moment_visual
from figs.metrics_visualizers import (
    plot_metric_by_feature,
    reconstruction_scatter,
)


DEFAULT_RANDOM_STATE = 0


# ----------------------------------------------------------------------------------------------------
# small shared helpers

def _is_scalar_number(value):
    '''return whether a value is a non-boolean scalar number'''

    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _record_key(record):
    '''return the molecule-conformer key for one metrics record'''

    chembl_id = str(record["chembl_id"])
    conformer_id = str(record["conformer_id"]).zfill(2)
    return chembl_id, conformer_id


def _sample_base_name(record):
    '''return the standard figure filename stem for one sampled conformer'''

    chembl_id, conformer_id = _record_key(record)
    return f"{chembl_id}_conf_{conformer_id}"


def _sample_output_dir(output_dir: str, record):
    '''return the per-conformer figure directory for one sampled record'''

    sample_dir = os.path.join(output_dir, _sample_base_name(record))
    os.makedirs(sample_dir, exist_ok=True)
    return sample_dir


def _average_pool_matrix(matrix: np.ndarray, pool_size: int):
    '''average pool a square matrix by non-overlapping blocks'''

    matrix = np.asarray(matrix, dtype=np.float64) # (n_basis, n_basis)
    if pool_size is None or int(pool_size) <= 1:
        return matrix

    pool_size = int(pool_size)
    n_rows, n_cols = matrix.shape
    pooled_rows = int(np.ceil(n_rows / pool_size))
    pooled_cols = int(np.ceil(n_cols / pool_size))
    pooled_matrix = np.zeros((pooled_rows, pooled_cols), dtype=np.float64)

    # edge blocks may be smaller when the matrix size is not divisible by pool_size
    for i_row in range(pooled_rows):
        row_start = i_row * pool_size
        row_stop = min(row_start + pool_size, n_rows)
        for i_col in range(pooled_cols):
            col_start = i_col * pool_size
            col_stop = min(col_start + pool_size, n_cols)
            pooled_matrix[i_row, i_col] = np.mean(matrix[row_start:row_stop, col_start:col_stop])

    return pooled_matrix


def _average_pool_mask(mask: np.ndarray, pool_size: int):
    '''pool a boolean mask by requiring all entries in a block to be true'''

    if mask is None or pool_size is None or int(pool_size) <= 1:
        return mask

    mask = np.asarray(mask).astype(bool) # (n_basis, n_basis)
    pool_size = int(pool_size)
    n_rows, n_cols = mask.shape
    pooled_rows = int(np.ceil(n_rows / pool_size))
    pooled_cols = int(np.ceil(n_cols / pool_size))
    pooled_mask = np.ones((pooled_rows, pooled_cols), dtype=bool)

    # if any entry in a pooled block was withheld, show the block as withheld
    for i_row in range(pooled_rows):
        row_start = i_row * pool_size
        row_stop = min(row_start + pool_size, n_rows)
        for i_col in range(pooled_cols):
            col_start = i_col * pool_size
            col_stop = min(col_start + pool_size, n_cols)
            pooled_mask[i_row, i_col] = np.all(mask[row_start:row_stop, col_start:col_stop])

    return pooled_mask


def _load_sdf_bonds(sample, qmugs_data_dir: str = None):
    '''load molecular bonds from the QMugs SDF file when available'''

    record = sample["record"]
    chembl_id, conformer_id = _record_key(record)

    candidate_paths = []
    if qmugs_data_dir is not None:
        # prefer the explicit QMugs data directory when it was saved in config
        candidate_paths.append(
            os.path.join(qmugs_data_dir, "structures", chembl_id, f"conf_{conformer_id}.sdf")
        )

    wfn_path = sample.get("wfn_path", None)
    if wfn_path is not None and "/wfns/" in wfn_path:
        # otherwise infer the data root from the wavefunction path convention
        data_dir = wfn_path.split("/wfns/")[0]
        candidate_paths.append(
            os.path.join(data_dir, "structures", chembl_id, f"conf_{conformer_id}.sdf")
        )

    for sdf_path in candidate_paths:
        if not os.path.isfile(sdf_path):
            continue
        # RDKit supplies only bond connectivity for visual context
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
        if len(supplier) == 0 or supplier[0] is None:
            continue
        mol = supplier[0]
        return [
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in mol.GetBonds()
        ]

    return None


# ----------------------------------------------------------------------------------------------------
# sampled inference

def sample_inference(pretrained_model, dataset, selected_records):
    '''rerun inference on a small selected dataset and return unpadded matrices'''

    # output_dim is N_pad ** 2 for the total density matrix head
    var_config = pretrained_model.config["NeuralNetwork"]["Variables_of_interest"]
    output_dim = int(var_config["output_dim"][0])
    max_size = int(round(output_dim ** 0.5))
    if max_size * max_size != output_dim:
        raise ValueError(f"Density matrix output_dim must be a square, got {output_dim}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    sample_results = []
    for i_sample, batch in enumerate(loader):
        record = selected_records[i_sample]

        # batch_size is one, so node-level arrays correspond to one conformer
        positions_angstrom = batch.pos.detach().cpu().numpy().astype(np.float64) # (n_atoms, 3)
        if hasattr(batch, "atomic_numbers"):
            atomic_numbers = batch.atomic_numbers.detach().cpu().numpy().reshape(-1).astype(int) # (n_atoms,)
        else:
            atomic_numbers = batch.x[:, 0].detach().cpu().numpy().reshape(-1).astype(int) # (n_atoms,)

        wfn_path = None
        if hasattr(batch, "wfn_path"):
            # PyG collation can present string metadata as lists or tensors
            wfn_path_value = batch.wfn_path
            if isinstance(wfn_path_value, (list, tuple)):
                wfn_path = str(wfn_path_value[0])
            elif torch.is_tensor(wfn_path_value):
                if wfn_path_value.dim() == 0:
                    wfn_path = str(wfn_path_value.item())
                else:
                    wfn_path = str(wfn_path_value[0].item())
            else:
                wfn_path = str(wfn_path_value)

        # inference runs on the model device, but stored plot inputs are converted back to NumPy
        batch = batch.to(pretrained_model.device)
        with torch.no_grad():
            with pretrained_model.autocast_ctx:
                pred = pretrained_model.model(batch)

            if var_config["output_names"] != ["density_matrix"]:
                raise NotImplementedError("Figure generation currently supports only total density matrix predictions")

            pred_batch = pred[0].reshape(-1, max_size, max_size) # (batch_size, N_pad, N_pad)
            true_batch = batch.y.reshape(-1, max_size, max_size) # (batch_size, N_pad, N_pad)
            dims_batch = batch.density_matrix_dim # (batch_size,)

            raw_pred_mat = pred_batch[0, :, :] # (N_pad, N_pad)
            raw_true_mat = true_batch[0, :, :] # (N_pad, N_pad)
            dim = int(dims_batch[0].item())

            # remove padding before any plotting or downstream scalar-field calculation
            pred_mat = raw_pred_mat[:dim, :dim].detach().cpu().numpy().astype(np.float64) # (n_basis, n_basis)
            true_mat = raw_true_mat[:dim, :dim].detach().cpu().numpy().astype(np.float64) # (n_basis, n_basis)
            assert pred_mat.shape == true_mat.shape

            mask = None
            if hasattr(batch, "y_mask") and batch.y_mask is not None:
                # compact masks are stored over the unpadded density matrix entries
                compact_mask = batch.y_mask.detach().cpu().numpy().reshape(-1).astype(bool) # (n_basis * n_basis,)
                if compact_mask.size == dim * dim:
                    # mask == True means the entry was supervised by the loss
                    mask = compact_mask.reshape(dim, dim) # (n_basis, n_basis)

        sample_results.append({
            "record": record,
            "pred_matrix": pred_mat,
            "true_matrix": true_mat,
            "mask": mask,
            "positions_angstrom": positions_angstrom,
            "atomic_numbers": atomic_numbers,
            "wfn_path": wfn_path,
        })

    return sample_results


# ----------------------------------------------------------------------------------------------------
# figure group functions

def plot_metric_visuals(
    metrics,
    output_dir: str,
    metrics_max_natoms=None,
    metrics_max_density_matrix_dim=None,
    metrics_natoms_bin_size=None,
    metrics_density_matrix_dim_bin_size=None,
):
    '''plot all scalar metrics over atom counts and density matrix dimensions'''

    # ----------------------------------------------------------------------------------------------------
    # local helpers

    def _sanitize_filename(name: str):
        '''sanitize a string for use in figure filenames'''

        sanitized = []
        for char in str(name):
            if char.isalnum() or char in ["_", "-"]:
                sanitized.append(char)
            else:
                sanitized.append("_")
        return "".join(sanitized)

    def _metric_columns():
        '''return scalar per-matrix metric columns to plot'''

        # identifiers and raw molecule properties are features, not metrics
        metadata_columns = {
            "chembl_id",
            "conformer_id",
            "density_matrix_dim",
            "natoms",
            "charge",
            "multiplicity",
        }

        metric_names = []
        seen = set()
        for record in metrics:
            for key, value in record.items():
                # plot only scalar diagnostics, excluding paths, vectors, and lists
                if key in metadata_columns or key in seen:
                    continue
                if _is_scalar_number(value):
                    metric_names.append(key)
                    seen.add(key)

        return metric_names

    # ----------------------------------------------------------------------------------------------------
    # plot scalar metrics

    # metrics has one record per conformer from per_matrix_metrics.json
    if len(metrics) == 0:
        log("[WARNING] No per-matrix metrics found; skipping metric plots", rank=0)
        return

    metrics_frame = pd.DataFrame(metrics)
    metric_names = _metric_columns()
    if len(metric_names) == 0:
        log("[WARNING] No scalar metric columns found; skipping metric plots", rank=0)
        return

    for feature_name in [
        "natoms", 
        # "density_matrix_dim",
    ]:
        if feature_name not in metrics_frame.columns:
            log(f"[WARNING] Feature {feature_name} is missing; skipping metric plots over this feature", rank=0)
            continue

        if feature_name == "natoms":
            max_feature_value = metrics_max_natoms
            feature_bin_size = metrics_natoms_bin_size
        elif feature_name == "density_matrix_dim":
            max_feature_value = metrics_max_density_matrix_dim
            feature_bin_size = metrics_density_matrix_dim_bin_size
        else:
            max_feature_value = None
            feature_bin_size = None

        for metric_name in metric_names:
            output_path = os.path.join(
                output_dir,
                f"{_sanitize_filename(metric_name)}_by_{feature_name}",
            )
            try:
                plot_metric_by_feature(
                    metrics_frame=metrics_frame,
                    metric_name=metric_name,
                    feature_name=feature_name,
                    output_path=output_path,
                    title=f"{metric_name} by {feature_name}",
                    xlabel=feature_name,
                    ylabel=metric_name,
                    max_feature_value=max_feature_value,
                    feature_bin_size=feature_bin_size,
                )
            except ValueError as exc:
                log(f"[WARNING] Skipping metric plot {metric_name} by {feature_name}: {exc}", rank=0)


def plot_density_matrix_visuals(
    sample_results,
    output_dir: str,
    matrix_pool_size: int = 1,
):
    '''plot matrix heatmaps and reconstruction scatters for sampled conformers'''

    # collect sampled matrices for the pooled reconstruction scatter
    all_true_matrices = []
    all_pred_matrices = []

    for sample in sample_results:
        # unpack one sampled conformer
        record = sample["record"]
        base_name = _sample_base_name(record)
        sample_dir = _sample_output_dir(output_dir=output_dir, record=record)
        true_matrix = sample["true_matrix"] # (n_basis, n_basis)
        pred_matrix = sample["pred_matrix"] # (n_basis, n_basis)
        mask = sample["mask"] # (n_basis, n_basis) or None

        all_true_matrices.append(true_matrix)
        all_pred_matrices.append(pred_matrix)

        heatmap_true_matrix = _average_pool_matrix(true_matrix, matrix_pool_size) # (n_plot, n_plot)
        heatmap_pred_matrix = _average_pool_matrix(pred_matrix, matrix_pool_size) # (n_plot, n_plot)
        heatmap_mask = _average_pool_mask(mask, matrix_pool_size) # (n_plot, n_plot) or None

        # true and predicted heatmaps share one color scale for each conformer
        matrix_vmin = float(min(np.min(heatmap_true_matrix), np.min(heatmap_pred_matrix)))
        matrix_vmax = float(max(np.max(heatmap_true_matrix), np.max(heatmap_pred_matrix)))
        if matrix_vmax <= matrix_vmin:
            matrix_vmax = matrix_vmin + 1.0

        difference = heatmap_pred_matrix - heatmap_true_matrix # (n_plot, n_plot)
        abs_difference = np.abs(difference) # (n_plot, n_plot)
        difference_absmax = float(np.max(np.abs(difference)))
        if difference_absmax <= 0.0:
            difference_absmax = 1.0
        abs_difference_max = float(np.max(abs_difference))
        if abs_difference_max <= 0.0:
            abs_difference_max = 1.0

        # each tuple defines one heatmap family for this conformer
        plot_specs = [
            ("true_density_matrix", heatmap_true_matrix, "True density matrix", "viridis", matrix_vmin, matrix_vmax),
            ("predicted_density_matrix", heatmap_pred_matrix, "Predicted density matrix", "viridis", matrix_vmin, matrix_vmax),
            ("density_matrix_difference", difference, "Predicted - true density matrix", "viridis", -difference_absmax, difference_absmax),
            ("absolute_density_matrix_difference", abs_difference, "Absolute density matrix difference", "viridis", 0.0, abs_difference_max),
        ]

        pool_suffix = "" if int(matrix_pool_size) <= 1 else f"_pooled_{int(matrix_pool_size)}x{int(matrix_pool_size)}"
        for suffix, matrix, title, cmap, vmin, vmax in plot_specs:
            output_path = os.path.join(sample_dir, f"{base_name}_{suffix}{pool_suffix}")
            density_matrix_heatmap(
                matrix=matrix,
                output_path=output_path,
                title=title,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            # overlay only entries withheld from the loss
            if heatmap_mask is not None and np.any(~heatmap_mask):
                masked_output_path = os.path.join(sample_dir, f"{base_name}_{suffix}{pool_suffix}_with_mask")
                density_matrix_heatmap_with_mask(
                    matrix=matrix,
                    mask=heatmap_mask,
                    output_path=masked_output_path,
                    title=title + " with withheld entries",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )

        # reconstruction scatter plots are disabled for now
        # reconstruction_scatter(
        #     true_matrices=true_matrix,
        #     predicted_matrices=pred_matrix,
        #     output_path=os.path.join(sample_dir, f"{base_name}_reconstruction_scatter"),
        #     title=f"{base_name} reconstruction scatter",
        #     zscore=True,
        # )

    # reconstruction scatter plots are disabled for now
    # if len(all_true_matrices) > 0:
    #     # pooled scatter uses one z-score normalization over all sampled entries
    #     reconstruction_scatter(
    #         true_matrices=all_true_matrices,
    #         predicted_matrices=all_pred_matrices,
    #         output_path=os.path.join(output_dir, "sampled_reconstruction_scatter"),
    #         title="Sampled reconstruction scatter",
    #         zscore=True,
    #     )


def plot_electron_density_visuals(
    sample_results,
    output_dir: str,
    artifacts_dir: str,
    inference_config,
    qmugs_data_dir: str = None,
    density_point_filter: str = "top_percentile",
    density_top_percentile: float = 99.5,
    density_absolute_cutoff: float = 1.0e-4,
    density_max_points: int = 5000,
    density_isosurface_percentile: float = 99.0,
    density_isosurface_absolute_level=None,
    show_axes: bool = False,
    show_legend: bool = True,
):
    '''plot true, predicted, and error electron density isosurfaces'''

    # ----------------------------------------------------------------------------------------------------
    # local helpers for density grid recovery

    def _existing_cubeprop_dirs(record):
        '''return candidate true and predicted cubeprop directories for one record'''

        candidates = []

        # new inference records store exact cube file paths when cubeprop was used
        true_paths = record.get("electron_density_true_cube_paths", None)
        pred_paths = record.get("electron_density_predicted_cube_paths", None)
        if isinstance(true_paths, list) and isinstance(pred_paths, list):
            if len(true_paths) > 0 and len(pred_paths) > 0:
                candidates.append((os.path.dirname(true_paths[0]), os.path.dirname(pred_paths[0])))

        # older records may store only the molecule-level artifact directory
        artifact_dir = record.get("electron_density_artifact_dir", None)
        if isinstance(artifact_dir, str) and len(artifact_dir) > 0:
            candidates.append((
                os.path.join(artifact_dir, "true"),
                os.path.join(artifact_dir, "predicted"),
            ))

        # fall back to the rank-sharded cubeprop directory convention
        if artifacts_dir is not None:
            chembl_id, conformer_id = _record_key(record)
            pattern = os.path.join(
                artifacts_dir,
                "cubeprops",
                "rank_*",
                f"{chembl_id}_conf_{conformer_id}",
                "electron_density",
            )
            for molecule_dir in glob.glob(pattern):
                candidates.append((
                    os.path.join(molecule_dir, "true"),
                    os.path.join(molecule_dir, "predicted"),
                ))

        return candidates

    def _read_existing_density(record):
        '''read existing true and predicted cubeprop densities if both are available'''

        from utils.inference_utils import assert_same_cube_grid, read_cubeprops

        # use cubeprops only when both true and predicted density cubes exist
        for true_dir, pred_dir in _existing_cubeprop_dirs(record=record):
            true_cube_files = glob.glob(os.path.join(true_dir, "*.cube"))
            pred_cube_files = glob.glob(os.path.join(pred_dir, "*.cube"))
            if len(true_cube_files) == 0 or len(pred_cube_files) == 0:
                continue

            try:
                true_cubeprops = read_cubeprops(cube_dir=true_dir, cubeprop_names=["density"])
                pred_cubeprops = read_cubeprops(cube_dir=pred_dir, cubeprop_names=["density"])
                true_density = true_cubeprops["density"]["values"] # (n_x, n_y, n_z)
                pred_density = pred_cubeprops["density"]["values"] # (n_x, n_y, n_z)
                true_metadata = true_cubeprops["density"]["metadata"]
                pred_metadata = pred_cubeprops["density"]["metadata"]
                assert_same_cube_grid(
                    first_cube_metadata=true_metadata,
                    second_cube_metadata=pred_metadata,
                )
                return true_density, pred_density, true_metadata
            except Exception as exc:
                log(f"[WARNING] Failed to read cubeprops from {true_dir} and {pred_dir}: {exc}", rank=0)

        return None

    def _compute_manual_density(sample):
        '''compute true and predicted electron densities on the same manual AO grid'''

        import psi4
        from utils.inference_utils import (
            assert_same_cube_grid,
            electron_density_from_density_matrix,
        )

        wfn_path = sample.get("wfn_path", None)
        if wfn_path is None or not os.path.isfile(wfn_path):
            raise FileNotFoundError(f"Cannot recompute electron density without wfn_path: {wfn_path}")

        psi4_wfn = psi4.core.Wavefunction.from_file(wfn_path)
        inference_config_local = inference_config or {}
        padding_angstrom = inference_config_local.get("manual_density_padding_angstrom", 3.0)
        spacing_angstrom = inference_config_local.get("manual_density_spacing_angstrom", None)
        num_grid_points = inference_config_local.get("manual_density_num_grid_points", None)
        block_size = inference_config_local.get("manual_density_block_size", 2000)

        # true and predicted densities are evaluated on the same manual AO grid
        true_density, true_metadata = electron_density_from_density_matrix(
            density_matrix=sample["true_matrix"],
            psi4_wfn=psi4_wfn,
            padding_angstrom=padding_angstrom,
            spacing_angstrom=spacing_angstrom,
            num_grid_points=num_grid_points,
            block_size=block_size,
        ) # (n_x, n_y, n_z)
        pred_density, pred_metadata = electron_density_from_density_matrix(
            density_matrix=sample["pred_matrix"],
            psi4_wfn=psi4_wfn,
            padding_angstrom=padding_angstrom,
            spacing_angstrom=spacing_angstrom,
            num_grid_points=num_grid_points,
            block_size=block_size,
        ) # (n_x, n_y, n_z)

        assert_same_cube_grid(
            first_cube_metadata=true_metadata,
            second_cube_metadata=pred_metadata,
        )

        return true_density, pred_density, true_metadata

    def _density_arrays_for_sample(sample):
        '''load cubeprop densities when possible and otherwise recompute them manually'''

        record = sample["record"]
        existing_density = _read_existing_density(record=record)
        if existing_density is not None:
            return existing_density

        # incomplete cubeprop artifacts are discarded to guarantee a common grid
        log(
            f"[WARNING] Cubeprops unavailable or incomplete for {_sample_base_name(record)}; "
            "recomputing true and predicted densities manually",
            rank=0,
        )
        return _compute_manual_density(sample=sample)

    # ----------------------------------------------------------------------------------------------------
    # plot density visuals

    if artifacts_dir is None:
        log("[WARNING] artifacts_dir was not provided; skipping electron density visualizations", rank=0)
        return

    for sample in sample_results:
        # molecule bonds are used only for visual context
        record = sample["record"]
        base_name = _sample_base_name(record)
        sample_dir = _sample_output_dir(output_dir=output_dir, record=record)
        bonds = _load_sdf_bonds(sample=sample, qmugs_data_dir=qmugs_data_dir)

        try:
            true_density, pred_density, metadata = _density_arrays_for_sample(sample=sample)
        except Exception as exc:
            log(f"[WARNING] Skipping electron density visuals for {base_name}: {exc}", rank=0)
            continue

        # density differences are plotted on the same grid as true and predicted densities
        density_difference = pred_density - true_density # (n_x, n_y, n_z)
        abs_difference = np.abs(density_difference) # (n_x, n_y, n_z)

        molecule_visual(
            positions_angstrom=sample["positions_angstrom"],
            atomic_numbers=sample["atomic_numbers"],
            output_path=os.path.join(sample_dir, f"{base_name}_molecule"),
            bonds=bonds,
            title=f"{base_name} molecule",
            show_axes=show_axes,
            show_legend=show_legend,
        )

        # density point-cloud plots are disabled for now
        # density_visual(
        #     positions_angstrom=sample["positions_angstrom"],
        #     atomic_numbers=sample["atomic_numbers"],
        #     density_values=true_density,
        #     density_metadata=metadata,
        #     output_path=os.path.join(sample_dir, f"{base_name}_true_electron_density"),
        #     bonds=bonds,
        #     title=f"{base_name} true electron density",
        #     point_filter=density_point_filter,
        #     top_percentile=density_top_percentile,
        #     absolute_cutoff=density_absolute_cutoff,
        #     max_points=density_max_points,
        #     density_label="True electron density (a.u.)",
        #     show_axes=show_axes,
        #     show_legend=show_legend,
        # )
        density_isosurface_visual(
            positions_angstrom=sample["positions_angstrom"],
            atomic_numbers=sample["atomic_numbers"],
            density_values=true_density,
            density_metadata=metadata,
            output_path=os.path.join(sample_dir, f"{base_name}_true_electron_density_isosurface"),
            bonds=bonds,
            title=f"{base_name} true electron density isosurface",
            isosurface_percentile=density_isosurface_percentile,
            isosurface_absolute_level=density_isosurface_absolute_level,
            density_cmap="viridis",
            show_axes=show_axes,
            show_legend=show_legend,
        )
        # density point-cloud plots are disabled for now
        # density_visual(
        #     positions_angstrom=sample["positions_angstrom"],
        #     atomic_numbers=sample["atomic_numbers"],
        #     density_values=pred_density,
        #     density_metadata=metadata,
        #     output_path=os.path.join(sample_dir, f"{base_name}_predicted_electron_density"),
        #     bonds=bonds,
        #     title=f"{base_name} predicted electron density",
        #     point_filter=density_point_filter,
        #     top_percentile=density_top_percentile,
        #     absolute_cutoff=density_absolute_cutoff,
        #     max_points=density_max_points,
        #     density_label="Predicted electron density (a.u.)",
        #     show_axes=show_axes,
        #     show_legend=show_legend,
        # )
        density_isosurface_visual(
            positions_angstrom=sample["positions_angstrom"],
            atomic_numbers=sample["atomic_numbers"],
            density_values=pred_density,
            density_metadata=metadata,
            output_path=os.path.join(sample_dir, f"{base_name}_predicted_electron_density_isosurface"),
            bonds=bonds,
            title=f"{base_name} predicted electron density isosurface",
            isosurface_percentile=density_isosurface_percentile,
            isosurface_absolute_level=density_isosurface_absolute_level,
            density_cmap="viridis",
            show_axes=show_axes,
            show_legend=show_legend,
        )
        # density point-cloud plots are disabled for now
        # density_visual(
        #     positions_angstrom=sample["positions_angstrom"],
        #     atomic_numbers=sample["atomic_numbers"],
        #     density_values=abs_difference,
        #     density_metadata=metadata,
        #     output_path=os.path.join(sample_dir, f"{base_name}_absolute_density_difference"),
        #     bonds=bonds,
        #     title=f"{base_name} absolute electron density difference",
        #     point_filter=density_point_filter,
        #     top_percentile=density_top_percentile,
        #     absolute_cutoff=density_absolute_cutoff,
        #     max_points=density_max_points,
        #     density_label="Absolute density difference (a.u.)",
        #     density_cmap="magma",
        #     show_axes=show_axes,
        #     show_legend=show_legend,
        # )
        density_isosurface_visual(
            positions_angstrom=sample["positions_angstrom"],
            atomic_numbers=sample["atomic_numbers"],
            density_values=abs_difference,
            density_metadata=metadata,
            output_path=os.path.join(sample_dir, f"{base_name}_absolute_density_difference_isosurface"),
            bonds=bonds,
            title=f"{base_name} absolute electron density difference isosurface",
            isosurface_percentile=density_isosurface_percentile,
            isosurface_absolute_level=density_isosurface_absolute_level,
            density_cmap="magma",
            show_axes=show_axes,
            show_legend=show_legend,
        )
        density_signed_isosurface_visual(
            positions_angstrom=sample["positions_angstrom"],
            atomic_numbers=sample["atomic_numbers"],
            density_values=density_difference,
            density_metadata=metadata,
            output_path=os.path.join(sample_dir, f"{base_name}_signed_density_difference_isosurface"),
            bonds=bonds,
            title=f"{base_name} signed electron density difference isosurface",
            isosurface_percentile=density_isosurface_percentile,
            isosurface_absolute_level=density_isosurface_absolute_level,
            positive_color="#d62728",
            negative_color="#1f77b4",
            show_axes=show_axes,
            show_legend=show_legend,
        )


def plot_dipole_visuals(
    sample_results,
    output_dir: str,
    qmugs_data_dir: str = None,
    show_axes: bool = False,
    show_legend: bool = True,
):
    '''plot true and predicted dipole moment arrows for sampled conformers'''

    # ----------------------------------------------------------------------------------------------------
    # local helpers

    def _dipoles_for_sample(sample):
        '''return true and predicted dipole moments from records or recomputation'''

        record = sample["record"]
        if "true_dipole_au" in record and "predicted_dipole_au" in record:
            # reuse downstream records when inference already computed dipoles
            true_dipole = np.asarray(record["true_dipole_au"], dtype=np.float64) # (3,)
            pred_dipole = np.asarray(record["predicted_dipole_au"], dtype=np.float64) # (3,)
            return true_dipole, pred_dipole

        import psi4
        from utils.inference_utils import dipole_moment_from_density_matrix

        wfn_path = sample.get("wfn_path", None)
        if wfn_path is None or not os.path.isfile(wfn_path):
            raise FileNotFoundError(f"Cannot compute dipole moment without wfn_path: {wfn_path}")

        psi4_wfn = psi4.core.Wavefunction.from_file(wfn_path)
        # recompute dipoles only for the small sampled subset
        true_dipole, _, _ = dipole_moment_from_density_matrix(
            density_matrix=sample["true_matrix"],
            psi4_wfn=psi4_wfn,
        ) # (3,)
        pred_dipole, _, _ = dipole_moment_from_density_matrix(
            density_matrix=sample["pred_matrix"],
            psi4_wfn=psi4_wfn,
        ) # (3,)

        return true_dipole, pred_dipole

    # ----------------------------------------------------------------------------------------------------
    # plot dipole visuals

    for sample in sample_results:
        record = sample["record"]
        base_name = _sample_base_name(record)
        sample_dir = _sample_output_dir(output_dir=output_dir, record=record)
        bonds = _load_sdf_bonds(sample=sample, qmugs_data_dir=qmugs_data_dir)

        try:
            true_dipole, pred_dipole = _dipoles_for_sample(sample=sample)
        except Exception as exc:
            log(f"[WARNING] Skipping dipole visuals for {base_name}: {exc}", rank=0)
            continue

        # vectors remain in atomic units; plotting scales only their displayed length
        dipole_vectors = np.vstack((true_dipole, pred_dipole)) # (2, 3)
        # first plot preserves relative vector magnitudes
        multiple_dipole_moment_visual(
            positions_angstrom=sample["positions_angstrom"],
            atomic_numbers=sample["atomic_numbers"],
            dipole_vectors_au=dipole_vectors,
            output_path=os.path.join(sample_dir, f"{base_name}_dipole_moments_au"),
            labels=["true dipole (a.u.)", "predicted dipole (a.u.)"],
            colors=["#1f77b4", "#d62728"],
            bonds=bonds,
            normalize_vectors=False,
            title=f"{base_name} dipole moments",
            show_axes=show_axes,
            show_legend=show_legend,
        )
        # second plot compares directions only
        multiple_dipole_moment_visual(
            positions_angstrom=sample["positions_angstrom"],
            atomic_numbers=sample["atomic_numbers"],
            dipole_vectors_au=dipole_vectors,
            output_path=os.path.join(sample_dir, f"{base_name}_normalized_dipole_moments_au"),
            labels=["true dipole direction", "predicted dipole direction"],
            colors=["#1f77b4", "#d62728"],
            bonds=bonds,
            normalize_vectors=True,
            title=f"{base_name} normalized dipole directions",
            show_axes=show_axes,
            show_legend=show_legend,
        )


# ----------------------------------------------------------------------------------------------------
# main driver

def main(
    log_dir: str,
    artifacts_dir: str = None,
    run_sample_visuals: bool = False,
    run_electron_density_visuals: bool = False,
    run_dipole_visuals: bool = False,
    sample_mode: str = "random_molecules",
    num_random_molecules: int = 5,
    num_best_conformers: int = 10,
    num_smallest_molecules: int = 10,
    seed: int = DEFAULT_RANDOM_STATE,
    density_point_filter: str = "top_percentile",
    density_top_percentile: float = 99.5,
    density_absolute_cutoff: float = 1.0e-4,
    density_max_points: int = 5000,
    density_isosurface_percentile: float = 99.0,
    density_isosurface_absolute_level=None,
    show_3d_axes: bool = False,
    show_legend: bool = True,
    matrix_pool_size: int = 1,
    metrics_max_natoms=None,
    metrics_max_density_matrix_dim=None,
    metrics_natoms_bin_size=None,
    metrics_density_matrix_dim_bin_size=None,
    dataset=None,
    inference_runner=None,
):
    '''create all requested figures for one QMugs density-matrix inference run'''

    # ----------------------------------------------------------------------------------------------------
    # local helpers for sample selection and loading

    def _select_sample_records(metrics):
        '''select per-matrix records for sampled visualizations'''

        if len(metrics) == 0:
            return []

        sample_mode_local = str(sample_mode).strip().lower()
        rng = np.random.default_rng(int(seed))

        if sample_mode_local == "random_molecules":
            molecule_ids = sorted({str(record["chembl_id"]) for record in metrics})
            if len(molecule_ids) == 0:
                return []
            sample_size = min(int(num_random_molecules), len(molecule_ids))
            sampled_molecules = set(rng.choice(molecule_ids, size=sample_size, replace=False).tolist())
            # random molecule mode includes every test conformer for each selected molecule
            return [record for record in metrics if str(record["chembl_id"]) in sampled_molecules]

        if sample_mode_local == "best_conformers":
            records_with_rmse = [
                record for record in metrics
                if "rmse" in record and _is_scalar_number(record["rmse"])
            ]
            # best conformer mode uses the saved per-matrix RMSE values
            records_with_rmse = sorted(records_with_rmse, key=lambda record: float(record["rmse"]))
            return records_with_rmse[: int(num_best_conformers)]

        if sample_mode_local == "smallest_molecules":
            molecule_natoms = {}
            for record in metrics:
                if "natoms" not in record or not _is_scalar_number(record["natoms"]):
                    continue
                chembl_id = str(record["chembl_id"])
                natoms = int(record["natoms"])
                if chembl_id not in molecule_natoms or natoms < molecule_natoms[chembl_id]:
                    molecule_natoms[chembl_id] = natoms
            if len(molecule_natoms) == 0:
                return []
            ordered_molecules = sorted(
                molecule_natoms.keys(),
                key=lambda chembl_id: (molecule_natoms[chembl_id], chembl_id),
            )
            selected_molecules = set(ordered_molecules[: int(num_smallest_molecules)])
            # smallest molecule mode includes every test conformer for each selected molecule
            selected_records = [
                record for record in metrics
                if str(record["chembl_id"]) in selected_molecules
            ]
            return sorted(
                selected_records,
                key=lambda record: (
                    molecule_natoms[str(record["chembl_id"])],
                    str(record["chembl_id"]),
                    str(record["conformer_id"]),
                ),
            )

        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    def _prepare_sample_results(selected_records, run_config):
        '''load selected test samples and rerun inference on them'''

        def _data_attr_value(data, name: str):
            '''return one attribute from a torch geometric data object as a Python value'''

            value = getattr(data, name)
            if torch.is_tensor(value):
                if value.numel() == 1:
                    return value.item()
                return value.detach().cpu().numpy()
            return value

        def _find_testset_indices(basedir: str, var_config):
            '''find global pickle indices for selected test-set conformers'''

            from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

            target_keys = {_record_key(record) for record in selected_records}
            found_indices = {}

            testset = SimplePickleDataset(
                basedir=basedir,
                label="testset",
                preload=False,
                var_config=var_config,
            )

            # scan lazy pickle metadata until all selected conformers are matched
            for global_index in range(testset.ntotal):
                if len(found_indices) == len(target_keys):
                    break

                data = testset.read(global_index)
                chembl_id = str(_data_attr_value(data, "chembl_id"))
                conformer_id = str(_data_attr_value(data, "conformer_id")).zfill(2)
                key = (chembl_id, conformer_id)
                if key in target_keys and key not in found_indices:
                    found_indices[key] = global_index

            ordered_indices = []
            ordered_records = []
            for record in selected_records:
                key = _record_key(record)
                if key not in found_indices:
                    log(f"[WARNING] Could not find testset pickle index for {key[0]} conf {key[1]}", rank=0)
                    continue
                ordered_indices.append(found_indices[key])
                ordered_records.append(record)

            return ordered_indices, ordered_records

        inference_config_local = run_config.get("inference_config", {})
        model_dir = inference_config_local.get("model_dir", None)
        checkpoint_path = inference_config_local.get("checkpoint_path", None)
        basedir = inference_config_local.get("basedir", None)
        if model_dir is None or basedir is None:
            log("[WARNING] Saved inference config lacks model_dir or basedir; skipping sampled visuals", rank=0)
            return []

        # inference.py passes its already-loaded wrapper when figs.main() is called in-process
        local_inference_runner = inference_runner
        if local_inference_runner is None:
            from qmugs import QMugsInference
            # standalone plotting redeclares the inference wrapper from saved config
            local_inference_runner = QMugsInference(
                model_dir=model_dir,
                checkpoint_path=checkpoint_path,
            )

        var_config = local_inference_runner.config["NeuralNetwork"]["Variables_of_interest"]
        selected_indices, ordered_records = _find_testset_indices(
            basedir=basedir,
            var_config=var_config,
        )
        if len(selected_indices) == 0:
            log("[WARNING] No selected conformers were found in the testset pickle files", rank=0)
            return []

        from hydragnn.utils.datasets.pickledataset import SimplePickleDataset
        selected_dataset = SimplePickleDataset(
            basedir=basedir,
            label="testset",
            subset=selected_indices,
            preload=False,
            var_config=var_config,
        )

        # small selected dataset is loaded lazily and inferred in batch_size one
        return sample_inference(
            pretrained_model=local_inference_runner,
            dataset=selected_dataset,
            selected_records=ordered_records,
        )

    # ----------------------------------------------------------------------------------------------------
    # logging setup

    # configure logging only for standalone figure generation
    if len(logging.getLogger().handlers) == 0:
        rank = MPI.COMM_WORLD.Get_rank()
        logging.basicConfig(
            level=logging.INFO,
            format=f"%(levelname)s (rank {rank}): %(message)s",
            datefmt="%H:%M:%S"
        )
        log_name = os.path.basename(os.path.normpath(log_dir))
        if log_name == "":
            log_name = "QMugs_figs"
        hydragnn.utils.print.setup_log(log_name)
        writer = hydragnn.utils.model.get_summary_writer(log_name)
        log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)

    if dataset is not None:
        log("[INFO] A dataset was provided to figs.main(); standalone reload will still be used for selected samples", rank=0)

    # ----------------------------------------------------------------------------------------------------
    # directory and JSON setup

    log_dir = os.path.abspath(log_dir)
    if artifacts_dir is not None:
        artifacts_dir = os.path.abspath(artifacts_dir)

    # all figures are stored under the inference run log directory
    base_figure_dir = os.path.join(log_dir, "figs")
    figure_dirs = {
        "base": base_figure_dir,
        "metrics": os.path.join(base_figure_dir, "metrics"),
        "density_matrices": os.path.join(base_figure_dir, "density_matrices"),
        "electron_density": os.path.join(base_figure_dir, "electron_density"),
        "dipole_moments": os.path.join(base_figure_dir, "dipole_moments"),
    }
    for directory in figure_dirs.values():
        os.makedirs(directory, exist_ok=True)

    # per_matrix_metrics.json is the only required input for scalar metric plots
    metrics_path = os.path.join(log_dir, "per_matrix_metrics.json")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"per_matrix_metrics.json not found: {metrics_path}")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    if not isinstance(metrics, list):
        raise ValueError(f"per_matrix_metrics.json must contain a list, got {type(metrics)}")

    config_path = os.path.join(log_dir, "inference_run_config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            run_config = json.load(f)
    else:
        log(f"[WARNING] inference_run_config.json not found in {log_dir}; sampled visuals will be skipped", rank=0)
        run_config = None
    inference_config = {} if run_config is None else run_config.get("inference_config", {})

    # ----------------------------------------------------------------------------------------------------
    # metric visuals

    # metric plots require only per_matrix_metrics.json
    log(f"[INFO] Plotting scalar metrics for {len(metrics)} conformers", rank=0)
    plot_metric_visuals(
        metrics=metrics,
        output_dir=figure_dirs["metrics"],
        metrics_max_natoms=metrics_max_natoms,
        metrics_max_density_matrix_dim=metrics_max_density_matrix_dim,
        metrics_natoms_bin_size=metrics_natoms_bin_size,
        metrics_density_matrix_dim_bin_size=metrics_density_matrix_dim_bin_size,
    )

    # ----------------------------------------------------------------------------------------------------
    # sampled visuals

    # density and dipole visuals require selected conformers and a quick inference rerun
    run_sample_visuals = (
        run_sample_visuals
        or run_electron_density_visuals
        or run_dipole_visuals
    )
    if not run_sample_visuals:
        log("[INFO] Sampled visuals were not requested; metric plots are complete", rank=0)
        return

    if run_config is None:
        log("[WARNING] Skipping sampled visuals because inference_run_config.json is unavailable", rank=0)
        return

    selected_records = _select_sample_records(metrics=metrics)
    if len(selected_records) == 0:
        log("[WARNING] No sampled conformers were selected; sampled visuals are skipped", rank=0)
        return

    # rerun inference only on the selected matrices because streaming inference does not store predictions
    log(f"[INFO] Rerunning inference for {len(selected_records)} sampled conformers", rank=0)
    sample_results = _prepare_sample_results(
        selected_records=selected_records,
        run_config=run_config,
    )
    if len(sample_results) == 0:
        log("[WARNING] Sample inference did not produce any results", rank=0)
        return

    plot_density_matrix_visuals(
        sample_results=sample_results,
        output_dir=figure_dirs["density_matrices"],
        matrix_pool_size=matrix_pool_size,
    )

    qmugs_data_dir = inference_config.get("qmugs_data_dir", None)
    if run_electron_density_visuals:
        # electron-density plots may reuse cubeprops or recompute true/predicted densities
        plot_electron_density_visuals(
            sample_results=sample_results,
            output_dir=figure_dirs["electron_density"],
            artifacts_dir=artifacts_dir,
            inference_config=inference_config,
            qmugs_data_dir=qmugs_data_dir,
            density_point_filter=density_point_filter,
            density_top_percentile=density_top_percentile,
            density_absolute_cutoff=density_absolute_cutoff,
            density_max_points=density_max_points,
            density_isosurface_percentile=density_isosurface_percentile,
            density_isosurface_absolute_level=density_isosurface_absolute_level,
            show_axes=show_3d_axes,
            show_legend=show_legend,
        )

    if run_dipole_visuals:
        # dipole plots use saved values when present and otherwise recompute on the sample
        plot_dipole_visuals(
            sample_results=sample_results,
            output_dir=figure_dirs["dipole_moments"],
            qmugs_data_dir=qmugs_data_dir,
            show_axes=show_3d_axes,
            show_legend=show_legend,
        )


# ----------------------------------------------------------------------------------------------------
# command line interface

if __name__ == "__main__":

    # ----------------------------------------------------------------------------------------------------
    # args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="inference run log directory containing per_matrix_metrics.json",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default=None,
        help="optional artifacts directory containing cubeprop outputs",
    )
    parser.add_argument(
        "--run_sample_visuals",
        action="store_true",
        help="rerun inference on selected samples and plot density matrices",
    )
    parser.add_argument(
        "--run_electron_density_visuals",
        action="store_true",
        help="plot electron density point clouds for selected samples",
    )
    parser.add_argument(
        "--run_dipole_visuals",
        action="store_true",
        help="plot dipole moments for selected samples",
    )

    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--random_molecules",
        action="store_true",
        help="sample random molecules and plot all selected test conformers",
    )
    sample_group.add_argument(
        "--best_conformers",
        action="store_true",
        help="plot conformers with the lowest per-matrix RMSE",
    )
    sample_group.add_argument(
        "--smallest_molecules",
        action="store_true",
        help="sample molecules with the fewest atoms and plot all selected test conformers",
    )

    parser.add_argument(
        "--num_random_molecules",
        type=int,
        default=5,
        help="number of random molecules to sample",
    )
    parser.add_argument(
        "--num_best_conformers",
        type=int,
        default=10,
        help="number of lowest-RMSE conformers to sample",
    )
    parser.add_argument(
        "--num_smallest_molecules",
        type=int,
        default=10,
        help="number of smallest molecules by atom count to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="random seed for molecule sampling",
    )
    parser.add_argument(
        "--density_point_filter",
        choices=["top_percentile", "absolute_cutoff", "max_points"],
        default="top_percentile",
        help="method used to select density grid points for point-cloud plots",
    )
    parser.add_argument(
        "--density_top_percentile",
        type=float,
        default=99.0,
        help="percentile cutoff used when density_point_filter=top_percentile",
    )
    parser.add_argument(
        "--density_absolute_cutoff",
        type=float,
        default=1.0e-4,
        help="absolute density cutoff used when density_point_filter=absolute_cutoff",
    )
    parser.add_argument(
        "--density_max_points",
        type=int,
        default=5000,
        help="maximum number of density grid points shown in one point cloud",
    )
    parser.add_argument(
        "--density_isosurface_percentile",
        type=float,
        default=99.0,
        help="percentile level used for density isosurface plots",
    )
    parser.add_argument(
        "--density_isosurface_absolute_level",
        type=float,
        default=None,
        help="absolute density level for isosurface plots; overrides percentile when provided",
    )
    parser.add_argument(
        "--show_3d_axes",
        action="store_true",
        help="show 3D axis grid and labels behind molecule visualizations",
    )
    parser.add_argument(
        "--hide_legend",
        action="store_true",
        help="hide legends on molecule, density, and dipole visualizations",
    )
    parser.add_argument(
        "--matrix_pool_size",
        type=int,
        default=1,
        help="average-pooling block size for sampled density matrix heatmaps",
    )
    parser.add_argument(
        "--metrics_max_natoms",
        type=float,
        default=None,
        help="maximum atom count included in metric-by-size plots",
    )
    parser.add_argument(
        "--metrics_max_density_matrix_dim",
        type=float,
        default=None,
        help="maximum density matrix dimension included in metric-by-size plots",
    )
    parser.add_argument(
        "--metrics_natoms_bin_size",
        type=float,
        default=None,
        help="atom-count bin size for metric-by-size plots",
    )
    parser.add_argument(
        "--metrics_density_matrix_dim_bin_size",
        type=float,
        default=None,
        help="density-matrix-dimension bin size for metric-by-size plots",
    )
    args = parser.parse_args()

    if args.smallest_molecules:
        sample_mode = "smallest_molecules"
    elif args.best_conformers:
        sample_mode = "best_conformers"
    else:
        sample_mode = "random_molecules"

    main(
        log_dir=args.log_dir,
        artifacts_dir=args.artifacts_dir,
        run_sample_visuals=args.run_sample_visuals,
        run_electron_density_visuals=args.run_electron_density_visuals,
        run_dipole_visuals=args.run_dipole_visuals,
        sample_mode=sample_mode,
        num_random_molecules=args.num_random_molecules,
        num_best_conformers=args.num_best_conformers,
        num_smallest_molecules=args.num_smallest_molecules,
        seed=args.seed,
        density_point_filter=args.density_point_filter,
        density_top_percentile=args.density_top_percentile,
        density_absolute_cutoff=args.density_absolute_cutoff,
        density_max_points=args.density_max_points,
        density_isosurface_percentile=args.density_isosurface_percentile,
        density_isosurface_absolute_level=args.density_isosurface_absolute_level,
        show_3d_axes=args.show_3d_axes,
        show_legend=not args.hide_legend,
        matrix_pool_size=args.matrix_pool_size,
        metrics_max_natoms=args.metrics_max_natoms,
        metrics_max_density_matrix_dim=args.metrics_max_density_matrix_dim,
        metrics_natoms_bin_size=args.metrics_natoms_bin_size,
        metrics_density_matrix_dim_bin_size=args.metrics_density_matrix_dim_bin_size,
    )
