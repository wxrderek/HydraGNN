###########################################################################################
# Add the overlap matrix and the number of electrons to an existing QMugs pickle dataset
#
# Every conformer pickle of the source dataset is read, its psi4 wavefunction is opened from
# the stored wfn_path, and two fields are attached:
#   overlap_upper_triangle: padded row-major upper triangle of the overlap matrix, float32,
#       same layout and padding as promolecular_upper_triangle
#   n_electrons: sum of atomic numbers minus the charge, int32, checked against the wavefunction
# The object is written to the same relative path under the destination, and the three
# *-meta.pkl files are copied unchanged, so the splits and the sample order are identical.
#
# For every conformer the script also evaluates the new density matrix metrics on the true
# density matrix, once from the wavefunction in float64 and once from the stored float32
# fields, and writes them to ground_truth_checks.csv and summary.json in --log_dir.
# With --check_only nothing is written to a destination.
#
# Run from examples/qmugs as: python -m utils.add_overlap --source ... --destination ... --log_dir ...
###########################################################################################

import os, json
import logging
import argparse
import pickle
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import torch
from mpi4py import MPI

import psi4

from utils import (
    overlap_matrix,
    number_of_electrons,
    upper_triangle_vector,
    triangular_number,
    triangular_side,
    symmetric_matrix_from_upper_triangle_vector,
)

SPLITS = ['trainset', 'valset', 'testset']


def unfold_stored_matrix(vector: np.ndarray, n_pad: int, dim: int):
    '''full symmetric dim by dim matrix from a stored padded upper triangle or padded square'''
    if vector.size == n_pad * n_pad:
        return vector.reshape(n_pad, n_pad)[:dim, :dim]
    # row-major upper triangle of the padded matrix, whose physical block is the leading one
    full = symmetric_matrix_from_upper_triangle_vector(vector, n_pad) # (n_pad, n_pad)
    return full[:dim, :dim]


def ground_truth_metrics(density, overlap, promolecular, prefix: str, n_electrons: int):
    '''electron count, idempotency defect, natural occupations and promolecular overlap-weighted error of a true density matrix'''
    density = np.asarray(density, dtype=np.float64) # (n_basis, n_basis)
    overlap = np.asarray(overlap, dtype=np.float64) # (n_basis, n_basis)
    assert density.shape == overlap.shape

    # square root of the overlap matrix from its eigendecomposition
    overlap_eigenvalues, overlap_eigenvectors = np.linalg.eigh(overlap)
    overlap_sqrt = (overlap_eigenvectors * np.sqrt(overlap_eigenvalues)) @ overlap_eigenvectors.T

    # natural occupations are the eigenvalues of the overlap-symmetrized density matrix
    occupations = np.linalg.eigvalsh(overlap_sqrt @ density @ overlap_sqrt)

    density_overlap = density @ overlap
    idempotency_defect = np.linalg.norm(density_overlap @ density - 2.0 * density) / np.linalg.norm(density)

    # overlap-weighted relative error of the promolecular baseline against the true density matrix
    residual_overlap = (np.asarray(promolecular, dtype=np.float64) - density) @ overlap
    promolecular_s_metric = np.sqrt(
        np.trace(residual_overlap @ residual_overlap) / np.trace(density_overlap @ density_overlap)
    )

    return {
        f'{prefix}_electron_count_error': float(np.sum(overlap * density)) - n_electrons,
        f'{prefix}_idempotency_relative_defect': float(idempotency_defect),
        f'{prefix}_occupation_min': float(occupations.min()),
        f'{prefix}_occupation_max': float(occupations.max()),
        f'{prefix}_fractional_occupations': int(np.sum((occupations > 0.05) & (occupations < 1.95))),
        f'{prefix}_occupations_above_2': int(np.sum(occupations > 2.0 + 1e-6)),
        f'{prefix}_promolecular_s_metric_relative_error': float(promolecular_s_metric),
        f'{prefix}_overlap_min_eigenvalue': float(overlap_eigenvalues.min()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='existing pickle dataset directory')
    parser.add_argument('--destination', default=None, help='new pickle dataset directory, must not exist or be empty')
    parser.add_argument('--log_dir', required=True, help='directory for summary.json and ground_truth_checks.csv')
    parser.add_argument('--check_only', action='store_true', help='run the ground-truth checks and write no dataset')
    parser.add_argument('--max_samples', type=int, default=None, help='process at most this many conformers per split')
    parser.add_argument(
        '--electron_count_tolerance',
        type=float,
        default=None,
        help='assert that the electron count of the wavefunction density matrix is within this tolerance; no assert when unset',
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s rank {rank}: %(message)s',
    )

    # one thread per rank, since the ranks already fill the node
    psi4.set_num_threads(1)
    psi4.core.be_quiet()

    source = os.path.abspath(args.source)
    destination = os.path.abspath(args.destination) if args.destination is not None else None
    if not args.check_only:
        if destination is None:
            raise ValueError('--destination is required unless --check_only is set')
        if destination == source:
            raise ValueError('--destination must differ from --source')

    # the destination must be new, so no existing file can be overwritten
    destination_in_use = False
    if rank == 0 and not args.check_only:
        destination_in_use = os.path.isdir(destination) and len(os.listdir(destination)) > 0
    if comm.bcast(destination_in_use, root=0):
        raise FileExistsError(f'destination {destination} exists and is not empty')

    if rank == 0:
        if not args.check_only:
            os.makedirs(destination, exist_ok=True)
            for label in SPLITS:
                shutil.copy2(
                    os.path.join(source, f'{label}-meta.pkl'),
                    os.path.join(destination, f'{label}-meta.pkl'),
                )
        os.makedirs(args.log_dir, exist_ok=True)
        logging.info(f'source {source}, destination {destination}, check_only {args.check_only}, ranks {world_size}')
    comm.Barrier()

    local_rows = []
    for label in SPLITS:
        with open(os.path.join(source, f'{label}-meta.pkl'), 'rb') as f:
            _ = pickle.load(f) # minmax_node_feature
            _ = pickle.load(f) # minmax_graph_feature
            ntotal = pickle.load(f)
            use_subdir = pickle.load(f)
            nmax_persubdir = pickle.load(f)

        n_process = ntotal if args.max_samples is None else min(ntotal, args.max_samples)
        local_indices = range(rank, n_process, world_size)
        logging.info(f'split {label}: {ntotal} conformers, processing {n_process}, {len(local_indices)} on this rank')

        for count, global_i in enumerate(local_indices):
            # same path layout as QMugsDataset.load_and_split_dataset_pickle_streaming
            relative_path = f'{label}-{global_i}.pkl'
            if use_subdir:
                relative_path = os.path.join(str(global_i // nmax_persubdir), relative_path)
            with open(os.path.join(source, relative_path), 'rb') as f:
                data_object = pickle.load(f)

            # the wavefunction must belong to this conformer
            wfn_path = str(data_object.wfn_path)
            chembl_id = wfn_path.split('/')[-2]
            conformer_id = wfn_path.split('/')[-1].split('.')[0][-2:]
            if chembl_id != data_object.chembl_id or conformer_id != data_object.conformer_id:
                raise ValueError(
                    f'{relative_path}: wfn_path {wfn_path} does not match '
                    f'{data_object.chembl_id} conformer {data_object.conformer_id}'
                )
            if hasattr(data_object, 'overlap_upper_triangle') or hasattr(data_object, 'n_electrons'):
                raise ValueError(f'{relative_path} already stores an overlap matrix')

            psi4_wfn = psi4.core.Wavefunction.from_file(wfn_path)
            dim = int(data_object.density_matrix_dim)
            if int(psi4_wfn.basisset().nbf()) != dim:
                raise ValueError(f'{relative_path}: wavefunction has {psi4_wfn.basisset().nbf()} basis functions, data has {dim}')

            # padding width read from the stored promolecular triangle, never assumed
            promolecular_vector = data_object.promolecular_upper_triangle.numpy() # (T(n_pad),)
            n_pad = triangular_side(promolecular_vector.size)
            y_vector = data_object.y.reshape(-1).numpy() # (T(n_pad),) or (n_pad * n_pad,)
            if y_vector.size not in (triangular_number(n_pad), n_pad * n_pad):
                raise ValueError(f'{relative_path}: target length {y_vector.size} does not match padding {n_pad}')

            # overlap matrix stored exactly as QMugsDataset._mol_to_graph stores it
            overlap = overlap_matrix(psi4_wfn) # (dim, dim)
            overlap_vector = upper_triangle_vector(
                np.asarray(overlap, dtype=np.float32),
                n_pad,
                triu_indices=np.triu_indices(dim),
            ) # (T(n_pad),)
            n_electrons = number_of_electrons(psi4_wfn)

            # true density matrix from the wavefunction, float64
            density_alpha = np.asarray(psi4_wfn.Da().np, dtype=np.float64) # (dim, dim)
            density_beta = np.asarray(psi4_wfn.Db().np, dtype=np.float64) # (dim, dim)
            density_wfn = density_alpha + density_beta

            # true density matrix as the decoder sees it, the stored delta plus the stored baseline in float32
            promolecular_stored = unfold_stored_matrix(promolecular_vector, n_pad, dim) # (dim, dim)
            density_stored = unfold_stored_matrix(y_vector, n_pad, dim) + promolecular_stored # (dim, dim)
            overlap_stored = unfold_stored_matrix(overlap_vector, n_pad, dim) # (dim, dim)

            row = {
                'split': label,
                'global_index': global_i,
                'chembl_id': data_object.chembl_id,
                'conformer_id': data_object.conformer_id,
                'density_matrix_dim': dim,
                'n_electrons': n_electrons,
                'charge': int(data_object.charge),
                'multiplicity': int(data_object.multiplicity),
                'overlap_max_diagonal_deviation': float(np.max(np.abs(np.diag(overlap) - 1.0))),
                'stored_density_max_abs_difference': float(np.max(np.abs(density_stored - density_wfn))),
            }
            row.update(ground_truth_metrics(density_wfn, overlap, promolecular_stored, 'wfn', n_electrons))
            row.update(ground_truth_metrics(density_stored, overlap_stored, promolecular_stored, 'stored', n_electrons))

            # spin-resolved idempotency defects, which hold for open-shell conformers where the total one does not
            for spin, density_spin in (('alpha', density_alpha), ('beta', density_beta)):
                norm = np.linalg.norm(density_spin)
                defect = np.linalg.norm(density_spin @ overlap @ density_spin - density_spin) / norm if norm > 0 else 0.0
                row[f'wfn_{spin}_idempotency_relative_defect'] = float(defect)
            local_rows.append(row)

            if args.electron_count_tolerance is not None:
                if abs(row['wfn_electron_count_error']) > args.electron_count_tolerance:
                    raise ValueError(
                        f"{relative_path}: electron count error {row['wfn_electron_count_error']:.3e} "
                        f"exceeds {args.electron_count_tolerance:.1e}"
                    )

            if not args.check_only:
                data_object.overlap_upper_triangle = torch.from_numpy(overlap_vector).to(torch.float32) # (T(n_pad),)
                data_object.n_electrons = torch.tensor(n_electrons, dtype=torch.int32)
                output_path = os.path.join(destination, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                if os.path.exists(output_path):
                    raise FileExistsError(output_path)
                with open(output_path, 'wb') as f:
                    pickle.dump(data_object, f)

            if (count + 1) % 100 == 0:
                logging.info(f'split {label}: {count + 1}/{len(local_indices)} conformers done')

    # gather all rows on rank 0 and summarize per split
    all_rows = comm.gather(local_rows, root=0)
    if rank == 0:
        table = pd.DataFrame([row for rows in all_rows for row in rows])
        table = table.sort_values(['split', 'global_index']).reset_index(drop=True)
        table.to_csv(os.path.join(args.log_dir, 'ground_truth_checks.csv'), index=False)

        summary = {
            'source': source,
            'destination': destination,
            'check_only': args.check_only,
            'max_samples': args.max_samples,
            'electron_count_tolerance': args.electron_count_tolerance,
            'world_size': world_size,
            'splits': {},
        }
        metric_columns = [
            column for column in table.columns
            if column not in ('split', 'global_index', 'chembl_id', 'conformer_id', 'charge', 'multiplicity')
        ]
        for label, split_table in table.groupby('split'):
            summary['splits'][label] = {
                'count': int(len(split_table)),
                'multiplicity_histogram': {str(k): int(v) for k, v in Counter(split_table['multiplicity']).items()},
                'charge_histogram': {str(k): int(v) for k, v in Counter(split_table['charge']).items()},
                'metrics': {
                    column: {
                        'mean': float(split_table[column].mean()),
                        'min': float(split_table[column].min()),
                        'max': float(split_table[column].max()),
                        'max_abs': float(split_table[column].abs().max()),
                    }
                    for column in metric_columns
                },
            }
        with open(os.path.join(args.log_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        logging.info(f"processed {len(table)} conformers, summary in {os.path.join(args.log_dir, 'summary.json')}")


if __name__ == '__main__':
    main()
