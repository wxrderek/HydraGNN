'''
Verification for the sequential Schur decoder in hydragnn/decoders/schur/.

Everything here runs in float64 on CPU, because the point is to separate real errors
from round-off. The checks are, in order:

  1. PSD and symmetry of the free-running reconstruction, for random latents
  2. oracle round trip: exact latents taken from an independent unpivoted LDL^T must drive
     the recursion back to the matrix they came from
  3. teacher forcing with oracle latents, and PSD of every individual step block
  4. detach_every changes gradients, not values
  5. the per-step loss collapses to the plain full-matrix loss at the matching weights
  6. gradcheck through the recursion in both routings
  7. delta learning: Equation 17 agrees entrywise, and the two targets agree under MAE
  8. conditioning_dim() agrees with conditioning() for every head and width option
  9. end to end through SchurDecoderModel on a two-molecule batch of different sizes

Run it from the qmugs example directory, or through
scripts/dcc/qmugs_schur_decoder_test_dcc.sh.
'''

import os
import sys
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

QMUGS_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
if QMUGS_EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, QMUGS_EXAMPLE_DIR)

from hydragnn.decoders.schur.heads import (
    AlphaHead,
    D1Head,
    GlobalEmbedding,
    OrbitalEmbedding,
    ZHead,
)
from hydragnn.decoders.schur.losses import reconstruction_loss, step_weights
from hydragnn.decoders.schur.model import SchurDecoderModel, dense_from_padded_target
from hydragnn.decoders.schur.recursion import bordering_recursion

# the oracle latents come from the diagnostic script, which computes them by an independent
# route (a jittered Cholesky), so agreement is a genuine cross-check rather than a tautology
from schur_decomp_test import unpivoted_ldl, z_rows_from_unit_lower

WORKING_DTYPE = torch.float64


# ----------------------------------------------------------------------------------------------------
# helpers


def random_psd_matrix(n_basis: int, rank: int, generator):
    '''a symmetric PSD matrix of the given rank, standing in for a density matrix'''

    # QMugs density matrices are about 80% rank deficient, so the synthetic ones are too
    factor = torch.randn((n_basis, rank), dtype=WORKING_DTYPE, generator=generator)
    matrix = factor @ factor.transpose(0, 1) # (n_basis, n_basis)
    return 0.5 * (matrix + matrix.transpose(0, 1))


def random_latents(n_basis: int, generator):
    '''random (z, delta) of the shapes the recursion expects'''

    pairs = torch.tril(
        torch.randn((n_basis, n_basis), dtype=WORKING_DTYPE, generator=generator), diagonal=-1
    )
    delta = F.softplus(torch.randn(n_basis, dtype=WORKING_DTYPE, generator=generator))
    return [pairs[k, :k] for k in range(n_basis)], delta, pairs


def oracle_latents(matrix, jitter_relative: float = 1e-9):
    '''the exact (z_k, delta_k) that reproduce a given PSD matrix'''

    # unpivoted_ldl factors matrix + jitter*I, so the reconstruction below should land at the
    # jitter level rather than at machine precision
    jitter = jitter_relative * abs(float(torch.diagonal(matrix).mean()))
    unit_lower, pivots = unpivoted_ldl(matrix, jitter=jitter)
    pairs = z_rows_from_unit_lower(unit_lower) # (n, n), row k is z_k
    return [pairs[k, :k] for k in range(matrix.shape[0])], pivots, pairs, jitter


def report(name: str, passed: bool, detail: str):
    logging.info('[%s] %s -- %s', 'PASS' if passed else 'FAIL', name, detail)
    return passed


# ----------------------------------------------------------------------------------------------------
# the checks


def check_psd_and_symmetry(generator):
    '''1. free running is PSD and exactly symmetric for arbitrary latents'''

    worst_eigenvalue = 0.0
    worst_asymmetry = 0.0
    for n_basis in (8, 33, 97, 200):
        z, delta, _ = random_latents(n_basis, generator)
        D_hat, _, _ = bordering_recursion(z, delta)

        eigenvalues = torch.linalg.eigvalsh(D_hat)
        relative = float(eigenvalues.min() / eigenvalues.max())
        worst_eigenvalue = min(worst_eigenvalue, relative)
        worst_asymmetry = max(
            worst_asymmetry, float((D_hat - D_hat.transpose(0, 1)).abs().max())
        )

    passed = worst_eigenvalue > -1e-12 and worst_asymmetry == 0.0
    return report(
        'psd_and_symmetry',
        passed,
        f'min lambda / lambda_max = {worst_eigenvalue:.3e}, '
        f'max |D_hat - D_hat^T| = {worst_asymmetry:.3e}',
    )


def check_oracle_round_trip(matrix, label: str):
    '''2. exact latents drive the recursion back to the matrix they came from'''

    z, delta, _, jitter = oracle_latents(matrix)
    D_hat, _, _ = bordering_recursion(z, delta)

    error = float(torch.linalg.norm(D_hat - matrix) / torch.linalg.norm(matrix))
    # the factorization is of matrix + jitter*I, so this is the size to beat
    expected = jitter * float(np.sqrt(matrix.shape[0])) / float(torch.linalg.norm(matrix))
    tolerance = max(10.0 * expected, 1e-9)

    return report(
        f'oracle_round_trip[{label}]',
        error < tolerance,
        f'relative error {error:.3e} against tolerance {tolerance:.3e} '
        f'(jitter contributes {expected:.3e})',
    )


def check_teacher_forcing(matrix, label: str):
    '''3. teacher forcing with exact latents reproduces the matrix, step blocks are PSD'''

    n_basis = int(matrix.shape[0])
    z, delta, _, jitter = oracle_latents(matrix)
    D_hat, columns, diagonals = bordering_recursion(z, delta, density_matrix=matrix)

    error = float(torch.linalg.norm(D_hat - matrix) / torch.linalg.norm(matrix))
    expected = jitter * float(np.sqrt(n_basis)) / float(torch.linalg.norm(matrix))
    tolerance = max(10.0 * expected, 1e-9)

    # Theorem 2.1 applies to every individual teacher-forced block, even though it does not
    # apply to the assembled matrix, whose input side is ground truth
    worst_block = 0.0
    for k in range(1, n_basis, max(1, n_basis // 16)):
        block = torch.zeros((k + 1, k + 1), dtype=WORKING_DTYPE)
        block[:k, :k] = matrix[:k, :k]
        block[:k, k] = columns[k]
        block[k, :k] = columns[k]
        block[k, k] = diagonals[k]
        eigenvalues = torch.linalg.eigvalsh(block)
        worst_block = min(worst_block, float(eigenvalues.min() / eigenvalues.max()))

    passed = error < tolerance and worst_block > -1e-10
    return report(
        f'teacher_forcing[{label}]',
        passed,
        f'relative error {error:.3e} against tolerance {tolerance:.3e}, '
        f'worst step block min lambda / lambda_max = {worst_block:.3e}',
    )


def check_detach_is_value_preserving(generator):
    '''4. detach_every changes gradients, not values'''

    n_basis = 64
    z, delta, _ = random_latents(n_basis, generator)

    joint, _, _ = bordering_recursion(z, delta)
    chunked, _, _ = bordering_recursion(z, delta, detach_every=8)

    difference = float((joint - chunked).abs().max())
    return report(
        'detach_is_value_preserving',
        difference == 0.0,
        f'max |D_hat(joint) - D_hat(detach_every=8)| = {difference:.3e}',
    )


def check_loss_identity(generator):
    '''5. the per-step loss collapses to the plain full-matrix loss at matching weights'''

    # with w_k proportional to (2k + 1) the entry-count normalization cancels and every entry
    # of the residual is counted exactly once, which is the plain mean absolute error
    n_basis = 57
    matrix = random_psd_matrix(n_basis, rank=12, generator=generator)
    z, delta, _ = random_latents(n_basis, generator)
    D_hat, _, _ = bordering_recursion(z, delta)

    # the 'per_entry' schedule is exactly this weighting, and is what makes the decoder's
    # training curve comparable with a dense head trained on the same target and loss
    weights = step_weights(
        n_basis, 'per_entry', 0, n_basis, D_hat.device, D_hat.dtype
    )
    counts = 2.0 * torch.arange(n_basis, dtype=WORKING_DTYPE) + 1.0
    expected = counts / counts.sum()
    schedule_error = float((weights - expected).abs().max())

    per_step = float(
        reconstruction_loss(
            D_hat, matrix, weights, 'mae', 'entry_count', 'total', promolecular=None
        )
    )
    plain = float(F.l1_loss(D_hat, matrix))

    difference = abs(per_step - plain) / max(plain, 1e-30)
    return report(
        'loss_identity',
        difference < 1e-12 and schedule_error < 1e-15,
        f"w_schedule 'per_entry' matches (2k+1)/sum to {schedule_error:.3e}; "
        f'per-step {per_step:.12e} against full-matrix {plain:.12e}, '
        f'relative difference {difference:.3e}',
    )


def check_gradients(generator):
    '''6. gradcheck through the recursion, both routings'''

    n_basis = 6
    matrix = random_psd_matrix(n_basis, rank=3, generator=generator)

    results = []
    for teacher_forcing in (False, True):

        def recursion(raw_pairs, raw_alpha):
            z_all = torch.tril(raw_pairs, diagonal=-1)
            delta = F.softplus(raw_alpha) # keeps the nonnegativity assert satisfied
            D_hat, _, _ = bordering_recursion(
                [z_all[k, :k] for k in range(n_basis)],
                delta,
                density_matrix=matrix if teacher_forcing else None,
            )
            return D_hat

        raw_pairs = torch.randn(
            (n_basis, n_basis), dtype=WORKING_DTYPE, generator=generator
        ).requires_grad_(True)
        raw_alpha = torch.randn(
            n_basis, dtype=WORKING_DTYPE, generator=generator
        ).requires_grad_(True)

        label = 'teacher_forcing' if teacher_forcing else 'free_running'
        try:
            torch.autograd.gradcheck(recursion, (raw_pairs, raw_alpha), eps=1e-6, atol=1e-7)
            results.append(report(f'gradcheck[{label}]', True, 'analytic and numeric agree'))
        except Exception as failure:  # gradcheck raises on mismatch
            results.append(report(f'gradcheck[{label}]', False, str(failure)[:200]))

    return all(results)


def check_delta_learning(generator):
    '''7. Equation 17 agrees entrywise, and both targets agree under MAE'''

    n_basis = 48
    density_matrix = random_psd_matrix(n_basis, rank=10, generator=generator)
    promolecular = random_psd_matrix(n_basis, rank=8, generator=generator)

    z, delta, _ = random_latents(n_basis, generator)
    D_hat, columns, diagonals = bordering_recursion(z, delta, density_matrix=density_matrix)

    # Equation 17 assembled step by step, against the entrywise subtraction the loss performs
    delta_D_hat = torch.zeros((n_basis, n_basis), dtype=WORKING_DTYPE)
    delta_D_hat[0, 0] = diagonals[0] - promolecular[0, 0]
    for k in range(1, n_basis):
        delta_c_k = columns[k] - promolecular[:k, k] # Delta c_k
        delta_d_k = diagonals[k] - promolecular[k, k] # Delta d_k
        delta_D_hat[:k, k] = delta_c_k
        delta_D_hat[k, :k] = delta_c_k
        delta_D_hat[k, k] = delta_d_k

    assembly_error = float((delta_D_hat - (D_hat - promolecular)).abs().max())

    weights = step_weights(n_basis, 'uniform', 0, n_basis, D_hat.device, D_hat.dtype)
    as_total = float(
        reconstruction_loss(
            D_hat, density_matrix, weights, 'mae', 'entry_count', 'total', promolecular=None
        )
    )
    as_delta = float(
        reconstruction_loss(
            D_hat, density_matrix, weights, 'mae', 'entry_count', 'delta',
            promolecular=promolecular,
        )
    )
    difference = abs(as_total - as_delta) / max(as_total, 1e-30)

    passed = assembly_error < 1e-12 and difference < 1e-12
    return report(
        'delta_learning',
        passed,
        f'max |Delta D_hat - (D_hat - D_pro)| = {assembly_error:.3e}; '
        f'loss total {as_total:.12e} against delta {as_delta:.12e}, '
        f'relative difference {difference:.3e}',
    )


def check_conditioning_widths():
    '''8. conditioning_dim() agrees with conditioning() for every head and width option'''

    torch.set_default_dtype(WORKING_DTYPE)
    node_dim, pooled_dim, n_basis = 16, 24, 11
    n_l, n_m, n_shell, n_species = 3, 5, 4, 6
    activation = torch.nn.SELU()

    results = []
    for use_sequence_position in (False, True):
        for condition_z_on_graph in (False, True):
            orbital_embedding = OrbitalEmbedding(
                node_dim, n_l, n_m, n_shell, n_species, 8, [12], activation,
                'concat', use_sequence_position,
            )
            global_embedding = GlobalEmbedding(pooled_dim, 5, [7], activation, 'concat')
            z_head = ZHead(8, 5, 4, 10.0, [9], activation, 'concat',
                           condition_z_on_graph, 6, False)
            alpha_head = AlphaHead(8, 5, [9], activation, 'concat', 'softplus', -2.0)
            d1_head = D1Head(8, 5, [9], activation, 'concat', 'softplus', -2.0)

            node_embedding = torch.randn(n_basis, node_dim)
            labels = [
                torch.randint(0, n_l, (n_basis,)),
                torch.randint(0, n_m, (n_basis,)),
                torch.randint(0, n_shell, (n_basis,)),
                torch.randint(0, n_species, (n_basis,)),
            ]
            orbital = orbital_embedding(node_embedding, *labels) # (n_basis, 8)
            g = global_embedding(torch.randn(pooled_dim), 12.0, float(n_basis), 0.0, 1.0)

            widths = [
                (
                    'orbital',
                    orbital_embedding.conditioning_dim(),
                    orbital_embedding.conditioning(node_embedding, *labels).shape[-1],
                ),
                (
                    'global',
                    global_embedding.conditioning_dim(),
                    global_embedding.conditioning(
                        torch.randn(pooled_dim), 12.0, float(n_basis), 0.0, 1.0
                    ).shape[-1],
                ),
                (
                    'z',
                    z_head.conditioning_dim(),
                    z_head.conditioning(
                        orbital, orbital, torch.rand(n_basis, n_basis), g
                    ).shape[-1],
                ),
                (
                    'alpha',
                    alpha_head.conditioning_dim(),
                    alpha_head.conditioning(orbital, g).shape[-1],
                ),
                (
                    'd1',
                    d1_head.conditioning_dim(),
                    d1_head.conditioning(orbital[0], g).shape[-1],
                ),
            ]
            for name, declared, actual in widths:
                results.append(
                    report(
                        f'conditioning[{name}, seq={use_sequence_position}, '
                        f'graph={condition_z_on_graph}]',
                        declared == actual,
                        f'declared {declared}, assembled {actual}',
                    )
                )

    return all(results)


# ----------------------------------------------------------------------------------------------------
# end to end


class StubBackbone(torch.nn.Module):
    '''the smallest object exposing what SchurDecoderModel needs from a HydraGNN backbone'''

    def __init__(self, hidden_dim: int, pooled_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pooled_dim = pooled_dim
        self.activation_function = torch.nn.SELU()
        self.loss_function_type = 'mae'
        self.num_heads = 1
        # the wrapper freezes these, so they have to exist and hold parameters
        self.heads_NN = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
        self.graph_shared = torch.nn.ModuleDict({'branch-0': torch.nn.Linear(1, 1)})
        self.node_projection = torch.nn.Linear(4, hidden_dim)
        self.graph_projection = torch.nn.Linear(hidden_dim, pooled_dim)

    def encode(self, data):
        features = torch.cat([data.atomic_numbers, data.pos], dim=1) # (num_nodes, 4)
        v = self.node_projection(features) # (num_nodes, hidden_dim)
        num_graphs = int(data.batch.max()) + 1
        pooled = torch.zeros((num_graphs, self.hidden_dim), dtype=v.dtype)
        pooled = pooled.index_add(0, data.batch, v) # (num_graphs, hidden_dim)
        return v, self.graph_projection(pooled), None, None


def build_molecule(n_atoms: int, orbitals_per_atom: int, n_padded: int, generator):
    '''one synthetic conformer carrying every field the decoder reads'''

    n_basis = n_atoms * orbitals_per_atom
    matrix = random_psd_matrix(n_basis, rank=max(2, n_basis // 5), generator=generator)

    # store the row-major upper triangle padded to n_padded, exactly as qmugs.py does, so that
    # molecules of different sizes still produce equal-length targets to batch
    rows, cols = torch.triu_indices(n_basis, n_basis)
    upper = torch.zeros(n_padded * (n_padded + 1) // 2, dtype=WORKING_DTYPE)
    upper[rows * n_padded - (rows * (rows - 1)) // 2 + (cols - rows)] = matrix[rows, cols]

    orbital_to_atom = torch.arange(n_basis, dtype=torch.int32) // orbitals_per_atom
    return matrix, Data(
        x=torch.randn((n_atoms, 4), dtype=WORKING_DTYPE, generator=generator),
        pos=torch.randn((n_atoms, 3), dtype=WORKING_DTYPE, generator=generator),
        atomic_numbers=torch.full((n_atoms, 1), 6.0, dtype=WORKING_DTYPE),
        y=upper,
        density_matrix_dim=torch.tensor(n_basis, dtype=torch.int32),
        natoms_int=torch.tensor(n_atoms, dtype=torch.int32),
        charge=torch.tensor(0, dtype=torch.int32),
        multiplicity=torch.tensor(1, dtype=torch.int32),
        orbital_to_atom=orbital_to_atom,
        orbital_angular_momentum=torch.arange(n_basis, dtype=torch.int32) % 3,
        orbital_within_shell=torch.arange(n_basis, dtype=torch.int32) % 5,
        orbital_shell=torch.arange(n_basis, dtype=torch.int32) % 2,
    )


def check_end_to_end(generator):
    '''9. a two-molecule batch of different sizes, through the wrapper and its loss'''

    torch.set_default_dtype(WORKING_DTYPE)
    n_padded = 20 # the dataset-wide padding width, as max_padded_density_matrix_dimension is
    first_matrix, first = build_molecule(4, 5, n_padded, generator) # n_basis 20
    second_matrix, second = build_molecule(3, 4, n_padded, generator) # n_basis 12
    batch = Batch.from_data_list([first, second])

    results = []

    # the trap this check exists for: Data.__inc__ offsets any attribute whose key contains
    # "index", so a field named e.g. orbital_index would silently gain num_nodes here
    expected = torch.cat([first.orbital_to_atom, second.orbital_to_atom])
    results.append(
        report(
            'batching_offsets',
            bool(torch.equal(batch.orbital_to_atom, expected)),
            'orbital_to_atom survives batching without being offset by num_nodes',
        )
    )

    backbone = StubBackbone(hidden_dim=12, pooled_dim=9)
    model = SchurDecoderModel(
        backbone,
        n_l=3,
        n_m=5,
        n_shell=2,
        species=[1, 6, 7, 8],
        orbital_dim=8,
        orbital_hidden_dims=[10],
        global_dim=5,
        global_hidden_dims=[7],
        z_hidden_dims=[9],
        alpha_hidden_dims=[9],
        d1_hidden_dims=[9],
        num_radial_basis=4,
        z_row_chunk=8,
        checkpoint_z_head=False,
        max_basis_dim=None,
    )

    prediction = model(batch)
    recovered = [
        float((prediction['D'][0] - first_matrix).abs().max()),
        float((prediction['D'][1] - second_matrix).abs().max()),
    ]
    results.append(
        report(
            'target_unpacking',
            max(recovered) < 1e-12,
            f'padded upper triangles unpack to the original matrices, max error {max(recovered):.3e}',
        )
    )

    total, tasks = model.loss(prediction, batch.y, target_dim=batch.density_matrix_dim)
    total.backward()
    has_gradient = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    results.append(
        report(
            'end_to_end_loss',
            bool(torch.isfinite(total)) and len(tasks) == 1 and has_gradient,
            f'loss {float(total):.6e}, one reported task, finite gradients reach the heads',
        )
    )

    # the batched loss must be the mean of the per-molecule losses
    weights = [
        step_weights(int(matrix.shape[0]), 'uniform', 0, int(matrix.shape[0]),
                     matrix.device, matrix.dtype)
        for matrix in (prediction['D'][0], prediction['D'][1])
    ]
    separate = sum(
        float(
            reconstruction_loss(
                prediction['D_hat'][position].detach(),
                prediction['D'][position],
                weights[position],
                'mae',
                'entry_count',
                'total',
                promolecular=None,
            )
        )
        for position in (0, 1)
    ) / 2.0
    difference = abs(separate - float(tasks[0])) / max(separate, 1e-30)
    results.append(
        report(
            'batched_loss_matches_per_molecule',
            difference < 1e-12,
            f'per-molecule mean {separate:.12e} against batched {float(tasks[0]):.12e}',
        )
    )

    # unpadding is exercised in both storage layouts
    square = first_matrix.reshape(-1)
    dense = dense_from_padded_target(square, 20)
    results.append(
        report(
            'square_target_unpacking',
            float((dense - first_matrix).abs().max()) < 1e-12,
            'a stored square target unpacks as well as an upper triangle',
        )
    )

    return all(results)


# ----------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='verification for the sequential Schur decoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--wfn_path',
        type=str,
        default=None,
        help='optional Psi4 wavefunction file, to run the oracle checks on a real conformer',
    )
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    torch.manual_seed(arguments.seed)
    generator = torch.Generator().manual_seed(arguments.seed)
    torch.set_default_dtype(WORKING_DTYPE)

    synthetic = random_psd_matrix(120, rank=24, generator=generator)

    outcomes = [
        check_psd_and_symmetry(generator),
        check_oracle_round_trip(synthetic, 'synthetic'),
        check_teacher_forcing(synthetic, 'synthetic'),
        check_detach_is_value_preserving(generator),
        check_loss_identity(generator),
        check_gradients(generator),
        check_delta_learning(generator),
        check_conditioning_widths(),
        check_end_to_end(generator),
    ]

    if arguments.wfn_path is not None:
        from schur_decomp_test import load_wavefunction_matrices

        density_matrix, _, _, record = load_wavefunction_matrices(
            arguments.wfn_path, compute_promolecular=False
        )
        real = torch.from_numpy(density_matrix).to(WORKING_DTYPE)
        logging.info(
            '[INFO] real conformer %s, n_basis = %d',
            record.get('chembl_id', arguments.wfn_path),
            int(real.shape[0]),
        )
        outcomes.append(check_oracle_round_trip(real, 'real'))
        outcomes.append(check_teacher_forcing(real, 'real'))

    logging.info('')
    logging.info(
        '%d of %d checks passed', sum(1 for passed in outcomes if passed), len(outcomes)
    )
    sys.exit(0 if all(outcomes) else 1)


if __name__ == '__main__':
    main()
