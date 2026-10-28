'''
Verification for the trace normalization of the sequential Schur decoder.

Everything runs in float64 on CPU with synthetic inputs. The checks are, in order:

  1. after normalization the electron count equals the number of electrons
  2. the normalized free-running prediction stays symmetric and PSD
  3. gradcheck through the recursion and the normalization
  4. end to end through SchurDecoderModel: with the flag off the prediction is the plain
     recursion output, with the flag on it is that output times the scale, the loss and
     its gradients are finite for both targets, and the invalid settings are refused

Run it from the qmugs example directory, or through scripts/dcc/qmugs_schur_trace_test_dcc.sh.
'''

import sys
import logging
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from hydragnn.decoders.schur.model import SchurDecoderModel, trace_normalize
from hydragnn.decoders.schur.recursion import bordering_recursion

WORKING_DTYPE = torch.float64


# ----------------------------------------------------------------------------------------------------
# helpers


def random_psd_matrix(n_basis: int, rank: int, generator):
    '''a symmetric PSD matrix of the given rank, standing in for a density matrix'''
    factor = torch.randn((n_basis, rank), dtype=WORKING_DTYPE, generator=generator)
    matrix = factor @ factor.transpose(0, 1) # (n_basis, n_basis)
    return 0.5 * (matrix + matrix.transpose(0, 1))


def random_overlap_matrix(n_basis: int, generator):
    '''a symmetric positive definite matrix with unit diagonal, standing in for the overlap matrix'''
    factor = torch.randn((n_basis, n_basis), dtype=WORKING_DTYPE, generator=generator)
    matrix = factor @ factor.transpose(0, 1) / n_basis + 0.5 * torch.eye(n_basis, dtype=WORKING_DTYPE)
    scale = torch.diagonal(matrix).rsqrt() # (n_basis,)
    matrix = scale[:, None] * matrix * scale[None, :]
    return 0.5 * (matrix + matrix.transpose(0, 1))


def random_latents(n_basis: int, generator):
    '''random latents and pivots of the shapes the recursion expects'''
    pairs = torch.tril(
        torch.randn((n_basis, n_basis), dtype=WORKING_DTYPE, generator=generator), diagonal=-1
    )
    delta = F.softplus(torch.randn(n_basis, dtype=WORKING_DTYPE, generator=generator))
    return [pairs[k, :k] for k in range(n_basis)], delta


def padded_upper_triangle(matrix, n_padded: int):
    '''row-major upper triangle padded to n_padded, as qmugs.py stores targets and the overlap matrix'''
    n_basis = int(matrix.shape[0])
    rows, cols = torch.triu_indices(n_basis, n_basis)
    upper = torch.zeros(n_padded * (n_padded + 1) // 2, dtype=WORKING_DTYPE)
    upper[rows * n_padded - (rows * (rows - 1)) // 2 + (cols - rows)] = matrix[rows, cols]
    return upper


def report(name: str, passed: bool, detail: str):
    logging.info('[%s] %s -- %s', 'PASS' if passed else 'FAIL', name, detail)
    return passed


# ----------------------------------------------------------------------------------------------------
# the checks


def check_electron_count_and_psd(generator):
    '''1 and 2. exact electron count, symmetry and PSD after normalization, free running'''
    worst_count, worst_eigenvalue, worst_asymmetry = 0.0, float('inf'), 0.0
    for n_basis in (5, 12, 25, 40):
        for _ in range(5):
            z, delta = random_latents(n_basis, generator)
            D_hat, _, _ = bordering_recursion(z, delta)
            overlap = random_overlap_matrix(n_basis, generator)
            n_electrons = torch.tensor(float(2 * n_basis), dtype=WORKING_DTYPE)
            normalized, scale = trace_normalize(D_hat, overlap, n_electrons)

            count = float(torch.sum(overlap * normalized))
            worst_count = max(worst_count, abs(count - float(n_electrons)))
            eigenvalues = torch.linalg.eigvalsh(normalized)
            worst_eigenvalue = min(worst_eigenvalue, float(eigenvalues.min() / eigenvalues.max().abs()))
            worst_asymmetry = max(worst_asymmetry, float((normalized - normalized.T).abs().max()))
            assert float(scale) > 0.0

    return all([
        report('electron_count', worst_count < 1e-10, f'max electron count error {worst_count:.3e}'),
        report('psd', worst_eigenvalue >= -1e-12, f'smallest eigenvalue over largest {worst_eigenvalue:.3e}'),
        report('symmetry', worst_asymmetry == 0.0, f'max asymmetry {worst_asymmetry:.3e}'),
    ])


def check_gradients(generator):
    '''3. gradcheck through the recursion followed by the normalization'''
    n_basis = 6
    z, delta = random_latents(n_basis, generator)
    overlap = random_overlap_matrix(n_basis, generator)
    n_electrons = torch.tensor(12.0, dtype=WORKING_DTYPE)
    pairs = torch.zeros((n_basis, n_basis), dtype=WORKING_DTYPE)
    for k in range(n_basis):
        pairs[k, :k] = z[k]
    lower = torch.tril_indices(n_basis, n_basis, offset=-1)
    z_flat = pairs[lower[0], lower[1]].clone().requires_grad_(True)
    delta = delta.clone().requires_grad_(True)

    def normalized_prediction(z_flat, delta):
        rows = [z_flat.new_zeros(0)]
        start = 0
        for k in range(1, n_basis):
            rows.append(z_flat[start:start + k])
            start += k
        D_hat, _, _ = bordering_recursion(rows, delta)
        return trace_normalize(D_hat, overlap, n_electrons)[0]

    passed = torch.autograd.gradcheck(normalized_prediction, (z_flat, delta), eps=1e-6, atol=1e-6)
    return report('gradcheck', bool(passed), 'analytic and numerical gradients agree')


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
    '''one synthetic conformer carrying every field the decoder reads, including the overlap matrix'''
    n_basis = n_atoms * orbitals_per_atom
    matrix = random_psd_matrix(n_basis, rank=max(2, n_basis // 5), generator=generator)
    promolecular = random_psd_matrix(n_basis, rank=max(2, n_basis // 5), generator=generator)
    overlap = random_overlap_matrix(n_basis, generator)
    return overlap, Data(
        x=torch.randn((n_atoms, 4), dtype=WORKING_DTYPE, generator=generator),
        pos=torch.randn((n_atoms, 3), dtype=WORKING_DTYPE, generator=generator),
        atomic_numbers=torch.full((n_atoms, 1), 6.0, dtype=WORKING_DTYPE),
        # the stored target is the delta against the promolecular matrix, as in the QMugs data
        y=padded_upper_triangle(matrix - promolecular, n_padded),
        promolecular_upper_triangle=padded_upper_triangle(promolecular, n_padded),
        overlap_upper_triangle=padded_upper_triangle(overlap, n_padded),
        n_electrons=torch.tensor(6 * n_atoms, dtype=torch.int32),
        density_matrix_dim=torch.tensor(n_basis, dtype=torch.int32),
        natoms_int=torch.tensor(n_atoms, dtype=torch.int32),
        charge=torch.tensor(0, dtype=torch.int32),
        multiplicity=torch.tensor(1, dtype=torch.int32),
        orbital_to_atom=torch.arange(n_basis, dtype=torch.int32) // orbitals_per_atom,
        orbital_angular_momentum=torch.arange(n_basis, dtype=torch.int32) % 3,
        orbital_within_shell=torch.arange(n_basis, dtype=torch.int32) % 5,
        orbital_shell=torch.arange(n_basis, dtype=torch.int32) % 2,
    )


def build_model(trace_normalization: bool, target: str, routing: str, supervision: str = 'joint'):
    '''a small decoder on the stub backbone, initialized from a fixed seed'''
    torch.manual_seed(1)
    return SchurDecoderModel(
        StubBackbone(hidden_dim=12, pooled_dim=9),
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
        target=target,
        routing=routing,
        supervision=supervision,
        chunk_width=4 if supervision == 'semi_sequential' else None,
        w_schedule='per_entry',
        trace_normalization=trace_normalization,
    )


def check_end_to_end(generator):
    '''4. the flag through SchurDecoderModel, on a two-molecule batch of different sizes'''
    n_padded = 20
    first_overlap, first = build_molecule(4, 5, n_padded, generator) # n_basis 20
    second_overlap, second = build_molecule(3, 4, n_padded, generator) # n_basis 12
    batch = Batch.from_data_list([first, second])
    overlaps = [first_overlap, second_overlap]
    results = []

    for target in ('total', 'delta'):
        for routing in ('free_running', 'teacher_forcing'):
            plain = build_model(False, target, routing)
            normalized = build_model(True, target, routing)
            plain_prediction = plain(batch)
            prediction = normalized(batch)

            # the flag off leaves the recursion output untouched; the flag on scales exactly that
            untouched = all(scale is None for scale in plain_prediction['trace_scale'])
            scale_error, count_error = 0.0, 0.0
            for position in range(2):
                raw = plain_prediction['D_hat'][position].detach()
                scale = prediction['trace_scale'][position].detach()
                expected_scale = float(batch.n_electrons[position]) / float(torch.sum(overlaps[position] * raw))
                scale_error = max(scale_error, abs(float(scale) - expected_scale) / expected_scale)
                count = float(torch.sum(overlaps[position] * prediction['D_hat'][position].detach()))
                count_error = max(count_error, abs(count - float(batch.n_electrons[position])))
                untouched = untouched and bool(torch.equal(prediction['D_hat'][position].detach(), scale * raw))

            total, tasks = normalized.loss(prediction, batch.y)
            total.backward()
            finite_gradients = all(
                torch.isfinite(parameter.grad).all()
                for parameter in normalized.parameters()
                if parameter.requires_grad and parameter.grad is not None
            )
            results.append(report(
                f'end_to_end_{target}_{routing}',
                untouched and scale_error < 1e-12 and count_error < 1e-10
                and bool(torch.isfinite(total)) and finite_gradients,
                f'scale error {scale_error:.3e}, electron count error {count_error:.3e}, loss {float(total.detach()):.6e}',
            ))

    # windowed supervision and a batch without the overlap matrix are refused
    try:
        build_model(True, 'delta', 'teacher_forcing', supervision='semi_sequential')
        refused_window = False
    except ValueError:
        refused_window = True
    results.append(report('refuses_windowed_supervision', refused_window, 'semi_sequential with the flag raises'))

    stripped = [molecule.clone() for molecule in (first, second)]
    for molecule in stripped:
        del molecule.overlap_upper_triangle
    try:
        build_model(True, 'delta', 'teacher_forcing')(Batch.from_data_list(stripped))
        refused_missing = False
    except ValueError:
        refused_missing = True
    results.append(report('refuses_missing_overlap', refused_missing, 'a batch without the overlap matrix raises'))
    plain_runs = build_model(False, 'delta', 'teacher_forcing')(Batch.from_data_list(stripped))
    results.append(report('flag_off_needs_no_overlap', len(plain_runs['D_hat']) == 2, 'the flag off runs without the overlap matrix'))

    return all(results)


# ----------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='verification for the Schur decoder trace normalization')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    torch.set_default_dtype(WORKING_DTYPE)
    generator = torch.Generator().manual_seed(arguments.seed)

    outcomes = [
        check_electron_count_and_psd(generator),
        check_gradients(generator),
        check_end_to_end(generator),
    ]
    logging.info('ALL PASS' if all(outcomes) else 'SOME CHECKS FAILED')
    sys.exit(0 if all(outcomes) else 1)


if __name__ == '__main__':
    main()
