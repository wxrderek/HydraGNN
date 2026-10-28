'''
Verification for the composite loss terms in hydragnn/domain_losses/.

Everything runs in float64 on CPU with synthetic inputs. The checks are, in order:

  1. each density matrix term is zero at the truth, and the delta-relative term needs the promolecular matrix
  2. each term matches an independent numpy evaluation
  3. gradcheck of each term with respect to the predicted density matrix
  4. the Schur and dense adapters give the same views and term values for the same molecules
  5. without data or without the provider, Base.loss and SchurDecoderModel.loss are unchanged
  6. trace normalization with the electron count term keeps the trace scale bounded over
     training steps on the stub model, where trace normalization alone drifts
  7. dense trace normalization: exact electron count, padding untouched, agreement with the
     Schur scaling, gradcheck, and Base.loss comparing the normalized rows with the target

Run it from the qmugs example directory, or through scripts/dcc/qmugs_composite_loss_test_dcc.sh.
'''

import sys
import types
import logging
import argparse

import numpy as np
import torch
from torch_geometric.data import Batch

from hydragnn.domain_losses import CompositeLoss, create_domain_loss
from hydragnn.domain_losses.density_matrix import (
    density_matrix_views,
    electron_count_term,
    frobenius_delta_term,
    frobenius_total_term,
    idempotency_term,
    overlap_metric_term,
)
from hydragnn.models.Base import Base

from schur_trace_test import (
    WORKING_DTYPE,
    build_model,
    build_molecule,
    padded_upper_triangle,
    random_overlap_matrix,
    random_psd_matrix,
    report,
)

TERMS = {
    'electron_count': electron_count_term,
    'overlap_metric': overlap_metric_term,
    'idempotency': idempotency_term,
    'frobenius_total': frobenius_total_term,
    'frobenius_delta': frobenius_delta_term,
}


def random_view(n_basis: int, generator, multiplicity: int = 1):
    '''one synthetic per-molecule view with a predicted, raw and true density matrix'''
    D = random_psd_matrix(n_basis, rank=max(2, n_basis // 4), generator=generator)
    D_hat = random_psd_matrix(n_basis, rank=max(2, n_basis // 4), generator=generator)
    return {
        'D_hat': D_hat,
        'D_hat_raw': 1.3 * D_hat,
        'D': D,
        'promolecular': random_psd_matrix(n_basis, rank=max(2, n_basis // 4), generator=generator),
        'overlap': random_overlap_matrix(n_basis, generator),
        'n_electrons': torch.tensor(float(2 * n_basis), dtype=WORKING_DTYPE),
        'multiplicity': multiplicity,
    }


def numpy_terms(views):
    '''the three terms evaluated independently in numpy'''
    count, metric, idempotency, total, delta = [], [], [], [], []
    for view in views:
        D_hat, D_raw, D, S, P = (view[key].numpy() for key in ('D_hat', 'D_hat_raw', 'D', 'overlap', 'promolecular'))
        total.append(np.linalg.norm(D_hat - D) ** 2 / np.linalg.norm(D) ** 2)
        delta.append(np.linalg.norm(D_hat - D) ** 2 / np.linalg.norm(D - P) ** 2)
        N = float(view['n_electrons'])
        count.append(((np.trace(S @ D_raw) - N) / N) ** 2)
        R = D_hat - D
        metric.append(np.trace(R @ S @ R @ S) / np.trace(D @ S @ D @ S))
        if view['multiplicity'] == 1:
            idempotency.append(np.linalg.norm(D_hat @ S @ D_hat - 2.0 * D_hat) ** 2 / np.linalg.norm(D) ** 2)
    return {'electron_count': np.mean(count), 'overlap_metric': np.mean(metric), 'idempotency': np.mean(idempotency),
            'frobenius_total': np.mean(total), 'frobenius_delta': np.mean(delta)}


def check_zero_at_truth(generator):
    '''1. every term vanishes when the prediction is a closed-shell density matrix equal to the truth'''
    n_basis, n_occupied = 14, 5
    overlap = random_overlap_matrix(n_basis, generator)
    # orbitals orthonormal in the overlap metric: coefficients times overlap times coefficients is the identity
    cholesky = torch.linalg.cholesky(overlap)
    orthonormal, _ = torch.linalg.qr(torch.randn((n_basis, n_occupied), dtype=WORKING_DTYPE, generator=generator))
    coefficients = torch.linalg.solve_triangular(cholesky.T, orthonormal, upper=True) # (n_basis, n_occupied)
    D = 2.0 * coefficients @ coefficients.T # (n_basis, n_basis)
    view = {
        'D_hat': D, 'D_hat_raw': D, 'D': D, 'overlap': overlap,
        'promolecular': random_psd_matrix(n_basis, rank=4, generator=generator),
        'n_electrons': torch.tensor(float(2 * n_occupied), dtype=WORKING_DTYPE), 'multiplicity': 1,
    }
    values = {name: float(term([view])) for name, term in TERMS.items()}
    try:
        frobenius_delta_term([dict(view, promolecular=None)])
        refused = False
    except ValueError:
        refused = True
    return report('zero_at_truth', max(values.values()) < 1e-24 and refused,
                  ', '.join(f'{name} {value:.2e}' for name, value in values.items())
                  + f'; frobenius_delta without the promolecular matrix raises: {refused}')


def check_against_numpy(generator):
    '''2. agreement with numpy, including an open-shell molecule left out of the idempotency term'''
    views = [random_view(9, generator), random_view(17, generator), random_view(12, generator, multiplicity=3)]
    reference = numpy_terms(views)
    worst = max(abs(float(TERMS[name](views)) - reference[name]) / abs(reference[name]) for name in TERMS)
    return report('against_numpy', worst < 1e-12, f'max relative difference {worst:.2e}')


def check_gradients(generator):
    '''3. gradcheck of each term with respect to the predicted and raw density matrices'''
    passed = True
    for name, term in TERMS.items():
        view = random_view(5, generator)
        D_hat = view['D_hat'].clone().requires_grad_(True)

        def evaluate(D_hat):
            return term([dict(view, D_hat=D_hat, D_hat_raw=1.3 * D_hat)])

        passed = passed and torch.autograd.gradcheck(evaluate, (D_hat,), eps=1e-6, atol=1e-6)
    return report('gradcheck', bool(passed), f'analytic and numerical gradients agree for all {len(TERMS)} terms')


def check_adapters(generator):
    '''4. the Schur and dense adapters agree on the same molecules, delta target'''
    n_padded = 20
    molecules, overlaps, totals, promolecular = [], [], [], []
    for n_atoms, per_atom in ((4, 5), (3, 4)):
        overlap, molecule = build_molecule(n_atoms, per_atom, n_padded, generator)
        molecules.append(molecule)
        overlaps.append(overlap)
    batch = Batch.from_data_list(molecules)

    from hydragnn.decoders.schur.model import dense_from_padded_target
    schur_pred = {'graph_index': [0, 1], 'D_hat': [], 'D': [], 'promolecular': [], 'trace_scale': [None, None]}
    dense_rows = []
    for position, molecule in enumerate(molecules):
        n_basis = int(molecule.density_matrix_dim)
        pro = dense_from_padded_target(molecule.promolecular_upper_triangle, n_basis)
        D = dense_from_padded_target(molecule.y, n_basis) + pro
        D_hat = random_psd_matrix(n_basis, rank=4, generator=generator)
        schur_pred['D_hat'].append(D_hat)
        schur_pred['D'].append(D)
        schur_pred['promolecular'].append(pro)
        # the dense head predicts the delta against the promolecular matrix as a padded upper triangle
        dense_rows.append(padded_upper_triangle(D_hat - pro, n_padded))
    dense_pred = [torch.stack(dense_rows)]

    schur_views = density_matrix_views(schur_pred, batch, None, {})
    dense_views = density_matrix_views(dense_pred, batch, None, {'dense_target': 'delta'})
    worst = 0.0
    for a, b in zip(schur_views, dense_views):
        for key in ('D_hat', 'D_hat_raw', 'D', 'promolecular', 'overlap', 'n_electrons'):
            worst = max(worst, float((a[key] - b[key]).abs().max()))
    term_difference = max(abs(float(term(schur_views)) - float(term(dense_views))) for term in TERMS.values())
    return report('adapters_agree', worst < 1e-12 and term_difference < 1e-12,
                  f'max view difference {worst:.2e}, max term difference {term_difference:.2e}')


def check_unchanged_without_provider(generator):
    '''5. no data or no provider leaves both loss methods unchanged; the provider adds exactly its terms'''
    n_padded = 20
    batch = Batch.from_data_list([build_molecule(4, 5, n_padded, generator)[1], build_molecule(3, 4, n_padded, generator)[1]])
    config = {'enabled': True, 'provider': 'composite', 'density_matrix': {'dense_target': 'delta'},
              'terms': [{'name': 'electron_count', 'weight': 0.3}, {'name': 'overlap_metric', 'weight': 0.025},
                        {'name': 'idempotency', 'weight': 0.013}, {'name': 'frobenius_total', 'weight': 0.020},
                        {'name': 'frobenius_delta', 'weight': 0.0066}]}
    results = []

    # Schur decoder
    model = build_model(False, 'delta', 'teacher_forcing')
    pred = model(batch)
    plain, _ = model.loss(pred, batch.y)
    with_data_no_provider, _ = model.loss(pred, batch.y, data=batch)
    create_domain_loss(model.backbone, config)
    with_provider, _ = model.loss(pred, batch.y, data=batch)
    reported = 'domain.electron_count' in model.last_loss_components
    extra, _ = model.backbone.composite_loss(pred, batch, model)
    without_data, _ = model.loss(pred, batch.y)
    # in eval mode with include_in_evaluation false the terms are reported but not added
    model.backbone.composite_loss = CompositeLoss(dict(config, include_in_evaluation=False))
    model.eval()
    evaluated, _ = model.loss(pred, batch.y, data=batch)
    left_out = bool(torch.equal(evaluated, plain)) and 'domain.electron_count' in model.last_loss_components
    model.train()
    training_mode, _ = model.loss(pred, batch.y, data=batch)
    left_out = left_out and abs(float(training_mode - with_provider)) < 1e-14
    results.append(report(
        'include_in_evaluation_false', left_out,
        'eval mode returns the plain loss with the terms reported; training mode still adds them',
    ))
    results.append(report(
        'schur_loss_unchanged',
        bool(torch.equal(plain, with_data_no_provider)) and bool(torch.equal(plain, without_data))
        and abs(float(with_provider - plain - extra)) < 1e-14 and reported,
        f'plain {float(plain):.6e}, with provider {float(with_provider):.6e}, extra {float(extra):.6e}',
    ))

    # Base.loss, on a minimal object carrying only what the method reads
    dense_pred = [torch.randn((2, n_padded * (n_padded + 1) // 2), dtype=WORKING_DTYPE)]
    supervised = torch.tensor(0.5, dtype=WORKING_DTYPE)
    fake = types.SimpleNamespace(
        var_output=False, ilossweights_nll=0, ilossweights_hyperp=1,
        loss_hpweighted=lambda *args, **kwargs: (supervised, [supervised]),
    )
    base_plain, _ = Base.loss(fake, dense_pred, batch.y, None)
    base_no_provider, _ = Base.loss(fake, dense_pred, batch.y, None, data=batch)
    fake.composite_loss = CompositeLoss(config)
    base_with_provider, _ = Base.loss(fake, dense_pred, batch.y, None, data=batch)
    base_extra, _ = fake.composite_loss(dense_pred, batch, fake)
    results.append(report(
        'base_loss_unchanged',
        float(base_plain) == 0.5 and float(base_no_provider) == 0.5
        and abs(float(base_with_provider) - 0.5 - float(base_extra)) < 1e-14,
        f'plain {float(base_plain):.3f}, with provider {float(base_with_provider):.6e}',
    ))
    return all(results)


def check_trace_scale_bounded(generator, steps: int):
    '''6. trace normalization with and without the electron count term, Adam on the stub model'''
    n_padded = 20
    batch = Batch.from_data_list([build_molecule(4, 5, n_padded, generator)[1], build_molecule(3, 4, n_padded, generator)[1]])
    finals = {}
    for weight in (0.0, 0.3):
        model = build_model(True, 'delta', 'teacher_forcing')
        if weight > 0.0:
            create_domain_loss(model.backbone, {'enabled': True, 'provider': 'composite',
                                                'terms': [{'name': 'electron_count', 'weight': weight}]})
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        for _ in range(steps):
            optimizer.zero_grad()
            pred = model(batch)
            total, _ = model.loss(pred, batch.y, data=batch)
            total.backward()
            optimizer.step()
        finals[weight] = [float(scale) for scale in model(batch)['trace_scale']]
    bounded = all(0.5 < scale < 2.0 for scale in finals[0.3])
    return report('trace_scale_bounded', bounded,
                  f'after {steps} steps, trace scales without the term {finals[0.0]}, with it {finals[0.3]}')


def check_dense_trace_normalization(generator):
    '''7. the dense head's trace normalization through CompositeLoss.transform_prediction and Base.loss'''
    from hydragnn.decoders.schur.model import dense_from_padded_target, trace_normalize
    from hydragnn.domain_losses.density_matrix import trace_normalize_dense_rows

    n_padded = 20
    molecules = [build_molecule(4, 5, n_padded, generator)[1], build_molecule(3, 4, n_padded, generator)[1]]
    batch = Batch.from_data_list(molecules)
    rows = []
    for molecule in molecules:
        n_basis = int(molecule.density_matrix_dim)
        pro = dense_from_padded_target(molecule.promolecular_upper_triangle, n_basis)
        # a raw dense prediction of the delta, with nonzero padding to check it is left alone
        row = padded_upper_triangle(random_psd_matrix(n_basis, rank=4, generator=generator) - pro, n_padded)
        physical = row != 0
        row = torch.where(physical, row, torch.randn(row.shape, dtype=WORKING_DTYPE, generator=generator))
        rows.append(row)
    raw_rows = torch.stack(rows) # (2, T(n_padded))
    config = {'enabled': True, 'provider': 'composite',
              'terms': [{'name': 'electron_count', 'weight': 0.24}],
              'density_matrix': {'dense_target': 'delta', 'trace_normalization': True}}
    composite = CompositeLoss(config)
    normalized_rows = composite.transform_prediction([raw_rows], batch, None)[0]

    count_error, padding_change, schur_difference = 0.0, 0.0, 0.0
    for graph, molecule in enumerate(molecules):
        n_basis = int(molecule.density_matrix_dim)
        pro = dense_from_padded_target(molecule.promolecular_upper_triangle, n_basis)
        overlap = dense_from_padded_target(molecule.overlap_upper_triangle, n_basis)
        total = dense_from_padded_target(normalized_rows[graph], n_basis) + pro
        count_error = max(count_error, abs(float(torch.sum(overlap * total)) - float(molecule.n_electrons)))
        # the Schur scaling applied to the unpacked raw total gives the same matrix
        expected, _ = trace_normalize(dense_from_padded_target(raw_rows[graph], n_basis) + pro, overlap,
                                      molecule.n_electrons.to(WORKING_DTYPE))
        schur_difference = max(schur_difference, float((total - expected).abs().max()))
        upper = torch.triu_indices(n_basis, n_basis)
        offsets = upper[0] * n_padded - (upper[0] * (upper[0] - 1)) // 2 + (upper[1] - upper[0])
        untouched = torch.ones(raw_rows.shape[1], dtype=torch.bool)
        untouched[offsets] = False
        # a molecule as large as the padding width has no padding entries
        if bool(untouched.any()):
            padding_change = max(padding_change, float((normalized_rows[graph][untouched] - raw_rows[graph][untouched]).abs().max()))
    results = [report('dense_trace_exact', count_error < 1e-10 and padding_change == 0.0 and schur_difference < 1e-12,
                      f'electron count error {count_error:.2e}, padding change {padding_change:.2e}, '
                      f'difference from the Schur scaling {schur_difference:.2e}')]

    # gradcheck of the transform with respect to the raw rows
    small = raw_rows.clone().requires_grad_(True)
    passed = torch.autograd.gradcheck(
        lambda x: trace_normalize_dense_rows(x, batch, 'delta')[0], (small,), eps=1e-6, atol=1e-6
    )
    results.append(report('dense_trace_gradcheck', bool(passed), 'analytic and numerical gradients agree'))

    # Base.loss: the supervised loss sees the normalized rows, the terms the raw ones
    target_rows = batch.y.reshape(2, -1)
    fake = types.SimpleNamespace(
        var_output=False, ilossweights_nll=0, ilossweights_hyperp=1, composite_loss=composite,
        loss_hpweighted=lambda pred, value, head_index, **kwargs: (
            torch.mean(torch.abs(pred[0] - target_rows)), [torch.mean(torch.abs(pred[0] - target_rows))]),
    )
    total, _ = Base.loss(fake, [raw_rows], batch.y, None, data=batch)
    extra, _ = composite([raw_rows], batch, fake)
    expected_total = torch.mean(torch.abs(normalized_rows - target_rows)) + extra
    views = density_matrix_views([raw_rows], batch, None, composite.options['density_matrix'])
    view_difference = max(float((views[g]['D_hat'] - (dense_from_padded_target(normalized_rows[g], int(m.density_matrix_dim))
                                + dense_from_padded_target(m.promolecular_upper_triangle, int(m.density_matrix_dim)))).abs().max())
                          for g, m in enumerate(molecules))
    results.append(report('dense_trace_base_loss', abs(float(total - expected_total)) < 1e-14 and view_difference < 1e-12,
                          f'Base.loss {float(total):.6e} against expected {float(expected_total):.6e}, '
                          f'view difference {view_difference:.2e}'))
    return all(results)


def main():
    parser = argparse.ArgumentParser(description='verification for the composite loss terms')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--steps', type=int, default=300, help='optimizer steps in check 6')
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    torch.set_default_dtype(WORKING_DTYPE)
    generator = torch.Generator().manual_seed(arguments.seed)

    outcomes = [
        check_zero_at_truth(generator),
        check_against_numpy(generator),
        check_gradients(generator),
        check_adapters(generator),
        check_unchanged_without_provider(generator),
        check_trace_scale_bounded(generator, arguments.steps),
        check_dense_trace_normalization(generator),
    ]
    logging.info('ALL PASS' if all(outcomes) else 'SOME CHECKS FAILED')
    sys.exit(0 if all(outcomes) else 1)


if __name__ == '__main__':
    main()
