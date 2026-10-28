###########################################################################################
# Density matrix loss terms for the composite loss
# Reference: references/schur_decoder.tex, Sections 2.1-2.2 and 4.4
# Author: Derek Wang
###########################################################################################

import math

import torch


def density_matrix_views(pred, data, model, options):
    '''per-molecule predicted and true density matrices with their overlap matrix and number of electrons'''

    # imported here because hydragnn.decoders imports from hydragnn.models, which imports this package
    from hydragnn.decoders.schur.model import dense_from_padded_target

    if not (hasattr(data, 'overlap_upper_triangle') and hasattr(data, 'n_electrons')):
        raise ValueError(
            'density matrix loss terms need overlap_upper_triangle and n_electrons in the batch; use a '
            'dataset written by examples/qmugs/utils/add_overlap.py or preprocessed with --store_overlap_matrix'
        )
    sizes = data.density_matrix_dim.reshape(-1).to(torch.int64) # (num_graphs,)
    num_graphs = int(sizes.shape[0])
    overlap_rows = data.overlap_upper_triangle.reshape(num_graphs, -1)
    electron_numbers = data.n_electrons.reshape(-1)
    multiplicities = data.multiplicity.reshape(-1)

    views = []
    if isinstance(pred, dict):
        if options.get('trace_normalization', False):
            raise ValueError(
                'density_matrix.trace_normalization is for the dense head; for the Schur decoder set '
                'SchurDecoder.trace_normalization instead'
            )
        # Schur decoder: dense totals per kept conformer, already unpadded in its forward
        for position, graph in enumerate(pred['graph_index']):
            D_hat = pred['D_hat'][position] # (n_basis, n_basis)
            n_basis = int(D_hat.shape[0])
            scale = pred['trace_scale'][position] if 'trace_scale' in pred else None
            promolecular = pred['promolecular'][position] if 'promolecular' in pred else None
            views.append({
                'D_hat': D_hat,
                # the prediction before trace normalization, whose scale the electron count term pins
                'D_hat_raw': D_hat if scale is None else D_hat / scale,
                'D': pred['D'][position],
                'promolecular': promolecular,
                'overlap': dense_from_padded_target(overlap_rows[graph], n_basis).to(D_hat.dtype),
                'n_electrons': electron_numbers[graph].to(D_hat.dtype),
                'multiplicity': int(multiplicities[graph]),
            })
        return views

    # imported here for the same reason as dense_from_padded_target
    from hydragnn.decoders.schur.model import trace_normalize

    # dense head: one padded row per molecule, a full matrix or its row-major upper triangle
    if 'dense_target' not in options:
        raise ValueError("density matrix loss terms on a dense head need density_matrix.dense_target, 'delta' or 'total'")
    dense_target = options['dense_target']
    if dense_target not in ('delta', 'total'):
        raise ValueError(f"density_matrix.dense_target must be 'delta' or 'total', got {dense_target}")
    prediction_rows = pred[0].reshape(num_graphs, -1)
    target_rows = data.y.reshape(num_graphs, -1)
    # the promolecular matrix is read whenever stored, since the delta-relative term needs it for either target
    promolecular_rows = None
    if hasattr(data, 'promolecular_upper_triangle'):
        promolecular_rows = data.promolecular_upper_triangle.reshape(num_graphs, -1)
    elif dense_target == 'delta':
        raise ValueError("density_matrix.dense_target='delta' needs promolecular_upper_triangle in the batch")

    for graph in range(num_graphs):
        n_basis = int(sizes[graph])
        D_hat = dense_from_padded_target(prediction_rows[graph], n_basis) # (n_basis, n_basis)
        D = dense_from_padded_target(target_rows[graph], n_basis).to(D_hat.dtype) # (n_basis, n_basis)
        promolecular = None
        if promolecular_rows is not None:
            promolecular = dense_from_padded_target(promolecular_rows[graph], n_basis).to(D_hat.dtype)
        if dense_target == 'delta':
            # the head predicts the delta against the promolecular matrix; the terms act on totals
            D_hat = D_hat + promolecular
            D = D + promolecular
        overlap = dense_from_padded_target(overlap_rows[graph], n_basis).to(D_hat.dtype)
        n_electrons = electron_numbers[graph].to(D_hat.dtype)
        # with trace normalization the terms see the scaled total, the electron count term the raw one
        D_hat_final = D_hat
        if options.get('trace_normalization', False):
            D_hat_final, _ = trace_normalize(D_hat, overlap, n_electrons)
        views.append({
            'D_hat': D_hat_final,
            'D_hat_raw': D_hat,
            'D': D,
            'promolecular': promolecular,
            'overlap': overlap,
            'n_electrons': n_electrons,
            'multiplicity': int(multiplicities[graph]),
        })
    return views


def trace_normalize_dense_rows(prediction_rows, data, dense_target: str):
    '''scale each dense prediction to the number of electrons, returned in the same padded upper-triangle layout'''

    # imported here because hydragnn.decoders imports from hydragnn.models, which imports this package
    from hydragnn.decoders.schur.model import dense_from_padded_target, trace_normalize

    if not (hasattr(data, 'overlap_upper_triangle') and hasattr(data, 'n_electrons')):
        raise ValueError(
            'dense trace normalization needs overlap_upper_triangle and n_electrons in the batch; use a dataset '
            'written by examples/qmugs/utils/add_overlap.py or preprocessed with --store_overlap_matrix'
        )
    num_graphs, length = prediction_rows.shape # (num_graphs, padded row length)
    n_padded = (math.isqrt(8 * length + 1) - 1) // 2
    if n_padded * (n_padded + 1) // 2 != length:
        raise NotImplementedError('dense trace normalization supports only upper-triangle outputs')
    sizes = data.density_matrix_dim.reshape(-1).to(torch.int64) # (num_graphs,)
    assert int(sizes.shape[0]) == num_graphs
    overlap_rows = data.overlap_upper_triangle.reshape(num_graphs, -1)
    electron_numbers = data.n_electrons.reshape(-1)
    promolecular_rows = None
    if dense_target == 'delta':
        if not hasattr(data, 'promolecular_upper_triangle'):
            raise ValueError("dense trace normalization with the delta target needs promolecular_upper_triangle")
        promolecular_rows = data.promolecular_upper_triangle.reshape(num_graphs, -1)

    new_rows, scales = [], []
    for graph in range(num_graphs):
        n_basis = int(sizes[graph])
        prediction = dense_from_padded_target(prediction_rows[graph], n_basis) # (n_basis, n_basis)
        promolecular = None
        if promolecular_rows is not None:
            promolecular = dense_from_padded_target(promolecular_rows[graph], n_basis).to(prediction.dtype)
            prediction = prediction + promolecular
        overlap = dense_from_padded_target(overlap_rows[graph], n_basis).to(prediction.dtype)
        # the same scaling as the Schur decoder, applied to the total
        normalized, scale = trace_normalize(prediction, overlap, electron_numbers[graph].to(prediction.dtype))
        if promolecular is not None:
            normalized = normalized - promolecular

        # write the physical entries back; the zero padding of the row stays as predicted
        upper = torch.triu_indices(n_basis, n_basis, device=prediction_rows.device) # (2, T(n_basis))
        rows, cols = upper[0], upper[1]
        offsets = rows * n_padded - (rows * (rows - 1)) // 2 + (cols - rows) # (T(n_basis),)
        new_rows.append(prediction_rows[graph].scatter(0, offsets, normalized[rows, cols]))
        scales.append(scale)
    return torch.stack(new_rows), scales # (num_graphs, padded row length), list of scalars


def electron_count_term(views):
    '''squared relative electron count error of the prediction before trace normalization, mean over molecules'''
    values = []
    for view in views:
        # trace of the overlap matrix times the density matrix; both symmetric
        electron_count = torch.sum(view['overlap'] * view['D_hat_raw'])
        values.append(((electron_count - view['n_electrons']) / view['n_electrons']) ** 2)
    return torch.stack(values).mean()


def overlap_metric_term(views):
    '''squared relative error in the norm weighted by the overlap matrix, mean over molecules'''
    values = []
    for view in views:
        n_basis = int(view['D'].shape[0])
        assert view['D_hat'].shape == (n_basis, n_basis) and view['overlap'].shape == (n_basis, n_basis)
        residual_overlap = (view['D_hat'] - view['D']) @ view['overlap'] # (n_basis, n_basis)
        true_overlap = view['D'] @ view['overlap'] # (n_basis, n_basis)
        values.append(torch.trace(residual_overlap @ residual_overlap) / torch.trace(true_overlap @ true_overlap))
    return torch.stack(values).mean()


def idempotency_term(views):
    '''squared relative closed-shell idempotency defect, mean over molecules with multiplicity 1'''
    values = []
    for view in views:
        # zero only for doubly occupied orbitals, so open-shell molecules are left out
        if view['multiplicity'] != 1:
            continue
        D_hat = view['D_hat']
        defect = D_hat @ view['overlap'] @ D_hat - 2.0 * D_hat # (n_basis, n_basis)
        values.append(torch.sum(defect ** 2) / torch.sum(view['D'] ** 2))
    if len(values) == 0:
        return views[0]['D_hat'].new_zeros(())
    return torch.stack(values).mean()


def frobenius_total_term(views):
    '''squared relative Frobenius error against the true total density matrix, mean over molecules'''
    values = []
    for view in views:
        n_basis = int(view['D'].shape[0])
        assert view['D_hat'].shape == (n_basis, n_basis)
        values.append(torch.sum((view['D_hat'] - view['D']) ** 2) / torch.sum(view['D'] ** 2))
    return torch.stack(values).mean()


def frobenius_delta_term(views):
    '''squared Frobenius error relative to the true delta against the promolecular matrix, mean over molecules'''
    values = []
    for view in views:
        if view['promolecular'] is None:
            raise ValueError(
                'frobenius_delta needs the promolecular matrix; preprocess with a density_matrix_delta output name '
                'so promolecular_upper_triangle is stored'
            )
        n_basis = int(view['D'].shape[0])
        assert view['D_hat'].shape == (n_basis, n_basis) and view['promolecular'].shape == (n_basis, n_basis)
        # the residual is the same for the total and the delta; only the normalizer differs
        delta_norm_squared = torch.sum((view['D'] - view['promolecular']) ** 2)
        if float(delta_norm_squared.detach()) == 0.0:
            raise ValueError('the true density matrix equals the promolecular matrix, so the delta-relative error is undefined')
        values.append(torch.sum((view['D_hat'] - view['D']) ** 2) / delta_norm_squared)
    return torch.stack(values).mean()
