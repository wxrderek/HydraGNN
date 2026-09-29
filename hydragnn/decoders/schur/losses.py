###########################################################################################
# Losses for the sequential Schur decoder
# Reference: references/schur_decoder.tex, Sections 4.1-4.3 and 6.1-6.3
# Author: Derek Wang
###########################################################################################

import torch


def step_weights(n_basis: int, schedule: str, active_lo: int, active_hi: int, device, dtype):
    '''the w_k of Section 4.2, zero outside the currently active window of steps'''

    # The three supervision schemes of Sections 6.1-6.3 differ only in which steps are active:
    # joint keeps all of them, semi_sequential a contiguous window that advances with the epoch,
    # per_step a window of width one. Adding a future schedule means one more branch here and
    # nothing else.
    assert 0 <= active_lo < active_hi <= n_basis

    weights = torch.zeros(n_basis, device=device, dtype=dtype) # (n_basis,)
    if schedule == 'uniform':
        # every step counts equally, so the entries introduced at an early step count for
        # much more each than the entries introduced at a late one. Normalized over the
        # window so the loss scale does not depend on the window width.
        weights[active_lo:active_hi] = 1.0 / (active_hi - active_lo)
    elif schedule == 'per_entry':
        # every matrix ENTRY counts equally instead. Step k introduces 2k + 1 entries, so
        # weighting by that exactly cancels the entry_count normalization N_k = 1/(2k + 1)
        # and the reconstruction loss below collapses to the plain full-matrix loss. This is
        # what makes the decoder's training curve comparable with a dense head trained on
        # the same target under the same loss function.
        counts = 2.0 * torch.arange(
            active_lo, active_hi, device=device, dtype=dtype
        ) + 1.0 # (active_hi - active_lo,)
        weights[active_lo:active_hi] = counts / counts.sum()
    else:
        raise ValueError(
            f"unsupported w_schedule: {schedule}; use 'uniform' or 'per_entry'"
        )

    return weights


def elementwise_residual(loss_function_type: str):
    '''the elementwise part of the norm in Section 4.2, before the sum over entries'''

    # hydragnn.utils.model.loss_function_selection returns reduced losses; the per-step form
    # needs the elementwise quantity so it can be summed per step and weighted by w_k.
    if loss_function_type == 'mae':
        return torch.abs
    if loss_function_type in ('mse', 'rmse'):
        return lambda residual: residual ** 2
    raise ValueError(
        f"unsupported loss_function_type for the Schur decoder: {loss_function_type}; "
        "use 'mae' or 'mse'"
    )


def reconstruction_loss(
    D_hat,
    density_matrix,
    weights,
    loss_function_type: str,
    normalization: str,
    target: str,
    promolecular=None,
):
    '''the per-step reconstruction loss of Section 4.2, on the newly predicted entries only'''

    # Step k introduces c_k, which enters the matrix twice by symmetry, and the single diagonal
    # entry d_k. Supervising only those is the efficiency note at the end of Section 4.2; every
    # earlier entry is already accounted for at its own step. Reading them off the assembled
    # matrix is equivalent to reading the per-step outputs, since row k of the strictly lower
    # triangle is exactly c_k and the diagonal is exactly d_k.
    n_basis = int(density_matrix.shape[0])
    assert D_hat.shape == (n_basis, n_basis)
    assert weights.shape == (n_basis,)

    if target == 'delta':
        # Section 3.5, Equation 17. Delta c_k = c_k - c_pro_k and Delta d_k = d_k - d_pro_k for
        # the prediction and for the target alike, which entrywise is just subtracting D_pro
        # from both matrices. The two subtractions are written out rather than cancelled
        # analytically: they do cancel for the entry-count normalization used here, but they
        # stop cancelling the moment N_k or the weight matrix W of Section 4.1 is made to
        # depend on the target's own magnitude, which is when delta learning starts to change
        # the objective rather than only the reported number.
        assert promolecular is not None, "target='delta' requires the promolecular matrix"
        assert promolecular.shape == (n_basis, n_basis)
        predicted = D_hat - promolecular # (n_basis, n_basis)
        truth = density_matrix - promolecular # (n_basis, n_basis)
    elif target == 'total':
        predicted = D_hat
        truth = density_matrix
    else:
        raise ValueError(f"unsupported target: {target}; use 'total' or 'delta'")

    elementwise = elementwise_residual(loss_function_type)
    residual = predicted - truth # (n_basis, n_basis)

    # row k of the strictly lower triangle holds the c_k residual; entries at or above the
    # diagonal are zeroed so the row sum runs over i < k only
    c_term = elementwise(torch.tril(residual, diagonal=-1)).sum(dim=1) # (n_basis,)
    d_term = elementwise(torch.diagonal(residual)) # (n_basis,)

    if normalization == 'entry_count':
        # step k contributes 2(k) off-diagonal entries and 1 diagonal entry
        counts = 2.0 * torch.arange(
            n_basis, device=residual.device, dtype=residual.dtype
        ) + 1.0 # (n_basis,)
    elif normalization == 'none':
        counts = torch.ones(n_basis, device=residual.device, dtype=residual.dtype)
    else:
        raise ValueError(
            f"unsupported step_normalization: {normalization}; use 'entry_count' or 'none'"
        )

    # sanity anchor: with weights[k] proportional to (2k + 1), which is w_schedule
    # 'per_entry' above, this collapses exactly to the plain full-matrix loss, so the
    # per-step form is a reweighting of the familiar objective
    per_step = (2.0 * c_term + d_term) / counts # (n_basis,)
    return (weights * per_step).sum()


def regularization_loss(z_pairs, delta, weights, z_weight: float, delta_weight: float):
    '''the latent regularization of Section 4.3, Equation 21'''

    # z_weight is the lever on error amplification: a perturbation of D_{k-1} reaches c_k as
    # dc_k = dD_{k-1} z_k, so large latents are what let an error at one step grow at the next.
    # Section 10.5 of the diagnostics measured that growth directly under free running.
    n_basis = int(delta.shape[0])
    assert z_pairs.shape == (n_basis, n_basis)
    assert weights.shape == (n_basis,)

    total = torch.zeros((), device=delta.device, dtype=delta.dtype)
    if z_weight != 0.0:
        z_norm_squared = (z_pairs ** 2).sum(dim=1) # (n_basis,), this is ||z_k||^2
        total = total + z_weight * (weights * z_norm_squared).sum()
    if delta_weight != 0.0:
        total = total + delta_weight * (weights * delta ** 2).sum()

    return total


@torch.no_grad()
def relative_errors(D_hat, density_matrix, promolecular=None):
    '''relative Frobenius errors against the total matrix and against the delta'''

    # The delta figure is the more demanding and more informative of the two, because
    # ||Delta D|| is much smaller than ||D||, so it is reported whatever the training target is.
    total = float(
        torch.linalg.norm(D_hat - density_matrix) / torch.linalg.norm(density_matrix)
    )
    if promolecular is None:
        return total, None

    delta_truth = density_matrix - promolecular
    delta_norm = float(torch.linalg.norm(delta_truth))
    if delta_norm == 0.0:
        return total, None

    # the residual is the same matrix either way, only the normalizer differs
    return total, float(torch.linalg.norm(D_hat - density_matrix) / delta_norm)
