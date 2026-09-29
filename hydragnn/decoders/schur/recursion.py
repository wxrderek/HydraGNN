###########################################################################################
# The bordering recursion of the sequential Schur decoder
# Reference: references/schur_decoder.tex, Sections 3.4-3.5 and 6.4
# Author: Derek Wang
###########################################################################################

import torch


def bordering_recursion(z, delta, density_matrix=None, detach_every=None):
    '''build D_hat one orbital at a time by the bordering recursion of Section 3.5'''

    # At step k the decoder emits the latents (z_k, alpha_k) and we set
    #
    #   c_k = D_{k-1} z_k,    d_k = z_k^T c_k + delta_k,    delta_k = f(alpha_k) >= 0
    #
    # which is Equation 8. Theorem 2.1 says the bordered matrix is PSD for ANY z_k and any
    # delta_k >= 0, so no pseudoinverse and no eigendecomposition appear anywhere here.
    #
    # Indexing note, stated once: the reference is 1-based with z_k in R^{k-1}, so the Python
    # list z is 0-based with z[k] of length k. z[0] is empty and delta[0] = d_1.
    #
    # density_matrix is None for free running (Section 6.4), which feeds D_hat_{k-1} back into
    # the recursion, and is the ground truth D for teacher forcing, which feeds D_{k-1}. That
    # one substitution is the entire difference between the two routings.
    #
    # PSD caveat: under teacher forcing every individual step block is PSD by Theorem 2.1, but
    # the assembled D_hat is not the output of any single valid recursion, since its input side
    # is ground truth while its output side is predicted. Only the free-running D_hat carries
    # the PSD guarantee, and free running is what inference uses.
    #
    # Delta learning does not appear here: by Theorem 2.2 the recursion runs on the totals
    # whichever matrix the loss looks at, and the subtraction of D_pro happens in losses.py.
    n_basis = len(z)
    assert n_basis > 0
    assert delta.shape == (n_basis,)
    # These two are separate on purpose. f is softplus or exp, both non-negative for every
    # finite input, so a failure of "delta >= 0" is in practice always a NaN somewhere upstream
    # rather than a violation of Theorem 2.1, and reporting it as the latter sends the reader
    # looking in the wrong place.
    assert torch.all(torch.isfinite(delta)), (
        'f(alpha) is not finite; alpha or the parameters upstream of it already hold NaN or Inf'
    )
    assert torch.all(delta >= 0.0), 'f(alpha) produced a negative pivot, breaking Theorem 2.1'
    for k in range(n_basis):
        assert z[k].shape == (k,)
        assert z[k].dtype == delta.dtype and z[k].device == delta.device

    teacher_forcing = density_matrix is not None
    if teacher_forcing:
        assert density_matrix.shape == (n_basis, n_basis)
        assert density_matrix.dtype == delta.dtype
        assert density_matrix.device == delta.device

    D_hat = delta[0].reshape(1, 1) # (1, 1), this is D_hat_1 = d_1 of Section 5.2
    columns = [z[0]] # c_k per step, entry 0 is the empty column above d_1
    diagonals = [delta[0]] # d_k per step

    for k in range(1, n_basis):
        # the only line that distinguishes the two routings of Section 6.4
        D_prev = density_matrix[:k, :k] if teacher_forcing else D_hat # (k, k)

        c_k = D_prev @ z[k] # (k,)
        d_k = torch.dot(z[k], c_k) + delta[k] # scalar

        # D_hat is grown out of place. Preallocating (n_basis, n_basis) and assigning row and
        # column k looks cheaper but is wrong: the matmul above saves the view D_hat[:k, :k]
        # for backward, the assignment bumps the storage version counter, and backward then
        # raises. PyTorch tracks versions per storage, so writing only into the untouched
        # border does not help.
        top = torch.cat([D_hat, c_k.unsqueeze(1)], dim=1) # (k, k + 1)
        bottom = torch.cat([c_k.unsqueeze(0), d_k.view(1, 1)], dim=1) # (1, k + 1)
        D_hat = torch.cat([top, bottom], dim=0) # (k + 1, k + 1)

        columns.append(c_k)
        diagonals.append(d_k)

        # truncated backpropagation for the semi_sequential and per_step supervision of
        # Sections 6.2-6.3. Under teacher forcing D_hat is never fed back, so this changes
        # nothing there; under free running it is what keeps the retained graph flat.
        if detach_every is not None and (k + 1) % detach_every == 0:
            D_hat = D_hat.detach()

    assert D_hat.shape == (n_basis, n_basis)
    return D_hat, columns, torch.stack(diagonals) # (n_basis, n_basis), list, (n_basis,)
