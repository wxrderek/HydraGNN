'''
Diagnostic for the sequential Schur-complement decoder proposed in
references/schur_decoder.tex.

The decoder builds the 1-RDM one orbital at a time by bordering the leading
principal submatrix,

    c_k = D_{k-1} z_k,    d_k = z_k^T c_k + delta_k,    delta_k >= 0,

which is PSD by construction (Albert's theorem). This recursion is exactly the
unpivoted LDL^T factorization reparameterized: with L unit lower triangular and
Delta = diag(delta_k), it produces D = L Delta L^T, and the strictly lower part
of row k of L is l_k = L_{k-1}^T z_k.

This script factors ground-truth QMugs density matrices in that fixed AO order
and measures the three things that decide how the decoder heads must be built:

  1. the dynamic range of z, which the pair head has to emit,
  2. where the rank cutoff sits, since rank(D) = N_e / 2 exactly,
  3. how sharply the pivots delta_k decay to zero, which says what the
     nonnegativity map f must be able to represent.

The jittered factorization above answers these only up to the regularization.
Because jitter*I is diagonal it leaves c_k untouched, so the latent it produces
is exactly z_k(eps) = (D_{k-1} + eps I)^{-1} c_k, a Tikhonov path whose norms are
monotone LOWER bounds on the exact ||D_{k-1}^+ c_k||. The path cannot reach the
limit: below jitter_relative ~ 1e-7 the Cholesky fails outright. Five further
diagnostics therefore work from eigendecompositions instead:

  M1  discrete Picard analysis  -- is there a modest-norm truncated latent that
      reproduces c_k to useful accuracy? (the decisive measurement)
  M2  rollout stability         -- how much does teacher forcing buy over free
      running, at a given model accuracy?
  M3  identifiability budget    -- how much of z_k is unconstrained by the target?
  M4  gradient conditioning     -- is per-step supervision well conditioned in the
      directions carrying signal? (falls out of M1)
  M5  target mass profile       -- do the ill-conditioned late steps carry any
      appreciable share of ||D||_F?

Every term above is defined precisely in the companion document
logs/qmugs-schur-diagnostics/schur_decomp_test.md.

Nothing here trains or predicts anything; it only measures ground truth.
'''

import os
import sys
import json
import logging
import argparse

import numpy as np
import pandas as pd

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import psi4
    psi4.core.be_quiet()
except ImportError:
    print('[WARNING] psi4 package not detected')

# allow schur_decomp_test.py to be run as a standalone script from shell scripts
QMUGS_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
if QMUGS_EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, QMUGS_EXAMPLE_DIR)

from utils import promolecular_density_matrix


DEFAULT_RANDOM_STATE = 0

TITLE_FONTSIZE = 9
AXIS_LABEL_FONTSIZE = 8

# all factorizations run in float64 regardless of how the matrices are stored,
# so that "numerically zero pivot" is not a statement about float32 precision
WORKING_DTYPE = torch.float64

# relative eigenvalue cutoffs tau for the M1 truncation sweep, spanning from an
# aggressive truncation down past the float64 noise floor of eigh
PICARD_CUTOFFS = [10.0 ** -exponent for exponent in range(1, 13)]

# relative residuals rho at which the required latent norm is reported; 1e-2 and
# 1e-3 bracket any accuracy a trained model would plausibly reach on c_k
RESIDUAL_TARGETS = [1e-2, 1e-3]

# multiplicative latent perturbations for the M2 rollout, standing in for models
# of decreasing accuracy. Zero is included as the truncation-error baseline: the
# truncated latents do not reproduce c_k exactly, so without it an amplified
# perturbation cannot be told apart from the error the truncation already carries.
ROLLOUT_NOISE_LEVELS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]


# ----------------------------------------------------------------------------------------------------
# small shared helpers


def _save_figure(fig, output_path: str, dpi: int = 300):
    '''save a matplotlib figure as both PNG and PDF'''

    base_path, extension = os.path.splitext(output_path)
    output_base = base_path if extension.lower() in ['.png', '.pdf'] else output_path

    png_path = output_base + '.png'
    pdf_path = output_base + '.pdf'

    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    return [png_path, pdf_path]


def _quantile_band(curves, num_points: int):
    '''resample ragged per-molecule curves onto a common grid and return quantiles'''

    # every molecule has a different K, so curves are compared on the fractional
    # orbital index k / K rather than on k itself
    grid = np.linspace(0.0, 1.0, num_points) # (num_points,)
    resampled = []
    for curve in curves:
        curve = np.asarray(curve, dtype=np.float64)
        if curve.size < 2:
            continue
        source = np.linspace(0.0, 1.0, curve.size) # (len(curve),)
        resampled.append(np.interp(grid, source, curve))

    if len(resampled) == 0:
        raise ValueError('No curves with at least two points were provided')

    stacked = np.stack(resampled) # (n_curves, num_points)
    lower = np.nanquantile(stacked, 0.25, axis=0) # (num_points,)
    median = np.nanquantile(stacked, 0.50, axis=0) # (num_points,)
    upper = np.nanquantile(stacked, 0.75, axis=0) # (num_points,)

    return grid, lower, median, upper


# ----------------------------------------------------------------------------------------------------
# conformer selection


def density_matrix_sizes():
    '''load per-molecule density matrix sizes from the EDA output file'''

    eda_path = os.path.join(QMUGS_EXAMPLE_DIR, 'eda', 'densmat_eda.json')
    if not os.path.isfile(eda_path):
        raise FileNotFoundError(f'Density matrix EDA file not found: {eda_path}')

    with open(eda_path, 'r', encoding='utf-8') as f:
        densmat_eda = json.load(f)

    sizes = densmat_eda.get('sizes', None)
    if not isinstance(sizes, dict):
        raise ValueError(f"Density matrix EDA file {eda_path} must contain a dictionary under key 'sizes'")

    return sizes


def sample_conformers(
    download_config,
    num_size_bins: int,
    num_per_bin: int,
    seed: int,
):
    '''select conformers stratified across density matrix size K'''

    data_dir = download_config.get('data_dir', 'dataset')
    wfns_dir = os.path.join(data_dir, download_config.get('wfns_subdir', 'wfns'))
    tarball_assignment = pd.read_csv(os.path.join(data_dir, 'tarball_assignment.csv'))

    sizes = density_matrix_sizes()

    # map each molecule to the wfn directory extracted from its archive
    archive_by_id = dict(zip(tarball_assignment['chembl_id'], tarball_assignment['archive_name']))

    candidates = []
    for chembl_id, size in sizes.items():
        archive_name = archive_by_id.get(chembl_id, None)
        if archive_name is None:
            continue
        wfn_path = os.path.join(
            wfns_dir,
            archive_name.split('.')[0],
            chembl_id,
            'wfn_conf_00.npy',
        )
        candidates.append((chembl_id, int(size), wfn_path))

    if len(candidates) == 0:
        raise ValueError('No molecules found by cross-referencing the EDA sizes with tarball_assignment.csv')

    # the large-K tail is a small fraction of the dataset but is exactly where the
    # rank deficiency is most severe, so sampling is stratified rather than uniform
    all_sizes = np.array([size for _, size, _ in candidates], dtype=np.float64) # (n_candidates,)
    bin_edges = np.linspace(all_sizes.min(), all_sizes.max() + 1.0, num_size_bins + 1) # (num_size_bins + 1,)
    bin_index = np.digitize(all_sizes, bin_edges) - 1 # (n_candidates,)
    bin_index = np.clip(bin_index, 0, num_size_bins - 1)

    rng = np.random.default_rng(seed)
    selected = []
    for target_bin in range(num_size_bins):
        members = [candidates[i] for i in np.flatnonzero(bin_index == target_bin)]
        if len(members) == 0:
            logging.info(f'[INFO] Size bin {target_bin} ({bin_edges[target_bin]:.0f}-{bin_edges[target_bin + 1]:.0f}) is empty')
            continue

        # sample more than needed, since some wfn archives may not be extracted locally
        order = rng.permutation(len(members))
        taken = 0
        for position in order:
            if taken >= num_per_bin:
                break
            chembl_id, size, wfn_path = members[position]
            if not os.path.isfile(wfn_path):
                continue
            selected.append({'chembl_id': chembl_id, 'eda_size': size, 'wfn_path': wfn_path})
            taken += 1

        if taken < num_per_bin:
            logging.info(
                f'[INFO] Size bin {target_bin} ({bin_edges[target_bin]:.0f}-{bin_edges[target_bin + 1]:.0f}) '
                f'yielded {taken} of {num_per_bin} requested conformers'
            )

    if len(selected) == 0:
        raise ValueError('No conformers with an existing wfn file were selected')

    return selected


# ----------------------------------------------------------------------------------------------------
# factorization


def unpivoted_ldl(matrix, jitter: float):
    '''unpivoted LDL^T of a symmetric PSD matrix, in the given fixed ordering'''

    # The decoder recursion corresponds to the UNPIVOTED factorization in the fixed
    # AO order. torch.linalg.ldl_factor and scipy.linalg.ldl both apply Bunch-Kaufman
    # pivoting and so factor a permuted matrix; torch.linalg.cholesky requires strict
    # positive definiteness, which these rank-deficient matrices do not have. A
    # jittered Cholesky gives the right factorization in one LAPACK call:
    #
    #   D + jitter*I = Ltil Ltil^T,   Ltil = L diag(sqrt(delta))
    #
    # so dividing column j of Ltil by its diagonal entry recovers the unit lower
    # triangular L, and the squared diagonal gives the pivots.
    n_basis = int(matrix.shape[0])
    assert matrix.shape == (n_basis, n_basis)
    assert matrix.dtype == WORKING_DTYPE

    jittered = matrix + jitter * torch.eye(n_basis, dtype=WORKING_DTYPE) # (n_basis, n_basis)
    cholesky_factor, info = torch.linalg.cholesky_ex(jittered)
    if int(info.item()) != 0:
        raise ValueError(
            f'Jittered Cholesky failed at leading minor {int(info.item())} with jitter {jitter:.3e}; '
            'increase --jitter_relative'
        )

    diagonal = torch.diagonal(cholesky_factor) # (n_basis,)
    assert torch.all(diagonal > 0.0)

    unit_lower = cholesky_factor / diagonal.unsqueeze(0) # (n_basis, n_basis), column scaling
    pivots = diagonal ** 2 # (n_basis,)

    return unit_lower, pivots


def z_rows_from_unit_lower(unit_lower):
    '''return the matrix whose row k is the decoder latent z_k'''

    # z_k is defined by L_{k-1}^T z_k = l_k. Bordering the inverse of a unit lower
    # triangular matrix,
    #
    #   L_k = [[L_{k-1}, 0], [l_k^T, 1]]  =>  L_k^{-1} = [[L_{k-1}^{-1}, 0], [-l_k^T L_{k-1}^{-1}, 1]]
    #
    # so the strictly lower part of row k of L^{-1} is exactly -z_k^T. All K latent
    # vectors therefore come from one triangular inverse, with no per-step solve and
    # no pseudoinverse anywhere.
    n_basis = int(unit_lower.shape[0])
    assert unit_lower.shape == (n_basis, n_basis)

    identity = torch.eye(n_basis, dtype=unit_lower.dtype) # (n_basis, n_basis)
    inverse = torch.linalg.solve_triangular(
        unit_lower,
        identity,
        upper=False,
        unitriangular=True,
    ) # (n_basis, n_basis)

    z_rows = -torch.tril(inverse, diagonal=-1) # (n_basis, n_basis), row k is z_k
    assert z_rows.shape == (n_basis, n_basis)

    return z_rows


def factorization_error(unit_lower, pivots, matrix, jitter: float):
    '''relative Frobenius error of the reconstruction L Delta L^T against D'''

    # the factorization is of D + jitter*I, so the residual against D should sit at
    # the jitter level rather than at machine precision; the expected size is
    # returned alongside it so the two can be compared directly
    reconstruction = (unit_lower * pivots.unsqueeze(0)) @ unit_lower.transpose(0, 1) # (n_basis, n_basis)
    denominator = float(torch.linalg.norm(matrix))
    expected = jitter * float(matrix.shape[0]) ** 0.5 / denominator

    return float(torch.linalg.norm(reconstruction - matrix)) / denominator, expected


def recursion_error(z_rows, pivots, matrix):
    '''relative error of the decoder recursion itself, run from the exact latents'''

    # c_k[i] = (Z D)[k, i] for i < k, because row k of Z vanishes at and above the
    # diagonal, so the matvec c_k = D_{k-1} z_k over the leading block is the same
    # as one row of Z D. This checks the claim the decoder rests on, in one shot.
    column_predictions = z_rows @ matrix # (n_basis, n_basis)

    off_diagonal_error = torch.linalg.norm(
        torch.tril(column_predictions - matrix, diagonal=-1)
    ) / torch.linalg.norm(torch.tril(matrix, diagonal=-1))

    # d_k = z_k^T c_k + delta_k
    diagonal_predictions = (z_rows * column_predictions).sum(dim=1) + pivots # (n_basis,)
    diagonal_error = torch.linalg.norm(
        diagonal_predictions - torch.diagonal(matrix)
    ) / torch.linalg.norm(torch.diagonal(matrix))

    return float(off_diagonal_error), float(diagonal_error)


def explicit_recursion_error(z_rows, pivots, matrix):
    '''run the decoder recursion literally, one orbital at a time, and compare'''

    # The vectorized check above is the one used everywhere; this loop exists so the
    # bordering construction is confirmed in the form it is actually written in the
    # document, not only in its collapsed form.
    n_basis = int(matrix.shape[0])
    reconstruction = torch.zeros_like(matrix) # (n_basis, n_basis)
    reconstruction[0, 0] = pivots[0]

    for k in range(1, n_basis):
        previous = reconstruction[:k, :k] # (k, k)
        z_k = z_rows[k, :k] # (k,)
        c_k = previous @ z_k # (k,)
        d_k = torch.dot(z_k, c_k) + pivots[k]
        reconstruction[:k, k] = c_k
        reconstruction[k, :k] = c_k
        reconstruction[k, k] = d_k

    return float(torch.linalg.norm(reconstruction - matrix) / torch.linalg.norm(matrix))


# ----------------------------------------------------------------------------------------------------
# M1, M3, M4: truncated pseudoinverse of a single bordering step


def picard_analysis(matrix, k_index: int, cutoffs, residual_targets):
    '''truncated-pseudoinverse analysis of the bordering step at index k_index'''

    # The step solves D_{k-1} z_k = c_k. Expanding c_k in the eigenbasis of D_{k-1}
    # turns the solve into a sum of terms b_i / lambda_i, so "is there a modest-norm
    # latent" becomes "does |b_i| decay at least as fast as lambda_i" -- the discrete
    # Picard condition. Unlike the jittered factorization this separates genuinely zero
    # eigenvalues from round-off explicitly, through the cutoff tau, which is why it can
    # reach the exact latent that the Tikhonov path only bounds from below.
    leading = matrix[:k_index, :k_index]  # (k_index, k_index), D_{k-1}
    c_vector = matrix[:k_index, k_index]  # (k_index,), c_k
    d_scalar = float(matrix[k_index, k_index])
    assert leading.shape == (k_index, k_index)
    assert c_vector.shape == (k_index,)

    eigenvalues, eigenvectors = torch.linalg.eigh(leading)  # ascending
    # descending order, so that "retain the top r modes" is a prefix of the arrays below
    eigenvalues = torch.flip(eigenvalues, dims=[0]) # (k_index,)
    eigenvectors = torch.flip(eigenvectors, dims=[1]) # (k_index, k_index)
    coefficients = eigenvectors.transpose(0, 1) @ c_vector # (k_index,), b_i = u_i^T c_k

    max_eigenvalue = float(eigenvalues[0])
    c_norm = float(torch.linalg.norm(c_vector))
    if max_eigenvalue <= 0.0 or c_norm == 0.0:
        return None

    # eigh returns small negative values where the true eigenvalue is zero; those modes
    # carry no usable information and are dropped from every sum
    positive = eigenvalues > 0.0 # (k_index,)
    safe = torch.where(positive, eigenvalues, torch.ones_like(eigenvalues)) # (k_index,)
    zeros = torch.zeros_like(eigenvalues)

    # prefix sums give every truncation level at once, with no loop over tau
    z_cumulative = torch.cumsum(torch.where(positive, coefficients ** 2 / safe ** 2, zeros), dim=0)
    zc_cumulative = torch.cumsum(torch.where(positive, coefficients ** 2 / safe, zeros), dim=0)
    energy_cumulative = torch.cumsum(coefficients ** 2, dim=0)
    total_energy = float(energy_cumulative[-1])

    # rho(r), the fraction of c_k unreproduced when the top r modes are retained. Since
    # D_{k-1} z(tau) = sum_{lambda_i > tau} b_i u_i, the residual is the energy in the
    # DROPPED modes. Retaining a mode with lambda_i <= 0 adds nothing to z, so its b_i
    # must still count as unreproduced; the curve is therefore frozen past the last
    # positive eigenvalue rather than allowed to keep falling.
    num_positive = int(positive.sum())
    residual_curve = torch.sqrt(
        torch.clamp(total_energy - energy_cumulative, min=0.0)
    ) / c_norm # (k_index,), index r - 1
    residual_curve[num_positive:] = residual_curve[num_positive - 1]

    def reading(requested: int):
        '''the reported quantities at a given number of retained modes'''
        # modes past the last positive eigenvalue are not retainable
        num_retained = min(max(requested, 0), num_positive)
        if num_retained <= 0:
            return {
                'num_retained': 0,
                'z_norm': 0.0,
                'residual': 1.0,
                'delta': d_scalar,
                'min_retained_eigenvalue': 1.0,
            }
        # Lemma (schur_decomp_test.md 6.3): truncation only ever raises the pivot the
        # alpha head must supply, since z(tau)^T c_k is a sub-sum of nonnegative terms
        return {
            'num_retained': num_retained,
            'z_norm': float(torch.sqrt(z_cumulative[num_retained - 1])),
            'residual': float(residual_curve[num_retained - 1]),
            'delta': d_scalar - float(zc_cumulative[num_retained - 1]),
            # how far down the spectrum this truncation reaches, relative to lambda_max.
            # Terms b_i^2 / lambda_i are inverted eigenvalues, so this is the amplification
            # the smallest retained mode contributes, and the quantity that decides whether
            # the lemma survives in floating point.
            'min_retained_eigenvalue': float(eigenvalues[num_retained - 1]) / max_eigenvalue,
        }

    rows = []
    for cutoff in cutoffs:
        entry = reading(int((eigenvalues > cutoff * max_eigenvalue).sum()))
        rows.append({
            'k_index': k_index,
            'cutoff': cutoff,
            'num_retained': entry['num_retained'],
            # M3: the unconstrained directions of z_k at this cutoff
            'null_fraction': 1.0 - entry['num_retained'] / k_index,
            'z_norm': entry['z_norm'],
            'residual': entry['residual'],
            # M4: the share of ||c_k||^2 living above the cutoff, which is what a
            # gradient on the per-step loss can actually see
            'mass_above_cutoff': 1.0 - entry['residual'] ** 2,
            'delta': entry['delta'],
            'delta_nonnegative': bool(entry['delta'] >= 0.0),
        })

    # the headline numbers: how large a latent is needed to hit a given accuracy on c_k
    targets = {}
    for target in residual_targets:
        meets = residual_curve <= target # (k_index,)
        # the fewest modes meeting the accuracy bar, i.e. the smallest latent that clears
        # it; if the bar is unreachable, take everything the spectrum has to offer
        requested = int(torch.argmax(meets.to(torch.int32))) + 1 if bool(meets.any()) else num_positive
        entry = reading(requested)
        # This path uses the same stopping rule as truncated_latents, and unlike the
        # cutoff rows above it is not floored, so it is the one that can invert an
        # eigenvalue at the noise floor. It is checked for the same reason.
        entry['target_met'] = bool(entry['residual'] <= target)
        entry['delta_nonnegative'] = bool(entry['delta'] >= 0.0)
        targets[target] = entry

    # the Picard plot itself, subsampled to keep the stored arrays small
    ratios = (coefficients[positive].abs() / eigenvalues[positive]).numpy()
    stride = max(1, ratios.size // 200)

    return {
        'k_index': k_index,
        'dimension': k_index,
        'max_eigenvalue': max_eigenvalue,
        'c_norm': c_norm,
        'd_scalar': d_scalar,
        'rows': rows,
        'targets': targets,
        'picard_ratios': ratios[::stride],
        'eigenvalue_profile': (eigenvalues[positive].numpy() / max_eigenvalue)[::stride],
    }


def sample_step_indices(n_basis: int, num_samples: int):
    '''bordering steps to analyze, geometrically spaced so small k stays resolved'''

    # cost is O(k^3) per step, so a geometric grid buys coverage of the whole sequence
    # for roughly twice the price of the single largest step
    if n_basis <= num_samples + 1:
        return list(range(1, n_basis))

    spaced = np.geomspace(1.0, float(n_basis - 1), num_samples)
    return sorted(set(int(round(value)) for value in spaced if 1 <= round(value) <= n_basis - 1))


# ----------------------------------------------------------------------------------------------------
# M2: rollout stability


def truncated_latents(matrix, residual_target: float, rank_tolerance: float):
    '''exact truncated latents (z_k, delta_k) for every bordering step'''

    # One eigendecomposition per step, so O(K^4) overall; only affordable for the small
    # molecules, which is where the rollout is run. Truncation is at the coarsest level
    # that still reproduces c_k to residual_target, i.e. the smallest latent that meets
    # the accuracy bar, which is exactly the latent a well-behaved model would emit.
    #
    # Retention is additionally floored at rank_tolerance * lambda_max(D_{k-1}), the same
    # definition of "numerically zero eigenvalue" spectral_summary uses. Without the floor
    # the accuracy bar can force a mode whose eigenvalue is at the float64 noise floor of
    # eigh, where the term b_i^2 / lambda_i is a fabricated inversion of round-off and the
    # lemma of schur_decomp_test.md 6.3 fails numerically even though it holds exactly.
    n_basis = int(matrix.shape[0])
    z_rows = torch.zeros((n_basis, n_basis), dtype=matrix.dtype) # (n_basis, n_basis)
    deltas = torch.zeros(n_basis, dtype=matrix.dtype) # (n_basis,)
    residuals = torch.ones(n_basis, dtype=matrix.dtype) # (n_basis,), achieved rho per step
    min_eigenvalues = torch.ones(n_basis, dtype=matrix.dtype) # (n_basis,), relative to lambda_max
    deltas[0] = matrix[0, 0]
    residuals[0] = 0.0

    for k_index in range(1, n_basis):
        leading = matrix[:k_index, :k_index] # (k_index, k_index)
        c_vector = matrix[:k_index, k_index] # (k_index,)

        eigenvalues, eigenvectors = torch.linalg.eigh(leading)
        eigenvalues = torch.flip(eigenvalues, dims=[0]) # (k_index,)
        eigenvectors = torch.flip(eigenvectors, dims=[1]) # (k_index, k_index)
        coefficients = eigenvectors.transpose(0, 1) @ c_vector # (k_index,)

        c_norm = float(torch.linalg.norm(c_vector))
        total_energy = float((coefficients ** 2).sum())
        max_eigenvalue = float(eigenvalues[0])
        if c_norm == 0.0 or max_eigenvalue <= 0.0:
            deltas[k_index] = matrix[k_index, k_index]
            residuals[k_index] = 0.0 if c_norm == 0.0 else 1.0
            continue

        energy_cumulative = torch.cumsum(coefficients ** 2, dim=0) # (k_index,)
        residual_curve = torch.sqrt(
            torch.clamp(total_energy - energy_cumulative, min=0.0)
        ) / c_norm # (k_index,)
        meets = residual_curve <= residual_target
        num_retained = int(torch.argmax(meets.to(torch.int32))) + 1 if bool(meets.any()) else k_index

        # modes below the resolvable floor carry no usable information and are dropped
        num_resolvable = int((eigenvalues > rank_tolerance * max_eigenvalue).sum())
        num_retained = min(num_retained, num_resolvable)
        if num_retained <= 0:
            deltas[k_index] = matrix[k_index, k_index]
            continue

        retained = eigenvectors[:, :num_retained] # (k_index, num_retained)
        weights = coefficients[:num_retained] / eigenvalues[:num_retained] # (num_retained,)
        z_vector = retained @ weights # (k_index,)

        z_rows[k_index, :k_index] = z_vector
        # nonnegative by the lemma in schur_decomp_test.md 6.3, up to round-off
        deltas[k_index] = matrix[k_index, k_index] - torch.dot(z_vector, c_vector)
        # the bar may now be unreachable at this step, so what was actually achieved is
        # recorded rather than assumed
        residuals[k_index] = residual_curve[num_retained - 1]
        min_eigenvalues[k_index] = float(eigenvalues[num_retained - 1]) / max_eigenvalue

    return z_rows, deltas, residuals, min_eigenvalues


def rollout_errors(matrix, z_rows, deltas, noise_level: float, generator):
    '''per-step reconstruction error under teacher forcing and under free running'''

    # Teacher forcing feeds the ground-truth D_{k-1} into c_k = D_{k-1} z_k; free running
    # feeds back the model's own reconstruction. The difference between the two is
    # dc_k = dD z_k, so the error committed at earlier steps is amplified by ||z_k||.
    # This measures that amplification directly.
    n_basis = int(matrix.shape[0])
    assert z_rows.shape == (n_basis, n_basis)
    assert deltas.shape == (n_basis,)

    perturbed = z_rows * (
        1.0 + noise_level * torch.randn(z_rows.shape, dtype=z_rows.dtype, generator=generator)
    ) # (n_basis, n_basis)

    forced = torch.zeros_like(matrix) # (n_basis, n_basis)
    free = torch.zeros_like(matrix) # (n_basis, n_basis)
    forced[0, 0] = deltas[0]
    free[0, 0] = deltas[0]

    rows = []
    for k_index in range(1, n_basis):
        z_vector = perturbed[k_index, :k_index] # (k_index,)
        c_true = matrix[:k_index, k_index] # (k_index,)
        c_norm = float(torch.linalg.norm(c_true))

        # teacher forced: exact leading block on the input side, no feedback
        c_forced = matrix[:k_index, :k_index] @ z_vector # (k_index,)
        forced[:k_index, k_index] = c_forced
        forced[k_index, :k_index] = c_forced
        forced[k_index, k_index] = torch.dot(z_vector, c_forced) + deltas[k_index]

        # free running: the model's own reconstruction on the input side
        c_free = free[:k_index, :k_index] @ z_vector # (k_index,)
        free[:k_index, k_index] = c_free
        free[k_index, :k_index] = c_free
        free[k_index, k_index] = torch.dot(z_vector, c_free) + deltas[k_index]

        if c_norm > 0.0:
            rows.append({
                'k_index': k_index,
                'noise_level': noise_level,
                'forced_step_error': float(torch.linalg.norm(c_forced - c_true)) / c_norm,
                'free_step_error': float(torch.linalg.norm(c_free - c_true)) / c_norm,
                'forced_block_error': float(
                    torch.linalg.norm(forced[:k_index + 1, :k_index + 1] - matrix[:k_index + 1, :k_index + 1])
                    / torch.linalg.norm(matrix[:k_index + 1, :k_index + 1])
                ),
                'free_block_error': float(
                    torch.linalg.norm(free[:k_index + 1, :k_index + 1] - matrix[:k_index + 1, :k_index + 1])
                    / torch.linalg.norm(matrix[:k_index + 1, :k_index + 1])
                ),
            })

    return rows


# ----------------------------------------------------------------------------------------------------
# M5: where the target mass lives


def target_mass_profile(matrix):
    '''Frobenius mass of the target carried by each bordering step'''

    # c_k is the strictly lower part of row k, by symmetry, and it enters ||D||_F^2 twice
    # (once above and once below the diagonal) while d_k enters once
    n_basis = int(matrix.shape[0])
    strictly_lower = torch.tril(matrix, diagonal=-1) # (n_basis, n_basis)
    c_squared = (strictly_lower ** 2).sum(dim=1) # (n_basis,), ||c_k||^2
    d_squared = torch.diagonal(matrix) ** 2 # (n_basis,)

    total = float(2.0 * c_squared.sum() + d_squared.sum())
    assert abs(total - float(torch.linalg.norm(matrix)) ** 2) <= 1e-8 * total

    return c_squared.numpy(), d_squared.numpy(), total


# ----------------------------------------------------------------------------------------------------
# Lowdin orthogonalization


def lowdin_transform(matrix, overlap):
    '''symmetric orthogonalization P = S^{1/2} D S^{1/2}'''

    # P is the density matrix in the orthonormal basis obtained by symmetric
    # orthogonalization of the AOs. There tr(P) = N_e literally and the eigenvalues of P
    # are the occupation numbers in [0, 2]. Note rank(P) = rank(D), since S^{1/2} is
    # invertible: orthogonalization cannot remove the rank deficiency, only the
    # ill-conditioning that the overlap itself contributes.
    n_basis = int(matrix.shape[0])
    assert overlap.shape == (n_basis, n_basis)

    eigenvalues, eigenvectors = torch.linalg.eigh(overlap) # (n_basis,), (n_basis, n_basis)
    assert torch.all(eigenvalues > 0.0), 'overlap matrix is not positive definite'

    sqrt_overlap = (eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.transpose(0, 1)
    transformed = sqrt_overlap @ matrix @ sqrt_overlap # (n_basis, n_basis)

    return 0.5 * (transformed + transformed.transpose(0, 1))


# ----------------------------------------------------------------------------------------------------
# per-conformer analysis


def load_wavefunction_matrices(wfn_path: str, compute_promolecular: bool):
    '''load the total AO density matrix and the overlap for one conformer'''

    psi4_wfn = psi4.core.Wavefunction.from_file(wfn_path)

    density_matrix = np.asarray(psi4_wfn.Da().np, dtype=np.float64) + np.asarray(psi4_wfn.Db().np, dtype=np.float64)
    n_basis = int(density_matrix.shape[0])
    assert density_matrix.shape == (n_basis, n_basis)

    overlap = np.asarray(
        psi4.core.MintsHelper(psi4_wfn.basisset()).ao_overlap().to_array(),
        dtype=np.float64,
    ) # (n_basis, n_basis)
    assert overlap.shape == (n_basis, n_basis)

    promolecular = None
    if compute_promolecular:
        promolecular = np.asarray(promolecular_density_matrix(psi4_wfn), dtype=np.float64) # (n_basis, n_basis)
        assert promolecular.shape == (n_basis, n_basis)

    psi4_mol = psi4_wfn.molecule()
    charge = int(psi4_mol.molecular_charge())
    multiplicity = int(psi4_mol.multiplicity())

    # the occupation counts come from the stored wavefunction when it exposes them,
    # and otherwise from the molecule, so the rank prediction never depends on which
    # Psi4 build wrote the file
    if hasattr(psi4_wfn, 'nalpha') and hasattr(psi4_wfn, 'nbeta'):
        n_alpha = int(psi4_wfn.nalpha())
        n_beta = int(psi4_wfn.nbeta())
    else:
        n_electrons = int(sum(psi4_mol.ftrue_atomic_number(i) for i in range(psi4_mol.natom()))) - charge
        n_beta = (n_electrons - (multiplicity - 1)) // 2
        n_alpha = n_electrons - n_beta

    record = {
        'n_basis': n_basis,
        'n_atoms': int(psi4_mol.natom()),
        'charge': charge,
        'multiplicity': multiplicity,
        'n_alpha': n_alpha,
        'n_beta': n_beta,
    }

    return density_matrix, overlap, promolecular, record


def spectral_summary(matrix, overlap, rank_tolerance: float):
    '''rank, conditioning and electron count from the exact spectra'''

    eigenvalues = torch.linalg.eigvalsh(matrix) # (n_basis,)
    max_eigenvalue = float(eigenvalues[-1])
    above = eigenvalues[eigenvalues > rank_tolerance * max_eigenvalue] # (rank,)

    overlap_eigenvalues = torch.linalg.eigvalsh(overlap) # (n_basis,)

    return {
        'spectral_rank': int(above.numel()),
        'max_eigenvalue': max_eigenvalue,
        'min_eigenvalue': float(eigenvalues[0]),
        'min_nonzero_eigenvalue': float(above[0]) if above.numel() > 0 else float('nan'),
        # the condition number restricted to the range is what governs how large the
        # exact z_k must become, since z_k = D_{k-1}^+ c_k
        'range_condition_number': float(above[-1] / above[0]) if above.numel() > 0 else float('nan'),
        'overlap_condition_number': float(overlap_eigenvalues[-1] / overlap_eigenvalues[0]),
        'overlap_min_eigenvalue': float(overlap_eigenvalues[0]),
        # tr(SD) is the electron count in a non-orthogonal AO basis; tr(D) is not
        'trace_SD': float(torch.sum(overlap * matrix)),
        'trace_D': float(torch.trace(matrix)),
    }


def analyze_matrix(
    matrix,
    jitter_relative: float,
    pivot_floor_factor: float,
    num_entry_samples: int,
    rng,
):
    '''factor one matrix and record everything the decoder design depends on'''

    n_basis = int(matrix.shape[0])

    # jitter is set relative to the average diagonal entry so it does not depend on
    # the overall scale of the matrix
    mean_diagonal = float(torch.mean(torch.diagonal(matrix)))
    jitter = jitter_relative * abs(mean_diagonal)

    unit_lower, pivots = unpivoted_ldl(matrix, jitter=jitter)
    z_rows = z_rows_from_unit_lower(unit_lower)

    # pivots that have collapsed onto the jitter floor are the numerically zero ones
    pivot_floor = pivot_floor_factor * jitter
    above_floor = pivots > pivot_floor # (n_basis,)

    # per-step norms of the two competing latent parameterizations
    z_norms = torch.linalg.norm(z_rows, dim=1) # (n_basis,)
    l_norms = torch.linalg.norm(torch.tril(unit_lower, diagonal=-1), dim=1) # (n_basis,)

    # storing every entry would be K^2 / 2 numbers per molecule, so the per-entry
    # distributions are subsampled from the strictly lower triangle
    row_index, column_index = torch.tril_indices(n_basis, n_basis, offset=-1)
    num_entries = int(row_index.numel())
    if num_entries > num_entry_samples:
        chosen = torch.from_numpy(
            rng.choice(num_entries, size=num_entry_samples, replace=False)
        )
        row_index = row_index[chosen]
        column_index = column_index[chosen]

    off_diagonal_error, diagonal_error = recursion_error(z_rows, pivots, matrix)
    reconstruction_error, expected_error = factorization_error(unit_lower, pivots, matrix, jitter=jitter)

    first_floor_index = int(torch.argmax((~above_floor).to(torch.int32))) if bool((~above_floor).any()) else n_basis

    return {
        'scalars': {
            'jitter': jitter,
            'pivot_floor': pivot_floor,
            'num_pivots_above_floor': int(above_floor.sum()),
            'first_pivot_at_floor': first_floor_index,
            'max_pivot': float(pivots.max()),
            'min_pivot': float(pivots.min()),
            'max_z_norm': float(z_norms.max()),
            'median_z_norm': float(torch.median(z_norms)),
            'max_l_norm': float(l_norms.max()),
            'median_l_norm': float(torch.median(l_norms)),
            'max_abs_z_entry': float(z_rows.abs().max()),
            'max_abs_l_entry': float(torch.tril(unit_lower, diagonal=-1).abs().max()),
            'factorization_error': reconstruction_error,
            'expected_factorization_error': expected_error,
            'recursion_off_diagonal_error': off_diagonal_error,
            'recursion_diagonal_error': diagonal_error,
        },
        'curves': {
            'pivots': pivots.numpy(),
            'z_norms': z_norms.numpy(),
            'l_norms': l_norms.numpy(),
        },
        'entries': {
            'z': z_rows[row_index, column_index].numpy(),
            'l': unit_lower[row_index, column_index].numpy(),
        },
    }


# ----------------------------------------------------------------------------------------------------
# plots


def plot_pivot_decay(curves, output_path: str, title: str):
    '''plot the pivot decay profile against fractional orbital index'''

    fig, axis = plt.subplots(figsize=(6.0, 4.0))

    for curve in curves:
        fraction = np.linspace(0.0, 1.0, curve.size) # (n_basis,)
        axis.semilogy(fraction, np.maximum(curve, 1e-300), color='C0', alpha=0.15, linewidth=0.8)

    grid, lower, median, upper = _quantile_band(curves, num_points=400)
    axis.semilogy(grid, median, color='C3', linewidth=1.8, label='median')
    axis.fill_between(grid, lower, upper, color='C3', alpha=0.25, label='interquartile range')

    axis.set_xlabel('fractional orbital index $k / K$', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_ylabel(r'pivot $\delta_k$', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_title(title, fontsize=TITLE_FONTSIZE)
    axis.legend(fontsize=AXIS_LABEL_FONTSIZE)
    axis.grid(True, alpha=0.3)

    return _save_figure(fig, output_path)


def plot_latent_norms(records, output_path: str):
    '''compare the per-step norms of the two latent parameterizations'''

    fig, axis = plt.subplots(figsize=(6.0, 4.0))

    for name, color, label in [
        ('z_norms', 'C0', r'$\|z_k\|$  (Schur parameterization)'),
        ('l_norms', 'C1', r'$\|\ell_k\|$  ($L \Delta L^T$ parameterization)'),
    ]:
        # step 1 has no latent vector at all, so its norm is exactly zero; including it
        # on a log axis drags the lower limit to the clamp value and flattens everything
        curves = [record['curves'][name][1:] for record in records]
        grid, lower, median, upper = _quantile_band(curves, num_points=400)
        axis.semilogy(grid, np.maximum(median, 1e-300), color=color, linewidth=1.8, label=label)
        axis.fill_between(grid, np.maximum(lower, 1e-300), np.maximum(upper, 1e-300), color=color, alpha=0.25)

    axis.set_xlabel('fractional orbital index $k / K$', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_ylabel('latent norm', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_title('Magnitude the pair head must emit, median and interquartile range', fontsize=TITLE_FONTSIZE)
    axis.legend(fontsize=AXIS_LABEL_FONTSIZE)
    axis.grid(True, alpha=0.3)

    return _save_figure(fig, output_path)


def plot_norm_ratio(records, output_path: str):
    '''plot the ratio of the two latent norms against fractional orbital index'''

    fig, axis = plt.subplots(figsize=(6.0, 4.0))

    ratios = []
    for record in records:
        z_norms = record['curves']['z_norms']
        l_norms = record['curves']['l_norms']
        # step 1 has no latent vector at all, so the ratio starts at step 2
        ratios.append(z_norms[1:] / np.maximum(l_norms[1:], 1e-300))

    grid, lower, median, upper = _quantile_band(ratios, num_points=400)
    axis.semilogy(grid, np.maximum(median, 1e-300), color='C2', linewidth=1.8, label='median')
    axis.fill_between(grid, np.maximum(lower, 1e-300), np.maximum(upper, 1e-300), color='C2', alpha=0.25, label='interquartile range')
    axis.axhline(1.0, color='k', linestyle='--', linewidth=0.8)

    axis.set_xlabel('fractional orbital index $k / K$', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_ylabel(r'$\|z_k\| / \|\ell_k\|$', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_title(r'Cost of the Schur parameterization relative to $L \Delta L^T$', fontsize=TITLE_FONTSIZE)
    axis.legend(fontsize=AXIS_LABEL_FONTSIZE)
    axis.grid(True, alpha=0.3)

    return _save_figure(fig, output_path)


def plot_entry_histograms(records, output_path: str):
    '''plot the distribution of individual latent entries'''

    fig, axis = plt.subplots(figsize=(6.0, 4.0))

    for name, color, label in [
        ('z', 'C0', r'$z_{k,i}$'),
        ('l', 'C1', r'$\ell_{k,i}$'),
    ]:
        entries = np.concatenate([record['entries'][name] for record in records])
        magnitudes = np.abs(entries)
        magnitudes = magnitudes[magnitudes > 0.0]
        axis.hist(
            np.log10(magnitudes),
            bins=120,
            histtype='step',
            color=color,
            linewidth=1.5,
            density=True,
            label=label,
        )

    axis.set_xlabel(r'$\log_{10} |\mathrm{entry}|$', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_ylabel('density', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_title('Distribution of individual latent entries, pooled over conformers', fontsize=TITLE_FONTSIZE)
    axis.legend(fontsize=AXIS_LABEL_FONTSIZE)
    axis.grid(True, alpha=0.3)

    return _save_figure(fig, output_path)


def plot_rank_cutoff(summary_frame, output_path: str):
    '''check the counted rank against the closed-shell prediction N_e / 2'''

    fig, axis = plt.subplots(figsize=(6.0, 4.0))

    predicted = summary_frame['n_alpha'].to_numpy(dtype=np.float64) # (n_conformers,)
    counted = summary_frame['num_pivots_above_floor'].to_numpy(dtype=np.float64) # (n_conformers,)
    n_basis = summary_frame['n_basis'].to_numpy(dtype=np.float64) # (n_conformers,)

    scatter = axis.scatter(predicted, counted, c=n_basis, cmap='viridis', s=18)
    limits = [0.0, max(predicted.max(), counted.max()) * 1.05]
    axis.plot(limits, limits, 'k--', linewidth=0.8, label='$y = x$')

    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label('$K$', fontsize=AXIS_LABEL_FONTSIZE)

    axis.set_xlabel(r'$N_e / 2$ (number of occupied orbitals)', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_ylabel('pivots above the jitter floor', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_title('Counted rank against the closed-shell prediction', fontsize=TITLE_FONTSIZE)
    axis.legend(fontsize=AXIS_LABEL_FONTSIZE)
    axis.grid(True, alpha=0.3)

    return _save_figure(fig, output_path)


def plot_z_range_against_overlap(summary_frame, output_path: str):
    '''test whether z's dynamic range tracks basis near-linear-dependence'''

    fig, axis = plt.subplots(figsize=(6.0, 4.0))

    condition = summary_frame['overlap_condition_number'].to_numpy(dtype=np.float64) # (n_conformers,)
    max_z = summary_frame['max_z_norm'].to_numpy(dtype=np.float64) # (n_conformers,)
    n_basis = summary_frame['n_basis'].to_numpy(dtype=np.float64) # (n_conformers,)

    scatter = axis.scatter(condition, max_z, c=n_basis, cmap='viridis', s=18)
    axis.set_xscale('log')
    axis.set_yscale('log')

    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label('$K$', fontsize=AXIS_LABEL_FONTSIZE)

    axis.set_xlabel(r'$\kappa(S)$, AO overlap condition number', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_ylabel(r'$\max_k \|z_k\|$', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_title('Latent magnitude against basis near-linear-dependence', fontsize=TITLE_FONTSIZE)
    axis.grid(True, alpha=0.3, which='both')

    return _save_figure(fig, output_path)


def plot_jitter_sweep(sweep_frame, output_path: str):
    '''confirm the rank reading and the latent magnitude are not jitter artifacts'''

    fig, (rank_axis, norm_axis) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    for chembl_id, group in sweep_frame.groupby('chembl_id'):
        group = group.sort_values('jitter_relative')
        rank_axis.semilogx(group['jitter_relative'], group['num_pivots_above_floor'], marker='o', markersize=3, linewidth=1.0)
        norm_axis.loglog(group['jitter_relative'], group['max_z_norm'], marker='o', markersize=3, linewidth=1.0, label=chembl_id)
        norm_axis.loglog(
            group['jitter_relative'], group['max_l_norm'],
            marker='s', markersize=3, linewidth=1.0, linestyle='--', alpha=0.6,
        )

    rank_axis.set_xlabel('relative jitter', fontsize=AXIS_LABEL_FONTSIZE)
    rank_axis.set_ylabel('pivots above the jitter floor', fontsize=AXIS_LABEL_FONTSIZE)
    rank_axis.set_title('Rank reading against jitter', fontsize=TITLE_FONTSIZE)
    rank_axis.grid(True, alpha=0.3, which='both')

    norm_axis.set_xlabel('relative jitter', fontsize=AXIS_LABEL_FONTSIZE)
    norm_axis.set_ylabel('latent norm', fontsize=AXIS_LABEL_FONTSIZE)
    # This is a Tikhonov path: z_k(eps) = (D_{k-1} + eps I)^{-1} c_k, so the norms are
    # monotone LOWER bounds on the exact ||D_{k-1}^+ c_k||, approached from below. Solid
    # is max||z_k||, dashed is max||l_k||.
    norm_axis.set_title(r'Tikhonov path: solid $\max_k\|z_k\|$, dashed $\max_k\|\ell_k\|$', fontsize=TITLE_FONTSIZE)
    norm_axis.legend(fontsize=6)
    norm_axis.grid(True, alpha=0.3, which='both')

    return _save_figure(fig, output_path)


def plot_picard_tradeoff(picard_frame, output_path: str, title: str):
    '''M1: the latent norm needed against the accuracy achieved on c_k'''

    fig, axis = plt.subplots(figsize=(6.0, 4.0))

    # each (molecule, k) traces one curve as the cutoff tau is swept; the useful reading
    # is how far left one can move before the required norm turns upward
    for (chembl_id, k_index), group in picard_frame.groupby(['chembl_id', 'k_index']):
        group = group.sort_values('cutoff', ascending=False)
        axis.loglog(
            np.maximum(group['residual'], 1e-16), np.maximum(group['z_norm'], 1e-16),
            color='C0', alpha=0.12, linewidth=0.8,
        )

    for target, color in zip(RESIDUAL_TARGETS, ['C3', 'C2']):
        axis.axvline(target, color=color, linestyle='--', linewidth=1.0, label=rf'$\rho = {target:g}$')

    axis.set_xlabel(r'relative residual $\rho(\tau) = \|c_k - D_{k-1} z_k(\tau)\| / \|c_k\|$', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_ylabel(r'$\|z_k(\tau)\|$', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_title(title, fontsize=TITLE_FONTSIZE)
    axis.legend(fontsize=AXIS_LABEL_FONTSIZE)
    axis.grid(True, alpha=0.3, which='both')

    return _save_figure(fig, output_path)


def plot_picard_ratios(picard_records, output_path: str):
    '''M1: the Picard plot, |b_i| / lambda_i against mode index'''

    fig, axis = plt.subplots(figsize=(6.0, 4.0))

    curves = [record['picard_ratios'] for record in picard_records if record['picard_ratios'].size >= 2]
    if len(curves) == 0:
        plt.close(fig)
        return []

    for curve in curves:
        fraction = np.linspace(0.0, 1.0, curve.size)
        axis.semilogy(fraction, np.maximum(curve, 1e-300), color='C0', alpha=0.12, linewidth=0.8)

    grid, lower, median, upper = _quantile_band(curves, num_points=200)
    axis.semilogy(grid, np.maximum(median, 1e-300), color='C3', linewidth=1.8, label='median')
    axis.fill_between(grid, np.maximum(lower, 1e-300), np.maximum(upper, 1e-300), color='C3', alpha=0.25, label='interquartile range')

    axis.set_xlabel(r'fractional mode index (eigenvalues sorted descending)', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_ylabel(r'$|b_i| / \lambda_i$', fontsize=AXIS_LABEL_FONTSIZE)
    # a curve that stays flat or falls means |b_i| decays at least as fast as lambda_i,
    # so the truncated latent stays bounded; a curve that rises at the right does not
    axis.set_title(r'Discrete Picard condition, $b_i = u_i^T c_k$', fontsize=TITLE_FONTSIZE)
    axis.legend(fontsize=AXIS_LABEL_FONTSIZE)
    axis.grid(True, alpha=0.3)

    return _save_figure(fig, output_path)


def plot_required_norm_by_step(target_frame, output_path: str):
    '''M1: the latent norm a given accuracy demands, against position in the sequence'''

    fig, axis = plt.subplots(figsize=(6.0, 4.0))

    for target, color in zip(RESIDUAL_TARGETS, ['C0', 'C1']):
        subset = target_frame[target_frame['residual_target'] == target]
        axis.scatter(
            subset['k_fraction'], np.maximum(subset['z_norm'], 1e-16),
            c=color, s=12, alpha=0.5, label=rf'$\rho = {target:g}$',
        )

    axis.set_yscale('log')
    axis.set_xlabel('fractional orbital index $k / K$', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_ylabel(r'$\|z_k(\tau^*)\|$', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_title('Latent magnitude required to hit a fixed accuracy on $c_k$', fontsize=TITLE_FONTSIZE)
    axis.legend(fontsize=AXIS_LABEL_FONTSIZE)
    axis.grid(True, alpha=0.3, which='both')

    return _save_figure(fig, output_path)


def plot_null_fraction(picard_frame, output_path: str, reference_cutoff: float):
    '''M3: the share of z_k that the target does not constrain'''

    fig, axis = plt.subplots(figsize=(6.0, 4.0))

    subset = picard_frame[np.isclose(picard_frame['cutoff'], reference_cutoff)]
    scatter = axis.scatter(
        subset['k_fraction'], subset['null_fraction'],
        c=subset['n_basis'], cmap='viridis', s=14,
    )
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label('$K$', fontsize=AXIS_LABEL_FONTSIZE)

    axis.set_xlabel('fractional orbital index $k / K$', fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_ylabel(r'$\dim \mathrm{null}(D_{k-1}) / (k-1)$', fontsize=AXIS_LABEL_FONTSIZE)
    # rank saturates at N_e/2 while k grows to K, so this rises toward roughly 0.8; the
    # unconstrained directions are why per-step supervision on (c_k, d_k) is well posed
    # even though z_k itself is not unique
    axis.set_title(rf'Unconstrained directions of $z_k$, cutoff $\tau = {reference_cutoff:g}$', fontsize=TITLE_FONTSIZE)
    axis.grid(True, alpha=0.3)

    return _save_figure(fig, output_path)


def plot_rollout(rollout_frame, output_path: str):
    '''M2: teacher forcing against free running, per noise level'''

    noise_levels = sorted(rollout_frame['noise_level'].unique())
    fig, axes = plt.subplots(1, len(noise_levels), figsize=(3.4 * len(noise_levels), 3.6), sharey=True)
    axes = np.atleast_1d(axes)

    for axis, noise_level in zip(axes, noise_levels):
        subset = rollout_frame[rollout_frame['noise_level'] == noise_level]
        for chembl_id, group in subset.groupby('chembl_id'):
            group = group.sort_values('k_index')
            fraction = (group['k_index'] / group['k_index'].max()).to_numpy()
            # At k = 1 the block is reproduced exactly and the error is 0, which a log axis
            # cannot show. Clamping it to a tiny constant drags the shared y-range down to
            # that constant and flattens every curve, so the zeros are masked out instead.
            for column, color in [('forced_block_error', 'C0'), ('free_block_error', 'C3')]:
                curve = group[column].to_numpy()
                visible = curve > 0.0
                axis.semilogy(fraction[visible], curve[visible], color=color, alpha=0.7, linewidth=1.0)

        axis.set_xlabel('fractional orbital index $k / K$', fontsize=AXIS_LABEL_FONTSIZE)
        axis.set_title(rf'$\eta = {noise_level:g}$', fontsize=TITLE_FONTSIZE)
        axis.grid(True, alpha=0.3, which='both')

    # the first two steps are reproduced to machine precision, so without a floor they
    # stretch the shared log axis over 20 decades and flatten everything of interest
    axes[0].set_ylim(bottom=1e-8)
    axes[0].set_ylabel(r'$\|\hat D_k - D_k\|_F / \|D_k\|_F$', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].plot([], [], color='C0', label='teacher forced')
    axes[0].plot([], [], color='C3', label='free running')
    axes[0].legend(fontsize=AXIS_LABEL_FONTSIZE)
    fig.suptitle('Error propagation with and without teacher forcing', fontsize=TITLE_FONTSIZE)

    return _save_figure(fig, output_path)


def plot_mass_profile(mass_records, output_path: str):
    '''M5: the share of the Frobenius mass each bordering step carries'''

    fig, (step_axis, cumulative_axis) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    step_curves = []
    cumulative_curves = []
    for c_squared, d_squared, total in mass_records:
        per_step = (2.0 * c_squared + d_squared) / total # (n_basis,)
        step_curves.append(per_step)
        cumulative_curves.append(np.cumsum(per_step))

    for curves, axis, label in [
        (step_curves, step_axis, 'per step'),
        (cumulative_curves, cumulative_axis, 'cumulative'),
    ]:
        grid, lower, median, upper = _quantile_band(curves, num_points=400)
        axis.plot(grid, median, color='C3', linewidth=1.8, label='median')
        axis.fill_between(grid, lower, upper, color='C3', alpha=0.25, label='interquartile range')
        axis.set_xlabel('fractional orbital index $k / K$', fontsize=AXIS_LABEL_FONTSIZE)
        axis.set_ylabel(f'{label} share of $\\|D\\|_F^2$', fontsize=AXIS_LABEL_FONTSIZE)
        axis.legend(fontsize=AXIS_LABEL_FONTSIZE)
        axis.grid(True, alpha=0.3)

    step_axis.set_yscale('log')
    step_axis.set_title('Target mass carried by each step', fontsize=TITLE_FONTSIZE)
    cumulative_axis.set_title('Cumulative target mass', fontsize=TITLE_FONTSIZE)

    return _save_figure(fig, output_path)


# ----------------------------------------------------------------------------------------------------


def main(
    log_dir: str,
    num_size_bins: int,
    num_per_bin: int,
    seed: int,
    jitter_relative: float,
    pivot_floor_factor: float,
    rank_tolerance: float,
    num_entry_samples: int,
    num_promolecular: int,
    num_jitter_sweep: int,
    explicit_check_max_dim: int,
    num_spectral: int,
    num_k_samples: int,
    num_rollout: int,
    rollout_max_dim: int,
    rollout_residual_target: float,
    num_lowdin: int,
    num_threads: int,
):
    '''factor sampled QMugs density matrices and write the diagnostic figures'''

    # ----------------------------------------------------------------------------------------------------
    # logging and directory setup

    if len(logging.getLogger().handlers) == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s',
            datefmt='%H:%M:%S',
        )
    logging.info('Command: {0}'.format(' '.join([x for x in sys.argv])))

    log_dir = os.path.abspath(log_dir)
    figure_dir = os.path.join(log_dir, 'figs')
    os.makedirs(figure_dir, exist_ok=True)
    logging.info(f'[INFO] Writing results to {log_dir}')

    torch.set_num_threads(num_threads)
    if 'psi4' in sys.modules:
        psi4.set_num_threads(num_threads)

    with open(os.path.join(QMUGS_EXAMPLE_DIR, 'utils/download_data.json'), 'r', encoding='utf-8') as f:
        download_config = json.load(f)

    # ----------------------------------------------------------------------------------------------------
    # conformer selection

    selected = sample_conformers(
        download_config=download_config,
        num_size_bins=num_size_bins,
        num_per_bin=num_per_bin,
        seed=seed,
    )
    logging.info(f'[INFO] Selected {len(selected)} conformers across {num_size_bins} size bins')

    rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------------------------------------
    # per-conformer factorization

    records = []
    promolecular_records = []
    summary_rows = []
    sweep_rows = []
    explicit_checks = []

    # M1 - M5 accumulators
    picard_rows = []
    picard_records = []
    target_rows = []
    lowdin_picard_rows = []
    lowdin_target_rows = []
    rollout_rows = []
    rollout_delta_rows = []
    mass_records = []
    truncation_violations = 0
    target_truncation_violations = 0
    max_lowdin_trace_error = 0.0

    # the rollout perturbations are drawn from a dedicated generator so that adding or
    # removing conformers does not reshuffle them
    rollout_generator = torch.Generator().manual_seed(seed)

    # sample_conformers walks the size bins in order, so taking the first num_spectral
    # conformers would analyze only the smallest molecules. Spreading the positions evenly
    # over the selection keeps M1 stratified over K, which is the whole point of the
    # stratified sample.
    def spread_positions(count: int):
        '''evenly spaced positions in the selection, so the K range stays covered'''
        if count >= len(selected):
            return set(range(len(selected)))
        return set(np.linspace(0, len(selected) - 1, count).round().astype(int).tolist())

    spectral_positions = spread_positions(num_spectral)
    lowdin_positions = spread_positions(num_lowdin)

    for position, entry in enumerate(selected):
        chembl_id = entry['chembl_id']
        compute_promolecular = len(promolecular_records) < num_promolecular

        try:
            density_matrix, overlap, promolecular, molecule_record = load_wavefunction_matrices(
                wfn_path=entry['wfn_path'],
                compute_promolecular=compute_promolecular,
            )
        except Exception as error:
            logging.info(f'[WARNING] Skipping {chembl_id}: {error}')
            continue

        n_basis = molecule_record['n_basis']
        density_matrix = torch.from_numpy(density_matrix).to(WORKING_DTYPE) # (n_basis, n_basis)
        overlap = torch.from_numpy(overlap).to(WORKING_DTYPE) # (n_basis, n_basis)

        # the stored matrices are symmetric only up to round-off, and the bordering
        # construction assumes exact symmetry
        asymmetry = float(torch.linalg.norm(density_matrix - density_matrix.transpose(0, 1)))
        assert asymmetry <= 1e-8 * float(torch.linalg.norm(density_matrix)), (
            f'{chembl_id} density matrix is not symmetric to round-off: {asymmetry:.3e}'
        )
        density_matrix = 0.5 * (density_matrix + density_matrix.transpose(0, 1))

        analysis = analyze_matrix(
            matrix=density_matrix,
            jitter_relative=jitter_relative,
            pivot_floor_factor=pivot_floor_factor,
            num_entry_samples=num_entry_samples,
            rng=rng,
        )
        spectra = spectral_summary(density_matrix, overlap, rank_tolerance=rank_tolerance)

        records.append(analysis)
        summary_rows.append({
            'chembl_id': chembl_id,
            'wfn_path': entry['wfn_path'],
            **molecule_record,
            **spectra,
            **analysis['scalars'],
        })

        # the literal step-by-step recursion is only run on small matrices, where its
        # K-iteration Python loop is affordable
        if n_basis <= explicit_check_max_dim and len(explicit_checks) < 3:
            explicit_error = explicit_recursion_error(
                z_rows=z_rows_from_unit_lower(
                    unpivoted_ldl(density_matrix, jitter=analysis['scalars']['jitter'])[0]
                ),
                pivots=torch.from_numpy(analysis['curves']['pivots']).to(WORKING_DTYPE),
                matrix=density_matrix,
            )
            explicit_checks.append({'chembl_id': chembl_id, 'n_basis': n_basis, 'error': explicit_error})
            logging.info(f'[INFO] {chembl_id}: explicit recursion relative error {explicit_error:.3e}')

        # the jitter sweep separates a genuinely large latent from one whose size is
        # set by the regularization
        if len(sweep_rows) < num_jitter_sweep * 5:
            for sweep_jitter in [jitter_relative * factor for factor in [0.01, 0.1, 1.0, 10.0, 100.0]]:
                try:
                    sweep_analysis = analyze_matrix(
                        matrix=density_matrix,
                        jitter_relative=sweep_jitter,
                        pivot_floor_factor=pivot_floor_factor,
                        num_entry_samples=0,
                        rng=rng,
                    )
                except ValueError as error:
                    logging.info(f'[INFO] {chembl_id}: jitter {sweep_jitter:.1e} rejected ({error})')
                    continue
                sweep_rows.append({
                    'chembl_id': chembl_id,
                    'jitter_relative': sweep_jitter,
                    'num_pivots_above_floor': sweep_analysis['scalars']['num_pivots_above_floor'],
                    'max_z_norm': sweep_analysis['scalars']['max_z_norm'],
                    # the L parameterization follows the same Tikhonov path, so recording it
                    # here says whether reparameterizing in terms of l_k would help
                    'max_l_norm': sweep_analysis['scalars']['max_l_norm'],
                    'median_z_norm': sweep_analysis['scalars']['median_z_norm'],
                    'median_l_norm': sweep_analysis['scalars']['median_l_norm'],
                })

        # ------------------------------------------------------------------------------------
        # M5: where the Frobenius mass of the target lives across the sequence. Cheapest of
        # the five, and if the ill-conditioned late steps carry no mass then much of the
        # conditioning concern is moot.

        mass_records.append(target_mass_profile(density_matrix))

        # ------------------------------------------------------------------------------------
        # M1, M3, M4: the truncated pseudoinverse at sampled bordering steps. This is the
        # measurement the jittered factorization cannot make, because it reaches the exact
        # latent instead of only bounding it from below.

        if position in spectral_positions:
            for k_index in sample_step_indices(n_basis, num_k_samples):
                picard = picard_analysis(
                    matrix=density_matrix,
                    k_index=k_index,
                    cutoffs=PICARD_CUTOFFS,
                    residual_targets=RESIDUAL_TARGETS,
                )
                if picard is None:
                    continue
                picard_records.append(picard)
                for row in picard['rows']:
                    # the lemma of schur_decomp_test.md 6.3 says truncation can never make
                    # the required pivot negative; count any violation rather than assert,
                    # so one bad conformer does not kill the run
                    if not row['delta_nonnegative']:
                        truncation_violations += 1
                    picard_rows.append({
                        'chembl_id': chembl_id,
                        'n_basis': n_basis,
                        'n_alpha': molecule_record['n_alpha'],
                        'k_fraction': k_index / n_basis,
                        **row,
                    })
                for target, reading in picard['targets'].items():
                    # the unfloored path, which is the one truncated_latents also takes
                    if not reading['delta_nonnegative']:
                        target_truncation_violations += 1
                    target_rows.append({
                        'chembl_id': chembl_id,
                        'n_basis': n_basis,
                        'k_index': k_index,
                        'k_fraction': k_index / n_basis,
                        'residual_target': target,
                        **reading,
                    })

            logging.info(
                f'[INFO] {chembl_id}: Picard analysis at {len(sample_step_indices(n_basis, num_k_samples))} steps'
            )

        # ------------------------------------------------------------------------------------
        # Lowdin variant of M1. rank(P) = rank(D) since S^{1/2} is invertible, so this can
        # only test whether removing the overlap's own ill-conditioning helps within the
        # range; it cannot remove the rank deficiency.

        if position in lowdin_positions:
            lowdin_matrix = lowdin_transform(density_matrix, overlap) # (n_basis, n_basis)
            n_electrons = molecule_record['n_alpha'] + molecule_record['n_beta']
            trace_error = abs(float(torch.trace(lowdin_matrix)) - n_electrons)
            # In the orthogonalized basis the plain trace IS the electron count, which the
            # AO basis cannot provide as a check. S^{1/2} is built by eigendecomposing an
            # overlap with kappa(S) ~ 3.3e4, so this is reported rather than asserted --
            # a diagnostic-quality discrepancy here should not cost M2 and M5.
            max_lowdin_trace_error = max(max_lowdin_trace_error, trace_error)
            if trace_error > 1e-6 * max(1.0, n_electrons):
                logging.info(f'[WARN] {chembl_id}: Lowdin trace off by {trace_error:.3e}')

            for k_index in sample_step_indices(n_basis, num_k_samples):
                picard = picard_analysis(
                    matrix=lowdin_matrix,
                    k_index=k_index,
                    cutoffs=PICARD_CUTOFFS,
                    residual_targets=RESIDUAL_TARGETS,
                )
                if picard is None:
                    continue
                for row in picard['rows']:
                    lowdin_picard_rows.append({
                        'chembl_id': chembl_id,
                        'n_basis': n_basis,
                        'k_fraction': k_index / n_basis,
                        **row,
                    })
                for target, reading in picard['targets'].items():
                    lowdin_target_rows.append({
                        'chembl_id': chembl_id,
                        'n_basis': n_basis,
                        'k_index': k_index,
                        'k_fraction': k_index / n_basis,
                        'residual_target': target,
                        **reading,
                    })

        # ------------------------------------------------------------------------------------
        # M2: teacher forcing against free running. Needs exact truncated latents at every
        # step, which costs O(K^4), so it runs only on the smallest molecules.

        if n_basis <= rollout_max_dim and len(set(row['chembl_id'] for row in rollout_rows)) < num_rollout:
            rollout_z, rollout_deltas, rollout_residuals, rollout_min_eigenvalues = truncated_latents(
                density_matrix,
                residual_target=rollout_residual_target,
                rank_tolerance=rank_tolerance,
            )

            # The lemma of schur_decomp_test.md 6.3 gives delta_k >= 0 exactly, but past
            # k = N_e/2 the exact value IS zero, so the computed one is a difference of
            # nearly equal quantities and its sign is set by round-off. The magnitude is
            # therefore the measurement, not the sign: ~1e-14 is that round-off, while a
            # violation of order the diagonal entries would mean the truncated decoder is
            # not PSD-safe in floating point. Recorded rather than asserted, so one bad
            # conformer does not cost the whole run.
            min_delta = float(rollout_deltas.min())
            num_negative = int((rollout_deltas < 0.0).sum())
            rollout_delta_rows.append({
                'chembl_id': chembl_id,
                'n_basis': n_basis,
                'min_delta': min_delta,
                'min_delta_k_index': int(torch.argmin(rollout_deltas)),
                'num_negative_deltas': num_negative,
                'max_diagonal': float(torch.diagonal(density_matrix).max()),
                'median_min_retained_eigenvalue': float(rollout_min_eigenvalues.median()),
                'fraction_target_unmet': float((rollout_residuals > rollout_residual_target).to(WORKING_DTYPE).mean()),
            })
            if num_negative > 0:
                logging.info(
                    f'[INFO] {chembl_id}: {num_negative} of {n_basis} truncated pivots negative, '
                    f'min {min_delta:.3e}; clamped to zero'
                )
            # a decoder head emits delta through exp or softplus and structurally cannot
            # produce a negative pivot, so clamping is what the architecture does anyway
            rollout_deltas = torch.clamp(rollout_deltas, min=0.0)

            for noise_level in ROLLOUT_NOISE_LEVELS:
                for row in rollout_errors(
                    matrix=density_matrix,
                    z_rows=rollout_z,
                    deltas=rollout_deltas,
                    noise_level=noise_level,
                    generator=rollout_generator,
                ):
                    rollout_rows.append({'chembl_id': chembl_id, 'n_basis': n_basis, **row})

            logging.info(
                f'[INFO] {chembl_id}: rollout done, max||z_trunc||={float(torch.linalg.norm(rollout_z, dim=1).max()):.3e}, '
                f'min delta={min_delta:.3e}'
            )

        # the promolecular baseline enters the delta formulation as fixed data, so its
        # own rank and pivot profile matter even though the latents are unchanged
        if compute_promolecular:
            promolecular = torch.from_numpy(promolecular).to(WORKING_DTYPE) # (n_basis, n_basis)
            promolecular = 0.5 * (promolecular + promolecular.transpose(0, 1))
            promolecular_analysis = analyze_matrix(
                matrix=promolecular,
                jitter_relative=jitter_relative,
                pivot_floor_factor=pivot_floor_factor,
                num_entry_samples=num_entry_samples,
                rng=rng,
            )
            promolecular_records.append(promolecular_analysis)

            # the delta formulation reconstructs D_pro + Delta_D, which is D exactly,
            # so its latents must agree with the direct ones to round-off
            delta_total = promolecular + (density_matrix - promolecular)
            delta_analysis = analyze_matrix(
                matrix=delta_total,
                jitter_relative=jitter_relative,
                pivot_floor_factor=pivot_floor_factor,
                num_entry_samples=0,
                rng=rng,
            )
            summary_rows[-1]['delta_latent_agreement'] = float(np.max(np.abs(
                delta_analysis['curves']['z_norms'] - analysis['curves']['z_norms']
            )))
            summary_rows[-1]['promolecular_pivots_above_floor'] = promolecular_analysis['scalars']['num_pivots_above_floor']

        logging.info(
            f'[INFO] ({position + 1}/{len(selected)}) {chembl_id}: K={n_basis}, '
            f"rank={analysis['scalars']['num_pivots_above_floor']}, N_e/2={molecule_record['n_alpha']}, "
            f"max||z||={analysis['scalars']['max_z_norm']:.3e}, "
            f"recon err={analysis['scalars']['factorization_error']:.3e}"
        )

    if len(records) == 0:
        raise ValueError('No conformers were analyzed successfully')

    # ----------------------------------------------------------------------------------------------------
    # outputs

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(os.path.join(log_dir, 'per_conformer_summary.csv'), index=False)

    np.savez_compressed(
        os.path.join(log_dir, 'per_step_curves.npz'),
        **{
            f"{row['chembl_id']}_{name}": record['curves'][name]
            for row, record in zip(summary_rows, records)
            for name in ['pivots', 'z_norms', 'l_norms']
        },
    )

    overall = {
        'num_conformers': int(len(records)),
        'n_basis_range': [int(summary_frame['n_basis'].min()), int(summary_frame['n_basis'].max())],
        'jitter_relative': jitter_relative,
        'pivot_floor_factor': pivot_floor_factor,
        'rank_tolerance': rank_tolerance,
        'seed': seed,
        # the factorization residual should sit at the jitter level, not below it
        'max_factorization_error': float(summary_frame['factorization_error'].max()),
        'max_expected_factorization_error': float(summary_frame['expected_factorization_error'].max()),
        'max_recursion_off_diagonal_error': float(summary_frame['recursion_off_diagonal_error'].max()),
        'max_recursion_diagonal_error': float(summary_frame['recursion_diagonal_error'].max()),
        'explicit_recursion_checks': explicit_checks,
        # question 2: does the counted rank match N_e / 2
        'median_rank_minus_n_alpha': float((summary_frame['num_pivots_above_floor'] - summary_frame['n_alpha']).median()),
        'median_rank_fraction_of_K': float((summary_frame['num_pivots_above_floor'] / summary_frame['n_basis']).median()),
        'median_spectral_rank_minus_n_alpha': float((summary_frame['spectral_rank'] - summary_frame['n_alpha']).median()),
        # question 1: how large must the pair head's output be
        'median_max_z_norm': float(summary_frame['max_z_norm'].median()),
        'max_max_z_norm': float(summary_frame['max_z_norm'].max()),
        'median_max_l_norm': float(summary_frame['max_l_norm'].median()),
        'median_z_over_l_max_norm': float((summary_frame['max_z_norm'] / summary_frame['max_l_norm']).median()),
        'median_overlap_condition_number': float(summary_frame['overlap_condition_number'].median()),
        'median_range_condition_number': float(summary_frame['range_condition_number'].median()),
        # the trace convention: tr(SD) should equal N_e, tr(D) should not
        'median_trace_SD_minus_n_electrons': float(
            (summary_frame['trace_SD'] - summary_frame['n_alpha'] - summary_frame['n_beta']).median()
        ),
        'median_trace_D_minus_n_electrons': float(
            (summary_frame['trace_D'] - summary_frame['n_alpha'] - summary_frame['n_beta']).median()
        ),
    }
    if 'delta_latent_agreement' in summary_frame.columns:
        overall['max_delta_latent_disagreement'] = float(summary_frame['delta_latent_agreement'].max())

    # the range condition number is only meaningful when the rank tolerance separated the
    # true nonzero spectrum from the eigendecomposition noise floor; reporting the median
    # over all conformers mixes the two populations and is misleading
    clean = summary_frame[summary_frame['min_nonzero_eigenvalue'] > 0.1]
    overall['num_clean_rank_cutoff'] = int(len(clean))
    if len(clean) > 0:
        overall['clean_median_range_condition_number'] = float(clean['range_condition_number'].median())
        overall['clean_max_range_condition_number'] = float(clean['range_condition_number'].max())
        overall['clean_max_spectral_rank_minus_n_alpha'] = int((clean['spectral_rank'] - clean['n_alpha']).abs().max())

    # ------------------------------------------------------------------------------------
    # M1 / M3 / M4

    target_frame = pd.DataFrame(target_rows)
    picard_frame = pd.DataFrame(picard_rows)
    if len(target_frame) > 0:
        target_frame.to_csv(os.path.join(log_dir, 'picard_targets.csv'), index=False)
        picard_frame.to_csv(os.path.join(log_dir, 'picard_samples.csv'), index=False)

        overall['picard_num_samples'] = int(len(picard_records))
        overall['picard_truncation_violations'] = int(truncation_violations)
        # the unfloored path; nonzero here means an accuracy bar forced an eigenvalue at
        # the noise floor, which is what crashed the 2026-08-27 run
        overall['picard_target_truncation_violations'] = int(target_truncation_violations)
        for target in RESIDUAL_TARGETS:
            subset = target_frame[target_frame['residual_target'] == target]
            # how far down the spectrum the accuracy bar reaches, relative to lambda_max.
            # Approaching rank_tolerance means the bar and the resolvable floor conflict.
            overall[f'picard_median_min_retained_eigenvalue_rho_{target:g}'] = float(
                subset['min_retained_eigenvalue'].median()
            )
            overall[f'picard_min_retained_eigenvalue_rho_{target:g}'] = float(
                subset['min_retained_eigenvalue'].min()
            )
            # THE headline number: the latent norm a modest accuracy on c_k demands
            overall[f'picard_median_z_norm_at_rho_{target:g}'] = float(subset['z_norm'].median())
            overall[f'picard_p95_z_norm_at_rho_{target:g}'] = float(subset['z_norm'].quantile(0.95))
            overall[f'picard_max_z_norm_at_rho_{target:g}'] = float(subset['z_norm'].max())
            overall[f'picard_median_modes_at_rho_{target:g}'] = float(subset['num_retained'].median())
            # a step where the accuracy bar is unreachable at any truncation means c_k has
            # mass the spectrum simply cannot represent; the z_norm reported there is a
            # floor, not an answer
            overall[f'picard_fraction_target_unmet_rho_{target:g}'] = float(1.0 - subset['target_met'].mean())

        # M4: the share of ||c_k||^2 a gradient on the per-step loss can see
        for cutoff in [1e-2, 1e-4, 1e-6]:
            subset = picard_frame[np.isclose(picard_frame['cutoff'], cutoff)]
            if len(subset) > 0:
                overall[f'median_mass_above_cutoff_{cutoff:g}'] = float(subset['mass_above_cutoff'].median())

        # M3: how much of z_k the target leaves free
        late = picard_frame[(picard_frame['k_fraction'] > 0.8) & np.isclose(picard_frame['cutoff'], 1e-6)]
        if len(late) > 0:
            overall['median_null_fraction_late_steps'] = float(late['null_fraction'].median())

    lowdin_target_frame = pd.DataFrame(lowdin_target_rows)
    if len(lowdin_target_frame) > 0:
        lowdin_target_frame.to_csv(os.path.join(log_dir, 'picard_targets_lowdin.csv'), index=False)
        pd.DataFrame(lowdin_picard_rows).to_csv(os.path.join(log_dir, 'picard_samples_lowdin.csv'), index=False)
        for target in RESIDUAL_TARGETS:
            subset = lowdin_target_frame[lowdin_target_frame['residual_target'] == target]
            overall[f'lowdin_median_z_norm_at_rho_{target:g}'] = float(subset['z_norm'].median())
            overall[f'lowdin_p95_z_norm_at_rho_{target:g}'] = float(subset['z_norm'].quantile(0.95))
        overall['lowdin_max_trace_error'] = float(max_lowdin_trace_error)

    # ------------------------------------------------------------------------------------
    # M2

    delta_frame = pd.DataFrame(rollout_delta_rows)
    if len(delta_frame) > 0:
        delta_frame.to_csv(os.path.join(log_dir, 'rollout_pivots.csv'), index=False)
        # THE number for the lemma of schur_decomp_test.md 6.3 in floating point. Compare
        # it against max_diagonal: a ratio of ~1e-15 is round-off on a quantity whose exact
        # value is zero, while anything approaching 1 means the truncated decoder is not
        # PSD-safe as formulated.
        overall['rollout_min_delta'] = float(delta_frame['min_delta'].min())
        overall['rollout_min_delta_over_diagonal'] = float(
            (delta_frame['min_delta'] / delta_frame['max_diagonal']).min()
        )
        overall['rollout_num_negative_deltas'] = int(delta_frame['num_negative_deltas'].sum())
        overall['rollout_median_min_retained_eigenvalue'] = float(
            delta_frame['median_min_retained_eigenvalue'].median()
        )
        # with the floor in place the accuracy bar may be unreachable, which is what the
        # eta = 0 rollout curve then measures
        overall['rollout_fraction_target_unmet'] = float(delta_frame['fraction_target_unmet'].median())

    rollout_frame = pd.DataFrame(rollout_rows)
    if len(rollout_frame) > 0:
        rollout_frame.to_csv(os.path.join(log_dir, 'rollout.csv'), index=False)
        overall['rollout_residual_target'] = rollout_residual_target
        overall['rollout_num_conformers'] = int(rollout_frame['chembl_id'].nunique())
        for noise_level in ROLLOUT_NOISE_LEVELS:
            subset = rollout_frame[rollout_frame['noise_level'] == noise_level]
            if len(subset) == 0:
                continue
            # the error at the end of the sequence is what an inference pass would suffer
            final = subset.loc[subset.groupby('chembl_id')['k_index'].idxmax()]
            overall[f'rollout_median_forced_final_error_eta_{noise_level:g}'] = float(final['forced_block_error'].median())
            overall[f'rollout_median_free_final_error_eta_{noise_level:g}'] = float(final['free_block_error'].median())
            overall[f'rollout_median_free_over_forced_eta_{noise_level:g}'] = float(
                (final['free_block_error'] / final['forced_block_error'].clip(lower=1e-300)).median()
            )

    # ------------------------------------------------------------------------------------
    # M5

    if len(mass_records) > 0:
        np.savez_compressed(
            os.path.join(log_dir, 'mass_profile.npz'),
            **{
                f"{row['chembl_id']}_{name}": array
                for row, record in zip(summary_rows, mass_records)
                for name, array in [('c_squared', record[0]), ('d_squared', record[1])]
            },
        )
        # the share of ||D||_F^2 sitting in the last fifth of the sequence, where D_{k-1}
        # is most rank deficient
        late_shares = []
        for c_squared, d_squared, total in mass_records:
            per_step = (2.0 * c_squared + d_squared) / total
            late_shares.append(float(per_step[int(0.8 * per_step.size):].sum()))
        overall['median_mass_share_last_fifth'] = float(np.median(late_shares))

    with open(os.path.join(log_dir, 'schur_decomp_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(overall, f, indent=4)

    logging.info('[INFO] Summary: ' + json.dumps(overall, indent=4))

    # ----------------------------------------------------------------------------------------------------
    # figures

    plot_pivot_decay(
        curves=[record['curves']['pivots'] for record in records],
        output_path=os.path.join(figure_dir, 'pivot_decay'),
        title=r'Pivot decay of the ground-truth 1-RDM, $D = L \Delta L^T$',
    )
    plot_latent_norms(records, os.path.join(figure_dir, 'latent_norms'))
    plot_norm_ratio(records, os.path.join(figure_dir, 'latent_norm_ratio'))
    plot_entry_histograms(records, os.path.join(figure_dir, 'latent_entry_histograms'))
    plot_rank_cutoff(summary_frame, os.path.join(figure_dir, 'rank_cutoff'))
    plot_z_range_against_overlap(summary_frame, os.path.join(figure_dir, 'latent_range_vs_overlap'))

    if len(sweep_rows) > 0:
        sweep_frame = pd.DataFrame(sweep_rows)
        sweep_frame.to_csv(os.path.join(log_dir, 'jitter_sweep.csv'), index=False)
        plot_jitter_sweep(sweep_frame, os.path.join(figure_dir, 'jitter_sweep'))

    if len(picard_frame) > 0:
        plot_picard_tradeoff(
            picard_frame,
            os.path.join(figure_dir, 'picard_tradeoff'),
            title=r'M1: latent norm against accuracy on $c_k$, AO basis',
        )
        plot_picard_ratios(picard_records, os.path.join(figure_dir, 'picard_ratios'))
        plot_required_norm_by_step(target_frame, os.path.join(figure_dir, 'required_latent_norm'))
        plot_null_fraction(picard_frame, os.path.join(figure_dir, 'null_fraction'), reference_cutoff=1e-6)

    if len(lowdin_picard_rows) > 0:
        plot_picard_tradeoff(
            pd.DataFrame(lowdin_picard_rows),
            os.path.join(figure_dir, 'picard_tradeoff_lowdin'),
            title=r'M1: latent norm against accuracy on $c_k$, L\"owdin basis',
        )

    if len(rollout_frame) > 0:
        plot_rollout(rollout_frame, os.path.join(figure_dir, 'rollout'))

    if len(mass_records) > 0:
        plot_mass_profile(mass_records, os.path.join(figure_dir, 'mass_profile'))

    if len(promolecular_records) > 0:
        plot_pivot_decay(
            curves=[record['curves']['pivots'] for record in promolecular_records],
            output_path=os.path.join(figure_dir, 'promolecular_pivot_decay'),
            title=r'Pivot decay of the promolecular baseline $D^{pro}$',
        )

    logging.info(f'[INFO] Wrote {len(os.listdir(figure_dir))} figure files to {figure_dir}')


# ----------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(QMUGS_EXAMPLE_DIR)), 'logs', 'qmugs-schur-diagnostics'),
        help='directory for figures and summary files',
    )
    parser.add_argument(
        '--num_size_bins',
        type=int,
        default=10,
        help='number of density matrix size bins to stratify the sample over',
    )
    parser.add_argument(
        '--num_per_bin',
        type=int,
        default=20,
        help='number of conformers to sample from each size bin',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help='random seed for conformer and entry sampling',
    )
    parser.add_argument(
        '--jitter_relative',
        type=float,
        default=1e-10,
        help='Cholesky jitter as a fraction of the mean diagonal entry',
    )
    parser.add_argument(
        '--pivot_floor_factor',
        type=float,
        default=10.0,
        help='pivots below this multiple of the jitter count as numerically zero',
    )
    parser.add_argument(
        '--rank_tolerance',
        type=float,
        default=1e-6,
        help=(
            'eigenvalue threshold, relative to the largest, for the independent rank check; '
            '1e-8 sits inside the float64 noise floor of eigh and admits spurious eigenvalues'
        ),
    )
    parser.add_argument(
        '--num_entry_samples',
        type=int,
        default=20000,
        help='number of strictly-lower-triangle entries sampled per conformer for the histograms',
    )
    parser.add_argument(
        '--num_promolecular',
        type=int,
        default=10,
        help='number of conformers to also factor the promolecular baseline for',
    )
    parser.add_argument(
        '--num_jitter_sweep',
        type=int,
        default=5,
        help='number of conformers to run the jitter sweep on',
    )
    parser.add_argument(
        '--explicit_check_max_dim',
        type=int,
        default=400,
        help='largest K for which the literal step-by-step recursion check is run',
    )
    parser.add_argument(
        '--num_spectral',
        type=int,
        default=20,
        help='number of conformers to run the M1 Picard analysis on',
    )
    parser.add_argument(
        '--num_k_samples',
        type=int,
        default=24,
        help='bordering steps per conformer for M1, geometrically spaced over k',
    )
    parser.add_argument(
        '--num_rollout',
        type=int,
        default=6,
        help='number of conformers to run the M2 rollout on',
    )
    parser.add_argument(
        '--rollout_max_dim',
        type=int,
        default=260,
        help='largest K for the M2 rollout, whose truncated latents cost O(K^4)',
    )
    parser.add_argument(
        '--rollout_residual_target',
        type=float,
        default=1e-3,
        help='relative residual on c_k at which the M2 rollout truncates its latents',
    )
    parser.add_argument(
        '--num_lowdin',
        type=int,
        default=8,
        help='number of conformers to repeat the M1 analysis on in the Lowdin basis',
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=32,
        help='BLAS and psi4 thread count',
    )
    args = parser.parse_args()

    main(
        log_dir=args.log_dir,
        num_size_bins=args.num_size_bins,
        num_per_bin=args.num_per_bin,
        seed=args.seed,
        jitter_relative=args.jitter_relative,
        pivot_floor_factor=args.pivot_floor_factor,
        rank_tolerance=args.rank_tolerance,
        num_entry_samples=args.num_entry_samples,
        num_promolecular=args.num_promolecular,
        num_jitter_sweep=args.num_jitter_sweep,
        explicit_check_max_dim=args.explicit_check_max_dim,
        num_spectral=args.num_spectral,
        num_k_samples=args.num_k_samples,
        num_rollout=args.num_rollout,
        rollout_max_dim=args.rollout_max_dim,
        rollout_residual_target=args.rollout_residual_target,
        num_lowdin=args.num_lowdin,
        num_threads=args.num_threads,
    )
