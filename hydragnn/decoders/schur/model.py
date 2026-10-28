###########################################################################################
# The sequential Schur decoder, composed around a HydraGNN backbone
# Reference: references/schur_decoder.tex
# Author: Derek Wang
###########################################################################################

import logging
import math

import torch
from torch.nn import Module

from hydragnn.decoders.schur.heads import (
    AlphaHead,
    D1Head,
    GlobalEmbedding,
    OrbitalEmbedding,
    ZHead,
)
from hydragnn.decoders.schur.losses import (
    reconstruction_loss,
    regularization_loss,
    relative_errors,
    step_weights,
)
from hydragnn.decoders.schur.recursion import bordering_recursion


def dense_from_padded_target(row, n_basis: int):
    '''unpad one molecule's stored target into a dense (n_basis, n_basis) matrix'''

    # The QMugs pipeline stores either the padded matrix itself or its row-major upper triangle,
    # depending on the output name in use, so both are accepted and told apart by the length.
    # This is duplicated rather than imported from examples/qmugs/utils/utils.py because
    # hydragnn/ must not depend on examples/.
    length = int(row.numel())

    n_padded = int(math.isqrt(length))
    if n_padded * n_padded == length:
        return row.reshape(n_padded, n_padded)[:n_basis, :n_basis] # (n_basis, n_basis)

    # row-major upper triangle of the padded matrix, so length = n_padded(n_padded + 1)/2
    n_padded = (math.isqrt(8 * length + 1) - 1) // 2
    if n_padded * (n_padded + 1) // 2 != length:
        raise ValueError(
            f'stored target of length {length} is neither a square nor an upper triangle'
        )

    upper = torch.triu_indices(n_basis, n_basis, device=row.device) # (2, T(n_basis))
    rows, cols = upper[0], upper[1]
    # position of the physical entry (i, j) inside the row-major padded upper triangle
    flat = rows * n_padded - (rows * (rows - 1)) // 2 + (cols - rows)

    dense = torch.zeros((n_basis, n_basis), dtype=row.dtype, device=row.device)
    values = row[flat]
    dense[rows, cols] = values
    dense[cols, rows] = values # the diagonal is written twice with the same value
    return dense


def trace_normalize(D_hat, overlap_matrix, n_electrons):
    '''scale the predicted density matrix so its electron count equals the number of electrons'''

    # Section 2.2 and Theorem 2.1: after step K the prediction is multiplied by the number of
    # electrons over the trace of the overlap matrix times the prediction. A positive scalar
    # keeps it symmetric and PSD. The basis is not orthogonal, so the plain trace is not the
    # electron count. Both matrices are symmetric, so the trace of their product is the sum of
    # their entrywise product.
    n_basis = int(D_hat.shape[0])
    assert D_hat.shape == (n_basis, n_basis) and overlap_matrix.shape == (n_basis, n_basis)
    assert overlap_matrix.dtype == D_hat.dtype and overlap_matrix.device == D_hat.device

    electron_count = torch.sum(overlap_matrix * D_hat) # scalar
    # positive for a nonzero PSD prediction; a teacher-forced prediction carries no such guarantee
    count_value = float(electron_count.detach())
    assert math.isfinite(count_value) and count_value > 0.0, (
        f'electron count of the predicted density matrix is {count_value:.6e}, '
        'so it cannot be scaled to the number of electrons'
    )
    trace_scale = n_electrons / electron_count # scalar
    return trace_scale * D_hat, trace_scale


class SchurDecoderModel(Module):
    '''HydraGNN backbone plus the sequential Schur decoder of Sections 5.1-5.5'''

    # the training loop passes the batch to loss() as data= only for models that declare this
    loss_accepts_data = True

    def __init__(
        self,
        backbone,
        n_l: int,
        n_m: int,
        n_shell: int,
        species,
        conditioning_type: str = 'concat',
        use_sequence_position: bool = True,
        condition_z_on_graph: bool = False,
        orbital_dim: int = 64,
        orbital_hidden_dims=(128,),
        global_dim: int = 32,
        global_hidden_dims=(64,),
        z_hidden_dims=(128, 128),
        alpha_hidden_dims=(128, 128),
        d1_hidden_dims=(64,),
        num_radial_basis: int = 16,
        radial_cutoff: float = 10.0,
        f: str = 'softplus',
        alpha_bias_init: float = -2.0,
        target: str = 'total',
        routing: str = 'teacher_forcing',
        supervision: str = 'joint',
        chunk_width=None,
        epochs_per_chunk: int = 1,
        w_schedule: str = 'uniform',
        step_normalization: str = 'entry_count',
        z_regularization_weight: float = 0.0,
        delta_regularization_weight: float = 0.0,
        z_row_chunk: int = 128,
        checkpoint_z_head: bool = True,
        max_basis_dim=None,
        report_stride: int = 50,
        trace_normalization: bool = False,
    ):
        super().__init__()

        if target not in ('total', 'delta'):
            raise ValueError(f"routing target must be 'total' or 'delta', got {target}")
        if routing not in ('teacher_forcing', 'free_running'):
            raise ValueError(
                f"routing must be 'teacher_forcing' or 'free_running', got {routing}"
            )
        if supervision not in ('joint', 'semi_sequential', 'per_step'):
            raise ValueError(
                "supervision must be 'joint', 'semi_sequential' or 'per_step', "
                f'got {supervision}'
            )
        if supervision == 'semi_sequential' and chunk_width is None:
            raise ValueError("supervision='semi_sequential' requires chunk_width to be set")
        # the scale depends on every entry, so under windowed supervision the loss of a window
        # would depend on steps outside it
        if trace_normalization and supervision != 'joint':
            raise ValueError(
                f"trace_normalization requires supervision='joint', got {supervision}"
            )

        # The decoder consumes Base.encode, which is the encoder half of Base.forward. A stack
        # that overrides forward does not go through it: MACEStack, for instance, interleaves
        # per-layer readouts and never pools, so Base.encode would quietly hand back features
        # that never saw a convolution. Refuse rather than return a wrong answer.
        from hydragnn.models.Base import Base

        if isinstance(backbone, Base) and type(backbone).forward is not Base.forward:
            raise ValueError(
                f'{type(backbone).__name__} overrides Base.forward, so its encoder does not '
                'match Base.encode and the Schur decoder cannot read node and graph '
                'embeddings from it. Use a stack that inherits Base.forward, such as PAINN.'
            )

        self.backbone = backbone
        self.target = target
        self.routing = routing
        self.supervision = supervision
        self.chunk_width = 1 if supervision == 'per_step' else chunk_width
        self.epochs_per_chunk = int(epochs_per_chunk)
        self.w_schedule = w_schedule
        self.step_normalization = step_normalization
        self.z_regularization_weight = float(z_regularization_weight)
        self.delta_regularization_weight = float(delta_regularization_weight)
        self.max_basis_dim = None if max_basis_dim is None else int(max_basis_dim)
        self.trace_normalization = bool(trace_normalization)
        self.current_epoch = 0
        # every report_stride calls to loss(), one diagnostic line; 0 turns it off
        self.report_stride = int(report_stride)
        self.report_counter = 0

        # the backbone's own multihead decoder is never called, so its parameters would receive
        # no gradient and DDP is constructed with find_unused_parameters=False; freezing them
        # takes them out of DDP's bookkeeping entirely
        self.backbone.heads_NN.requires_grad_(False)
        self.backbone.graph_shared.requires_grad_(False)

        # atomic number to species slot, so preprocessing does not have to know the dataset-wide
        # species list before it has seen every molecule
        species_lookup = torch.full((119,), -1, dtype=torch.long)
        for slot, atomic_number in enumerate(species):
            species_lookup[int(atomic_number)] = slot
        self.register_buffer('species_lookup', species_lookup)
        n_species = len(species)

        activation = backbone.activation_function
        self.orbital_embedding = OrbitalEmbedding(
            node_dim=backbone.hidden_dim,
            n_l=int(n_l),
            n_m=int(n_m),
            n_shell=int(n_shell),
            n_species=n_species,
            orbital_dim=orbital_dim,
            hidden_dims=list(orbital_hidden_dims),
            activation=activation,
            conditioning_type=conditioning_type,
            use_sequence_position=use_sequence_position,
        )
        self.global_embedding = GlobalEmbedding(
            pooled_dim=backbone.pooled_dim,
            global_dim=global_dim,
            hidden_dims=list(global_hidden_dims),
            activation=activation,
            conditioning_type=conditioning_type,
        )
        self.z_head = ZHead(
            orbital_dim=orbital_dim,
            global_dim=global_dim,
            num_radial_basis=num_radial_basis,
            radial_cutoff=radial_cutoff,
            hidden_dims=list(z_hidden_dims),
            activation=activation,
            conditioning_type=conditioning_type,
            condition_on_graph=condition_z_on_graph,
            row_chunk=int(z_row_chunk),
            use_checkpoint=bool(checkpoint_z_head),
        )
        self.alpha_head = AlphaHead(
            orbital_dim=orbital_dim,
            global_dim=global_dim,
            hidden_dims=list(alpha_hidden_dims),
            activation=activation,
            conditioning_type=conditioning_type,
            f=f,
            alpha_bias_init=alpha_bias_init,
        )
        self.d1_head = D1Head(
            orbital_dim=orbital_dim,
            global_dim=global_dim,
            hidden_dims=list(d1_hidden_dims),
            activation=activation,
            conditioning_type=conditioning_type,
            f=f,
            alpha_bias_init=alpha_bias_init,
        )

        # create_model_config has already placed the backbone on its device, while everything
        # built above is on CPU. DDP refuses a module whose parameters span device types
        # ("input module parameters locate in {'cuda', 'cpu'}"), so the decoder is moved to
        # meet the backbone rather than leaving it to the caller.
        self.to(device=next(backbone.parameters()).device)

    def __getattr__(self, name):
        # delegate anything the wrapper does not define to the backbone, so the training loop
        # keeps finding num_heads, head_dims, var_output and the rest. Mirrors the composition
        # pattern already used for interatomic potentials in hydragnn/models/create.py.
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        if name == 'backbone':
            raise AttributeError(name)
        return getattr(self.backbone, name)

    def set_epoch(self, epoch: int):
        '''tell the decoder which epoch it is, so the supervision window can advance'''

        # pushed in by the training loop rather than counted inside forward, which would be a
        # hidden side effect
        self.current_epoch = int(epoch)

    def active_window(self, n_basis: int):
        '''the range of steps k carrying nonzero w_k this epoch, per Sections 6.1-6.3'''

        if self.supervision == 'joint':
            return 0, n_basis

        width = min(int(self.chunk_width), n_basis)
        num_windows = math.ceil(n_basis / width)
        which = (self.current_epoch // self.epochs_per_chunk) % num_windows
        low = which * width
        return low, min(low + width, n_basis)

    def forward(self, data):
        v, G, _, _ = self.backbone.encode(data)

        sizes = data.density_matrix_dim.reshape(-1).to(torch.int64) # (num_graphs,)
        num_graphs = int(sizes.shape[0])
        assert int(sizes.sum()) == int(data.orbital_to_atom.numel()), (
            'orbital metadata length does not match the sum of density_matrix_dim; the dataset '
            'was probably preprocessed before the orbital metadata was added'
        )

        # PyG concatenates orbital-level attributes without offsetting them, since none of their
        # names contains "index" or "batch" (see Data.__inc__), so the slices below are already
        # in each molecule's own local numbering
        zero = torch.zeros(1, dtype=torch.int64, device=sizes.device)
        orbital_ptr = torch.cat([zero, torch.cumsum(sizes, dim=0)]) # (num_graphs + 1,)

        # a batched Data carries ptr; a single unbatched graph does not, so it is rebuilt
        node_ptr = getattr(data, 'ptr', None)
        if node_ptr is None:
            counts = data.natoms_int.reshape(-1).to(torch.int64)
            node_ptr = torch.cat([zero, torch.cumsum(counts, dim=0)])
        # the slices below assume the trunk returned one embedding per input node, which is not
        # true of a stack that drops isolated nodes
        assert int(v.shape[0]) == int(node_ptr[-1]), (
            f'the backbone returned {int(v.shape[0])} node embeddings for '
            f'{int(node_ptr[-1])} nodes, so orbital-to-atom slicing would be misaligned'
        )

        targets = data.y.reshape(num_graphs, -1) # (num_graphs, padded target length)
        promolecular_rows = None
        if hasattr(data, 'promolecular_upper_triangle'):
            promolecular_rows = data.promolecular_upper_triangle.reshape(num_graphs, -1)
        elif self.target == 'delta':
            raise ValueError(
                "target='delta' needs the promolecular matrix in the batch; preprocess with "
                '--density_matrix_delta so promolecular_upper_triangle is stored'
            )

        # the overlap matrix and the number of electrons, needed only for trace normalization
        overlap_rows, electron_numbers = None, None
        if self.trace_normalization:
            if not (hasattr(data, 'overlap_upper_triangle') and hasattr(data, 'n_electrons')):
                raise ValueError(
                    'trace_normalization needs overlap_upper_triangle and n_electrons in the batch; '
                    'use a dataset written by examples/qmugs/utils/add_overlap.py or preprocessed '
                    'with --store_overlap_matrix'
                )
            overlap_rows = data.overlap_upper_triangle.reshape(num_graphs, -1)
            electron_numbers = data.n_electrons.reshape(-1)

        charges = data.charge.reshape(-1)
        multiplicities = data.multiplicity.reshape(-1)
        atom_counts = data.natoms_int.reshape(-1)

        kept, matrices, promolecular_matrices = [], [], []
        predicted_matrices, latents, pivots, trace_scales = [], [], [], []

        for graph in range(num_graphs):
            n_basis = int(sizes[graph])
            if self.max_basis_dim is not None and n_basis > self.max_basis_dim:
                logging.info(
                    'Schur decoder: skipping a conformer with n_basis=%d, above max_basis_dim=%d',
                    n_basis,
                    self.max_basis_dim,
                )
                continue

            node_low, node_high = int(node_ptr[graph]), int(node_ptr[graph + 1])
            orbital_low, orbital_high = int(orbital_ptr[graph]), int(orbital_ptr[graph + 1])

            j_of_k = data.orbital_to_atom[orbital_low:orbital_high].to(torch.int64) # (n_basis,)
            assert int(j_of_k.max()) < node_high - node_low

            node_embedding = v[node_low:node_high][j_of_k] # (n_basis, hidden_dim), v^{j(k)}
            positions = data.pos[node_low:node_high][j_of_k] # (n_basis, 3), r_{j(k)}
            atomic_numbers = (
                data.atomic_numbers[node_low:node_high].reshape(-1)[j_of_k].to(torch.int64)
            ) # (n_basis,)
            species_slot = self.species_lookup[atomic_numbers] # (n_basis,)
            assert int(species_slot.min()) >= 0, (
                'a conformer contains a species missing from the dataset species list; re-run '
                'preprocessing so the metadata covers it'
            )

            orbital = self.orbital_embedding(
                node_embedding,
                data.orbital_angular_momentum[orbital_low:orbital_high],
                data.orbital_within_shell[orbital_low:orbital_high],
                data.orbital_shell[orbital_low:orbital_high],
                species_slot,
            ) # (n_basis, orbital_dim)
            g = self.global_embedding(
                G[graph],
                float(atom_counts[graph]),
                float(n_basis),
                float(charges[graph]),
                float(multiplicities[graph]),
            ) # (global_dim,)

            z_pairs = self.z_head(orbital, positions, g) # (n_basis, n_basis), strictly lower
            delta = self.alpha_head(orbital, g) # (n_basis,)
            # Section 5.2: the first pivot comes from its own head, and delta_1 = d_1
            delta = torch.cat([self.d1_head(orbital[0], g).reshape(1), delta[1:]], dim=0)

            # When a NaN reaches this point the recursion asserts and the traceback names only
            # the recursion, which says nothing about where the NaN was born. Walk the stages
            # in order and report the first one that is already non-finite: everything from v
            # and G onwards belongs to the backbone, everything after that to the decoder.
            if not (torch.isfinite(z_pairs).all() and torch.isfinite(delta).all()):
                stages = (
                    ('backbone v', v[node_low:node_high]),
                    ('backbone G', G[graph]),
                    ('positions', positions),
                    ('orbital embedding', orbital),
                    ('global embedding g', g),
                    ('z', z_pairs),
                    ('delta', delta),
                )
                identifiers = getattr(data, 'chembl_id', None)
                identifier = identifiers[graph] if isinstance(identifiers, list) else '?'
                for name, tensor in stages:
                    mask = torch.isfinite(tensor)
                    largest = float(tensor[mask].abs().max()) if bool(mask.any()) else float('nan')
                    logging.error(
                        'Schur decoder non-finite trace, conformer %s n_basis=%d: '
                        '%s finite=%s max|finite entry|=%.6e',
                        identifier,
                        n_basis,
                        name,
                        bool(mask.all()),
                        largest,
                    )

            # the stored target is the delta against the promolecular baseline when the dataset
            # was preprocessed that way, so the total is rebuilt before the recursion, which
            # Theorem 2.2 always runs on totals
            density_matrix = dense_from_padded_target(targets[graph], n_basis)
            promolecular = None
            if promolecular_rows is not None:
                promolecular = dense_from_padded_target(promolecular_rows[graph], n_basis)
                density_matrix = density_matrix + promolecular

            # the recursion always runs to n_basis, even when only an early window is
            # supervised, so that D_hat has its full shape for the loss; stopping early would
            # be a real saving for per_step and is left for later
            detach_every = None if self.supervision == 'joint' else int(self.chunk_width)
            D_hat, _, _ = bordering_recursion(
                [z_pairs[k, :k] for k in range(n_basis)],
                delta,
                density_matrix=density_matrix if self.routing == 'teacher_forcing' else None,
                detach_every=detach_every,
            )

            # Section 2.2, after step K and before the loss; on the total, also for the delta target
            trace_scale = None
            if self.trace_normalization:
                overlap_matrix = dense_from_padded_target(overlap_rows[graph], n_basis).to(D_hat.dtype)
                D_hat, trace_scale = trace_normalize(
                    D_hat, overlap_matrix, electron_numbers[graph].to(D_hat.dtype)
                ) # (n_basis, n_basis), scalar

            kept.append(graph)
            matrices.append(density_matrix)
            promolecular_matrices.append(promolecular)
            predicted_matrices.append(D_hat)
            latents.append(z_pairs)
            pivots.append(delta)
            trace_scales.append(trace_scale)

        if len(kept) == 0:
            raise ValueError(
                'every conformer in the batch exceeded max_basis_dim; raise it or lower the '
                'batch size so at least one conformer survives'
            )

        return {
            'graph_index': kept,
            'D_hat': predicted_matrices,
            'D': matrices,
            'promolecular': promolecular_matrices,
            'z': latents,
            'delta': pivots,
            'trace_scale': trace_scales,
        }

    def loss(
        self,
        pred,
        value,
        head_index=None,
        mask=None,
        mask_loc=None,
        target_dim=None,
        use_padding_mask=False,
        data=None,
    ):
        '''the total loss of Section 4.4, restricted to the currently active steps'''

        # The signature matches Base.loss exactly so hydragnn/train/train_validate_test.py needs
        # no change. head_index, mask and mask_loc are unused: the decoder builds its target
        # from the batch itself, and padding never enters because every matrix is unpadded to
        # its own n_basis in forward().
        del value, head_index, mask, mask_loc, target_dim, use_padding_mask

        total_reconstruction = None
        total_regularization = None
        reported_total, reported_delta, num_reported = 0.0, 0.0, 0

        for position in range(len(pred['graph_index'])):
            D_hat = pred['D_hat'][position]
            density_matrix = pred['D'][position]
            promolecular = pred['promolecular'][position]
            n_basis = int(density_matrix.shape[0])

            low, high = self.active_window(n_basis)
            weights = step_weights(
                n_basis,
                self.w_schedule,
                low,
                high,
                device=D_hat.device,
                dtype=D_hat.dtype,
            ) # (n_basis,)

            molecule_loss = reconstruction_loss(
                D_hat,
                density_matrix,
                weights,
                loss_function_type=self.backbone.loss_function_type,
                normalization=self.step_normalization,
                target=self.target,
                promolecular=promolecular,
            )
            total_reconstruction = (
                molecule_loss if total_reconstruction is None
                else total_reconstruction + molecule_loss
            )

            molecule_regularization = regularization_loss(
                pred['z'][position],
                pred['delta'][position],
                weights,
                self.z_regularization_weight,
                self.delta_regularization_weight,
            )
            total_regularization = (
                molecule_regularization if total_regularization is None
                else total_regularization + molecule_regularization
            )

            relative_total, relative_delta = relative_errors(
                D_hat, density_matrix, promolecular
            )
            reported_total += relative_total
            if relative_delta is not None:
                reported_delta += relative_delta
            num_reported += 1

        total_reconstruction = total_reconstruction / num_reported
        total_regularization = total_regularization / num_reported

        # The delta-relative error is the more demanding of the two, so it is reported whatever
        # the training target is; it goes to the log rather than to tasks_loss so that
        # num_heads stays 1 and get_head_indices keeps working untouched.
        #
        # This used to be logging.debug, which the training script's root logger (pinned to
        # INFO) discarded, so it was never once emitted in a real run. It is on a stride
        # because a line per batch is unreadable, and it carries the latent magnitudes: the
        # ideal z_k solves c_k = D_{k-1} z_k with a rank-deficient D_{k-1}, so a slow growth of
        # max|z| is the drift to watch for, and it is invisible in the loss until it is fatal.
        self.report_counter += 1
        if self.report_stride > 0 and self.report_counter % self.report_stride == 0:
            largest_z = max(float(z.abs().max()) for z in pred['z'])
            largest_delta = max(float(d.abs().max()) for d in pred['delta'])
            largest_prediction = max(float(D.abs().max()) for D in pred['D_hat'])
            # mean scale, and the mean relative electron deficit of the prediction before scaling
            if self.trace_normalization:
                scales = [float(scale) for scale in pred['trace_scale']]
                logging.info(
                    'Schur decoder: mean trace scale=%.6e mean relative electron deficit=%.6e',
                    sum(scales) / len(scales),
                    sum(1.0 - 1.0 / scale for scale in scales) / len(scales),
                )
            logging.info(
                'Schur decoder: reconstruction=%.6e regularization=%.6e '
                'relative_total=%.6e relative_delta=%.6e '
                'max|z|=%.6e max delta=%.6e max|D_hat|=%.6e',
                float(total_reconstruction),
                float(total_regularization),
                reported_total / num_reported,
                reported_delta / num_reported,
                largest_z,
                largest_delta,
                largest_prediction,
            )

        # per-epoch report through hydragnn/loss_reporting.py, plus the composite loss terms, which
        # need the batch (hydragnn/domain_losses/composite.py); the composite loss lives on the backbone
        total = total_reconstruction + total_regularization
        report = {
            'supervised.reconstruction': {
                'raw': total_reconstruction.detach(),
                'weight': 1.0,
                'weighted': total_reconstruction.detach(),
            },
            'supervised.regularization': {
                'raw': total_regularization.detach(),
                'weight': 1.0,
                'weighted': total_regularization.detach(),
            },
        }
        composite_loss = getattr(self.backbone, 'composite_loss', None)
        if data is not None and composite_loss is not None:
            extra, extra_report = composite_loss(pred, data, self)
            if extra is not None:
                total = total + extra
            report.update(extra_report)
        self.last_loss_components = report

        return total, [total_reconstruction]

    def __str__(self):
        return 'SchurDecoderModel(' + str(self.backbone) + ')'
