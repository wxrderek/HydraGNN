###########################################################################################
# The three decoder heads of the sequential Schur decoder
# Reference: references/schur_decoder.tex, Sections 5.1-5.5
# Author: Derek Wang
###########################################################################################

import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, Linear
from torch.utils.checkpoint import checkpoint


def build_mlp(input_dim: int, hidden_dims, output_dim: int, activation):
    '''plain MLP with the given hidden widths and no activation on the output'''

    layers = []
    width = input_dim
    for hidden in hidden_dims:
        layers.append(Linear(width, hidden))
        layers.append(activation)
        width = hidden
    layers.append(Linear(width, output_dim))
    return Sequential(*layers)


def nonnegative_map(name: str):
    '''f from Section 3.4, mapping alpha_k to a nonnegative delta_k'''

    # Section 10 of logs/qmugs-schur-diagnostics/schur_decomp_test.md measured delta_k spanning
    # about seven decades, so f has to be able to reach small positive values smoothly. softplus
    # is exponential below zero and linear above, which covers that without the overflow risk of
    # exp at large positive alpha.
    if name == 'softplus':
        return F.softplus
    if name == 'exp':
        return torch.exp
    raise ValueError(f"unsupported f: {name}; use 'softplus' or 'exp'")


class RadialBasis(Module):
    '''fixed Gaussian basis on the interatomic distance of Section 5.3'''

    def __init__(self, num_basis: int, cutoff: float):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_basis) # (num_basis,)
        self.register_buffer('centers', centers)
        # one spacing wide, so neighbouring Gaussians overlap at exp(-1)
        self.width = float(cutoff / max(num_basis - 1, 1))
        self.num_basis = num_basis

    def forward(self, distances):
        # distances is (..., ), the output is (..., num_basis)
        scaled = (distances.unsqueeze(-1) - self.centers) / self.width
        return torch.exp(-scaled ** 2)


class OrbitalEmbedding(Module):
    '''the per-orbital vector shared by all three heads'''

    # Not reference notation. Section 5.3 feeds v^{j(k)} to the heads directly; projecting it
    # together with the orbital metadata first is an implementation choice, made so the pair
    # head of Section 5.3 sees an input of width ~150 instead of ~530, which matters because
    # that head is evaluated on all K^2 ordered orbital pairs.

    def __init__(
        self,
        node_dim: int,
        n_l: int,
        n_m: int,
        n_shell: int,
        n_species: int,
        orbital_dim: int,
        hidden_dims,
        activation,
        conditioning_type: str,
        use_sequence_position: bool,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.n_l = n_l
        self.n_m = n_m
        self.n_shell = n_shell
        self.n_species = n_species
        self.conditioning_type = conditioning_type
        self.use_sequence_position = use_sequence_position
        self.mlp = build_mlp(self.conditioning_dim(), hidden_dims, orbital_dim, activation)

    def conditioning_dim(self):
        '''width of the vector this module's MLP consumes; must agree with conditioning()'''

        width = self.node_dim + self.n_l + self.n_m + self.n_shell + self.n_species
        if self.use_sequence_position:
            width += 1
        return width

    def conditioning(
        self,
        node_embedding,
        angular_momentum,
        index_within_shell,
        shell_within_atom,
        species_index,
    ):
        '''assemble the per-orbital input; change the parametric form of the conditioning here'''

        if self.conditioning_type != 'concat':
            raise ValueError(f'unsupported conditioning_type: {self.conditioning_type}')

        n_basis = int(node_embedding.shape[0])
        dtype = node_embedding.dtype
        parts = [
            node_embedding, # (n_basis, node_dim), this is v^{j(k)} of Section 5.1
            F.one_hot(angular_momentum.long(), self.n_l).to(dtype), # (n_basis, n_l)
            F.one_hot(index_within_shell.long(), self.n_m).to(dtype), # (n_basis, n_m)
            F.one_hot(shell_within_atom.long(), self.n_shell).to(dtype), # (n_basis, n_shell)
            F.one_hot(species_index.long(), self.n_species).to(dtype), # (n_basis, n_species)
        ]
        if self.use_sequence_position:
            # Section 10.3 of the diagnostics found difficulty to be a strong function of k/K
            # (median ||z_k|| runs 0.70 early to 0.028 in the last 5%), so the heads are given
            # that coordinate outright rather than being left to infer it
            position = torch.arange(n_basis, device=node_embedding.device, dtype=dtype) / n_basis
            parts.append(position.unsqueeze(1)) # (n_basis, 1)

        features = torch.cat(parts, dim=1) # (n_basis, conditioning_dim)
        assert features.shape == (n_basis, self.conditioning_dim())
        return features

    def forward(
        self,
        node_embedding,
        angular_momentum,
        index_within_shell,
        shell_within_atom,
        species_index,
    ):
        features = self.conditioning(
            node_embedding,
            angular_momentum,
            index_within_shell,
            shell_within_atom,
            species_index,
        )
        return self.mlp(features) # (n_basis, orbital_dim)


class GlobalEmbedding(Module):
    '''the per-molecule vector, which is how the graph and the a priori metadata enter'''

    # This resolves the "G" and "metadata" of Sections 5.2-5.4, which the reference leaves open.

    def __init__(
        self,
        pooled_dim: int,
        global_dim: int,
        hidden_dims,
        activation,
        conditioning_type: str,
    ):
        super().__init__()
        self.pooled_dim = pooled_dim
        self.conditioning_type = conditioning_type
        self.mlp = build_mlp(self.conditioning_dim(), hidden_dims, global_dim, activation)

    def conditioning_dim(self):
        '''width of the vector this module's MLP consumes; must agree with conditioning()'''

        # the pooled graph embedding plus four molecular scalars
        return self.pooled_dim + 4

    def conditioning(self, graph_embedding, n_atoms, n_basis, charge, multiplicity):
        '''assemble the per-molecule input; change the parametric form of the conditioning here'''

        if self.conditioning_type != 'concat':
            raise ValueError(f'unsupported conditioning_type: {self.conditioning_type}')

        dtype = graph_embedding.dtype
        device = graph_embedding.device
        # divisors written out so the scaling is visible rather than hidden in a normalizer;
        # they only put the four scalars on the same O(1) footing as the pooled embedding
        scalars = torch.tensor(
            [n_atoms / 100.0, n_basis / 1000.0, charge, multiplicity],
            dtype=dtype,
            device=device,
        ) # (4,)

        features = torch.cat([graph_embedding, scalars], dim=0) # (conditioning_dim,)
        assert features.shape == (self.conditioning_dim(),)
        return features

    def forward(self, graph_embedding, n_atoms, n_basis, charge, multiplicity):
        features = self.conditioning(graph_embedding, n_atoms, n_basis, charge, multiplicity)
        return self.mlp(features) # (global_dim,)


class ZHead(Module):
    '''the z_{k,i} decoder head of Section 5.3, Equation 12'''

    def __init__(
        self,
        orbital_dim: int,
        global_dim: int,
        num_radial_basis: int,
        radial_cutoff: float,
        hidden_dims,
        activation,
        conditioning_type: str,
        condition_on_graph: bool,
        row_chunk: int,
        use_checkpoint: bool,
    ):
        super().__init__()
        self.orbital_dim = orbital_dim
        self.global_dim = global_dim
        self.conditioning_type = conditioning_type
        self.condition_on_graph = condition_on_graph
        self.row_chunk = row_chunk
        self.use_checkpoint = use_checkpoint
        self.radial_basis = RadialBasis(num_radial_basis, radial_cutoff)
        self.mlp = build_mlp(self.conditioning_dim(), hidden_dims, 1, activation)

    def conditioning_dim(self):
        '''width of the vector this head's MLP consumes; must agree with conditioning()'''

        width = 2 * self.orbital_dim + self.radial_basis.num_basis
        if self.condition_on_graph:
            width += self.global_dim
        return width

    def conditioning(self, rows, columns, distances, g):
        '''assemble the pair input; change the parametric form of the conditioning here'''

        # rows is (num_rows, orbital_dim), columns is (n_basis, orbital_dim),
        # distances is (num_rows, n_basis)
        if self.conditioning_type != 'concat':
            raise ValueError(f'unsupported conditioning_type: {self.conditioning_type}')

        num_rows = int(rows.shape[0])
        n_basis = int(columns.shape[0])
        parts = [
            rows.unsqueeze(1).expand(num_rows, n_basis, self.orbital_dim), # v^{j(k)}
            columns.unsqueeze(0).expand(num_rows, n_basis, self.orbital_dim), # v^{j(i)}
            self.radial_basis(distances), # (num_rows, n_basis, num_basis)
        ]
        if self.condition_on_graph:
            parts.append(g.reshape(1, 1, -1).expand(num_rows, n_basis, self.global_dim))

        features = torch.cat(parts, dim=2) # (num_rows, n_basis, conditioning_dim)
        assert features.shape == (num_rows, n_basis, self.conditioning_dim())
        return features

    def _chunk(self, rows, columns, row_positions, positions, g):
        '''one row block of the pair MLP, wrapped by checkpoint when enabled'''

        distances = torch.cdist(row_positions, positions) # (num_rows, n_basis)
        features = self.conditioning(rows, columns, distances, g)
        return self.mlp(features).squeeze(-1) # (num_rows, n_basis)

    def forward(self, orbital_embedding, positions, g):
        '''return the strictly lower triangular latent block whose row k holds z_{k+1}'''

        # The head is shared across all k and i (Section 5.3), so this is one MLP applied to
        # every ordered orbital pair. z_{k,i} != z_{i,k} is correct and intended; only k > i is
        # ever read, which is what the tril below keeps.
        #
        # Rows are evaluated in blocks under checkpoint because the hidden activations are
        # O(K^2 * width): at K = 800 with width 128 that is about 660 MB retained, against about
        # 2.5 MB when only the (K, K) output survives and the block is recomputed in backward.
        n_basis = int(orbital_embedding.shape[0])
        assert positions.shape == (n_basis, 3)

        blocks = []
        for lo in range(0, n_basis, self.row_chunk):
            hi = min(lo + self.row_chunk, n_basis)
            arguments = (
                orbital_embedding[lo:hi],
                orbital_embedding,
                positions[lo:hi],
                positions,
                g,
            )
            if self.use_checkpoint and torch.is_grad_enabled():
                blocks.append(checkpoint(self._chunk, *arguments, use_reentrant=False))
            else:
                blocks.append(self._chunk(*arguments))

        pairs = torch.cat(blocks, dim=0) # (n_basis, n_basis)
        assert pairs.shape == (n_basis, n_basis)

        # only the strictly lower triangle is used: row k holds z_{k+1} in its first k entries.
        # The block rather than a list of rows is returned so the regularization of Section 4.3
        # can take all the ||z_k||^2 in one reduction; the caller slices the rows for the
        # recursion, which costs nothing since the slices are views.
        return torch.tril(pairs, diagonal=-1)


class AlphaHead(Module):
    '''the alpha_k decoder head of Section 5.4, Equation 13'''

    def __init__(
        self,
        orbital_dim: int,
        global_dim: int,
        hidden_dims,
        activation,
        conditioning_type: str,
        f: str,
        alpha_bias_init: float,
    ):
        super().__init__()
        self.orbital_dim = orbital_dim
        self.global_dim = global_dim
        self.conditioning_type = conditioning_type
        self.f = nonnegative_map(f)
        self.mlp = build_mlp(self.conditioning_dim(), hidden_dims, 1, activation)
        # start delta_k near the median diagonal of the targets rather than near f(0)
        self.mlp[-1].bias.data.fill_(alpha_bias_init)

    def conditioning_dim(self):
        '''width of the vector this head's MLP consumes; must agree with conditioning()'''

        return self.orbital_dim + self.global_dim

    def conditioning(self, orbital_embedding, g):
        '''assemble the input; change the parametric form of the conditioning here'''

        if self.conditioning_type != 'concat':
            raise ValueError(f'unsupported conditioning_type: {self.conditioning_type}')

        n_basis = int(orbital_embedding.shape[0])
        features = torch.cat(
            [orbital_embedding, g.reshape(1, -1).expand(n_basis, self.global_dim)], dim=1
        ) # (n_basis, conditioning_dim)
        assert features.shape == (n_basis, self.conditioning_dim())
        return features

    def forward(self, orbital_embedding, g):
        alpha = self.mlp(self.conditioning(orbital_embedding, g)).squeeze(-1) # (n_basis,)
        return self.f(alpha) # (n_basis,), the delta_k of Equation 8


class D1Head(Module):
    '''the d_1 decoder head of Section 5.2, Equation 11'''

    # delta_1 = d_1 identically, so this head duplicates AlphaHead at k = 1. It is kept separate
    # because Section 5.5 prescribes three heads; its output overwrites delta[0].

    def __init__(
        self,
        orbital_dim: int,
        global_dim: int,
        hidden_dims,
        activation,
        conditioning_type: str,
        f: str,
        alpha_bias_init: float,
    ):
        super().__init__()
        self.orbital_dim = orbital_dim
        self.global_dim = global_dim
        self.conditioning_type = conditioning_type
        self.f = nonnegative_map(f)
        self.mlp = build_mlp(self.conditioning_dim(), hidden_dims, 1, activation)
        self.mlp[-1].bias.data.fill_(alpha_bias_init)

    def conditioning_dim(self):
        '''width of the vector this head's MLP consumes; must agree with conditioning()'''

        return self.orbital_dim + self.global_dim

    def conditioning(self, first_orbital_embedding, g):
        '''assemble the input; change the parametric form of the conditioning here'''

        if self.conditioning_type != 'concat':
            raise ValueError(f'unsupported conditioning_type: {self.conditioning_type}')

        features = torch.cat([first_orbital_embedding.reshape(-1), g.reshape(-1)], dim=0)
        assert features.shape == (self.conditioning_dim(),)
        return features

    def forward(self, first_orbital_embedding, g):
        alpha = self.mlp(self.conditioning(first_orbital_embedding, g)).reshape(())
        return self.f(alpha) # scalar, nonnegative as Section 5.2 requires
