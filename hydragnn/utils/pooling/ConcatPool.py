from hydragnn.utils.pooling.BasePool import BasePool

# Fit scopes for the non-parametric reductions F in the child classes below.
# "per_molecule": F is refit inside forward() for each molecule, no persistent state.
# "per_dataset":  F is fit once on training embeddings and frozen. NOT IMPLEMENTED
_FIT_SCOPES = ("per_molecule", "per_dataset")


class ConcatPool(BasePool):
    """
    Concatenate node embeddings in node order, zero-padded to max_nodes

    Base class of the non-parametric family: P(X) = [F(x_1) || ... || F(x_n) || 0 || ... ],
    with F = identity here and F in {PCA, Isomap, tSNE, UMAP} in the child classes.
    Preserves node ordering so the pooled embedding can be aligned column-by-column with
    atomic-orbital blocks. Has no learnable parameters.

    fit_scope="per_molecule" is the default because it is the only scope implementable.
    Note that refitting F per molecule gives each molecule a different embedding space, 
    so the reduced coordinates are not comparable across molecules.

    fit_scope="per_dataset" raises NotImplementedError at construction, pending a decision on:

      1. Chicken-and-egg. F must be fit on node embeddings produced by the model being
         trained, which do not exist before training. Fitting once on the epoch-0 random
         encoder fits noise; refitting each epoch rotates the head's input basis between
         epochs, staling its weights and making the loss a moving objective.
      2. No hook. hydragnn.train.train_validate_test has no per-dataset setup callback, so
         fit() must either be called from an edit to that file or lazily on the first
         forward, which silently fits random-encoder embeddings.
      3. DDP. Buffers are broadcast from rank 0 only at DDP wrapping. A fit performed later
         and independently per rank gives each rank a different projection while ranks share
         gradients. It needs an explicit dist.broadcast (or gather, fit on rank 0, broadcast).
    """

    def __init__(
        self,
        hidden_dim: int,
        max_nodes: int = None,
        reduce_dim: int = None,
        fit_scope: str = "per_molecule",
        fit_sample_size: int = 100000,
    ):
        super().__init__(
            hidden_dim,
            max_nodes,
            reduce_dim=reduce_dim,
            fit_scope=fit_scope,
            fit_sample_size=fit_sample_size,
        )
        if fit_scope not in _FIT_SCOPES:
            raise ValueError(
                f"fit_scope must be one of {_FIT_SCOPES}, got '{fit_scope}'"
            )
        if fit_scope == "per_dataset":
            raise NotImplementedError(
                "fit_scope='per_dataset' is scaffolded but not implemented; see the "
                "ConcatPool docstring for the 3 open questions blocking it"
            )
        
        self.reduce_dim = reduce_dim
        self.fit_scope = fit_scope
        self.fit_sample_size = fit_sample_size

    @staticmethod
    def compute_pooled_shape(
        hidden_dim: int,
        max_nodes: int = None,
        reduce_dim: int = None,
        fit_scope: str = "per_molecule",
        fit_sample_size: int = 100000,
    ):
        """Pooled shape is (max_nodes, reduce_dim or hidden_dim)."""
        if max_nodes is None:
            raise ValueError(
                "graph_pooling='concat' requires max_nodes; set 'max_nodes' in the "
                "pooling 'pool_config', or let the training script fill it from the "
                "dataset metadata"
            )
        return (max_nodes, hidden_dim if reduce_dim is None else reduce_dim)

    def fit(self, x_sample):
        """Fit F once on a sample of training embeddings and freeze it; no-op for F = identity."""
        return

    def _reduce(self, x, batch):
        """Apply F node-wise, (num_nodes, hidden_dim) -> (num_nodes, reduce_dim); identity here."""
        return x

    def forward(self, x, batch, data=None):
        # x is (num_nodes_in_batch, hidden_dim)
        x_red = self._reduce(x, batch)  # (num_nodes_in_batch, C)
        return self._densify(x_red, batch)  # (num_graphs, max_nodes, C)


class PCAConcatPool(ConcatPool):
    """
    ConcatPool with F = PCA reducing node embeddings to reduce_dim before concatenation.

    Implementable with torch.pca_lowrank / torch.linalg.svd, so it needs no new dependency;
    the natural first child of this family to fill in.
    """

    def fit(self, x_sample):
        raise NotImplementedError

    def _reduce(self, x, batch):
        raise NotImplementedError


class IsomapConcatPool(ConcatPool):
    """
    ConcatPool with F = Isomap reducing node embeddings to reduce_dim before concatenation.

    Would require scikit-learn, which is NOT a dependency of this project yet.
    """

    def fit(self, x_sample):
        raise NotImplementedError

    def _reduce(self, x, batch):
        raise NotImplementedError


class tSNEConcatPool(ConcatPool):
    """
    ConcatPool with F = t-SNE reducing node embeddings to reduce_dim before concatenation.

    Would require scikit-learn, which is NOT a dependency of this project yet. t-SNE has no
    out-of-sample transform, so this child can only ever run fit_scope="per_molecule".
    """

    def fit(self, x_sample):
        raise NotImplementedError

    def _reduce(self, x, batch):
        raise NotImplementedError


class UMAPConcatPool(ConcatPool):
    """
    ConcatPool with F = UMAP reducing node embeddings to reduce_dim before concatenation.

    Would require umap-learn, which is NOT a dependency of this project yet.
    """

    def fit(self, x_sample):
        raise NotImplementedError

    def _reduce(self, x, batch):
        raise NotImplementedError
