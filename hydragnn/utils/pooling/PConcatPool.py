from hydragnn.utils.pooling.BasePool import BasePool


class PConcatPool(BasePool):
    """
    Abstract base for parametric concatenation pooling.

    P(X) = [F_t(x_1) || ... || F_t(x_n) || 0 || ... ], where F_t is a learned node-wise map
    hidden_dim -> reduce_dim. Fixes the argument surface for the family. F_t itself is
    undefined here, so this class is NOT registered in _POOLERS. 

    Node ordering is preserved, so the pooled embedding stays alignable column-by-column
    with atomic-orbital blocks.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_nodes: int = None,
        reduce_dim: int = None,
        num_layers: int = 1,
        dim_layers=None,
        activation: str = "relu",
        bias: bool = True,
        use_node_metadata: bool = False,
        metadata_keys=None,
        metadata_dim: int = 0,
    ):
        super().__init__(
            hidden_dim,
            max_nodes,
            reduce_dim=reduce_dim,
            num_layers=num_layers,
            dim_layers=dim_layers,
            activation=activation,
            bias=bias,
            use_node_metadata=use_node_metadata,
            metadata_keys=metadata_keys,
            metadata_dim=metadata_dim,
        )
        if use_node_metadata and not metadata_keys:
            raise ValueError(
                "use_node_metadata=True requires 'metadata_keys' naming the per-node "
                "fields on the graph object to condition on"
            )
        self.reduce_dim = reduce_dim
        self.num_layers = num_layers
        self.dim_layers = dim_layers
        self.activation = activation
        self.bias = bias
        self.use_node_metadata = use_node_metadata
        self.metadata_keys = metadata_keys
        # width of the concatenated metadata; sizes the first Linear of F_t
        self.metadata_dim = metadata_dim

    @staticmethod
    def compute_pooled_shape(
        hidden_dim: int,
        max_nodes: int = None,
        reduce_dim: int = None,
        num_layers: int = 1,
        dim_layers=None,
        activation: str = "relu",
        bias: bool = True,
        use_node_metadata: bool = False,
        metadata_keys=None,
        metadata_dim: int = 0,
    ):
        """Pooled shape is (max_nodes, reduce_dim); both are required."""
        if max_nodes is None:
            raise ValueError(
                "parametric concatenation pooling requires max_nodes; set 'max_nodes' in "
                "the pooling 'pool_config', or let the training script fill it from the "
                "dataset metadata"
            )
        if reduce_dim is None:
            raise ValueError(
                "parametric concatenation pooling requires 'reduce_dim' in the pooling "
                "'pool_config'; there is no identity default for a learned map"
            )
        return (max_nodes, reduce_dim)

    def _reduce(self, x, batch, data):
        """Apply F_t node-wise, (num_nodes, hidden_dim) -> (num_nodes, reduce_dim)."""
        raise NotImplementedError

    def forward(self, x, batch, data=None):
        # x is (num_nodes_in_batch, hidden_dim)
        x_red = self._reduce(x, batch, data)  # (num_nodes_in_batch, reduce_dim)
        return self._densify(x_red, batch)  # (num_graphs, max_nodes, reduce_dim)


class MLPConcatPool(PConcatPool):
    """
    PConcatPool with F_t a node-wise MLP reducing hidden_dim -> reduce_dim.

    The recommended first parametric variant and the only one in this family needing no new
    dependency. Layer widths follow the num_headlayers / dim_headlayers convention used by
    the output heads in hydragnn/models/Base.py.
    """

    def _reduce(self, x, batch, data):
        raise NotImplementedError
