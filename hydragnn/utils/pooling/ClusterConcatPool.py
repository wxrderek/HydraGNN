from hydragnn.utils.pooling.BasePool import BasePool

# How the index chunks I_1, ..., I_c are formed for each graph.
# "contiguous": num_clusters equal consecutive chunks of that graph's own node count.
# "fixed_size": consecutive chunks of a fixed width, padded out to num_clusters.
# "metadata":   read a per-node cluster assignment off the graph object, e.g. an
#               atom-orbital block index recorded during preprocessing.
_CLUSTER_SCHEMES = ("contiguous", "fixed_size", "metadata")

# Cluster-local aggregation F applied to each chunk.
_AGGREGATIONS = ("mean", "max", "sum", "attn", "mlp")


class ClusterConcatPool(BasePool):
    """
    Aggregate node embeddings within consecutive index chunks, then concatenate the chunks.

    P(X) = [F(X_1) || ... || F(X_c)], where I_1, ..., I_c partition the graph's node indices
    and X_i = {x_j : j in I_i}. Emits a fixed c rows regardless of node count, so unlike the
    ConcatPool families it needs no max_nodes and does no zero padding; |I_i| varies across
    samples instead. This is the memory-relevant family: c is typically much smaller than the
    dataset's maximum atom count.

    Clusters are index-local, not geometry-local; chunk membership follows the canonical node
    ordering (which fixes the atomic-orbital block ordering), not spatial adjacency.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_nodes: int = None,
        num_clusters: int = None,
        cluster_scheme: str = "contiguous",
        cluster_key: str = None,
        aggregation: str = "mean",
        reduce_dim: int = None,
    ):
        super().__init__(
            hidden_dim,
            max_nodes,
            num_clusters=num_clusters,
            cluster_scheme=cluster_scheme,
            cluster_key=cluster_key,
            aggregation=aggregation,
            reduce_dim=reduce_dim,
        )
        if cluster_scheme not in _CLUSTER_SCHEMES:
            raise ValueError(
                f"cluster_scheme must be one of {_CLUSTER_SCHEMES}, got '{cluster_scheme}'"
            )
        if aggregation not in _AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {_AGGREGATIONS}, got '{aggregation}'"
            )
        if cluster_scheme == "metadata" and cluster_key is None:
            raise ValueError(
                "cluster_scheme='metadata' requires 'cluster_key' naming the per-node "
                "field on the graph object holding the cluster assignment"
            )
        self.num_clusters = num_clusters
        self.cluster_scheme = cluster_scheme
        self.cluster_key = cluster_key
        self.aggregation = aggregation
        self.reduce_dim = reduce_dim

    @staticmethod
    def compute_pooled_shape(
        hidden_dim: int,
        max_nodes: int = None,
        num_clusters: int = None,
        cluster_scheme: str = "contiguous",
        cluster_key: str = None,
        aggregation: str = "mean",
        reduce_dim: int = None,
    ):
        """Pooled shape is (num_clusters, reduce_dim or hidden_dim); independent of max_nodes."""
        if num_clusters is None:
            raise ValueError(
                "cluster concatenation pooling requires 'num_clusters' in the pooling "
                "'pool_config'; it fixes the row count of the graph embedding"
            )
        return (num_clusters, hidden_dim if reduce_dim is None else reduce_dim)

    def _assign_clusters(self, x, batch, data):
        """Return the per-node cluster index, shape (num_nodes_in_batch,), in [0, num_clusters)."""
        raise NotImplementedError

    def _aggregate(self, x, cluster_index, batch):
        """Reduce each (graph, cluster) segment with F, giving (num_graphs, num_clusters, C)."""
        raise NotImplementedError

    def forward(self, x, batch, data=None):
        # x is (num_nodes_in_batch, hidden_dim)
        raise NotImplementedError


class ConvConcatPool(ClusterConcatPool):
    """
    Convolve over index-local neighborhoods of the node axis, then concatenate.

    Index-aware rather than geometry-aware: x_j and x_{j+1} may be spatially distant but
    their density-matrix columns are adjacent, which is what these kernels see.

    NOTE the shape differs from the rest of this family. Per the design doc, the aggregate of
    the convolutions around x_j is appended at position j, so the output has max_nodes rows,
    not num_clusters. This class therefore OVERRIDES compute_pooled_shape to
    (max_nodes, channels[-1]) and does require max_nodes.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_nodes: int = None,
        channels=None,
        kernel_size: int = 3,
        stride: int = 1,
        **pool_config,
    ):
        raise NotImplementedError

    @staticmethod
    def compute_pooled_shape(
        hidden_dim: int,
        max_nodes: int = None,
        channels=None,
        kernel_size: int = 3,
        stride: int = 1,
        **pool_config,
    ):
        """Pooled shape is (max_nodes, channels[-1]); overrides the family formula."""
        raise NotImplementedError


class AttnConcatPool(ClusterConcatPool):
    """
    Aggregate index-local neighborhoods with learned attention weights, then concatenate.

    Generalizes ConvConcatPool's fixed kernels to learned, index-localized attention over a
    window of consecutive node indices.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_nodes: int = None,
        num_heads: int = 1,
        window: int = 3,
        **pool_config,
    ):
        raise NotImplementedError

    def _aggregate(self, x, cluster_index, batch):
        raise NotImplementedError
