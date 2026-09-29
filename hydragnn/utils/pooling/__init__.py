from hydragnn.utils.pooling.BasePool import BasePool
from hydragnn.utils.pooling.ConcatPool import (
    ConcatPool,
    PCAConcatPool,
    IsomapConcatPool,
    tSNEConcatPool,
    UMAPConcatPool,
)
from hydragnn.utils.pooling.PConcatPool import PConcatPool, MLPConcatPool
from hydragnn.utils.pooling.ClusterConcatPool import (
    ClusterConcatPool,
    ConvConcatPool,
    AttnConcatPool,
)

# Registry of hierarchical poolers
# The global pooling methods of mean/sum/max stay in Base.py
# PConcatPool is abstract so it is not registered
_POOLERS = {
    "concat": ConcatPool,
    "pca_concat": PCAConcatPool,
    "isomap_concat": IsomapConcatPool,
    "tsne_concat": tSNEConcatPool,
    "umap_concat": UMAPConcatPool,
    "mlp_concat": MLPConcatPool,
    "cluster_concat": ClusterConcatPool,
    "conv_concat": ConvConcatPool,
    "attn_concat": AttnConcatPool,
}


def build_graph_pooling(pool_type, hidden_dim, max_nodes, **pool_config):
    """Instantiate the graph pooling module registered under pool_type."""
    if pool_type not in _POOLERS:
        raise ValueError(
            f"Unsupported graph_pooling framework requested: {str(pool_type)}\n"
            + "The following are available: " + ", ".join(sorted(_POOLERS))
        )
    return _POOLERS[pool_type](hidden_dim, max_nodes, **pool_config)


def graph_pooled_dim(pool_type, hidden_dim, max_nodes, **pool_config):
    """Flattened pooled width R*C, computed without instantiating the model."""
    r, c = _POOLERS[pool_type].compute_pooled_shape(hidden_dim, max_nodes, **pool_config)
    return int(r * c)
