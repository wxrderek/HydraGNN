import torch
from torch.nn import Module
from torch_geometric.utils import to_dense_batch


class BasePool(Module):
    """
    Base interface for graph pooling mechanisms.

    Maps node embeddings of shape (num_nodes_in_batch, hidden_dim) to a fixed-shape
    graph embedding of shape (num_graphs, R, C). One subclass per mechanism.

    Node embeddings are stored as rows. A pooled embedding (R, C) has R = number of pooled rows
    and C = embedding vector dimension. For non-clustered concatenation-based pooling, R is picked 
    to be the max number of nodes occurring in the dataset.
    """

    def __init__(self, hidden_dim: int, max_nodes: int = None, **pool_config):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        # method-specific structured shape (R, C)
        self.pooled_shape = self.compute_pooled_shape(hidden_dim, max_nodes, **pool_config)

        # flattened width of input to standard NN head
        self.pooled_dim = int(self.pooled_shape[0] * self.pooled_shape[1])

    @staticmethod
    def compute_pooled_shape(hidden_dim: int, max_nodes: int = None, **pool_config):
        """Return the (R, C) shape of the pooled embedding"""
        raise NotImplementedError

    def forward(self, x, batch, data=None):
        """Pool node embeddings (num_nodes_in_batch, hidden_dim) to (num_graphs, R, C)."""
        raise NotImplementedError

    def _densify(self, x, batch):
        """Zero-pad each graph to max_nodes in node order, giving (num_graphs, max_nodes, C)."""
        # x is (num_nodes_in_batch, C), where C is hidden_dim or a reduced width
        if self.max_nodes is None:
            raise ValueError(
                "_densify requires max_nodes; set 'max_nodes' in the pooling "
                "'pool_config', or let the training script fill it from the "
                "dataset metadata"
            )

        if batch is None:
            num_nodes = x.shape[0]
            if num_nodes > self.max_nodes:
                raise ValueError(
                    f"Graph has {num_nodes} nodes but max_nodes is {self.max_nodes}; "
                    "concatenation pooling would silently drop nodes"
                )
            dense = x.new_zeros((self.max_nodes, x.shape[1]))
            dense[:num_nodes] = x  # (max_nodes, C)
            return dense.unsqueeze(0)  # (1, max_nodes, C)

        batch = batch.to(x.device)
        
        max_graph_nodes = int(torch.bincount(batch).max().item())
        if max_graph_nodes > self.max_nodes:
            raise ValueError(
                f"Batch contains a graph with {max_graph_nodes} nodes but max_nodes "
                f"is {self.max_nodes}; concatenation pooling would silently drop nodes"
            )

        # zero-pad every graph to max_nodes
        dense, _ = to_dense_batch(x, batch, max_num_nodes=self.max_nodes)
        return dense  # (num_graphs, max_nodes, C)
