# Pooling

Author: Derek Wang

This new submodule contains classes for hierarchical graph pooling methods.

Note the working assumption is that global pooling methods are integrated directly into `Base.py`. More sophisticated hierarchical pooling methods live in this submodule.

The first pass primarily targets integration with `EGNN` and `PAINN` type message passing.
We have also not integrated this with the multitask framework, though this would be an
action item for the module.

The pooling operations here assume accept to ONLY the output of the FINAL message passing layer
and possibly a priori loaded metadata, but nothing else (for now). 

Note that several of these pooling mechanisms do NOT preserve permutation invariance.
In particular, any method that concatenates node embeddings do not preserve 
permutation invariance (though some may preserve permutation equivariance). 
Thus, many of these methods are only suitable for graph-level targets with a fixed,
canonical node ordering (i.e. density matrices in a fixed basis). 

Conveniently, some methods in here do retain permutation invariance, and can thus
be used for more arbitrary tasks. 


## Documentation

### The `BasePool` contract

Every pooler is an `nn.Module` subclassing `BasePool` and provides three things.

- `compute_pooled_shape(hidden_dim, max_nodes=None, **pool_config) -> (R, C)`, a
  **staticmethod**. It returns the structured shape of the graph embedding. It is static so
  the training script can obtain the pooled width *before* the model is built, which is what
  the output-head auto-widening in `examples/qmugs/qmugs_densmat_train.py` needs.
- `forward(x, batch, data=None)`. Maps node embeddings `(num_nodes_in_batch, hidden_dim)` to
  a graph embedding `(num_graphs, R, C)`. `data` is the PyG batch object, carrying the a
  priori metadata some methods condition on; poolers that do not need it ignore it. Every
  `forward` must handle the single-graph case `batch is None`.
- `_densify(x, batch)`, inherited. Zero-pads each graph to `max_nodes` in node order and
  raises if a graph exceeds `max_nodes` — `to_dense_batch` truncates silently otherwise, so
  this guard is the only thing standing between a config error and silently dropped atoms.

`__init__` derives `pooled_shape = (R, C)` and `pooled_dim = R * C` from
`compute_pooled_shape`, so the width is never written down twice. **Nodes are rows**: an
embedding `(R, C)` has `R` pooled rows and `C` per-row width, the transpose of the design
doc's notation. `hydragnn/models/Base.py` flattens `(num_graphs, R, C)` to
`(num_graphs, pooled_dim)` at the head boundary; that flatten is the single seam a future
structure-aware head would remove, which is why `pooled_shape` is kept alongside `pooled_dim`.

### The three families

**`ConcatPool`** (`ConcatPool.py`) — non-parametric. `P(X) = [F(x_1) || ... || F(x_n) || 0 ...]`,
zero-padded to `max_nodes`. The base class is `F = identity`, i.e. plain concatenation, and is
parameter-free. Children take `F` from classical dimensionality reduction.

| arg | default | meaning |
|---|---|---|
| `max_nodes` | required | padding width; see below on how it is resolved |
| `reduce_dim` | `null` | output width `k`; `null` means `F` = identity |
| `fit_scope` | `"per_molecule"` | where `F` is fit; see the caveat below |
| `fit_sample_size` | `100000` | embeddings drawn for a `"per_dataset"` fit (inert for now) |

Children: `PCAConcatPool`, `IsomapConcatPool`, `tSNEConcatPool`, `UMAPConcatPool`. Only PCA is
implementable with what is already installed (`torch.pca_lowrank` / `torch.linalg.svd`); Isomap
and t-SNE would need `scikit-learn` and UMAP would need `umap-learn`, none of which are
dependencies of this project.

**`PConcatPool`** (`PConcatPool.py`) — parametric. Same concatenation, but `F_t` is a learned
node-wise map. The class is abstract and unregistered; use `MLPConcatPool`.

| arg | default | meaning |
|---|---|---|
| `max_nodes` | required | padding width |
| `reduce_dim` | **required** | output width `k`; no identity default for a learned map |
| `num_layers` / `dim_layers` | `1` / `null` | depth and hidden widths of `F_t`, following the `num_headlayers` / `dim_headlayers` convention of the output heads |
| `activation` / `bias` | `"relu"` / `true` | |
| `use_node_metadata` / `metadata_keys` / `metadata_dim` | `false` / `null` / `0` | condition `F_t` on a priori atom or orbital labels read off the graph object |

**`ClusterConcatPool`** (`ClusterConcatPool.py`) — cluster-then-concatenate. Partitions each
graph's node indices into `c` consecutive chunks, aggregates within each chunk, and
concatenates the results. It emits a fixed `c` rows regardless of node count, so it needs **no
`max_nodes` and does no padding** — `|I_i|` varies across samples instead. This is the
memory-relevant family, since `c` is typically far below the dataset's maximum atom count.
Clusters are index-local, not geometry-local: membership follows the canonical node ordering
that fixes the atomic-orbital block ordering, not spatial adjacency.

| arg | default | meaning |
|---|---|---|
| `num_clusters` | **required** | `c`, the fixed row count |
| `cluster_scheme` | `"contiguous"` | `"contiguous"`, `"fixed_size"`, or `"metadata"` |
| `cluster_key` | `null` | per-node field holding the assignment, when `cluster_scheme="metadata"` |
| `aggregation` | `"mean"` | `"mean"`, `"max"`, `"sum"`, `"attn"`, `"mlp"` |
| `reduce_dim` | `null` | optional width applied to the aggregated cluster embeddings |

Children: `ConvConcatPool`, `AttnConcatPool`. Note `ConvConcatPool` breaks the family's shape
formula — it appends each convolution aggregate at position `j`, so its output has `max_nodes`
rows rather than `c`, and it overrides `compute_pooled_shape` accordingly.

### Registry and configuration

`_POOLERS` in `__init__.py` maps a config string to a class; adding a mechanism means writing
one class and one row. Current keys: `concat`, `pca_concat`, `isomap_concat`, `tsne_concat`,
`umap_concat`, `mlp_concat`, `cluster_concat`, `conv_concat`, `attn_concat`. The global
reductions (`mean`, `add`, `sum`, `max`) are deliberately absent — they stay in `Base.py`.

`graph_pooling` in a `config.json` accepts either a bare string (legacy, e.g. `"mean"`) or an
object:

```json
"graph_pooling": {
  "type": "concat",
  "pool_config": {
    "reduce_dim": null,
    "fit_scope": "per_molecule"
  }
}
```

Keys in `pool_config` are passed straight to the pooler's `__init__`, so an unrecognized key
raises a `TypeError` at model construction rather than failing silently at runtime.

**On `max_nodes`.** Omit it. Resolution order is `pool_config.max_nodes`, then the top-level
`max_nodes`, then the dataset metadata — and the metadata value is the real one, computed
during preprocessing as the maximum atom count over all three splits and all-reduced across
ranks. Leaving it out of the config is what selects that value; pin it only to reproduce an
older model. The training script writes the resolved value back into the saved config so a run
is reproducible from its own `logs/<run>/config.json`.

### Standing caveats

- **`fit_scope="per_molecule"` is the default because it is implementable, not because it is
  sound.** Refitting `F` per molecule puts every molecule in its own eigenbasis, so the reduced
  coordinates are not comparable across graphs.
- **`fit_scope="per_dataset"` is scaffolded but raises `NotImplementedError` at construction.**
  Three questions block it: (1) `F` must be fit on embeddings produced by the model being
  trained, which do not exist before training — fitting on the epoch-0 random encoder fits
  noise, refitting each epoch rotates the head's input basis and stales its weights; (2)
  `hydragnn.train.train_validate_test` has no per-dataset setup hook to call `fit()` from; (3)
  buffers are broadcast from rank 0 only at DDP wrapping, so a later per-rank fit gives each
  rank a different projection while ranks share gradients, and an explicit `dist.broadcast` is
  required.
- **t-SNE has no out-of-sample transform**, so `tSNEConcatPool` could only ever run
  `"per_molecule"` even once `"per_dataset"` exists.