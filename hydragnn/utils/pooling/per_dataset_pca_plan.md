# Plan: `fit_scope="per_dataset"` for the non-parametric pooling family (PCA only)

**Status: TABLED.** This is a design document, not an implementation trigger. No code is to
be written from it until explicitly requested. It records the agreed design for implementing
the `per_dataset` fit scope of `ConcatPool` so the decision surface is fixed before any edit.

---

## 0. Scope — PCA only, emphatically

This plan implements `fit_scope="per_dataset"` for **exactly one** member of the non-parametric
family: **PCA** (`PCAConcatPool`). It does **not** implement `per_dataset` for `IsomapConcatPool`,
`UMAPConcatPool`, or `tSNEConcatPool`. Those remain scaffolded stubs that raise
`NotImplementedError`. Section 6 states precisely why generalizing to them is hard; the short
version is that PCA is the only member whose fit is a fixed-size, deterministic, distributable
statistic with a closed-form out-of-sample transform, and the entire fit-once-then-freeze design
depends on all three of those properties.

The base `ConcatPool` (`F = identity`) and `fit_scope="per_molecule"` are unaffected.

---

## 1. The scientific framing: fit once, freeze, co-adapt

`per_dataset` PCA is implemented as **fit `F` once, freeze it, then train through it as a fixed
non-learnable linear layer.** Concretely, `F(x) = (x - mu) @ V`, where `mu` (shape `hidden_dim`)
and `V` (shape `hidden_dim x reduce_dim`) are computed once from a sample of training-set node
embeddings and then held constant for the entire run.

### Why this does not contradict the `ConcatPool` docstring warning

The `ConcatPool` docstring warns that *refitting `F` each epoch* rotates the head's input basis
between epochs and stales the head weights. That warning is about **recomputing `V` repeatedly**.
The design here does the opposite: `V` is computed **once and frozen**, so the head's input basis
is **stationary** for all of training. What changes during training is the encoder output fed into
that fixed basis — which is ordinary training. `F` ceases to equal "the exact PCA of the current
embeddings" after epoch 0, and that is acceptable: `F` has been demoted from an estimator that must
track the encoder to a **data-initialized fixed linear layer**. The staleness pathology is caused
by `V` changing; freezing `V` removes it. Refitting (`V` moves) and co-adapting (encoder moves, `V`
fixed) are different regimes, and only the former is what the docstring cautions against.

Connection worth noting: PCA-frozen `per_dataset` is formally identical to a one-layer
`MLPConcatPool` whose linear projection is initialized from data and then frozen. The
non-parametric constraint of this family *is* the freezing.

---

## 2. How the three deferral blockers dissolve

The `ConcatPool` docstring lists three open questions blocking `per_dataset`. Under the
fit-once-freeze design each collapses to a concrete, non-invasive mechanism.

1. **Chicken-and-egg (F needs embeddings from a model that does not yet exist).**
   Resolved by fitting `F` on a **warmed encoder** — weights loaded from a prior run's checkpoint
   (e.g. an identity-concat or mean-pool run) — rather than the random epoch-0 encoder. After the
   one-time fit, `F` is a fixed layer and the encoder/head co-adapt to it via backprop. No refit,
   no moving objective.

2. **No training-loop hook (`train_validate_test` has no per-dataset setup callback).**
   Resolved by fitting **before the training loop**, in the example training script, in the seam
   between raw model construction (`create_model_config`, `qmugs_densmat_train.py:524`) and DDP
   wrapping (`distributed_model_wrapper`, `:529`). `hydragnn.train.train_validate_test` is **never
   touched**.

3. **DDP (buffers broadcast from rank 0 only at wrap time).**
   Resolved by the structure of PCA itself. PCA depends on the data only through the mean and the
   second moment, which are **additive sufficient statistics**. Each rank accumulates `n`,
   `sum(x)`, and `sum(x x^T)` over its own shard; a `dist.all_reduce(SUM)` on the mean vector
   (`hidden_dim`) and the second-moment matrix (`hidden_dim x hidden_dim`) gives every rank the
   identical global statistics; each rank then runs the **same** eigendecomposition of a
   `hidden_dim x hidden_dim` (here 200x200) matrix and obtains the **same** `V`. No raw embeddings
   are gathered, communication is `O(hidden_dim^2)` independent of dataset size, and there is no
   fragile rank-0 buffer broadcast. This mirrors the existing `dist.all_reduce(..., MAX)` for
   `max_natoms` at `examples/qmugs/qmugs.py:911`. A fixed sign convention (each eigenvector signed
   so its largest-magnitude loading is positive) is applied so the frame is reproducible.

---

## 3. Decisions locked for this plan

1. **Encoder co-adapts (not frozen).** After the one-time fit, `V` and `mu` are frozen buffers,
   but the encoder and head continue to train normally. Correctness does not depend on `F`
   remaining the exact PCA of the evolving embeddings; see Section 1.
2. **Integrated pre-pass (Option A).** The fit runs inside `qmugs_densmat_train.py` between
   `:527` and `:529`. No standalone fit script and no serialized `F` artifact in this plan.
3. **Fit-source checkpoint is a parameter.** The encoder used for the fit is supplied by a config
   key (`fit_encoder_checkpoint`), not hard-coded.
4. **Exact PCA.** The full `hidden_dim x hidden_dim` covariance is eigendecomposed exactly
   (`torch.linalg.eigh`). No `pca_lowrank` sampling — a 200x200 eig is trivially cheap.

---

## 4. File-by-file change surface (all additive; no existing path removed)

| File | Change | Editability |
|---|---|---|
| `hydragnn/utils/pooling/ConcatPool.py` (and `BasePool.py` if a shared helper is cleaner) | When `fit_scope=="per_dataset"`, register buffers `pca_mean` (`hidden_dim`), `pca_proj` (`hidden_dim x reduce_dim`), and `fitted` (bool scalar) at `__init__` — shapes known up front, so `state_dict` save/load and the DDP-wrap broadcast handle them automatically. Add `fit_from_moments(n, sum_x, sum_xxT)`: form the covariance, `torch.linalg.eigh`, take the top `reduce_dim` eigenvectors, apply the sign convention, fill `pca_mean`/`pca_proj`, set `fitted=True`. `_reduce` applies `(x - pca_mean) @ pca_proj` when `per_dataset`. Validation: `per_dataset` now **requires** `reduce_dim` (no identity default); remove the construction-time `NotImplementedError`; `forward`/`_reduce` raise a clear error if `fitted` is False so a misconfigured run fails loudly rather than silently projecting through zeros. | Pooling submodule — freely editable |
| `hydragnn/models/Base.py` | Extract the encoder block of `forward` (`_embedding` through the conv/feature loop ending at `x = inv_node_feat`, ~`:763-789`) into a new method `encode(data) -> x`; `forward` calls `x = self.encode(data)`. Behavior is byte-identical. This lets the pre-pass obtain pre-pool node embeddings without paying for the ~2M-wide output head. | `Base.py` is the one model file permitted for edits; this is a behavior-preserving extraction adding a new path, not altering the old one |
| `examples/qmugs/qmugs_densmat_train.py` | New conditional block between `:527` and `:529`: if the configured pooler is a `per_dataset` `ConcatPool`, load `fit_encoder_checkpoint` into the raw (pre-wrap) model, stream a `fit_sample_size` subsample of `train_loader` through `model.encode`, accumulate `n`/`sum(x)`/`sum(x x^T)`, `dist.all_reduce(SUM)`, call `model.graph_pool.fit_from_moments(...)`, then proceed to the existing DDP wrap. Read the new config keys. The `train_validate_test` call is unchanged. | Example script |
| `examples/qmugs/qmugs_densmat.json` | `pool_config`: set `fit_scope: "per_dataset"`, `reduce_dim: <k>`, `fit_sample_size`, `fit_encoder_checkpoint: "<path>"`. | Config only |

Footprint summary: `hydragnn/` is touched only in the pooling submodule (open for edits) and by one
additive, behavior-preserving method in `Base.py`. The training loop, model backbone, and all other
files are untouched.

---

## 5. Reproducibility notes

- `mu` and `V` are `register_buffer`s of fixed shape, so the frozen `F` is part of the model
  `state_dict` and travels with every checkpoint automatically — a `per_dataset` run reloads its
  exact projection.
- The fit is deterministic given (fit-encoder checkpoint, `fit_sample_size`, sampling order, sign
  convention). The resolved `fit_encoder_checkpoint` and `fit_sample_size` are written into the
  saved config so a run is reproducible from its own `logs/<run>/config.json`, consistent with how
  `max_nodes` is already persisted.
- No random seeds, dataset splits, or evaluation protocol are changed by this plan.

---

## 6. Why generalizing to Isomap, UMAP, and t-SNE is hard

The fit-once-freeze design in Sections 1-2 relies on PCA having three properties simultaneously:
(a) its fit is a **fixed-size, additive** statistic (mean + second moment), (b) the fit is
**deterministic**, and (c) it admits a **closed-form out-of-sample transform** (one matrix
multiply). The manifold-learning members violate each of these, and t-SNE violates (c) fatally.

1. **No sufficient statistics; the fit is not decomposable across shards.**
   Isomap fits on geodesic distances over a kNN graph of the whole point set; UMAP fits a fuzzy
   simplicial (neighbor-set) topology over the whole set; t-SNE fits pairwise affinities over the
   whole set. None reduces to a fixed-size accumulator. The fit requires **all** node embeddings
   materialized together, which in DDP forces an all-gather of the entire training set's node
   embeddings to a single rank — `O(N * hidden_dim)` communication and memory, versus PCA's
   `O(hidden_dim^2)`. At QMugs node counts this is prohibitive, and it reintroduces blocker 3 in a
   worse form (a full-model, variable-size broadcast rather than a fixed 200x200 all_reduce).

2. **No, or no closed-form, out-of-sample transform.**
   - **t-SNE has no out-of-sample map at all.** It optimizes the coordinates of a fixed set
     directly; a new point has no defined image without re-solving the whole embedding. There is
     therefore no frozen `F` to apply at train/val/test time — the fit-once-apply-everywhere design
     has nothing to freeze. This alone rules t-SNE out of `per_dataset`.
   - **Isomap** has only an approximate Nystrom-style landmark extension; `F` would have to retain
     the landmark set and geodesic machinery and is not a single tensor op.
   - **UMAP** exposes a `.transform()` for new points, but it is itself an iterative optimization
     against the trained topology model run per batch, not a closed form, and it must carry the
     fitted model through every forward.

3. **Non-convex, stochastic fits are neither reproducible nor rank-consistent.**
   t-SNE and UMAP are non-convex optimizations with random initialization; two runs, or two ranks,
   on identical data produce different embeddings. PCA's eigendecomposition is deterministic (up to
   the pinned sign convention), so every rank agrees bit-for-bit with no broadcast. Making UMAP or
   t-SNE rank-consistent requires fitting on exactly one rank and broadcasting the entire fitted
   model.

4. **Variable-size fitted state breaks the buffer/checkpoint model.**
   PCA's frozen state is two fixed-shape tensors, so it lives as `register_buffer`s and serializes
   with `state_dict` for free. The manifold methods' fitted state (landmark sets, neighbor graphs,
   trained coordinates) is variable-size and does not fit the buffer model; it would need bespoke
   serialization outside the module.

5. **New dependencies.** Isomap and t-SNE would require `scikit-learn`; UMAP would require
   `umap-learn`. PCA requires only `torch.linalg`. Adding dependencies needs explicit permission
   and complicates the cross-cluster environment transfer.

In one sentence: PCA reduces `per_dataset` fitting to *accumulate a covariance, eigendecompose,
freeze two tensors* — fixed-size, deterministic, DDP-native, and checkpoint-native — while the
manifold methods violate each of those, and t-SNE additionally lacks the out-of-sample transform
the whole design is built on.

---

## 7. Open questions (deferred; not to be resolved in chat)

1. **Fit-source provenance.** What run produces `fit_encoder_checkpoint`? Options: a prior
   identity-concat run, a mean-pool run, or a short warmup of the same architecture. The plan
   leaves the path a free parameter but does not prescribe the run that generates it.
2. **Sampling policy for `fit_sample_size`.** Whether to draw the fit sample by whole molecules or
   by nodes, and whether to stratify by molecule size, since a node-uniform sample over-weights
   large molecules in the covariance.
3. **Interaction with `mask_output_loss` and the delta/upper-triangle target.** The fit uses node
   embeddings only, so it is nominally independent of the output masking and delta-learning
   settings, but this should be confirmed rather than assumed.
4. **Behavior on `continue`/resume.** On a resumed run, `load_existing_model_config` (`:543`) will
   restore the saved `pca_mean`/`pca_proj` buffers, which should take precedence over any re-fit.
   Confirm the pre-pass is skipped (or made idempotent) when `fitted` is already True in a loaded
   checkpoint.
5. **Optional exposure of an encoder-freeze switch.** The plan chooses co-adaptation. If a future
   experiment wants `F` to remain the exact PCA of a frozen encoder, a `freeze_encoder` training
   flag would be needed; not included here.
6. **Numerical guard on `reduce_dim` vs. covariance rank.** If `reduce_dim` exceeds the numerical
   rank of the sampled covariance, trailing eigenvectors are noise. Decide whether to warn, clamp,
   or error.
