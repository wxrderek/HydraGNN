# QMugs density matrix learning — recap of runs to date

Compiled 2026-08-27 from `logs/`, `examples/qmugs/*-job-*.out`, `/work/xw262/qmugs`,
`/work/xw262/artifacts`, and the saved per-run `config.json` files. Schur-decoder work is
excluded. Nothing in this document was produced by re-running anything; every number is read
off an existing log, config, or file size.

---

## 1. Summary in three sentences

Ten training runs exist: four on Perlmutter (plus two early ones whose configs live only on
Perlmutter) and six on DCC. Every run that trained to convergence collapsed to what is
effectively a mean predictor — on DCC the three architectural variants agree on the test loss
to three significant figures, and on Perlmutter the three delta-learning variants agree to two
— so **no architecture tried so far has produced a signal that separates it from the others**.
The only two models ever evaluated with real (unpadded, reconstructed) error metrics reach
relative Frobenius error 0.68–0.73 and per-matrix R² 0.46–0.53, i.e. the predictions capture
roughly half the variance of the target and are not yet chemically useful.

---

## 2. Datasets

Naming: `QMugs<perc_load>[_max<N_pad>][_delta][_uptr]`. `perc_load` is the fraction of QMugs
loaded (0.001 → `0001`, 0.01 → `001`, 0.1 → `01`); `max<N>` is `--max_density_matrix_size`,
which both filters out molecules whose density matrix exceeds `N` and (with
`--update_max_padded_dimension`) sets the padded dimension `N_pad = N`.

| Pickle | Machine | Target | `N_pad` | Head width | train / val / test | max natoms | Status |
|---|---|---|---|---|---|---|---|
| `QMugs001.pickle` | PM | `density_matrix` | 2002 | 4,008,004 | 15940 / 1990 / 1996 | — | used (55245318) |
| `QMugs001_new.pickle` | PM | `density_matrix` | 2002 | 4,008,004 | 15940 / 1990 / 1996 | — | used (55581365, inference) |
| `QMugs001_mask.pickle` | PM | `density_matrix` + mask | 2002 | 4,008,004 | 15940 / 1990 / 1996 | — | **prepared, never trained** |
| `QMugs001_max800.pickle` | PM | `density_matrix` | 800 | 640,000 | 14867 / 1853 / 1860 | — | used (55833314) |
| `QMugs001_max800_mask.pickle` | PM | `density_matrix` + mask | 800 | 640,000 | 14867 / 1853 / 1860 | — | **prepared, never trained** |
| `QMugs001_max800_delta.pickle` | PM | `density_matrix_delta` | 800 | 640,000 | 14867 / 1853 / 1860 | — | used (55914623, 55988061, 55991092) |
| `QMugs001_max800_delta_uptr.pickle` | PM | `density_matrix_delta_uptr` | 800 | 320,400 | 14867 / 1853 / 1860 | — | **prepared, never trained** |
| `QMugs0001_max500_delta_uptr.pickle` | DCC | `density_matrix_delta_uptr` | 500 | 125,250 | 636 / 78 / 84 | 53 | used (4 runs) — toy |
| `QMugs01_max500_delta_uptr.pickle` | DCC | `density_matrix_delta_uptr` | 500 | 125,250 | 63983 / 7998 / 7990 | 60 | used (2 runs, both aborted) |
| `QMugs01_max800_delta_uptr.pickle` | DCC | `density_matrix_delta_uptr` | 800 | 320,400 | 148636 / 18574 / 18589 | 93 | **prepared, never trained** — largest available |

Promolecular baselines (needed to reconstruct total densities from delta predictions) exist
under `/work/xw262/artifacts/` for all three DCC datasets: 266, 26 684 and 61 998 molecules
respectively.

### Padding dilution (important for reading any loss)

From `examples/qmugs/eda/densmat_eda.json` (664 303 molecules; density matrix dimension
min 47, median 534, mean 555, p90 744, max 2002):

| `N_pad` | molecules retained | mean fraction of padded entries that are real |
|---|---|---|
| 500 | 40.2 % | 0.686 |
| 800 | 93.3 % | 0.448 |
| 2002 | 100 % | 0.085 |

Reported losses are means over **all** padded entries. Since the padded region is exactly zero
and trivially learned, a first-order correction to a "per real entry" loss is
`L_real ≈ L_reported / f`. That correction is applied in §5.

---

## 3. Training runs — Perlmutter

All Perlmutter runs used DDP (`HYDRAGNN_USE_FSDP=0`), `batch_size=32` per rank, fp32, AdamW at
`lr=1e-3`, `perc_train=0.8`, MSE loss, early stopping with `patience=20`. `NN1` = 1 node =
4×A100-40GB; `NN2` = 2 nodes = 8 GPUs. "Bottleneck" is the width of the last hidden layer
feeding the output linear — it upper-bounds the rank of the affine family of matrices the head
can emit.

| Run | Dataset / target | MPNN | hidden | conv | pooling | shared → head | bottleneck | params | epochs | **test loss (MSE)** | mem alloc/reserved per GPU | s/epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `55245318` (NN2) | `QMugs001`, raw | EGNN | 200 | 4 | mean | 2×50 → [50, 25] | 25 | ~105 M | 66 (ES) | **3.908e-5** | 8.16 / 9.70 GB | — |
| `55581365` (NN2) | `QMugs001_new`, raw | EGNN | 200 | 4 | mean | — | — | — | 10 (killed, 154 min wall) | 5.294e-4 | 9.98 / 10.34 GB | — |
| `55833314` (NN1) | `QMugs001_max800`, raw | EGNN | 200 | 4 | mean | 2×50 → [50, 100] | 100 | 65.6 M | 86 (ES) | **4.319e-4** | 2.42 / 3.53 GB | 28 |
| `55914623` (NN1) | `QMugs001_max800_delta` | EGNN | 400 | 4 | mean | 1×400 → [400, 1000] | 1000 | 645 M | 112 (ES) | **2.486e-4** | 14.6 / 15.5 GB | 37 |
| `55988061` (NN1) | `QMugs001_max800_delta` | PAINN | 400 | 4 (r=5.0, k=50) | mean | 1×400 → [400, 1000] | 1000 | 648 M | 112 (ES) | **2.470e-4** | 16.9 / 19.0 GB | 30 |
| `55991092` (NN1) | `QMugs001_max800_delta` | PAINN | 400 | 6 (r=7.0, k=5) | **sum** | 1×400 → [400, 1000] | 1000 | 652 M | 102 (ES) | **2.459e-4** | 19.4 / 23.9 GB | 31 |

ES = stopped by early stopping. Parameter counts for the four runs with local checkpoints are
derived from checkpoint size (`bytes = 12 × params` for fp32 weights + two AdamW moments) and
agree with hand-counted layer widths to <1 %.

Notes:
- The three delta runs (`55914623`, `55988061`, `55991092`) differ in message passing (EGNN vs
  PAINN), depth (4 vs 6), cutoff (5.0 vs 7.0 Å), neighbour cap (50 vs 5) and pooling (mean vs
  sum), yet land within **1.1 %** of each other on test loss. Their epoch-1 losses
  (2.582e-4, 2.581e-4, 2.580e-4) are already within 0.1 % of their epoch-100 losses. Nearly all
  of the "learning" happens in epoch 0 and consists of matching the dataset mean.
- Raw-density `55833314` → delta `55914623` on the same dataset halves the reported MSE, but
  the delta target is intrinsically smaller in magnitude, so this is not by itself evidence of
  a better model. See §5.

---

## 4. Training runs — DCC

All DCC runs used FSDP2 `FULL_SHARD` (`HYDRAGNN_USE_FSDP=1`, `HYDRAGNN_FSDP_VERSION=2`),
`batch_size=8` per rank, fp32, AdamW at `lr=1e-3`, `HYDRAGNN_MAX_NUM_BATCH=1000`.
`NN1` = 4×RTX-A5000-24GB, `NN2` = 8. All targets are `density_matrix_delta_uptr`.

| Run | Dataset | loss | hidden | pooling | head | bottleneck | params | epochs | **test loss** | mem alloc/reserved per GPU | time/epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `50518658` (NN1) | `QMugs0001_max500` (798) | MSE | 200 | concat (`max_nodes`=53) | [10600] | 10600 | 1443 M | 5 (smoke test) | 3.429e-4 | **16.1 / 23.7 GB** | 52 s |
| `50932324` (NN1) | `QMugs0001_max500` (798) | MAE | 200 | sum | [400, 1200] | 1200 | 154 M | 71 (ES @70) | **4.865e-3** | 1.93 / 3.62 GB | 4.8 s |
| `50932742` (NN1) | `QMugs0001_max500` (798) | MAE | 200 | concat (`max_nodes`=53) | [10600] | 10600 | 1443 M | 40 (ES @39) | **4.870e-3** | **16.1 / 23.7 GB** | 49 s |
| `50932793` (NN1) | `QMugs0001_max500` (798) | MAE | 200 | sum | [400, 1600, 8000] | 8000 | 1019 M | 71 (ES @70) | **4.858e-3** | 11.4 / 22.7 GB | 27 s |
| `51082693` (NN2) | `QMugs01_max500` (79 971) | MAE | 256 | sum | [400, 3200] | 3200 | 407 M | 10, **all NaN** | NaN | 4.17 / 8.90 GB | 58 min |
| `51083128` (NN2) | `QMugs01_max500` (79 971) | MAE | 256 | sum | [400, 3200, 10000] | 10000 | 1291 M | 3 (killed at 30 h wall) | 5.095e-3 @ ep2 | 12.6 / 21.7 GB | 3 h 12 min |

Notes:
- **The three converged DCC runs agree to 0.25 % on test loss** (4.858e-3, 4.865e-3, 4.870e-3)
  despite spanning 154 M–1443 M parameters, sum vs concat pooling, and bottleneck widths 1200
  vs 8000 vs 10600. Combined with train ≈ val ≈ test throughout, this is a clean signature of
  mean collapse, not of overfitting or of a capacity limit. Note also that these four runs used
  the 798-conformer toy dataset, so they cannot distinguish architectures anyway.
- `51082693` produced `nan` from epoch 0 onward and ran for 10 epochs regardless. This is the
  only NaN in the record. It shares hidden dim, pooling, dataset and optimizer with `51083128`,
  which did not NaN; the only architectural difference is the head ([400, 3200] vs
  [400, 3200, 10000]). Cause not diagnosed.
- Both `NN2` runs hit the 30 h wall clock. At 1000 capped batches/epoch, the 10000-wide head
  costs **3 h 12 min/epoch**, 3.3× the 3200-wide head's 58 min. Any real training run on
  `QMugs01_*` needs either a smaller head, more nodes, or checkpoint-restart.
- `50518658` and `50932742` (concat pooling, `max_nodes`=53 → pooled width 10 600) sit at
  23.7 GB reserved on a 24 GB card. Concat pooling on `QMugs01_max800`
  (`max_natoms` = 93 → pooled width 93×hidden) will not fit at these head sizes.
- No run in the record used masked loss: `mask_output_loss.mask` is `false` in every saved
  `config.json`, even though two masked datasets were prepared on Perlmutter.

---

## 5. Putting the losses on a common footing

The reported losses are not comparable across runs: they mix MSE and MAE, different padded
dimensions, and raw vs delta targets. Dividing by the real-entry fraction `f` from §2 gives an
approximate per-real-entry loss, and for the MSE runs a per-entry RMS error.

| Run | target | loss | `N_pad` | `f` | reported test | per-real-entry | RMS error per entry |
|---|---|---|---|---|---|---|---|
| `55245318` | raw | MSE | 2002 | 0.085 | 3.908e-5 | 4.6e-4 | 0.0214 |
| `55833314` | raw | MSE | 800 | 0.448 | 4.319e-4 | 9.64e-4 | 0.0310 |
| `55914623` | delta | MSE | 800 | 0.448 | 2.486e-4 | 5.55e-4 | 0.0236 |
| `55988061` | delta | MSE | 800 | 0.448 | 2.470e-4 | 5.51e-4 | 0.0235 |
| `55991092` | delta | MSE | 800 | 0.448 | 2.459e-4 | 5.49e-4 | **0.0234** |
| `50932793` | delta uptr | MAE | 500 | 0.686 | 4.858e-3 | 7.08e-3 | — (MAE) |
| `50932324` | delta uptr | MAE | 500 | 0.686 | 4.865e-3 | 7.09e-3 | — (MAE) |
| `50932742` | delta uptr | MAE | 500 | 0.686 | 4.870e-3 | 7.10e-3 | — (MAE) |
| `51083128` | delta uptr | MAE | 500 | 0.686 | 5.095e-3 (ep 2) | 7.43e-3 | — (MAE) |

Caveats: the correction assumes the model predicts the padded region as exactly zero (a good
assumption after epoch 0, but not verified per run); delta and raw targets have different
intrinsic magnitudes, so a delta RMS of 0.0234 vs a raw RMS of 0.0310 on the same dataset is
suggestive but not conclusive; and the `N_pad`=2002 run is trained and tested on a different
(unfiltered, hence larger-molecule) population than the `max800` runs.

The cross-check against measured inference metrics is reassuring for `55245318`
(predicted 0.0214 here vs measured RMSE 0.0229 at inference) and off by ~1.4× for `55833314`
(predicted 0.0310 vs measured 0.0217), so treat this column as order-of-magnitude.

---

## 6. Inference runs and the only trustworthy error numbers

Inference has been run **only on Perlmutter, and only on non-FSDP checkpoints**. Seven
inference jobs exist; four crashed or were killed:

| Job | Model | Outcome |
|---|---|---|
| `55802391` | `55245318` | completed → `logs/QMugs_inference` |
| `55804843` | `55245318` | crashed: `manual_ao` density path, psi4 constructor TypeError |
| `55824752` | `55245318` | crashed: `RKSFunctions has no attribute basis_value` |
| `55825234` | `55245318` | killed by wall clock (159 s/conformer at 2 % after 14 min) |
| `55834360` | `55833314` | crashed: `shape '[-1, 2002, 2002]' invalid for input of size 640000` (padding mismatch) |
| `55834804` | `55833314` | completed → `logs/qmugs-inference-55834804-NN1` |
| `55934921` | `55914623` | killed by wall clock |
| `55939913` / `55935478` | `55914623` | completed, but the log dir is tagged `WRONG-DATA-DIR` |

Metrics below are computed on **cropped, unpadded** matrices with the promolecular baseline
added back for delta models (`qmugs.py:1426–1433`), so they are the real quantities of interest.

| Metric | `55245318` (raw, `N_pad`=2002) | `55833314` (raw, `N_pad`=800) | `55914623` (delta) *suspect* |
|---|---|---|---|
| conformers evaluated | 1996 | 1860 | 1860 |
| per-matrix R² | 0.457 ± 0.163 | **0.533 ± 0.125** | 0.683 ± 0.014 |
| relative Frobenius error | 0.730 ± 0.103 | **0.677 ± 0.087** | 0.563 ± 0.013 |
| cosine similarity | 0.676 ± 0.124 | **0.727 ± 0.091** | 0.952 ± 0.006 |
| RMSE | 0.0229 | 0.0217 | 0.0338 |
| trace error | 0.132 | 0.0949 | 0.519 |
| hermitian error | 0.0029 | 0.0544 | 0.0182 |
| min eigenvalue | +0.0014 | −0.0488 | −0.318 |
| # negative eigenvalues | 1.3 | 43.8 | 145.8 |
| electron density abs. fractional error | 0.352 | 0.307 | 0.162 |
| dipole cosine similarity | −0.256 | −0.352 | −0.636 |
| dipole Euclidean error (a.u.) | 177 | 131 | 214 |
| wall time | 2 901 s × 32 ranks | 17 456 s × 32 ranks | 19 710 s × 32 ranks |

Reading these:
- **The `55914623` column should not be used.** Its run config shows
  `--basedir=.../QMugs001_max800.pickle` (the raw-density dataset) with a model trained on
  `QMugs001_max800_delta`, which is exactly what the `WRONG-DATA-DIR` tag records. Its
  apparently excellent cosine similarity (0.952) is also the expected artifact of adding a
  promolecular baseline back: the baseline alone already correlates strongly with the true
  density, so cosine similarity measures the baseline, not the model.
- Physical constraints are violated everywhere: predictions are not symmetric, not PSD (43–146
  negative eigenvalues per matrix), and the trace (electron count) is off by 0.09–0.52.
- Dipole moments are worse than useless — negative cosine similarity means the predicted dipole
  points *away* from the true one on average, with errors of order 100 a.u.
- Downstream cost is the dominant expense: 46–300 GPU-seconds per conformer with
  `electron_density` + `dipole_moment` enabled. A forward pass alone is negligible.

---

## 7. Which combination is best

**By measured, unpadded error: `55833314`** — EGNN, hidden 200, 4 conv layers, mean pooling,
raw total density matrix, `N_pad`=800, `QMugs001_max800`. R² 0.533, relative Frobenius error
0.677, cosine similarity 0.727. It beats `55245318` on every matrix metric while using a 65 M
parameter model with a 100-wide bottleneck, i.e. the *smallest* head tried.

**By training loss, normalized: `55991092`** — PAINN, hidden 400, 6 conv layers, cutoff 7.0 Å,
`max_neighbours` 5, **sum** pooling, delta target, `N_pad`=800. Per-entry RMS 0.0234 vs 0.0310
for raw density on the same dataset, a ~25 % reduction. This has **never been validated by
inference**, and it is the single most valuable inference run to do next.

**The honest conclusion:** no architectural variation tried so far has separated itself from the
others. The evidence is that (a) DCC's three variants agree to 0.25 %, (b) Perlmutter's three
delta variants agree to 1.1 %, (c) all runs reach ~99 % of their final loss within one or two
epochs, and (d) the two validated models capture roughly half the target variance with R² ≈ 0.46–0.53.
The two axes that did move the number are **the target** (delta beats raw) and **the padding
regime** (smaller `N_pad`, i.e. less dilution and a narrower head, beats larger). Neither is an
architecture change. This is consistent with the working diagnosis in
`examples/qmugs/CLAUDE.md`: a global pooling + dense head cannot represent a
basis-function-indexed matrix, and the fix is structural (block-wise / irrep-based prediction),
not a matter of width or depth.

---

## 8. Readiness to run inference on DCC

### Blocker: DCC checkpoints are FSDP2 shards, not full state dicts

`hydragnn/utils/model/model.py:112` consolidates a full state dict only for FSDP **v1**
(`fsdp_version == 1`). Every DCC run used `HYDRAGNN_FSDP_VERSION=2`, which falls through to the
`else` branch and calls `model.state_dict()` on an FSDP2-wrapped model, then writes it from rank 0
only. Under `fully_shard` that returns DTensors holding each rank's shard.

Checkpoint file sizes confirm this arithmetically. With `bytes = 12 × params` for fp32 weights
plus two AdamW moments:

| Run | ranks | hand-counted params | expected full ckpt | expected 1/N shard | actual file |
|---|---|---|---|---|---|
| `50932324` | 4 | 152 M | 1.83 GB | 0.457 GB | **0.462 GB** |
| `50932793` | 4 | 1016 M | 12.2 GB | 3.05 GB | **3.056 GB** |
| `50932742` | 4 | 1440 M | 17.3 GB | 4.32 GB | **4.329 GB** |
| `51082693` | 8 | 402 M | 4.83 GB | 0.603 GB | **0.611 GB** |
| `51083128` | 8 | 1286 M | 15.4 GB | 1.93 GB | **1.936 GB** |
| `55914623` (PM, DDP) | 1 | 641 M | 7.69 GB | n/a | **7.743 GB** |

Every DCC file matches the 1/N shard prediction; the Perlmutter DDP file matches the full
prediction. **No DCC checkpoint currently on disk can be loaded for inference.**

`QMugsInference._load_model` (`examples/qmugs/qmugs.py:1046–1052`) does
`torch.load` → strip `module.` → `load_state_dict(..., strict=False)`. There is no
consolidation step anywhere in the repo (`grep` for `get_model_state_dict`,
`distributed.checkpoint`, `full_tensor`, `DTensor` returns nothing). Because the load is
`strict=False`, a shape or type mismatch that manifests as a missing key would be *warned*, not
raised — so a silently wrong model is a real possibility here, not only a crash.

Definitive one-line check before doing anything else (on a login node, in the DCC env):

```
python -c "import torch; sd=torch.load('<ckpt>.pk', map_location='cpu')['model_state_dict']; k=list(sd)[-1]; print(k, type(sd[k]), sd[k].shape)"
```

If the type is `DTensor` or the shape is 1/N of the expected width, the checkpoint needs
consolidation. Three routes, in increasing order of work: (a) re-run training with
`HYDRAGNN_USE_FSDP=0` — memory permitting, since the largest DCC configs used 23.7 of 24 GB
*with* sharding; (b) re-run with `HYDRAGNN_FSDP_VERSION=1`, which already has a working
`FULL_STATE_DICT` path; (c) add an FSDP2 consolidation branch to `save_model` using
`torch.distributed.checkpoint.state_dict.get_model_state_dict` with
`StateDictOptions(full_state_dict=True, cpu_offload=True)`. Route (c) is the correct fix but
does not recover the six existing checkpoints; only re-training does.

### Everything else on DCC is ready

- **Environment**: `torch`, `psi4`, `mpi4py`, `torch_geometric` 2.6.1 all present in
  `/hpc/group/youlab/xw262/HydraGNN-Installation-Duke-Compute-Cluster/hydragnn_venv`. `adios2`
  is absent by design and the scripts report rather than fail on it.
- **Script**: `scripts/dcc/qmugs_inference_dcc.sh` is complete and mirrors the Perlmutter
  script, launching via `mpirun` with the `mpi_rank_env.sh` shim and `HYDRAGNN_USE_FSDP=0`.
- **Target support**: the inference path handles all four output names, including
  `density_matrix_delta_uptr`, folds the upper triangle back to a square matrix, and adds the
  promolecular baseline back (`qmugs.py:1426–1433`).
- **Raw data for downstream tasks**: `/work/xw262/qmugs/wfns` (100 shard directories) and
  `/work/xw262/qmugs/structures` (665 911 files) are on `/work`.
- **Promolecular baselines**: present for all three DCC datasets (§2).
- **Model code**: rebuilding a DCC model from its saved `config.json` requires the *uncommitted*
  working tree — the `graph_pooling` dict form, `max_nodes`, `num_sharedlayers: 0`, and
  `hydragnn/utils/pooling/` are all local changes. Do not run inference from a clean checkout.

### Two things to fix in `qmugs_inference_dcc.sh` before submitting

1. `MODEL_DIR=$HYDRAGNN_ROOT/logs/qmugs-train-50461062-NN1-DCC-FSDP0-V2` and its
   `CHECKPOINT_PATH` — **this run directory does not exist**. (Its name implies `FSDP0`, so it
   may have been an unsharded run that was since deleted.)
2. `DATASET_BASEDIR=/work/$USER/qmugs/QMugs0001_max800_delta_uptr-DCC.pickle` — **does not
   exist**. The nearest real datasets are `QMugs0001_max500_delta_uptr.pickle` (toy),
   `QMugs01_max500_delta_uptr.pickle`, and `QMugs01_max800_delta_uptr.pickle`.

### Budget the wall clock

The Perlmutter reference cost is 46–300 GPU-seconds per conformer with `electron_density` and
`dipole_moment` enabled. The DCC script requests `-N 4 -t 04:00:00` = 16 A5000 ranks for 4 h,
i.e. roughly 230 000 GPU-seconds. Against the 7 990-conformer `QMugs01_max500` test set that is
short by an order of magnitude at the high end. For a first pass, either set
`PERC_DOWNSAMPLE_TEST` to 0.02–0.05, or set `RUN_DOWNSTREAM_CALCULATION=0` and get the matrix
metrics alone, which are nearly free.

---

## 9. Suggested order of operations

1. Confirm the FSDP2 shard diagnosis with the one-liner in §8, and pick a consolidation route.
2. Do not spend GPU time on inference for the four `QMugs0001_max500` (798-conformer) DCC runs —
   they are mean predictors on a toy split and cannot rank architectures.
3. The single highest-value inference run is `55991092` (PAINN + sum + delta, best normalized
   training loss, never validated). Its checkpoint is a full DDP state dict, so it is loadable —
   but it lives on Perlmutter.
4. On DCC, the untouched `QMugs01_max800_delta_uptr` dataset (148 636 train conformers,
   `max_natoms` 93) is the right training target, with a head narrow enough to keep the epoch
   under the 30 h wall clock — the 10000-wide head costs 3 h 12 min/epoch.
5. Before more width/depth sweeps: the flat result across six architectural variants says the
   bottleneck is representational, not capacity. Block-wise / irrep-structured prediction is the
   change that would move the number.
