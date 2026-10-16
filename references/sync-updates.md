# QMugs 1-RDM learning: project status after the October 2026 restore and upstream sync

Compiled 2026-10-16. **This is a status document only:** it records what exists, what has been run and what is known, and it makes no recommendations. Every number is read from a file that exists now: run logs under `logs/`, SLURM outputs `examples/qmugs/*-job-<id>.out` (since moved to `examples/qmugs/slurm_output/<type>/`), saved `config.json` files, the pickles under `/work/xw262/qmugs`, or git. Each number cites its source.

Two earlier documents are folded in rather than repeated:
- `references/recap.md` (2026-08-27) covers dense-head runs up to job 51083128 and the Perlmutter inference runs.
- `logs/qmugs-schur-diagnostics/schur_decomp_test.md` covers the ground-truth numerical diagnostics for the Schur decoder.

Section numbers below refer to this document unless stated otherwise.

---

## 1. Repository, environment and provenance

### 1.1 Repository

| Item | State |
|---|---|
| Working copy | `/work/xw262/HydraGNN`, git repository again, branch `derek/learn-density`, tracking `origin` = `git@github.com:wxrderek/HydraGNN.git` |
| Head | `1bc53cf` "feat - adapt QMugs density-matrix code to upstream named data schema" |
| Below it | `8979bfd` merge of ORNL/HydraGNN `main` @ `24ea300` (2026-10-09, 50 upstream commits); `d86486f` (DCC partition change); `dd8ef91` (Perlmutter backup after the DCC data loss) |
| Fork point with upstream | `2fb0bd0` (2026-05-20); 22 fork commits since then |
| `upstream` remote | `git@github.com:ORNL/HydraGNN.git`, fetch only (push URL disabled) |
| `derek/learn-density-sync` | identical to `derek/learn-density`, left on GitHub |

**How the directory got here.** The DCC `/work` purge deleted `.git/` and files. The tree was restored from Perlmutter. The transfer dropped every dotfile, every symlink and every executable bit; the repository-tracked ones were restored when the repository was re-attached.

The transfer also dropped the untracked symlinks inside `logs/`. **No run directory has its `logs/<run>/<run>.pk` latest-checkpoint symlink** (0 symlinks under `logs/`), and that symlink is exactly what `Training.continue = 1` loads (`hydragnn/utils/model/model.py`, `load_existing_model`: `os.path.join(path, model_name, fname + ".pk")`). Only the `<run>_epoch_<n>.pk` files survive. Every directory under `logs/` shows the date 2026-10-03, the restore date, so chronology below comes from SLURM job ids.

### 1.2 DCC environment

The venv is `/hpc/group/youlab/xw262/HydraGNN-Installation-Duke-Compute-Cluster/hydragnn_venv`. It lives outside `/work` because VAST rejects psi4 basis-set filenames containing `*`.

| Package | Version | Upstream pin after the sync |
|---|---|---|
| Python | 3.11.15 | `>=3.11,<3.15` |
| torch | 2.8.0+cu128 | 2.13 (`requirements-torch.txt`); 2.14 in the Perlmutter installer |
| numpy | 1.26.4 | 2.4.6 |
| torch_geometric | 2.6.1 | 2.8.0 |
| psi4 | 1.11 (job 57900832 header) | — |

The merged code imports, passes 86 upstream unit tests (schema, loss reporting, config, `create_model`), and trains end to end on this venv. That was measured during the sync on node `dcc-youlab-gpu-ferc-s-aa25-7`, RTX A5000. `setup.py` keeps the fork's commented-out torch/PyG requirements, so `pip install -e .` does not replace torch.

Machines: Perlmutter GPU nodes are 4× A100-40GB. DCC `youlab-gpu` nodes are 4× RTX A5000-24GB with 48 cores. `dcc-youlab-gpu-ferc-s-aa25-19` has a faulty GPU and is excluded in every DCC script.

---

## 2. Pipeline components and their status

| Component | Where | Status | Exercised by |
|---|---|---|---|
| Raw data download | `utils/download_data.*` | done, frozen | — (out of scope) |
| Preprocessing: streaming pickle writer, molecule-level split, size filter, padding to `N_pad`, delta target with promolecular (SAD) baseline, upper-triangle packing, per-orbital metadata | `qmugs.py` (`QMugsDataset`, `_mol_to_graph`), `qmugs_densmat_train.py --preonly --streaming_preonly` | working | DCC jobs 50507706, 51067594, 51067851, 53631291 |
| Load-time sample transform (x = Z, y and y_loc per head, compact masks) | was `SimplePickleDataset.update_data_object` → `update_predicted_values`; now `utils.DensityMatrixDataset` | ported in `1bc53cf`; bit-identical to the old path on 378 tensors per dataset (sync check D2) | sync D2 |
| Dense-head training | `qmugs_densmat_train.py` + `qmugs_densmat.json` | working; every converged dense run so far is a near mean predictor (§4.1) | §4.1 |
| Graph pooling: sum / mean / max, plus concat / cluster-concat / p-concat poolers | `hydragnn/utils/pooling/`, `Base.py` | sum used in every DCC run since 50932324; concat used twice (50518658, 50932742) | §4.1 |
| Schur decoder (sequential bordering recursion, 3 heads) | `hydragnn/decoders/schur/`, `qmugs_schur.json` | trains under teacher forcing, joint supervision; one 100-epoch run (§4.2) | §4.2 |
| NaN guards (`HYDRAGNN_SKIP_NONFINITE_STEP`, `HYDRAGNN_CLIP_GRAD_NORM`, `HYDRAGNN_GRAD_NORM_STRIDE`); Schur non-finite stage trace | `hydragnn/train/train_validate_test.py`, `decoders/schur/model.py` | implemented; the stage trace fired in 53692784, 53704085, 53710269 | §4.2 |
| FSDP2 full-state checkpoint consolidation | `hydragnn/utils/model/model.py` (`fsdp_version == 2` branches) | implemented after `recap.md` §8. The `decodercmp` dense checkpoints (5.9 GB = full) were written by it | §4.1 |
| Inference: matrix metrics, upper-triangle folding, promolecular add-back | `qmugs_densmat_inference.py`, `QMugsInference` | run only on Perlmutter (`recap.md` §6); **never run on DCC** | — |
| Downstream: electron density (cubeprop), dipole moment | `utils/inference_utils.py`, `QMugsInference` | run on Perlmutter only | `recap.md` §6 |
| Figures | `figs.py`, `figs/` | run on Perlmutter only | — |
| Schur inference and figures | — | **not implemented**: the inference path expects a dense head's padded vector (`README.md`, "What to expect") | — |
| Output constraints `trace_norm`, `trace_zero` | `hydragnn/utils/model/model.py:67` | **stub only**: `output_constraint_selection` body is `pass` and nothing calls it, although `README.md` "Specifying output constraints" documents them | — |
| Masked loss (`mask_output_loss.mask = true`) | `qmugs.py`, `Base.loss_hpweighted` | implemented; **never used in any run** (every saved config has `mask: false`); masked pickles exist only on Perlmutter | — |
| ADIOS datasets | — | unsupported by design; `adios2` absent on DCC | — |
| Diagnostics tools | `schur_decomp_test.py`, `schur_decoder_test.py`, `scan_dataset_health.py`, `inspect_checkpoints.py` | all have DCC scripts and at least one completed run (§4.3) | §4.3 |

---

## 3. Datasets

Naming: `QMugs<perc_load>[_max<N>][_delta][_uptr][_suffix]`. `perc_load` 0.001 → `0001`, 0.01 → `001`, 0.1 → `01`. `max<N>` both filters molecules with `n_basis > N` and sets `N_pad = N`. Molecules are drawn with `random.sample` on the `tarball_assignment.csv` order under seed 0, so different `perc_load` draws are independent, not nested.

### 3.1 On DCC (`/work/xw262/qmugs`)

| Pickle | Target | `N_pad` | head width | train / val / test | `max_natoms` | orbital metadata | size | made by |
|---|---|---|---|---|---|---|---|---|
| `QMugs0001_max500_delta_uptr` | `density_matrix_delta_uptr` | 500 | 125,250 | 636 / 78 / 84 | 53 | no | 460 MB | DCC 50507706 (2 min) |
| `QMugs001_max800_delta` | `density_matrix_delta` | 800 | 640,000 | 14867 / 1853 / 1860 | — | no | 51 GB | Perlmutter (copied) |
| `QMugs001_max800_delta_uptr` | `density_matrix_delta_uptr` | 800 | 320,400 | 14867 / 1853 / 1860 | — | no | 26 GB | Perlmutter (copied); never trained |
| `QMugs001_max800_delta_uptr_decodercmp` | `density_matrix_delta_uptr` | 800 | 320,400 | 14867 / 1853 / 1860 | 93 | yes (n_l=3, n_m=5, n_shell=12, species [1,6,7,8,9,15,16,17,35,53]) | 48 GB | DCC 53631291 (36 min) |
| `QMugs01_max500_delta_uptr` | `density_matrix_delta_uptr` | 500 | 125,250 | 63983 / 7998 / 7990 | 60 | no | 45 GB | DCC 51067594 |
| `QMugs01_max800_delta_uptr` | `density_matrix_delta_uptr` | 800 | 320,400 | 148636 / 18574 / 18589 | 93 | no | 251 GB | DCC 51067851; never trained to completion |

- The three `QMugs001_max800*` pickles share the same split sizes, which is the same seed-0 draw.
- **Perlmutter-made pickles store Perlmutter paths.** Samples of `QMugs001_max800_delta` carry `wfn_path = /pscratch/sd/w/wxrderek/qmugs/wfns/...` and `promolecular_density_matrix_path = /pscratch/sd/w/wxrderek/artifacts_model-QMugs001_max800_delta_data-QMugs001_max800_delta/...`, read from the pickle bytes. DCC-made pickles carry `/work/xw262/...`.
- Before this sync, `QMugsInference._resolve_wfn_path` and `_resolve_promolecular_density_matrix_path` used the stored path first and fell back to `--qmugs_data_dir` / `--artifacts_dir` only when it was empty, so delta-target inference or any downstream calculation on a Perlmutter-made pickle looked for `/pscratch/...` on DCC. Since decision D-1 of `sync-checks.md`, both fall back whenever the stored file does not exist, and log it once.

Dataset scan (job 53684846, `scan_dataset_health.py`, radius 7.0): no degenerate geometry, isolated atom, self-loop or non-finite input or target in `QMugs0001_max500_delta_uptr`, `decodercmp` or `QMugs01_max800_delta_uptr`. For `decodercmp`: 18580 conformers, minimum stored edge 0.954591 Å, `max|y|` p50/p90/p99/max = 0.528093/0.622816/0.692347/1.39478, `n_basis` min/p50/max = 128/526/800.

### 3.2 Perlmutter only

`QMugs001`, `QMugs001_new`, `QMugs001_mask`, `QMugs001_max800`, `QMugs001_max800_mask` (`recap.md` §2). The raw-target (`density_matrix`) pickles exist only on Perlmutter.

### 3.3 Promolecular artifacts (`/work/xw262/artifacts`)

- `DCC_MODEL-<name>_DATA-<name>/promolecular_density_matrices/` exists for each of the four DCC-made pickles.
- `artifacts_model-QMugs001_max800_delta_data-QMugs001_max800_delta/` and `..._delta_uptr_.../` hold `promolecular_density_matrices/` and `cubeprops/` copied from Perlmutter.
- `artifacts_model-QMugs001_data-QMugs001_TEST/` and `artifacts_model-QMugs001_max800_data-QMugs001_max800/` hold cubeprops only.

### 3.4 Padding dilution

From `eda/densmat_eda.json` via `recap.md` §2: mean fraction of real entries in the padded square is 0.686 at `N_pad` = 500, 0.448 at 800 and 0.085 at 2002. Reported dense-head losses are means over all padded entries. The Schur loss and the inference metrics are computed on unpadded matrices.

---

## 4. Training record

### 4.1 Dense head

**Perlmutter.** Six runs, all fully in `recap.md` §3:
- EGNN / PAINN, hidden 200–400, mean or sum pooling, MSE, DDP 4–8× A100, batch 32 per rank.
- Raw target: 55245318, 55833314.
- Delta target, on `QMugs001_max800_delta`: 55914623 (EGNN), 55988061 (PAINN, mean), 55991092 (PAINN, sum). Their test MSEs, 2.486e-4 / 2.470e-4 / 2.459e-4, agree to 1.1 %, and each reached about 99 % of its final loss by epoch 1.
- Exact per-epoch record of 55914623, used as a reference in the checks (`logs/qmugs-train-55914623-NN1-PM-FSDP0-V2/run.log`):
  - epoch 0 train/val/test = 2.8432e-4 / 2.5459e-4 / 2.5891e-4
  - epoch 111, which is the saved checkpoint `_epoch_111.pk`: 2.4676e-4 / 2.4392e-4 / 2.4862e-4.

**DCC, up to 51083128.** Six runs on `QMugs0001_max500` and `QMugs01_max500` (`recap.md` §4). The three converged toy runs agree to 0.25 % (MAE 4.858e-3 to 4.870e-3). 51082693 was NaN from epoch 0. Their checkpoints are FSDP2 DTensor shards (`recap.md` §8; file size = 1/N of the full state), loadable by no code.

**DCC, the `decodercmp` series (after `recap.md`).** PAINN, hidden 400, 6 layers, sum pooling, head `[400, 800, 1600] → 320400`, `qmugs_densmat.json`, batch 32 per rank, AdamW lr 1e-3, `NN2` = 8 GPUs unless noted:

| Job | Variant | epoch 0 val / test | last finite epoch: val / test | outcome |
|---|---|---|---|---|
| 53630889 | `QMugs01_max800_delta_uptr`, batch 8, FSDP | — | — | NaN from epoch 0 (2 epochs) |
| 53631819 | FSDP2, MAE | 4.1223e-3 / 4.1938e-3 | ep 15: 3.9588e-3 / 4.0236e-3 | NaN from epoch 16 (ran to 83) |
| 53686488 | FSDP2, MAE | 4.0436e-3 / 4.1170e-3 | ep 9: 4.0312e-3 / 4.1039e-3 | NaN from epoch 10 |
| 53686507 | FSDP2, **MSE** | 2.7648e-4 / 2.8048e-4 | ep 6: 2.7227e-4 / 2.7633e-4 | NaN from epoch 7 |
| 53692783 | **DDP** (`FSDP0`), MAE | 4.0718e-3 / 4.1449e-3 | ep 18: 3.9577e-3 / 4.0253e-3 | NaN from epoch 19 |
| **53709019** | DDP, MAE, `HYDRAGNN_GRAD_NORM_STRIDE` logging on | 4.0873e-3 / 4.1603e-3 | **ep 99: 3.9101e-3 / 3.9841e-3** | completed 100 epochs, 194 min wall (1.9 min/epoch) |

- 53709019's gradient norm falls from 3.14e-1 at epoch 0 batch 0 to 7.07e-4 at epoch 49 and 4.89e-4 at epoch 99 (1600 logged values; maximum 3.14e-1). Its job output does not echo `HYDRAGNN_CLIP_GRAD_NORM` or `HYDRAGNN_SKIP_NONFINITE_STEP`, so whether clipping was active is unknown. No step was logged as skipped.
- Checkpoints `_epoch_81.pk` and `_epoch_99.pk` are full DDP state dicts (6,310,166,533 B).
- Its test MAE fell 4.2 % over 100 epochs (4.1603e-3 → 3.9841e-3), the same flat profile as the earlier runs.

### 4.2 Schur decoder

Configuration of every run: `qmugs_schur.json`, PAINN backbone as in §4.1 (head pinned to width 1 and frozen), `QMugs001_max800_delta_uptr_decodercmp`, `--schur_routing=teacher_forcing --schur_supervision=joint --schur_target=total` (the target in the JSON is `delta`, overridden on the command line), MAE, `w_schedule = per_entry`, `f = softplus`, DDP (`HYDRAGNN_USE_FSDP=0`), 1 node = 4 GPUs, AdamW lr 1e-3.

| Job | batch / rank | start | epochs | outcome |
|---|---|---|---|---|
| 53631821 | — | — | 0 | `AssertionError` at setup |
| 53631991 | 32 | scratch | 0 | DDP refused a module with parameters on both CPU and CUDA |
| 53632070 | 32 | scratch | 26 | ep 25 val/test 2.8979e-3 / 2.9583e-3, then `AssertionError: f(alpha) produced a negative pivot, breaking Theorem 2.1` |
| 53692784 | 32 | scratch | 25 | ep 24 val/test 2.9037e-3 / 2.9618e-3; NaN in epoch 25. Stage trace: positions finite, backbone `v` non-finite, orbital embedding non-finite → NaN weights in the PAINN backbone |
| 53704085 | 32 | **resumed** from 53692784 | 3 | ep 0–2 val 2.880e-3 → 2.861e-3; NaN in epoch 3, same trace |
| 53710269 | 32 | scratch | 11 | ep 10 val/test 3.4445e-3 / 3.5162e-3; NaN in epoch 11, same trace |
| **53692804** | **128** | scratch | **100** | ep 0 val/test 4.8514e-3 / 4.9418e-3 → **ep 99 2.9083e-3 / 2.9734e-3** (train 2.8017e-3); 1333 min wall (13.3 min/epoch, 30 batches); peak 10.21 GB allocated / 24.88 GB reserved per GPU |
| 57900832 | — | intended resume of 53692804 | 0 | after the restore: `mpirun` could not execute `scripts/dcc/mpi_rank_env.sh` (executable bit lost in the transfer, since restored). It would next have failed on the missing `<run>.pk` symlink (§1.1) |

**What 53692804 measured.** Diagnostic lines every 50 `loss()` calls, under teacher forcing, on training batches:

| line in `schur-train-job-53692804.out` | `relative_total` = ‖D̂ − D‖/‖D‖ | `relative_delta` = ‖D̂ − D‖/‖D − D_pro‖ | max\|z\| | max δ |
|---|---|---|---|---|
| 284 (start) | 0.951 | 1.666 | 2.98e-3 | 6.35 |
| 1121 | 0.533 | 0.941 | 0.151 | 3.48 |
| 2010 | 0.441 | 0.776 | 1.056 | 3.40 |
| 2894 | 0.394 | 0.696 | 1.038 | 3.12 |
| 3772 | 0.381 | 0.663 | 0.895 | 3.11 |
| 4654 (end) | 0.357 | 0.622 | 0.998 | 2.81 |

- `relative_total` was still decreasing at epoch 99.
- `relative_delta` > 1 at the start means the reconstruction was further from D than the promolecular baseline is. By the end it is 0.62.
- These are teacher-forced numbers: each step uses the ground-truth leading block D_{k−1}.
- No free-running (autoregressive) evaluation of any trained decoder exists. The ground-truth rollout in `schur_decomp_test.md` §10.5 shows free running diverges to O(1) relative error by k/K ≈ 0.5 even with exact latents.

**Loss comparability.**
- Under `w_schedule = per_entry`, the Schur loss is the plain MAE over the unpadded K×K matrix (`README.md`, "Comparing the two decoders").
- The dense-head loss is the MAE over the padded upper triangle with the compact mask, i.e. over the unpadded upper triangle.
- The README states the two differ by the factor K²/T(K) ≈ 2. That makes 53692804's test 2.9734e-3 and 53709019's test 3.9841e-3 related through a K-dependent factor rather than directly comparable.

### 4.3 Diagnostics and verification runs

| Job | Script | Result |
|---|---|---|
| 50973471 | `schur_decomp_test.py`, `jitter_relative=1e5` (typo) | completed with factorization error 4.65e4 — invalid run |
| 50974223 | same, `jitter_relative=1e-7` | `ValueError: Jittered Cholesky failed at leading minor 129` |
| 50974384 | same, `1e-5`, `rank_tolerance=1e-8` | completed ("July" run of `schur_decomp_test.md`) |
| 53615301 | diagnostics v2 | `AssertionError: CHEMBL362550 truncated pivots went negative` |
| **53618973** | diagnostics v2, `rank_tolerance=1e-6` | completed, 49 min; the results of `schur_decomp_test.md` §10–11 (200 conformers, K 157–1935) |
| 53629940, 53631178 | `schur_decoder_test.py` | 33/33 PASS; identical numbers (53631178 adds the `w_schedule` identity to one message) |
| 53684846 | `scan_dataset_health.py` | clean, §3.1 |
| 53703971 | `inspect_checkpoints.py` | per-module NaN localization of the poisoned checkpoints (README "Diagnosing NaNs") |

Established by the diagnostics (summarized from `schur_decomp_test.md` §11):
- rank(D) = N_e/2 exactly, with range condition number ≈ 5 (max 7.7).
- tr(SD) = N_e, while tr(D) differs from N_e by O(100).
- Pivots span about seven decades, so `f` must be log-scale.
- The median exact latent norm is 0.72 at ρ = 1e-3.
- The hard steps are early (k/K < 0.4) and carry above-average target mass.
- Teacher forcing is stable with an end-to-end error ≈ 40–70× the relative latent error; free running is unstable at zero injected noise.
- Löwdin orthogonalization and the l_k parameterization do not help.

---

## 5. Inference record

All inference so far ran on Perlmutter, on non-FSDP checkpoints (`recap.md` §6). The only trustworthy unpadded metrics are:
- 55245318 (raw, `N_pad` 2002): per-matrix R² 0.457, relative Frobenius error 0.730.
- 55833314 (raw, `N_pad` 800): R² 0.533, relative Frobenius error 0.677.

The 55914623 inference (job 55939913, `logs/qmugs-inference-55935478-NN1-WRONG-DATA-DIR`, since moved to `logs/archive/perlmutter/`) evaluated a delta model on the raw-density pickle and is invalid. Across the valid runs:
- Predicted matrices are not symmetric and not PSD (43–146 negative eigenvalues per matrix).
- Dipole cosine similarity is negative, with dipole errors of 131–214 a.u.

The pipeline's "true" dipole is computed from the true D (`qmugs.py:1688`) and has never been compared against the QMugs reference dipole (`summary.csv` columns `DFT_DIPOLE_X/Y/Z/TOT`).

`scripts/dcc/qmugs_inference_dcc.sh` and `qmugs_figs_dcc.sh` contain a hard-coded "SET THESE" block that overrides their environment-variable defaults with paths that do not exist: `logs/qmugs-train-50461062-NN1-DCC-FSDP0-V2`, `QMugs0001_max800_delta_uptr-DCC.pickle`, `logs/qmugs-inference-50469262-NN1-DCC`. As of this sync that block can be bypassed with `USE_ENV_ONLY=1`; unset, it applies as before.

---

## 6. Which checkpoints the merged code can load

Upstream #511 (`e9ff339`, "Fix biased vector projections in PaiNN and PNAEq") removed the biases of PaiNN's `update_U`, `update_V` and `vec_embed_out`. A bias on a vector channel breaks rotation equivariance: W(Rv) + b ≠ R(Wv + b) unless b = 0. Strict loading of an older PAINN checkpoint therefore fails on the unexpected `*.bias` keys. Dropping them changes the function: on the Schur checkpoint 53692804 the test loss went from 2.6463e-3 to 3.0191e-3 on a 4-conformer batch. With the 17 biases set to zero in the old code, the result was bit-identical to the merged code (sync check D3).

| Checkpoints | Backbone | Loadable on merged code |
|---|---|---|
| PM 55245318 / 55581365 / 55833314 / 55914623 | EGNN | yes (EGNN unaffected by #511; key compatibility to be confirmed by sync check E1) |
| PM 55988061, 55991092 | PAINN | not exactly (#511) |
| DCC 50518658 → 51083128 | PAINN | no: FSDP2 DTensor shards, unloadable by any code |
| DCC 53631819 → 53709019 (`decodercmp` dense) | PAINN | not exactly (#511); full state dicts, so loadable by the pre-merge tree `d86486f` |
| DCC Schur 53632070 → 53710269 | PAINN backbone | not exactly (#511); loadable by `d86486f` |

Resuming any of these with `continue: 1` additionally needs the missing `<run>.pk` symlink (§1.1). `qmugs_schur.json` is still set to `continue: 1, startfrom: qmugs-schur-train-53692804-NN1-DCC-teacher_forcing-joint-total`.

---

## 7. What the upstream sync changed for this project

- **Configuration.** HydraGNN now requires a top-level `Variables` schema and no longer reads `NeuralNetwork.Variables_of_interest`.
  - `qmugs_densmat_train.py` builds `Variables` from `output_names` and `N_pad` (`utils.density_matrix_variables`) and writes it into the saved config.
  - `Variables_of_interest` remains as the example's own settings (`output_names`, `output_dim`, `mask_output_loss`).
  - Configs saved before the sync have `Variables` rebuilt at inference time.
- **Data path.** `utils.DensityMatrixDataset` reproduces the removed load-time transform. Old pickles load unchanged, verified bit-identical (sync check D2). `edge_attr` (interatomic distances) is no longer kept on samples; no current config reads it (PAINN/EGNN with `edge_dim` unset).
- **Models.** PaiNN vector projections are bias-free (#511). EGNN's embedding path was refactored (#526) with no change for configs without input encoders. Base `forward` and `encode` were merged with upstream's input encoders, BF16 output assembly and gated pool conditioning; on shared weights, dense-head and Schur outputs are bit-identical to the pre-merge code (sync check D3).
- **Loss reporting.** Upstream reports per-column loss components. On the fork's masked path this is replaced by one entry per head, because per-column evaluation of a 320,400-wide head is one loss call per column on padded entries.
- **Training loop.** Upstream additions (named task losses, local SGD, performer redraw, FSDP2 checks) merged with the fork's mask arguments, gradient guards and `skip_prediction_plots`.
- **Installers and pins.** The Perlmutter installer moved to `scripts/hpc/nersc/perlmutter/installation/install.sh` with torch 2.14 / CUDA 13; the fork's psi4, matscipy, matplotlib and torch-ema additions were kept. The DCC installer `installation_other/hydragnn_installation_bash_script_dcc.sh` is unchanged (torch 2.8).
- **Scripts.** `qmugs_pre_dcc.sh`, `qmugs_densmat_dcc.sh` and `qmugs_schur_dcc.sh` gained environment overrides with unchanged defaults (`INPUTFILE`, `MAX_DENSITY_MATRIX_SIZE`, `BATCH_SIZE`, `NUM_EPOCH`, `PERC_LOAD`, `MODELNAME`, `HYDRAGNN_USE_FSDP`, `SCHUR_INPUTFILE`). The inference and figures scripts gained `USE_ENV_ONLY`. These were added for `sync-checks.md`.

---

## 8. Open problems and unknowns, as they stand

1. **Mean collapse of the dense head.** Among the dense runs that converged (EGNN/PAINN, mean/sum/concat pooling, 65 M to 1.44 B parameters, raw vs delta), architectural variants on the same dataset agree on test loss within 0.25 % (DCC toy set) and 1.1 % (Perlmutter delta set), and reach about 99 % of their final loss within one or two epochs (`recap.md` §3–5). On `decodercmp`, 53709019's test MAE fell 4.2 % in 100 epochs (§4.1).
2. **NaN in the PAINN backbone.** Of the 13 DCC PAINN training runs since 51082693 that trained at least one epoch, 9 ended in NaN:
   - dense: 6 of 8 (51082693 and 53630889 at epoch 0; the others at epochs 7–19);
   - Schur: 3 of 5 (epochs 3–25). 53632070 ended on the negative-pivot assertion instead.
   
   The two runs that completed are 53709019 (dense) and 53692804 (Schur, batch 128).
   - Excluded as causes (README "Diagnosing NaNs"): the inputs (scan clean), the loss (MSE fails too), FSDP (DDP fails too), the Schur decoder (max|z| plateaus near 1).
   - Localized: the stage trace shows NaN weights in the shared PAINN backbone, written by an optimizer step. PAINN has no normalization, and `node_scalar += a_sv * inner_prod` is quadratic in the vector features over 6 layers.
   - Unknown: the triggering batch and whether #511's bias removal changes the behaviour.
3. **Schur decoder, evaluated only under teacher forcing.** No free-running evaluation, no inference path, and no comparison against the dense head on unpadded metrics.
4. **Inference on DCC has never run.** Perlmutter-made pickles carry `/pscratch` paths; since D-1 the inference path falls back to `--artifacts_dir` / `--qmugs_data_dir` when they are missing (§3.1).
5. **The downstream dipole is unvalidated against QMugs' own DFT dipoles** (§5).
6. **Documentation that no longer matches the code.**
   - `README.md` documents output constraints that are a stub (§2).
   - The root `CLAUDE.md` points to `examples/qmugs/AGENTS.md`, which does not exist.
   - `examples/qmugs/CLAUDE.md` still lists only the dense pipeline.
   - `tasks/schur_decoder.md` is a stub with the title copied from the DCC-transfer task.
7. **`recap.md` §8, now resolved:** FSDP2 consolidation exists; DCC inference scripts can be pointed at real paths.
   **Still open:** none of the 50932324-era shards is recoverable.

---

## 9. Task files (`tasks/`)

| File | Status line |
|---|---|
| `delta_learn_densmat.md` | Done |
| `downstream_density.md` | Done |
| `downstream_dipole_moment.md` | Done |
| `memory_efficient_masks.md` | Done |
| `size_aware_data_loading.md` | Done |
| `upgrade_inference.md` | Done |
| `upper_triangle.md` | Done |
| `visualization.md` | Done (title says "Downstream Density") |
| `perlmutter_to_dcc_transfer.md` | no status line; work documented in `references/dcc_info/perlmutter_to_dcc_transfer.md` |
| `schur_decoder.md` | no status line; stub ("High level plan: 1. read the ideation document, 2. TBD"), title copied from the transfer task; the implementation exists (§2) |
