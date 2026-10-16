# Sync checks: DCC jobs that confirm the pipeline after the restore and upstream sync

Compiled 2026-10-16, against `derek/learn-density` @ `1bc53cf` plus the script and helper changes listed in §0.3. This document lists the jobs to run; nothing in it has been run yet. After the runs, `sync-checks-posttest.md` will repeat every check with the actual numbers, the deviation from the reference, and a verdict.

Status context, with the run numbers and file paths cited below, is in `references/sync-updates.md`. Initial data download is out of scope.

---

## 0. Conventions

### 0.1 Where things run

```
cd /work/xw262/HydraGNN/examples/qmugs        # every sbatch below is submitted from here
mkdir -p /work/xw262/HydraGNN/logs/sync-checks  # JSON outputs of the helper scripts
git -C /work/xw262/HydraGNN worktree add --detach /work/xw262/HydraGNN-premerge d86486f
```

- SLURM `*.out` files land in `examples/qmugs/`, like every earlier run.
- Training and inference run directories land in `/work/xw262/HydraGNN/logs/`, because every pipeline script `cd`s to `HYDRAGNN_ROOT` before launching Python.
- `/work/xw262/HydraGNN-premerge` is the pre-merge tree at `d86486f`. It is read-only and never pushed, and is used only as a reference by E1b, E2 and E3. It is removed with `git worktree remove` after the posttest.
- The environment is the existing DCC venv, loaded by each script exactly as in production.
- Every new dataset, artifacts directory and output file carries the suffix `_synccheck` or lives in `logs/sync-checks/`, so nothing existing is overwritten.

### 0.2 Tolerance classes

| Class | Used when | Criterion |
|---|---|---|
| **exact** | same code path, deterministic CPU computation | `torch.equal` / `np.array_equal`, or identical printed numbers |
| **reproduction** | same evaluation of the same weights on the same samples, possibly on different GPUs | relative deviation ≤ 1e-4 |
| **trend** | training from scratch, where GPU atomics are nondeterministic and upstream #511 changed the PaiNN architecture | stated band, with the reason written next to the check |

For the reproduction class, the logged losses come from `validate()` / `test()` under a `DistributedSampler` (seed 0, `set_epoch(epoch)`), which pads each split to a multiple of the world size by repeating samples. `eval_checkpoint.py` therefore runs with **the same number of ranks and the same `set_epoch` as the original run**. The sample multiset is then identical, and only the floating-point summation order differs.

### 0.3 Files added or changed for these checks

| File | Change |
|---|---|
| `scripts/dcc/qmugs_pre_dcc.sh` | `INPUTFILE`, `MAX_DENSITY_MATRIX_SIZE` overridable (defaults unchanged) |
| `scripts/dcc/qmugs_densmat_dcc.sh` | `INPUTFILE`, `MODELNAME`, `PERC_LOAD`, `BATCH_SIZE`, `NUM_EPOCH`, `MAX_DENSITY_MATRIX_SIZE`, `HYDRAGNN_USE_FSDP` overridable (defaults unchanged) |
| `scripts/dcc/qmugs_schur_dcc.sh` | `SCHUR_INPUTFILE` overridable (default unchanged) |
| `scripts/dcc/qmugs_inference_dcc.sh`, `qmugs_figs_dcc.sh` | hard-coded "SET THESE" block skipped when `USE_ENV_ONLY=1`; unset, it applies as before |
| `sync_checks/eval_checkpoint.py` | E1–E4: rebuild model from `config.json`, load checkpoint (strict unless `--drop_unexpected`), run hydragnn's `validate`/`test` |
| `sync_checks/compare_pickles.py` | P1: exact comparison of two pickle datasets and their promolecular artifacts |
| `sync_checks/ground_truth_check.py` | P4: recompute stored targets from Psi4 wavefunctions; physics checks |
| `sync_checks/make_resume_config.py` | T5: copy a config with `continue: 1`, `startfrom: <run>` |
| `sync_checks/sync_python.sh` | sbatch launcher for the helpers, with the pipeline's environment setup |
| `sync_checks/qmugs_egnn_synccheck.json` | T1: `qmugs_densmat.json` with run 55914623's architecture (EGNN, 4 layers, radius 5.0, `max_neighbours` 50, default mean pooling, head 1×400 → [400, 1000], MSE, batch 32, `density_matrix_delta`, `output_dim` 640000) |
| `sync_checks/qmugs_schur_synccheck.json` | T4: `qmugs_schur.json` with `continue: 0`, `startfrom: null` |
| `sync_checks/qmugs_raw_synccheck.json`, `qmugs_raw_uptr_synccheck.json` | P2/P3/T7/T8: `qmugs_densmat.json` with `output_names` `density_matrix` (640000) / `density_matrix_uptr` (320400) and last head layer 400 instead of 1600 to fit 24 GB |

### 0.4 Decision D-1 (approved and applied 2026-10-16)

**D-1. Stored-path fallback in inference.**
- **The problem:** samples of Perlmutter-made pickles (`QMugs001_max800_delta`) store `/pscratch/...` paths for `wfn_path` and `promolecular_density_matrix_path`. `QMugsInference` uses the stored path whenever it is non-empty (`qmugs.py:1202`, `qmugs.py:2381`), so on DCC it will raise `FileNotFoundError` at the first conformer.
- **The fix (applied):** in both resolvers, the stored path is used only if `os.path.isfile(stored)`; otherwise they fall back to `--artifacts_dir` / `--qmugs_data_dir` and log the fallback once.
- **Without the fix,** check I1 is run as an **expected failure** that documents the problem.
- **With it,** I1 becomes a real inference check of the only pretrained model the merged code can load exactly, and the fix itself is validated by I1.

---

## 1. Environment and unit checks

### U1: environment

- **Purpose:** the venv, CUDA, psi4, mpi4py and hydragnn import on a compute node.
- **Run:** interactively; `test.sh` has no SBATCH header.
  ```
  srun -A youlab -p youlab-gpu --gres=gpu:4 -c 12 --mem=32G -t 00:20:00 --pty bash
  cd /work/xw262/HydraGNN && bash examples/qmugs/scripts/dcc/test.sh
  ```
- **Pass:**
  - `torch 2.8.0+cu128` from the venv path
  - `psi4 1.11`
  - `import hydragnn` succeeds
  - `cuda available: True | device count: 4`
  - an mpi4py version line for OpenMPI 4.1.6

### U2: Schur decoder verification suite

- **Purpose:** the decoder's float64 CPU checks reproduce exactly after the merge.
- **Run:**
  ```
  sbatch scripts/dcc/qmugs_schur_decoder_test_dcc.sh
  ```
- **Pass (exact):** 33 `[PASS]` lines, 0 `[FAIL]`, and
  ```
  diff <(grep -aE '^\[(PASS|FAIL)\]' schur-decoder-job-<new>.out) <(grep -aE '^\[(PASS|FAIL)\]' schur-decoder-job-53631178.out)
  ```
  prints nothing.
- **Reference:** job 53631178, e.g. `oracle_round_trip[synthetic]` relative error 4.080e-10, `loss_identity` 4.379542345766e+09, `end_to_end_loss` 1.405450e+00.

### U3: ground-truth decomposition diagnostics

- **Purpose:** `schur_decomp_test.py`, which depends on psi4 and numpy and not on the hydragnn model code, reproduces job 53618973.
- **Run:** the script's defaults equal 53618973's arguments. `LOG_DIR` must be overridden, or the existing results are overwritten.
  ```
  LOG_DIR=/work/xw262/HydraGNN/logs/qmugs-schur-diagnostics-synccheck sbatch scripts/dcc/qmugs_schur_decomp_test_dcc.sh
  ```
- **Compare:** login node, json only.
  ```
  python3 - <<'EOF'
  import json, math
  a = json.load(open('/work/xw262/HydraGNN/logs/qmugs-schur-diagnostics/schur_decomp_summary.json'))
  b = json.load(open('/work/xw262/HydraGNN/logs/qmugs-schur-diagnostics-synccheck/schur_decomp_summary.json'))
  def flat(d, p=''):
      for k, v in d.items():
          if isinstance(v, dict): yield from flat(v, p + k + '.')
          else: yield p + k, v
  fa, fb = dict(flat(a)), dict(flat(b))
  print('keys only in one:', sorted(set(fa) ^ set(fb)))
  worst = max((abs(fa[k] - fb[k]) / max(abs(fa[k]), 1e-300), k) for k in fa if k in fb and isinstance(fa[k], (int, float)) and isinstance(fb[k], (int, float)) and not isinstance(fa[k], bool))
  print('max relative difference:', worst)
  print('non-numeric differences:', [k for k in fa if k in fb and not isinstance(fa[k], (int, float)) and fa[k] != fb[k]])
  EOF
  ```
- **Pass (exact up to threaded LAPACK):** no key differences; maximum relative difference ≤ 1e-10. Exact equality is expected, since the seed and thread count (32) are fixed.
- **Reference:** `logs/qmugs-schur-diagnostics/schur_decomp_summary.json`, 76 keys; e.g. `max_factorization_error` 4.654814938036785e-06.
- About 50 min, CPU only.

### U4: dataset health scan of the regenerated dataset

- **Depends on:** P1.
- **Run:**
  ```
  SCAN_DATASETS="QMugs001_max800_delta_uptr_decodercmp QMugs001_max800_delta_uptr_decodercmp_synccheck" sbatch scripts/dcc/qmugs_scan_dcc.sh
  ```
- **Pass (exact):** both blocks print identical statistics, equal to job 53684846's `decodercmp` block:
  - 14867 / 1853 / 1860 conformers, 18580 scanned
  - minimum stored edge length 0.954591
  - 0 isolated atoms, 0 self loops, 0 non-finite pos / target entries
  - `max |target entry|` p50 = 0.528093, p90 = 0.622816, p99 = 0.692347, max = 1.39478
  - `n_basis` min = 128, p50 = 526, max = 800
- About 1 h, CPU only.

### U5: upstream unit tests

- **Run:** interactively on a GPU node, as during the sync.
  ```
  srun -A youlab -p youlab-gpu --gres=gpu:1 -c 4 --mem=16G -t 00:30:00 --pty bash
  cd /work/xw262/HydraGNN
  PYTHONPATH=/work/xw262/HydraGNN /hpc/group/youlab/xw262/HydraGNN-Installation-Duke-Compute-Cluster/hydragnn_venv/bin/python -m pytest -q -p no:cacheprovider \
      tests/test_named_variable_schema.py tests/test_loss_reporting.py tests/test_config.py tests/test_create_model.py tests/test_example_training_loss_schema.py \
      -m "not mpi and not deepspeed and not network and not fairchem"
  ```
- **Pass:** `86 passed`, as measured during the sync.

---

## 2. Preprocessing checks

### P1: regenerate the main dataset and compare it exactly

- **Purpose:** preprocessing on the merged code reproduces `QMugs001_max800_delta_uptr_decodercmp` byte for byte, in tensors, metadata and promolecular baselines.
  - That pickle was written by the pre-merge code with exactly `qmugs_pre_dcc.sh`'s defaults (job 53631291: 4 ranks, streaming, `perc_load` 0.01, `--max_density_matrix_size=800 --update_max_padded_dimension`, `qmugs_densmat.json`).
  - The molecule draw is `random.sample` under seed 0, so the splits must coincide as well.
- **Run:**
  ```
  MODELNAME=QMugs001_max800_delta_uptr_decodercmp_synccheck sbatch scripts/dcc/qmugs_pre_dcc.sh
  ```
  This writes `/work/xw262/qmugs/QMugs001_max800_delta_uptr_decodercmp_synccheck.pickle` (about 48 GB) and `/work/xw262/artifacts/DCC_MODEL-QMugs001_max800_delta_uptr_decodercmp_synccheck_DATA-QMugs001_max800_delta_uptr_decodercmp_synccheck/`. Reference time: 36 min.
- **Compare:**
  ```
  sbatch -t 08:00:00 --mem=64G sync_checks/sync_python.sh compare_pickles.py \
      --tree_root /work/xw262/HydraGNN \
      --reference /work/xw262/qmugs/QMugs001_max800_delta_uptr_decodercmp.pickle \
      --candidate /work/xw262/qmugs/QMugs001_max800_delta_uptr_decodercmp_synccheck.pickle \
      --path_substitution DCC_MODEL-QMugs001_max800_delta_uptr_decodercmp_DATA-QMugs001_max800_delta_uptr_decodercmp \
                          DCC_MODEL-QMugs001_max800_delta_uptr_decodercmp_synccheck_DATA-QMugs001_max800_delta_uptr_decodercmp_synccheck \
      --reference_artifacts /work/xw262/artifacts/DCC_MODEL-QMugs001_max800_delta_uptr_decodercmp_DATA-QMugs001_max800_delta_uptr_decodercmp \
      --candidate_artifacts /work/xw262/artifacts/DCC_MODEL-QMugs001_max800_delta_uptr_decodercmp_synccheck_DATA-QMugs001_max800_delta_uptr_decodercmp_synccheck \
      --out /work/xw262/HydraGNN/logs/sync-checks/P1_compare_pickles.json
  ```
- **Pass (exact):** exit code 0 and `n_mismatch_total = 0`. That means:
  - split sizes 14867 / 1853 / 1860
  - identical `attrs`: `pna_deg`, `max_natoms` = 93, `n_l` = 3, `n_m` = 5, `n_shell` = 12, species [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
  - every sample's tensors `torch.equal`, and every string equal after the artifacts-directory substitution
  - the same set of promolecular `.npy` files, each `np.array_equal`
- **If it fails:** the JSON names the first 200 mismatches by split, index, key and ChEMBL id.

### P2: raw total density, full matrix (new tiny dataset)

- **Run:**
  ```
  INPUTFILE=$PWD/sync_checks/qmugs_raw_synccheck.json MODELNAME=QMugs0001_max800_synccheck PERC_LOAD=0.001 sbatch scripts/dcc/qmugs_pre_dcc.sh
  ```
- **Pass:**
  - completes
  - target `density_matrix` with padded `y` of shape (800, 800), i.e. head width 640000
  - split sizes logged (to be recorded)
  - no promolecular files written

### P3: raw total density, upper triangle (new tiny dataset)

- **Run:**
  ```
  INPUTFILE=$PWD/sync_checks/qmugs_raw_uptr_synccheck.json MODELNAME=QMugs0001_max800_uptr_synccheck PERC_LOAD=0.001 sbatch scripts/dcc/qmugs_pre_dcc.sh
  ```
- **Pass:**
  - completes; `y` of length T(800) = 320400
  - **split sizes identical to P2**, since it is the same seed, filter and draw
  - the same ChEMBL / conformer ids in the same order. P4 checks the targets of both against Psi4 independently.

### P4: ground truth against Psi4

- **Purpose:** stored targets are what the physics says they are, independent of the code history.
- Four CPU jobs, 30 conformers each, evenly spaced over the test split:
  ```
  for spec in "QMugs001_max800_delta_uptr_decodercmp density_matrix_delta_uptr" \
              "QMugs0001_max800_synccheck density_matrix" \
              "QMugs0001_max800_uptr_synccheck density_matrix_uptr"; do
      set -- $spec
      sbatch -t 03:00:00 sync_checks/sync_python.sh ground_truth_check.py --tree_root /work/xw262/HydraGNN \
          --basedir /work/xw262/qmugs/$1.pickle --output_name $2 --num 30 \
          --out /work/xw262/HydraGNN/logs/sync-checks/P4_$1.json
  done
  sbatch -t 03:00:00 sync_checks/sync_python.sh ground_truth_check.py --tree_root /work/xw262/HydraGNN \
      --basedir /work/xw262/qmugs/QMugs001_max800_delta.pickle --output_name density_matrix_delta --num 30 \
      --path_substitution /pscratch/sd/w/wxrderek/qmugs /work/xw262/qmugs \
      --path_substitution /pscratch/sd/w/wxrderek/artifacts_model-QMugs001_max800_delta_data-QMugs001_max800_delta \
                          /work/xw262/artifacts/artifacts_model-QMugs001_max800_delta_data-QMugs001_max800_delta \
      --out /work/xw262/HydraGNN/logs/sync-checks/P4_QMugs001_max800_delta.json
  ```
- **Pass, exact (exit code 0):**
  - For every conformer, stored `y` equals the float32 cast of the independent float64 recomputation: `Da + Db`, minus `promolecular_density_matrix(wfn)` for delta targets, padded or upper-triangle packed.
  - The promolecular artifact `.npy` equals float32(D_pro).
  - The stored `promolecular_upper_triangle` equals its packing.
  - The SDF atomic numbers equal Psi4's.
- **Pass, physics:**
  - `max|D − Dᵀ|` = 0 to round-off
  - `λ_min/λ_max` ≥ −1e-10, i.e. PSD to round-off
  - `|tr(S D) − N_e|` ≤ 1e-6. `tr(D) − N_e` is recorded and is expected to be O(100), as in `schur_decomp_test.md` §11.
  - SDF positions (Å) equal the Psi4 geometry (bohr × 0.52917721092) to ≤ 1e-4 Å
- **Dipole:** `dipole_norm_ratio_if_debye` = |μ_pipeline| · 2.541746 / |μ_QMugs| is recorded per conformer.
  - **Pass:** the ratio lies in [0.999, 1.001] for all conformers under one unit hypothesis (Debye or atomic units, both reported). This validates the pipeline's dipole implementation against QMugs' own DFT dipoles.
  - Components are compared too; they may differ by a rotation if Psi4 reoriented the molecule. The norm is the primary criterion.
- **The fourth job** also confirms that the Perlmutter-made pickle matches the DCC wavefunctions and the copied artifacts once paths are substituted. That is the precondition for I1 under D-1.

---

## 3. Evaluation of saved checkpoints

All four jobs use `sync_checks/sync_python.sh eval_checkpoint.py`, one rank per GPU.

| ID | Tree | Checkpoint | Data | Ranks | `sampler_epoch` | batch | Reference (logged) |
|---|---|---|---|---|---|---|---|
| E1 | merged | `logs/qmugs-train-55914623-NN1-PM-FSDP0-V2/..._epoch_111.pk` (EGNN, DDP, 7.7 GB) | `QMugs001_max800_delta` | 4 | 111 | 32 | val **2.4392e-4**, test **2.4862e-4** (`run.log`, epoch 111) |
| E1b | pre-merge | same | same | 4 | 111 | 32 | same; must equal E1 to ≤ 1e-6 relative |
| E2 | pre-merge | `logs/qmugs-train-53709019-NN2-DCC-FSDP0-V2/..._epoch_99.pk` (PAINN dense, 6.3 GB) | `decodercmp` | 8 | 99 | 32 | val **3.91011e-3**, test **3.98406e-3** (`train-job-53709019.out`, epoch 99) |
| E3 | pre-merge | `logs/qmugs-schur-train-53692804-.../..._epoch_99.pk` (Schur) | `decodercmp` | 4 | 99 | 128 | val **2.90826e-3**, test **2.97336e-3** (`schur-train-job-53692804.out`, epoch 99) |
| E4a | merged, strict | E2's and E3's checkpoints | as E2 / E3 | 8 / 4 | 99 | 32 / 128 | **expected failure**: `RuntimeError` listing unexpected `*.update_U.bias`, `*.update_V.bias`, `*.module_3.bias` keys (upstream #511) |
| E4b | merged, `--drop_unexpected` | E2's and E3's checkpoints | as E2 / E3 | 8 / 4 | 99 | 32 / 128 | records the loss shift caused by #511 relative to E2 / E3 (no pass band; this is the measurement) |

Commands. Substitute `T`, `M`, `C`, `B`, `R`, `E`, `S` per row; `-N 2` for 8 ranks:

```
sbatch -N 1 --ntasks-per-node=4 --gres=gpu:4 -c 12 --mem=256G -t 02:00:00 \
    sync_checks/sync_python.sh eval_checkpoint.py \
    --tree_root /work/xw262/HydraGNN \
    --model_dir /work/xw262/HydraGNN/logs/qmugs-train-55914623-NN1-PM-FSDP0-V2 \
    --checkpoint qmugs-train-55914623-NN1-PM-FSDP0-V2_epoch_111.pk \
    --basedir /work/xw262/qmugs/QMugs001_max800_delta.pickle \
    --sampler_epoch 111 --batch_size 32 \
    --out /work/xw262/HydraGNN/logs/sync-checks/E1.json
```

- **E1b:** as E1 with `--tree_root /work/xw262/HydraGNN-premerge` and `--out .../E1b.json`.
- **E2:**
  ```
  sbatch -N 2 --ntasks-per-node=4 --gres=gpu:4 -c 12 --mem=256G -t 02:00:00 sync_checks/sync_python.sh eval_checkpoint.py \
      --tree_root /work/xw262/HydraGNN-premerge \
      --model_dir /work/xw262/HydraGNN/logs/qmugs-train-53709019-NN2-DCC-FSDP0-V2 --checkpoint qmugs-train-53709019-NN2-DCC-FSDP0-V2_epoch_99.pk \
      --basedir /work/xw262/qmugs/QMugs001_max800_delta_uptr_decodercmp.pickle --sampler_epoch 99 --batch_size 32 \
      --out /work/xw262/HydraGNN/logs/sync-checks/E2.json
  ```
- **E3:**
  ```
  sbatch -N 1 --ntasks-per-node=4 --gres=gpu:4 -c 12 --mem=256G -t 02:00:00 sync_checks/sync_python.sh eval_checkpoint.py \
      --tree_root /work/xw262/HydraGNN-premerge \
      --model_dir /work/xw262/HydraGNN/logs/qmugs-schur-train-53692804-NN1-DCC-teacher_forcing-joint-total \
      --checkpoint qmugs-schur-train-53692804-NN1-DCC-teacher_forcing-joint-total_epoch_99.pk \
      --basedir /work/xw262/qmugs/QMugs001_max800_delta_uptr_decodercmp.pickle --sampler_epoch 99 --batch_size 128 \
      --out /work/xw262/HydraGNN/logs/sync-checks/E3.json
  ```
- **E4a / E4b:** E2 and E3 with `--tree_root /work/xw262/HydraGNN`, without and with `--drop_unexpected`; outputs `E4a_*` / `E4b_dense.json`, `E4b_schur.json`.

**Pass (reproduction class):** E1, E2 and E3 reproduce val and test to relative deviation ≤ 1e-4.
- The logs print 8 decimals, so the rounding unit is ≤ 2e-5 relative for all three.
- E1 additionally exercises loading a pre-sync config with no `Variables` section through the merged `DensityMatrixDataset` path.
- E2/E3 failing would mean the evaluation method or the data has changed, not the merge, because they run the pre-merge code.

Memory: every rank loads the whole checkpoint, including the AdamW moments, into host memory: 7.7 GB for E1 and 6.3 GB for E2. Hence `--mem=256G` per node.

---

## 4. Training checks (merged code, from scratch unless stated)

### T1: EGNN reproduction of Perlmutter run 55914623, first 5 epochs

- **Run:**
  ```
  INPUTFILE=$PWD/sync_checks/qmugs_egnn_synccheck.json MODELNAME=QMugs001_max800_delta NUM_EPOCH=5 BATCH_SIZE=32 \
      sbatch -N 1 scripts/dcc/qmugs_densmat_dcc.sh
  ```
- **Reference:** `logs/qmugs-train-55914623-NN1-PM-FSDP0-V2/run.log`; 4× A100, DDP, batch 32 per rank, same pickle.

  | epoch | train | val | test |
  |---|---|---|---|
  | 0 | 2.8432e-4 | 2.5459e-4 | 2.5891e-4 |
  | 1 | 2.5906e-4 | 2.5431e-4 | 2.5822e-4 |
  | 2 | 2.5901e-4 | 2.5455e-4 | 2.5853e-4 |
  | 3 | 2.5894e-4 | 2.5428e-4 | 2.5831e-4 |
  | 4 | 2.5895e-4 | 2.5499e-4 | 2.5913e-4 |

- **Pass (trend):** every value within ±0.5 % of the reference.
  - EGNN is not touched by #511, initialization is seeded (`create_model` seeds 0) and the sampler order is identical. The remaining differences are the GPU type and nondeterministic scatter reductions, which perturb the trajectory slightly.
  - Peak memory per GPU is recorded; the reference is 14.6 GB allocated / 15.5 GB reserved on A100, which must fit in 24 GB.

### T2: dense PAINN on `decodercmp` (configuration of 53709019), 15 epochs

- **Run:**
  ```
  NUM_EPOCH=15 HYDRAGNN_GRAD_NORM_STRIDE=50 sbatch scripts/dcc/qmugs_densmat_dcc.sh
  ```
  These are the script defaults: 2 nodes, batch 32, DDP, `qmugs_densmat.json`, `decodercmp`.
- **Reference band:** the four pre-merge runs with the same configuration (53709019, 53692783, 53631819, 53686488):

  | epoch | val band | test band |
  |---|---|---|
  | 0 | 4.0436e-3 – 4.1223e-3 | 4.1170e-3 – 4.1938e-3 |
  | 1 | 3.9733e-3 – 3.9845e-3 | 4.0466e-3 – 4.0574e-3 |
  | 4 | 3.9630e-3 – 3.9706e-3 | 4.0314e-3 – 4.0388e-3 |
  | 9 | 3.9526e-3 – 3.9571e-3 | 4.0276e-3 – 4.0321e-3 (53686488 excluded: diverging before its NaN) |
  | 14 | 3.9557e-3 – 3.9583e-3 | 4.0273e-3 – 4.0299e-3 |

- **Pass (trend):**
  - finite losses
  - epoch 0 within ±2 % of the band edges and epochs 1–14 within ±1 %. The band is widened because #511 changes the PaiNN architecture and its random initialization stream.
  - gradient norms of the same order as 53709019's: 3.1e-1 at epoch 0 batch 0, then O(1e-3)
- **Recorded, not a pass criterion:** whether and when a NaN occurs. Four of the five pre-merge runs of this configuration went NaN at epochs 7–19; 53709019 survived 100 epochs.

### T3: FSDP2 save path, 2 epochs

- **Run:**
  ```
  HYDRAGNN_USE_FSDP=1 NUM_EPOCH=2 sbatch -N 1 scripts/dcc/qmugs_densmat_dcc.sh
  ```
  The run name will end in `-NN1-DCC-FSDP1-V2`.
- **Pass:**
  - The checkpoint file size equals that of the DDP checkpoints of the same model, 6,310,166,533 B (53709019), to within 1 MB. The optimizer state is keyed by parameter name under FSDP2, so a small difference is possible. That shows the consolidated full state dict, not a 1/4 shard of about 1.6 GB.
  - Every state-dict value is a plain `torch.Tensor`; I3 loads it and `eval_checkpoint.py` asserts this.
  - Losses are in T2's epoch-0/1 band (±2 %).

### T4: Schur decoder from scratch (configuration of 53692804), 10 epochs

- **Run:**
  ```
  SCHUR_INPUTFILE=$PWD/sync_checks/qmugs_schur_synccheck.json SCHUR_NUM_EPOCH=10 sbatch scripts/dcc/qmugs_schur_dcc.sh
  ```
  Script defaults: batch 128, 1 node, teacher forcing, joint, target total, `decodercmp`.
- **Reference:** `schur-train-job-53692804.out`

  | epoch | val | test |
  |---|---|---|
  | 0 | 4.8514e-3 | 4.9418e-3 |
  | 3 | 4.5311e-3 | 4.6217e-3 |
  | 6 | 4.3476e-3 | 4.4307e-3 |
  | 9 | 4.2293e-3 | 4.3142e-3 |

  The decoder diagnostic `relative_total` starts at 0.951; peak memory is 10.21 GB allocated per GPU; 13.3 min per epoch.
- **Pass (trend):**
  - no negative-pivot assertion, finite losses
  - epoch 9 val within ±10 % of 4.2293e-3 with a decreasing trend from epoch 2. The wide band is because the PAINN backbone changed (#511) and the early Schur curve is not monotone (epochs 1–2 rose in 53692804).
  - first `relative_total` within 0.90–1.00
  - peak memory within ±10 % of 10.21 GB
- NaN occurrence is recorded.

### T5: resume from a checkpoint (dense and Schur), 2 epochs each

- **Depends on:** T2 and T4. `<T2>` and `<T4>` are their SLURM job ids.
- **Run:**
  ```
  python3 sync_checks/make_resume_config.py qmugs_densmat.json qmugs-train-<T2>-NN2-DCC-FSDP0-V2 sync_checks/qmugs_densmat_resume_synccheck.json
  INPUTFILE=$PWD/sync_checks/qmugs_densmat_resume_synccheck.json NUM_EPOCH=2 sbatch scripts/dcc/qmugs_densmat_dcc.sh

  python3 sync_checks/make_resume_config.py sync_checks/qmugs_schur_synccheck.json qmugs-schur-train-<T4>-NN1-DCC-teacher_forcing-joint-total sync_checks/qmugs_schur_resume_synccheck.json
  SCHUR_INPUTFILE=$PWD/sync_checks/qmugs_schur_resume_synccheck.json SCHUR_NUM_EPOCH=2 sbatch scripts/dcc/qmugs_schur_dcc.sh
  ```
- **Pass:**
  - The log prints `Load existing model: ./logs/<run>/<run>.pk` (`save_model` creates that symlink in new runs) with no exception. The load is strict and includes the optimizer state.
  - The first resumed epoch's train loss is within ±3 % of the source run's last epoch, not back at epoch-0 level (dense ≈ 7.6e-3, Schur ≈ 6.0e-3).
  - For Schur, the first `relative_total` is within ±5 % of T4's last one.

### T7 / T8: dense PAINN on the raw-target datasets, 5 epochs each

- **Depends on:** P2 (T7) and P3 (T8).
- **Run:**
  ```
  INPUTFILE=$PWD/sync_checks/qmugs_raw_synccheck.json      MODELNAME=QMugs0001_max800_synccheck      NUM_EPOCH=5 sbatch -N 1 scripts/dcc/qmugs_densmat_dcc.sh
  INPUTFILE=$PWD/sync_checks/qmugs_raw_uptr_synccheck.json MODELNAME=QMugs0001_max800_uptr_synccheck NUM_EPOCH=5 sbatch -N 1 scripts/dcc/qmugs_densmat_dcc.sh
  ```
- **Pass:**
  - finite losses; train loss decreasing from epoch 0 to 4
  - saved config has `Variables.outputs[0].dim` = 640000 / 320400
- **Recorded:** the epoch-4 val MAE of T7 vs T8. On the same molecules, the upper-triangle MAE excludes the symmetric duplicates, so the two are not expected to be equal. There is no prior DCC reference: Perlmutter's raw-target runs used MSE and `N_pad` 2002 / 800 on a different draw.

---

## 5. Inference and figures

All inference jobs use the real script with `USE_ENV_ONLY=1`, one node, and a fresh artifacts directory. `CHECKPOINT_PATH=<run>.pk` is the symlink `save_model` creates to the last saved epoch file; epoch files are written only for checkpointed epochs, so their numbers are not known in advance. Inference writes cubeprops and electron densities there, so pointing it at a dataset's artifacts directory would add files to it.

### I1: pretrained EGNN 55914623 on its own test split, matrix metrics only

- **Depends on:** D-1.
- **Run:**
  ```
  USE_ENV_ONLY=1 \
  MODEL_DIR=/work/xw262/HydraGNN/logs/qmugs-train-55914623-NN1-PM-FSDP0-V2 CHECKPOINT_PATH=qmugs-train-55914623-NN1-PM-FSDP0-V2_epoch_111.pk \
  DATASET_BASEDIR=/work/xw262/qmugs/QMugs001_max800_delta.pickle \
  ARTIFACTS_DIR=/work/xw262/artifacts/artifacts_model-QMugs001_max800_delta_data-QMugs001_max800_delta \
  PERC_DOWNSAMPLE_TEST=1.0 RUN_DOWNSTREAM_CALCULATION=0 RUN_FIGS=0 \
      sbatch -N 1 -t 02:00:00 scripts/dcc/qmugs_inference_dcc.sh
  ```
  Here `ARTIFACTS_DIR` only supplies the baselines. With the D-1 fallback the inference outputs (metrics) are written under the run's log directory; no downstream calculation runs, so nothing is added to the artifacts directory.
- **Without D-1 — expected failure:** `FileNotFoundError: Promolecular density matrix not found: /pscratch/sd/w/wxrderek/...`.
- **With D-1 — pass:**
  - The job completes over all 1860 test conformers, with no missing or unexpected keys in `QMugsInference._load_model`'s warnings.
  - Metrics are finite, and none is directly comparable to Perlmutter: that run used the wrong data, see `sync-updates.md` §5.
  - **Consistency identity:** for this delta model, the mean over the test split of Σ_ij (D̂ − D)²_ij / N_pad², computed from the unpadded per-matrix squared errors, equals E1's padded test MSE 2.4862e-4 to ±1 %. The small slack is because the model's predictions in the padded region are not exactly zero; that contribution is recorded.

### I2: newly trained dense model with downstream calculations and sample figures

- **Depends on:** T2.
- **Run:**
  ```
  USE_ENV_ONLY=1 \
  MODEL_DIR=/work/xw262/HydraGNN/logs/qmugs-train-<T2>-NN2-DCC-FSDP0-V2 CHECKPOINT_PATH=qmugs-train-<T2>-NN2-DCC-FSDP0-V2.pk \
  DATASET_BASEDIR=/work/xw262/qmugs/QMugs001_max800_delta_uptr_decodercmp.pickle \
  ARTIFACTS_DIR=/work/xw262/artifacts/synccheck-inference-I2 \
  PERC_DOWNSAMPLE_TEST=0.01 RUN_DOWNSTREAM_CALCULATION=1 DOWNSTREAM_CALCULATIONS="electron_density dipole_moment" \
  RUN_FIGS=1 RUN_SAMPLE_FIGS=1 RUN_ELECTRON_DENSITY_FIGS=1 RUN_DIPOLE_FIGS=1 \
      sbatch -N 1 -t 03:00:00 scripts/dcc/qmugs_inference_dcc.sh
  ```
  That is about 19 conformers on 4 ranks.
- **Pass:**
  - completes
  - per-conformer electron-density and dipole records written
  - sample figures written
  - the "true" dipoles in the output equal P4's pipeline dipoles for any conformer the two share
- **Recorded:** wall time per conformer, against Perlmutter's 46–300 GPU-seconds (`recap.md` §6).

### I3: inference from the FSDP2-consolidated checkpoint

- **Depends on:** T3.
- **Run:**
  ```
  USE_ENV_ONLY=1 \
  MODEL_DIR=/work/xw262/HydraGNN/logs/qmugs-train-<T3>-NN1-DCC-FSDP1-V2 CHECKPOINT_PATH=qmugs-train-<T3>-NN1-DCC-FSDP1-V2.pk \
  DATASET_BASEDIR=/work/xw262/qmugs/QMugs001_max800_delta_uptr_decodercmp.pickle \
  ARTIFACTS_DIR=/work/xw262/artifacts/synccheck-inference-I3 \
  PERC_DOWNSAMPLE_TEST=0.1 RUN_DOWNSTREAM_CALCULATION=0 RUN_FIGS=0 \
      sbatch -N 1 -t 01:00:00 scripts/dcc/qmugs_inference_dcc.sh
  ```
- **Pass:**
  - completes on about 186 conformers
  - no missing or unexpected keys at load
  - finite matrix metrics
- This exercises a saved config that has a `Variables` section, upper-triangle folding, and the promolecular add-back from stored DCC paths.

### I4: inference on the raw-target model

- **Depends on:** T7.
- **Run:**
  ```
  USE_ENV_ONLY=1 \
  MODEL_DIR=/work/xw262/HydraGNN/logs/qmugs-train-<T7>-NN1-DCC-FSDP0-V2 CHECKPOINT_PATH=qmugs-train-<T7>-NN1-DCC-FSDP0-V2.pk \
  DATASET_BASEDIR=/work/xw262/qmugs/QMugs0001_max800_synccheck.pickle \
  ARTIFACTS_DIR=/work/xw262/artifacts/synccheck-inference-I4 \
  PERC_DOWNSAMPLE_TEST=1.0 RUN_DOWNSTREAM_CALCULATION=0 RUN_FIGS=0 \
      sbatch -N 1 -t 01:00:00 scripts/dcc/qmugs_inference_dcc.sh
  ```
- **Pass:** completes with finite metrics, on the raw path with no baseline add-back.

### F1: standalone figures from I2's outputs

- **Depends on:** I2.
- **Run:**
  ```
  USE_ENV_ONLY=1 LOG_DIR=/work/xw262/HydraGNN/logs/qmugs-inference-<I2>-NN1-DCC ARTIFACTS_DIR=/work/xw262/artifacts/synccheck-inference-I2 \
  RUN_METRICS_FIGS=1 RUN_SAMPLE_FIGS=1 RUN_ELECTRON_DENSITY_FIGS=1 RUN_DIPOLE_FIGS=1 \
      sbatch scripts/dcc/qmugs_figs_dcc.sh
  ```
- **Pass:** metric-vs-size figures and the sample density-matrix, electron-density and dipole figures are written under `LOG_DIR`, with no exception.

Schur inference is not implemented (`sync-updates.md` §2), so there is no Schur inference check.

---

## 6. Order of submission

1. **Independent; submit together:**
   - CPU: U2, U3, P1, P2, P3, and the decodercmp P4 job
   - GPU: E1, E1b, E2, E3, E4a, E4b, T1, T2, T3, T4
   - interactive: U1, U5
2. **After their dependencies:**
   - U4 and P1-compare (after P1)
   - the P4 jobs on P2/P3 (after P2/P3)
   - the P4 job on `QMugs001_max800_delta` (anytime)
   - T7 (after P2), T8 (after P3)
   - T5 (after T2, T4)
   - I2 (after T2), I3 (after T3), I4 (after T7)
   - I1 (after the D-1 decision)
3. **Last:** F1 (after I2).

## 7. Budget

| Group | Jobs | GPU node-hours | CPU node-hours |
|---|---|---|---|
| Unit | U1–U5 | ≈ 0.3 | ≈ 2.5 (U3 ≈ 0.8, U4 ≈ 1) |
| Preprocessing | P1 (+ compare), P2, P3, P4 ×4 | 0 | ≈ 6 (P1 0.6, compare ≈ 2, P4 ≈ 3) |
| Evaluation | E1, E1b, E2, E3, E4a, E4b | ≈ 3.5 | 0 |
| Training | T1, T2, T3, T4, T5 ×2, T7, T8 | ≈ 7 (T4 ≈ 2.5, T2 ≈ 1.5 on 2 nodes) | 0 |
| Inference / figures | I1–I4, F1 | ≈ 2 | ≈ 0.3 |
| **Total** | | **≈ 13** | **≈ 9** |

## 8. What `sync-checks-posttest.md` must report for each check

- the SLURM job id(s), wall time and node(s)
- the measured numbers in the same units and digits as the reference, with the absolute and relative deviation
- the verdict: PASS, FAIL or EXPLAINED (a deviation with an identified cause), plus the evidence: the file and line, or the JSON key
- anything unexpected in the logs: warnings at model load, NaN, skipped steps, memory outside the stated range
