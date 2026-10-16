# `slurm_output/`: SLURM job logs of the QMugs scripts

- **What lives here:** every SLURM `.out` (stdout and stderr together) written by the scripts in `scripts/dcc/`, `scripts/perlmutter/` and `sync_checks/`. Each job type has its own subdirectory.
- **What does not:** the run outputs of the Python programs (`run.log`, `config.json`, checkpoints, metrics). Those go to `logs/<run>/` at the repository root, as before.
- **Submit from `examples/qmugs/`.** The `#SBATCH -o/-e` paths are relative to the submission directory, and SLURM does not create a missing directory: the job then fails with no log at all. Each subdirectory therefore carries a tracked `.gitkeep`.
- **Git:** `*.out` is git-ignored (`examples/qmugs/.gitignore`). Only this README and the `.gitkeep` files are tracked.
- **History:** the logs were reorganized on 2026-10-16. Before that they sat loose in `examples/qmugs/`. Thirteen logs of jobs that failed at start-up or were superseded by a later successful job were deleted with the user's approval (list at the end).

**Run directories.**
- The `logs/<run>` named next to each log is at `logs/` top level, unless it is marked:
  - **[A: group]** for `logs/archive/<group>/`, a stale run kept for the record;
  - **[S]** for `logs/sync-checks/runs/`.
- "No run dir" means the job wrote nothing under `logs/`, or its directory exists only on Perlmutter.

**Machines.**
- **DCC:** Duke Compute Cluster, `youlab-gpu`; node names `dcc-youlab-gpu-ferc-s-aa25-*`.
- **PM:** Perlmutter (NERSC), node names `nid*`. PM logs were copied over when the repository was restored on 2026-10-03.

**Datasets** are named `QMugs<perc>_max<N>_<target>[_uptr][_decodercmp]`:
- `<perc>`: fraction of QMugs loaded (`0001` = 0.1 %, `001` = 1 %, `01` = 10 %);
- `<N>`: maximum basis size `N_pad`;
- `delta`: target D − D_promolecular;
- `uptr`: packed upper triangle;
- `decodercmp`: the dataset used to compare the dense head with the Schur decoder.

---

## Subdirectories

| subdir | written by | content |
|---|---|---|
| `train/` | `qmugs_densmat_dcc.sh` (`train-job-`), `qmugs_densmat_perlmutter.sh` (`train-debug-job-`) | dense-head training (`qmugs_densmat_train.py`) |
| `schur_train/` | `qmugs_schur_dcc.sh` | Schur-decoder training (`qmugs_densmat_train.py` with `qmugs_schur.json`) |
| `dataload/` | `qmugs_pre_{dcc,perlmutter}.sh` (`dataload-job-`), `qmugs_schur_pre_dcc.sh` (`schur-dataload-job-`) | preprocessing: wavefunctions → pickled datasets and promolecular artifacts (`--preonly`) |
| `inference/` | `qmugs_inference_{dcc,perlmutter}.sh` | `qmugs_densmat_inference.py`: matrix metrics, optional downstream calculations and figures |
| `figs/` | `qmugs_figs_{dcc,perlmutter}.sh` | standalone figures from a finished inference run (`figs.py`) |
| `schur_decomp/` | `qmugs_schur_decomp_test_dcc.sh` (`schur-job-`) | `schur_decomp_test.py`: ground-truth LDLᵀ / Schur diagnostics of QMugs densities (CPU) |
| `schur_decoder_test/` | `qmugs_schur_decoder_test_dcc.sh` | `schur_decoder_test.py`: float64 unit checks of the Schur decoder (CPU) |
| `scan/` | `qmugs_scan_dcc.sh` | `scan_dataset_health.py`: dataset statistics (CPU) |
| `inspect_ckpt/` | `qmugs_inspect_ckpt_dcc.sh` | `inspect_checkpoints.py`: parameter and optimizer statistics of saved checkpoints |
| `eda/` | `qmugs_eda_{dcc,perlmutter}.sh` | `eda.py` (no longer used) |
| `sync_checks/` | the 2026-10-16 sync-check campaign: `sync_checks/sync_python.sh` (`sync-check-job-`), `sync_checks/submit_all.sh` (`sync-U1-`, `sync-U5-`), and the pipeline scripts above as submitted by it | all logs of that campaign, whatever their type; see `references/sync-checks.md` (spec) and `references/sync-checks-posttest.md` (results) |

---

## `train/`

| log | machine | what ran | outcome | run dir |
|---|---|---|---|---|
| `train-job-50518658.out` | DCC | first DCC test: dense PAINN, FSDP, 0.1 % / max500 delta_uptr, batch 8, 5 epochs | completed; epoch 4 val 3.73e-4 | `qmugs-train-50518658-NN1-DCC-FSDP1-V2` [A: dense_dcc] |
| `train-job-50932324.out` | DCC | dense PAINN, FSDP, 0.1 % / max500 delta_uptr, batch 8, 300 epochs requested | early-stopped at epoch 70 (val 5.12e-3) | `qmugs-train-50932324-…` [A: dense_dcc] |
| `train-job-50932742.out` | DCC | same configuration as 50932324 | early-stopped at epoch 39 (val 5.21e-3) | `qmugs-train-50932742-…` [A: dense_dcc] |
| `train-job-50932793.out` | DCC | same configuration as 50932324 | early-stopped at epoch 70 (val 5.12e-3) | `qmugs-train-50932793-…` [A: dense_dcc] |
| `train-job-51082693.out` | DCC | dense PAINN, FSDP, 2 nodes, 10 % / max500 delta_uptr, batch 8 | NaN from epoch 0; cancelled after epoch 9 | `qmugs-train-51082693-NN2-…` [A: dense_dcc] |
| `train-job-51083128.out` | DCC | as 51082693 | cancelled during epoch 3 (last logged epoch 2, val 5.10e-3) | `qmugs-train-51083128-NN2-…` [A: dense_dcc] |
| `train-job-53630889.out` | DCC | dense PAINN, FSDP, 2 nodes, 10 % / max800 delta_uptr, batch 8 | NaN from epoch 0; cancelled after epoch 1 | `qmugs-train-53630889-NN2-…` [A: dense_dcc] |
| `train-job-53631819.out` | DCC | dense PAINN, FSDP, 2 nodes, decodercmp (1 % / max800 delta_uptr), batch 32, 300 epochs | NaN from epoch 16; cancelled after epoch 83. One of the four T2 reference-band runs | `qmugs-train-53631819-NN2-…` [A: dense_dcc] |
| `train-job-53686488.out` | DCC | as 53631819 | NaN from epoch 10; cancelled after epoch 13. T2 band run | `qmugs-train-53686488-NN2-…` [A: dense_dcc] |
| `train-job-53686507.out` | DCC | as 53631819 | NaN from epoch 7; cancelled after epoch 12 | `qmugs-train-53686507-NN2-…` [A: dense_dcc] |
| `train-job-53692783.out` | DCC | dense PAINN, DDP (FSDP0), 2 nodes, decodercmp, batch 32, 100 epochs | NaN from epoch 19; early-stopped at epoch 37. T2 band run | `qmugs-train-53692783-NN2-DCC-FSDP0-V2` [A: dense_dcc] |
| `train-job-53709019.out` | DCC | same configuration as 53692783 | **completed 100 epochs**: epoch 99 val 3.91011e-3, test 3.98406e-3. The only run of this configuration without NaN; reference for E2 and T2 | `qmugs-train-53709019-NN2-DCC-FSDP0-V2` |
| `train-job-55245318.out` | PM | dense model, 2 nodes, `examples/qmugs_run`, 1 % (`QMugs001`), 400 epochs requested | early-stopped at epoch 65 (val 3.99e-5); the model behind `logs/QMugs_inference` | none on DCC (`qmugs-55245318-NN2-PM-FSDP0-V2-TP0` exists only on PM) |
| `train-job-55581365.out` | PM | as 55245318 on `QMugs001_new` | OOM-killed during epoch 10 (last logged epoch 9) | none on DCC |

## `schur_train/`

| log | machine | what ran | outcome | run dir |
|---|---|---|---|---|
| `schur-train-job-53632070.out` | DCC | Schur decoder, teacher forcing, joint, total target, decodercmp, batch 32 | stopped after epoch 25 by the assertion "f(alpha) produced a negative pivot" | `qmugs-schur-train-53632070-…` [A: schur_dcc] |
| `schur-train-job-53692784.out` | DCC | as 53632070, 100 epochs | stopped after epoch 24: "f(alpha) is not finite" (NaN upstream); its epoch-24 checkpoint was studied by `inspect-ckpt-job-53703971` | `qmugs-schur-train-53692784-…` [A: schur_dcc] |
| `schur-train-job-53692804.out` | DCC | as 53692784 with batch 128 | **completed 100 epochs**: epoch 99 val 2.90826e-3, test 2.97336e-3; reference for E3 and T4 | `qmugs-schur-train-53692804-…` |
| `schur-train-job-53704085.out` | DCC | as 53692784, 5 epochs (resume test) | stopped after epoch 2: "f(alpha) is not finite" | `qmugs-schur-train-53704085-…` [A: schur_dcc] |
| `schur-train-job-53710269.out` | DCC | as 53692784, batch 32 | stopped after epoch 10: "f(alpha) is not finite" | `qmugs-schur-train-53710269-…` [A: schur_dcc] |

## `dataload/`

| log | machine | what ran | outcome (dataset written) | run dir |
|---|---|---|---|---|
| `dataload-job-50507706.out` | DCC | preprocessing, 0.1 %, max500, delta_uptr | `QMugs0001_max500_delta_uptr.pickle` | `qmugs-pre-50507706-…` [A: preprocessing] |
| `dataload-job-51067594.out` | DCC | 10 %, max500, delta_uptr | `QMugs01_max500_delta_uptr.pickle` | `qmugs-pre-51067594-…` [A: preprocessing] |
| `dataload-job-51067851.out` | DCC | 10 %, max800, delta_uptr | `QMugs01_max800_delta_uptr.pickle` | `qmugs-pre-51067851-…` [A: preprocessing] |
| `dataload-job-53631291.out` | DCC | 1 %, max800, delta_uptr, decodercmp | `QMugs001_max800_delta_uptr_decodercmp.pickle`, the main DCC dataset (reference for P1) | `qmugs-pre-53631291-NN1-DCC-FSDP0-V2` |
| `dataload-job-55032835.out` | PM | 0.1 % | `QMugs0001_two.pickle` (PM only) | none on DCC |
| `dataload-job-55605055.out` | PM | 10 %, `examples/qmugs_run` | `QMugs01.pickle` (PM only), 4.2 h | none on DCC |

## `inference/`

| log | machine | what ran | outcome | run dir |
|---|---|---|---|---|
| `inference-job-55802391.out` | PM | PM model 55245318 epoch 65 on `QMugs001_new` | completed (50 min) | `QMugs_inference` (the default `LOG_DIR` of the figs scripts) |
| `inference-job-55939913.out` | PM | PM EGNN 55914623 epoch 111 on 8 nodes | completed, but on the raw-density pickle instead of the delta one, so the metrics are invalid (`references/sync-updates.md` §5) | `qmugs-inference-55935478-NN1-WRONG-DATA-DIR` [A: perlmutter] |

## `figs/`
No pre-campaign logs; the one figures job (F1) is in `sync_checks/`.

## `schur_decomp/`

| log | machine | what ran | outcome | run dir |
|---|---|---|---|---|
| `schur-job-50973471.out` | DCC | `schur_decomp_test.py`, jitter_relative 1e5, 10 size bins × 20 | completed; 16 figures | `qmugs-schur-decomp-test` [A: schur_decomp_test] |
| `schur-job-50974384.out` | DCC | as 50973471 with jitter_relative 1e-5; overwrote the same directory | completed; 16 figures | `qmugs-schur-decomp-test` [A: schur_decomp_test] |
| `schur-job-53618973.out` | DCC | final diagnostics run, rank_tolerance 1e-6; the source of `schur_decomp_test.md` | completed (5:10); reference for U3 | `qmugs-schur-diagnostics` |

## `schur_decoder_test/`

| log | machine | what ran | outcome | run dir |
|---|---|---|---|---|
| `schur-decoder-job-53629940.out` | DCC | `schur_decoder_test.py --seed=0`, first full run | 33 PASS | no run dir |
| `schur-decoder-job-53631178.out` | DCC | `schur_decoder_test.py --seed=0`, full suite | 33 PASS; the reference for U2 (read by `sync_checks/collect_results.py`) | no run dir |
| `schur-decoder-job-58798338.out` | DCC | the same suite, 2026-10-16, after the log reorganization; it confirms that the repointed `#SBATCH -o` writes here | 33 PASS | no run dir |

## `scan/`

| log | machine | what ran | outcome | run dir |
|---|---|---|---|---|
| `scan-job-53684846.out` | DCC | `scan_dataset_health.py` on QMugs0001_max500_delta_uptr, decodercmp and QMugs01_max800_delta_uptr (radius 7 Å) | completed; reference for U4 | no run dir |

## `inspect_ckpt/`

| log | machine | what ran | outcome | run dir |
|---|---|---|---|---|
| `inspect-ckpt-job-53703971.out` | DCC | `inspect_checkpoints.py` on the Schur 53692784 epoch-24 checkpoint, `/work/xw262/schur_alive_snapshot.pk` and the dense 53692783 epoch-37 checkpoint | completed; found non-finite Adam `exp_avg` entries, i.e. the NaN had already reached the optimizer state | reads `logs/archive/schur_dcc/qmugs-schur-train-53692784-…` |

## `eda/`

| log | machine | what ran | outcome | run dir |
|---|---|---|---|---|
| `job-54703432.out` | PM | `eda.py --get_densmat_info` over all of QMugs (8.7 h) | completed; maximum density matrix 2002 × 2002 | no run dir |

---

## `sync_checks/` (2026-10-16 campaign)

Check ids refer to `references/sync-checks.md`; measured results and verdicts are in `references/sync-checks-posttest.md`. The check id ↔ job id ↔ exact command mapping is in `logs/sync-checks/jobs.tsv`. "[S]" marks `logs/sync-checks/runs/`.

| log | check | what ran | outcome | run dir |
|---|---|---|---|---|
| `train-job-58774741.out` | T2 (1st) | dense PAINN, 2 nodes aa25-[20-21] | FAILED at launch: `mpirun` could not start ranks on aa25-21 (infrastructure) | none |
| `sync-check-job-58774742.out` | E2 (1st) | 2-node evaluation on aa25-[20-21] | FAILED at launch (aa25-21) | none |
| `sync-check-job-58774743.out` | E4a_dense (1st) | same | FAILED at launch (aa25-21) | none |
| `sync-check-job-58774744.out` | E4b_dense (1st) | same | FAILED at launch (aa25-21) | none |
| `dataload-job-58774746.out` | P1 | regenerate decodercmp on the merged code | completed (39:47) | `qmugs-pre-58774746-…` [S] |
| `dataload-job-58774747.out` | P2 | new raw full-matrix dataset `QMugs0001_max800_synccheck` | completed | `qmugs-pre-58774747-…` [S] |
| `dataload-job-58774748.out` | P3 | new raw upper-triangle dataset `QMugs0001_max800_uptr_synccheck` | completed | `qmugs-pre-58774748-…` [S] |
| `sync-check-job-58774749.out` | E1 | re-evaluate PM EGNN 55914623 epoch 111, merged code | PASS | none (E1.json) |
| `sync-check-job-58774750.out` | E3 | re-evaluate Schur 53692804 epoch 99, pre-merge code | PASS | none (E3.json) |
| `train-job-58774751.out` | T1 | EGNN reproduction of 55914623, 5 epochs | PASS | `qmugs-train-58774751-…` [S] |
| `train-job-58774752.out` | T3 | FSDP2 save path, 2 epochs | PASS | `qmugs-train-58774752-NN1-DCC-FSDP1-V2` [S] |
| `sync-check-job-58774753.out` | E1b | E1 on the pre-merge code | PASS (bit-identical to E1) | none |
| `sync-U1-job-58774754.out` | U1 | environment check | PASS | none |
| `sync-U5-job-58774755.out` | U5 | upstream unit tests | PASS (86 passed) | none |
| `inference-job-58774756.out` | I3 | inference from T3's FSDP2 checkpoint | PASS | `qmugs-inference-58774756-NN1-DCC` [S] |
| `inference-job-58774757.out` | I1 | PM EGNN with the D-1 path fallback | PASS | `qmugs-inference-58774757-NN1-DCC` [S] |
| `schur-train-job-58774758.out` | T4 | Schur decoder from scratch, 10 epochs | PASS | `qmugs-schur-train-58774758-…` (top level) |
| `schur-train-job-58774759.out` | T5_schur | Schur resume, 2 epochs | PASS | `qmugs-schur-train-58774759-…` [S] |
| `sync-check-job-58774760.out` | E4a_schur | strict load of the old Schur checkpoint on the merged code | expected failure (17 unexpected `*.bias` keys) | none |
| `sync-check-job-58774761.out` | E4b_schur | the same with `--drop_unexpected` | measured +13.6 % val | none |
| `schur-job-58774762.out` | U3 | ground-truth decomposition diagnostics | EXPLAINED (73/74 exact) | `qmugs-schur-diagnostics-synccheck` [S] |
| `train-job-58774763.out` | T7 | dense PAINN on the raw full-matrix target, 5 epochs | PASS | `qmugs-train-58774763-…` [S] |
| `train-job-58774764.out` | T8 | dense PAINN on the raw upper-triangle target, 5 epochs | PASS | `qmugs-train-58774764-…` [S] |
| `inference-job-58774766.out` | I4 | inference on T7's model | PASS | `qmugs-inference-58774766-NN1-DCC` [S] |
| `schur-decoder-job-58774767.out` | U2 | Schur decoder unit suite | EXPLAINED (1 ulp) | none |
| `scan-job-58774768.out` | U4 | dataset health scan of the regenerated dataset | PASS | none |
| `sync-check-job-58774769.out` | P1_compare | exact comparison with the existing decodercmp | FAIL on `natoms` only (uninitialized tensor; pre-existing) | none |
| `sync-check-job-58774770.out` | P4_decodercmp | targets against Psi4 | PASS / EXPLAINED (thresholds) | none |
| `sync-check-job-58774771.out` | P4_delta | targets of the PM delta dataset against Psi4 | finding: PM and DCC promolecular baselines differ | none |
| `sync-check-job-58774772.out` | P4_raw | targets of P2's dataset | PASS / EXPLAINED | none |
| `sync-check-job-58774773.out` | P4_raw_uptr | targets of P3's dataset | PASS / EXPLAINED | none |
| `sync-check-job-58774775.out` | COLLECT (1st) | `collect_results.py` after the lanes | completed | none |
| `nodetest-aa25-21-58774920.out` | — (diagnostic) | direct execution on aa25-21 | works | none |
| `mpitest-single21-58774934.out` | — (diagnostic) | single-node `mpirun` on aa25-21 | works | none |
| `mpitest-pair21-58774935.out` | — (diagnostic) | 2-node `mpirun` on aa25-[20-21] | both ranks report aa25-20: the remote rank never runs on 21 | none |
| `mpitest-pairother-58774938.out` | — (diagnostic) | 2-node `mpirun` on aa25-[7-8] | correct placement, one rank per node | none |
| `mpitest-pair21b-58774939.out` | — (diagnostic) | repeat of 58774935 | same failure | none |
| `train-job-58774955.out` | T2 | dense PAINN, 2 nodes aa25-[9-10], 15 epochs | PASS | `qmugs-train-58774955-NN2-DCC-FSDP0-V2` (top level) |
| `sync-check-job-58774956.out` | E2 | re-evaluate 53709019 epoch 99, pre-merge code | PASS | none |
| `sync-check-job-58774958.out` | E4a_dense | strict load of 53709019 on the merged code | expected failure (17 unexpected keys) | none |
| `sync-check-job-58774959.out` | E4b_dense | the same with `--drop_unexpected` | measured +0.86 % val | none |
| `train-job-58774960.out` | T5_dense | dense resume from T2, 2 epochs | PASS | `qmugs-train-58774960-…` [S] |
| `inference-job-58774961.out` | I2 | T2's model with downstream electron density and dipole, then in-process figures | inference complete; FAILED in figures (NCCL deadlock, pre-existing) | `qmugs-inference-58774961-NN1-DCC` [S] |
| `sync-check-job-58774963.out` | COLLECT (2nd) | `collect_results.py` after block X | completed | none |
| `sync-check-job-58776278.out` | P1_compare_ignore_natoms | P1 comparison excluding `natoms` | PASS (0 mismatches) | none |
| `sync-check-job-58780054.out` | I1_split | padded-region split of E1's test loss | completed | none (I1_split.json) |
| `sync-check-job-58780094.out` | I1_identity | rebuild of the logged masked loss from I1's per-matrix errors | completed (rel 1.2e-7) | none (I1_identity.json) |
| `schur-decoder-job-58780304.out` | U2_repeat | U2 again on the merged code | same 1-ulp line | none |
| `U2_premerge-58780335.out` | U2_premerge | U2 on the pre-merge worktree | same 1-ulp line, so not caused by the merge | none |
| `figs-job-58796970.out` | F1 | standalone figures from I2's outputs, on youlab-gpu | PASS | writes into `qmugs-inference-58774961-NN1-DCC/figs` [S] |

---

## Deleted on 2026-10-16 (user-approved)
Each of these failed at start-up or was superseded by a later successful job.

| log | reason |
|---|---|
| `debug-job-54964688.out`, `debug-job-54972795.out`, `debug-job-54973354.out` | PM debug runs of the old `qmugs.py` entry point: a SyntaxError and two task failures; no results |
| `schur-train-job-53631821.out` | AssertionError at start-up |
| `schur-train-job-53631991.out` | DDP device-type error at start-up; its 7.5 KB run dir (config only) was deleted with it |
| `schur-train-job-57900832.out` | "4 total processes failed to start" (0.4 s) |
| `schur-job-50974223.out` | ValueError (invalid jitter); rerun as 50974384 |
| `schur-job-53615301.out` | negative-pivot assertion; rerun as 53618973 |
| `inference-job-55804843.out`, `55824752`, `55825234`, `55834360`, `55934921` | PM inference attempts that failed (TypeError; psi4 `basis_value` AttributeError; reshape 2002 vs 800) or were cancelled; superseded by 55802391 and 55939913 |
