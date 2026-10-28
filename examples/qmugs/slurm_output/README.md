# `slurm_output/`: SLURM job logs of the QMugs scripts

- **What lives here:** every SLURM `.out` (stdout and stderr together) written by the scripts in `scripts/dcc/` and `schur_long/`. The inference logs below ran in the inference copy `/work/xw262/HydraGNN-schur-inference`, which was merged into this tree and deleted on 2026-10-21. Each job type has its own subdirectory.
- **What does not:** the run outputs of the Python programs (`run.log`, `config.json`, checkpoints, metrics). Those go to `logs/<run>/` at the repository root.
- **Submit from `examples/qmugs/`.** The `#SBATCH -o/-e` paths are relative to the submission directory, and SLURM does not create a missing directory: the job then fails with no log at all. Each subdirectory therefore carries a tracked `.gitkeep`.
- **Git:** `*.out` is git-ignored (`examples/qmugs/.gitignore`). Only this README and the `.gitkeep` files are tracked.
- **History:**
  - On 2026-10-16 the logs were reorganized out of `examples/qmugs/`.
  - On 2026-10-21, every log of a run that is no longer used was moved to `/work/xw262/HydraGNN_archive/`. Its `slurm_output_README-2026-10-16.md` is the complete per-job index from before that move, and `MANIFEST.tsv` gives each log's new path.
  - The archived job types are Perlmutter, pre-sync dense and Schur, sync checks, decomposition and decoder tests, EDA, scan and inspect_ckpt.
  - On 2026-10-28, the logs of the loss trials (segments, evaluations, figures, dense trials), of the overlap dataset build and checks, and of the old baseline evaluations (W-FR-e391, W-DS-e391, D1, D2) were moved to the same archive, topics `loss_trials/`, `overlap_data/` and `schur_long_evaluation/`.

---

## Subdirectories

| subdir | written by | content |
|---|---|---|
| `train/` | `qmugs_densmat_dcc.sh` (`train-job-`) | dense-head training (`qmugs_densmat_train.py`) |
| `schur_train/` | `qmugs_schur_dcc.sh` | Schur-decoder training (`qmugs_densmat_train.py` with `qmugs_schur.json`); empty |
| `schur_long/` | `schur_long/run_segment.sh` (`segment-`) | segments of the 400-epoch Schur run; see `schur_long/README.md` |
| `dataload/` | `qmugs_pre_dcc.sh` (`dataload-job-`), `qmugs_schur_pre_dcc.sh` (`schur-dataload-job-`) | preprocessing (`--preonly`); empty |
| `inference/` | `qmugs_inference_dcc.sh` | `qmugs_densmat_inference.py` on dense-head models |
| `schur_inference/` | `qmugs_schur_inference_dcc.sh` | `qmugs_densmat_inference.py` on Schur-decoder models |
| `figs/` | `qmugs_figs_dcc.sh`, `qmugs_schur_figs_dcc.sh` (`schur-figs-job-`) | standalone figures from a finished inference run (`figs.py`); empty |

---

## `train/`

| log | what ran | outcome | run dir |
|---|---|---|---|
| `train-job-58774955.out` | sync check T2: dense PAINN, 2 nodes aa25-[9-10], decodercmp, 15 epochs, on the merged code | PASS; the dense reference model (epoch 13) | `qmugs-train-58774955-NN2-DCC-FSDP0-V2` |

## `schur_long/`

These are the segments of the kept model's lineage, in order. `logs/schur-long/segments.tsv` is the ledger.

| log | what ran | outcome | run dir |
|---|---|---|---|
| `segment-58799458.out` | epochs 0–77, 1 node | exited 1 after epoch 77 (the crash diagnosed by the replays) | `qmugs-schur-train-58799458-NN1-DCC-teacher_forcing-joint-total` |
| `segment-58802161.out` | resume from 58799458, epoch 78 | exited 1 after epoch 78 | `qmugs-schur-train-58802161-NN1-DCC-teacher_forcing-joint-total` |
| `segment-58857551.out` | replay from `58802161 …_epoch_78.pk` at LR 5e-4, epochs 79–81, imported into the lineage | exit 0 | `qmugs-schur-replay-58857551-lr` |
| `segment-58867528.out` | resume from the replay, epochs 82–399, 2 nodes | finished at epoch 399 (exit 0); best val epoch 391 | `qmugs-schur-train-58867528-NN2-DCC-teacher_forcing-joint-total` |
| `segment-58867530.out` | chained segment after 58867528 | nothing to do (epoch 399 already reached) | none |

The deleted restart 58802162 (exit 134), the smoke runs (58799459, 58799460, 58867318) and the other replays are recorded in `/work/xw262/HydraGNN_archive/schur_long_campaign/README.md` and `MANIFEST.tsv`.

## `inference/` and `schur_inference/`

These are the reference evaluations of the kept models, on the overlap dataset. Labels, exact commands and results are in `logs/baselines/` (`README.md`, `jobs.tsv`).

| log | label | what ran | run dir |
|---|---|---|---|
| `schur_inference/schur-inference-job-59254259.out` | T5-schur | Schur 58867528 epoch 391, free running, full test set, matrix metrics | `logs/qmugs-schur-inference-59254259-NN1-DCC-free_running` |
| `inference/inference-job-59254260.out` | T5-dense | dense T2 epoch 13, full test set, matrix metrics | `logs/qmugs-inference-59254260-NN1-DCC` |
| `schur_inference/schur-inference-job-59411997.out` | ds-schur-baseline-e391 | Schur epoch 391, free running, 10 % of the test set, electron density and dipole | `logs/qmugs-schur-inference-59411997-NN1-DCC-free_running` |
| `inference/inference-job-59399568.out` | ds-dense-baseline-e13 | dense epoch 13, the same 10 %, electron density and dipole | `logs/qmugs-inference-59399568-NN1-DCC` |

The density cube files of these runs were deleted on 2026-10-28; rerun the downstream evaluation to regenerate them.
