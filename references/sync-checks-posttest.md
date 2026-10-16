# Sync checks: post-test report

Results of the checks specified in `references/sync-checks.md`, run on DCC on 2026-10-16. They verify the QMugs 1-RDM pipeline on `derek/learn-density` after the sync with ORNL/HydraGNN main (`24ea300`).

- **Code under test:** commit `8908264`, local and not pushed. After submission, only helpers under `examples/qmugs/sync_checks/` changed:
  - `compare_pickles.py --ignore_keys`
  - `eval_checkpoint.py --split_padding`
  - the new `i1_identity.py`
  - `resubmit_blockX.sh`
  - collector handlers

  No pipeline or `hydragnn/` file changed after submission.
- **Pre-merge reference tree:** `/work/xw262/HydraGNN-premerge`, worktree at `d86486f`. E1b, E2, E3 and U2_premerge import from it.
- **Ledger and raw results:**
  - `logs/sync-checks/STATUS.md`: running analysis, the source of every number below;
  - `logs/sync-checks/jobs.tsv`: check id → SLURM id → exact submission command; a later row for the same check supersedes an earlier one;
  - `logs/sync-checks/results.json` / `results_table.md`: the collector's mechanical verdicts;
  - helper JSONs in `logs/sync-checks/`;
  - job outputs in `examples/qmugs/*-job-<id>.out`;
  - run directories in `logs/<run>/`.
- **Verdicts:**
  - **PASS:** the criterion of `sync-checks.md` holds.
  - **EXPLAINED:** the criterion fails, but the cause is identified with evidence and is not a defect introduced by the sync. Usually the criterion itself was set wrongly a priori; this is stated each time.
  - **FAIL:** a real defect. The cause is stated, and so is whether it predates the sync.
  - **MEASURED:** a measurement with no pass band.

---

## 1. Summary

| check | job | nodes (aa25-) | wall | verdict | one line |
|---|---|---|---|---|---|
| U1 | 58774754 | 20 | 0:24 | PASS | torch 2.8.0+cu128, 4 CUDA devices, imports |
| U2 | 58774767 (+ 58780304, 58780335) | 7 | 1:05 | EXPLAINED | 33/33 PASS; one line differs by 1 float64 ulp; the pre-merge tree gives the same ulp today |
| U3 | 58774762 | 19 | 5:07 | EXPLAINED | 73/74 values exact; the one difference is a round-off residual (2.0e-9 → 1.5e-9) |
| U4 | 58774768 | 7 | 6:27 | PASS | dataset statistics identical to 53684846 |
| U5 | 58774755 | 8 | 0:33 | PASS | 86 passed |
| P1 | 58774746 | 19 | 39:47 | PASS | splits 14867/1853/1860 |
| P1_compare | 58774769 (+ 58776278) | 7 / 21 | 10:10 | FAIL (pre-existing) | `natoms` is an uninitialized tensor; everything else bit-identical |
| P2 / P3 | 58774747 / 58774748 | 19 | 4:24 / 4:17 | PASS | splits 1480/183/189, same draw |
| P4 (four datasets) | 58774770–58774773 | 7, 21 | ≤ 1:33 | PASS (decodercmp, raw, raw_uptr), EXPLAINED (thresholds); **FINDING** (PM delta) | targets exact; dipoles in Debye confirmed; the Perlmutter promolecular baseline differs from DCC's |
| E1 | 58774749 | 20 | 1:50 | PASS | PM EGNN val/test reproduced, rel ≤ 1e-5 |
| E1b | 58774753 | 20 | 1:11 | PASS | bit-identical to E1 (pre-merge tree) |
| E2 | 58774956 | 9–10 | 1:37 | PASS | dense PAINN, pre-merge tree, rel ≤ 1e-6 |
| E3 | 58774750 | 20 | 1:46 | PASS | Schur, pre-merge tree, rel ≤ 1.2e-6 |
| E4a dense / Schur | 58774958 / 58774760 | 9–10 / 20 | 0:48 / 0:35 | PASS (expected failure) | strict load refuses 17 unexpected `*.bias` keys, 0 missing |
| E4b dense / Schur | 58774959 / 58774761 | 9–10 / 20 | 0:54 / 1:33 | MEASURED | dropping the #511 biases: dense +0.86 % val, Schur +13.6 % val |
| T1 | 58774751 | 20 | 8:34 | PASS | EGNN 5 epochs, max rel dev 4.5e-3 (band 5e-3) |
| T2 | 58774955 | 9–10 | 31:32 | PASS | dense PAINN 15 epochs inside ±1 % of the band; loss spike at epoch 12, recovered |
| T3 | 58774752 | 20 | 7:15 | PASS | FSDP2 consolidated full-state checkpoint |
| T4 | 58774758 | 9 | 2:16:33 | PASS | Schur 10 epochs, epoch-9 val −0.16 % from reference |
| T5 dense / Schur | 58774960 / 58774759 | 9–10 / 20 | 5:50 / 28:33 | PASS | resumed loss within 0.5 % of the source run |
| T7 / T8 | 58774763 / 58774764 | 20 | 1:54 / 1:31 | PASS | raw full / upper-triangle targets train |
| I1 | 58774757 (+ 58780054, 58780094) | 20 | 1:21 | PASS (D-1); EXPLAINED (identity) | the corrected identity reproduces E1's loss to 1.2e-7 |
| I2 | 58774961 | 20 | 37:15 | FAIL (pre-existing) | inference and downstream complete; in-process figures deadlock with 4 ranks |
| I3 | 58774756 | 20 | 0:51 | PASS | inference from the FSDP2 checkpoint |
| I4 | 58774766 | 21 | 1:27 | PASS | raw path, no baseline add-back |
| F1 | 58795783 (first 58774962 cancelled) | see §7 | see §7 | see §7 | standalone figures on I2's outputs |

**Bottom line.**
- Every comparison between the merged and the pre-merge code agrees to within float32 summation order or better:
  - evaluation of four saved checkpoints (E1/E1b/E2/E3);
  - preprocessing of the main dataset (P1_compare, apart from the pre-existing `natoms` bug);
  - dataset statistics (U4);
  - the Schur unit suite (U2, apart from 1 ulp that the pre-merge tree shows too).
- Training from scratch lands inside the pre-merge bands (T1, T2, T4). Resume works (T5), FSDP2 saving works (T3), and inference works on all paths (I1, I3, I4, and I2 up to figures).
- The defects found all predate the sync:
  - the `natoms` tensor;
  - the in-process figures deadlock;
  - the machine dependence of the SAD baseline.
- The one intended behaviour change is #511, which removes PaiNN vector biases. It makes old PAINN checkpoints non-loadable under strict loading. Dropping the biases costs 0.8 % (dense) and 13.6 % (Schur) in loss.

---

## 2. Unit and environment checks

### U1: environment — PASS
58774754 (aa25-20, 0:24): torch 2.8.0+cu128, 4 CUDA devices visible, all imports succeed.

### U2: Schur decoder verification suite — EXPLAINED
- 58774767 (aa25-7, 1:05): 33 `[PASS]`, 0 `[FAIL]`. `sync-checks.md` asked for the `[PASS]/[FAIL]` lines to be identical to 53631178; one line differs:

  | | 53631178 (2026-08-28) | 58774767 |
  |---|---|---|
  | `loss_identity` relative difference | 0.000e+00 | 2.178e-16 |

  That is 1 float64 ulp on a value of 4.379542345766e9. The check's own criterion, < 1e-12 (`schur_decoder_test.py:219`), passes.
- **Investigation.**
  - U2_repeat 58780304 (merged tree, aa25-7) gives 2.178e-16 again, so the difference is deterministic.
  - U2_premerge 58780335 runs the pre-merge tree (it imports `/work/xw262/HydraGNN-premerge/hydragnn`; output `logs/sync-checks/U2_premerge-58780335.out`). It **also gives 2.178e-16**, and its other 32 lines are identical.
  - Hence the merge is not the cause. The decoder code is unchanged by the sync, and torch has been installed unchanged since 2026-07-21 on the same node.
  - Two changes happened after 2026-08-28, and these runs cannot separate them:
    - the node OS went from el9_7 to el9_8 (glibc-2.34-274 on 2026-08-31, kernel 5.14.0-687 on 2026-09-01);
    - the code was restored from the Perlmutter backup after the DCC data loss (commit dd8ef91, 2026-09-28).

### U3: ground-truth decomposition diagnostics — EXPLAINED
- 58774762 (aa25-19, 5:07); reference 53618973 (aa25-7, 5:10). `sync-checks.md`'s "about 50 min" was the reference's 49 min of CPU (user) time misread as wall time.
- `schur_decomp_summary.json`:
  - 76/76 keys, none present in only one file;
  - 73 of 74 numeric values bit-identical;
  - no non-numeric differences.
- The only difference is `max_delta_latent_disagreement`: 2.0472157e-9 → 1.4892976e-9 (relative 0.27, against the ≤ 1e-10 criterion).
  - In `per_conformer_summary.csv` only its source column `delta_latent_agreement` differs. That column is defined only for the 10 conformers that get a promolecular baseline.
  - The metric is max |z(D_pro + (D − D_pro)) − z(D)| (`schur_decomp_test.py` ≈1528–1538). In exact arithmetic it is identically zero, so it is a pure round-off residual. Its value depends on the bits of D_pro, which is recomputed by SAD each run (≈694; see §3, P4), and on threaded LAPACK.
  - Both runs show the agreement the metric is meant to show: ~1e-9 on O(1) latent norms.
  - The relative criterion was misapplied to a residual.

### U4: dataset health scan — PASS
58774768 (aa25-7, 6:27). The regenerated dataset and the reference give identical statistics, equal to 53684846:
- 18580 conformers;
- minimum edge 0.954591;
- max|y| p50/p90/p99/max 0.528093 / 0.622816 / 0.692347 / 1.39478;
- n_basis 128 / 526 / 800.

### U5: upstream unit tests — PASS
58774755 (aa25-8, 0:33): 86 passed.

---

## 3. Preprocessing and ground truth

### P1: regenerate `QMugs001_max800_delta_uptr_decodercmp` — PASS
58774746 (aa25-19, 39:47; reference 53631291 36:00). Splits 14867 / 1853 / 1860, as in the reference.

### P1_compare: exact comparison with the existing dataset — FAIL (pre-existing defect); otherwise bit-identical
- 58774769: 18577 sample mismatches, **all on key `natoms`**. There are 0 metadata mismatches, and all 18580 promolecular artifacts are identical.
- **Cause:** `qmugs.py:413`, `natoms = torch.IntTensor(natoms_psi4)`. With an integer argument this allocates an **uninitialized** int32 tensor of that length instead of storing the count, so its contents are whatever was in memory and differ per run; 3 samples matched by chance. `natoms_int` holds the correct count.
- **Impact:**
  - `data.natoms` is read only by the UMA and EScAIP backbones (`hydragnn/models/UMAStack.py`, `hydragnn/utils/model/escaip/...`);
  - it is never read by PAINN, EGNN or the QMugs code;
  - so no model trained so far is affected.

  The defect predates the sync. Not fixed; it needs a decision.
- P1_compare_ignore_natoms 58776278 (aa25-21, 6:47, `--ignore_keys natoms`) gives **0 mismatches** over 14867 / 1853 / 1860 samples, covering every tensor and string. Preprocessing on the merged code reproduces the pre-merge dataset bit for bit, apart from `natoms`.

### P2 / P3: new tiny raw-target datasets — PASS
58774747 / 58774748 (aa25-19, 4:24 / 4:17): splits 1480 / 183 / 189 for both, the same draw.

### P4: targets against Psi4 — PASS / EXPLAINED / FINDING
Jobs: 58774770 (decodercmp), 58774771 (Perlmutter `QMugs001_max800_delta`), 58774772 (raw), 58774773 (raw_uptr). Each sampled conformers from the test split and recomputed everything from the wavefunctions.

- **decodercmp, raw, raw_uptr: PASS.**
  - Stored targets equal the recomputation exactly (`target_max_abs_diff` 0.0); the symmetric-matrix and packed-upper-triangle forms both hold.
  - Atomic numbers are exact; symmetry error 0.
  - Geometry agrees with QMugs to ≤ 4.6e-7 Å.
  - Dipole norm ratio, pipeline (au → Debye) over QMugs `DFT_DIPOLE_*`, is 0.999994–1.000006, with max component difference 2.4e-5 D. The QMugs dipoles are in Debye, and the pipeline dipole implementation is validated.
- **EXPLAINED: two thresholds were set too strictly a priori.**
  - min λ/λ_max is −1.44e-8 to −1.54e-8 on three disjoint samples, against the −1e-10 criterion. Float64 round-off in forming D_α + D_β and in `eigvalsh` is O(1e-14) relative, so the stored QMugs densities themselves are PSD only to ~1e-8 relative.
  - max |tr(SD) − N_e| is 1.06e-6 against the 1e-6 criterion, about 4e-9 relative at N_e ≈ 250. This is the precision of the stored wavefunctions.
- **FINDING (Perlmutter delta dataset, 58774771): the promolecular (SAD) baseline is not machine-independent.**
  - Physics checks are identical to decodercmp, but the stored delta target and artifact differ from today's recomputation by exactly the baseline difference: max |Δ| = 0.3371555 on every sampled conformer.
  - For CHEMBL1388109 conf_00, comparing the Perlmutter and DCC D_pro:
    - differences occur only in on-atom blocks of C (max 0.337), O (0.332) and Cl (0.033);
    - H and N blocks are identical, and so are all off-atom blocks;
    - the traces are equal (201.1535), and tr(S D_pro) = 254 = N_e for both;
    - at the largest entry (C, l = 1, shell 3) PM = +0.160 and DCC = −0.177.
  - Reading: the atomic UHF solutions of the open-shell p-block atoms are symmetry-broken, and they converged to different p-orientations on the two machines. On DCC the baseline is reproducible: decodercmp's artifacts are exact.
  - Consequences:
    - delta targets depend on an arbitrary orientation of the open-shell atomic densities;
    - Perlmutter-made and DCC-made delta datasets and baselines are not interchangeable;
    - any inference that adds back a baseline must use the baseline of the dataset the model was trained on (D-1's fallback does: I1 uses the artifacts of the PM dataset copied to DCC).

  The defect predates the sync. It needs a decision.

---

## 4. Evaluation of saved checkpoints

All of these runs use `sync_checks/eval_checkpoint.py`. It runs hydragnn's `validate`/`test` with the original rank count, batch size and `DistributedSampler` epoch. The references are the logged 5–6 significant digits.

| check | tree | checkpoint | val (ref) | test (ref) | rel. dev. |
|---|---|---|---|---|---|
| E1 | merged | EGNN 55914623 epoch 111 (PM), 4 ranks | 2.43922492e-4 (2.4392e-4) | 2.48618628e-4 (2.4862e-4) | +1.0e-5 / −5.5e-6 |
| E1b | pre-merge | same | identical to E1 | identical to E1 | 0.0 |
| E2 | pre-merge | dense PAINN 53709019 epoch 99, 8 ranks / 2 nodes | 3.91011406e-3 (3.91011e-3) | 3.98406014e-3 (3.98406e-3) | +1.0e-6 / +3.5e-8 |
| E3 | pre-merge | Schur 53692804 epoch 99, 4 ranks | 2.90825730e-3 (2.90826e-3) | 2.97335652e-3 (2.97336e-3) | −9.3e-7 / −1.2e-6 |

- E1 uses the old-config path (no `Variables` block) of the merged code. It also confirms that `DensityMatrixDataset` reproduces the old load path; E1b being bit-identical to E1 confirms the same thing.
- **E4a (merged tree, strict): PASS as the expected failure.**
  - 58774958 (dense): `RuntimeError: Error(s) in loading state_dict for PAINNStack`.
  - 58774760 (Schur): the same for `SchurDecoderModel`.
  - Both report 17 unexpected keys and 0 missing: `update_U.bias` and `update_V.bias` for all 6 layers, and `module_3.bias` for layers 0–4. These are the parameters removed by upstream #511.
- **E4b (merged tree, `--drop_unexpected`): MEASURED.** The loss cost of loading a pre-sync model without its vector biases:

  | | val | test | vs E2 / E3 |
  |---|---|---|---|
  | dense 58774959 | 3.94357e-3 | 4.01559e-3 | +0.86 % / +0.79 % |
  | Schur 58774761 | 3.30430e-3 | 3.37155e-3 | +13.6 % / +13.4 % |

  Pre-sync PAINN checkpoints cannot be resumed exactly on the merged code. A bias-dropped Schur model is noticeably worse; the dense one is nearly unaffected. This is also recorded in the auto-memory `upstream-sync-2026-10-painn-checkpoints`.

---

## 5. Training

### T1: EGNN reproduction of Perlmutter 55914623, 5 epochs — PASS
- 58774751 (`qmugs-train-58774751-NN1-DCC-FSDP0-V2`, aa25-20, 8:34).
- Max |relative deviation| of val/test from 55914623, per epoch: 1.96e-4, 3.9e-5, 4.7e-4, 9.0e-4, 4.51e-3; the maximum, at epoch 4, is val, against a 5e-3 band.
- The growth with epoch is the expected divergence of nondeterministic GPU trajectories.
- Peak memory 14.59 GB (A100 reference 14.6 GB).

### T2: dense PAINN on decodercmp, 15 epochs — PASS
- 58774955 (`qmugs-train-58774955-NN2-DCC-FSDP0-V2`, aa25-9/10, 31:32). 8 ranks; ranks 4–7 log from the second node.
- Against the band of the four pre-merge runs:

  | epoch | val | test | position |
  |---|---|---|---|
  | 0 | 4.07077e-3 | 4.14389e-3 | inside |
  | 1 | 3.98612e-3 | 4.05873e-3 | +0.04 % / +0.03 % above the upper edge |
  | 4 | 3.98318e-3 | 4.05137e-3 | +0.32 % / +0.31 % above |
  | 9 | 3.96151e-3 | 4.03643e-3 | +0.11 % / +0.11 % above |
  | 14 | 3.94802e-3 | 4.01969e-3 | −0.19 % / −0.19 % below the lower edge |

  All are within ±1 %, and epoch 0 is within ±2 %.
- **Gradient norm:**
  - at epoch 0, batch 0: 3.188e-1 against 53709019's 3.139e-1;
  - epochs 2–12: 7e-3–1.1e-2, about 2× the reference's 3–5e-3;
  - epochs 13–14: 1.5–2.7e-3.

  Same order, as required; the factor 2 is recorded. #511 changes the architecture, so it is not expected to match.
- No NaN.
- **Loss spike at epoch 12:** train 4.248e-3 (+5.4 %), val 4.035e-3. It recovered at epoch 13 (val 3.945e-3, the best). This configuration was already unstable before the merge: four of five runs went NaN at epochs 7–19.
- Peak memory 12.71 GB.
- **Checkpoint semantics** (relevant to any later resume or inference):
  - `Checkpoint` saves when val ≤ best, so epochs 0, 1, 2, 3, 5, 7, 9 and 13 are saved, matching the "Creating Checkpoint" log lines;
  - `qmugs_densmat_train.py:714` also saves unconditionally at the end;
  - so `<run>.pk` points to the **last** epoch (here 14), not to the best (13). T5_dense and I2 used the epoch-14 state.

### T3: FSDP2 save path, 2 epochs — PASS
- 58774752 (`qmugs-train-58774752-NN1-DCC-FSDP1-V2`, aa25-20, 7:15).
- Both checkpoints are 6,310,064,631 B, i.e. the DDP size minus 101,902 B: a consolidated full state, not a 1/4 shard.
- Epoch 0/1 val 3.99481e-3 / 3.97665e-3, test 4.06992e-3 / 4.04576e-3. Epoch 0 val is 1.2 % below the pre-merge NN2 band, within ±2 %; this run uses 1 node, i.e. global batch 128 instead of 256.
- Peak memory 8.05 GB.

### T4: Schur decoder from scratch (53692804 configuration), 10 epochs — PASS
- 58774758 (aa25-9, 2:16:33).
- Val per epoch: 4.93156, 4.79921, 4.74546, 4.50994, 4.40313, 4.35166, 4.35957, 4.29593, 4.23430, 4.22241 (×1e-3).
  - Epoch 9 val 4.22241e-3 against 4.2293e-3: −0.16 %, band ±10 %.
  - Epoch 9 test 4.30670e-3 against 4.3142e-3: −0.17 %.
  - The curve decreases from epoch 2 except for a +0.18 % rise at epoch 6. The reference rose at epochs 1–2, so the band anticipated this.
- First `relative_total`: 0.95165 on rank 0, 0.9511–0.9518 across ranks; reference 0.951, band 0.90–1.00.
- Peak memory 10.2005 GB against 10.2104 GB (−0.10 %). About 13.5 min per epoch (reference 13.3).
- No negative-pivot assertion; 0 NaN.

### T5: resume, 2 epochs each — PASS
- **Dense 58774960** (aa25-9/10, 5:50):
  - `Load existing model: ./logs/qmugs-train-58774955-NN2-DCC-FSDP0-V2/...V2.pk`, which resolves to `_epoch_14.pk`; strict load, with the optimizer state.
  - First resumed train loss 4.00956e-3 against T2's last 4.01059e-3: −0.03 %, band ±3 %. The epoch-0 level was 6.9e-3.
- **Schur 58774759** (aa25-20, 28:33):
  - loaded T4's `.pk`, which resolves to `_epoch_9.pk`;
  - first train loss 4.31542e-3 against T4's last 4.29614e-3: +0.45 %;
  - first `relative_total` 0.54798 against T4's last 0.55668: −1.6 %, band ±5 %.

### T7 / T8: dense PAINN on raw targets, 5 epochs — PASS
- **T7** 58774763 (`density_matrix`, dim 640000, aa25-20, 1:54):
  - train MAE 2.635e-2 → 1.134e-2 → 6.329e-3 → 5.111e-3 → 4.688e-3;
  - epoch-4 val MAE **4.56892e-3**.
- **T8** 58774764 (`density_matrix_uptr`, dim 320400, aa25-20, 1:31):
  - train MAE 2.667e-2 → 1.134e-2 → 6.650e-3 → 5.518e-3 → 5.086e-3;
  - epoch-4 val MAE **4.99426e-3**.
- Both are finite and decreasing, and the saved `Variables.outputs[0].dim` is correct.
- T8's MAE is 9.3 % above T7's. That is not a contradiction: the upper-triangle mean counts each off-diagonal pair once, so the larger diagonal errors carry more weight. The two are not expected to be equal.

---

## 6. Inference

### I1: Perlmutter EGNN 55914623 with the D-1 fallback — PASS (D-1); EXPLAINED (identity)
- 58774757 (aa25-20, 1:21).
- **D-1 behaviour:**
  - all 1860 test conformers ran (4 ranks × 465);
  - exactly one fallback warning was logged, `stored promolecular path not found (/pscratch/...)`, and the baseline then resolved to `/work/xw262/artifacts/artifacts_model-QMugs001_max800_delta_data-QMugs001_max800_delta/...`;
  - no missing or unexpected keys;
  - metrics are finite: rmse 0.0182, r2 0.726, relative Frobenius 0.523, cosine 0.854.
- **Consistency identity: EXPLAINED.** The formula in `sync-checks.md` was wrong; the correct one holds exactly.
  - As written, the mean over the test set of SSE_i / N_pad² (N_pad = 800) gives 1.11236e-4 against E1's 2.48619e-4, i.e. −55.3 %.
  - I1_split 58780054 (`eval_checkpoint.py --split_padding`, `I1_split.json`) measured the pieces exactly:
    - in-block SSE / N_pad² is 1.11236e-4, equal to inference to 6 digits;
    - the padded region contributes only 4.97e-6.

    So the discrepancy is not padding leakage.
  - The logged loss is not SSE / N_pad²:
    - the samples carry `y_mask`, and with `mask: false` that mask is all-true on the n_i × n_i block;
    - `Base._compact_or_padding_mask_loss` therefore returns, per batch b, L_b = Σ_{i∈b} SSE_i / Σ_{i∈b} n_i² (`hydragnn/models/Base.py` ≈1328);
    - `test()` takes the num_graphs-weighted mean of L_b on each rank, and then the mean over ranks.

    The padded region never enters the loss.
  - I1_identity 58780094 (`sync_checks/i1_identity.py`, CPU, `I1_identity.json`) rebuilds E1's test batches: `DistributedSampler`, seed 0, `set_epoch(111)`, 4 ranks, batch 32. It applies that formula to the inference SSE_i.
    - Result: **2.48618657e-4 against E1's 2.48618628e-4, relative +1.2e-7**, at the float32 level.
    - The global entry-weighted mean Σ SSE_i / Σ n_i² is 2.47885e-4 (−0.29 %); the difference is the batch grouping.
  - So the inference path (`QMugsInference`, unpadding `raw[:n, :n]`) is consistent with the training loss.

### I2: newly trained dense model, downstream calculations and figures — FAIL (pre-existing deadlock)
- 58774961: T2's final (epoch-14) model on decodercmp, 1 % of the test set, electron density and dipole, aa25-20. The job ran 37:15 and ended with a SIGABRT from the NCCL watchdog.
- **What completed:** inference, metrics and downstream calculations for all 18 conformers.
  - Matrix: rmse 0.01639, r2 0.710, relative Frobenius 0.538, cosine 0.845, hermitian 0.0, trace error 0.030.
  - Electron density: integrated L2 0.0437, absolute fractional error 0.149, mean absolute voxel error 1.98e-3.
  - Dipole: euclidean error 30.5 au, cosine −0.094.
    - On every conformer the predicted density integrates to 16.8–20.7 electrons **fewer** than the true density on the same cube grid (`electron_density_predicted_integral − electron_density_true_integral`).
    - A net charge of ~20 e makes the predicted dipole depend on the origin. The dipole error grows with the deficit: 4.2 au at −16.8 e, 47–59 au at −20.4 to −20.7 e.
    - That accounts for the size of the error. The model, trained for 15 epochs on an unconstrained head, does not conserve the electron count.
  - Per-conformer records are in `logs/qmugs-inference-58774961-NN1-DCC/per_matrix_metrics.json`, and the cube files are under `/work/xw262/artifacts/synccheck-inference-I2/cubeprops`.
  - **Timing:** `inference_time` 266.5 s on 4 ranks, i.e. ≈ 14.8 s wall and ≈ 59 GPU-s per conformer, inside Perlmutter's 46–300 GPU-s.
- **True dipoles: PASS, on a substitute check.**
  - The criterion "equal to P4's pipeline dipoles for shared conformers" turned out to be vacuous: none of the 18 is in any P4 sample.
  - Instead I compared them with QMugs `summary.csv` `DFT_DIPOLE_*`, the comparison P4 makes. Norm ratio 0.999993–1.000007; max component difference 2.7e-5 D.
- **Failure:** sample figures were not produced. Sequence, with `RUN_FIGS=1` and 4 ranks:
  1. Rank 0 alone calls `figs.main`.
  2. `qmugs_densmat_inference.py:758` does not pass its loaded model, so `_prepare_sample_results` (`figs.py` ≈1099) builds a new `QMugsInference`.
  3. That calls `create_model_config` → `Timer("create_model").stop()` → `torch.distributed.reduce` (`hydragnn/utils/profiling_and_tracing/time_utils.py:70`).
  4. Ranks 1–3 are waiting in the MPI `comm.Barrier()`, so the NCCL REDUCE (sequence 4) never completes. The watchdog aborts after 1800 s.
- **Predates the sync:** the same `Timer` is in pre-merge `create.py` (`d86486f:hydragnn/models/create.py:164`), and the sync changed only the dataset wrapper in `figs.py`. In-process figures with more than one rank cannot have worked before either.

  Not fixed; it needs a decision. Candidate fixes:
  - pass the loaded `QMugsInference` to `figs.main` (`inference_runner`);
  - or make figure generation collective;
  - or keep figures to the standalone 1-rank `qmugs_figs_dcc.sh`, as F1 does.

### I3: inference from the FSDP2-consolidated checkpoint — PASS
- 58774756: T3's model, decodercmp, downsample 0.1, aa25-20, 0:51.
- 186 conformers; no missing or unexpected keys; no path fallback needed.
- rmse 0.01723, r2 0.703, relative Frobenius 0.544, cosine 0.844.
- Hermitian 0.0, as it must be when the matrix is folded from an upper triangle. Trace error 0.030.

### I4: inference on the raw-target model — PASS
- 58774766: T7's model, aa25-21 as a single node, 1:27.
- 189 conformers; all metrics finite. `promolecular_density_matrix_path` is `None`, so this is the raw path with no add-back.
- mse 9.21e-4, mae 4.88e-3, r2 0.083, cosine 0.405.
- The hermitian error of 0.41 is expected: the full-matrix head does not symmetrize. The weak scores reflect 5 epochs on 1480 conformers.

---

## 7. Figures

### F1: standalone figures from I2's outputs
- The first F1 (58774962) was cancelled, because it depended `afterok` on I2. It was resubmitted once, without the dependency, as 58795783 on I2's completed outputs.
- It runs as 1 rank, so the reduce of §6/I2 is trivial. It also exercises the sample-figure path that I2 could not reach.

**Status at writing (2026-10-16 ≈12:30): not run, still PENDING (Resources).**
- `qmugs_figs_dcc.sh` requests the `youlab` CPU partition, which has one node. Another user's 7-day job (58718624) holds that node; it has been running for 1 day 5 h.
- This section will be completed when F1 has run. Until then, the sample-figure path is untested on the merged code.

---

## 8. Corrections to `sync-checks.md`

These criteria were wrong as written. The checks above use the corrected forms, and each correction is backed by the measurement cited.

1. **I1 identity.** The logged loss is Σ_b (B_b / N) · Σ_{i∈b} SSE_i / Σ_{i∈b} |mask_i|, averaged over ranks. It is not mean_i SSE_i / N_pad².
2. **P4 PSD threshold.** The stored QMugs densities are PSD only to ~1.5e-8 relative. A threshold such as −1e-7 is appropriate; −1e-10 is not.
3. **P4 electron count.** |tr(SD) − N_e| reaches 1.06e-6. A relative threshold of 1e-8 is appropriate.
4. **U2 exactness.** Exact identity with a run made before the 2026-08-31 OS update is not achievable; the pre-merge tree shows the same 1 ulp. The check's own 1e-12 criterion is the meaningful one.
5. **U3.** "About 50 min" should read about 5 min of wall time (≈ 50 min of CPU). The ≤ 1e-10 relative criterion must exclude `max_delta_latent_disagreement`, which is a round-off residual.
6. **I2 true dipoles.** With a 1 % downsample the I2 and P4 samples do not overlap. Comparing with `summary.csv` directly is the meaningful check.

---

## 9. Findings that need a decision

None of these was introduced by the sync, and none was fixed.

1. **`natoms` is an uninitialized tensor** in every pickle (`qmugs.py:413`). It is harmless to PAINN and EGNN, but would be wrong input for UMA or EScAIP. Fixing it changes the stored datasets (the `natoms` key only).
2. **The SAD promolecular baseline depends on the machine** (§3, P4): the symmetry-broken open-shell atomic solutions differ between Perlmutter and DCC. This affects the definition of every delta target. Options:
   - an orientation-averaged (spherical) atomic density;
   - or a pinned, stored baseline treated as part of the dataset, which is the current de facto behaviour.
3. **In-process figures deadlock with more than one rank** (§6, I2).
4. **`<run>.pk` points to the last epoch, not the best** (§5, T2). Inference with `CHECKPOINT_PATH=<run>.pk` uses the final model.
5. **dcc-youlab-gpu-ferc-s-aa25-21 cannot serve as the remote node of a 2-node `mpirun`** (§10). The DCC scripts' `--exclude` lists only aa25-19.
6. **#511 makes pre-sync PAINN checkpoints non-loadable under strict loading** (§4, E4). This is intended upstream behaviour; it is recorded here because resuming the old Schur runs costs 13.6 %.

---

## 10. Infrastructure log and resubmissions

- **aa25-21 as a remote `mpirun` node.**
  - T2, E2, E4a_dense and E4b_dense (58774741–58774744, nodes aa25-[20-21]) failed in 1–50 s with `mpirun was unable to launch ... on node dcc-youlab-gpu-ferc-s-aa25-21`. T5_dense, I2 and F1 were cancelled through `afterok`.
  - Isolated by test jobs:
    - 58774920: direct execution on 21 works;
    - 58774934: single-node `mpirun` on 21 works;
    - 58774935 / 58774937: 2-node `mpirun` with 21; both ranks report aa25-20, so the remote rank never runs on 21;
    - 58774936: the aa25-7/8 pair places ranks correctly.
  - Block X was resubmitted once by `sync_checks/resubmit_blockX.sh`, excluding aa25-19 and aa25-21 and gated on the end of both lanes. Those are the job ids in this report.
  - Single-node jobs on aa25-21 (I4, P4_raw_uptr, P1_compare_ignore_natoms) are unaffected.
- **F1** was resubmitted once after its dependency cancelled it (§7).
- No numerical result was resubmitted.
- **Extra diagnostic jobs**, not in `sync-checks.md` and each submitted to resolve a failed criterion:
  - P1_compare_ignore_natoms 58776278;
  - I1_split 58780054;
  - I1_identity 58780094;
  - U2_repeat 58780304;
  - U2_premerge 58780335.
- **2-node cap:** held throughout. The lanes and block X were sequenced by SLURM dependencies; the extra diagnostics were 1-node jobs submitted while at most one lane node was busy.

---

## 11. Where the data is

All data is kept, by user decision.

- **Run directories:** `logs/qmugs-train-{58774751,58774752,58774955,58774960,58774763,58774764}-*`, `logs/qmugs-schur-train-{58774758,58774759}-*`, `logs/qmugs-inference-{58774756,58774757,58774766,58774961}-NN1-DCC`, `logs/qmugs-schur-diagnostics-synccheck/`.
- **Datasets regenerated by P1–P3:** under `/work/xw262/qmugs/`, as named in `jobs.tsv`.
- **Inference artifacts:** `/work/xw262/artifacts/synccheck-inference-I2`.
- **Helper outputs:** `logs/sync-checks/*.json`, `U2_premerge-58780335.out`.
