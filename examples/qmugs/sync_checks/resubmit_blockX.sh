#!/bin/bash
# Resubmission of block X (the two-node checks) after the first attempt failed at launch on
# dcc-youlab-gpu-ferc-s-aa25-21 (2026-10-16, jobs 58774741-58774744: "mpirun was unable to
# launch ... mpi_rank_env.sh ... Node: dcc-youlab-gpu-ferc-s-aa25-21"). Isolated by test jobs
# 58774934/58774935/58774936/58774937: as a remote mpirun node, aa25-21 either fails to start
# its ranks or they silently run on the other node; other node pairs place ranks correctly. Every
# two-node job therefore also excludes aa25-21, the same kind of exclusion the DCC scripts keep
# for aa25-19.
#
# The block waits (afterany) for the last jobs of both lanes, so the 2-node cap still holds.
# T5_dense, I2 and F1 were cancelled with the first T2 and are resubmitted after the new T2.
# Appends to logs/sync-checks/jobs.tsv like submit_all.sh; run once, with plain bash.

set -u
R=/work/xw262/HydraGNN
EX=$R/examples/qmugs
L=$R/logs
Q=/work/xw262/qmugs
A=/work/xw262/artifacts
PRE=/work/xw262/HydraGNN-premerge
SC=$L/sync-checks
JOBS=$SC/jobs.tsv
cd "$EX" || exit 1

GPU2="-N 2 --ntasks-per-node=4 --gres=gpu:4 -c 12 --mem=256G"
EXCL2="--exclude=dcc-youlab-gpu-ferc-s-aa25-19,dcc-youlab-gpu-ferc-s-aa25-21"
jid() { awk -F'\t' -v c="$1" '$1 == c { id = $2 } END { print id }' "$JOBS"; }

sub() {
    local check=$1 deps=$2
    shift 2
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    local id
    id=$(env "${envs[@]}" sbatch --parsable --kill-on-invalid-dep=yes --dependency="$deps" "$@") || { echo "ERROR: sbatch failed for $check" >&2; exit 1; }
    id=${id%%;*}
    printf '%s\t%s\t%s\n' "$check" "$id" "${envs[*]} sbatch --dependency=$deps $*" >> "$JOBS"
    echo "resubmitted $check as $id"
}

# end of lane A (I4) and lane B (T8; I2 was cancelled), plus everything still queued in the lanes
LANES="afterany:$(jid I4),afterany:$(jid T8),afterany:$(jid I1),afterany:$(jid U3)"

sub T2 "$LANES" NUM_EPOCH=15 HYDRAGNN_GRAD_NORM_STRIDE=50 -- \
    $EXCL2 -t 03:00:00 scripts/dcc/qmugs_densmat_dcc.sh

sub E2 "afterany:$(jid T2)" -- $GPU2 $EXCL2 -t 01:30:00 sync_checks/sync_python.sh eval_checkpoint.py \
    --tree_root $PRE \
    --model_dir $L/qmugs-train-53709019-NN2-DCC-FSDP0-V2 --checkpoint qmugs-train-53709019-NN2-DCC-FSDP0-V2_epoch_99.pk \
    --basedir $Q/QMugs001_max800_delta_uptr_decodercmp.pickle --sampler_epoch 99 --batch_size 32 \
    --out $SC/E2.json

sub E4a_dense "afterany:$(jid E2)" -- $GPU2 $EXCL2 -t 01:30:00 sync_checks/sync_python.sh eval_checkpoint.py \
    --tree_root $R \
    --model_dir $L/qmugs-train-53709019-NN2-DCC-FSDP0-V2 --checkpoint qmugs-train-53709019-NN2-DCC-FSDP0-V2_epoch_99.pk \
    --basedir $Q/QMugs001_max800_delta_uptr_decodercmp.pickle --sampler_epoch 99 --batch_size 32 \
    --out $SC/E4a_dense.json

sub E4b_dense "afterany:$(jid E4a_dense)" -- $GPU2 $EXCL2 -t 01:30:00 sync_checks/sync_python.sh eval_checkpoint.py \
    --tree_root $R --drop_unexpected \
    --model_dir $L/qmugs-train-53709019-NN2-DCC-FSDP0-V2 --checkpoint qmugs-train-53709019-NN2-DCC-FSDP0-V2_epoch_99.pk \
    --basedir $Q/QMugs001_max800_delta_uptr_decodercmp.pickle --sampler_epoch 99 --batch_size 32 \
    --out $SC/E4b_dense.json

T2_RUN=qmugs-train-$(jid T2)-NN2-DCC-FSDP0-V2
python3 sync_checks/make_resume_config.py $EX/qmugs_densmat.json "$T2_RUN" $EX/sync_checks/qmugs_densmat_resume_synccheck.json
sub T5_dense "afterany:$(jid E4b_dense),afterok:$(jid T2)" \
    INPUTFILE=$EX/sync_checks/qmugs_densmat_resume_synccheck.json NUM_EPOCH=2 -- \
    $EXCL2 -t 01:00:00 scripts/dcc/qmugs_densmat_dcc.sh

# I2 is a one-node job; it follows the block, so at most 2 nodes are ever in use
sub I2 "afterany:$(jid T5_dense),afterok:$(jid T2)" \
    USE_ENV_ONLY=1 MODEL_DIR=$L/$T2_RUN CHECKPOINT_PATH=$T2_RUN.pk \
    DATASET_BASEDIR=$Q/QMugs001_max800_delta_uptr_decodercmp.pickle ARTIFACTS_DIR=$A/synccheck-inference-I2 \
    PERC_DOWNSAMPLE_TEST=0.01 RUN_DOWNSTREAM_CALCULATION=1 "DOWNSTREAM_CALCULATIONS=electron_density dipole_moment" \
    RUN_FIGS=1 RUN_SAMPLE_FIGS=1 RUN_ELECTRON_DENSITY_FIGS=1 RUN_DIPOLE_FIGS=1 -- \
    -N 1 -t 03:00:00 scripts/dcc/qmugs_inference_dcc.sh

I2_LOG=$L/qmugs-inference-$(jid I2)-NN1-DCC
sub F1 "afterok:$(jid I2)" \
    USE_ENV_ONLY=1 LOG_DIR=$I2_LOG ARTIFACTS_DIR=$A/synccheck-inference-I2 \
    RUN_METRICS_FIGS=1 RUN_SAMPLE_FIGS=1 RUN_ELECTRON_DENSITY_FIGS=1 RUN_DIPOLE_FIGS=1 -- \
    -t 00:20:00 scripts/dcc/qmugs_figs_dcc.sh

# a second collector after everything, including this block
sub COLLECT "afterany:$(jid F1),afterany:$(jid I2),afterany:$(jid T5_dense),afterany:$(jid E4b_dense)" -- \
    -t 00:30:00 sync_checks/sync_python.sh collect_results.py --jobs $JOBS --out_dir $SC
