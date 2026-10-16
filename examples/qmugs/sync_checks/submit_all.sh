#!/bin/bash
# Submits every check of references/sync-checks.md with its dependencies, from the login side
# (no SBATCH header: run it with plain bash, NOT sbatch).
#
#   bash examples/qmugs/sync_checks/submit_all.sh                 # submit everything not yet submitted
#   RESUBMIT=T4 bash examples/qmugs/sync_checks/submit_all.sh     # resubmit one check, without dependencies
#
# Idempotent: every submission is appended to logs/sync-checks/jobs.tsv at once
# (check id, SLURM job id, command), and a check already listed there is skipped. If this
# script is interrupted, running it again submits only what is missing.
#
# Concurrency: at most 2 youlab-gpu nodes are held at once by jobs that use GPUs, a whole
# node, or >= 32 cores. This is enforced with dependencies, not by watching the queue:
#   block X (2 nodes):  T2 -> E2 -> E4a_dense -> E4b_dense -> T5_dense
#   lane A  (1 node):   P1 -> P2 -> P3 -> E1 -> E3 -> T1 -> T3 -> E1b -> U1 -> U5 -> I3 -> I1 -> I4
#   lane B  (1 node):   T4 -> T5_schur -> E4a_schur -> E4b_schur -> U3 -> T7 -> T8 -> I2
# Both lanes start after block X. Within a lane, jobs follow each other with afterany, so one
# failure does not block the rest; afterok is added only where a job needs another's output.
# Small CPU jobs (-c 8: U2, U4, P1_compare, P4_*) and F1 (partition youlab) are not capped.

set -u
R=/work/xw262/HydraGNN
EX=$R/examples/qmugs
L=$R/logs
Q=/work/xw262/qmugs
A=/work/xw262/artifacts
PRE=/work/xw262/HydraGNN-premerge
V=/hpc/group/youlab/$USER/HydraGNN-Installation-Duke-Compute-Cluster/hydragnn_venv
SC=$L/sync-checks
JOBS=$SC/jobs.tsv
mkdir -p "$SC"
touch "$JOBS"
cd "$EX" || exit 1

GPU1="-N 1 --ntasks-per-node=4 --gres=gpu:4 -c 12 --mem=256G"
GPU2="-N 2 --ntasks-per-node=4 --gres=gpu:4 -c 12 --mem=256G"
EXCL="--exclude=dcc-youlab-gpu-ferc-s-aa25-19"
RESUBMIT=${RESUBMIT:-}

# latest SLURM id recorded for a check, or empty
jid() { awk -F'\t' -v c="$1" '$1 == c { id = $2 } END { print id }' "$JOBS"; }

# dependency list: dep afterany:T2 afterok:P1 ... -> "afterany:<id>,afterok:<id>"
dep() {
    local out="" spec type check id
    for spec in "$@"; do
        type=${spec%%:*}
        check=${spec#*:}
        id=$(jid "$check")
        if [ -z "$id" ]; then
            # runs inside $(...), so exiting here would not stop the script: flag it instead
            echo "DEPENDENCY_ERROR:$check"
            return
        fi
        out="${out:+$out,}$type:$id"
    done
    echo "$out"
}

# submit CHECK "<dependency list>" VAR=value ... -- sbatch arguments ...
submit() {
    local check=$1 deps=$2
    shift 2
    if [[ "$deps" == DEPENDENCY_ERROR* ]]; then
        echo "ERROR: $check depends on a check that has not been submitted ($deps)" >&2
        exit 1
    fi
    local envs=()
    while [ "$1" != "--" ]; do
        envs+=("$1")
        shift
    done
    shift
    if [ -n "$(jid "$check")" ] && [ "$RESUBMIT" != "$check" ]; then
        echo "skip $check (already submitted as $(jid "$check"))"
        return
    fi
    if [ -n "$RESUBMIT" ] && [ "$RESUBMIT" != "$check" ]; then
        return
    fi
    # a job whose afterok dependency failed is cancelled instead of pending forever, so the
    # afterany chain of its lane continues
    local depflag=(--kill-on-invalid-dep=yes)
    # a resubmission runs without dependencies: its predecessors may be gone from SLURM
    if [ -n "$deps" ] && [ -z "$RESUBMIT" ]; then
        depflag+=(--dependency="$deps")
    fi
    local id
    id=$(env "${envs[@]}" sbatch --parsable "${depflag[@]}" "$@") || { echo "ERROR: sbatch failed for $check" >&2; exit 1; }
    id=${id%%;*}
    printf '%s\t%s\t%s\n' "$check" "$id" "${envs[*]} sbatch ${depflag[*]} $*" >> "$JOBS"
    echo "submitted $check as $id"
}

# ---------------------------------------------------------------------------------------------
# block X: two-node jobs, first

submit T2 "" NUM_EPOCH=15 HYDRAGNN_GRAD_NORM_STRIDE=50 -- \
    -t 03:00:00 scripts/dcc/qmugs_densmat_dcc.sh

submit E2 "$(dep afterany:T2)" -- $GPU2 $EXCL -t 01:30:00 sync_checks/sync_python.sh eval_checkpoint.py \
    --tree_root $PRE \
    --model_dir $L/qmugs-train-53709019-NN2-DCC-FSDP0-V2 --checkpoint qmugs-train-53709019-NN2-DCC-FSDP0-V2_epoch_99.pk \
    --basedir $Q/QMugs001_max800_delta_uptr_decodercmp.pickle --sampler_epoch 99 --batch_size 32 \
    --out $SC/E2.json

submit E4a_dense "$(dep afterany:E2)" -- $GPU2 $EXCL -t 01:30:00 sync_checks/sync_python.sh eval_checkpoint.py \
    --tree_root $R \
    --model_dir $L/qmugs-train-53709019-NN2-DCC-FSDP0-V2 --checkpoint qmugs-train-53709019-NN2-DCC-FSDP0-V2_epoch_99.pk \
    --basedir $Q/QMugs001_max800_delta_uptr_decodercmp.pickle --sampler_epoch 99 --batch_size 32 \
    --out $SC/E4a_dense.json

submit E4b_dense "$(dep afterany:E4a_dense)" -- $GPU2 $EXCL -t 01:30:00 sync_checks/sync_python.sh eval_checkpoint.py \
    --tree_root $R --drop_unexpected \
    --model_dir $L/qmugs-train-53709019-NN2-DCC-FSDP0-V2 --checkpoint qmugs-train-53709019-NN2-DCC-FSDP0-V2_epoch_99.pk \
    --basedir $Q/QMugs001_max800_delta_uptr_decodercmp.pickle --sampler_epoch 99 --batch_size 32 \
    --out $SC/E4b_dense.json

# resume config for T5_dense, from T2's run name (fixed once T2 has a job id)
T2_RUN=qmugs-train-$(jid T2)-NN2-DCC-FSDP0-V2
python3 sync_checks/make_resume_config.py $EX/qmugs_densmat.json "$T2_RUN" $EX/sync_checks/qmugs_densmat_resume_synccheck.json
submit T5_dense "$(dep afterany:E4b_dense afterok:T2)" \
    INPUTFILE=$EX/sync_checks/qmugs_densmat_resume_synccheck.json NUM_EPOCH=2 -- \
    -t 01:00:00 scripts/dcc/qmugs_densmat_dcc.sh

# ---------------------------------------------------------------------------------------------
# lane A (one node)

submit P1 "$(dep afterany:T5_dense)" MODELNAME=QMugs001_max800_delta_uptr_decodercmp_synccheck -- \
    -t 02:00:00 scripts/dcc/qmugs_pre_dcc.sh

submit P2 "$(dep afterany:P1)" \
    INPUTFILE=$EX/sync_checks/qmugs_raw_synccheck.json MODELNAME=QMugs0001_max800_synccheck PERC_LOAD=0.001 -- \
    -t 01:00:00 scripts/dcc/qmugs_pre_dcc.sh

submit P3 "$(dep afterany:P2)" \
    INPUTFILE=$EX/sync_checks/qmugs_raw_uptr_synccheck.json MODELNAME=QMugs0001_max800_uptr_synccheck PERC_LOAD=0.001 -- \
    -t 01:00:00 scripts/dcc/qmugs_pre_dcc.sh

submit E1 "$(dep afterany:P3)" -- $GPU1 $EXCL -t 01:30:00 sync_checks/sync_python.sh eval_checkpoint.py \
    --tree_root $R \
    --model_dir $L/qmugs-train-55914623-NN1-PM-FSDP0-V2 --checkpoint qmugs-train-55914623-NN1-PM-FSDP0-V2_epoch_111.pk \
    --basedir $Q/QMugs001_max800_delta.pickle --sampler_epoch 111 --batch_size 32 \
    --out $SC/E1.json

submit E3 "$(dep afterany:E1)" -- $GPU1 $EXCL -t 01:30:00 sync_checks/sync_python.sh eval_checkpoint.py \
    --tree_root $PRE \
    --model_dir $L/qmugs-schur-train-53692804-NN1-DCC-teacher_forcing-joint-total \
    --checkpoint qmugs-schur-train-53692804-NN1-DCC-teacher_forcing-joint-total_epoch_99.pk \
    --basedir $Q/QMugs001_max800_delta_uptr_decodercmp.pickle --sampler_epoch 99 --batch_size 128 \
    --out $SC/E3.json

submit T1 "$(dep afterany:E3)" \
    INPUTFILE=$EX/sync_checks/qmugs_egnn_synccheck.json MODELNAME=QMugs001_max800_delta NUM_EPOCH=5 BATCH_SIZE=32 -- \
    -N 1 -t 01:30:00 scripts/dcc/qmugs_densmat_dcc.sh

submit T3 "$(dep afterany:T1)" HYDRAGNN_USE_FSDP=1 NUM_EPOCH=2 -- \
    -N 1 -t 01:00:00 scripts/dcc/qmugs_densmat_dcc.sh

submit E1b "$(dep afterany:T3)" -- $GPU1 $EXCL -t 01:30:00 sync_checks/sync_python.sh eval_checkpoint.py \
    --tree_root $PRE \
    --model_dir $L/qmugs-train-55914623-NN1-PM-FSDP0-V2 --checkpoint qmugs-train-55914623-NN1-PM-FSDP0-V2_epoch_111.pk \
    --basedir $Q/QMugs001_max800_delta.pickle --sampler_epoch 111 --batch_size 32 \
    --out $SC/E1b.json

# U1 and U5: test.sh has no header; header of qmugs_schur_decoder_test_dcc.sh plus GPUs
submit U1 "$(dep afterany:E1b)" -- -A youlab -p youlab-gpu -N 1 --ntasks-per-node=1 -c 8 --mem=32G \
    --gres=gpu:4 $EXCL -t 00:20:00 -J QMugs-SyncU1 -o slurm_output/sync_checks/sync-U1-job-%j.out -e slurm_output/sync_checks/sync-U1-job-%j.out \
    --wrap "cd $R && bash examples/qmugs/scripts/dcc/test.sh"

submit U5 "$(dep afterany:U1)" -- -A youlab -p youlab-gpu -N 1 --ntasks-per-node=1 -c 8 --mem=32G \
    --gres=gpu:1 $EXCL -t 00:30:00 -J QMugs-SyncU5 -o slurm_output/sync_checks/sync-U5-job-%j.out -e slurm_output/sync_checks/sync-U5-job-%j.out \
    --wrap "cd $R && source installation_other/module_loads_dcc.sh && load_dcc_modules 12.8 && git log --oneline -1 && PYTHONPATH=$R $V/bin/python -m pytest -q -p no:cacheprovider tests/test_named_variable_schema.py tests/test_loss_reporting.py tests/test_config.py tests/test_create_model.py tests/test_example_training_loss_schema.py -m 'not mpi and not deepspeed and not network and not fairchem'"

T3_RUN=qmugs-train-$(jid T3)-NN1-DCC-FSDP1-V2
submit I3 "$(dep afterany:U5 afterok:T3)" \
    USE_ENV_ONLY=1 MODEL_DIR=$L/$T3_RUN CHECKPOINT_PATH=$T3_RUN.pk \
    DATASET_BASEDIR=$Q/QMugs001_max800_delta_uptr_decodercmp.pickle ARTIFACTS_DIR=$A/synccheck-inference-I3 \
    PERC_DOWNSAMPLE_TEST=0.1 RUN_DOWNSTREAM_CALCULATION=0 RUN_FIGS=0 -- \
    -N 1 -t 01:00:00 scripts/dcc/qmugs_inference_dcc.sh

submit I1 "$(dep afterany:I3)" \
    USE_ENV_ONLY=1 MODEL_DIR=$L/qmugs-train-55914623-NN1-PM-FSDP0-V2 CHECKPOINT_PATH=qmugs-train-55914623-NN1-PM-FSDP0-V2_epoch_111.pk \
    DATASET_BASEDIR=$Q/QMugs001_max800_delta.pickle ARTIFACTS_DIR=$A/artifacts_model-QMugs001_max800_delta_data-QMugs001_max800_delta \
    PERC_DOWNSAMPLE_TEST=1.0 RUN_DOWNSTREAM_CALCULATION=0 RUN_FIGS=0 -- \
    -N 1 -t 02:00:00 scripts/dcc/qmugs_inference_dcc.sh

# ---------------------------------------------------------------------------------------------
# lane B (one node)

submit T4 "$(dep afterany:T5_dense)" \
    SCHUR_INPUTFILE=$EX/sync_checks/qmugs_schur_synccheck.json SCHUR_NUM_EPOCH=10 -- \
    -t 04:30:00 scripts/dcc/qmugs_schur_dcc.sh

T4_RUN=qmugs-schur-train-$(jid T4)-NN1-DCC-teacher_forcing-joint-total
python3 sync_checks/make_resume_config.py $EX/sync_checks/qmugs_schur_synccheck.json "$T4_RUN" $EX/sync_checks/qmugs_schur_resume_synccheck.json
submit T5_schur "$(dep afterok:T4)" \
    SCHUR_INPUTFILE=$EX/sync_checks/qmugs_schur_resume_synccheck.json SCHUR_NUM_EPOCH=2 -- \
    -t 01:30:00 scripts/dcc/qmugs_schur_dcc.sh

submit E4a_schur "$(dep afterany:T5_schur)" -- $GPU1 $EXCL -t 01:30:00 sync_checks/sync_python.sh eval_checkpoint.py \
    --tree_root $R \
    --model_dir $L/qmugs-schur-train-53692804-NN1-DCC-teacher_forcing-joint-total \
    --checkpoint qmugs-schur-train-53692804-NN1-DCC-teacher_forcing-joint-total_epoch_99.pk \
    --basedir $Q/QMugs001_max800_delta_uptr_decodercmp.pickle --sampler_epoch 99 --batch_size 128 \
    --out $SC/E4a_schur.json

submit E4b_schur "$(dep afterany:E4a_schur)" -- $GPU1 $EXCL -t 01:30:00 sync_checks/sync_python.sh eval_checkpoint.py \
    --tree_root $R --drop_unexpected \
    --model_dir $L/qmugs-schur-train-53692804-NN1-DCC-teacher_forcing-joint-total \
    --checkpoint qmugs-schur-train-53692804-NN1-DCC-teacher_forcing-joint-total_epoch_99.pk \
    --basedir $Q/QMugs001_max800_delta_uptr_decodercmp.pickle --sampler_epoch 99 --batch_size 128 \
    --out $SC/E4b_schur.json

submit U3 "$(dep afterany:E4b_schur)" LOG_DIR=$L/qmugs-schur-diagnostics-synccheck -- \
    -t 02:00:00 scripts/dcc/qmugs_schur_decomp_test_dcc.sh

submit T7 "$(dep afterany:U3 afterok:P2)" \
    INPUTFILE=$EX/sync_checks/qmugs_raw_synccheck.json MODELNAME=QMugs0001_max800_synccheck NUM_EPOCH=5 -- \
    -N 1 -t 01:00:00 scripts/dcc/qmugs_densmat_dcc.sh

submit T8 "$(dep afterany:T7 afterok:P3)" \
    INPUTFILE=$EX/sync_checks/qmugs_raw_uptr_synccheck.json MODELNAME=QMugs0001_max800_uptr_synccheck NUM_EPOCH=5 -- \
    -N 1 -t 01:00:00 scripts/dcc/qmugs_densmat_dcc.sh

submit I2 "$(dep afterany:T8 afterok:T2)" \
    USE_ENV_ONLY=1 MODEL_DIR=$L/$T2_RUN CHECKPOINT_PATH=$T2_RUN.pk \
    DATASET_BASEDIR=$Q/QMugs001_max800_delta_uptr_decodercmp.pickle ARTIFACTS_DIR=$A/synccheck-inference-I2 \
    PERC_DOWNSAMPLE_TEST=0.01 RUN_DOWNSTREAM_CALCULATION=1 "DOWNSTREAM_CALCULATIONS=electron_density dipole_moment" \
    RUN_FIGS=1 RUN_SAMPLE_FIGS=1 RUN_ELECTRON_DENSITY_FIGS=1 RUN_DIPOLE_FIGS=1 -- \
    -N 1 -t 03:00:00 scripts/dcc/qmugs_inference_dcc.sh

# lane A's last job needs T7 (lane B), so it is submitted after lane B
T7_RUN=qmugs-train-$(jid T7)-NN1-DCC-FSDP0-V2
submit I4 "$(dep afterany:I1 afterok:T7)" \
    USE_ENV_ONLY=1 MODEL_DIR=$L/$T7_RUN CHECKPOINT_PATH=$T7_RUN.pk \
    DATASET_BASEDIR=$Q/QMugs0001_max800_synccheck.pickle ARTIFACTS_DIR=$A/synccheck-inference-I4 \
    PERC_DOWNSAMPLE_TEST=1.0 RUN_DOWNSTREAM_CALCULATION=0 RUN_FIGS=0 -- \
    -N 1 -t 01:00:00 scripts/dcc/qmugs_inference_dcc.sh

# ---------------------------------------------------------------------------------------------
# uncapped CPU jobs

submit U2 "" -- -t 00:30:00 scripts/dcc/qmugs_schur_decoder_test_dcc.sh

submit U4 "$(dep afterok:P1)" \
    "SCAN_DATASETS=QMugs001_max800_delta_uptr_decodercmp QMugs001_max800_delta_uptr_decodercmp_synccheck" -- \
    -t 03:00:00 scripts/dcc/qmugs_scan_dcc.sh

submit P1_compare "$(dep afterok:P1)" -- -t 06:00:00 sync_checks/sync_python.sh compare_pickles.py \
    --tree_root $R \
    --reference $Q/QMugs001_max800_delta_uptr_decodercmp.pickle \
    --candidate $Q/QMugs001_max800_delta_uptr_decodercmp_synccheck.pickle \
    --path_substitution DCC_MODEL-QMugs001_max800_delta_uptr_decodercmp_DATA-QMugs001_max800_delta_uptr_decodercmp \
                        DCC_MODEL-QMugs001_max800_delta_uptr_decodercmp_synccheck_DATA-QMugs001_max800_delta_uptr_decodercmp_synccheck \
    --reference_artifacts $A/DCC_MODEL-QMugs001_max800_delta_uptr_decodercmp_DATA-QMugs001_max800_delta_uptr_decodercmp \
    --candidate_artifacts $A/DCC_MODEL-QMugs001_max800_delta_uptr_decodercmp_synccheck_DATA-QMugs001_max800_delta_uptr_decodercmp_synccheck \
    --out $SC/P1_compare_pickles.json

submit P4_decodercmp "" -- -t 03:00:00 sync_checks/sync_python.sh ground_truth_check.py --tree_root $R \
    --basedir $Q/QMugs001_max800_delta_uptr_decodercmp.pickle --output_name density_matrix_delta_uptr --num 30 \
    --out $SC/P4_QMugs001_max800_delta_uptr_decodercmp.json

submit P4_delta "" -- -t 03:00:00 sync_checks/sync_python.sh ground_truth_check.py --tree_root $R \
    --basedir $Q/QMugs001_max800_delta.pickle --output_name density_matrix_delta --num 30 \
    --path_substitution /pscratch/sd/w/wxrderek/qmugs /work/xw262/qmugs \
    --path_substitution /pscratch/sd/w/wxrderek/artifacts_model-QMugs001_max800_delta_data-QMugs001_max800_delta \
                        $A/artifacts_model-QMugs001_max800_delta_data-QMugs001_max800_delta \
    --out $SC/P4_QMugs001_max800_delta.json

submit P4_raw "$(dep afterok:P2)" -- -t 03:00:00 sync_checks/sync_python.sh ground_truth_check.py --tree_root $R \
    --basedir $Q/QMugs0001_max800_synccheck.pickle --output_name density_matrix --num 30 \
    --out $SC/P4_QMugs0001_max800_synccheck.json

submit P4_raw_uptr "$(dep afterok:P3)" -- -t 03:00:00 sync_checks/sync_python.sh ground_truth_check.py --tree_root $R \
    --basedir $Q/QMugs0001_max800_uptr_synccheck.pickle --output_name density_matrix_uptr --num 30 \
    --out $SC/P4_QMugs0001_max800_uptr_synccheck.json

I2_LOG=$L/qmugs-inference-$(jid I2)-NN1-DCC
submit F1 "$(dep afterok:I2)" \
    USE_ENV_ONLY=1 LOG_DIR=$I2_LOG ARTIFACTS_DIR=$A/synccheck-inference-I2 \
    RUN_METRICS_FIGS=1 RUN_SAMPLE_FIGS=1 RUN_ELECTRON_DENSITY_FIGS=1 RUN_DIPOLE_FIGS=1 -- \
    -t 00:20:00 scripts/dcc/qmugs_figs_dcc.sh

# ---------------------------------------------------------------------------------------------
# collector: runs after everything, whether or not the checks succeeded

ALL=$(awk -F'\t' '$1 != "COLLECT" { print "afterany:" $2 }' "$JOBS" | sort -u | paste -sd, -)
submit COLLECT "$ALL" -- -t 00:30:00 sync_checks/sync_python.sh collect_results.py --jobs $JOBS --out_dir $SC
