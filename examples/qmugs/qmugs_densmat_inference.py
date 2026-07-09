import os, json
import logging
import sys
from mpi4py import MPI
import argparse
import time
import glob

import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.distributed as dist

# ----------------------------------------------------------------------------------------------------
# FIX random seed
random_state = 0
torch.manual_seed(random_state)
random.seed(random_state)
rng = np.random.default_rng(random_state)
# ----------------------------------------------------------------------------------------------------

import hydragnn

from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

from hydragnn.models.create import create_model_config
from hydragnn.train.train_validate_test import move_batch_to_device, resolve_precision, get_autocast_and_scaler
from hydragnn.utils.distributed import get_device
from hydragnn.utils.input_config_parsing.config_utils import update_config
from hydragnn.utils.print.print_utils import iterate_tqdm, log
from hydragnn.utils.model import print_model

try:
    import psi4
    psi4.core.be_quiet()
except ImportError:
    print('[WARNING] psi4 package not detected')

from qmugs import QMugsDataset, QMugsInference

# ----------------------------------------------------------------------------------------------------
# setup logging

logging.basicConfig(
    level=logging.INFO,
    format=f"%(levelname)s : %(message)s",
    datefmt="%H:%M:%S"
)
log_name = "QMugs_inference"
hydragnn.utils.print.setup_log(log_name)
writer = hydragnn.utils.model.get_summary_writer(log_name)

log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)
log(f'Random seed used for run: {random_state}', rank=0)

# ----------------------------------------------------------------------------------------------------
# additional evaluation criteria

def evaluate_matrix_constraints(all_pred, all_true, eps=1e-12):
    '''metrics for evaluating constraints on predicted density matrices'''

     # validate input and convert to numpy
    assert len(all_pred) == len(all_true)
    if torch.is_tensor(all_pred[0]):
        all_pred = [pred_mat.detach().cpu().numpy() for pred_mat in all_pred]
    if torch.is_tensor(all_true[0]):
        all_true = [true_mat.detach().cpu().numpy() for true_mat in all_true]

    trace_errors = []
    trace_comp = []
    hermitian_errors = []
    min_eigenvalues = []
    num_negative_eigenvalues = []

    for pred_mat, true_mat in zip(all_pred, all_true):
        assert pred_mat.shape == true_mat.shape

        # relative trace error
        trace_error = (np.trace(pred_mat) - np.trace(true_mat)) / (np.trace(true_mat) + eps)
        trace_errors.append(float(abs(trace_error)))
        trace_comp.append([float(np.trace(pred_mat)), float(np.trace(true_mat))])

        # hermitianity error in relative frobenius norm
        hermitian_error = np.linalg.norm(pred_mat - pred_mat.T, ord="fro") / (np.linalg.norm(pred_mat, ord="fro") + eps)
        hermitian_errors.append(float(hermitian_error))

        # PSD violation
        eigvals = np.linalg.eigvalsh((pred_mat + pred_mat.T) / 2)
        min_eigenvalues.append(float(np.min(eigvals)))

        negative = eigvals[eigvals < 0.0]
        num_negative_eigenvalues.append(int(len(negative)))


    return {
        "trace_error": {
            "mean": float(np.mean(trace_errors)),
            "std": float(np.std(trace_errors)),
            "values": trace_errors,
            "trace_comp": trace_comp,
        },

        "hermitian_error": {
            "mean": float(np.mean(hermitian_errors)),
            "std": float(np.std(hermitian_errors)),
            "values": hermitian_errors,
        },

        "minimum_eigenvalue": {
            "mean": float(np.mean(min_eigenvalues)),
            "std": float(np.std(min_eigenvalues)),
            "values": min_eigenvalues,
        },

        "negative_eigenvalue_count": {
            "mean": float(np.mean(num_negative_eigenvalues)),
            "std": float(np.std(num_negative_eigenvalues)),
            "values": num_negative_eigenvalues,
        },
    }


def evaluate_matrix_reconstruction(all_pred, all_true, eps=1e-12):
    '''other reconstruction metrics'''

    # validate input and convert to numpy
    assert len(all_pred) == len(all_true)
    if torch.is_tensor(all_pred[0]):
        all_pred = [pred_mat.detach().cpu().numpy() for pred_mat in all_pred]
    if torch.is_tensor(all_true[0]):
        all_true = [true_mat.detach().cpu().numpy() for true_mat in all_true]

    # flatten matrices and concatenate across full dataset
    y_pred = np.concatenate([mat.ravel() for mat in all_pred])
    y_true = np.concatenate([mat.ravel() for mat in all_true])

    # compute global R2
    ss_res_global = np.sum((y_true - y_pred) ** 2)
    ss_tot_global = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_global = 1.0 - ss_res_global / (ss_tot_global + eps)

    # compute other metrics
    r2_per_matrix = []
    relative_frobenius_errors = []
    cosine_similarities = []

    for pred_mat, true_mat in zip(all_pred, all_true):
        assert pred_mat.shape == true_mat.shape

        pred_flat = pred_mat.ravel()
        true_flat = true_mat.ravel()

        # R2 per matrix
        ss_res = np.sum((true_flat - pred_flat) ** 2)
        ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + eps)
        r2_per_matrix.append(float(r2))

        # relative frobenius error
        frob_rel = np.linalg.norm(pred_mat - true_mat, ord="fro") / (np.linalg.norm(true_mat, ord="fro") + eps)
        relative_frobenius_errors.append(float(frob_rel))

        # cosine similarity between flattened matrices
        cosine = np.dot(pred_flat, true_flat) / ((np.linalg.norm(pred_flat) * np.linalg.norm(true_flat)) + eps)
        cosine_similarities.append(float(cosine))

    return {
        "r2_global": float(r2_global),

        "r2_per_matrix": {
            "mean": float(np.mean(r2_per_matrix)),
            "std": float(np.std(r2_per_matrix)),
            "values": r2_per_matrix,
        },

        "relative_frobenius_error": {
            "mean": float(np.mean(relative_frobenius_errors)),
            "std": float(np.std(relative_frobenius_errors)),
            "values": relative_frobenius_errors,
        },

        "cosine_similarity": {
            "mean": float(np.mean(cosine_similarities)),
            "std": float(np.std(cosine_similarities)),
            "values": cosine_similarities,
        },
    }

# ----------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    model_dir = "/global/homes/w/wxrderek/HydraGNN/logs/qmugs-55245318-NN2-PM-FSDP0-V2-TP0"
    checkpoint_path = "qmugs-55245318-NN2-PM-FSDP0-V2-TP0_epoch_65.pk"
    basedir = "/pscratch/sd/w/wxrderek/qmugs/QMugs01.pickle"

    # ----------------------------------------------------------------------------------------------------
    # distributed setup

    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    
    # ----------------------------------------------------------------------------------------------------
    # load pretrained model

    log(f"[START] Loading HydraGNN pretrained model from {checkpoint_path}", rank=0)
    pretrained_model = QMugsInference(
        model_dir=model_dir,
        checkpoint_path=checkpoint_path,
    )
    config = pretrained_model.config
    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    log(f"[DONE] Loaded HydraGNN pretrained model from {checkpoint_path}", rank=0)

    # ----------------------------------------------------------------------------------------------------
    # load data

    log("[START] Lazy pickle data load for test set", rank=0)

    # open pickle metadata without preloading graph objects
    testset_meta = SimplePickleDataset(
        basedir=basedir,
        label="testset",
        preload=False,
        var_config=var_config,
    )

    # assign disjoint global pickle indices to each MPI rank
    local_subset = list(range(rank, testset_meta.ntotal, world_size))

    # create rank-local lazy dataset view
    testset = SimplePickleDataset(
        basedir=basedir,
        label="testset",
        subset=local_subset,
        preload=False,
        var_config=var_config,
    )

    log(
        "[DONE] testset global size: %d, local size on rank %d: %d"
        % (testset_meta.ntotal, rank, len(testset)),
        rank=0,
    )

    del testset_meta


    # ----------------------------------------------------------------------------------------------------
    # run inference

    local_evaluation, per_matrix_metrics = pretrained_model.run_streaming_inference(
        dataset=testset,
        evaluate=True,
        return_numpy_matrices=False,
        save_per_matrix_metrics=True,
        batch_size=1,
    )

    # ----------------------------------------------------------------------------------------------------
    # reduce streaming metrics across ranks

    def _reduce_metric_block(local_block):
        '''reduce one streaming metric block across MPI ranks'''

        # reconstruct local first and second moments
        local_count = float(local_evaluation['count'])
        local_sum = float(local_block['mean']) * local_count
        local_sumsq = (
            float(local_block['std']) ** 2
            + float(local_block['mean']) ** 2
        ) * local_count

        # reduce moments across all ranks
        global_count = comm.allreduce(local_count, op=MPI.SUM)
        global_sum = comm.allreduce(local_sum, op=MPI.SUM)
        global_sumsq = comm.allreduce(local_sumsq, op=MPI.SUM)

        # compute global statistics
        global_mean = global_sum / global_count
        global_var = max(global_sumsq / global_count - global_mean * global_mean, 0.0)

        return {
            'mean': global_mean,
            'std': global_var ** 0.5,
        }

    # reduce all streaming metrics
    global_evaluation = {}
    for key, value in local_evaluation.items():
        if isinstance(value, dict) and 'mean' in value and 'std' in value:
            global_evaluation[key] = _reduce_metric_block(value)

    # reduce total sample count
    global_evaluation['count'] = int(comm.allreduce(local_evaluation['count'], op=MPI.SUM))

    # store inference time of slowest rank
    global_evaluation['inference_time'] = comm.allreduce(
        local_evaluation['inference_time'],
        op=MPI.MAX,
    )

    # gather per matrix metrics from all MPI ranks
    all_per_matrix_metrics = comm.gather(per_matrix_metrics, root=0)

    # save combined per-matrix metrics on rank 0
    if rank == 0:
        combined_per_matrix_metrics = []
        for rank_metrics in all_per_matrix_metrics:
            combined_per_matrix_metrics.extend(rank_metrics)

        output_path = os.path.join(
            "logs",
            log_name,
            "per_matrix_metrics.json",
        )

        with open(output_path, "w") as f:
            json.dump(combined_per_matrix_metrics, f, indent=4)

    # log final metrics on rank 0
    if rank == 0:
        log(json.dumps(global_evaluation, indent=4), rank=0)

    dist.destroy_process_group()