import os, json
import logging
import sys
from mpi4py import MPI
import argparse
import time
import glob
import importlib.util

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


def _load_figs_main():
    '''load figs.py without ambiguity from the figs/ package directory'''

    # figs.py and the figs/ helper directory share a name, so load by path
    figs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs.py")
    spec = importlib.util.spec_from_file_location("qmugs_figs_main", figs_path)
    figs_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figs_module)
    return figs_module.main


def _figs_sample_mode(args):
    '''return the selected figure sampling mode'''

    # random molecule sampling is the default when neither flag is passed
    if args.figs_smallest_molecules:
        return "smallest_molecules"
    if args.figs_best_conformers:
        return "best_conformers"
    return "random_molecules"


def _density_matrix_max_size_from_var_config(var_config):
    '''return padded density matrix size from model output dimension'''

    total_density_names = [
        ["density_matrix"],
        ["density_matrix_delta"],
        ["density_matrix_uptr"],
        ["density_matrix_delta_uptr"],
    ]
    upper_triangle_names = [["density_matrix_uptr"], ["density_matrix_delta_uptr"]]
    if var_config["output_names"] not in total_density_names:
        return None

    output_dim = int(var_config["output_dim"][0])
    if var_config["output_names"] in upper_triangle_names:
        # upper-triangle head length is T(N_pad) = N_pad(N_pad+1)/2
        from utils import triangular_side
        return triangular_side(output_dim)

    # output_dim is N_pad ** 2 for the full total density matrix
    max_size = int(round(output_dim ** 0.5))
    if max_size * max_size != output_dim:
        raise ValueError(f"Density matrix output_dim must be a square, got {output_dim}")

    return max_size


def _model_output_names_from_config(model_dir: str):
    '''return model output names from a saved HydraGNN config'''

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(config_path, "r") as f:
        model_config = json.load(f)

    return model_config["NeuralNetwork"]["Variables_of_interest"]["output_names"]


def _pre_mpi_rank():
    '''infer rank before distributed setup is initialized'''

    # use common launcher rank variables, falling back to single-process rank zero
    for rank_name in ['SLURM_PROCID', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'RANK']:
        rank_value = os.environ.get(rank_name)
        if rank_value is not None:
            return int(rank_value)

    return 0


def _save_inference_run_config(args, random_state: int):
    '''save run-defining inference configuration before inference starts'''

    if _pre_mpi_rank() != 0:
        return None

    log_dir = os.path.join("logs", args.log)
    os.makedirs(log_dir, exist_ok=True)
    output_path = os.path.join(log_dir, "inference_run_config.json")

    psi4_version = None
    if 'psi4' in globals():
        psi4_version = getattr(psi4, '__version__', None)

    # only store configs that define this inference task
    inference_config = {
        'model_dir': args.model_dir,
        'checkpoint_path': args.checkpoint_path,
        'model_output_names': args.model_output_names,
        'basedir': args.basedir,
        'qmugs_data_dir': args.qmugs_data_dir,
        'artifacts_dir': args.artifacts_dir,
        'log': args.log,
        'batch_size': args.batch_size,
        'run_downstream_calculation': args.run_downstream_calculation,
        'downstream_calculations': args.downstream_calculations,
        'run_downstream_prediction': args.run_downstream_prediction,
        'downstream_predictions': args.downstream_predictions,
        'electron_density_method': args.electron_density_method,
        'cubeprop_grid_spacing': args.cubeprop_grid_spacing,
        'cubeprop_num_grid_points': args.cubeprop_num_grid_points,
        'manual_density_padding_angstrom': args.manual_density_padding_angstrom,
        'manual_density_spacing_angstrom': args.manual_density_spacing_angstrom,
        'manual_density_num_grid_points': args.manual_density_num_grid_points,
        'manual_density_block_size': args.manual_density_block_size,
        # figure settings are saved so standalone plotting can reproduce the run
        'run_figs': args.run_figs,
        'run_sample_figs': args.run_sample_figs,
        'run_electron_density_figs': args.run_electron_density_figs,
        'run_dipole_figs': args.run_dipole_figs,
        'figs_sample_mode': _figs_sample_mode(args),
        'figs_num_random_molecules': args.figs_num_random_molecules,
        'figs_num_best_conformers': args.figs_num_best_conformers,
        'figs_num_smallest_molecules': args.figs_num_smallest_molecules,
        'figs_seed': args.figs_seed,
        'figs_density_point_filter': args.figs_density_point_filter,
        'figs_density_top_percentile': args.figs_density_top_percentile,
        'figs_density_absolute_cutoff': args.figs_density_absolute_cutoff,
        'figs_density_max_points': args.figs_density_max_points,
        'figs_density_isosurface_percentile': args.figs_density_isosurface_percentile,
        'figs_density_isosurface_absolute_level': args.figs_density_isosurface_absolute_level,
        'figs_show_3d_axes': args.figs_show_3d_axes,
        'figs_hide_legend': args.figs_hide_legend,
        'figs_matrix_pool_size': args.figs_matrix_pool_size,
        'figs_metrics_max_natoms': args.figs_metrics_max_natoms,
        'figs_metrics_max_density_matrix_dim': args.figs_metrics_max_density_matrix_dim,
        'figs_metrics_natoms_bin_size': args.figs_metrics_natoms_bin_size,
        'figs_metrics_density_matrix_dim_bin_size': args.figs_metrics_density_matrix_dim_bin_size,
    }

    run_config = {
        'command': " ".join(sys.argv),
        'argv': list(sys.argv),
        'timestamp_utc': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'cwd': os.getcwd(),
        'random_state': random_state,
        'python_executable': sys.executable,
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'psi4_version': psi4_version,
        'inference_config': inference_config,
    }

    with open(output_path, "w") as f:
        json.dump(run_config, f, indent=4)

    return output_path


def _update_downsampled_inference_run_config(
    run_config_path: str,
    downsample_config,
    downsampled_test_conformers,
):
    '''record downsampled conformers in the saved inference config'''

    if run_config_path is None or _pre_mpi_rank() != 0:
        return

    with open(run_config_path, "r") as f:
        run_config = json.load(f)

    inference_config = run_config["inference_config"]
    inference_config["downsample_test_config"] = downsample_config
    inference_config["downsampled_test_conformers"] = downsampled_test_conformers

    with open(run_config_path, "w") as f:
        json.dump(run_config, f, indent=4)


if __name__ == "__main__":

    # ----------------------------------------------------------------------------------------------------
    # args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="directory containing model config and checkpoint files",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="checkpoint filename relative to model_dir, or absolute checkpoint path",
    )
    parser.add_argument(
        "--basedir",
        type=str,
        required=True,
        help="preprocessed pickle dataset directory",
    )
    parser.add_argument(
        "--qmugs_data_dir",
        type=str,
        default=None,
        help="QMugs raw data directory containing tarball_assignment.csv and wfns/",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default=None,
        help="directory for downstream calculation artifacts such as cubeprop files",
    )
    parser.add_argument("--log", help="log name", default="QMugs_inference")
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument(
        "--perc_downsample_test",
        type=float,
        default=1.0,
        help="proportion of test conformers to use for inference; 1.0 uses the full test set",
    )
    parser.add_argument(
        "--run_downstream_calculation",
        action="store_true",
        help="run requested downstream physics calculations during inference",
    )
    parser.add_argument(
        "--downstream_calculations",
        nargs="*",
        default=None,
        help="downstream physics calculations to run, e.g. dipole_moment electron_density",
    )
    parser.add_argument(
        "--electron_density_method",
        choices=["cubeprop", "manual_ao"],
        default="cubeprop",
        help="method for downstream electron density calculations",
    )
    parser.add_argument(
        "--cubeprop_grid_spacing",
        type=float,
        nargs="+",
        default=None,
        help="Psi4 cubeprop grid spacing in bohr; provide one value or three axis values",
    )
    parser.add_argument(
        "--cubeprop_num_grid_points",
        type=int,
        nargs="+",
        default=None,
        help="target number of cubeprop grid points; provide one value or three axis values",
    )
    parser.add_argument(
        "--manual_density_padding_angstrom",
        type=float,
        default=3.0,
        help="manual AO density grid padding around molecular bounding box in angstrom",
    )
    parser.add_argument(
        "--manual_density_spacing_angstrom",
        type=float,
        nargs="+",
        default=None,
        help="manual AO density grid spacing in angstrom; provide one value or three axis values",
    )
    parser.add_argument(
        "--manual_density_num_grid_points",
        type=int,
        nargs="+",
        default=None,
        help="manual AO density grid point count; provide one value or three axis values",
    )
    parser.add_argument(
        "--manual_density_block_size",
        type=int,
        default=2000,
        help="number of grid points per block for manual AO density evaluation",
    )
    parser.add_argument(
        "--run_downstream_prediction",
        action="store_true",
        help="run requested downstream ML predictions during inference",
    )
    parser.add_argument(
        "--downstream_predictions",
        nargs="*",
        default=None,
        help="downstream ML predictions to run, e.g. dipole_moment",
    )
    parser.add_argument(
        "--run_figs",
        action="store_true",
        help="run figure generation after inference metrics are written",
    )
    # sampled figures rerun inference on a small subset after streaming inference
    parser.add_argument(
        "--run_sample_figs",
        action="store_true",
        help="rerun inference on sampled conformers and plot sampled density matrix figures",
    )
    parser.add_argument(
        "--run_electron_density_figs",
        action="store_true",
        help="plot electron density isosurfaces for sampled conformers",
    )
    parser.add_argument(
        "--run_dipole_figs",
        action="store_true",
        help="plot dipole moment arrows for sampled conformers",
    )
    figs_sample_group = parser.add_mutually_exclusive_group()
    figs_sample_group.add_argument(
        "--figs_random_molecules",
        action="store_true",
        help="sample random molecules for figure generation",
    )
    figs_sample_group.add_argument(
        "--figs_best_conformers",
        action="store_true",
        help="sample conformers with lowest per-matrix RMSE for figure generation",
    )
    figs_sample_group.add_argument(
        "--figs_smallest_molecules",
        action="store_true",
        help="sample molecules with the fewest atoms for figure generation",
    )
    parser.add_argument(
        "--figs_num_random_molecules",
        type=int,
        default=5,
        help="number of random molecules to sample for figures",
    )
    parser.add_argument(
        "--figs_num_best_conformers",
        type=int,
        default=10,
        help="number of lowest-RMSE conformers to sample for figures",
    )
    parser.add_argument(
        "--figs_num_smallest_molecules",
        type=int,
        default=10,
        help="number of smallest molecules by atom count to sample for figures",
    )
    parser.add_argument(
        "--figs_seed",
        type=int,
        default=random_state,
        help="random seed for figure sampling",
    )
    parser.add_argument(
        "--figs_density_point_filter",
        choices=["top_percentile", "absolute_cutoff", "max_points"],
        default="top_percentile",
        help="method used to select density grid points for figure point clouds",
    )
    parser.add_argument(
        "--figs_density_top_percentile",
        type=float,
        default=99.0,
        help="percentile cutoff for density point cloud figures",
    )
    parser.add_argument(
        "--figs_density_absolute_cutoff",
        type=float,
        default=1.0e-4,
        help="absolute cutoff for density point cloud figures",
    )
    parser.add_argument(
        "--figs_density_max_points",
        type=int,
        default=5000,
        help="maximum number of points in each density point cloud figure",
    )
    parser.add_argument(
        "--figs_density_isosurface_percentile",
        type=float,
        default=99.0,
        help="percentile level used for density isosurface figures",
    )
    parser.add_argument(
        "--figs_density_isosurface_absolute_level",
        type=float,
        default=None,
        help="absolute density level for isosurface figures; overrides percentile when provided",
    )
    parser.add_argument(
        "--figs_show_3d_axes",
        action="store_true",
        help="show 3D axis grid and labels behind molecule visualizations",
    )
    parser.add_argument(
        "--figs_hide_legend",
        action="store_true",
        help="hide legends on molecule, density, and dipole visualizations",
    )
    parser.add_argument(
        "--figs_matrix_pool_size",
        type=int,
        default=1,
        help="average-pooling block size for sampled density matrix heatmaps",
    )
    parser.add_argument(
        "--figs_metrics_max_natoms",
        type=float,
        default=None,
        help="maximum atom count included in metric-by-size plots",
    )
    parser.add_argument(
        "--figs_metrics_max_density_matrix_dim",
        type=float,
        default=None,
        help="maximum density matrix dimension included in metric-by-size plots",
    )
    parser.add_argument(
        "--figs_metrics_natoms_bin_size",
        type=float,
        default=None,
        help="atom-count bin size for metric-by-size plots",
    )
    parser.add_argument(
        "--figs_metrics_density_matrix_dim_bin_size",
        type=float,
        default=None,
        help="density-matrix-dimension bin size for metric-by-size plots",
    )
    args = parser.parse_args()

    # validate data dir
    if args.qmugs_data_dir is None:
        args.qmugs_data_dir = os.path.dirname(os.path.abspath(args.basedir))
    args.model_output_names = _model_output_names_from_config(model_dir=args.model_dir)
    if args.artifacts_dir is None:
        if args.model_output_names in [["density_matrix_delta"], ["density_matrix_delta_uptr"]]:
            raise ValueError("--artifacts_dir is required for delta density matrix inference")
        args.artifacts_dir = os.path.join("logs", args.log, "artifacts")
    args.artifacts_dir = os.path.abspath(args.artifacts_dir)
    
    # validate downsampling request
    if args.perc_downsample_test <= 0.0 or args.perc_downsample_test > 1.0:
        raise ValueError(
            f"--perc_downsample_test must satisfy 0 < perc_downsample_test <= 1, "
            f"got {args.perc_downsample_test}"
        )

    # save run-defining config before validation, distributed setup, data loading, or inference
    run_config_path = _save_inference_run_config(
        args=args,
        random_state=random_state,
    )

    # validate grid config inputs
    if args.cubeprop_grid_spacing is not None and args.cubeprop_num_grid_points is not None:
        raise ValueError("Specify only one of --cubeprop_grid_spacing or --cubeprop_num_grid_points")
    if args.manual_density_spacing_angstrom is not None and args.manual_density_num_grid_points is not None:
        raise ValueError("Specify only one of --manual_density_spacing_angstrom or --manual_density_num_grid_points")

    # ----------------------------------------------------------------------------------------------------
    # distributed setup

    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    # ----------------------------------------------------------------------------------------------------
    # setup logging

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s (rank {rank}): %(message)s",
        datefmt="%H:%M:%S"
    )
    log_name = args.log
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)

    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)
    log(f'Random seed used for run: {random_state}', rank=0)
    log(f'Artifacts directory: {args.artifacts_dir}', rank=0)
    if run_config_path is not None:
        log(f'Inference run config: {run_config_path}', rank=0)
    
    # ----------------------------------------------------------------------------------------------------
    # load pretrained model

    checkpoint_log_name = (
        args.checkpoint_path
        if args.checkpoint_path is not None
        else "latest .pk checkpoint in model_dir"
    )
    log(f"[START] Loading HydraGNN pretrained model from {checkpoint_log_name}", rank=0)
    pretrained_model = QMugsInference(
        model_dir=args.model_dir,
        checkpoint_path=args.checkpoint_path,
    )
    config = pretrained_model.config
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    max_size = _density_matrix_max_size_from_var_config(var_config=var_config)

    log(f"[DONE] Loaded HydraGNN pretrained model from {pretrained_model.checkpoint_path}", rank=0)
    if max_size is not None:
        log(f"[INFO] Density matrix padded dimension from model config: {max_size}", rank=0)

    # ----------------------------------------------------------------------------------------------------
    # load data

    log("[START] Lazy pickle data load for test set", rank=0)

    # open pickle metadata without preloading graph objects
    testset_meta = SimplePickleDataset(
        basedir=args.basedir,
        label="testset",
        preload=False,
        var_config=var_config,
    )

    # choose global pickle indices before assigning disjoint work to MPI ranks
    global_subset = list(range(testset_meta.ntotal))
    downsample_config = None
    downsampled_test_conformers = None

    # perform downsampling
    if args.perc_downsample_test < 1.0:
        num_downsampled_conformers = max(1, int(testset_meta.ntotal * args.perc_downsample_test))
        if num_downsampled_conformers < testset_meta.ntotal:
            downsample_rng = np.random.default_rng(random_state)
            global_subset = sorted([
                int(index)
                for index in downsample_rng.choice(
                    testset_meta.ntotal,
                    size=num_downsampled_conformers,
                    replace=False,
                )
            ])
            downsample_config = {
                'perc_downsample_test': args.perc_downsample_test,
                'downsample_test_seed': random_state,
                'original_test_num_conformers': testset_meta.ntotal,
                'downsampled_test_num_conformers': len(global_subset),
            }

            if rank == 0:
                downsampled_test_conformers = []
                for global_index in global_subset:
                    data_object = testset_meta.read(global_index)
                    downsampled_test_conformers.append([
                        str(data_object.chembl_id),
                        str(data_object.conformer_id),
                    ])
                    del data_object

                _update_downsampled_inference_run_config(
                    run_config_path=run_config_path,
                    downsample_config=downsample_config,
                    downsampled_test_conformers=downsampled_test_conformers,
                )

                log(
                    "[INFO] Downsampled test conformers for inference: "
                    f"{len(global_subset)}/{testset_meta.ntotal} "
                    f"with perc_downsample_test={args.perc_downsample_test} "
                    f"and seed={random_state}",
                    rank=0,
                )
                if run_config_path is not None:
                    log(
                        "[INFO] Updated inference run config with downsampled test conformers: "
                        f"{run_config_path}",
                        rank=0,
                    )
        else:
            log(
                "[INFO] Requested test conformer downsampling selected the full test set; "
                "using all test conformers",
                rank=0,
            )

    # assign disjoint selected global pickle indices to each MPI rank
    local_subset = global_subset[rank::world_size]

    # create rank-local lazy dataset view
    testset = SimplePickleDataset(
        basedir=args.basedir,
        label="testset",
        subset=local_subset,
        preload=False,
        var_config=var_config,
    )

    log(
        "[DONE] testset global size: %d, selected size: %d, local size on rank %d: %d"
        % (testset_meta.ntotal, len(global_subset), rank, len(testset)),
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
        batch_size=args.batch_size,
        max_size=max_size,
        run_downstream_calculation=args.run_downstream_calculation,
        downstream_calculations=args.downstream_calculations,
        qmugs_data_dir=args.qmugs_data_dir,
        artifacts_dir=args.artifacts_dir,
        electron_density_method=args.electron_density_method,
        cubeprop_grid_spacing=args.cubeprop_grid_spacing,
        cubeprop_num_grid_points=args.cubeprop_num_grid_points,
        manual_density_padding_angstrom=args.manual_density_padding_angstrom,
        manual_density_spacing_angstrom=args.manual_density_spacing_angstrom,
        manual_density_num_grid_points=args.manual_density_num_grid_points,
        manual_density_block_size=args.manual_density_block_size,
        run_downstream_prediction=args.run_downstream_prediction,
        downstream_predictions=args.downstream_predictions,
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

    # save metrics on rank 0
    if rank == 0:
        combined_per_matrix_metrics = []
        for rank_metrics in all_per_matrix_metrics:
            combined_per_matrix_metrics.extend(rank_metrics)

        # save global metrics
        global_output_path = os.path.join(
            "logs",
            log_name,
            "global_evaluation_metrics.json",
        )

        with open(global_output_path, "w") as f:
            json.dump(global_evaluation, f, indent=4)

        # save per matrix metrics
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

    # ensure all ranks have finished writing and reducing before rank 0 makes figures
    comm.Barrier()

    if args.run_figs and rank == 0:
        log("[START] Generating inference figures", rank=0)
        figs_main = _load_figs_main()
        # plotting is outside streaming inference and uses saved per-matrix JSONs
        figs_main(
            log_dir=os.path.abspath(os.path.join("logs", log_name)),
            artifacts_dir=args.artifacts_dir,
            run_sample_visuals=args.run_sample_figs,
            run_electron_density_visuals=args.run_electron_density_figs,
            run_dipole_visuals=args.run_dipole_figs,
            sample_mode=_figs_sample_mode(args),
            num_random_molecules=args.figs_num_random_molecules,
            num_best_conformers=args.figs_num_best_conformers,
            num_smallest_molecules=args.figs_num_smallest_molecules,
            seed=args.figs_seed,
            density_point_filter=args.figs_density_point_filter,
            density_top_percentile=args.figs_density_top_percentile,
            density_absolute_cutoff=args.figs_density_absolute_cutoff,
            density_max_points=args.figs_density_max_points,
            density_isosurface_percentile=args.figs_density_isosurface_percentile,
            density_isosurface_absolute_level=args.figs_density_isosurface_absolute_level,
            show_3d_axes=args.figs_show_3d_axes,
            show_legend=not args.figs_hide_legend,
            matrix_pool_size=args.figs_matrix_pool_size,
            metrics_max_natoms=args.figs_metrics_max_natoms,
            metrics_max_density_matrix_dim=args.figs_metrics_max_density_matrix_dim,
            metrics_natoms_bin_size=args.figs_metrics_natoms_bin_size,
            metrics_density_matrix_dim_bin_size=args.figs_metrics_density_matrix_dim_bin_size,
        )
        log("[DONE] Generated inference figures", rank=0)

    comm.Barrier()

    dist.destroy_process_group()
