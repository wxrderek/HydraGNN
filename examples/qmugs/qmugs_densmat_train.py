import os, json
import logging
import sys
from mpi4py import MPI
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

import random
import torch
import torch.distributed as dist

# ----------------------------------------------------------------------------------------------------
# FIX random seed
random_state = 0
torch.manual_seed(random_state)
random.seed(random_state)
rng = np.random.default_rng(random_state)
# ----------------------------------------------------------------------------------------------------

from torch_geometric.transforms import Distance

import hydragnn
from hydragnn.utils.model import print_model
from hydragnn.utils.print.print_utils import iterate_tqdm, log, log0
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
    gather_deg,
)

import hydragnn.utils.profiling_and_tracing.tracer as tr
from hydragnn.utils.profiling_and_tracing.time_utils import Timer

from hydragnn.utils.datasets.distdataset import DistDataset
from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

try:
    import psi4
    psi4.core.be_quiet()
except ImportError:
    print('[WARNING] psi4 package not detected')

from qmugs import (
    QMugsDataset,
)

# ----------------------------------------------------------------------------------------------------
transform_coordinates = Distance(norm=False, cat=False)
# ----------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # ----------------------------------------------------------------------------------------------------
    # args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only (no training)",
    )
    parser.add_argument(
        "--streaming_preonly",
        action="store_true",
        help="preprocess only (no training), with streaming dataset creation, compatible with pickle format only",
    )
    parser.add_argument(
        "--inputfile", help="input config file", type=str, default="qmugs_densmat.json"
    )
    parser.add_argument("--ddstore", action="store_true", help="ddstore dataset")
    parser.add_argument("--ddstore_width", type=int, help="ddstore width", default=None)
    parser.add_argument("--log", help="log name")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--num_epoch", type=int, help="number of epochs to train", default=None)
    parser.add_argument("--everyone", action="store_true", help="gptimer")
    parser.add_argument("--modelname", help="model name")
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default=None,
        help="directory for preprocessing artifacts such as promolecular density matrices",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp64", "bf16"],
        default=None,
        help="Override precision; defaults to fp32 when not set",
    )

    parser.add_argument("--perc_load", type=float, help="percentage of all molecules in dataset to load", default=0.0001)
    parser.add_argument("--perc_train", type=float, help="percentage of loaded moledules to assign to train set", default=0.8)
    parser.add_argument(
        "--max_density_matrix_size",
        type=int,
        help="maximum density matrix dimension allowed in sampled molecules",
        default=None,
    )
    parser.add_argument(
        "--update_max_padded_dimension",
        action="store_true",
        help="set the padded density matrix dimension to max_density_matrix_size",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    parser.set_defaults(format="pickle") # not sure how ADIOS2 works on Perlmutter
    args = parser.parse_args()

    # ----------------------------------------------------------------------------------------------------
    # set up names, files, logs
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, args.inputfile)

    # read config files
    with open(input_filename, 'r') as f:
        config = json.load(f)
    with open(os.path.join(dirpwd, 'utils/download_data.json'), 'r') as f:
        download_config = json.load(f)
        
    dataset_dir = download_config.get('data_dir', './dataset')    
    verbosity = config["Verbosity"]["level"]

    if args.update_max_padded_dimension:
        if args.max_density_matrix_size is None:
            raise ValueError(
                "--update_max_padded_dimension requires --max_density_matrix_size"
            )
        QMugsDataset.max_padded_density_matrix_dimension = args.max_density_matrix_size

    # N_pad already reflects any --update_max_padded_dimension override above
    N_pad = QMugsDataset.max_padded_density_matrix_dimension

    # set up features
    var_config = config["NeuralNetwork"]["Variables_of_interest"]

    # determine the task variant from the configured output names
    if var_config["output_names"] == ['density_matrix']:
        predict_alpha_beta = False
        density_matrix_delta = False
        upper_triangle = False
    elif var_config["output_names"] == ['density_matrix_delta']:
        predict_alpha_beta = False
        density_matrix_delta = True
        upper_triangle = False
    elif var_config["output_names"] == ['density_matrix_uptr']:
        predict_alpha_beta = False
        density_matrix_delta = False
        upper_triangle = True
    elif var_config["output_names"] == ['density_matrix_delta_uptr']:
        predict_alpha_beta = False
        density_matrix_delta = True
        upper_triangle = True
    elif var_config["output_names"] == ['alpha_density_matrix', 'beta_density_matrix']:
        predict_alpha_beta = True
        density_matrix_delta = False
        upper_triangle = False
    else:
        raise ValueError("Unknown configuration in 'Variables_of_interest' for density matrix learning task")

    # the padded head length is T(N_pad) = N_pad(N_pad+1)/2 for upper-triangle targets, else N_pad**2
    if upper_triangle:
        padding_dim = int(N_pad * (N_pad + 1) // 2)
    else:
        padding_dim = int(N_pad ** 2)

    graph_feature_names = list(var_config["output_names"])
    graph_feature_dims = [padding_dim] * len(var_config["output_names"])

    if args.update_max_padded_dimension:
        var_config["output_dim"] = [padding_dim] * len(var_config["output_names"])

    if density_matrix_delta and args.preonly and args.artifacts_dir is None:
        raise ValueError("--artifacts_dir is required for delta density matrix preprocessing")
    if density_matrix_delta and args.format != "pickle":
        raise NotImplementedError("delta density matrix learning is implemented only for pickle datasets")
    
    node_feature_names = ['atomic_number', 'cartesian_coordinates']
    node_feature_dims = [1, 3]
    
    # variables of interest configs
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims

    # mask config
    mask_output_loss = var_config.get("mask_output_loss", None)

    # ensure consistent padded matrix dimensions
    assert all(output_dim == padding_dim for output_dim in var_config["output_dim"])

    # reset batch size and epochs if specified
    if args.batch_size is not None:
        config["NeuralNetwork"]["Training"]["batch_size"] = args.batch_size
    if args.num_epoch is not None: 
        config["NeuralNetwork"]["Training"]["num_epoch"] = args.num_epoch
    
    comm_size, rank = hydragnn.utils.distributed.setup_ddp()
    comm = MPI.COMM_WORLD

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(levelname)s (rank {rank}): %(message)s",
        datefmt="%H:%M:%S"
    )
    log_name = "QMugs" if args.log is None else args.log
    hydragnn.utils.print.setup_log(log_name)
    writer = hydragnn.utils.model.get_summary_writer(log_name)
    
    log("Command: {0}\n".format(" ".join([x for x in sys.argv])), rank=0)
    log(f'Random seed used for run: {random_state}', rank=0)

    # ----------------------------------------------------------------------------------------------------
    # load data and perform splitting
    modelname = "QMugs" if args.modelname is None else args.modelname
    if args.preonly:
        
        log(f"[START] Initializing dataset object", rank=0)

        # initialize dataset object
        dataset_object = QMugsDataset(
            config=config,
            download_config=download_config,
            graphgps_transform=None,
            energy_per_atom=False,
            dist=True,
            pad_density_matrix=True,
            mask_output_loss=mask_output_loss,
            predict_alpha_beta=predict_alpha_beta,
            density_matrix_delta=density_matrix_delta,
            upper_triangle=upper_triangle,
            artifacts_dir=args.artifacts_dir,
        )

        if args.streaming_preonly:
            # use streaming to overcome memory constraints, available for pickle format only

            if args.format != "pickle":
                raise NotImplementedError("Streaming preonly path is implemented only for pickle")

            start = time.perf_counter()
            basedir = os.path.join(dataset_dir, "%s.pickle" % modelname)
            streaming_info = dataset_object.load_and_split_dataset_pickle_streaming(
                basedir=basedir,
                comm=comm,
                perc_load=args.perc_load,
                perc_train=args.perc_train,
                max_density_matrix_size=args.max_density_matrix_size,
                update_max_padded_dimension=args.update_max_padded_dimension,
                use_subdir=True,
                nmax_persubdir=10_000,
                compute_pna_deg=True,
            )

            config["pna_deg"] = streaming_info["pna_deg"]
            end = time.perf_counter()

            log(f"[INFO] Streaming split info: {json.dumps({k: v for k, v in streaming_info.items() if k != 'pna_deg'}, indent=4)}", rank=0)
            log(f"[INFO] Time for streaming dataset preprocessing: {end-start}", rank=0)
            log(f"[DONE] Saved trainset, testset, valset to {basedir}\n", rank=0)

            sys.exit(0)
        
        else:
            # load and split locally downloaded and extracted data
            start = time.perf_counter()
            trainset, valset, testset, split_sizes = dataset_object.load_and_split_dataset(
                perc_load=args.perc_load, 
                perc_train=args.perc_train,
                max_density_matrix_size=args.max_density_matrix_size,
                update_max_padded_dimension=args.update_max_padded_dimension,
            )
            end = time.perf_counter()

            log(f'[INFO] Split sizes: {json.dumps(split_sizes, indent=4)}', rank=0)
            log(f'[INFO] Time for dataset loading + splitting (without accounting for object initialization): {end-start}', rank=0)

            deg = gather_deg(trainset)
            config["pna_deg"] = deg

            setnames = ["trainset", "valset", "testset"]

            # adios 
            if args.format == "adios":
                fname = os.path.join(
                    dataset_dir, "%s.bp" % modelname
                )
                adwriter = AdiosWriter(fname, comm)
                adwriter.add("trainset", trainset)
                adwriter.add("valset", valset)
                adwriter.add("testset", testset)
                adwriter.add_global("pna_deg", deg)
                adwriter.save()
            
            # pickle
            elif args.format == "pickle":
                basedir = os.path.join(dataset_dir, "%s.pickle" % modelname)
                attrs = {"pna_deg": deg}
                SimplePickleWriter(
                    trainset,
                    basedir,
                    "trainset",
                    use_subdir=True,
                    attrs=attrs,
                )
                SimplePickleWriter(
                    valset,
                    basedir,
                    "valset",
                    use_subdir=True,
                )
                SimplePickleWriter(
                    testset,
                    basedir,
                    "testset",
                    use_subdir=True,
                )
            log(f'[DONE] Saved trainset, testset, valset to {basedir}\n', rank=0)
            sys.exit(0)
            

    # ----------------------------------------------------------------------------------------------------
    # load preprocessed data

    tr.initialize()
    tr.disable()
    timer = Timer('load_data')
    timer.start()

    # adios
    if args.format == "adios":
        assert AdiosDataset is not None, "ADIOS support not available"
        log("[START] Adios data load", rank=0)
        assert not (args.shmem and args.ddstore), "Cannot use both ddstore and shmem"
        opt = {
            "preload": False,
            "shmem": args.shmem,
            "ddstore": args.ddstore,
            "ddstore_width": args.ddstore_width,
        }
        fname = os.path.join(dataset_dir, "%s.bp" % modelname)
        trainset = AdiosDataset(fname, "trainset", comm, **opt, var_config=var_config)
        valset = AdiosDataset(fname, "valset", comm, **opt, var_config=var_config)
        testset = AdiosDataset(fname, "testset", comm, **opt, var_config=var_config)
    
    # pickle
    elif args.format == "pickle":
        log("[START] Pickle data load", rank=0)
        basedir = os.path.join(dataset_dir, "%s.pickle" % modelname)
        trainset = SimplePickleDataset(
            basedir=basedir, label="trainset", var_config=var_config
        )
        valset = SimplePickleDataset(
            basedir=basedir, label="valset", var_config=var_config
        )
        testset = SimplePickleDataset(
            basedir=basedir, label="testset", var_config=var_config
        )
        pna_deg = trainset.pna_deg
        if args.ddstore:
            opt = {"ddstore_width": args.ddstore_width}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset = DistDataset(valset, "valset", comm, **opt)
            testset = DistDataset(testset, "testset", comm, **opt)
            trainset.pna_deg = pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    log(
        "[DONE] trainset, valset, testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset)), rank=0
    )

    if args.ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"
    
    # get data loaders
    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )

    comm.Barrier()

    # LINE BELOW THROWS DIMENSION MISMATCH ERROR
    # if output dimension is not invariant and not specified in the config and script
    hydragnn.utils.input_config_parsing.save_config(config, log_name)

    timer.stop()

    # ----------------------------------------------------------------------------------------------------
    # train
    log("[START] Declaring new HydraGNN model", rank=0)
    precision = args.precision.lower() if args.precision is not None else "fp32"
    config["NeuralNetwork"]["Training"]["precision"] = precision

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    model, optimizer = hydragnn.utils.distributed.distributed_model_wrapper(
        model, optimizer, verbosity
    )

    # Print details of neural network architecture
    log("[DONE] New HydraGNN model loaded", rank=0)
    print_model(model)

    hydragnn.utils.model.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    log("[START] Training HydraGNN model", rank=0)
    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        plot_init_solution=False,
        plot_hist_solution=False,
        skip_prediction_plots=True,
        create_plots=True,
        compute_grad_energy=config["NeuralNetwork"]["Architecture"].get(
            "enable_interatomic_potential", False
        ),
        precision=precision,
    )
    log("[DONE] HydraGNN model training complete", rank=0)

    hydragnn.utils.model.save_model(model, optimizer, log_name)
    hydragnn.utils.profiling_and_tracing.print_timers(verbosity)
    if writer is not None:
        writer.close()

    if tr.has("GPTLTracer"):
        import gptl4py as gp

        eligible = rank if args.everyone else 0
        if rank == eligible:
            gp.pr_file(os.path.join("logs", log_name, "gp_timing.p%d" % rank))
        gp.pr_summary_file(os.path.join("logs", log_name, "gp_timing.summary"))
        gp.finalize()

    dist.destroy_process_group()
    sys.exit(0)

    # ----------------------------------------------------------------------------------------------------
