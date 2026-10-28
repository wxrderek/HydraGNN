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
from hydragnn.decoders import SchurDecoderModel
from hydragnn.utils.model import print_model
from hydragnn.utils.pooling import graph_pooled_dim
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
from utils import density_matrix_variables, DensityMatrixDataset

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
    parser.add_argument(
        "--store_overlap_matrix",
        action="store_true",
        help="also store the overlap matrix and the number of electrons of each conformer during preprocessing",
    )

    # Schur decoder overrides. Everything else about the decoder is configured in the JSON;
    # these three are exposed because sweeping them is the point of the first runs, and
    # editing the JSON per job invites the two from getting out of step.
    parser.add_argument(
        "--schur_routing",
        type=str,
        choices=["teacher_forcing", "free_running"],
        help="override SchurDecoder.routing from the config file",
        default=None,
    )
    parser.add_argument(
        "--schur_supervision",
        type=str,
        choices=["joint", "semi_sequential", "per_step"],
        help="override SchurDecoder.supervision from the config file",
        default=None,
    )
    parser.add_argument(
        "--schur_target",
        type=str,
        choices=["total", "delta"],
        help="override SchurDecoder.target from the config file",
        default=None,
    )
    parser.add_argument(
        "--learning_rate_override",
        type=float,
        help="learning rate to use after loading a checkpoint, in place of the one the checkpoint restores",
        default=None,
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

    # resolved here rather than at model construction because it changes how the output head is
    # sized below: the sequential Schur decoder never calls the Base head
    schur_config = config["NeuralNetwork"].get("SchurDecoder", None)
    use_schur_decoder = schur_config is not None and schur_config.get("enable", False)

    # command line wins over the config file, and the override is written back so the saved
    # config records what actually ran
    if use_schur_decoder:
        for name in ("routing", "supervision", "target"):
            override = getattr(args, f"schur_{name}")
            if override is not None:
                schur_config[name] = override
    elif any(
        getattr(args, f"schur_{name}") is not None
        for name in ("routing", "supervision", "target")
    ):
        raise ValueError(
            "--schur_* options were given but the config has no enabled SchurDecoder block"
        )

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

    # The Schur decoder builds the matrix itself and never calls the Base output head, so the
    # head keeps the stub width set in the config. Widening it to T(N_pad) here would rebuild
    # exactly the multi-billion-parameter head the decoder exists to replace.
    if args.update_max_padded_dimension and not use_schur_decoder:
        var_config["output_dim"] = [padding_dim] * len(var_config["output_names"])

    if density_matrix_delta and args.preonly and args.artifacts_dir is None:
        raise ValueError("--artifacts_dir is required for delta density matrix preprocessing")
    if density_matrix_delta and args.format != "pickle":
        raise NotImplementedError("delta density matrix learning is implemented only for pickle datasets")
    
    # HydraGNN's named data schema: one graph-level head of width padding_dim per output name.
    # Built here rather than in the JSON because padding_dim follows N_pad, which
    # --update_max_padded_dimension can override; written to the config so the saved config
    # records it. Variables_of_interest stays as the example's own task-variant settings.
    config["Variables"] = density_matrix_variables(
        output_names=var_config["output_names"],
        padding_dim=padding_dim,
    )

    # mask config
    mask_output_loss = var_config.get("mask_output_loss", None)

    # ensure consistent padded matrix dimensions. The Schur decoder is exempt for the same
    # reason it is exempt from the widening above: it never calls the Base output head, so its
    # output_dim is a deliberate stub rather than the width of a padded target.
    if not use_schur_decoder:
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

    # placed after setup_log so the lines below reach run.log
    # a dense head output is the delta against the promolecular matrix exactly for delta output names
    loss_config = config["NeuralNetwork"]["Training"].get("loss") or {}
    if loss_config.get("enabled", False) and loss_config.get("provider") == "composite":
        loss_config.setdefault("density_matrix", {})
        loss_config["density_matrix"]["dense_target"] = "delta" if density_matrix_delta else "total"
        log(f"[INFO] composite loss: density_matrix.dense_target={loss_config['density_matrix']['dense_target']}, "
            f"from output_names={var_config['output_names']}", rank=0)
        active_terms = {term["name"] for term in loss_config.get("terms", []) if float(term.get("weight", 0.0)) != 0.0}
    else:
        active_terms = set()
    # trace normalization alone leaves the raw scale free and diverged in job 59258340
    dense_trace_normalization = bool(loss_config.get("density_matrix", {}).get("trace_normalization", False))
    if dense_trace_normalization and not (loss_config.get("enabled", False) and loss_config.get("provider") == "composite"):
        raise ValueError("density_matrix.trace_normalization needs the composite loss provider enabled")
    if use_schur_decoder and dense_trace_normalization:
        raise ValueError("density_matrix.trace_normalization is for the dense head; set SchurDecoder.trace_normalization instead")
    schur_trace_normalization = use_schur_decoder and schur_config.get("trace_normalization", False)
    if (schur_trace_normalization or dense_trace_normalization) and "electron_count" not in active_terms:
        log("[WARNING] trace_normalization is on without an active electron_count composite loss term; "
            "the raw scale of the prediction is then unconstrained and training diverged in job 59258340", rank=0)

    
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
            store_overlap_matrix=args.store_overlap_matrix,
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
                # atom count maximum over all splits, for concat pooling padding
                max_natoms = max(
                    int(data_object.num_nodes)
                    for split in (trainset, valset, testset)
                    for data_object in split
                )
                # orbital metadata one-hot widths and species list for the Schur decoder, taken
                # from the data rather than assumed. The widths are the label maxima plus one
                # because the labels are 0-based.
                all_objects = [
                    data_object
                    for split in (trainset, valset, testset)
                    for data_object in split
                ]
                n_l = 1 + max(
                    int(data_object.orbital_angular_momentum.max()) for data_object in all_objects
                )
                n_m = 1 + max(
                    int(data_object.orbital_within_shell.max()) for data_object in all_objects
                )
                n_shell = 1 + max(
                    int(data_object.orbital_shell.max()) for data_object in all_objects
                )
                species = sorted(
                    {
                        int(number)
                        for data_object in all_objects
                        for number in data_object.atomic_numbers.reshape(-1).tolist()
                    }
                )
                attrs = {
                    "pna_deg": deg,
                    "max_natoms": max_natoms,
                    "n_l": n_l,
                    "n_m": n_m,
                    "n_shell": n_shell,
                    "species": species,
                }
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

    # only the pickle path records the atom count maximum used by concat pooling
    max_natoms = None
    # orbital metadata widths and species list recorded by preprocessing, for the Schur decoder
    orbital_metadata_attrs = {}

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
        trainset = AdiosDataset(fname, "trainset", comm, **opt)
        valset = AdiosDataset(fname, "valset", comm, **opt)
        testset = AdiosDataset(fname, "testset", comm, **opt)
    
    # pickle
    elif args.format == "pickle":
        log("[START] Pickle data load", rank=0)
        basedir = os.path.join(dataset_dir, "%s.pickle" % modelname)
        trainset = SimplePickleDataset(basedir=basedir, label="trainset")
        valset = SimplePickleDataset(basedir=basedir, label="valset")
        testset = SimplePickleDataset(basedir=basedir, label="testset")
        pna_deg = trainset.pna_deg
        # older pickle datasets predate max_natoms, so it may be absent
        max_natoms = getattr(trainset, "max_natoms", None)
        # likewise for the orbital metadata, which predates nothing older than the Schur decoder
        orbital_metadata_attrs = {
            name: getattr(trainset, name)
            for name in ("n_l", "n_m", "n_shell", "species")
            if getattr(trainset, name, None) is not None
        }
        if args.ddstore:
            opt = {"ddstore_width": args.ddstore_width}
            trainset = DistDataset(trainset, "trainset", comm, **opt)
            valset = DistDataset(valset, "valset", comm, **opt)
            testset = DistDataset(testset, "testset", comm, **opt)
            trainset.pna_deg = pna_deg
            trainset.max_natoms = max_natoms
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    log(
        "[DONE] trainset, valset, testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset)), rank=0
    )

    if args.ddstore:
        os.environ["HYDRAGNN_AGGR_BACKEND"] = "mpi"
        os.environ["HYDRAGNN_USE_ddstore"] = "1"

    # stored samples -> model-ready samples (x = Z, y and y_loc per head, masks), see DensityMatrixDataset
    trainset = DensityMatrixDataset(trainset, config["Variables"])
    valset = DensityMatrixDataset(valset, config["Variables"])
    testset = DensityMatrixDataset(testset, config["Variables"])

    # get data loaders. variables= is not passed: DensityMatrixDataset has already applied the schema
    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.input_config_parsing.update_config(
        config, train_loader, val_loader, test_loader
    )

    # Shrink the Base output head back to its stub for the Schur decoder. This has to happen
    # AFTER update_config, not before: update_config sets output_dim from the Variables output
    # widths, which are padding_dim because the schema validates the stored target against
    # them. Left alone, the decoder would build the same T(N_pad)-wide head it exists to
    # replace -- about 5e8 frozen parameters that are never called.
    if use_schur_decoder:
        stub_dim = config["NeuralNetwork"]["Variables_of_interest"]["output_dim"]
        config["NeuralNetwork"]["Architecture"]["output_dim"] = list(stub_dim)
        log(
            f"[INFO] Schur decoder: Base output head pinned to output_dim={list(stub_dim)} "
            f"instead of the padded target width {padding_dim}; the head is never called",
            rank=0,
        )

    # ----------------------------------------------------------------------------------------------------
    # resolve pooling widths before the config is saved and the model is built

    architecture = config["NeuralNetwork"]["Architecture"]

    # graph_pooling may be a bare string or a dict {"type", "pool_config"}
    graph_pooling_cfg = architecture.get("graph_pooling", "mean")
    if isinstance(graph_pooling_cfg, dict):
        pool_type = graph_pooling_cfg.get("type", "mean").lower()
        pool_config = dict(graph_pooling_cfg.get("pool_config", {}))
    else:
        pool_type = graph_pooling_cfg.lower()
        pool_config = {}

    # hierarchical poolers concatenate node embeddings and need max_nodes and a wide enough first head layer
    legacy_reductions = {"mean", "add", "sum", "max"}
    if pool_type not in legacy_reductions:

        # max_nodes may be pinned in pool_config or top-level, else fall back to dataset metadata
        max_nodes = pool_config.get("max_nodes", architecture.get("max_nodes", None))
        if max_nodes is None:
            if max_natoms is None:
                raise ValueError(
                    f"graph_pooling='{pool_type}' requires 'max_nodes' in the pooling "
                    "'pool_config' or 'max_natoms' in the pickle metadata; re-run "
                    "preprocessing to record max_natoms"
                )
            max_nodes = int(max_natoms)
        max_nodes = int(max_nodes)

        # write the resolved max_nodes back so the saved config fully reproduces the model.
        # top-level max_nodes is what create.py threads to PAINN/EGNN
        pool_config["max_nodes"] = max_nodes
        architecture["max_nodes"] = max_nodes
        architecture["graph_pooling"] = {"type": pool_type, "pool_config": pool_config}

        # pooled width is exact once max_nodes is known; pooler kwargs exclude max_nodes
        pooler_kwargs = {k: v for k, v in pool_config.items() if k != "max_nodes"}
        pooled_dim = graph_pooled_dim(
            pool_type, architecture["hidden_dim"], max_nodes, **pooler_kwargs
        )
        log(
            f"[INFO] Pooling '{pool_type}': max_nodes={max_nodes}, "
            f"hidden_dim={architecture['hidden_dim']}, pooled_dim={pooled_dim}",
            rank=0,
        )

        # update_config has already normalized output_heads into the branch list form
        for branch in architecture["output_heads"]["graph"]:
            brancharct = branch["architecture"]
            num_sharedlayers = brancharct["num_sharedlayers"]

            # the layer consuming the pooled embedding is the trunk, or the head when there is none
            if num_sharedlayers == 0:
                target_key = "dim_headlayers"
                configured = brancharct["dim_headlayers"][0]
            else:
                target_key = "dim_sharedlayers"
                configured = brancharct["dim_sharedlayers"]

            # widen that layer to the pooled width so it does not contract the concatenation
            if configured < pooled_dim:
                if num_sharedlayers == 0:
                    brancharct["dim_headlayers"][0] = pooled_dim
                else:
                    brancharct["dim_sharedlayers"] = pooled_dim

                logging.warning(
                    f"Pooling '{pool_type}' override on branch '{branch['type']}': "
                    f"{target_key} raised from {configured} to {pooled_dim} because the "
                    f"configured width contracts the pooled embedding by a factor of "
                    f"{pooled_dim / configured:.1f}. This layer now holds "
                    f"{pooled_dim * pooled_dim:.3e} parameters."
                )
                if num_sharedlayers > 1:
                    logging.warning(
                        f"Branch '{branch['type']}' has num_sharedlayers={num_sharedlayers}; "
                        f"each layer beyond the first adds a further "
                        f"{pooled_dim} x {pooled_dim} linear layer."
                    )

    # ----------------------------------------------------------------------------------------------------
    # sequential Schur decoder: fill the widths that must come from the dataset

    if use_schur_decoder:
        missing = [
            name
            for name in ("n_l", "n_m", "n_shell", "species")
            if name not in orbital_metadata_attrs
        ]
        if missing:
            raise ValueError(
                f"The Schur decoder needs {missing} in the pickle metadata, which datasets "
                "preprocessed before the orbital metadata was added do not carry. Re-run "
                "preprocessing with --preonly to record them."
            )

        # written into the config so the saved config alone reproduces the identical model
        schur_config.update(orbital_metadata_attrs)
        log(
            "[INFO] Schur decoder orbital metadata widths from the dataset (label maxima plus "
            f"one, species sorted by atomic number): n_l={schur_config['n_l']}, "
            f"n_m={schur_config['n_m']}, n_shell={schur_config['n_shell']}, "
            f"species={schur_config['species']}",
            rank=0,
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

    # attach the sequential Schur decoder, which replaces the dense output head with the
    # bordering recursion of references/schur_decoder.tex
    if use_schur_decoder:
        # create_model_config sets the default dtype before returning, so the decoder's own
        # modules are already built at run precision; matching the backbone explicitly makes
        # that dependence visible instead of implicit
        backbone_dtype = next(model.parameters()).dtype
        backbone_device = next(model.parameters()).device
        decoder_kwargs = {k: v for k, v in schur_config.items() if k != "enable"}
        model = SchurDecoderModel(model, **decoder_kwargs).to(
            device=backbone_device, dtype=backbone_dtype
        )
        log(f"[INFO] Schur decoder attached: routing={schur_config.get('routing')}, "
            f"supervision={schur_config.get('supervision')}, "
            f"target={schur_config.get('target')}", rank=0)

    model, _ = hydragnn.utils.distributed.distributed_model_wrapper(
        model, None, verbosity
    )

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    # Print details of neural network architecture
    log("[DONE] New HydraGNN model loaded", rank=0)
    print_model(model)

    hydragnn.utils.model.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    # A resumed optimizer carries the learning rate saved in its checkpoint, so the config's
    # learning_rate has no effect on a resume. This is the way to change it at a resume, e.g.
    # after the LR 1e-3 blow-up of the Schur 400-epoch campaign (schur_long/README.md).
    if args.learning_rate_override is not None:
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate_override
        log(f"[INFO] learning rate overridden to {args.learning_rate_override} after loading", rank=0)

    log("[START] Training HydraGNN model", rank=0)
    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config,
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
