import os, json
import logging
import sys
from mpi4py import MPI
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

import random
import torch
import torch.distributed as dist

# ----------------------------------------------------------------------------------------------------
# FIX random seed
random_state = 0
torch.manual_seed(random_state)
random.seed(random_state)
# ----------------------------------------------------------------------------------------------------

from torch_geometric.data import Data
from torch_geometric.transforms import Distance

import hydragnn
from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.model import print_model
from hydragnn.utils.print.print_utils import iterate_tqdm, log
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

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

try:
    import psi4
    psi4.core.be_quiet()
except ImportError:
    print('[WARNING] psi4 package not detected')


# ----------------------------------------------------------------------------------------------------
transform_coordinates = Distance(norm=False, cat=False)
# ----------------------------------------------------------------------------------------------------

NUM_MOLECULES = 665911
NUM_CONFORMERS = 1992984

BOHR_TO_ANGSTROM = 0.52917721092

# ----------------------------------------------------------------------------------------------------
# util functions (might move)

def _build_xyz_grid(
    center = (0.0, 0.0, 0.0), # (x, y, z), Angstrom by default
    box_size = 30.0, # Angstrom by default
    spacing = 0.5, # Angstrom by default
    input_units: str = 'Angstrom', # 'Angstrom' or 'Bohr'
    output_units: str = 'Angstrom', # 'Angstrom' or 'Bohr'
):
    '''builds (N, 3) dim array of grid points to express scalar field quantities, in Angstrom units by default, covering a square box'''

    if input_units.lower() not in ['angstrom', 'bohr']:
        raise ValueError(f'Input units not recognized: {input_units}')
    if output_units.lower() not in ['angstrom', 'bohr']:
        raise ValueError(f'Output units not recognized: {output_units}')
    
    # change everything to Angstrom
    if input_units.lower() == 'bohr':
        center = center * BOHR_TO_ANGSTROM
        box_size = box_size * BOHR_TO_ANGSTROM
        spacing = spacing * BOHR_TO_ANGSTROM
    
    center = np.asarray(center)
    n = int(round(box_size / spacing))
    half = box_size / 2

    # ensure box_size is integer multiple of spacing
    if abs(n * spacing - box_size) > 1e-8:
        raise ValueError('box_size must be an integer multiple of spacing')

    x = np.linspace(
        center[0] - half + spacing / 2,
        center[0] + half - spacing / 2,
        n,
    )

    y = np.linspace(
        center[1] - half + spacing / 2,
        center[1] + half - spacing / 2,
        n,
    )

    z = np.linspace(
        center[2] - half + spacing / 2,
        center[2] + half - spacing / 2,
        n,
    )

    # build grid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid = np.column_stack(
        (X.ravel(), Y.ravel(), Z.ravel())
    )

    # convert to Bohr if specified
    if output_units.lower() == 'bohr':
        grid = grid * 1.0 / BOHR_TO_ANGSTROM

    return grid

# ----------------------------------------------------------------------------------------------------


class QMugsDataset(AbstractBaseDataset):
    '''QMugs dataset class'''

    # pad density matrix to ensure dimension of output is invariant across moledules
    max_padded_density_matrix_dimension = 2002

    def __init__(
        self,
        # dirpath, # info for this is contained in download_config
        config,
        download_config,
        graphgps_transform=None,
        energy_per_atom=False,
        dist=False,
        pad_density_matrix=False,
        normalize=False, # NOT IMPLEMENTED YET
    ):
        super().__init__()

        self.config = config
        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]
        self.energy_per_atom = energy_per_atom

        self.pad_density_matrix = pad_density_matrix
        self.normalize=normalize

        self.radius_graph = RadiusGraph(
            self.radius, loop=False, max_num_neighbors=self.max_neighbours
        )

        self.graphgps_transform = graphgps_transform
        
        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        # paths to data
        self.download_config = download_config or {}
        self.data_dir = download_config.get('data_dir', 'dataset')
        self.wfns_dir = os.path.join(self.data_dir, download_config.get('wfns_subdir', 'wfns'))

        # get chembl_id's and dirs of all molecules via tarball_assignment.csv
        tarball_assignment = pd.read_csv(os.path.join(self.data_dir, 'tarball_assignment.csv'))

        # store dir paths
        self.molecule_dirs = {}
        for _, row in tarball_assignment.iterrows():
            chembl_id = row['chembl_id']
            wfn_dir = os.path.join(
                self.wfns_dir,
                row['archive_name'].split('.')[0],
                chembl_id,
            )
            sdf_dir = os.path.join(
                self.data_dir, 
                'structures',
                chembl_id,
            )
            
            # check id match
            wfn_chembl_id = os.path.basename(wfn_dir)
            sdf_chembl_id = os.path.basename(sdf_dir)
            if wfn_chembl_id != chembl_id or sdf_chembl_id != chembl_id:
                raise ValueError(
                    f"chembl_id mismatch for {chembl_id}: "
                    f"wfn_dir ends with {wfn_chembl_id}, "
                    f"sdf_dir ends with {sdf_chembl_id}"
                )
            
            self.molecule_dirs[chembl_id] = {
                'wfn_dir': wfn_dir,
                'sdf_dir': sdf_dir,
            }


    # ----------------------------------------------------------------------------------------------------
    # pre and post processing of densities
    def _pad_density_matrix(self, density_matrix):
        '''pad density matrix with 0's to match maximal dimension in dataset'''
        N = self.__class__.max_padded_density_matrix_dimension
        n = density_matrix.shape[0]

        padded = np.zeros((N, N), dtype=density_matrix.dtype)
        padded[:n, :n] = density_matrix

        return padded


    def _get_density_matrix_mask(
        self, 
        wfn: psi4.core.Wavefunction,
        mask_method: str = "unif_atoms",
        **kwargs,
    ):
        '''outputs binary mask for density matrix'''
        return


    def _preprocess_scalar_density(
        self, 
        wfn: psi4.core.Wavefunction, 
        grid_xyz: np.ndarray, # (N, 3) dim array of grid points measured in Angstrom units
        spin: str = 'total', # 'total', 'alpha', 'beta', 'spin'
    ):
        '''for each conformer wfn, map density matrix -> scalar density'''

        Da = wfn.Da().np
        Db = wfn.Db().np
        basis = wfn.basisset()

        # prep matrix
        if spin == 'alpha':
            D = Da.copy()
        elif spin == 'beta':
            D = Db.copy()
        elif spin == 'total':
            D = Da.copy() + Db.copy()
        elif spin == 'spin':
            D = Da.copy() - Db.copy()
        else:
            raise ValueError(F'Unrecognized density type input: {spin}')
        
        # FINISH

    # ----------------------------------------------------------------------------------------------------
    # dataset prep

    def _mol_to_graph(self, wfn_path, sdf_path):
        '''convert molecule data to torch_geometric.data.Data object and return'''    

        # track chembl_id since each object is a conformer
        chembl_id = wfn_path.split('/')[-2]

        # load objects
        psi4_wfn = psi4.core.Wavefunction.from_file(wfn_path)
        psi4_mol = psi4_wfn.molecule().clone()
        sdf_mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]

        sdf_mol_props = sdf_mol.GetPropsAsDict()
                
        # ----------------------------------------------------------------------------------------------------
        # sdf_mol data (at xTB level unless specified)
        pos = torch.from_numpy(
            sdf_mol.GetConformer().GetPositions()
        ).to(torch.float32)
        energy_xTB = torch.tensor(sdf_mol_props['GFN2:TOTAL_ENERGY'], dtype=torch.float32)
        energy_DFT = torch.tensor(sdf_mol_props['DFT:TOTAL_ENERGY'], dtype=torch.float32)
        
        # ----------------------------------------------------------------------------------------------------
        # psi4_mol data (at DFT level unless specified)
        Da = psi4_wfn.Da().np
        Db = psi4_wfn.Db().np
        D_tot = Da + Db

        # check if density matrix is too big for padding
        if D_tot.shape[0] > self.__class__.max_padded_density_matrix_dimension:
            raise ValueError(f'A loaded density matrix has size {D_tot.shape}, the output dimension is {self.__class__.max_padded_density_matrix_dimension}')

        # pad density matrix
        if self.pad_density_matrix:
            D_tot = self._pad_density_matrix(D_tot.copy())

        D_tot = torch.from_numpy(D_tot.copy()).to(torch.float32)

        # save original density matrix dimension
        density_matrix_dim = torch.IntTensor(Da.shape[0])

        # ----------------------------------------------------------------------------------------------------
        # data available in sdf or psi4

        # natoms
        natoms_psi4 = int(psi4_mol.natom())
        natoms_sdf = sdf_mol.GetNumAtoms()
        assert(natoms_psi4 == natoms_sdf)
        assert(len(pos) == natoms_psi4)
        natoms = torch.IntTensor(natoms_psi4)

        # atomic_numbers
        atomic_numbers_psi4 = [int(psi4_mol.ftrue_atomic_number(i)) for i in range(psi4_mol.natom())]
        atomic_numbers_sdf = [atom.GetAtomicNum() for atom in sdf_mol.GetAtoms()]
        assert(atomic_numbers_psi4 == atomic_numbers_sdf)
        atomic_numbers = torch.as_tensor(atomic_numbers_psi4, dtype=torch.float32).unsqueeze(1)

        # chemical_composition
        hist, _ = np.histogram(atomic_numbers.tolist(), bins=range(1, 118 + 2))
        chemical_composition = torch.tensor(hist).unsqueeze(1).to(torch.float32)

        # ----------------------------------------------------------------------------------------------------
        # placeholder data
        cell = torch.eye(3, dtype=torch.float32) # no period boundary conditions
        pbc = torch.tensor([False, False, False], dtype=torch.bool) # no period boundary conditions

        # ----------------------------------------------------------------------------------------------------
        # other attributes
        graph_attr = None # not adding charge / spin for now, both are available in QMugss summary.csv file
                
        # ----------------------------------------------------------------------------------------------------
        # declare data object

        density_mask = self._get_density_matrix_mask(
            wfn=psi4_wfn,
            mask_method="unif_atoms",
            perc_atoms_masked = 0.2,
        )

        x = torch.cat([atomic_numbers, pos], dim=1) # atomic numbers + geometry only

        data_object = Data(
            dataset_name="qmugs",
            natoms=natoms,
            pos=pos,
            cell=cell, # not needed
            pbc=pbc, # not needed
            edge_index=None,
            edge_attr=None,
            atomic_numbers=atomic_numbers,
            chemical_composition=chemical_composition,
            smiles_string=None, # available in QMugs summary.csv file, which is currently not being parsed
            x=x,
            density_matrix=D_tot,
            # auxiliary inputs
            chembl_id=chembl_id,
            graph_attr=graph_attr,
        )
        data_object.y = data_object.density_matrix
        data_object.y_mask = density_mask

        data_object = self.radius_graph(data_object)
        data_object = transform_coordinates(data_object)

        data_object.edge_shifts = torch.zeros(
            (data_object.edge_index.size(1), 3), dtype=torch.float32
        )

        if self.graphgps_transform is not None:
            data_object = self.graphgps_transform(data_object)

        return data_object


    def _train_val_test_ids(self, perc_load: float = 1.0, perc_train: float = 0.8):
        '''return list of chembl_id's of molecules to load'''

        # sample subset of chembl_id's of molecules to load for task
        all_ids = list(self.molecule_dirs.keys()).copy()
        num_molecules_to_load = max(1, int(len(all_ids) * perc_load))
        molecules_to_load = random.sample(all_ids, num_molecules_to_load)

        perc_val = (1 - perc_train) / 2
        num_train = int(num_molecules_to_load * perc_train)
        num_val = int(num_molecules_to_load * perc_val)

        # get chembl_id's for train, val, test sets
        trainset_ids = molecules_to_load[: num_train]
        valset_ids = molecules_to_load[num_train : num_train + num_val]
        testset_ids = molecules_to_load[num_train + num_val :]

        return trainset_ids, valset_ids, testset_ids


    def load_and_split_dataset(self, 
        perc_load: float = 1.0, 
        perc_train: float = 0.8,
    ):
        '''load and split data, with splitting done per molecule, not per conformer'''

        # get chembl_id's for train, val, test sets
        trainset_ids, valset_ids, testset_ids = self._train_val_test_ids(
            perc_load=perc_load,
            perc_train=perc_train
        )

        # helper loader
        def _load_from_id_list(id_list, res_list, pbar):
            for id in id_list:
                wfn_dir = self.molecule_dirs[id]['wfn_dir']
                sdf_dir = self.molecule_dirs[id]['sdf_dir']
                for _, _, files in os.walk(wfn_dir):
                    for filename in files:
                        wfn_path = os.path.join(wfn_dir, filename)
                        
                        # match conformer id
                        conf_id = wfn_path.split('/')[-1].split('.')[0][-2:]
                        sdf_path = os.path.join(sdf_dir, f'conf_{conf_id}.sdf')

                        # create Data object
                        data_object = self._mol_to_graph(wfn_path=wfn_path, sdf_path=sdf_path)

                        # append to reslist and self.dataset
                        res_list.append(data_object)
                        self.dataset.append(data_object)
                pbar.update(1)


        # load trainset
        trainset = []
        with tqdm(total=len(trainset_ids), desc='Loading train set molecules') as pbar:
            _load_from_id_list(trainset_ids, trainset, pbar)
        
        # load valset
        valset = []
        with tqdm(total=len(valset_ids), desc='Loading validation set molecules') as pbar:
            _load_from_id_list(valset_ids, valset, pbar)
        
        # load testset
        testset = []
        with tqdm(total=len(testset_ids), desc='Loading test set molecules') as pbar:
            _load_from_id_list(testset_ids, testset, pbar)
        
        split_sizes = {
            'train_num_molecules': len(trainset_ids),
            'train_num_conformers': len(trainset),
            'val_num_molecules': len(valset_ids),
            'val_num_conformers': len(valset),
            'test_num_molecules': len(testset_ids),
            'test_num_conformers': len(testset),
            'total_num_molecules': len(trainset_ids) + len(valset_ids) + len(testset_ids),
            'total_num_conformers': self.len(),
        }

        return trainset, valset, testset, split_sizes
        
        
    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


# runs trianing
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
        "--precision",
        type=str,
        choices=["fp32", "fp64", "bf16"],
        default=None,
        help="Override precision; defaults to fp32 when not set",
    )

    parser.add_argument("--perc_load", type=float, help="percentage of all molecules in dataset to load", default=0.0001)
    parser.add_argument("--perc_train", type=float, help="percentage of loaded moledules to assign to train set", default=0.8)

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
    padding_dim = int((QMugsDataset.max_padded_density_matrix_dimension)**2)
    
    # set up features
    graph_feature_names = ['density_matrix']
    graph_feature_dims = [padding_dim]
    node_feature_names = ['atomic_number', 'cartesian_coordinates']
    node_feature_dims = [1, 3]
    
    # variables of interest configs
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dims
    var_config["node_feature_names"] = node_feature_names
    var_config["node_feature_dims"] = node_feature_dims

    # ensure consistent padded matrix dimensions
    assert (var_config["output_dim"][0] == padding_dim)

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
    log(f'Random seed used for run: {random_state}')

    # ----------------------------------------------------------------------------------------------------
    # load data and perform splitting
    modelname = "QMugs" if args.modelname is None else args.modelname
    if args.preonly:
        
        # initialize dataset object
        dataset_object = QMugsDataset(
            config=config,
            download_config=download_config,
            graphgps_transform=None,
            energy_per_atom=False,
            dist=True,
            pad_density_matrix=True,
        )

        # load and split locally downloaded and extracted data
        start = time.perf_counter()
        trainset, valset, testset, split_sizes = dataset_object.load_and_split_dataset(
            perc_load=args.perc_load, 
            perc_train=args.perc_train,
        )
        end = time.perf_counter()

        log(f'Split sizes: {json.dumps(split_sizes, indent=4)}')
        log(f'Time for dataset loading + splitting (without accounting for object initialization): {end-start}')

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
        log(f'Saved trainset, testset, valset to {basedir}\n')
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
        log("Adios load")
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
        log("Pickle load")
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
        "trainset, valset, testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
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
    print_model(model)

    hydragnn.utils.model.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

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
        create_plots=False,
        compute_grad_energy=config["NeuralNetwork"]["Architecture"].get(
            "enable_interatomic_potential", False
        ),
        precision=precision,
    )

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