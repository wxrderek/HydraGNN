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
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.distributed import nsplit
from hydragnn.utils.print.print_utils import iterate_tqdm
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
)

from hydragnn.utils.datasets.pickledataset import (
    SimplePickleWriter,
    SimplePickleDataset,
)

try:
    from hydragnn.utils.datasets.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

from rdkit import Chem

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


class QMugsDataset(AbstractBaseDataset):
    '''QMugs dataset class'''

    def __init__(
        self,
        # dirpath, # info for this is contained in download_config
        config,
        download_config,
        graphgps_transform=None,
        energy_per_atom=False,
        dist=False,
        normalize=False, # NOT IMPLEMENTED YET
    ):
        super().__init__()

        self.config = config
        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]
        self.energy_per_atom = energy_per_atom
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

        # get list of chembl_id's of all molecules via tarball_assignment.csv
        tarball_assignment = pd.read_csv(os.path.join(self.data_dir, 'tarball_assignment.csv'))
        self.chembl_ids = tarball_assignment['chembl_id'].tolist()

        # get list of dirs where wfns for molecules are stored
        self.molecule_wfn_dirs = tarball_assignment.apply(
            lambda row: os.path.join(
                self.wfns_dir,
                row['archive_name'].split('.')[0],
                row['chembl_id'],
            ), axis=1
        ).tolist()



    def _mol_to_graph(self, wfn_path):
        '''convert wfn .npy file to torch_geometric.data.Data object and return'''

        # track CHEMBL id
        chembl_id = wfn_path.split('/')[-2]

        wfn = psi4.core.Wavefunction.from_file(wfn_path)
        mol = wfn.molecule().clone() # for now, working with a clone of the psi4.core.Molecule object
                
        # ----------------------------------------------------------------------------------------------------
        # core data input
        natoms = torch.IntTensor(mol.natom())
        pos = torch.from_numpy(
            np.array(mol.geometry())
        ).to(torch.float32)
        assert(len(pos) == mol.natom())

        cell = torch.eye(3, dtype=torch.float32) # no period boundary conditions
        pbc = torch.tensor([False, False, False], dtype=torch.bool) # no period boundary conditions

        atomic_numbers = torch.as_tensor(
            [int(mol.ftrue_atomic_number(i)) for i in range(mol.natom())],
            dtype=torch.float32
        ).unsqueeze(1)

        hist, _ = np.histogram(atomic_numbers.tolist(), bins=range(1, 118 + 2))
        chemical_composition = torch.tensor(hist).unsqueeze(1).to(torch.float32)

        # ----------------------------------------------------------------------------------------------------
        # density learning specific data
        x = torch.cat([atomic_numbers, pos], dim=1) # geometric predictor only
        Da = torch.from_numpy(wfn.Da().np).to(torch.float32)
        Db = torch.from_numpy(wfn.Db().np).to(torch.float32)
        D_tot = torch.add(Da, Db)

        density = None # not computing electron density as scalar field yet

        # ----------------------------------------------------------------------------------------------------
        # other attributes
        graph_attr = None # not adding charge / spin for now, both are available in QMugss summary.csv file
                
        # ----------------------------------------------------------------------------------------------------

        # declare data object
        data_object = Data(
            dataset_name='qmugs',
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
            density=density,
            chembl_id=chembl_id,
            graph_attr=graph_attr,
        )
        data_object.y = D_tot

        data_object = self.radius_graph(data_object)
        data_object = transform_coordinates(data_object)

        data_object.edge_shifts = torch.zeros(
            (data_object.edge_index.size(1), 3), dtype=torch.float32
        )

        if self.graphgps_transform is not None:
            data_object = self.graphgps_transform(data_object)

        return data_object


    def _train_val_test_dirs(self, perc_load: float = 1.0, perc_train: float = 0.8):
        '''return list of directory paths of molecules to load'''

        # sample subset of chembl_id's of molecules to load for task
        all_molecule_dirs = self.molecule_wfn_dirs.copy()
        num_molecules_to_load = max(1, int(len(all_molecule_dirs) * perc_load))
        molecule_dirs_to_load = random.sample(all_molecule_dirs, num_molecules_to_load)

        perc_val = (1 - perc_train) / 2
        num_train = int(num_molecules_to_load * perc_train)
        num_val = int(num_molecules_to_load * perc_val)

        # get chembl_id's for train, val, test sets
        trainset_dirs = molecule_dirs_to_load[: num_train]
        valset_dirs = molecule_dirs_to_load[num_train : num_train + num_val]
        testset_dirs = molecule_dirs_to_load[num_train + num_val :]

        return trainset_dirs, valset_dirs, testset_dirs


    def load_and_split_dataset(self, 
        perc_load: float = 1.0, 
        perc_train: float = 0.8,
    ):
        '''load and split data, with splitting done per molecule, not per conformer'''

        # get chembl_id's for train, val, test sets
        trainset_dirs, valset_dirs, testset_dirs = self._train_val_test_dirs(
            perc_load=perc_load,
            perc_train=perc_train
        )

        # helper loader
        def _load_molecules_from_dir_list(dir_list, res_list, pbar):
            for dir in dir_list:
                for _, _, files in os.walk(dir):
                    for filename in files:
                        wfn_path = os.path.join(dir, filename)
                        data_object = self._mol_to_graph(wfn_path=wfn_path)

                        # append to trainset and self.dataset
                        res_list.append(data_object)
                        self.dataset.append(data_object)
                pbar.update(1)

        # load trainset
        trainset = []
        with tqdm(total=len(trainset_dirs), desc='Loading train set molecules') as pbar:
            _load_molecules_from_dir_list(trainset_dirs, trainset, pbar)
        
        # load valset
        valset = []
        with tqdm(total=len(valset_dirs), desc='Loading validation set molecules') as pbar:
            _load_molecules_from_dir_list(valset_dirs, valset, pbar)
        
        # load testset
        testset = []
        with tqdm(total=len(testset_dirs), desc='Loading test set molecules') as pbar:
            _load_molecules_from_dir_list(testset_dirs, testset, pbar)

        return trainset, valset, testset
        
        
    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


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
    parser.add_argument("--shmem", action="store_true", help="shmem")
    parser.add_argument("--log", help="log name")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=None)
    parser.add_argument("--everyone", action="store_true", help="gptimer")
    parser.add_argument("--modelname", help="model name")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp64", "bf16"],
        default=None,
        help="Override precision; defaults to fp32 when not set",
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
    # set up dataset object
    with open(args.inputfile, 'r') as f:
        config = json.load(f)
    with open('utils/download_data.json', 'r') as f:
        download_config = json.load(f)
    
    dataset_object = QMugsDataset(
        config=config,
        download_config=download_config,
    )

    # data loading and splitting
    start = time.perf_counter()
    trainset, valset, testset = dataset_object.load_and_split_dataset(
        perc_load=0.01, 
        perc_train=0.8,
    )
    end = time.perf_counter()

    print(f'Loaded {dataset_object.len()} conformers')
    print(f'trainset {len(trainset)}, valset {len(valset)}, testset {len(testset)}')
    print(f'Time for dataset loading + splitting (without accounting for object initialization): {end-start}')