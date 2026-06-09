import os, json
import logging
import sys
from mpi4py import MPI
import argparse
import time

import numpy as np
from tqdm import tqdm

import random
import torch
import torch.distributed as dist

# FIX random seed
random_state = 0
torch.manual_seed(random_state)

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

        self.download_config = download_config or {}
        self.data_dir = download_config.get('data_dir', 'dataset')
        self.wfns_dir = os.path.join(self.data_dir, download_config.get('wfns_subdir', 'wfns'))

        self._load_molecules()
    

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

    
    # THIS MAY NEED TO BE FURTHER OPTIMIZED
    # use tarball_assignments.csv to construct paths to wfns instead of walking thru the entire dataset
    def _load_molecules(self, use_tarball_assignments=True):
        '''walks through locally downloaded QMugs molecules and maps CHEMBL ids to wfn paths'''
        molecules = {}
        
        # map molecules (CHEMBL ids) to the paths of its conformers
        with tqdm(total=NUM_CONFORMERS, desc='Walking through molecules') as pbar:
            for root, _, files in os.walk(self.wfns_dir):
                for filename in files:
                    wfn_path = os.path.join(root, filename)

                    chembl_id = wfn_path.split('/')[-2]
                    if chembl_id not in molecules:
                        molecules[chembl_id] = []
                    molecules[chembl_id].append(wfn_path)
                    pbar.update(1)
        
        print(f'Walked through {pbar.n} total conformers')
        self.molecules_dict = molecules
    

    def load_dataset(self, sampling_ratio: float = 1.0):
        '''samples a subset of all molecules and loads them as Data objects'''
        molecules = self.molecules_dict.copy()
        chembl_ids = list(molecules.keys())
        num_molecules_to_load = max(1, int(len(chembl_ids) * sampling_ratio))

        random.seed(random_state)
        sampled_molecule_ids = random.sample(chembl_ids, num_molecules_to_load)
        
        # load all conformers for each sampled molecule
        for chembl_id in sampled_molecule_ids:
            for wfn_path in sorted(molecules[chembl_id]):
                data_obj = self._mol_to_graph(wfn_path=wfn_path)
                self.dataset.append(data_obj)

        
    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":

    with open('qmugs_densmat.json', 'r') as f:
        config = json.load(f)

    with open('utils/download_data.json', 'r') as f:
        download_config = json.load(f)
    
    dataset_object = QMugsDataset(
        config=config,
        download_config=download_config,
    )

    start = time.perf_counter()
    dataset_object.load_dataset(sampling_ratio=0.0001)
    end = time.perf_counter()

    print(f'Loaded {dataset_object.len()} conformers')
    print(f'Time for dataset sampling + loading (without accounting for object initialization): {end-start}')