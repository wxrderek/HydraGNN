import os, json
import logging
import time
import pickle
import gc

import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
from collections import OrderedDict

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

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Distance

import hydragnn
from hydragnn.train.train_validate_test import move_batch_to_device, resolve_precision, get_autocast_and_scaler
from hydragnn.utils.datasets.abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.model import print_model, loss_function_selection
from hydragnn.utils.distributed import get_device
from hydragnn.utils.print.print_utils import iterate_tqdm, log
from hydragnn.preprocess.graph_samples_checks_and_updates import (
    RadiusGraph,
    gather_deg,
)

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from utils.inference_utils import (
    assert_same_cube_grid,
    electron_density_from_density_matrix,
    read_cubeprops,
    write_cubeprops,
    wavefunction_from_total_density_matrix,
    dipole_moment_from_density_matrix,
)

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
DIPOLE_AU_TO_DEBYE = 2.541746473


# ----------------------------------------------------------------------------------------------------
# QMugs dataset class

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
        pad_density_matrix=True,
        mask_output_loss=None,
        predict_alpha_beta=False,
        normalize=False, # NOT IMPLEMENTED YET
    ):
        super().__init__()

        self.config = config
        self.radius = config["NeuralNetwork"]["Architecture"]["radius"]
        self.max_neighbours = config["NeuralNetwork"]["Architecture"]["max_neighbours"]
        self.energy_per_atom = energy_per_atom

        self.pad_density_matrix = pad_density_matrix
        self.predict_alpha_beta = predict_alpha_beta
        self.normalize=normalize

        # parse masking configuration as specified
        if mask_output_loss is not None:
            self.mask = mask_output_loss.get('mask', False)
            self.mask_method = mask_output_loss.get('mask_method', 'unif_atoms')
            self.mask_config = mask_output_loss.get('mask_config', {})
        else:
            self.mask = False

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

        from utils import pad_density_matrix
        N = self.__class__.max_padded_density_matrix_dimension
        padded = pad_density_matrix(density_matrix=density_matrix, max_size=N)

        return padded


    def _get_density_matrix_mask(
        self, 
        rng,
        wfn: psi4.core.Wavefunction,
        mask_method: str = "unif_atoms",
        **kwargs,
    ):
        '''returns binary mask for density matrix'''

        from utils import get_density_matrix_mask
        mask, _ = get_density_matrix_mask(
            wfn=wfn,
            rng=rng,
            random_state=random_state,
            mask_method=mask_method,
            kwargs=kwargs,
        )

        return mask

    # ----------------------------------------------------------------------------------------------------
    # dataset prep

    def _mol_to_graph(self, wfn_path, sdf_path):
        '''convert molecule data to torch_geometric.data.Data object and return'''    

        # track chembl_id and conformer_id since each object is a conformer
        chembl_id = wfn_path.split('/')[-2]
        conformer_id = wfn_path.split('/')[-1].split('.')[0][-2:]

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

        # save original density matrix dimension
        density_matrix_dim = torch.tensor(int(Da.shape[0]), dtype=torch.int32)
        density_matrix_dim_int = int(density_matrix_dim.item())

        # check if density matrix is too big for padding
        if D_tot.shape[0] > self.__class__.max_padded_density_matrix_dimension:
            raise ValueError(f'A loaded density matrix has size {D_tot.shape}, the output dimension is {self.__class__.max_padded_density_matrix_dimension}')

        # pad density matrix
        if self.pad_density_matrix:
            if self.predict_alpha_beta:
                Da = self._pad_density_matrix(Da.copy())
                Db = self._pad_density_matrix(Db.copy())

                Da = torch.from_numpy(Da).to(torch.float32)
                Db = torch.from_numpy(Db).to(torch.float32)
            else: 
                D_tot = self._pad_density_matrix(D_tot.copy())
                D_tot = torch.from_numpy(D_tot).to(torch.float32)
        else:
            if self.predict_alpha_beta:
                Da = torch.from_numpy(Da.copy()).to(torch.float32)
                Db = torch.from_numpy(Db.copy()).to(torch.float32)
            else:
                D_tot = torch.from_numpy(D_tot.copy()).to(torch.float32)

        # charge and multiplicity
        charge = torch.tensor(
            int(psi4_mol.molecular_charge()),
            dtype=torch.int32,
        )

        multiplicity = torch.tensor(
            int(psi4_mol.multiplicity()),
            dtype=torch.int32,
        )

        # ----------------------------------------------------------------------------------------------------
        # data available in sdf or psi4

        # natoms
        natoms_psi4 = int(psi4_mol.natom())
        natoms_sdf = sdf_mol.GetNumAtoms()
        assert(natoms_psi4 == natoms_sdf)
        assert(len(pos) == natoms_psi4)
        natoms = torch.IntTensor(natoms_psi4)
        natoms_int = torch.tensor(
            int(natoms_psi4),
            dtype=torch.int32,
        )

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
        # masking

        if self.predict_alpha_beta:
            # mask alpha and beta independently
            if self.mask:
                alpha_density_mask = self._get_density_matrix_mask(
                    wfn=psi4_wfn,
                    rng=rng,
                    mask_method=self.mask_method,
                    perc_entries_masked = self.mask_config.get('perc_entries_masked', 0.2),
                    perc_atom_pairs_masked = self.mask_config.get('perc_atom_pairs_masked', 0.2),
                )
                beta_density_mask = self._get_density_matrix_mask(
                    wfn=psi4_wfn,
                    rng=rng,
                    mask_method=self.mask_method,
                    perc_entries_masked = self.mask_config.get('perc_entries_masked', 0.2),
                    perc_atom_pairs_masked = self.mask_config.get('perc_atom_pairs_masked', 0.2),
                )
            # mask out the padded region only
            else:
                alpha_density_mask = np.ones((density_matrix_dim_int, density_matrix_dim_int), dtype=bool)
                beta_density_mask = np.ones((density_matrix_dim_int, density_matrix_dim_int), dtype=bool)

            # store compact masks with shape (n_basis, n_basis), not padded to (2002, 2002)
            assert alpha_density_mask.shape == (density_matrix_dim_int, density_matrix_dim_int)
            assert beta_density_mask.shape == (density_matrix_dim_int, density_matrix_dim_int)
            alpha_density_mask = torch.from_numpy(alpha_density_mask).to(torch.bool)
            beta_density_mask = torch.from_numpy(beta_density_mask).to(torch.bool)
        
        else:
            # mask total density
            if self.mask:
                density_mask = self._get_density_matrix_mask(
                    wfn=psi4_wfn,
                    rng=rng,
                    mask_method=self.mask_method,
                    perc_entries_masked=self.mask_config.get('perc_entries_masked', 0.2),
                    perc_atom_pairs_masked=self.mask_config.get('perc_atom_pairs_masked', 0.2),
                )
            # mask out the padded region only
            else: 
                density_mask = np.ones((density_matrix_dim_int, density_matrix_dim_int), dtype=bool)

            # store compact mask with shape (n_basis, n_basis), not padded to (2002, 2002)
            assert density_mask.shape == (density_matrix_dim_int, density_matrix_dim_int)
            density_mask = torch.from_numpy(density_mask).to(torch.bool)
        

        # ----------------------------------------------------------------------------------------------------
        # declare data object

        # predictor: atom species and positions only
        x = torch.cat([atomic_numbers, pos], dim=1)
        
        if self.predict_alpha_beta:
            density_target = torch.cat((Da.flatten(), Db.flatten()))
            density_target_mask = torch.cat(
                (alpha_density_mask.flatten(), beta_density_mask.flatten())
            )

            # predict alpha and beta matrices jointly as separate vectors
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
                y=density_target,
                y_mask=density_target_mask,
                # auxiliary inputs
                density_matrix_dim=density_matrix_dim,
                chembl_id=chembl_id,
                conformer_id=conformer_id,
                wfn_path=wfn_path,
                natoms_int=natoms_int,
                charge=charge,
                multiplicity=multiplicity,
                graph_attr=graph_attr,
            )
            # y_mask_loc records compact per-head mask offsets: (1, 3) for alpha and beta heads
            data_object.y_mask_loc = torch.tensor(
                [[0, density_matrix_dim_int ** 2, 2 * density_matrix_dim_int ** 2]],
                dtype=torch.int64,
            )
            
        else:
            # predict the full density matrix only
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
                y=D_tot, # y has shape (2002, 2002) when padding is enabled
                y_mask=density_mask, # y_mask has compact shape (n_basis, n_basis), not padded
                # auxiliary inputs
                density_matrix_dim=density_matrix_dim,
                chembl_id=chembl_id,
                conformer_id=conformer_id,
                wfn_path=wfn_path,
                natoms_int=natoms_int,
                charge=charge,
                multiplicity=multiplicity,
                graph_attr=graph_attr,
            )
            
            # y_mask_loc records compact mask offsets: (1, 2) for one density-matrix head
            data_object.y_mask_loc = torch.tensor(
                [[0, density_matrix_dim_int ** 2]],
                dtype=torch.int64,
            )

        # apply graph transforms
        data_object = self.radius_graph(data_object)
        data_object = transform_coordinates(data_object)
        data_object.edge_shifts = torch.zeros(
            (data_object.edge_index.size(1), 3), dtype=torch.float32
        )
        
        if self.graphgps_transform is not None:
            data_object = self.graphgps_transform(data_object)

        return data_object


    def _density_matrix_sizes_from_eda(self):
        '''load density matrix sizes from the EDA output file'''

        eda_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'eda',
            'densmat_eda.json',
        )
        if not os.path.isfile(eda_path):
            raise FileNotFoundError(
                f'Density matrix EDA file not found: {eda_path}'
            )

        with open(eda_path, 'r', encoding='utf-8') as f:
            densmat_eda = json.load(f)

        density_matrix_sizes = densmat_eda.get('sizes', None)
        if not isinstance(density_matrix_sizes, dict):
            raise ValueError(
                f"Density matrix EDA file {eda_path} must contain a dictionary under key 'sizes'"
            )

        return density_matrix_sizes


    def _filter_ids_by_density_matrix_size(self, all_ids, max_density_matrix_size: int):
        '''return molecule IDs whose density matrix size is at most the requested size'''

        if max_density_matrix_size <= 0:
            raise ValueError('max_density_matrix_size must be a positive integer')

        density_matrix_sizes = self._density_matrix_sizes_from_eda()

        filtered_ids = []
        for chembl_id in all_ids:
            if chembl_id not in density_matrix_sizes:
                log(
                    f"[WARNING] Missing density matrix size for molecule {chembl_id}; "
                    "skipping this molecule during size-aware sampling",
                    rank=0,
                )
                continue

            try:
                density_matrix_size = int(density_matrix_sizes[chembl_id])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'Density matrix size for molecule {chembl_id} must be an integer'
                ) from exc

            if density_matrix_size <= max_density_matrix_size:
                filtered_ids.append(chembl_id)

        if len(filtered_ids) == 0:
            raise ValueError(
                f'No molecules have density matrix size <= {max_density_matrix_size}'
            )

        return filtered_ids


    def _update_max_padded_density_matrix_dimension(self, max_density_matrix_size: int):
        '''update the class-level padded density matrix dimension'''

        if max_density_matrix_size <= 0:
            raise ValueError('max_density_matrix_size must be a positive integer')

        self.__class__.max_padded_density_matrix_dimension = max_density_matrix_size


    def _train_val_test_ids(
        self,
        perc_load: float = 1.0,
        perc_train: float = 0.8,
        max_density_matrix_size: int = None,
    ):
        '''return list of chembl_id's of molecules to load'''

        # sample subset of chembl_id's of molecules to load for task
        all_ids = list(self.molecule_dirs.keys()).copy()
        if max_density_matrix_size is not None:
            all_ids = self._filter_ids_by_density_matrix_size(
                all_ids=all_ids,
                max_density_matrix_size=max_density_matrix_size,
            )

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
        max_density_matrix_size: int = None,
        update_max_padded_dimension: bool = False,
    ):
        '''load and split data, with splitting done per molecule, not per conformer'''

        if update_max_padded_dimension and max_density_matrix_size is not None:
            self._update_max_padded_density_matrix_dimension(max_density_matrix_size=max_density_matrix_size)

        # get chembl_id's for train, val, test sets
        trainset_ids, valset_ids, testset_ids = self._train_val_test_ids(
            perc_load=perc_load,
            perc_train=perc_train,
            max_density_matrix_size=max_density_matrix_size,
        )

        # distributed loading
        if self.dist:
            trainset_ids = trainset_ids[self.rank :: self.world_size]
            valset_ids = valset_ids[self.rank :: self.world_size]
            testset_ids = testset_ids[self.rank :: self.world_size]

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


    def load_and_split_dataset_pickle_streaming(
        self,
        basedir,
        comm,
        perc_load: float = 1.0,
        perc_train: float = 0.8,
        max_density_matrix_size: int = None,
        update_max_padded_dimension: bool = False,
        use_subdir: bool = True,
        nmax_persubdir: int = 10_000,
        compute_pna_deg: bool = True,
        verbosity: int = 2,
        log_every: int = 100,
    ):
        '''load, split, and stream QMugs dataset directly to pickle files'''

        rank = comm.Get_rank()
        world_size = comm.Get_size()

        if update_max_padded_dimension and max_density_matrix_size is not None:
            self._update_max_padded_density_matrix_dimension(
                max_density_matrix_size=max_density_matrix_size,
            )

        # log config
        log("[START] Streaming QMugs pickle preprocessing", rank=0)
        log(f"[INFO] Output pickle directory: {basedir}", rank=0)
        log(
            f"[INFO] perc_load={perc_load}, perc_train={perc_train}, "
            f"max_density_matrix_size={max_density_matrix_size}, world_size={world_size}",
            rank=0,
        )

        # generate train/val/test split indices for all molecules
        trainset_ids, valset_ids, testset_ids = self._train_val_test_ids(
            perc_load=perc_load,
            perc_train=perc_train,
            max_density_matrix_size=max_density_matrix_size,
        )

        # shard indices across MPI ranks
        if self.dist:
            trainset_ids = trainset_ids[rank::world_size]
            valset_ids = valset_ids[rank::world_size]
            testset_ids = testset_ids[rank::world_size]
        
        log(
            f"[INFO] Rank {rank}: assigned molecules "
            f"train={len(trainset_ids)}, val={len(valset_ids)}, test={len(testset_ids)}"
        )

        def _count_conformers(id_list):
            '''count conformers contained in a list of molecule IDs'''
            n = 0
            for chembl_id in id_list:
                wfn_dir = self.molecule_dirs[chembl_id]["wfn_dir"]
                n += len([
                    f for f in os.listdir(wfn_dir)
                    if os.path.isfile(os.path.join(wfn_dir, f))
                ])
            return n

        def _update_deg_hist(deg_hist, data_object):
            '''update running pna degree histogram'''
            deg = torch.bincount(
                data_object.edge_index[1],
                minlength=data_object.num_nodes,
            ).cpu()
            local_hist = torch.bincount(deg)

            if deg_hist.numel() < local_hist.numel():
                tmp = torch.zeros(local_hist.numel(), dtype=torch.long)
                tmp[:deg_hist.numel()] = deg_hist
                deg_hist = tmp

            deg_hist[:local_hist.numel()] += local_hist
            return deg_hist

        def _write_split(label, id_list, attrs=None):
            '''stream a single dataset split to pickle files'''

            # determine local and global graph counts
            local_n = _count_conformers(id_list)
            ns = comm.allgather(local_n)
            noffset = sum(ns[:rank])
            ntotal = sum(ns)

            log(
                f"[START] Streaming split={label}: "
                f"global_graphs={ntotal}, local_graphs_rank0={ns[0] if len(ns) > 0 else 0}", 
                rank=0,
            )

            log(f"[INFO] Rank {rank}: split={label}, local_graphs={local_n}, global_offset={noffset}")

            # write metadata file once
            if rank == 0:
                os.makedirs(basedir, exist_ok=True)
                with open(os.path.join(basedir, f"{label}-meta.pkl"), "wb") as f:
                    pickle.dump(None, f)
                    pickle.dump(None, f)
                    pickle.dump(ntotal, f)
                    pickle.dump(use_subdir, f)
                    pickle.dump(nmax_persubdir, f)
                    pickle.dump(attrs if attrs is not None else {}, f)

            comm.Barrier()

            deg_hist = torch.zeros(0, dtype=torch.long)

            local_i = 0

            # iterate over assigned molecules
            pbar = iterate_tqdm(
                id_list,
                verbosity,
                total=len(id_list),
                desc=f"Streaming {label} molecules",
            )

            for chembl_id in pbar:
                wfn_dir = self.molecule_dirs[chembl_id]["wfn_dir"]
                sdf_dir = self.molecule_dirs[chembl_id]["sdf_dir"]

                # iterate over conformers belonging to the molecule
                for filename in sorted(os.listdir(wfn_dir)):
                    wfn_path = os.path.join(wfn_dir, filename)
                    if not os.path.isfile(wfn_path):
                        continue

                    conf_id = filename.split(".")[0][-2:]
                    sdf_path = os.path.join(sdf_dir, f"conf_{conf_id}.sdf")

                    # construct graph object
                    data_object = self._mol_to_graph(
                        wfn_path=wfn_path,
                        sdf_path=sdf_path,
                    )
                    global_i = noffset + local_i

                    # determine pickle output path
                    if use_subdir:
                        subdir = os.path.join(basedir, str(global_i // nmax_persubdir))
                        os.makedirs(subdir, exist_ok=True)
                        fname = os.path.join(subdir, f"{label}-{global_i}.pkl")
                    else:
                        fname = os.path.join(basedir, f"{label}-{global_i}.pkl")
                    
                    # write graph to pickle
                    with open(fname, "wb") as f:
                        pickle.dump(data_object, f)

                    # update pna degree histogram using training graphs only
                    if compute_pna_deg and label == "trainset":
                        deg_hist = _update_deg_hist(deg_hist, data_object)

                    local_i += 1

                    # periodic progress logging
                    if log_every is not None and log_every > 0 and local_i % log_every == 0:
                        log(f"[INFO] Rank {rank}: split={label}, wrote {local_i}/{local_n} local graphs")
                    
                    # immediately release memory for current graph
                    del data_object
                    gc.collect()
            
            log(f"[DONE] Rank {rank}: split={label}, wrote {local_i} local graphs")
            log(f"[DONE] Streaming split={label}", rank=0)

            return deg_hist
        
        # stream each dataset split
        local_deg_hist = _write_split("trainset", trainset_ids)

        _write_split("valset", valset_ids)
        _write_split("testset", testset_ids)

        # reduce pna degree histogram across all MPI ranks
        local_max_deg = torch.tensor([local_deg_hist.numel()], dtype=torch.long)
        dist.all_reduce(local_max_deg, op=dist.ReduceOp.MAX)

        padded_deg_hist = torch.zeros(int(local_max_deg.item()), dtype=torch.long)
        padded_deg_hist[:local_deg_hist.numel()] = local_deg_hist
        dist.all_reduce(padded_deg_hist, op=dist.ReduceOp.SUM)

        pna_deg = padded_deg_hist.numpy()

        # update metadata with global PNA degree histogram
        if rank == 0:
            meta_path = os.path.join(basedir, "trainset-meta.pkl")
            with open(meta_path, "rb") as f:
                minmax_node_feature = pickle.load(f)
                minmax_graph_feature = pickle.load(f)
                ntotal = pickle.load(f)
                meta_use_subdir = pickle.load(f)
                meta_nmax_persubdir = pickle.load(f)
                attrs = pickle.load(f)

            attrs["pna_deg"] = pna_deg

            with open(meta_path, "wb") as f:
                pickle.dump(minmax_node_feature, f)
                pickle.dump(minmax_graph_feature, f)
                pickle.dump(ntotal, f)
                pickle.dump(meta_use_subdir, f)
                pickle.dump(meta_nmax_persubdir, f)
                pickle.dump(attrs, f)

        comm.Barrier()

        # gather global conformer counts across all MPI ranks
        train_global = sum(comm.allgather(_count_conformers(trainset_ids)))
        val_global = sum(comm.allgather(_count_conformers(valset_ids)))
        test_global = sum(comm.allgather(_count_conformers(testset_ids)))

        return {
            "pna_deg": pna_deg,
            "train_num_conformers": train_global,
            "val_num_conformers": val_global,
            "test_num_conformers": test_global,
        }
        
        
    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]


# ----------------------------------------------------------------------------------------------------
# QMugs inference class (with pretrained model)

class QMugsInference:
    '''wrapper for running QMugs density matrix inference'''

    def __init__(
        self,
        model_dir,
        checkpoint_path=None,
    ):
        # dir and path setup
        self.model_dir = model_dir
        self._qmugs_archive_maps = {}
        self._find_checkpoint(checkpoint_path)
        self._load_config()
        self._load_model()
    
    # ----------------------------------------------------------------------------------------------------
    # initialization
    
    def _find_checkpoint(self, checkpoint_path=None):
        '''return path to model checkpoint'''

        # return checkpoint file path if provided
        if checkpoint_path is not None:
            if os.path.isabs(checkpoint_path):
                self.checkpoint_path = checkpoint_path
            else:
                self.checkpoint_path = os.path.join(self.model_dir, checkpoint_path)
            if not os.path.isfile(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # get last available checkpoint if no checkpoint path provided
        else: 
            candidates = sorted(glob.glob(os.path.join(self.model_dir, "*.pk")))
            if len(candidates) == 0:
                raise FileNotFoundError(f"No .pk checkpoint files found in {self.model_dir}")
            
            self.checkpoint_path = candidates[-1]


    def _load_config(self):
        '''loads config file for the specified pretrained model'''
        config_path = os.path.join(self.model_dir, "config.json")
        self.config_path = config_path

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"config.json not found in {self.model_dir}")
        with open(config_path, "r") as f:
            config = json.load(f)
        self.config = config


    def _load_model(self):
        '''loads the specified pretrained model'''

        # resolve precision
        precision_str = self.config["NeuralNetwork"]["Training"].get("precision", "fp32")
        precision, param_dtype, _ = resolve_precision(precision=precision_str)
        self.param_dtype = param_dtype
        self.precision = precision
        torch.set_default_dtype(param_dtype)

        # device
        device = get_device()
        self.device = device
        autocast_ctx, _ = get_autocast_and_scaler(precision=precision)
        self.autocast_ctx = autocast_ctx

        # declare model
        self.model = hydragnn.models.create.create_model_config(
            config=self.config["NeuralNetwork"],
            verbosity=self.config["Verbosity"]["level"],
        )

        # load state dict
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        clean_state = OrderedDict(
            (k[len("module."):] if k.startswith("module.") else k, v)
            for k, v in state_dict.items()
        )
        missing, unexpected = self.model.load_state_dict(clean_state, strict=False)
        if missing:
            log(f"[WARNING] missing keys when loading model: {missing}", rank=0)
        if unexpected:
            log(f"[WARNING] unexpected keys when loading model: {unexpected}", rank=0)

        # load model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False) # freeze parameters
        self.model = self.model.to(dtype=param_dtype, device=device)
    

    ######################################################################################################
    # DEFUNCT
    # ----------------------------------------------------------------------------------------------------
    # naive inference without streaming

    def run_inference(
        self, 
        dataset,
        evaluate=True,
        criterions=['mse', 'mae', 'smooth_l1', 'rmse'],
        return_numpy_matrices=True,
    ):
        '''
        runs inference naively with model on preloaded dataset.
        suitable only for very small test sets. for larger datasets, use run_streaming_inference().
        '''

        max_size = QMugsDataset.max_padded_density_matrix_dimension # max padding size

        # run inference
        all_pred = []
        all_true = []
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for batch in loader:
            batch = batch.to(self.device)

            with torch.no_grad():
                pred = self.model(batch)

                # total density matrix prediction
                if self.config["NeuralNetwork"]["Variables_of_interest"]["output_names"] == ['density_matrix']:
                    pred_batch = pred[0].reshape(-1, max_size, max_size)
                    true_batch = batch.y.reshape(-1, max_size, max_size)
                    dims_batch = batch.density_matrix_dim
                else:
                    raise NotImplementedError

                # unpad
                for raw_pred_mat, raw_true_mat, dim in zip(
                    pred_batch,
                    true_batch,
                    dims_batch,
                ):
                    assert (raw_pred_mat.shape == raw_true_mat.shape)
                    assert (raw_pred_mat.shape[0] == max_size)

                    dim = int(dim.item())
                    pred_mat = raw_pred_mat[:dim, :dim]
                    true_mat = raw_true_mat[:dim, :dim]

                    all_pred.append(pred_mat)
                    all_true.append(true_mat)
        
        # evaluate
        evaluation = {}
        if evaluate:
            # iterate through all specified criterions
            for criterion in criterions:
                losses = []
                loss_function = loss_function_selection(criterion)

                # compute losses
                for pred_mat, true_mat in zip(all_pred, all_true):
                    losses.append(loss_function(pred_mat, true_mat))
                loss_mean = torch.stack(losses).mean()
                loss_std = torch.stack(losses).std()

                # save results
                evaluation[criterion] = {
                    'mean': loss_mean.item(),
                    'std': loss_std.item(),
                    'values': [l.item() for l in losses],
                }  

        # return results
        if return_numpy_matrices:
            all_pred_numpy = [pred_mat.detach().cpu().numpy() for pred_mat in all_pred]
            all_true_numpy = [true_mat.detach().cpu().numpy() for true_mat in all_true]
            return all_pred_numpy, all_true_numpy, evaluation            

        else: 
            return all_pred, all_true, evaluation

    ######################################################################################################

    # ----------------------------------------------------------------------------------------------------
    # inference with streaming across MPI ranks

    def run_streaming_inference(
        self,
        dataset,
        evaluate=True,
        criterions=['mse', 'mae', 'smooth_l1', 'rmse'],
        return_numpy_matrices=False,
        save_per_matrix_metrics=False,
        batch_size=1,
        # downstream calculations
        run_downstream_calculation=False,
        downstream_calculations=None,
        qmugs_data_dir: str = None,
        artifacts_dir: str = None,
        # electron density related configs
        electron_density_method: str = 'cubeprop',
        cubeprop_grid_spacing=None,
        cubeprop_num_grid_points=None,
        manual_density_padding_angstrom: float = 3.0,
        manual_density_spacing_angstrom=None,
        manual_density_num_grid_points=None,
        manual_density_block_size: int = 2000,
        # downstream predictions
        run_downstream_prediction=False,
        downstream_predictions=None,
        # logging
        verbosity: int = 2,
        log_every: int = 100,
    ):
        '''runs inference with model on preloaded or lazy dataset'''

        log("[START] Inference initiated", rank=0)

        # disable matrix storage for streaming inference
        if return_numpy_matrices:
            raise NotImplementedError('return_numpy_matrices=True is disabled for streaming inference')

        downstream_calculations = downstream_calculations or []
        downstream_predictions = downstream_predictions or []
        
        if run_downstream_calculation and not evaluate:
            raise ValueError("run_downstream_calculation=True requires evaluate=True")
        if run_downstream_calculation and len(downstream_calculations) == 0:
            raise ValueError("run_downstream_calculation=True requires at least one downstream calculation")
        if run_downstream_prediction:
            self.run_downstream_predictions(downstream_predictions=downstream_predictions)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        inference_start_time = time.time()
        max_size = QMugsDataset.max_padded_density_matrix_dimension

        # initialize local metric accumulators
        streaming_state = None
        if evaluate:
            streaming_state = self._init_streaming_metric_state(criterions=criterions)
        per_matrix_metrics = []

        # lazy dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        try:
            local_num_batches = len(loader)
        except TypeError:
            local_num_batches = None
        local_num_batches_label = local_num_batches if local_num_batches is not None else "unknown"
        log(
            f"[START] Streaming inference: rank={rank}, "
            f"local_batches={local_num_batches_label}, batch_size={batch_size}",
        )

        # run inference
        pbar = iterate_tqdm(
            loader,
            verbosity,
            total=local_num_batches,
            desc=f"Streaming inference rank {rank}",
        )
        local_batch_count = 0
        local_matrix_count = 0
        for batch in pbar:
            local_batch_count += 1
            batch = batch.to(self.device)

            with torch.no_grad():
                with self.autocast_ctx:
                    pred = self.model(batch)

                # total density matrix prediction
                if self.config["NeuralNetwork"]["Variables_of_interest"]["output_names"] == ['density_matrix']:
                    pred_batch = pred[0].reshape(-1, max_size, max_size)
                    true_batch = batch.y.reshape(-1, max_size, max_size)
                    dims_batch = batch.density_matrix_dim
                else:
                    raise NotImplementedError

                # postprocess matrices
                for batch_index, (raw_pred_mat, raw_true_mat, dim) in enumerate(zip(
                    pred_batch,
                    true_batch,
                    dims_batch,
                )):
                    assert(raw_pred_mat.shape == raw_true_mat.shape)
                    assert(raw_pred_mat.shape[0] == max_size)

                    # unpad matrices to original size
                    dim = int(dim.item())
                    pred_mat = raw_pred_mat[:dim, :dim]
                    true_mat = raw_true_mat[:dim, :dim]
                    local_matrix_count += 1

                    downstream_metric_values = None
                    downstream_record_values = None
                    if run_downstream_calculation:
                        downstream_metric_values, downstream_record_values = self.run_downstream_calculations(
                            downstream_calculations=downstream_calculations,
                            pred_mat=pred_mat,
                            true_mat=true_mat,
                            batch=batch,
                            batch_index=batch_index,
                            qmugs_data_dir=qmugs_data_dir,
                            artifacts_dir=artifacts_dir,
                            electron_density_method=electron_density_method,
                            cubeprop_grid_spacing=cubeprop_grid_spacing,
                            cubeprop_num_grid_points=cubeprop_num_grid_points,
                            manual_density_padding_angstrom=manual_density_padding_angstrom,
                            manual_density_spacing_angstrom=manual_density_spacing_angstrom,
                            manual_density_num_grid_points=manual_density_num_grid_points,
                            manual_density_block_size=manual_density_block_size,
                        )

                    # evaluate and update streaming metrics
                    if evaluate:
                        self._update_streaming_metric_state(
                            state=streaming_state,
                            pred_mat=pred_mat,
                            true_mat=true_mat,
                            batch=batch,
                            batch_index=batch_index,
                            criterions=criterions,
                            save_per_matrix_metrics=save_per_matrix_metrics,
                            per_matrix_metrics=per_matrix_metrics,
                            downstream_metric_values=downstream_metric_values,
                            downstream_record_values=downstream_record_values,
                        )

                    # release matrices from memory
                    del pred_mat
                    del true_mat

            del batch
            del pred

            # periodic progress logging
            if log_every is not None and log_every > 0 and local_batch_count % log_every == 0:
                log(
                    f"[INFO] Rank {rank}: streaming inference processed "
                    f"{local_batch_count}/{local_num_batches_label} batches, "
                    f"{local_matrix_count} matrices"
                )

        # finalize metrics
        evaluation = {}
        if evaluate:
            evaluation = self._finalize_streaming_metric_state(state=streaming_state)

        inference_time = time.time() - inference_start_time
        evaluation['inference_time'] = inference_time
        log(
            f"[DONE] Rank {rank}: streaming inference processed "
            f"{local_batch_count} batches, {local_matrix_count} matrices "
            f"in {inference_time:.2f} seconds"
        )

        return evaluation, per_matrix_metrics
    

    # ----------------------------------------------------------------------------------------------------
    # downstream calculations

    def run_downstream_dipole_moment_calculation(
        self,
        pred_mat,
        true_mat,
        batch,
        batch_index,
        qmugs_data_dir: str = None,
    ):
        '''compute true and predicted dipole moments for one conformer'''

        # only total density matrix predictions are supported
        if self.config["NeuralNetwork"]["Variables_of_interest"]["output_names"] != ['density_matrix']:
            raise NotImplementedError(
                "Downstream dipole moment calculations currently require total density matrix predictions"
            )

        # load stored ground truth Psi4 wavefunction
        wfn_path = self._resolve_wfn_path(
            batch=batch,
            batch_index=batch_index,
            qmugs_data_dir=qmugs_data_dir,
        )
        if not os.path.isfile(wfn_path):
            raise FileNotFoundError(f"Psi4 wavefunction file not found: {wfn_path}")

        psi4_wfn = psi4.core.Wavefunction.from_file(wfn_path)

        # Convert to float64 for contraction with Psi4 double precision AO integrals.
        pred_np = _as_numpy_density_matrix(pred_mat).astype(np.float64, copy=False) # (n_basis, n_basis)
        true_np = _as_numpy_density_matrix(true_mat).astype(np.float64, copy=False) # (n_basis, n_basis)
        assert pred_np.shape == true_np.shape

        n_basis = int(psi4_wfn.basisset().nbf())
        assert pred_np.shape == (n_basis, n_basis)
        assert true_np.shape == (n_basis, n_basis)

        # true and predicted dipoles use same geometry and basis
        true_dipole_au, true_nuclear_dipole_au, true_electronic_dipole_au = dipole_moment_from_density_matrix(
            density_matrix=true_np,
            psi4_wfn=psi4_wfn,
        ) # (3,)
        pred_dipole_au, pred_nuclear_dipole_au, pred_electronic_dipole_au = dipole_moment_from_density_matrix(
            density_matrix=pred_np,
            psi4_wfn=psi4_wfn,
        ) # (3,)

        dipole_error_au = pred_dipole_au - true_dipole_au # (3,)
        dipole_euclideian_error_au = np.linalg.norm(dipole_error_au)

        eps = 1e-12
        true_dipole_norm_au = np.linalg.norm(true_dipole_au)
        pred_dipole_norm_au = np.linalg.norm(pred_dipole_au)
        true_dipole_range_au = np.max(true_dipole_au) - np.min(true_dipole_au)
        true_dipole_unit = true_dipole_au / (true_dipole_norm_au + eps) # (3,)
        pred_dipole_unit = pred_dipole_au / (pred_dipole_norm_au + eps) # (3,)

        # cosine similarity of true and predicted dipole vectors
        dipole_cosine_similarity = np.dot(
            pred_dipole_au,
            true_dipole_au,
        ) / (pred_dipole_norm_au * true_dipole_norm_au + eps)

        # RMSE over vector components, normalized by range of true components
        dipole_rmse_au = np.sqrt(np.mean(dipole_error_au ** 2))
        dipole_range_normalized_rmse = dipole_rmse_au / (true_dipole_range_au + eps)

        # RMSE between unit-normalized dipole vectors
        dipole_unit_vector_rmse = np.sqrt(np.mean(
            (pred_dipole_unit - true_dipole_unit) ** 2
        ))

        # absolute error in vector magnitude, normalized by true magnitude
        dipole_relative_magnitude_error = abs(
            pred_dipole_norm_au - true_dipole_norm_au
        ) / (true_dipole_norm_au + eps)

        # scalar metrics for streaming aggregation
        metric_values = {
            'dipole_moment_absolute_component_error_x_au': abs(dipole_error_au[0]),
            'dipole_moment_absolute_component_error_y_au': abs(dipole_error_au[1]),
            'dipole_moment_absolute_component_error_z_au': abs(dipole_error_au[2]),
            'dipole_euclideian_error_au': dipole_euclideian_error_au,
            'dipole_moment_cosine_similarity': dipole_cosine_similarity,
            'dipole_moment_rmse': dipole_rmse_au,
            'dipole_moment_range_normalized_rmse': dipole_range_normalized_rmse,
            'dipole_moment_unit_vector_rmse': dipole_unit_vector_rmse,
            'dipole_moment_relative_magnitude_error': dipole_relative_magnitude_error,
        }

        # per molecule values for downstream debugging
        record_values = {
            'wfn_path': wfn_path,
            'true_dipole_au': true_dipole_au.tolist(),
            'predicted_dipole_au': pred_dipole_au.tolist(),
            'dipole_error_au': dipole_error_au.tolist(),
            'true_nuclear_dipole_au': true_nuclear_dipole_au.tolist(),
            'predicted_nuclear_dipole_au': pred_nuclear_dipole_au.tolist(),
            'true_electronic_dipole_au': true_electronic_dipole_au.tolist(),
            'predicted_electronic_dipole_au': pred_electronic_dipole_au.tolist(),
        }

        return metric_values, record_values


    def run_downstream_electron_density_calculation(
        self,
        pred_mat,
        true_mat,
        batch,
        batch_index,
        qmugs_data_dir: str = None,
        artifacts_dir: str = None,
        cubeprop_grid_spacing=None,
        cubeprop_num_grid_points=None,
        electron_density_method: str = 'cubeprop',
        manual_density_padding_angstrom: float = 3.0,
        manual_density_spacing_angstrom=None,
        manual_density_num_grid_points=None,
        manual_density_block_size: int = 2000,
    ):
        '''compute true and predicted electron density grid metrics for one conformer'''

        # only total density matrix predictions are supported
        if self.config["NeuralNetwork"]["Variables_of_interest"]["output_names"] != ['density_matrix']:
            raise NotImplementedError("Downstream electron density calculations currently require total density matrix predictions")
        
        # validate method
        electron_density_method = str(electron_density_method).strip().lower()
        if electron_density_method not in ['cubeprop', 'manual_ao']:
            raise ValueError(f"Unknown electron density method: {electron_density_method}")
        if electron_density_method == 'cubeprop' and artifacts_dir is None:
            raise ValueError("Downstream electron density calculations require artifacts_dir")
        if cubeprop_grid_spacing is not None and cubeprop_num_grid_points is not None:
            raise ValueError("Specify only one of cubeprop_grid_spacing or cubeprop_num_grid_points")
        
        if artifacts_dir is not None:
            artifacts_dir = os.path.abspath(artifacts_dir)

        # load stored ground truth Psi4 wavefunction
        wfn_path = self._resolve_wfn_path(
            batch=batch,
            batch_index=batch_index,
            qmugs_data_dir=qmugs_data_dir,
        )
        if not os.path.isfile(wfn_path):
            raise FileNotFoundError(f"Psi4 wavefunction file not found: {wfn_path}")

        true_wfn = psi4.core.Wavefunction.from_file(wfn_path)

        # Convert to float64 for Psi4 double precision density matrices.
        pred_np = _as_numpy_density_matrix(pred_mat).astype(np.float64, copy=False) # (n_basis, n_basis)
        true_np = _as_numpy_density_matrix(true_mat).astype(np.float64, copy=False) # (n_basis, n_basis)
       
        n_basis = int(true_wfn.basisset().nbf())
        assert pred_np.shape == true_np.shape
        assert true_np.shape == (n_basis, n_basis)

        molecule_dir = None
        true_cube_paths = []
        pred_cube_paths = []
        
        # use Psi4 cubeprops
        if electron_density_method == 'cubeprop':
            # predicted Psi4 wavefunction uses the same molecule and def2-SVP basis
            pred_wfn = wavefunction_from_total_density_matrix(
                reference_wfn=true_wfn,
                total_density_matrix=pred_np,
                wfn_path=wfn_path,
            )

            # unique artifact paths prevent cubeprop filename collisions across ranks
            chembl_id = str(self._get_batch_attr_value(
                batch=batch,
                name='chembl_id',
                batch_index=batch_index,
            ))
            conformer_id = str(self._get_batch_attr_value(
                batch=batch,
                name='conformer_id',
                batch_index=batch_index,
            ))

            rank = int(os.environ.get('SLURM_PROCID', os.environ.get('PMI_RANK', '0')))
            molecule_dir = os.path.join(
                artifacts_dir,
                'cubeprops',
                f'rank_{rank}',
                f'{chembl_id}_conf_{conformer_id}',
                'electron_density',
            )

            true_cube_dir = os.path.join(molecule_dir, 'true')
            pred_cube_dir = os.path.join(molecule_dir, 'predicted')
            cubeprop_grid_config = None
            if cubeprop_grid_spacing is not None:
                cubeprop_grid_config = {'spacing': cubeprop_grid_spacing}
            elif cubeprop_num_grid_points is not None:
                cubeprop_grid_config = {'num_grid_points': cubeprop_num_grid_points}

            # cubeprop evaluates density from the wavefunction density matrices on its default cube grid
            write_cubeprops(
                psi4_wfn=true_wfn,
                cube_dir=true_cube_dir,
                cubeprop_names=['density'],
                grid_config=cubeprop_grid_config,
            )
            write_cubeprops(
                psi4_wfn=pred_wfn,
                cube_dir=pred_cube_dir,
                cubeprop_names=['density'],
                grid_config=cubeprop_grid_config,
            )

            # read true and predicted density values from the written cube files
            true_cubeprops = read_cubeprops(
                cube_dir=true_cube_dir,
                cubeprop_names=['density'],
            )
            pred_cubeprops = read_cubeprops(
                cube_dir=pred_cube_dir,
                cubeprop_names=['density'],
            )

            true_density = true_cubeprops['density']['values'] # (n_x, n_y, n_z)
            true_cube_metadata = true_cubeprops['density']['metadata']
            true_cube_paths = true_cubeprops['density']['paths']
            pred_density = pred_cubeprops['density']['values'] # (n_x, n_y, n_z)
            pred_cube_metadata = pred_cubeprops['density']['metadata']
            pred_cube_paths = pred_cubeprops['density']['paths']

            assert_same_cube_grid(
                first_cube_metadata=true_cube_metadata,
                second_cube_metadata=pred_cube_metadata,
            )
        
        else:
            # manual AO path evaluates rho(r) = phi(r)^T P phi(r) directly
            true_density, true_cube_metadata = electron_density_from_density_matrix(
                density_matrix=true_np,
                psi4_wfn=true_wfn,
                padding_angstrom=manual_density_padding_angstrom,
                spacing_angstrom=manual_density_spacing_angstrom,
                num_grid_points=manual_density_num_grid_points,
                block_size=manual_density_block_size,
            ) # (n_x, n_y, n_z)
            pred_density, pred_cube_metadata = electron_density_from_density_matrix(
                density_matrix=pred_np,
                psi4_wfn=true_wfn,
                padding_angstrom=manual_density_padding_angstrom,
                spacing_angstrom=manual_density_spacing_angstrom,
                num_grid_points=manual_density_num_grid_points,
                block_size=manual_density_block_size,
            ) # (n_x, n_y, n_z)

            assert_same_cube_grid(
                first_cube_metadata=true_cube_metadata,
                second_cube_metadata=pred_cube_metadata,
            )
        assert pred_density.shape == true_density.shape

        # voxelwise density errors on the common cube grid
        density_error = pred_density - true_density # (n_x, n_y, n_z)
        abs_density_error = np.abs(density_error) # (n_x, n_y, n_z)
        squared_density_error = density_error ** 2 # (n_x, n_y, n_z)

        eps = 1e-12
        voxel_volume = float(true_cube_metadata['voxel_volume'])
        true_density_range = float(np.max(true_density) - np.min(true_density))
        true_density_integral = float(np.sum(true_density) * voxel_volume)
        pred_density_integral = float(np.sum(pred_density) * voxel_volume)

        # scalar metrics for streaming aggregation
        mean_absolute_voxel_grid_error = float(np.mean(abs_density_error))
        mean_squared_voxel_grid_error = float(np.mean(squared_density_error))
        root_mean_squared_voxel_grid_error = float(np.sqrt(mean_squared_voxel_grid_error))
        range_normalized_rmse = root_mean_squared_voxel_grid_error / (true_density_range + eps)
        true_density_l2_integral = float(np.sqrt(np.sum(true_density ** 2) * voxel_volume))
        integrated_l2_error = float(
            np.sqrt(np.sum(squared_density_error) * voxel_volume) / (true_density_l2_integral + eps)
        )
        absolute_fractional_error = float(
            np.sum(abs_density_error) * voxel_volume / (true_density_integral + eps)
        )

        metric_values = {
            'electron_density_mean_absolute_voxel_grid_error': mean_absolute_voxel_grid_error,
            'electron_density_mean_squared_voxel_grid_error': mean_squared_voxel_grid_error,
            'electron_density_root_mean_squared_voxel_grid_error': root_mean_squared_voxel_grid_error,
            'electron_density_range_normalized_rmse': range_normalized_rmse,
            'electron_density_integrated_l2_error': integrated_l2_error,
            'electron_density_absolute_fractional_error': absolute_fractional_error,
        }

        # per molecule values for downstream debugging
        record_values = {
            'electron_density_method': electron_density_method,
            'electron_density_wfn_path': wfn_path,
            'electron_density_artifact_dir': molecule_dir,
            'electron_density_true_cube_paths': true_cube_paths,
            'electron_density_predicted_cube_paths': pred_cube_paths,
            'electron_density_grid_shape': list(true_cube_metadata['grid_shape']),
            'electron_density_voxel_volume_au3': voxel_volume,
            'electron_density_true_integral': true_density_integral,
            'electron_density_predicted_integral': pred_density_integral,
        }

        return metric_values, record_values


    # ----------------------------------------------------------------------------------------------------
    # downstream predictions


    # ----------------------------------------------------------------------------------------------------
    # downstream task wrappers

    def run_downstream_calculations(
        self,
        downstream_calculations,
        pred_mat,
        true_mat,
        batch,
        batch_index,
        qmugs_data_dir: str = None,
        artifacts_dir: str = None,
        # electron density related configs
        electron_density_method: str = 'cubeprop',
        cubeprop_grid_spacing=None,
        cubeprop_num_grid_points=None,
        manual_density_padding_angstrom: float = 3.0,
        manual_density_spacing_angstrom=None,
        manual_density_num_grid_points=None,
        manual_density_block_size: int = 2000,
    ):
        '''run requested downstream physics calculations for one conformer'''

        metric_values = {}
        record_values = {}

        # run each requested downstream physics task
        for calculation in downstream_calculations:
            if calculation == 'dipole_moment':
                log("[INFO] Running downstream dipole moment calculations", rank=0)
                calculation_metrics, calculation_record = self.run_downstream_dipole_moment_calculation(
                    pred_mat=pred_mat,
                    true_mat=true_mat,
                    batch=batch,
                    batch_index=batch_index,
                    qmugs_data_dir=qmugs_data_dir,
                )
            elif calculation == 'electron_density':
                log("[INFO] Running downstream electron density calculations", rank=0)
                calculation_metrics, calculation_record = self.run_downstream_electron_density_calculation(
                    pred_mat=pred_mat,
                    true_mat=true_mat,
                    batch=batch,
                    batch_index=batch_index,
                    qmugs_data_dir=qmugs_data_dir,
                    artifacts_dir=artifacts_dir,
                    electron_density_method=electron_density_method,
                    cubeprop_grid_spacing=cubeprop_grid_spacing,
                    cubeprop_num_grid_points=cubeprop_num_grid_points,
                    manual_density_padding_angstrom=manual_density_padding_angstrom,
                    manual_density_spacing_angstrom=manual_density_spacing_angstrom,
                    manual_density_num_grid_points=manual_density_num_grid_points,
                    manual_density_block_size=manual_density_block_size,
                )
            else:
                raise NotImplementedError(f"Unknown downstream calculation: {calculation}")

            metric_values.update(calculation_metrics)
            record_values.update(calculation_record)

        return metric_values, record_values


    def run_downstream_predictions(self, downstream_predictions):
        '''raise for requested downstream ML predictions until they are implemented'''

        if downstream_predictions is None or len(downstream_predictions) == 0:
            raise ValueError("run_downstream_prediction=True requires at least one downstream prediction")

        # downstream ML predictions are a reserved interface for future work
        raise NotImplementedError(
            f"Downstream ML predictions are not implemented yet: {downstream_predictions}"
        )


    # ----------------------------------------------------------------------------------------------------
    # helpers for more optimized inference with streaming across MPI ranks

    def _init_streaming_metric_state(self, criterions):
        '''initialize local streaming metric accumulators'''

        state = {
            'count': 0,

            # loss metrics usable in training
            'losses': {
                criterion: {
                    'sum': 0.0,
                    'sumsq': 0.0,
                }
                for criterion in criterions
            },

            # density matrix constraint metrics
            'trace_error': {
                'sum': 0.0,
                'sumsq': 0.0,
            },
            'hermitian_error': {
                'sum': 0.0,
                'sumsq': 0.0,
            },
            'minimum_eigenvalue': {
                'sum': 0.0,
                'sumsq': 0.0,
            },
            'negative_eigenvalue_count': {
                'sum': 0.0,
                'sumsq': 0.0,
            },

            # other reconstruction metrics
            'r2_per_matrix': {
                'sum': 0.0,
                'sumsq': 0.0,
            },
            'relative_frobenius_error': {
                'sum': 0.0,
                'sumsq': 0.0,
            },
            'cosine_similarity': {
                'sum': 0.0,
                'sumsq': 0.0,
            },

            # downstream calculation metrics
            'downstream_metric_names': [],
        }

        return state


    def _ensure_streaming_metric(self, state, name):
        '''add a scalar metric accumulator if it has not been initialized'''

        if name in state:
            return

        state[name] = {
            'sum': 0.0,
            'sumsq': 0.0,
        }
        state['downstream_metric_names'].append(name)


    def _update_scalar_metric(self, state, name, value):
        '''update one scalar metric accumulator'''

        state[name]['sum'] += value
        state[name]['sumsq'] += value * value


    def _update_streaming_metric_state(
        self,
        state,
        pred_mat,
        true_mat,
        batch,
        batch_index,
        criterions,
        save_per_matrix_metrics=False,
        per_matrix_metrics=None, # persisted dict of per matrix metrics
        downstream_metric_values=None,
        downstream_record_values=None,
    ):
        '''update local streaming metrics from one unpadded matrix pair'''

        # increment local sample count
        state['count'] += 1

        # store per matrix quantities if requested
        record = None
        if save_per_matrix_metrics:
            record = {
                'chembl_id': batch.chembl_id[batch_index],
                'conformer_id': batch.conformer_id[batch_index],
                'density_matrix_dim': int(batch.density_matrix_dim[batch_index].item()),
                'natoms': int(batch.natoms_int[batch_index].item()),
                'charge': int(batch.charge[batch_index].item()),
                'multiplicity': int(batch.multiplicity[batch_index].item()),
            }

        # reconstruction losses available to train on
        for criterion in criterions:
            value = _loss_value(
                pred_mat=pred_mat,
                true_mat=true_mat,
                criterion=criterion,
            )

            state['losses'][criterion]['sum'] += value
            state['losses'][criterion]['sumsq'] += value * value

            if save_per_matrix_metrics:
                record[criterion] = value

        # ----------------------------------------------------------------------------------------------------
        # trace conservation
        trace_error, trace_pred, trace_true = _trace_error(
            pred_mat=pred_mat,
            true_mat=true_mat,
        )
        self._update_scalar_metric(
            state=state,
            name='trace_error',
            value=trace_error,
        )

        if save_per_matrix_metrics:
            record['trace_error'] = trace_error
            record['predicted_trace'] = trace_pred
            record['true_trace'] = trace_true

        # ----------------------------------------------------------------------------------------------------
        # hermitianity
        hermitian_error = _hermitian_error(
            pred_mat=pred_mat,
        )
        self._update_scalar_metric(
            state=state,
            name='hermitian_error',
            value=hermitian_error,
        )

        if save_per_matrix_metrics:
            record['hermitian_error'] = hermitian_error

        # ----------------------------------------------------------------------------------------------------
        # positive semidefinite metrics
        min_eigenvalue, negative_eigenvalue_count = _psd_metrics(
            pred_mat=pred_mat,
        )
        self._update_scalar_metric(
            state=state,
            name='minimum_eigenvalue',
            value=min_eigenvalue,
        )
        self._update_scalar_metric(
            state=state,
            name='negative_eigenvalue_count',
            value=float(negative_eigenvalue_count),
        )

        if save_per_matrix_metrics:
            record['minimum_eigenvalue'] = min_eigenvalue
            record['negative_eigenvalue_count'] = negative_eigenvalue_count

        # ----------------------------------------------------------------------------------------------------
        # per matrix R2
        r2_per_matrix = _r2_per_matrix(
            pred_mat=pred_mat,
            true_mat=true_mat,
        )
        self._update_scalar_metric(
            state=state,
            name='r2_per_matrix',
            value=r2_per_matrix,
        )

        if save_per_matrix_metrics:
            record['r2_per_matrix'] = r2_per_matrix

        # ----------------------------------------------------------------------------------------------------
        # relative frobenius erorr
        relative_frobenius_error = _relative_frobenius_error(
            pred_mat=pred_mat,
            true_mat=true_mat,
        )
        self._update_scalar_metric(
            state=state,
            name='relative_frobenius_error',
            value=relative_frobenius_error,
        )

        if save_per_matrix_metrics:
            record['relative_frobenius_error'] = relative_frobenius_error

        # ----------------------------------------------------------------------------------------------------
        # cosine similarity of matrices as flattened vectors
        cosine_similarity = _cosine_similarity(
            pred_mat=pred_mat,
            true_mat=true_mat,
        )
        self._update_scalar_metric(
            state=state,
            name='cosine_similarity',
            value=cosine_similarity,
        )

        # ----------------------------------------------------------------------------------------------------

        if save_per_matrix_metrics:
            record['cosine_similarity'] = cosine_similarity

        # ----------------------------------------------------------------------------------------------------
        # downstream calculation metrics
        if downstream_metric_values is not None:
            for name, value in downstream_metric_values.items():
                value = float(value)
                self._ensure_streaming_metric(
                    state=state,
                    name=name,
                )
                self._update_scalar_metric(
                    state=state,
                    name=name,
                    value=value,
                )

                if save_per_matrix_metrics:
                    record[name] = value

        if save_per_matrix_metrics and downstream_record_values is not None:
            record.update(downstream_record_values)

        # ----------------------------------------------------------------------------------------------------

        if save_per_matrix_metrics:
            # save record for downstream analysis
            per_matrix_metrics.append(record)


    def _finalize_streaming_metric_state(self, state):
        '''finalize local streaming metrics as mean and standard deviation'''

        count = state['count']
        if count == 0:
            raise ValueError('Cannot finalize streaming metrics with zero samples')

        results = {}

        # finalize reconstruction lossess
        for criterion, values in state['losses'].items():
            mean = values['sum'] / count
            var = max(values['sumsq'] / count - mean * mean, 0.0)
            results[criterion] = {
                'mean': mean,
                'std': var ** 0.5,
            }

        # finalize density matrix metrics
        for name in [
            'trace_error',
            'hermitian_error',
            'minimum_eigenvalue',
            'negative_eigenvalue_count',
            'r2_per_matrix',
            'relative_frobenius_error',
            'cosine_similarity',
        ]:
            mean = state[name]['sum'] / count
            var = max(state[name]['sumsq'] / count - mean * mean, 0.0)
            results[name] = {
                'mean': mean,
                'std': var ** 0.5,
            }

        # finalize downstream calculation metrics
        for name in state['downstream_metric_names']:
            mean = state[name]['sum'] / count
            var = max(state[name]['sumsq'] / count - mean * mean, 0.0)
            results[name] = {
                'mean': mean,
                'std': var ** 0.5,
            }

        results['count'] = count

        return results


    def _get_batch_attr_value(self, batch, name, batch_index):
        '''return one graph-level attribute from a PyG batch'''

        # PyG stores string attributes as lists after batching
        value = getattr(batch, name)
        if isinstance(value, (list, tuple)):
            return value[batch_index]

        # tensor attributes are indexed by batch position
        if torch.is_tensor(value):
            if value.dim() == 0:
                return value.item()
            return value[batch_index].item()

        return value


    def _load_qmugs_archive_map(self, qmugs_data_dir: str):
        '''load the CHEMBL ID to wavefunction archive map for QMugs'''

        # cache archive lookup by QMugs data directory
        qmugs_data_dir = os.path.abspath(qmugs_data_dir)
        if qmugs_data_dir in self._qmugs_archive_maps:
            return self._qmugs_archive_maps[qmugs_data_dir]

        # old pickled datasets need tarball_assignment.csv to infer paths
        tarball_path = os.path.join(qmugs_data_dir, 'tarball_assignment.csv')
        if not os.path.isfile(tarball_path):
            raise FileNotFoundError(
                f"Cannot infer wfn_path because tarball_assignment.csv was not found: {tarball_path}"
            )

        # map CHEMBL ID to wavefunction archive directory
        tarball_assignment = pd.read_csv(tarball_path)
        archive_map = {}
        for _, row in tarball_assignment.iterrows():
            archive_map[row['chembl_id']] = row['archive_name'].split('.')[0]

        self._qmugs_archive_maps[qmugs_data_dir] = archive_map
        return archive_map


    def _resolve_wfn_path(self, batch, batch_index, qmugs_data_dir: str = None):
        '''return the Psi4 wavefunction path for one batched conformer'''

        # use stored path for newly preprocessed datasets
        if hasattr(batch, 'wfn_path'):
            wfn_path = self._get_batch_attr_value(
                batch=batch,
                name='wfn_path',
                batch_index=batch_index,
            )
            if wfn_path is not None and str(wfn_path) != '':
                return str(wfn_path)

        # fall back to QMugs directory conventions for old datasets
        if qmugs_data_dir is None:
            raise ValueError(
                "Cannot infer wfn_path for an old pickle dataset without qmugs_data_dir"
            )

        # identify molecule and conformer in the batch
        chembl_id = str(self._get_batch_attr_value(
            batch=batch,
            name='chembl_id',
            batch_index=batch_index,
        ))
        conformer_id = str(self._get_batch_attr_value(
            batch=batch,
            name='conformer_id',
            batch_index=batch_index,
        )).zfill(2)

        # construct canonical wavefunction file path
        archive_map = self._load_qmugs_archive_map(qmugs_data_dir=qmugs_data_dir)
        if chembl_id not in archive_map:
            raise KeyError(f"CHEMBL ID {chembl_id} not found in tarball_assignment.csv")

        return os.path.join(
            qmugs_data_dir,
            'wfns',
            archive_map[chembl_id],
            chembl_id,
            f'wfn_conf_{conformer_id}.npy',
        )


# ----------------------------------------------------------------------------------------------------
# wrappers for per matrix reconstruction metrics

def _as_numpy_density_matrix(mat):
    '''convert a torch tensor or numpy array density matrix to a CPU numpy array'''
    if torch.is_tensor(mat):
        return mat.detach().cpu().numpy()
    return np.asarray(mat)


def _loss_value(pred_mat, true_mat, criterion):
    '''wrapper for per matrix scalar reconstruction metric'''
    loss_function = loss_function_selection(criterion)
    loss = loss_function(pred_mat, true_mat)
    return float(loss.detach().cpu().item() if torch.is_tensor(loss) else loss)


def _trace_error(pred_mat, true_mat, eps=1e-12):
    '''Absolute relative trace error for one matrix pair'''
    pred_np = _as_numpy_density_matrix(pred_mat)
    true_np = _as_numpy_density_matrix(true_mat)

    trace_pred = float(np.trace(pred_np))
    trace_true = float(np.trace(true_np))
    value = abs((trace_pred - trace_true) / (trace_true + eps))

    return value, trace_pred, trace_true


def _hermitian_error(pred_mat, eps=1e-12):
    '''relative Frobenius-norm hermitianity violation'''
    pred_np = _as_numpy_density_matrix(pred_mat)

    numerator = np.linalg.norm(pred_np - pred_np.T, ord="fro")
    denominator = np.linalg.norm(pred_np, ord="fro") + eps

    return float(numerator / denominator)


def _psd_metrics(pred_mat):
    '''minimum eigenvalue and number of negative eigenvalues of the symmetrized prediction'''
    pred_np = _as_numpy_density_matrix(pred_mat)

    sym_pred = 0.5 * (pred_np + pred_np.T)
    eigvals = np.linalg.eigvalsh(sym_pred)

    min_eigenvalue = float(np.min(eigvals))
    negative_eigenvalue_count = int(np.sum(eigvals < 0.0))

    return min_eigenvalue, negative_eigenvalue_count


def _r2_per_matrix(pred_mat, true_mat, eps=1e-12):
    '''per matrix R2 score'''
    pred_np = _as_numpy_density_matrix(pred_mat)
    true_np = _as_numpy_density_matrix(true_mat)

    pred_flat = pred_np.ravel()
    true_flat = true_np.ravel()

    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)

    return float(1.0 - ss_res / (ss_tot + eps))


def _relative_frobenius_error(pred_mat, true_mat, eps=1e-12):
    '''relative frobenius error'''
    pred_np = _as_numpy_density_matrix(pred_mat)
    true_np = _as_numpy_density_matrix(true_mat)

    numerator = np.linalg.norm(pred_np - true_np, ord="fro")
    denominator = np.linalg.norm(true_np, ord="fro") + eps

    return float(numerator / denominator)


def _cosine_similarity(pred_mat, true_mat, eps=1e-12):
    '''cosine similarity between flattened matrices'''
    pred_np = _as_numpy_density_matrix(pred_mat)
    true_np = _as_numpy_density_matrix(true_mat)

    pred_flat = pred_np.ravel()
    true_flat = true_np.ravel()

    numerator = np.dot(pred_flat, true_flat)
    denominator = np.linalg.norm(pred_flat) * np.linalg.norm(true_flat) + eps

    return float(numerator / denominator)

# ----------------------------------------------------------------------------------------------------
