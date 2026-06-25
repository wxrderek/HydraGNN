import os, json
import logging
import sys
from mpi4py import MPI
import argparse
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
from mpi4py import MPI

import re
from rdkit import Chem

try:
    import psi4
    psi4.core.be_quiet()
except ImportError:
    print('[WARNING] psi4 package not detected')

NUM_MOLECULES = 665911
NUM_CONFORMERS = 1992984


MINIMAL_SUMMARY_INFO = [
    'smiles', 
    'atoms',
    'nonunique_smiles',
    'GFN2_TOTAL_ENERGY',
    'DFT_TOTAL_ENERGY',
]

MINIMAL_WFN_MOLECULE_INFO = [
    'units',
    'geom',
    'elez',
    'elem',
    'mass',
]

MINIMAL_WFN_MATRIX_INFO = [
    'Da',
    'Db',
]

BOHR_TO_ANGSTROM = 0.52917721092


# ----------------------------------------------------------------------------------------------------
# checking across all data

def compile_molecule_dirs(download_config, outpath: str = None):
    '''compile the dirs for wfn files and sdf files in one dictionary'''

    # data download info
    data_dir = download_config.get('data_dir', 'dataset')
    wfns_dir = os.path.join(data_dir, download_config.get('wfns_subdir', 'wfns'))
    tarball_assignment = pd.read_csv(os.path.join(data_dir, 'tarball_assignment.csv'))

    # get dirs
    molecule_dirs = {} 
    for _, row in tarball_assignment.iterrows():
        chembl_id = row['chembl_id']
        wfn_dir = os.path.join(
            wfns_dir,
            row['archive_name'].split('.')[0],
            chembl_id,
        )
        sdf_dir = os.path.join(
            data_dir, 
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
        
        molecule_dirs[chembl_id] = {
            'wfn_dir': wfn_dir,
            'sdf_dir': sdf_dir,
        }

    if outpath:
        parent_dir = os.path.dirname(outpath)
        os.makedirs(parent_dir, exist_ok=True)
        with open(outpath, 'w', encoding='utf-8') as f:
            json.dump(molecule_dirs, f, indent=4)
    
    return molecule_dirs



def scan_density_matrices(download_config, scan_config, molecule_dirs = None, miniters = 1000):
    '''scan density matrix info for all downloaded data'''

    psi4.set_num_threads(1)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if rank == 0: 
        print('Scanning through density matrices...')

    # data download info
    data_dir = download_config.get('data_dir', 'dataset')
    wfns_dir = os.path.join(data_dir, download_config.get('wfns_subdir', 'wfns'))
    tarball_assignment = pd.read_csv(os.path.join(data_dir, 'tarball_assignment.csv'))

    local_max_size = 0
    local_info = {
        'sizes': {}, # per molecule
        'smallest_entry': {}, # per conformer
        'frobenius_norms': {}, # per conformer
        'conformer_diff_frobenius_norms': {}, # per molecule
    }

    # run through all wfn files
    if molecule_dirs:
        # NOT IMPLEMENTED
        raise NotImplementedError

    else:
        # get directories from tarball_assignment.csv
        for idx, (_, row) in enumerate(tarball_assignment.iterrows()):

            # simple round-robin assignment
            if idx % comm_size != rank:
                continue

            chembl_id = row['chembl_id']

            wfn_dir = os.path.join(
                wfns_dir,
                row['archive_name'].split('.')[0],
                chembl_id,
            )

            # warn if wfn dir does not exist
            if not os.path.exists(wfn_dir):
                print(f'[WARNING] The directory {wfn_dir} obtained from tarball_assignment.csv does not exist')
                continue

            # per molecule
            if scan_config['sizes']:
                conf_00_path = os.path.join(wfn_dir, 'wfn_conf_00.npy')

                if os.path.exists(conf_00_path):
                    psi4_wfn = psi4.core.Wavefunction.from_file(conf_00_path)

                    Da = psi4_wfn.Da().np
                    size = Da.shape[0]

                    local_info['sizes'][chembl_id] = size
                    local_max_size = max(local_max_size, size)

            # per conformer
            if (
                scan_config['smallest_entry']
                or scan_config['frobenius_norms']
                or scan_config['conformer_diff_frobenius_norms']
            ):
                for _, _, files in os.walk(wfn_dir):
                    for filename in files:
                        wfn_path = os.path.join(wfn_dir, filename)
                        psi4_wfn = psi4.core.Wavefunction.from_file(wfn_path)

                        Da = psi4_wfn.Da().np
                        Db = psi4_wfn.Db().np
                        D_tot = Da + Db

                        # NOT FINISHED


    gathered_max_sizes = comm.gather(local_max_size, root=0)
    gathered_infos = comm.gather(local_info, root=0)

    if rank != 0:
        return None, None

    max_size = max(gathered_max_sizes)

    info = {
        'sizes': {},
        'smallest_entry': {},
        'frobenius_norms': {},
        'conformer_diff_frobenius_norms': {},
    }

    for proc_info in gathered_infos:

        info['sizes'].update(
            proc_info['sizes']
        )

        info['smallest_entry'].update(
            proc_info['smallest_entry']
        )

        info['frobenius_norms'].update(
            proc_info['frobenius_norms']
        )

        info['conformer_diff_frobenius_norms'].update(
            proc_info['conformer_diff_frobenius_norms']
        )

    info['max_size'] = max_size
    return max_size, info

    #     with tqdm(total=NUM_MOLECULES, desc='Scanning molecules for EDA', miniters=miniters) as pbar:
    #         for _, row in tarball_assignment.iterrows():
    #             chembl_id = row['chembl_id']
    #             wfn_dir = os.path.join(
    #                 wfns_dir,
    #                 row['archive_name'].split('.')[0],
    #                 chembl_id,
    #             )

    #             # warn if wfn dir does not exist
    #             if not os.path.exists(wfn_dir):
    #                 print(f'[WARNING] The directory {wfn_dir} obtained from tarball_assignment.csv does not exist')
    #                 continue

    #             # scan through sizes of conformer 00 only
    #             if scan_config['sizes']:
    #                 conf_00_path = os.path.join(wfn_dir, 'wfn_conf_00.npy')
    #                 if not os.path.exists(conf_00_path):
    #                     print(f'[WARNING] The path {conf_00_path} obtained from tarball_assignment.csv does not exist')
    #                     continue

    #                 psi4_wfn = psi4.core.Wavefunction.from_file(conf_00_path)
    #                 Da = psi4_wfn.Da().np
    #                 size = Da.shape[0]
                    
    #                 info['sizes'][chembl_id] = size
    #                 if size > max_size:
    #                     max_size = size
                    
    #             # scan through all conformers of each wfn
    #             if (scan_config['smallest_entry'] or scan_config['frobenius_norms'] or scan_config['conformer_diff_frobenius_norms']):
    #                 # walk through wfn files
    #                 for _, _, files in os.walk(wfn_dir):
    #                     for filename in files:
    #                         wfn_path = os.path.join(wfn_dir, filename)
    #                         psi4_wfn = psi4.core.Wavefunction.from_file(wfn_path)

    #                         Da = psi4_wfn.Da().np
    #                         Db = psi4_wfn.Db().np
    #                         D_tot = Da + Db

    #                         # NOT FINISHED

    # info['max_size'] = max_size
    # return max_size, info



def scan_molecular_geometries(download_config, scan_config, molecule_dirs = None, miniters = 1000):
    '''scan sdf structure files for all downloaded data'''
    
    print('Scanning through molecule structures (sdfs)...')

    # data download info
    data_dir = download_config.get('data_dir', 'dataset')
    tarball_assignment = pd.read_csv(os.path.join(data_dir, 'tarball_assignment.csv'))

    max_box_size = 0
    info = {
        'box_sizes': {}, # per conformer, in Angstrom
        'max_avg_conformer_diff': {}, # per molecule, in Angstrom, max avg difference between coords in pairs of conformers
    }

    # run through all sdf files
    if molecule_dirs:
        # NOT IMPLEMENTED
        pass

    else:
        # get directories from tarball_assignment.csv
        for _, row in tarball_assignment.iterrows():
            chembl_id = row['chembl_id']
            sdf_dir = os.path.join(
                data_dir,
                'structures',
                chembl_id,
            )

            # warn if sdf dir does not exist
            if not os.path.exists(sdf_dir):
                print(f'[WARNING] The directory {sdf_dir} obtained from tarball_assignment.csv does not exist')
                continue
                
            # scan through all conformers of each molecule
            if (scan_config['box_sizes'] or scan_config['max_avg_conformer_diff']):
                # walk through sdf files
                for _, _, files in os.walk(sdf_dir):
                    for filename in files:
                        sdf_path = os.path.join(sdf_dir, filename)
                        sdf_mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]

                        # NOT FINISHED

    info['max_box_size'] = max_box_size
    return max_box_size, info


# ----------------------------------------------------------------------------------------------------

def _canon_CHEMBL_id(id: str):
    '''ensure the prefix CHEMBL is attached to the inputted CHEMBL_id'''
    if id.isdigit():
        return f'CHEMBL{id}'
    if re.fullmatch(r'CHEMBL\d+', id):
        return id
    
    raise ValueError(
        f'Invalid input format for CHEMBL ID: {id}. Must input either all digits, or start with string CHEMBL'
    )

# ----------------------------------------------------------------------------------------------------
# NEED TO ACCOUNT FOR MOLECULES WITH LESS THAN 3 CONFORMERS
# ----------------------------------------------------------------------------------------------------

class QMugsMolecule():
    '''molecule object for QMugs dataset for EDA purposes'''
    
    def __init__(
        self,
        CHEMBL_id: str,
        download_config: dict = None,
        load_minimal: bool = True,
        load_summary: bool = False,
        load_structures: bool = False,
        load_vibspectra: bool = False,
        load_wfns: bool = False,
        load_conf_00_only: bool = False,
    ):
        self.CHEMBL_id = _canon_CHEMBL_id(CHEMBL_id)
        self.download_config = download_config or {}
        self.data_dir = self.download_config.get('data_dir', 'dataset')
        self.structures_dir = os.path.join(self.data_dir, 'structures')
        self.vibspectra_dir = os.path.join(self.data_dir, 'vibspectra')
        self.wfns_dir = os.path.join(
            self.data_dir, 
            self.download_config.get('wfns_subdir', 'wfns')
        )

        self.load_minimal = load_minimal
        self.load_summary = load_summary
        self.load_structures = load_structures
        self.load_vibspectra = load_vibspectra
        self.load_wfns = load_wfns
        self.num_confs_to_load = 1 if load_conf_00_only else 3

        self.props = {}

        self._setup_paths()
        self._setup_props()


    def _setup_paths(self):
        '''store paths to data associated with this molecule'''
        self.paths = {}
        self.paths['summary.csv'] = os.path.join(self.data_dir, 'summary.csv')
        self.paths['tarball_assignment.csv'] = os.path.join(self.data_dir, 'tarball_assignment.csv')

        tarball_df = pd.read_csv(self.paths['tarball_assignment.csv'])
        wfn_tarball = tarball_df.loc[tarball_df['chembl_id'] == self.CHEMBL_id, 'archive_name'].iloc[0]
        wfn_dirname = wfn_tarball.split('.')[0]

        for i in range(self.num_confs_to_load):
            # per conformer files
            conf = str(i).zfill(2)
            self.paths[f'structures_conf_{conf}'] = os.path.join(
                self.structures_dir,
                self.CHEMBL_id,
                f'conf_{conf}.sdf'
            )
            self.paths[f'vibspectrum_conf_{conf}'] = os.path.join(
                self.vibspectra_dir, 
                self.CHEMBL_id, 
                f'vibspectrum_conf_{conf}'
            )
            self.paths[f'wfn_conf_{conf}'] = os.path.join(
                self.wfns_dir, 
                wfn_dirname, 
                self.CHEMBL_id,
                f'wfn_conf_{conf}.npy'
            )
    

    def _setup_props(self):
        '''read and store molecular properties per conformer in a friendly way'''

        for i in range(self.num_confs_to_load):
            conf = str(i).zfill(2)
            conf_props = {}

            # from summary.csv
            if self.load_summary:
                summary_df = pd.read_csv(self.paths['summary.csv'])
                mol_df = summary_df[summary_df['chembl_id'] == self.CHEMBL_id]
                conf_row = mol_df.loc[mol_df['conf_id'] == f'conf_{conf}'].iloc[0]

                if self.load_minimal:
                    summary_props = conf_row.loc[MINIMAL_SUMMARY_INFO].to_dict()
                else: 
                    summary_props = conf_row.loc['smiles':].to_dict()
                
                conf_props.update({'summary': summary_props})

            # from structures
            if self.load_structures:
                mol_object = Chem.SDMolSupplier(self.paths[f'structures_conf_{conf}'], removeHs=False)[0]
                
                # get coordinates and atomic numbers
                pos = mol_object.GetConformer().GetPositions().tolist()
                atomic_numbers = [atom.GetAtomicNum() for atom in mol_object.GetAtoms()]
                other_structure_props = next(Chem.SDMolSupplier(self.paths[f'structures_conf_{conf}'], removeHs=False)).GetPropsAsDict()

                conf_props.update({'structure': {
                    'geom': pos,
                    'atomic_numbers': atomic_numbers,
                    'other_props': other_structure_props,
                }})

            # from vibspectra
            if self.load_vibspectra:
                vibspectrum_props = []
                with open(self.paths[f'vibspectrum_conf_{conf}'], 'r') as f:
                    
                    for line in f:
                        # reg ex match the vibspectrum files
                        m = re.match(
                            r"\s*(\d+)\s+([a-zA-Z-]*)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(-|YES|NO)\s+(-|YES|NO)",
                            line
                        )
                        if m:
                            if self.load_minimal:
                                vibspectrum_props.append({
                                    'mode': int(m.group(1)),
                                    'wave_number': float(m.group(3)),
                                    'ir_intensity': float(m.group(4)),
                                })
                            else:
                                vibspectrum_props.append({
                                    'mode': int(m.group(1)),
                                    'symmetry': m.group(2),
                                    'wave_number': float(m.group(3)),
                                    'ir_intensity': float(m.group(4)),
                                    'ir_active': m.group(5),
                                    'raman_active': m.group(6),
                                })
                
                conf_props.update({'vibspectrum': vibspectrum_props})
            
            # from wfns
            if self.load_wfns:
                wfn = np.load(self.paths[f'wfn_conf_{conf}'], allow_pickle=True).tolist()
                wfn_props = {}

                if self.load_minimal:
                    wfn_props['molecule'] = {prop: value for prop, value in wfn['molecule'].items() if prop in MINIMAL_WFN_MOLECULE_INFO}
                    wfn_props['matrix'] = {prop: value for prop, value in wfn['matrix'].items() if prop in MINIMAL_WFN_MATRIX_INFO}
                else: 
                    wfn_props = wfn

                conf_props.update({'wfn': wfn_props})

            self.props[f'conf_{conf}'] = conf_props


    def smiles(self):
        '''return SMILES of molecule'''
        try:
            return self.props['conf_00']['summary']['smiles']
        except KeyError:
            print('SMILES data not loaded in object. Declare object with load_summary=True to load.\nIf load_minimal=True, ensure "smiles" is added to MINIMAL_SUMMARY_INFO.')
            return ''
    

    def draw(self, outpath: str):
        '''draw the skeletal structure of molecule'''
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(self.smiles())
        if mol is not None:
            img = Draw.MolToImage(mol, size=(500, 500), kekulize=True, wedgeBonds=True)
            img.save(outpath)
        else:
            print('Invalid SMILES string attached to molecule')


    def psi4_wfn(self, conf_idx: str):
        '''return psi4.core.Wavefunction object for the given conformer'''
        import psi4
        return psi4.core.Wavefunction.from_file(self.paths[f'wfn_conf_{conf_idx}'])


    def psi4_molecule(self, conf_idx: str):
        '''return psi4.core.Molecule object for the given conformer'''
        import psi4
        return self.psi4_wfn(conf_idx).molecule()

    
    def get_principle_axes(self, conf_idx: str):
        '''returns the spatial basis (normalized) given by the principle axes of inertia of the molecule'''
        import psi4
        mol = self.psi4_molecule(conf_idx).clone() # updates a clone of the Molecule object
        mol.update_geometry()
        mol.move_to_com()

        # compute principle axes and moments
        I = np.array(mol.inertia_tensor())
        moments, axes = np.linalg.eigh(I)
        ordered_idx = np.argsort(moments)[::-1]
        moments = moments[ordered_idx]
        axes = axes[:, ordered_idx]

        # ensure right handed frame
        if np.linalg.det(axes) < 0: axes[:, 0] *= -1
        return {
            'axes': (axes[:, 0], axes[:, 1], axes[:, 2]),
            'moments': moments
        }

# ----------------------------------------------------------------------------------------------------
# print file contents (early EDA)

def inspect_wfn(path, outpath: str = 'inspect_wfn.txt'):
    wfn = np.load(path, allow_pickle=True).tolist()
    with open(outpath, 'w', encoding='utf-8') as f:
        print(wfn, file=f)


def inspect_summary(path, output_file: str = 'inspect_summary.txt', n_head: int = 10):
    summary = pd.read_csv(path)
    with open(output_file, 'w', encoding='utf-8') as f:
        print(summary.head(n_head), file=f) 

# ----------------------------------------------------------------------------------------------------
# distance measures between density matrices

def frobenius(D1, D2, S1, S2, normalize=True):
    if normalize:
        D1 = D1 / np.trace(D1 @ S1)
        D2 = D2 / np.trace(D2 @ S2)
    D_delta = D1 - D2
    fro = np.sqrt(np.trace(D_delta @ D_delta))

    return float(np.real(fro))


def fidelity(D1, D2, S1, S2, normalize=True):
    '''fidelity in a quantum information sense, note this is NOT symmetric with respect to swapping args'''
    if normalize:
        D1 = D1 / np.trace(D1 @ S1)
        D2 = D2 / np.trace(D2 @ S2)
    sqrt_D1 = scipy.linalg.sqrtm(D1)
    fidelity = np.trace(
        scipy.linalg.sqrtm(sqrt_D1 @ D2 @ sqrt_D1)
    )

    return np.real(fidelity)**2
# ----------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    dirpwd = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=dirpwd,
        help="directory to output EDA results if applicable"
    )
    parser.add_argument(
        "--inspect_wfn",
        type=str,
        default=None,
        help="look at wfn file from QMugs",
    )
    parser.add_argument(
        "--inspect_summary",
        type=str,
        default=None,
        help="look at summary.csv"
    )
    parser.add_argument(
        "--get_densmat_info",
        action="store_true",
        help="extract size and other info of all density matrices in the downloaded dataset"
    )
    parser.add_argument(
        "--get_molecular_geometry_info",
        action="store_true",
        help="extract box sizes in Angstrom and other info of all molecules in the dataset"
    )
    parser.add_argument(
        "--miniters",
        type=int,
        default=1000,
        help="miniters argument on tdqm for scanning"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------------------------------------------------------
    
    output_dir = args.output_dir if args.output_dir else dirpwd
    with open(os.path.join(dirpwd, 'utils/download_data.json'), 'r') as f:
        download_config = json.load(f)

    if args.inspect_wfn:
        inspect_wfn(path=args.inspect_wfn, outpath=os.path.join(output_dir, 'inspect_wfn.txt'))
    
    if args.inspect_summary:
        inspect_summary(path=args.inspect_summary, outpath=os.path.join(output_dir, 'inspect_summary.txt'), n_head=50)
    
    # ----------------------------------------------------------------------------------------------------
    
    densmat_scan_config = {
        'sizes': True,
        'smallest_entry': False,
        'frobenius_norms': False,
        'conformer_diff_frobenius_norms': False,
    }

    if args.get_densmat_info:
        max_matrix_size, matrix_info = scan_density_matrices(download_config=download_config, scan_config=densmat_scan_config, miniters=args.miniters)
        print(f'Max density matrix size detected: {max_matrix_size} by {max_matrix_size}')

        # save scan results
        eda_dir = os.path.join(output_dir, 'eda')
        os.makedirs(eda_dir, exist_ok=True)
        with open(os.path.join(eda_dir, 'densmat_eda.json'), 'w', encoding='utf-8') as f:
            json.dump(matrix_info, f, indent=4)
    
    geo_scan_config = {
        'box_sizes': True,
        'max_avg_conformer_diff': False,
    }

    if args.get_molecular_geometry_info:
        max_box_size, geometry_info = scan_molecular_geometries(download_config=download_config, scan_config=geo_scan_config, miniters=args.miniters)
        print(f'Max box sizes for molecular configurations: {max_box_size} by {max_box_size}')
    
    # ----------------------------------------------------------------------------------------------------


    # mol_test = QMugsMolecule(
    #     CHEMBL_id='1', 
    #     download_config=download_config,
    #     load_structures=True,
    #     load_conf_00_only=True,
    # )

    # geom_sdf = np.array(mol_test.props['conf_00']['structure']['geom'])
    # geom_psi4 = np.array(mol_test.psi4_molecule('00').geometry()) * BOHR_TO_ANGSTROM
    # max_diff = np.abs(geom_sdf - geom_psi4).max()
    # print(max_diff)

    # print(json.dumps(mol_test.props['conf_00']['structure'], indent=4))

    # print(mol_test.props)
    # print(mol_test.smiles())
    # mol_test.draw(outpath='mol_test.png')


    # wfn_test = mol_test.psi4_wfn('00')
    # wfn_test_1 = mol_test.psi4_wfn('01')

    # molecule_test = mol_test.psi4_molecule('00')
    # print(molecule_test.natom())
    # for i in range(3):
    #     inertia_info = mol_test.get_principle_axes(f'0{i}')
    #     axis = inertia_info['axes'][i]
    #     moment = inertia_info['moments'][i]
    #     print(axis)
    #     print(np.linalg.norm(axis))
    #     print(moment)

    # D0 = wfn_test.Da().np + wfn_test.Db().np
    # D1 = wfn_test_1.Da().np + wfn_test_1.Db().np
    # S0 = psi4.core.MintsHelper(wfn_test.basisset()).ao_overlap().to_array()
    # S1 = psi4.core.MintsHelper(wfn_test_1.basisset()).ao_overlap().to_array()

    # D_delta = D0 - D1

    # print(f'D0 Frobenius: {np.linalg.norm(D0 @ S0, ord='fro')}')
    # print(f'D1 Frobenius: {np.linalg.norm(D1 @ S1, ord='fro')}')

    # print(f'D0-D1 Frobenius: {np.linalg.norm(D_delta, ord='fro')}')
    # print(f'Frobenius normalized: {frobenius(D0, D1, S0, S1, normalize=True)}')
    # print(f'Fidelity measure: {fidelity(D0, D1, S0, S1, normalize=True)}')