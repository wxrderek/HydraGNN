import os, json
import logging
import sys
from mpi4py import MPI
import argparse
import numpy as np
import pandas as pd

import re
from rdkit import Chem


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


def _canon_CHEMBL_id(id: str):
    '''ensure the prefix CHEMBL is attached to the inputted CHEMBL_id'''
    if id.isdigit():
        return f'CHEMBL{id}'
    if re.fullmatch(r'CHEMBL\d+', id):
        return id
    
    raise ValueError(
        f'Invalid input format for CHEMBL ID: {id}. Must input either all digits, or start with string CHEMBL'
    )


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
                structure_props = next(Chem.SDMolSupplier(self.paths[f'structures_conf_{conf}'], removeHs=False)).GetPropsAsDict()
                conf_props.update({'structure': structure_props})

            # from vibspectra
            if self.load_vibspectra:
                vibspectrum_props = []
                with open(self.paths[f'vibspectrum_conf_{conf}'], 'r') as f:
                    
                    for line in f:
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
        try:
            return self.props['conf_00']['summary']['smiles']
        except KeyError:
            print('SMILES data not loaded in object. Declare object with load_summary=True to load.\nIf load_minimal=True, ensure "smiles" is added to MINIMAL_SUMMARY_INFO.')
            return ''
    

    def draw(self, outpath: str):
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


def inspect_wfn(path):
    wfn = np.load(path, allow_pickle=True).tolist()
    print(wfn)

def inspect_summary(path):
    summary = pd.read_csv(path)
    print(summary.head(10)) 



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    args = parser.parse_args()

    if args.inspect_wfn:
        inspect_wfn(args.inspect_wfn)
    if args.inspect_summary:
        inspect_summary(args.inspect_summary)
    
    with open('utils/download_data.json', 'r') as f:
        download_config = json.load(f)


    mol_test = QMugsMolecule(
        CHEMBL_id='1000', 
        download_config=download_config,
        load_conf_00_only=True,
    )

    # print(mol_test.props)
    # print(mol_test.smiles())
    # mol_test.draw(outpath='mol_test.png')

    # import psi4
    # wfn_test = mol_test.psi4_wfn('00')
    # wfn_test_1 = mol_test.psi4_wfn('01')

    # psi4.set_options({
    #     'cubeprop_tasks': ['density'],
    #     'cubic_grid_spacing': [0.2, 0.2, 0.2],
    #     'cubic_grid_overage': [4.0, 4.0, 4.0],
    #     'cubeprop_filepath': './cube_files'
    # })
    # # psi4.cubeprop(wfn_test)

    # Da_delta = wfn_test.Da().np - wfn_test_1.Da().np

    # print(np.linalg.norm(wfn_test.Da().np, ord='fro'))
    # print(np.linalg.norm(wfn_test_1.Da().np, ord='fro'))
    
    # print(np.linalg.norm(Da_delta, ord='fro'))
