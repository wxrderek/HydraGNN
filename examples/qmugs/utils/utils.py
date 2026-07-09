import os, json
import logging
import time

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem

try:
    import psi4
    psi4.core.be_quiet()
except ImportError:
    print('[WARNING] psi4 package not detected')

NUM_MOLECULES = 665911
NUM_CONFORMERS = 1992984

BOHR_TO_ANGSTROM = 0.52917721092

# ----------------------------------------------------------------------------------------------------
# density matrix preprocessing

def pad_density_matrix(density_matrix: np.ndarray, max_size: int):
    '''pads density matrix with 0's to match max_size'''
    n = density_matrix.shape[0]
    if n > max_size:
        raise ValueError(f"Density matrix original dimension of {n} greater than padding size {max_size}")
    
    padded = np.zeros((max_size, max_size), dtype=density_matrix.dtype)
    padded[:n, :n] = density_matrix

    return padded


def unpad_density_matrix(padded_density_matrix: np.ndarray, original_size: int):
    '''unpads previously padded density matrix to original_size'''
    N = padded_density_matrix.shape[0]
    if N < original_size:
        print(f"[WARNING] Padded density matrix of size {N} is smaller than original size {original_size}")
        return padded_density_matrix

    unpadded = padded_density_matrix[:original_size, :original_size]
    return unpadded


def get_density_matrix_upper_tr(density_matrix: np.ndarray):
    '''returns the upper triangle of a density matrix'''
    pass

def get_density_matrix_from_upper_tr(density_matrix_upper_tr: np.ndarray):
    '''given the upper triangle of a density matrix, mirrors it and returns the full matrix'''
    pass


# ----------------------------------------------------------------------------------------------------
# density matrix postprocessing and downstream analysis



# ----------------------------------------------------------------------------------------------------
# masking

def get_density_matrix_mask(
    wfn: psi4.core.Wavefunction,
    rng, # np.random.default_rng(random_state)
    mask_method: str = None,
    debug: bool = False,
    **kwargs,
):
    '''
    Returns binary mask for density matrix

    Arguments
    ---------
    mask_method : str
        This specifies the method of masking. Options are:
        - 'unif_entries': uniformly masks out a specified percentage of entries
        - 'unif_atoms': uniformly masks out a specified percentage of blocks corresponding to atom pairs

    **kwargs
    --------
    perc_entries_masked : float
        If mask_method = 'unif_entries', this is the percentage of entries masked out
    perc_atom_pairs_masked : float
        If mask_method = 'unif_atoms', this is the percentage of atom pair blocks masked out
    '''

    # extract density matrix
    Da = wfn.Da().np
    n = Da.shape[0]

    # sample upper triangle only
    upper = np.triu_indices(n)
    n_entries_upper = len(upper[0])
    mask = np.ones((n, n), dtype=bool)

    # store atom masking results
    atoms_masked_dict = {}

    # randomly mask out blocks corresponding to certain atom pairs
    if mask_method == 'unif_atoms':
        perc_atom_pairs_masked = kwargs.get('perc_atom_pairs_masked', 0.2)
        basis = wfn.basisset()
        natoms = int(wfn.molecule().natom())
        atomic_symbols = [wfn.molecule().symbol(i) for i in range(natoms)]

        if debug:
            print(atomic_symbols)

        # build AO indices for each atom
        atom_to_aos = {A: [] for A in range(natoms)}
        ao_start = 0
        for ish in range(basis.nshell()):
            shell = basis.shell(ish)
            atom = shell.ncenter
            nfunc = shell.nfunction

            atom_to_aos[atom].extend(range(ao_start, ao_start + nfunc))
            ao_start += nfunc
        
        # sample atom pairs
        atom_pairs = [(i, j) for i in range(natoms) for j in range(i, natoms)]
        n_pairs = len(atom_pairs)
        n_pairs_masked = int(round(perc_atom_pairs_masked * n_pairs))

        if n_pairs_masked > 0:
            pair_idx = rng.choice(n_pairs, size=n_pairs_masked, replace=False)

            for idx in pair_idx:
                ai, aj = atom_pairs[idx]

                # store (atom idx, atom idx) = (atom symbol, atom symbol)
                atoms_masked_dict[(ai, aj)] = (
                    atomic_symbols[ai],
                    atomic_symbols[aj],
                )

                aos_i = atom_to_aos[ai]
                aos_j = atom_to_aos[aj]

                if debug:
                    print(f'(i) aos at atom {ai} which is {atomic_symbols[ai]}: {aos_i}')
                    print(f'(j) aos at atom {aj} which is {atomic_symbols[aj]}: {aos_j}')

                # mask block
                mask[np.ix_(aos_i, aos_j)] = False

                # enforce symmetry
                if ai != aj:
                    mask[np.ix_(aos_j, aos_i)] = False

    # randomly mask entries
    elif mask_method == 'unif_entries':
        perc_entries_masked = kwargs.get('perc_entries_masked', 0.2)
        n_entries_masked = int(round(perc_entries_masked * n_entries_upper))

        # sample n_entries_masked entries to mask
        upper_mask = np.ones(n_entries_upper, dtype=bool)
        if n_entries_masked > 0:
            mask_idx = rng.choice(n_entries_upper, size=n_entries_masked, replace=False)
            upper_mask[mask_idx] = False
        mask[upper] = upper_mask
        
        # mirror to lower triangle
        mask[(upper[1], upper[0])] = upper_mask
    
    # ...nothing else implemented
    elif mask_method is None:
        raise ValueError("Please specify a masking method, either 'unif_entries' or 'unif_atoms'")

    else:
        raise NotImplementedError
    
    # ensure mask is symmetric
    assert np.array_equal(mask, mask.T)

    # ensure mask is right dimensions
    assert Da.shape == mask.shape

    return mask, atoms_masked_dict


def plot_density_matrix_mask(
    wfn,
    mask,
    out_name,
):
    plt.figure(figsize=(8, 8))
    plt.imshow((wfn.Da().np + wfn.Db().np), cmap="RdBu_r", origin="lower", interpolation="none")
    plt.colorbar(label="Density Matrix")
    plt.xlabel("AO index")
    plt.ylabel("AO index")

    # plot mask as overlay
    overlay = np.where(mask, np.nan, 1.0) * 0.5
    plt.imshow(
        overlay,
        cmap="hot",
        origin="lower",
        interpolation="none",
        alpha=0.2,
    )
    plt.tight_layout()
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------------------------------
# grid building

def build_xyz_grid_by_spacing(
    center = (0.0, 0.0, 0.0), # (x, y, z), Angstrom by default
    box_size = 30.0, # Angstrom by default
    spacing = 0.5, # Angstrom by default
    input_units: str = 'Angstrom', # 'Angstrom' or 'Bohr'
    output_units: str = 'Angstrom', # 'Angstrom' or 'Bohr'
):
    '''
    builds (N, 3) dim array of grid points to express scalar field quantities, in Angstrom units by default, covering a square box.
    uses a defined box size and spatial resolution, NOT a fixed number of grid points
    '''

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


def build_xyz_grid_by_num_points(
    center = (0.0, 0.0, 0.0), # (x, y, z), Angstrom by default
    box_size = 30.0, # Angstrom by default
    num_grid_points: int = 100,
    input_units: str = 'Angstrom', # 'Angstrom' or 'Bohr'
    output_units: str = 'Angstrom', # 'Angstrom' or 'Bohr'
):
    '''
    builds (N, 3) dim array of grid points to express scalar field quantities, in Angstrom units by default, covering a square box.
    uses a defined box size and number of grid points, NOT a fixed spacing between grid points
    '''

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
    spacing = box_size / num_grid_points
    half = box_size / 2

    x = np.linspace(
        center[0] - half + spacing / 2,
        center[0] + half - spacing / 2,
        num_grid_points,
    )

    y = np.linspace(
        center[1] - half + spacing / 2,
        center[1] + half - spacing / 2,
        num_grid_points,
    )

    z = np.linspace(
        center[2] - half + spacing / 2,
        center[2] + half - spacing / 2,
        num_grid_points,
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
# distance measures between density matrices (UNUSED RIGHT NOW)

def frobenius_dist(D1, D2, S1, S2, normalize=True):
    '''frobenius norm induced distance'''
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
# direct utils testing

if __name__ == "__main__":

    dirpwd = os.path.dirname(os.path.abspath(__file__))

    wfn = psi4.core.Wavefunction.from_file('/pscratch/sd/w/wxrderek/qmugs/wfns/wfns_15/CHEMBL1592087/wfn_conf_00.npy')
    rng = np.random.default_rng(0)

    mask, atoms_masked_dict = get_density_matrix_mask(
        wfn=wfn,
        rng=rng,
        mask_method='unif_atoms',
        perc_entries_masked=0.2,
        perc_atom_pairs_masked=0.1,
    )
    print(mask.shape)

    print(atoms_masked_dict)
    plot_density_matrix_mask(
        wfn=wfn,
        mask=mask,
        out_name=os.path.join(dirpwd, 'densmat.png')
    )

    mask_padded = pad_density_matrix(density_matrix=mask, max_size=2002)
    print(mask_padded.shape)