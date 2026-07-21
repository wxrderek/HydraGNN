import os, json
import logging
import time
import math

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

SAD_REFERENCE_UNPAIRED_ELECTRONS = [
    0,
    1, 0,
    1, 0, 1, 2, 3, 2, 1, 0,
    1, 0, 1, 2, 3, 2, 1, 0,
    1, 0, 1, 2, 3, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0,
    1, 0, 1, 2, 5, 6, 5, 4, 3, 0, 1, 0, 1, 2, 3, 2, 1, 0,
    1, 0, 1, 0, 3, 4, 5, 6, 7, 8, 5, 4, 3, 2, 1, 0,
    1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0,
]
_PROMOLECULAR_ATOMIC_DENSITY_CACHE = {}

# ----------------------------------------------------------------------------------------------------
# density matrix processing

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


# ----------------------------------------------------------------------------------------------------
# upper triangle packing/unpacking (Hermitian constraint)

def triangular_number(n: int):
    '''return the number of upper-triangle entries (including diagonal) of an n x n matrix'''
    return n * (n + 1) // 2


def triangular_side(length: int):
    '''return n such that n(n+1)/2 == length, raising if length is not a triangular number'''
    n = (math.isqrt(8 * length + 1) - 1) // 2
    if triangular_number(n) != length:
        raise ValueError(f"Length {length} is not a triangular number")
    return n


def is_triangular_number(length: int):
    '''return whether length equals n(n+1)/2 for some non-negative integer n'''
    n = (math.isqrt(8 * length + 1) - 1) // 2
    return triangular_number(n) == length


def physical_upper_triangle_indices(dim: int, max_size: int, triu_indices=None):
    '''return indices into a padded upper-triangle vector selecting the top-left dim x dim block'''
    # np.triu_indices is O(dim**2); callers in hot loops pass a precomputed pair instead
    if triu_indices is None:
        triu_indices = np.triu_indices(dim)
    rows, cols = triu_indices # row-major physical (i, j), 0 <= i <= j < dim
    # position of padded entry (i, j) in the row-major upper triangle of a max_size matrix
    offsets = rows * max_size - (rows * (rows - 1)) // 2 + (cols - rows)
    return offsets # (dim(dim+1)/2,)


def upper_triangle_vector(density_matrix: np.ndarray, max_size: int, triu_indices=None):
    '''scatter the row-major upper triangle of density_matrix into the padded T(max_size) vector'''
    n = density_matrix.shape[0]
    if n > max_size:
        raise ValueError(f"Density matrix original dimension of {n} greater than padding size {max_size}")

    if triu_indices is None:
        triu_indices = np.triu_indices(n)
    rows, cols = triu_indices

    # write only the T(n) physical entries; the padded square is never materialized
    offsets = physical_upper_triangle_indices(
        dim=n,
        max_size=max_size,
        triu_indices=triu_indices,
    ) # (T(n),)
    padded_vector = np.zeros(triangular_number(max_size), dtype=density_matrix.dtype) # (T(max_size),)
    padded_vector[offsets] = density_matrix[rows, cols]

    return padded_vector # (T(max_size),)


def upper_triangle_mask_vector(mask: np.ndarray, triu_indices=None):
    '''return the row-major upper triangle (with diagonal) of a square boolean mask'''
    if triu_indices is None:
        triu_indices = np.triu_indices(mask.shape[0])
    rows, cols = triu_indices
    return mask[rows, cols] # (n(n+1)/2,)


def symmetric_matrix_from_upper_triangle_vector(upper_triangle: np.ndarray, size: int):
    '''reconstruct the full symmetric matrix from its row-major upper triangle vector'''
    rows, cols = np.triu_indices(size)
    if upper_triangle.shape[0] != len(rows):
        raise ValueError(
            f"Upper triangle vector has {upper_triangle.shape[0]} entries, "
            f"expected {len(rows)} for size {size}"
        )
    matrix = np.zeros((size, size), dtype=upper_triangle.dtype) # (size, size)
    matrix[rows, cols] = upper_triangle
    # mirror the strict upper triangle onto the lower triangle to enforce symmetry
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    return matrix

# ----------------------------------------------------------------------------------------------------
# promolecular density matrix computation (baseline for delta learning)

def promolecular_density_matrix(wfn, basis_name: str = "def2-SVP"):
    '''return the Psi4 SAD total density matrix for one wavefunction'''

    # the delta path is built from stored Psi4 wavefunctions, not bare molecules
    if not hasattr(wfn, 'Da') or not hasattr(wfn, 'Db'):
        raise TypeError("promolecular_density_matrix expects a psi4.core.Wavefunction")

    # reference density fixes the required AO ordering and matrix shape
    reference_density = wfn.Da().np + wfn.Db().np
    reference_density = np.asarray(reference_density, dtype=np.float64) # (n_basis, n_basis)
    if reference_density.ndim != 2 or reference_density.shape[0] != reference_density.shape[1]:
        raise ValueError(f"Reference density matrix must be square, got {reference_density.shape}")

    # try direct Psi4 SAD interfaces before constructing atomic densities ourselves
    sad_density = _promolecular_density_matrix_from_sadguess(wfn)
    if sad_density is None:
        sad_density = _promolecular_density_matrix_from_guess_wfn(wfn)
    if sad_density is None:
        sad_density = _promolecular_density_matrix_from_atomic_scfs(
            wfn=wfn,
            basis_name=basis_name,
        )

    if sad_density is None:
        raise NotImplementedError(
            "Psi4 SAD density extraction is not available from this Python build. "
            "No SCF calculation was run."
        )

    # require exact agreement with the stored AO matrix shape
    sad_density = np.asarray(sad_density, dtype=np.float64) # (n_basis, n_basis)
    if sad_density.shape != reference_density.shape:
        raise ValueError(
            "Promolecular density matrix shape does not match the stored AO density: "
            f"{sad_density.shape} != {reference_density.shape}"
        )

    return sad_density


def _promolecular_density_matrix_from_atomic_scfs(wfn, basis_name: str = "def2-SVP"):
    '''assemble a promolecular density from neutral atomic UHF densities'''

    psi4_mol = wfn.molecule()
    basis = wfn.basisset()
    n_basis = int(basis.nbf())
    atom_to_aos = {}
    ao_start = 0

    # build AO index blocks for each atom in Psi4 canonical basis ordering
    for shell_index in range(basis.nshell()):
        shell = basis.shell(shell_index)
        atom_index = int(shell.ncenter)
        n_functions = int(shell.nfunction)
        atom_to_aos.setdefault(atom_index, [])
        atom_to_aos[atom_index].extend(range(ao_start, ao_start + n_functions))
        ao_start += n_functions
    promolecular_density = np.zeros((n_basis, n_basis), dtype=np.float64) # (n_basis, n_basis)

    # fill only atom-centered AO blocks, matching the Psi4 SAD block-diagonal construction
    for atom_index in range(int(psi4_mol.natom())):
        atomic_number = int(psi4_mol.ftrue_atomic_number(atom_index))
        atomic_symbol = psi4_mol.symbol(atom_index)
        ao_indices = atom_to_aos[atom_index]
        atomic_density = _neutral_atomic_density_matrix(
            atomic_number=atomic_number,
            atomic_symbol=atomic_symbol,
            basis_name=basis_name,
            n_basis_atom=len(ao_indices),
        ) # (n_basis_atom, n_basis_atom)

        if atomic_density.shape != (len(ao_indices), len(ao_indices)):
            raise ValueError(
                f"Atomic density for {atomic_symbol} has shape {atomic_density.shape}, "
                f"expected {(len(ao_indices), len(ao_indices))}"
            )

        promolecular_density[np.ix_(ao_indices, ao_indices)] = atomic_density

    return promolecular_density


def _neutral_atomic_density_matrix(
    atomic_number: int,
    atomic_symbol: str,
    basis_name: str,
    n_basis_atom: int,
):
    '''return the cached neutral atomic UHF total density matrix'''

    # Psi4 SAD uses high-spin neutral atoms determined by Hund's rules
    atomic_number = int(atomic_number)
    if atomic_number <= 0 or atomic_number >= len(SAD_REFERENCE_UNPAIRED_ELECTRONS):
        raise ValueError(f"SAD atomic occupations are not available for Z={atomic_number}")

    multiplicity = int(SAD_REFERENCE_UNPAIRED_ELECTRONS[atomic_number]) + 1

    # cache by element, basis, AO block size, and multiplicity within each Python process
    cache_key = (
        str(atomic_symbol),
        atomic_number,
        str(basis_name).lower(),
        int(n_basis_atom),
        int(multiplicity),
    )
    if cache_key in _PROMOLECULAR_ATOMIC_DENSITY_CACHE:
        return _PROMOLECULAR_ATOMIC_DENSITY_CACHE[cache_key].copy()

    # isolated neutral atom at the origin; no molecular orientation convention is needed
    atomic_geometry = f"""
0 {multiplicity}
{atomic_symbol} 0.0 0.0 0.0
symmetry c1
no_com
no_reorient
"""
    atomic_mol = psi4.geometry(atomic_geometry)

    # this is an atomic UHF calculation, not a molecular SCF cycle
    psi4.set_options({
        "basis": basis_name,
        "reference": "UHF",
        "guess": "CORE",
    })
    _, atomic_wfn = psi4.energy(
        "hf",
        molecule=atomic_mol,
        return_wfn=True,
    )

    # total alpha plus beta atomic density in the atom's AO basis
    atomic_density = _total_density_from_psi4_matrices(
        da_matrix=atomic_wfn.Da(),
        db_matrix=atomic_wfn.Db(),
    )
    atomic_density = np.asarray(atomic_density, dtype=np.float64) # (n_basis_atom, n_basis_atom)
    if atomic_density.shape != (n_basis_atom, n_basis_atom):
        raise ValueError(
            f"Atomic wavefunction for {atomic_symbol} has density shape {atomic_density.shape}, "
            f"expected {(n_basis_atom, n_basis_atom)}"
        )

    _PROMOLECULAR_ATOMIC_DENSITY_CACHE[cache_key] = atomic_density.copy()
    return atomic_density


def _promolecular_density_matrix_from_sadguess(wfn):
    '''try direct Psi4 SADGuess extraction without molecular SCF'''

    # most Psi4 Python builds do not expose this internal C++ class
    if not hasattr(psi4.core, 'SADGuess'):
        return None

    basis = wfn.basisset()
    atomic_bases = None

    # use any exposed atomic SAD basis list if the wavefunction provides one
    for attr_name in ['sad_basissets', 'sad_basis_sets', 'atomic_basissets']:
        if hasattr(wfn, attr_name):
            atomic_bases = getattr(wfn, attr_name)()
            if atomic_bases is not None:
                break
    if atomic_bases is None:
        return None

    options = getattr(psi4.core, 'get_options', lambda: None)()
    if options is None:
        return None

    # direct route mirrors Psi4's internal SADGuess::compute_guess()
    sad_guess = psi4.core.SADGuess(basis, atomic_bases, options)
    if hasattr(wfn, 'sad_fitting_basissets'):
        sad_fit_bases = wfn.sad_fitting_basissets()
        if sad_fit_bases is not None:
            sad_guess.set_atomic_fit_bases(sad_fit_bases)

    sad_guess.compute_guess()
    return _total_density_from_psi4_matrices(
        da_matrix=sad_guess.Da(),
        db_matrix=sad_guess.Db(),
    )


def _promolecular_density_matrix_from_guess_wfn(wfn):
    '''try a Psi4 guess-only wavefunction path without calling energy'''

    # clone when possible so a guess build cannot overwrite the loaded reference wavefunction
    guess_wfn = wfn
    if hasattr(wfn, 'clone'):
        guess_wfn = wfn.clone()

    # some Psi4 builds expose HF::compute_SAD_guess on wavefunction-like objects
    if not hasattr(guess_wfn, 'compute_SAD_guess'):
        return None

    guess_wfn.compute_SAD_guess(False)
    return _total_density_from_psi4_matrices(
        da_matrix=guess_wfn.Da(),
        db_matrix=guess_wfn.Db(),
    )


def _total_density_from_psi4_matrices(da_matrix, db_matrix):
    '''convert Psi4 alpha and beta matrices to one total NumPy matrix'''

    # SAD stores alpha and beta pieces separately even when they are identical
    da = _psi4_matrix_to_array(da_matrix).astype(np.float64, copy=False)
    db = _psi4_matrix_to_array(db_matrix).astype(np.float64, copy=False)
    if da.shape != db.shape:
        raise ValueError(f"SAD alpha and beta density shapes differ: {da.shape} != {db.shape}")

    # total 1-RDM used everywhere else in the QMugs density-matrix path
    return da + db


def _psi4_matrix_to_array(matrix):
    '''convert one Psi4 matrix to a dense NumPy array'''

    # prefer Psi4's zero-copy NumPy view when it is available
    if hasattr(matrix, 'np'):
        try:
            return np.asarray(matrix.np)
        except Exception:
            pass
    if hasattr(matrix, 'nph'):
        # irrepped Psi4 matrices expose one NumPy block per irrep
        return scipy.linalg.block_diag(*[np.asarray(block) for block in matrix.nph])
    if hasattr(matrix, 'to_array'):
        return np.asarray(matrix.to_array())
    if hasattr(matrix, 'to_np_array'):
        return np.asarray(matrix.to_np_array())

    return np.asarray(matrix)

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
