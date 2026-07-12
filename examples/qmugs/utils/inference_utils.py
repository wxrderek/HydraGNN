import os
import numpy as np

import psi4
psi4.core.be_quiet()

BOHR_TO_ANGSTROM = 0.52917721092
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# ----------------------------------------------------------------------------------------------------
# density matrix processing

def psi4_matrix_to_array(mat):
    '''convert a Psi4 matrix-like object to a numpy array'''
    if hasattr(mat, 'np'):
        return np.asarray(mat.np)
    if hasattr(mat, 'to_array'):
        return np.asarray(mat.to_array())
    return np.asarray(mat)


# ----------------------------------------------------------------------------------------------------
# dipole moment from density matrix

def dipole_moment_from_density_matrix(density_matrix, psi4_wfn):
    '''compute one molecular dipole vector from a total AO density matrix'''

    # density matrix in AO basis
    density_matrix = np.asarray(density_matrix, dtype=np.float64) # (n_basis, n_basis)

    # ground truth Psi4 molecular geometry
    psi4_mol = psi4_wfn.molecule()
    positions_bohr = psi4_matrix_to_array(psi4_mol.geometry()).astype(np.float64) # (n_atoms, 3)
    atomic_numbers = np.asarray(
        [psi4_mol.ftrue_atomic_number(i) for i in range(psi4_mol.natom())],
        dtype=np.float64,
    ) # (n_atoms,)
    assert positions_bohr.shape == (psi4_mol.natom(), 3)
    assert atomic_numbers.shape == (psi4_mol.natom(),)

    # nuclear contribution to dipole moment
    nuclear_dipole = np.sum(
        atomic_numbers[:, None] * positions_bohr,
        axis=0,
    ) # (3,)

    # AO dipole integral matrices from stored wavefunction basis
    mints = psi4.core.MintsHelper(psi4_wfn.basisset())
    dipole_integrals = np.stack([
        psi4_matrix_to_array(dipole_component)
        for dipole_component in mints.ao_dipole()
    ]).astype(np.float64) # (3, n_basis, n_basis)

    n_basis = int(density_matrix.shape[0])
    assert density_matrix.shape == (n_basis, n_basis)
    assert dipole_integrals.shape == (3, n_basis, n_basis)

    # electronic contribution by contraction with total density matrix
    electronic_dipole = np.einsum(
        'mn,xmn->x',
        density_matrix,
        dipole_integrals,
    ) # (3,)

    # total dipole moment with current Psi4 integral sign convention
    total_dipole = nuclear_dipole + electronic_dipole # (3,)
    return total_dipole, nuclear_dipole, electronic_dipole


# ----------------------------------------------------------------------------------------------------
# electron density from density matrix

def electron_density_from_density_matrix(
    density_matrix,
    psi4_wfn,
    padding_angstrom: float = 3.0,
    spacing_angstrom=None,
    num_grid_points=None,
    block_size: int = 2000,
):
    '''compute electron density on a grid by direct AO-basis contraction'''

    density_matrix = np.asarray(density_matrix, dtype=np.float64) # (n_basis, n_basis)
    n_basis = int(psi4_wfn.basisset().nbf())
    assert density_matrix.shape == (n_basis, n_basis)
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    # construct rectangular grid around the molecular bounding box
    grid_points, metadata = _manual_density_grid(
        psi4_wfn=psi4_wfn,
        padding_angstrom=padding_angstrom,
        spacing_angstrom=spacing_angstrom,
        num_grid_points=num_grid_points,
    ) # (n_grid, 3)

    # evaluate AO functions and contract rho_g = phi_gm P_mn phi_gn in blocks
    density_values = np.empty(grid_points.shape[0], dtype=np.float64) # (n_grid,)
    basis = psi4_wfn.basisset()
    for start in range(0, grid_points.shape[0], block_size):
        stop = min(start + block_size, grid_points.shape[0])
        phi_block, function_map = _ao_values_on_grid_block(
            basis=basis,
            grid_points=grid_points[start:stop, :],
        ) # (n_block, n_local), (n_local,)

        # only contract AO functions nonzero on this grid block
        if len(function_map) == 0:
            density_values[start:stop] = 0.0
        else:
            density_block = density_matrix[np.ix_(function_map, function_map)] # (n_local, n_local)
            density_values[start:stop] = np.einsum(
                'gm,mn,gn->g',
                phi_block,
                density_block,
                phi_block,
            ) # (n_block,)

    density_values = density_values.reshape(metadata['grid_shape']) # (n_x, n_y, n_z)

    return density_values, metadata


# ----------------------------------------------------------------------------------------------------
# wavefunction object from density matrix

def wavefunction_from_total_density_matrix(
    reference_wfn,
    total_density_matrix: np.ndarray,
    wfn_path: str = None,
):
    '''build a Psi4 wavefunction with a supplied total density matrix'''

    total_density_matrix = np.asarray(total_density_matrix, dtype=np.float64) # (n_basis, n_basis)
    n_basis = int(reference_wfn.basisset().nbf())
    assert total_density_matrix.shape == (n_basis, n_basis)

    if wfn_path is None:
        raise ValueError("Building a predicted Psi4 wavefunction requires wfn_path")

    # load an independent copy so overwriting Da and Db leaves the true wavefunction unchanged
    pred_wfn = psi4.core.Wavefunction.from_file(wfn_path)
    assert int(pred_wfn.basisset().nbf()) == n_basis

    # declare alpha and beta density matrices from total density, assuming closed shells
    alpha_density = psi4.core.Matrix.from_array(0.5 * total_density_matrix) # (n_basis, n_basis)
    beta_density = psi4.core.Matrix.from_array(0.5 * total_density_matrix) # (n_basis, n_basis)
    if pred_wfn.Da() is None or pred_wfn.Db() is None:
        raise ValueError("Predicted Psi4 wavefunction copy does not contain Da and Db matrices")

    # copy in density matrices
    pred_wfn.Da().copy(alpha_density)
    pred_wfn.Db().copy(beta_density)

    # cubeprop reads Da_subset("AO") and Db_subset("AO"), not Da and Db directly
    pred_da = psi4_matrix_to_array(pred_wfn.Da()).astype(np.float64, copy=False) # (n_basis, n_basis)
    pred_db = psi4_matrix_to_array(pred_wfn.Db()).astype(np.float64, copy=False) # (n_basis, n_basis)
    pred_da_ao = psi4_matrix_to_array(pred_wfn.Da_subset("AO")).astype(np.float64, copy=False) # (n_basis, n_basis)
    pred_db_ao = psi4_matrix_to_array(pred_wfn.Db_subset("AO")).astype(np.float64, copy=False) # (n_basis, n_basis)
    alpha_np = psi4_matrix_to_array(alpha_density).astype(np.float64, copy=False) # (n_basis, n_basis)
    beta_np = psi4_matrix_to_array(beta_density).astype(np.float64, copy=False) # (n_basis, n_basis)

    assert pred_da.shape == (n_basis, n_basis)
    assert pred_db.shape == (n_basis, n_basis)
    assert pred_da_ao.shape == (n_basis, n_basis)
    assert pred_db_ao.shape == (n_basis, n_basis)

    if not np.allclose(pred_da, alpha_np):
        raise ValueError("Predicted Psi4 Da matrix was not overwritten")
    if not np.allclose(pred_db, beta_np):
        raise ValueError("Predicted Psi4 Db matrix was not overwritten")
    if not np.allclose(pred_da_ao, alpha_np):
        raise ValueError("Predicted Psi4 Da_subset('AO') does not match overwritten Da")
    if not np.allclose(pred_db_ao, beta_np):
        raise ValueError("Predicted Psi4 Db_subset('AO') does not match overwritten Db")

    return pred_wfn


# ----------------------------------------------------------------------------------------------------
# cubeprop writing and reading

def write_cubeprops(psi4_wfn, cube_dir: str, cubeprop_names, grid_config=None):
    '''write requested Psi4 cubeprop outputs for one wavefunction'''

    # normalize user-facing names to the internal property convention
    cubeprop_names = _normalize_cubeprop_names(cubeprop_names)
    _raise_for_unsupported_cubeprops(cubeprop_names)
    grid_options = _cubeprop_grid_options(
        psi4_wfn=psi4_wfn,
        grid_config=grid_config,
    )

    # each molecule gets its own directory because Psi4 writes fixed filenames
    cube_dir = os.path.abspath(cube_dir)
    os.makedirs(cube_dir, exist_ok=True)

    # configure Psi4 cubeprop tasks and output location
    cubeprop_options = {
        'cubeprop_tasks': [name.upper() for name in cubeprop_names],
        'cubeprop_filepath': cube_dir,
    }
    cubeprop_options.update(grid_options)
    psi4.set_options(cubeprop_options)

    # compute requested cubeprops from the supplied wavefunction
    psi4.cubeprop(psi4_wfn)


def read_cubeprops(cube_dir: str, cubeprop_names):
    '''read requested Psi4 cubeprop outputs from one directory'''

    # normalize user-facing names to the internal property convention
    cubeprop_names = _normalize_cubeprop_names(cubeprop_names)
    _raise_for_unsupported_cubeprops(cubeprop_names)

    # collect all cube files written by Psi4 in the requested directory
    cube_files = sorted([
        os.path.join(cube_dir, filename)
        for filename in os.listdir(cube_dir)
        if filename.endswith('.cube')
    ])
    if len(cube_files) == 0:
        raise FileNotFoundError(f"No cube files found in {cube_dir}")

    cube_by_name = {
        os.path.basename(cube_path).lower(): cube_path
        for cube_path in cube_files
    }

    # read each requested property into a small result dictionary
    cubeprops = {}
    for cubeprop_name in cubeprop_names:
        if cubeprop_name == 'density':
            values, metadata, paths = _read_density_cubeprop(cube_by_name, cube_files)
            cubeprops[cubeprop_name] = {
                'values': values,
                'metadata': metadata,
                'paths': paths,
            }

    return cubeprops


def read_cube_file(cube_path: str):
    '''read one Gaussian cube file into values and grid metadata'''

    # read full cube file because headers determine array shape
    with open(cube_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError(f"Cube file is too short: {cube_path}")

    atom_line = lines[2].split()
    n_atoms = abs(int(atom_line[0]))
    origin = np.asarray([float(x) for x in atom_line[1:4]], dtype=np.float64) # (3,)

    # parse grid dimensions and voxel axis vectors
    grid_shape = []
    axis_vectors = []
    for axis_line in lines[3:6]:
        fields = axis_line.split()
        grid_shape.append(abs(int(fields[0])))
        axis_vectors.append([float(x) for x in fields[1:4]])

    grid_shape = tuple(grid_shape)
    axis_vectors = np.asarray(axis_vectors, dtype=np.float64) # (3, 3)
    assert origin.shape == (3,)
    assert axis_vectors.shape == (3, 3)

    # parse flattened cube values after the atom block
    data_start = 6 + n_atoms
    values = []
    for line in lines[data_start:]:
        values.extend(float(x) for x in line.split())

    cube_values = np.asarray(values, dtype=np.float64) # (n_x * n_y * n_z,)
    expected_size = int(np.prod(grid_shape))
    if cube_values.size != expected_size:
        raise ValueError(
            f"Cube file {cube_path} has {cube_values.size} values, expected {expected_size}"
        )

    # reshape into the cube grid ordering stored by Psi4
    cube_values = cube_values.reshape(grid_shape) # (n_x, n_y, n_z)

    # store grid metadata needed for pointwise comparison and integration
    metadata = {
        'grid_shape': grid_shape,
        'origin': origin,
        'axis_vectors': axis_vectors,
        'voxel_volume': abs(float(np.linalg.det(axis_vectors))),
    }

    return cube_values, metadata


def assert_same_cube_grid(first_cube_metadata, second_cube_metadata):
    '''assert that two cube files use the same voxel grid'''

    # pointwise cube comparisons require identical grid dimensions
    if tuple(first_cube_metadata['grid_shape']) != tuple(second_cube_metadata['grid_shape']):
        raise ValueError(f"Cube grid shapes differ: {first_cube_metadata['grid_shape']} vs {second_cube_metadata['grid_shape']}")

    # values are compared pointwise, so origin and voxel axes must match
    if not np.allclose(first_cube_metadata['origin'], second_cube_metadata['origin']):
        raise ValueError("Cube grid origins differ")
    if not np.allclose(first_cube_metadata['axis_vectors'], second_cube_metadata['axis_vectors']):
        raise ValueError("Cube grid axis vectors differ")


# ----------------------------------------------------------------------------------------------------
# readers for specific cubeprop objects

def _read_density_cubeprop(cube_by_name, cube_files):
    '''read total density from Psi4 cubeprop outputs'''

    # Psi4 writes total density as Dt.cube
    if 'dt.cube' in cube_by_name:
        values, metadata = read_cube_file(cube_by_name['dt.cube'])
        return values, metadata, [cube_by_name['dt.cube']]

    # if only spin densities are written, recover total density as Da + Db
    if 'da.cube' in cube_by_name and 'db.cube' in cube_by_name:
        density_alpha, metadata_alpha = read_cube_file(cube_by_name['da.cube'])
        density_beta, metadata_beta = read_cube_file(cube_by_name['db.cube'])
        assert_same_cube_grid(metadata_alpha, metadata_beta)
        return (
            density_alpha + density_beta, # (n_x, n_y, n_z)
            metadata_alpha,
            [cube_by_name['da.cube'], cube_by_name['db.cube']],
        )

    # single cube output is accepted only when there is no ambiguity
    if len(cube_files) == 1:
        values, metadata = read_cube_file(cube_files[0])
        return values, metadata, [cube_files[0]]

    raise ValueError(f"Could not identify total density cube; found {cube_files}")


# ----------------------------------------------------------------------------------------------------
# manual electron density helpers

def _manual_density_grid(
    psi4_wfn,
    padding_angstrom: float = 3.0,
    spacing_angstrom=None,
    num_grid_points=None,
):
    '''build a rectangular molecular grid for manual density evaluation'''

    if spacing_angstrom is not None and num_grid_points is not None:
        raise ValueError("Specify only one of spacing_angstrom or num_grid_points")

    # build bounding box from the Psi4 geometry in bohr
    geometry_bohr = psi4_matrix_to_array(psi4_wfn.molecule().geometry()).astype(np.float64) # (n_atoms, 3)
    assert geometry_bohr.ndim == 2
    assert geometry_bohr.shape[1] == 3

    padding_bohr = float(padding_angstrom) * ANGSTROM_TO_BOHR
    min_xyz = np.min(geometry_bohr, axis=0) - padding_bohr # (3,)
    max_xyz = np.max(geometry_bohr, axis=0) + padding_bohr # (3,)
    grid_width = max_xyz - min_xyz # (3,)
    assert min_xyz.shape == (3,)
    assert max_xyz.shape == (3,)
    assert grid_width.shape == (3,)

    if num_grid_points is not None:
        grid_shape = tuple(_three_int_list(num_grid_points, 'num_grid_points'))
        axis_spacing = grid_width / (np.asarray(grid_shape, dtype=np.float64) - 1.0) # (3,)
    else:
        if spacing_angstrom is None:
            spacing_angstrom = 0.2 * BOHR_TO_ANGSTROM
        spacing_angstrom = _three_float_list(spacing_angstrom, 'spacing_angstrom')
        spacing_bohr = np.asarray(spacing_angstrom, dtype=np.float64) * ANGSTROM_TO_BOHR # (3,)
        grid_shape = tuple(np.floor(grid_width / spacing_bohr).astype(int) + 1)
        axis_spacing = grid_width / (np.asarray(grid_shape, dtype=np.float64) - 1.0) # (3,)

    if min(grid_shape) < 2:
        raise ValueError("Manual density grid must have at least two points per dimension")
    if np.any(axis_spacing <= 0.0):
        raise ValueError("Manual density grid spacing must be positive")

    # construct grid points with indexing matching cube file array order
    x = np.linspace(min_xyz[0], max_xyz[0], grid_shape[0])
    y = np.linspace(min_xyz[1], max_xyz[1], grid_shape[1])
    z = np.linspace(min_xyz[2], max_xyz[2], grid_shape[2])
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel())) # (n_grid, 3)

    axis_vectors = np.diag(axis_spacing).astype(np.float64) # (3, 3)
    metadata = {
        'grid_shape': grid_shape,
        'origin': min_xyz,
        'axis_vectors': axis_vectors,
        'voxel_volume': abs(float(np.linalg.det(axis_vectors))),
    }

    return grid_points, metadata


def _ao_values_on_grid_block(basis, grid_points):
    '''evaluate local AO basis functions on one grid block'''

    if not hasattr(psi4.core, 'BasisExtents'):
        raise NotImplementedError("Psi4 BasisExtents Python binding is required for manual AO density")
    if not hasattr(psi4.core, 'BlockOPoints'):
        raise NotImplementedError("Psi4 BlockOPoints Python binding is required for manual AO density")
    if not hasattr(psi4.core, 'RKSFunctions'):
        raise NotImplementedError("Psi4 RKSFunctions Python binding is required for manual AO density")

    grid_points = np.asarray(grid_points, dtype=np.float64) # (n_block, 3)
    assert grid_points.ndim == 2
    assert grid_points.shape[1] == 3

    n_points = int(grid_points.shape[0])
    weights = np.ones(n_points, dtype=np.float64) # (n_block,)

    # use the same Psi4 grid-function machinery as CubicScalarGrid
    extents = psi4.core.BasisExtents(basis, 1e-12)
    x_vector = _psi4_vector_from_array(grid_points[:, 0].copy())
    y_vector = _psi4_vector_from_array(grid_points[:, 1].copy())
    z_vector = _psi4_vector_from_array(grid_points[:, 2].copy())
    weight_vector = _psi4_vector_from_array(weights)

    # Python binding accepts coordinate vectors directly, without C++ block offsets
    block = psi4.core.BlockOPoints(
        x_vector,
        y_vector,
        z_vector,
        weight_vector,
        extents,
    )
    function_map = list(block.functions_local_to_global())
    max_functions = max(len(function_map), 1)

    points = psi4.core.RKSFunctions(basis, n_points, max_functions)
    points.set_ansatz(0)
    points.compute_functions(block)

    # PHI contains only basis functions nonzero on this block
    phi_local = psi4_matrix_to_array(_rks_basis_values(points, "PHI")) # (n_block, n_local)
    phi_local = _orient_phi_matrix(
        phi=phi_local,
        n_points=n_points,
        n_local=len(function_map),
    ) # (n_block, n_local)

    return phi_local, function_map


def _rks_basis_values(points, name: str):
    '''return one RKSFunctions basis-value matrix'''

    # Psi4 source uses basis_value, while some Python bindings expose basis_values
    if hasattr(points, 'basis_value'):
        return points.basis_value(name)
    if hasattr(points, 'basis_values'):
        return points.basis_values()[name]

    raise NotImplementedError("Psi4 RKSFunctions basis-value access is required for manual AO density")


def _psi4_vector_from_array(values):
    '''convert one numpy vector to a Psi4 Vector'''

    values = np.asarray(values, dtype=np.float64) # (n_values,)
    assert values.ndim == 1

    if hasattr(psi4.core.Vector, 'from_array'):
        return psi4.core.Vector.from_array(values)

    raise NotImplementedError("Psi4 Vector.from_array is required for manual AO density")


def _orient_phi_matrix(phi, n_points: int, n_local: int):
    '''orient Psi4 PHI matrix as n_points by n_local'''

    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError(f"Psi4 PHI matrix must be 2D, got shape {phi.shape}")

    if phi.shape[0] == n_points and phi.shape[1] >= n_local:
        return phi[:, :n_local]
    if phi.shape[1] == n_points and phi.shape[0] >= n_local:
        return phi[:n_local, :].T

    raise ValueError(
        f"Unexpected Psi4 PHI matrix shape {phi.shape}; "
        f"expected ({n_points}, {n_local}) up to extra local columns"
    )


# ----------------------------------------------------------------------------------------------------
# cubeprop helpers

def _normalize_cubeprop_names(cubeprop_names):
    '''normalize requested cubeprop property names'''

    # allow callers to pass either one string or a list of strings
    if isinstance(cubeprop_names, str):
        cubeprop_names = [cubeprop_names]
    if cubeprop_names is None or len(cubeprop_names) == 0:
        raise ValueError("At least one cubeprop name must be requested")

    # accept both lower-case user names and Psi4-style task names
    normalized_names = []
    for cubeprop_name in cubeprop_names:
        name = str(cubeprop_name).strip().lower()
        if name in ['density', 'densities']:
            name = 'density'
        normalized_names.append(name)

    return normalized_names


def _three_float_list(value, name: str):
    '''convert scalar or length-three input to a three-float list'''

    # scalar values are applied to all three grid axes
    if np.isscalar(value):
        return [float(value), float(value), float(value)]

    values = [float(x) for x in value]
    if len(values) == 1:
        return [values[0], values[0], values[0]]
    if len(values) != 3:
        raise ValueError(f"{name} must be a scalar or length-three sequence")

    return values


def _three_int_list(value, name: str):
    '''convert scalar or length-three input to a three-int list'''

    # scalar values are applied to all three grid axes
    if np.isscalar(value):
        values = [int(value), int(value), int(value)]
    else:
        values = [int(x) for x in value]

    if len(values) == 1:
        values = [values[0], values[0], values[0]]
    if len(values) != 3:
        raise ValueError(f"{name} must be a scalar or length-three sequence")
    if min(values) < 2:
        raise ValueError(f"{name} values must all be at least 2")

    return values


def _raise_for_unsupported_cubeprops(cubeprop_names):
    '''raise for cubeprop properties not yet supported here'''

    # density is the only cubeprop currently consumed by downstream metrics
    for cubeprop_name in cubeprop_names:
        if cubeprop_name != 'density':
            raise NotImplementedError(f"Unsupported cubeprop property: {cubeprop_name}")


def _cubeprop_grid_options(psi4_wfn=None, grid_config=None):
    '''return Psi4 cubeprop grid options from user configuration'''

    # explicit defaults prevent Psi4 global options from leaking between calls
    grid_options = {
        'cubic_grid_spacing': [0.2, 0.2, 0.2],
        'cubic_grid_overage': [4.0, 4.0, 4.0],
    }
    if grid_config is None:
        return grid_options

    num_grid_points = None

    # accept lower-case aliases or exact Psi4 option names
    for key, value in grid_config.items():
        normalized_key = str(key).strip().lower()
        if normalized_key in ['spacing', 'grid_spacing', 'cubic_grid_spacing']:
            grid_options['cubic_grid_spacing'] = _three_float_list(value, 'cubic_grid_spacing')
        elif normalized_key in ['overage', 'grid_overage', 'cubic_grid_overage']:
            grid_options['cubic_grid_overage'] = _three_float_list(value, 'cubic_grid_overage')
        elif normalized_key in ['num_points', 'grid_points', 'num_grid_points', 'cubic_grid_points']:
            num_grid_points = _three_int_list(value, 'num_grid_points')
        elif normalized_key in ['basis_tolerance', 'cubic_basis_tolerance']:
            grid_options['cubic_basis_tolerance'] = float(value)
        elif normalized_key in ['block_max_points', 'cubic_block_max_points']:
            grid_options['cubic_block_max_points'] = int(value)
        else:
            raise ValueError(f"Unknown cubeprop grid option: {key}")

    if num_grid_points is not None:
        grid_options['cubic_grid_spacing'] = _spacing_from_num_grid_points(
            psi4_wfn=psi4_wfn,
            num_grid_points=num_grid_points,
            overage=grid_options['cubic_grid_overage'],
        )

    return grid_options


def _spacing_from_num_grid_points(psi4_wfn, num_grid_points, overage):
    '''compute cubeprop spacings that give requested grid point counts'''

    if psi4_wfn is None:
        raise ValueError("psi4_wfn is required when num_grid_points is specified")

    # use the same molecular bounding box convention as Psi4 CubicScalarGrid
    geometry_bohr = psi4_matrix_to_array(psi4_wfn.molecule().geometry()).astype(np.float64) # (n_atoms, 3)
    assert geometry_bohr.ndim == 2
    assert geometry_bohr.shape[1] == 3

    min_xyz = np.min(geometry_bohr, axis=0) # (3,)
    max_xyz = np.max(geometry_bohr, axis=0) # (3,)
    molecular_width = max_xyz - min_xyz # (3,)
    grid_width = molecular_width + 2.0 * np.asarray(overage, dtype=np.float64) # (3,)

    # Psi4 writes N + 1 points, so N is one less than the requested point count
    num_intervals = np.asarray(num_grid_points, dtype=np.float64) - 1.0 # (3,)
    spacing = grid_width / num_intervals # (3,)
    if np.any(spacing <= 0.0):
        raise ValueError("Computed cubeprop grid spacing must be positive")

    # tiny enlargement avoids floating-point roundoff increasing Psi4's ceil count by one
    spacing = spacing * (1.0 + 1e-12) # (3,)

    return spacing.tolist()
