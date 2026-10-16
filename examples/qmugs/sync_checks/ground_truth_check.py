###########################################################################################
# Sync check P4: recompute stored QMugs targets from the Psi4 wavefunctions, independently
#
# For a sample of conformers of one pickle dataset, each stored quantity is rebuilt from the
# raw Psi4 wavefunction with the same utilities preprocessing uses, and compared exactly:
#   target y              float32 of D = Da + Db (or D - D_pro), padded or upper-triangle packed
#   promolecular baseline artifact .npy and stored promolecular_upper_triangle, float32 of D_pro
# and the physics of the ground truth itself is checked:
#   symmetry and PSD of D, electron count tr(S D) = N_alpha + N_beta,
#   SDF positions (Angstrom) against the Psi4 geometry (bohr),
#   the pipeline's dipole from D against the QMugs DFT dipole in summary.csv.
###########################################################################################

import os
import sys
import json
import argparse
import logging

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--tree_root", required=True, help="HydraGNN source tree to import from")
parser.add_argument("--basedir", required=True, help="pickle dataset directory")
parser.add_argument("--label", default="testset", help="split to sample from")
parser.add_argument(
    "--output_name",
    required=True,
    choices=["density_matrix", "density_matrix_uptr", "density_matrix_delta", "density_matrix_delta_uptr"],
)
parser.add_argument("--num", type=int, default=30, help="number of conformers, evenly spaced over the split")
parser.add_argument("--qmugs_data_dir", default="/work/xw262/qmugs", help="holds summary.csv")
parser.add_argument(
    "--path_substitution",
    nargs=2,
    action="append",
    default=[],
    metavar=("OLD", "NEW"),
    help="rewrite stored paths (e.g. Perlmutter /pscratch prefixes) before opening them",
)
parser.add_argument("--out", required=True, help="JSON results")
args = parser.parse_args()

sys.path.insert(0, os.path.join(os.path.abspath(args.tree_root), "examples", "qmugs"))
sys.path.insert(0, os.path.abspath(args.tree_root))

import psi4
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset
from utils import pad_density_matrix, upper_triangle_vector, promolecular_density_matrix
from utils.inference_utils import dipole_moment_from_density_matrix

psi4.core.be_quiet()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BOHR_TO_ANGSTROM = 0.52917721092
AU_TO_DEBYE = 2.541746473  # e * bohr in Debye

is_delta = "delta" in args.output_name
is_uptr = args.output_name.endswith("uptr")


def _resolve(path):
    '''apply the stored-path substitutions'''
    for old, new in args.path_substitution:
        path = path.replace(old, new)
    return path


summary_csv = pd.read_csv(os.path.join(args.qmugs_data_dir, "summary.csv"))
summary_csv = summary_csv.set_index(["chembl_id", "conf_id"])

dataset = SimplePickleDataset(basedir=args.basedir, label=args.label)
indices = np.unique(np.linspace(0, dataset.ntotal - 1, num=min(args.num, dataset.ntotal)).astype(int))

records = []
for k in indices:
    data = dataset.read(int(k))
    chembl_id = str(data.chembl_id)
    conformer_id = str(data.conformer_id).zfill(2)
    n_basis = int(data.density_matrix_dim)
    wfn_path = _resolve(str(data.wfn_path))
    wfn = psi4.core.Wavefunction.from_file(wfn_path)

    # ground truth total density, float64, (n_basis, n_basis)
    D = np.asarray(wfn.Da().np + wfn.Db().np, dtype=np.float64)
    assert D.shape == (n_basis, n_basis)
    y_stored = data.y.numpy() # padded (N_pad, N_pad) or packed (T(N_pad),)
    N_pad = int(round(np.sqrt(y_stored.size))) if not is_uptr else int((np.sqrt(8 * y_stored.size + 1) - 1) // 2)

    record = {"index": int(k), "chembl_id": chembl_id, "conformer_id": conformer_id, "n_basis": n_basis, "N_pad": N_pad}

    # target reconstruction, in the order preprocessing applies it (qmugs.py _mol_to_graph)
    target = D
    if is_delta:
        D_pro = promolecular_density_matrix(wfn) # float64 (n_basis, n_basis)
        target = D - D_pro
        artifact = np.load(_resolve(str(data.promolecular_density_matrix_path)))
        record["artifact_equal"] = bool(np.array_equal(artifact, D_pro.astype(np.float32)))
        record["artifact_max_abs_diff"] = float(np.max(np.abs(artifact.astype(np.float64) - D_pro)))
        if "promolecular_upper_triangle" in data:
            expected_pro = upper_triangle_vector(D_pro.astype(np.float32), N_pad, triu_indices=np.triu_indices(n_basis))
            record["promolecular_upper_triangle_equal"] = bool(
                np.array_equal(data.promolecular_upper_triangle.numpy(), expected_pro.astype(np.float32))
            )
    if is_uptr:
        expected_y = upper_triangle_vector(target, N_pad, triu_indices=np.triu_indices(n_basis)).astype(np.float32)
    else:
        expected_y = pad_density_matrix(target.copy(), N_pad).astype(np.float32)
    record["target_equal"] = bool(y_stored.shape == expected_y.shape and np.array_equal(y_stored, expected_y))
    record["target_max_abs_diff"] = float(np.max(np.abs(y_stored.astype(np.float64) - expected_y.astype(np.float64))))

    # physics of the ground truth density
    eigenvalues = np.linalg.eigvalsh(D)
    S = np.asarray(psi4.core.MintsHelper(wfn.basisset()).ao_overlap().np, dtype=np.float64)
    n_electrons = int(wfn.nalpha() + wfn.nbeta())
    record["symmetry_max_abs"] = float(np.max(np.abs(D - D.T)))
    record["min_eigenvalue_over_max"] = float(eigenvalues[0] / eigenvalues[-1])
    record["n_electrons"] = n_electrons
    record["trace_SD_error"] = float(np.trace(S @ D) - n_electrons)
    record["trace_D_minus_N_e"] = float(np.trace(D) - n_electrons)

    # geometry: SDF positions (Angstrom) used as model input vs the Psi4 geometry (bohr)
    psi4_positions = np.asarray(wfn.molecule().geometry().np, dtype=np.float64) * BOHR_TO_ANGSTROM
    record["geometry_max_abs_diff_angstrom"] = float(np.max(np.abs(data.pos.numpy().astype(np.float64) - psi4_positions)))
    psi4_numbers = [int(wfn.molecule().ftrue_atomic_number(i)) for i in range(wfn.molecule().natom())]
    record["atomic_numbers_equal"] = psi4_numbers == [int(z) for z in data.atomic_numbers.reshape(-1).tolist()]

    # dipole: pipeline implementation (atomic units) vs QMugs DFT dipole (summary.csv)
    dipole_au, _, _ = dipole_moment_from_density_matrix(D, wfn)
    row = summary_csv.loc[(chembl_id, f"conf_{conformer_id}")]
    dipole_qmugs = np.array([row["DFT_DIPOLE_X"], row["DFT_DIPOLE_Y"], row["DFT_DIPOLE_Z"]], dtype=np.float64)
    record["dipole_pipeline_au"] = dipole_au.tolist()
    record["dipole_qmugs"] = dipole_qmugs.tolist()
    record["dipole_qmugs_tot"] = float(row["DFT_DIPOLE_TOT"])
    # ratio of norms under the hypothesis that QMugs reports Debye; 1 means the units agree
    record["dipole_norm_ratio_if_debye"] = float(np.linalg.norm(dipole_au) * AU_TO_DEBYE / np.linalg.norm(dipole_qmugs))
    record["dipole_norm_ratio_if_au"] = float(np.linalg.norm(dipole_au) / np.linalg.norm(dipole_qmugs))
    record["dipole_component_max_abs_diff_if_debye"] = float(np.max(np.abs(dipole_au * AU_TO_DEBYE - dipole_qmugs)))

    records.append(record)
    logging.info(
        f"{chembl_id} conf {conformer_id} n_basis={n_basis}: target_equal={record['target_equal']} "
        f"trSD-Ne={record['trace_SD_error']:.2e} geom={record['geometry_max_abs_diff_angstrom']:.2e} "
        f"dipole ratio (Debye)={record['dipole_norm_ratio_if_debye']:.5f}"
    )

summary = {
    "basedir": args.basedir,
    "label": args.label,
    "output_name": args.output_name,
    "n_conformers": len(records),
    "all_targets_equal": all(r["target_equal"] for r in records),
    "all_artifacts_equal": all(r.get("artifact_equal", True) for r in records),
    "all_promolecular_upper_triangle_equal": all(r.get("promolecular_upper_triangle_equal", True) for r in records),
    "all_atomic_numbers_equal": all(r["atomic_numbers_equal"] for r in records),
    "max_symmetry_abs": max(r["symmetry_max_abs"] for r in records),
    "min_eigenvalue_over_max": min(r["min_eigenvalue_over_max"] for r in records),
    "max_abs_trace_SD_error": max(abs(r["trace_SD_error"]) for r in records),
    "max_geometry_abs_diff_angstrom": max(r["geometry_max_abs_diff_angstrom"] for r in records),
    "dipole_norm_ratio_if_debye_range": [min(r["dipole_norm_ratio_if_debye"] for r in records), max(r["dipole_norm_ratio_if_debye"] for r in records)],
    "dipole_norm_ratio_if_au_range": [min(r["dipole_norm_ratio_if_au"] for r in records), max(r["dipole_norm_ratio_if_au"] for r in records)],
    "records": records,
}
with open(args.out, "w") as f:
    json.dump(summary, f, indent=2)
logging.info(
    f"targets equal: {summary['all_targets_equal']}, artifacts equal: {summary['all_artifacts_equal']}, "
    f"max |tr(SD)-N_e| = {summary['max_abs_trace_SD_error']:.3e}, dipole ratio (Debye) range = "
    f"{summary['dipole_norm_ratio_if_debye_range']} -> {args.out}"
)
exact_ok = summary["all_targets_equal"] and summary["all_artifacts_equal"] and summary["all_promolecular_upper_triangle_equal"] and summary["all_atomic_numbers_equal"]
sys.exit(0 if exact_ok else 1)
