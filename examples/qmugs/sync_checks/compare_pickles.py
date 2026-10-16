###########################################################################################
# Sync check P1: compare a regenerated pickle dataset against the existing one, exactly
#
# Every split's metadata (ntotal, subdir layout, attrs such as pna_deg, max_natoms and the
# orbital metadata widths) and every sample's attributes are compared. Tensors must match in
# dtype, shape and value (torch.equal). Stored file paths differ by construction only in the
# artifacts directory name, so --path_substitution rewrites the reference prefix before the
# string comparison. Optionally the promolecular .npy files of the two artifacts directories
# are compared with np.array_equal.
###########################################################################################

import os
import sys
import json
import argparse
import logging

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--tree_root", required=True, help="HydraGNN source tree to import from")
parser.add_argument("--reference", required=True, help="existing pickle dataset directory")
parser.add_argument("--candidate", required=True, help="regenerated pickle dataset directory")
parser.add_argument(
    "--path_substitution",
    nargs=2,
    action="append",
    default=[],
    metavar=("OLD", "NEW"),
    help="replace OLD by NEW in reference string attributes before comparing",
)
parser.add_argument("--reference_artifacts", default=None, help="existing artifacts dir")
parser.add_argument("--candidate_artifacts", default=None, help="regenerated artifacts dir")
parser.add_argument("--stride", type=int, default=1, help="compare every stride-th sample")
parser.add_argument(
    "--ignore_keys",
    nargs="*",
    default=[],
    help="sample attributes excluded from the comparison (reported, e.g. natoms: qmugs.py stores an uninitialized torch.IntTensor(n))",
)
parser.add_argument("--out", required=True, help="JSON summary")
args = parser.parse_args()

sys.path.insert(0, os.path.abspath(args.tree_root))
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _equal(a, b):
    '''exact equality for tensors, arrays, containers and scalars'''
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(a, b)
        )
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _substitute(value):
    '''apply the reference -> candidate path substitutions to a string attribute'''
    if isinstance(value, str):
        for old, new in args.path_substitution:
            value = value.replace(old, new)
    return value


summary = {"splits": {}, "artifacts": None, "ignored_keys": list(args.ignore_keys)}
n_mismatch_total = 0
for label in ("trainset", "valset", "testset"):
    ref = SimplePickleDataset(basedir=args.reference, label=label)
    cand = SimplePickleDataset(basedir=args.candidate, label=label)
    split = {"ntotal_reference": ref.ntotal, "ntotal_candidate": cand.ntotal, "meta_mismatch": [], "sample_mismatch": [], "n_sample_mismatch": 0}

    for name in ("ntotal", "use_subdir", "nmax_persubdir"):
        if getattr(ref, name) != getattr(cand, name):
            split["meta_mismatch"].append(name)
    # attrs are written for the trainset only (pna_deg, max_natoms, n_l, n_m, n_shell, species)
    for key in sorted(set(ref.attrs) | set(cand.attrs)):
        if key not in ref.attrs or key not in cand.attrs:
            split["meta_mismatch"].append(f"attrs.{key} present in only one dataset")
        elif not _equal(ref.attrs[key], cand.attrs[key]):
            split["meta_mismatch"].append(f"attrs.{key}")

    n_compared = 0
    for k in range(0, min(ref.ntotal, cand.ntotal), args.stride):
        a, b = ref.read(k), cand.read(k)
        keys_a, keys_b = set(a.keys()), set(b.keys())
        if keys_a != keys_b:
            split["n_sample_mismatch"] += 1
            if len(split["sample_mismatch"]) < 200:
                split["sample_mismatch"].append({"index": k, "keys_only_reference": sorted(keys_a - keys_b), "keys_only_candidate": sorted(keys_b - keys_a)})
        for key in sorted((keys_a & keys_b) - set(args.ignore_keys)):
            if not _equal(_substitute(a[key]), b[key]):
                split["n_sample_mismatch"] += 1
                # the first 200 are recorded; the count covers all of them
                if len(split["sample_mismatch"]) < 200:
                    split["sample_mismatch"].append({"index": k, "key": key, "chembl_id": str(getattr(a, "chembl_id", ""))})
        n_compared += 1
    split["n_compared"] = n_compared
    n_mismatch_total += len(split["meta_mismatch"]) + split["n_sample_mismatch"]
    summary["splits"][label] = split
    logging.info(f"{label}: compared {n_compared} samples, {len(split['meta_mismatch'])} meta and {split['n_sample_mismatch']} sample mismatches")

if args.reference_artifacts is not None and args.candidate_artifacts is not None:
    # promolecular baselines, saved as float32 (n_basis, n_basis) arrays per conformer
    subdir = "promolecular_density_matrices"
    ref_root = os.path.join(args.reference_artifacts, subdir)
    cand_root = os.path.join(args.candidate_artifacts, subdir)
    ref_files = sorted(os.path.relpath(os.path.join(d, f), ref_root) for d, _, fs in os.walk(ref_root) for f in fs)
    cand_files = sorted(os.path.relpath(os.path.join(d, f), cand_root) for d, _, fs in os.walk(cand_root) for f in fs)
    differing = [
        f for f in sorted(set(ref_files) & set(cand_files))
        if not np.array_equal(np.load(os.path.join(ref_root, f)), np.load(os.path.join(cand_root, f)))
    ]
    summary["artifacts"] = {
        "n_reference": len(ref_files),
        "n_candidate": len(cand_files),
        "only_reference": sorted(set(ref_files) - set(cand_files))[:20],
        "only_candidate": sorted(set(cand_files) - set(ref_files))[:20],
        "n_differing": len(differing),
        "differing_examples": differing[:20],
    }
    n_mismatch_total += len(differing) + len(set(ref_files) ^ set(cand_files))
    logging.info(f"artifacts: {summary['artifacts']['n_reference']} vs {summary['artifacts']['n_candidate']} files, {len(differing)} differing")

summary["n_mismatch_total"] = n_mismatch_total
with open(args.out, "w") as f:
    json.dump(summary, f, indent=2)
logging.info(f"total mismatches: {n_mismatch_total} -> {args.out}")
sys.exit(1 if n_mismatch_total else 0)
