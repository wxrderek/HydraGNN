###########################################################################################
# Sync check I1: consistency of the inference metrics with the logged padded test loss
#
# The logged test loss of a masked QMugs run is, per batch b, the mean over all selected
# entries of the batch, L_b = sum_{i in b} SSE_i / sum_{i in b} |mask_i| (Base.py
# _compact_or_padding_mask_loss), then the graph-weighted mean over batches on each rank and
# the mean over ranks (train_validate_test.test). With mask: false, |mask_i| = n_i**2.
# This rebuilds the test batches of E1 (DistributedSampler, seed 0, set_epoch(epoch), shuffle)
# and applies that formula to the unpadded per-matrix SSE_i = mse_i * n_i**2 written by the
# inference path (QMugsInference), an independent code path from validate()/test().
###########################################################################################

import os
import json
import argparse
import logging

import torch
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset

parser = argparse.ArgumentParser()
parser.add_argument("--basedir", required=True, help="pickle dataset directory")
parser.add_argument("--per_matrix_metrics", required=True, help="per_matrix_metrics.json of the inference run")
parser.add_argument("--world_size", type=int, required=True, help="number of ranks of the logged run")
parser.add_argument("--batch_size", type=int, required=True, help="per-rank batch size of the logged run")
parser.add_argument("--sampler_epoch", type=int, required=True, help="epoch whose logged test loss is reproduced")
parser.add_argument("--out", required=True, help="JSON file for the results")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

with open(args.per_matrix_metrics, "r") as f:
    per_matrix = json.load(f)
# unpadded sum of squared errors of each conformer, keyed by (chembl_id, conformer_id)
sse = {(m["chembl_id"], m["conformer_id"]): m["mse"] * m["density_matrix_dim"] ** 2 for m in per_matrix}
dims = {(m["chembl_id"], m["conformer_id"]): m["density_matrix_dim"] for m in per_matrix}
assert len(sse) == len(per_matrix), "duplicate conformers in per_matrix_metrics"

# test-set index -> conformer key, in the stored order the sampler indexes
testset = SimplePickleDataset(basedir=args.basedir, label="testset")
keys = []
for index in range(len(testset)):
    sample = testset.read(index)
    key = (sample.chembl_id, sample.conformer_id)
    assert int(sample.density_matrix_dim) == dims[key], f"dimension mismatch for {key}"
    assert getattr(sample, "y_mask", None) is None or bool(sample.y_mask.all()), f"partial mask for {key}"
    keys.append(key)
assert set(keys) == set(sse), "test split and per_matrix_metrics hold different conformers"
logging.info(f"{len(keys)} test conformers matched")

rank_losses = []
for rank in range(args.world_size):
    sampler = torch.utils.data.distributed.DistributedSampler(
        range(len(keys)), num_replicas=args.world_size, rank=rank, shuffle=True
    )
    sampler.set_epoch(args.sampler_epoch)
    indices = list(sampler)
    total, count = 0.0, 0
    for start in range(0, len(indices), args.batch_size):
        batch = [keys[i] for i in indices[start : start + args.batch_size]]
        batch_loss = sum(sse[k] for k in batch) / sum(dims[k] ** 2 for k in batch)
        total += batch_loss * len(batch)
        count += len(batch)
    rank_losses.append(total / count)
test_loss = sum(rank_losses) / len(rank_losses)

# the same SSE under the global entry-weighted mean, for reference
global_entry_mean = sum(sse.values()) / sum(n ** 2 for n in dims.values())
results = {
    "basedir": args.basedir,
    "per_matrix_metrics": os.path.abspath(args.per_matrix_metrics),
    "world_size": args.world_size,
    "batch_size": args.batch_size,
    "sampler_epoch": args.sampler_epoch,
    "n_test": len(keys),
    "rank_losses": rank_losses,
    "reconstructed_test_loss": test_loss,
    "global_entry_weighted_mse": global_entry_mean,
}
with open(args.out, "w") as f:
    json.dump(results, f, indent=2)
logging.info(f"reconstructed test loss {test_loss:.8e}; global entry-weighted {global_entry_mean:.8e} -> {args.out}")
