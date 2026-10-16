###########################################################################################
# Sync check E1-E4: re-evaluate a saved QMugs checkpoint on its own val/test splits
#
# Rebuilds the model from the run's saved config.json, loads the checkpoint, and runs
# hydragnn's own validate() and test() exactly as train_validate_test() does at the end of an
# epoch: same loss, same DistributedSampler (seed 0), and sampler.set_epoch(<epoch>). Launched
# with the same number of ranks as the original run, the split padding and the duplicated
# samples it introduces are identical, so the result should reproduce the logged Val/Test Loss
# of that epoch up to floating point summation order.
#
# Works in either source tree: the merged tree (DensityMatrixDataset + named Variables) or the
# pre-merge tree (SimplePickleDataset with var_config), chosen by --tree_root.
###########################################################################################

import os
import sys
import json
import argparse
import logging
import subprocess
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("--tree_root", required=True, help="HydraGNN source tree to import from")
parser.add_argument("--model_dir", required=True, help="logs/<run> directory holding config.json")
parser.add_argument("--checkpoint", required=True, help="checkpoint file name inside model_dir")
parser.add_argument("--basedir", required=True, help="pickle dataset directory")
parser.add_argument("--sampler_epoch", type=int, required=True, help="epoch whose logged losses are reproduced")
parser.add_argument("--batch_size", type=int, required=True, help="per-rank batch size of the original run")
parser.add_argument(
    "--drop_unexpected",
    action="store_true",
    help="load non-strictly, dropping unexpected keys (E4: upstream #511 vector biases)",
)
parser.add_argument(
    "--split_padding",
    action="store_true",
    help="also split each test matrix's squared error into the (n, n) block and the padded region (I1)",
)
parser.add_argument("--out", required=True, help="JSON file for the results (rank 0)")
args = parser.parse_args()

# import hydragnn and the qmugs utils from the requested tree only
tree_root = os.path.abspath(args.tree_root)
sys.path.insert(0, os.path.join(tree_root, "examples", "qmugs"))
sys.path.insert(0, tree_root)

import torch
import hydragnn
from hydragnn.utils.datasets.pickledataset import SimplePickleDataset
from hydragnn.train.train_validate_test import validate, test

assert hydragnn.__file__.startswith(tree_root), f"hydragnn imported from {hydragnn.__file__}"

try:
    # present only in the merged tree
    from utils import DensityMatrixDataset, density_matrix_variables
    merged_tree = True
except ImportError:
    merged_tree = False

world_size, rank = hydragnn.utils.distributed.setup_ddp()
logging.basicConfig(level=logging.INFO, format=f"%(levelname)s (rank {rank}): %(message)s")
logging.info(f"tree_root={tree_root} merged_tree={merged_tree} world_size={world_size}")

with open(os.path.join(args.model_dir, "config.json"), "r") as f:
    config = json.load(f)
var_config = config["NeuralNetwork"]["Variables_of_interest"]
output_names = list(var_config["output_names"])
schur_config = config["NeuralNetwork"].get("SchurDecoder", None)
use_schur = schur_config is not None and schur_config.get("enable", False)

# padded width of one head, read off the stored target (all heads share one padding width)
probe = SimplePickleDataset(basedir=args.basedir, label="testset")
target_numel = int(probe.read(0).y.numel())
assert target_numel % len(output_names) == 0
padding_dim = target_numel // len(output_names)
logging.info(f"output_names={output_names} padding_dim={padding_dim} schur={use_schur}")

# val/test datasets through the same load path the training script of this tree uses
if merged_tree:
    variables = config.get("Variables") or density_matrix_variables(output_names, padding_dim)
    valset = DensityMatrixDataset(SimplePickleDataset(basedir=args.basedir, label="valset"), variables)
    testset = DensityMatrixDataset(SimplePickleDataset(basedir=args.basedir, label="testset"), variables)
else:
    # the pre-merge training script filled these before constructing SimplePickleDataset
    var_config["graph_feature_names"] = output_names
    var_config["graph_feature_dims"] = [padding_dim] * len(output_names)
    var_config["node_feature_names"] = ["atomic_number", "cartesian_coordinates"]
    var_config["node_feature_dims"] = [1, 3]
    valset = SimplePickleDataset(basedir=args.basedir, label="valset", var_config=var_config)
    testset = SimplePickleDataset(basedir=args.basedir, label="testset", var_config=var_config)

# valset stands in for the unused trainset; only the val and test loaders are iterated
_, val_loader, test_loader = hydragnn.preprocess.create_dataloaders(
    valset, valset, testset, args.batch_size
)
for loader in (val_loader, test_loader):
    sampler = getattr(loader, "batch_sampler", None)
    if getattr(sampler, "set_epoch", None) is None:
        sampler = loader.sampler
    if getattr(sampler, "set_epoch", None) is not None:
        sampler.set_epoch(args.sampler_epoch)

# model, exactly as qmugs_densmat_train.py builds it
model = hydragnn.models.create_model_config(config=config["NeuralNetwork"], verbosity=0)
if use_schur:
    from hydragnn.decoders import SchurDecoderModel
    backbone_parameter = next(model.parameters())
    model = SchurDecoderModel(model, **{k: v for k, v in schur_config.items() if k != "enable"}).to(
        device=backbone_parameter.device, dtype=backbone_parameter.dtype
    )

checkpoint_path = os.path.join(args.model_dir, args.checkpoint)
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_dict = checkpoint.get("model_state_dict", checkpoint)
state_dict = OrderedDict(
    (k[len("module."):] if k.startswith("module.") else k, v) for k, v in state_dict.items()
)
non_tensor = [k for k, v in state_dict.items() if type(v) is not torch.Tensor]
assert not non_tensor, f"checkpoint holds non-plain tensors (e.g. DTensor shards): {non_tensor[:3]}"
if args.drop_unexpected:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not missing, f"missing keys: {missing}"
else:
    # strict: raises on any missing or unexpected key
    model.load_state_dict(state_dict, strict=True)
    missing, unexpected = [], []
logging.info(f"loaded {checkpoint_path}; dropped unexpected keys: {list(unexpected)}")

model, _ = hydragnn.utils.distributed.distributed_model_wrapper(model, None, 0)
num_tasks = model.module.num_heads

val_loss, val_tasks = validate(val_loader, model, 0, num_tasks=num_tasks, precision="fp32")
test_loss, test_tasks, _, _ = test(
    test_loader, model, 0, num_tasks=num_tasks, return_samples=False, precision="fp32"
)

split = None
if args.split_padding:
    # square single-head targets only: padded (N_pad, N_pad) with the molecule in the top-left (n, n) block.
    # each rank takes testset[rank::world_size], so every conformer is counted exactly once (no sampler padding)
    assert len(output_names) == 1 and not use_schur
    max_size = int(round(padding_dim ** 0.5))
    assert max_size * max_size == padding_dim, "split_padding needs a square padded target"
    device = next(model.parameters()).device
    sse_block = torch.zeros((), dtype=torch.float64, device=device)
    sse_pad = torch.zeros((), dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    local_indices = list(range(rank, len(testset), world_size))
    local_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(testset, local_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_loader.collate_fn,
    )
    model.eval()
    with torch.no_grad():
        for batch in local_loader:
            batch = batch.to(device)
            pred = model(batch)[0].reshape(-1, max_size, max_size).double() # (B, N_pad, N_pad)
            true = batch.y.reshape(-1, max_size, max_size).double() # (B, N_pad, N_pad)
            assert pred.shape == true.shape
            sq = (pred - true) ** 2 # (B, N_pad, N_pad)
            dims = batch.density_matrix_dim.reshape(-1).long() # (B,)
            assert dims.shape[0] == sq.shape[0]
            # mask of the (n, n) block of every matrix
            idx = torch.arange(max_size, device=device)
            in_block = (idx[None, :, None] < dims[:, None, None]) & (idx[None, None, :] < dims[:, None, None]) # (B, N_pad, N_pad)
            sse_block += (sq * in_block).sum()
            sse_pad += (sq * (~in_block)).sum()
            count += sq.shape[0]
    for t in (sse_block, sse_pad, count):
        torch.distributed.all_reduce(t)
    n = float(count.item())
    split = {
        # a stored y_mask would change what the logged loss counts, so record whether one exists
        "stored_sample_keys": sorted(probe.read(0).to_dict()),
        "n_conformers": int(n),
        "mean_block_sse_over_padding_dim": float(sse_block.item()) / padding_dim / n,
        "mean_pad_sse_over_padding_dim": float(sse_pad.item()) / padding_dim / n,
        "mean_total_sse_over_padding_dim": float(sse_block.item() + sse_pad.item()) / padding_dim / n,
    }
    logging.info(f"split_padding: {split}")

if rank == 0:
    commit = subprocess.run(
        ["git", "-C", tree_root, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    results = {
        "tree_root": tree_root,
        "tree_commit": commit,
        "merged_tree": merged_tree,
        "model_dir": os.path.abspath(args.model_dir),
        "checkpoint": args.checkpoint,
        "basedir": args.basedir,
        "world_size": world_size,
        "batch_size": args.batch_size,
        "sampler_epoch": args.sampler_epoch,
        "padding_dim": padding_dim,
        "n_val": len(valset),
        "n_test": len(testset),
        "dropped_unexpected_keys": list(unexpected),
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
        "val_tasks": [float(t) for t in val_tasks],
        "test_tasks": [float(t) for t in test_tasks],
        "split_padding": split,
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"val_loss={float(val_loss):.8e} test_loss={float(test_loss):.8e} -> {args.out}")
