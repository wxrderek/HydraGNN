'''
Report where a NaN lives in a saved HydraGNN checkpoint, and how large the weights had grown.

Motivation: the Schur run 53692784 printed a stage trace showing that the parameters were
already non-finite when the forward ran -- its orbital embedding, which reads only discrete
metadata (l, m, shell, species), came back non-finite. So the NaN was created in a backward
pass and written into the weights by an optimizer step, and the question is which parameters
and how the magnitudes were behaving beforehand.

Two comparisons are intended:
  - a poisoned checkpoint against a healthy one, to see which modules hold the NaN. If it is
    confined to the PAINN convolution and update blocks, the backbone is the origin; if the
    AdamW second-moment state is also non-finite, no resume from that checkpoint can recover.
  - the last healthy checkpoint of a run that died against a checkpoint of a run that is still
    alive at a later epoch. Anomalously large weights in the first mean a slow blow-up, which
    gradient clipping cures; comparable weights mean one discrete gradient spike.

Read only: it opens the checkpoints and prints statistics, and writes nothing.

Example:
  python inspect_checkpoints.py \
      logs/archive/schur_dcc/qmugs-schur-train-53692784-NN1-DCC-teacher_forcing-joint-total/qmugs-schur-train-53692784-NN1-DCC-teacher_forcing-joint-total_epoch_24.pk \
      logs/qmugs-schur-train-53692804-NN1-DCC-teacher_forcing-joint-total/qmugs-schur-train-53692804-NN1-DCC-teacher_forcing-joint-total_epoch_48.pk
'''

import os
import logging
import argparse
from collections import OrderedDict

import torch


def scan_tensor(tensor, chunk: int):
    '''non-finite count and largest finite magnitude of one tensor, read in chunks'''

    # the dense head's last weight is 1600 x 320400, so boolean indexing or .abs() on the
    # whole tensor would allocate another 2 GB; chunking bounds the resident set instead
    flat = tensor.detach().reshape(-1)
    n_elements = int(flat.numel())
    n_nonfinite = 0
    largest = 0.0
    for start in range(0, n_elements, chunk):
        piece = flat[start : start + chunk].float()
        n_nonfinite += int((~torch.isfinite(piece)).sum())
        # nan_to_num keeps the reduction O(chunk) and never allocates a mask-selected copy
        largest = max(largest, float(torch.nan_to_num(piece, nan=0.0, posinf=0.0, neginf=0.0).abs().max()))
    return n_elements, n_nonfinite, largest


def inspect_checkpoint(path: str, top: int, chunk: int):
    '''print the per-parameter and per-module state of one checkpoint'''

    logging.info('')
    logging.info('=== %s', path)
    logging.info('  file size: %.3f GB', os.path.getsize(path) / 1.0e9)

    # mmap keeps the storages on disk until each tensor is touched, which matters for the
    # 6.3 GB dense checkpoint; it needs the zipfile serializer, so fall back if it is refused
    try:
        checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    except RuntimeError as error:
        logging.info('  mmap load refused (%s); loading eagerly', error)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    logging.info('  checkpoint keys: %s', sorted(checkpoint.keys()))
    for key in ('epoch', 'num_epoch', 'loss'):
        if key in checkpoint:
            logging.info('  %s: %s', key, checkpoint[key])

    # --- model parameters ---
    state_dict = checkpoint['model_state_dict']
    records = []
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            continue
        n_elements, n_nonfinite, largest = scan_tensor(tensor, chunk)
        records.append((name, tuple(tensor.shape), n_elements, n_nonfinite, largest))

    total_elements = sum(r[2] for r in records)
    total_nonfinite = sum(r[3] for r in records)
    logging.info('  model: %d float tensors, %d entries, %d non-finite',
                 len(records), total_elements, total_nonfinite)

    # roll up by the first two components of the parameter name, which is the module a
    # reader recognizes ("module.graph_convs.0..." rolls up to "module.graph_convs")
    rollup = OrderedDict()
    for name, _, n_elements, n_nonfinite, largest in records:
        parts = name.split('.')
        prefix = '.'.join(parts[:2]) if len(parts) > 2 else name
        entry = rollup.setdefault(prefix, [0, 0, 0.0])
        entry[0] += n_elements
        entry[1] += n_nonfinite
        entry[2] = max(entry[2], largest)
    logging.info('  %-44s %14s %14s %14s', 'module', 'entries', 'non-finite', 'max |w|')
    for prefix, (n_elements, n_nonfinite, largest) in rollup.items():
        logging.info('  %-44s %14d %14d %14.6e', prefix, n_elements, n_nonfinite, largest)

    nonfinite = [r for r in records if r[3] > 0]
    if nonfinite:
        logging.info('  non-finite parameters (%d):', len(nonfinite))
        for name, shape, n_elements, n_nonfinite, largest in nonfinite:
            logging.info('    %-60s %-24s %10d/%-10d max finite |w|=%.6e',
                         name, str(shape), n_nonfinite, n_elements, largest)

    logging.info('  largest %d parameters by max |w|:', top)
    for name, shape, _, n_nonfinite, largest in sorted(records, key=lambda r: r[4], reverse=True)[:top]:
        logging.info('    %-60s %-24s max |w|=%.6e non-finite=%d', name, str(shape), largest, n_nonfinite)

    # --- optimizer state ---
    optimizer_state = checkpoint.get('optimizer_state_dict', None)
    if optimizer_state is None or 'state' not in optimizer_state:
        logging.info('  optimizer state: absent')
        return

    # AdamW keeps exp_avg and exp_avg_sq per parameter. The plain DDP path keys them by the
    # parameter's index in param_groups, the FSDP2 path by its fully qualified name; both are
    # summarized the same way, and the key is printed as-is so a reader can match it up.
    state = optimizer_state['state']
    totals = OrderedDict()
    worst = []
    for param_key, entry in state.items():
        if not isinstance(entry, dict):
            continue
        for state_name, tensor in entry.items():
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                continue
            n_elements, n_nonfinite, largest = scan_tensor(tensor, chunk)
            accumulator = totals.setdefault(state_name, [0, 0, 0, 0.0])
            accumulator[0] += 1
            accumulator[1] += n_elements
            accumulator[2] += n_nonfinite
            accumulator[3] = max(accumulator[3], largest)
            worst.append((str(param_key), state_name, n_nonfinite, largest))

    logging.info('  optimizer state: %d parameters with state', len(state))
    logging.info('  %-20s %10s %14s %14s %14s', 'state', 'tensors', 'entries', 'non-finite', 'max |s|')
    for state_name, (n_tensors, n_elements, n_nonfinite, largest) in totals.items():
        logging.info('  %-20s %10d %14d %14d %14.6e', state_name, n_tensors, n_elements, n_nonfinite, largest)

    poisoned = [w for w in worst if w[2] > 0]
    if poisoned:
        logging.info('  non-finite optimizer state (%d tensors):', len(poisoned))
        for param_key, state_name, n_nonfinite, largest in poisoned[:top]:
            logging.info('    param %-24s %-14s non-finite=%d max finite=%.6e',
                         param_key, state_name, n_nonfinite, largest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='+', help='paths to .pk checkpoint files')
    parser.add_argument('--top', type=int, default=20, help='how many extreme tensors to name')
    parser.add_argument('--chunk', type=int, default=1 << 24, help='elements read at a time')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    for path in args.checkpoints:
        if not os.path.isfile(path):
            logging.warning('missing: %s', path)
            continue
        inspect_checkpoint(path, args.top, args.chunk)


if __name__ == '__main__':
    main()
