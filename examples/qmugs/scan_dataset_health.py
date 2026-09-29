'''
Scan preprocessed QMugs pickle datasets for the inputs that can produce a NaN in PAINN.

Motivation: several training runs went NaN after O(1000) optimizer steps, in both the dense
head and the Schur decoder, so the cause has to live in what they share -- the encoder and
its inputs. PAINN divides by the interatomic distance twice (hydragnn/models/PAINNStack.py,
sinc_expansion at line 338 and the edge vector at line 257), so a conformer with two atoms
almost on top of each other produces enormous but finite features: a trigger that only blows
up once the weights have drifted, which is exactly the observed timing.

The runs that stayed clean used a 636-conformer draw and the ones that failed used draws of
14867 and up. qmugs.py draws molecules with random.sample, so the draws are independent, not
nested, and a pathological conformer can be present in one and absent from another. This
script compares the draws directly.

Read only: it opens the pickled Data objects and computes statistics, and writes nothing.

Example:
  python scan_dataset_health.py \
      --datasets QMugs0001_max500_delta_uptr QMugs001_max800_delta_uptr_decodercmp \
      --dataset_dir /work/xw262/qmugs
'''

import os
import glob
import pickle
import logging
import argparse

import torch


def scan_conformer(data, radius: float):
    '''statistics for one conformer; every distance is in the units stored in data.pos'''

    pos = data.pos # (n_atoms, 3)
    n_atoms = int(pos.shape[0])
    assert pos.shape == (n_atoms, 3)

    # full pairwise distances, not just the stored edges: an atom pair closer than the graph
    # cutoff is always an edge, and n_atoms is at most a few tens, so the O(n^2) is free
    distances = torch.cdist(pos.double(), pos.double()) # (n_atoms, n_atoms)
    off_diagonal = distances + torch.eye(n_atoms, dtype=distances.dtype) * 1.0e30
    min_distance = float(off_diagonal.min()) if n_atoms > 1 else float('inf')

    # an atom with no neighbour inside the cutoff never receives a message
    neighbours = (off_diagonal < radius).sum(dim=1) # (n_atoms,)
    n_isolated = int((neighbours == 0).sum())

    # the stored edge list is what the model actually consumes
    min_edge = float('inf')
    n_self_loops = 0
    if 'edge_index' in data and data.edge_index is not None and data.edge_index.numel() > 0:
        source, target = data.edge_index[0].long(), data.edge_index[1].long()
        n_self_loops = int((source == target).sum())
        edge_lengths = (pos.double()[target] - pos.double()[source]).norm(dim=1) # (n_edges,)
        min_edge = float(edge_lengths.min())

    target = data.y if 'y' in data else None
    promolecular = data.promolecular_upper_triangle if 'promolecular_upper_triangle' in data else None

    return {
        'chembl_id': getattr(data, 'chembl_id', '?'),
        'conformer_id': getattr(data, 'conformer_id', '?'),
        'n_atoms': n_atoms,
        'n_basis': int(data.density_matrix_dim) if 'density_matrix_dim' in data else -1,
        'min_distance': min_distance,
        'min_edge': min_edge,
        'n_isolated': n_isolated,
        'n_self_loops': n_self_loops,
        'nonfinite_pos': int((~torch.isfinite(pos)).sum()),
        'nonfinite_y': 0 if target is None else int((~torch.isfinite(target)).sum()),
        'max_y': 0.0 if target is None else float(target.abs().max()),
        'max_promolecular': 0.0 if promolecular is None else float(promolecular.abs().max()),
    }


def scan_dataset(basedir: str, labels, radius: float, limit: int, n_report: int):
    '''walk every pickled conformer of one dataset and report the worst cases'''

    records = []
    for label in labels:
        files = sorted(glob.glob(os.path.join(basedir, '*', '%s-*.pkl' % label)))
        files += sorted(glob.glob(os.path.join(basedir, '%s-[0-9]*.pkl' % label)))
        if limit > 0:
            files = files[:limit]
        logging.info('  %s: %d conformers', label, len(files))
        for path in files:
            with open(path, 'rb') as handle:
                data = pickle.load(handle)
            record = scan_conformer(data, radius)
            record['label'] = label
            records.append(record)

    if len(records) == 0:
        logging.warning('  no conformers found under %s', basedir)
        return records

    def report(key, reverse, unit):
        '''the n_report most extreme conformers by one key'''

        worst = sorted(records, key=lambda r: r[key], reverse=reverse)[:n_report]
        for r in worst:
            logging.info(
                '    %-16s %-4s n_atoms=%3d n_basis=%4d %s=%.6g %s',
                r['chembl_id'], r['conformer_id'], r['n_atoms'], r['n_basis'], key, r[key], unit,
            )

    n = len(records)
    logging.info('  conformers scanned: %d', n)

    min_distance = min(r['min_distance'] for r in records)
    min_edge = min(r['min_edge'] for r in records)
    logging.info('  minimum interatomic distance: %.6g', min_distance)
    report('min_distance', False, '')
    logging.info('  minimum stored edge length: %.6g', min_edge)

    totals = {
        'isolated atoms': sum(r['n_isolated'] for r in records),
        'self loops': sum(r['n_self_loops'] for r in records),
        'non-finite pos entries': sum(r['nonfinite_pos'] for r in records),
        'non-finite target entries': sum(r['nonfinite_y'] for r in records),
    }
    for name, total in totals.items():
        logging.info('  %s: %d', name, total)
    if totals['isolated atoms'] > 0:
        report('n_isolated', True, 'atoms')

    max_y = torch.tensor([r['max_y'] for r in records], dtype=torch.float64)
    quantiles = torch.tensor([0.5, 0.9, 0.99, 1.0], dtype=torch.float64)
    values = torch.quantile(max_y, quantiles)
    logging.info(
        '  max |target entry| quantiles  p50=%.6g p90=%.6g p99=%.6g max=%.6g',
        float(values[0]), float(values[1]), float(values[2]), float(values[3]),
    )
    report('max_y', True, '')

    n_basis = torch.tensor([r['n_basis'] for r in records], dtype=torch.float64)
    logging.info(
        '  n_basis  min=%d p50=%d max=%d',
        int(n_basis.min()), int(torch.quantile(n_basis, 0.5)), int(n_basis.max()),
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', required=True, help='modelnames, without the .pickle suffix')
    parser.add_argument('--dataset_dir', default='/work/xw262/qmugs', help='directory holding the .pickle directories')
    parser.add_argument('--labels', nargs='+', default=['trainset', 'valset', 'testset'])
    parser.add_argument('--radius', type=float, default=7.0, help='cutoff used to count isolated atoms')
    parser.add_argument('--limit', type=int, default=0, help='scan at most this many conformers per label, 0 for all')
    parser.add_argument('--n_report', type=int, default=5, help='how many extreme conformers to name')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    for name in args.datasets:
        basedir = os.path.join(args.dataset_dir, '%s.pickle' % name)
        logging.info('')
        logging.info('=== %s', basedir)
        if not os.path.isdir(basedir):
            logging.warning('  missing')
            continue
        scan_dataset(basedir, args.labels, args.radius, args.limit, args.n_report)


if __name__ == '__main__':
    main()
