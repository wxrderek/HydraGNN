import os, json
import logging
import sys
import argparse
import subprocess
import time
from typing import Tuple


def download_one_wfn_file(i: int, data_dir: str = 'dataset'):
    os.makedirs(data_dir, exist_ok=True)
    index = str(i).zfill(2)
    start_time = time.perf_counter()
    subprocess.run(
        ['bash', 'utils/download_wfn_file.sh', index, data_dir],
        check=True,
    )
    end_time = time.perf_counter()
    return index, end_time - start_time


def download_wfn_files(
    index_range: Tuple[int, int] = (0, 1),
    data_dir: str = 'dataset',
    parallel: bool = False,
    max_workers: int = 1
):
    os.makedirs(data_dir, exist_ok=True)
    if index_range[0] < 0 or index_range[1] > 99 or index_range[0] >= index_range[1]:
        raise ValueError("Indices out of range, must be 0-99 inclusive and start < end")
    
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(lambda i: download_one_wfn_file(i, data_dir), range(*index_range))
            )
    else:
        results = []
        for i in range(*index_range):
            res = download_one_wfn_file(i, data_dir)
            results.append(res)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download QMugs wfn files from ETH Zurich collection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--start', type=int, default=0, help='Starting index of wfn file (inclusive)')
    parser.add_argument('--end', type=int, default=1, help='Ending index of wfn file (exclusive)')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Directory to save downloaded files')
    parser.add_argument('--parallel', action='store_true', help='Download files in parallel in the same terminal')
    parser.add_argument('--max_workers', type=int, default=1, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    results = download_wfn_files(
        index_range=(args.start, args.end),
        data_dir=args.data_dir,
        parallel=args.parallel,
        max_workers=args.max_workers
    )

    print("\n".join(res for res in results))