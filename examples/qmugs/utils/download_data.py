import os, json
import logging
import sys
import argparse
import subprocess
import time
from typing import Tuple


def download_file(out_file: str, url: str):    
    start_time = time.perf_counter()
    subprocess.run(
        ['wget', '-O', out_file, url], 
        check=True
    )
    end_time = time.perf_counter()
    return end_time - start_time


def download_one_wfn_file(
    i: int, 
    download_config: dict
):
    index = str(i).zfill(2)
    wfn_dir = os.path.join(
        download_config.get('data_dir', 'dataset'), 
        download_config.get('raw_data_subdir', 'raw'), 
        download_config.get('wfns_subdir', 'wfns')
    )
    os.makedirs(wfn_dir, exist_ok=True)
    wfn_file = os.path.join(wfn_dir, f'wfn_{index}.tar.gz')
    wfn_url = download_config['urls']['wfns'].replace('$', index)
    
    time_delta = download_file(wfn_file, wfn_url)
    return index, time_delta


def download_wfn_files(
    index_range: Tuple[int, int] = (0, 1),
    download_config: dict = None,
    parallel: bool = False,
    max_workers: int = 4
):
    if index_range[0] < 0 or index_range[1] > 100 or index_range[0] >= index_range[1]:
        raise ValueError("Indices out of range, must be 0-99 inclusive and start < end")
    download_config = download_config or {}
    
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(lambda i: download_one_wfn_file(i, download_config), range(*index_range))
            )
    else:
        results = []
        for i in range(*index_range):
            res = download_one_wfn_file(i, download_config)
            results.append(res)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download QMugs wfn files from ETH Zurich collection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--start', type=int, default=0, help='Starting index of wfns files (inclusive)')
    parser.add_argument('--end', type=int, default=1, help='Ending index of wfns files (exclusive)')
    parser.add_argument('--parallel', action='store_true', help='Download wfns files in parallel in the same terminal')
    parser.add_argument('--max_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--wfns_only', action='store_true', help='Only download wfn files, skip other files')
    args = parser.parse_args()

    with open('utils/download_data.json', 'r') as f:
        download_config = json.load(f)
    data_dir = download_config.get('data_dir', 'dataset')
    raw_data_dir = os.path.join(data_dir, download_config.get('raw_data_subdir', 'raw'))
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # download non wavefunction files
    if not args.wfns_only:
        for key in ['summary.csv', 'structures.tar.gz', 'vibspectra.tar.gz', 'tarball_assignment.csv']:
            time_delta = download_file(
                out_file = os.path.join(raw_data_dir, key), 
                url = download_config['urls'][key]
            )
            print(f"Downloaded {key} in {time_delta:.4f} seconds")

    wfns_results = download_wfn_files(
        index_range=(args.start, args.end),
        download_config=download_config,
        parallel=args.parallel,
        max_workers=args.max_workers
    )
    print("Download times for wfn files:")
    print("\n".join(f"{res[0] : res[1]}" for res in wfns_results))