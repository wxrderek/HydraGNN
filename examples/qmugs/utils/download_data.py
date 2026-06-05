import os, json
import argparse
import subprocess
import time
from typing import Tuple

import tarfile
import shutil

def download_file(out_file: str, url: str):  
    '''Download a single file with wget'''  
    start_time = time.perf_counter()
    subprocess.run(
        ['wget', '-O', out_file, url], 
        check=True
    )
    end_time = time.perf_counter()

    # returns time taken for download
    return end_time - start_time


def download_wfn_files(
    index_range: Tuple[int, int] = (0, 1),
    download_config: dict = None,
    parallel: bool = False,
    max_workers: int = 4
):
    '''download QMugs wfns files from ETH Zurich collections of given index range [0, 99]'''
    if index_range[0] < 0 or index_range[1] > 99 or index_range[0] >= index_range[1]:
        raise ValueError("Indices out of range, must be in range 0-99 inclusive with start < end")
    download_config = download_config or {}

    # download single wfns file
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
    
    index_range_adj = (index_range[0], index_range[1] + 1) # shift indices so arg entry can be inclusive on both sides
    if parallel:
        # run parallel download in single terminal
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(lambda i: download_one_wfn_file(i, download_config), range(*index_range_adj))
            )
    else:
        # run sequential download
        results = []
        for i in range(*index_range_adj):
            res = download_one_wfn_file(i, download_config)
            results.append(res)
    
    # returns list of tuples (INDEX: str, DOWNLOAD TIME: float)
    return results


def extract_files(
    data_dir: str,
    raw_data_dir: str,
    wfns_subdir: str,
    parallel: bool = False,
    max_workers: int = 4
):
    '''extracts all downloaded QMugs files of .tar.gz format'''
    def destination_dir(raw_file: str):
        rel_dir = os.path.relpath(raw_file, raw_data_dir)
        parts = rel_dir.split(os.sep)
        if parts[0] == wfns_subdir:
            return os.path.join(data_dir, wfns_subdir)
        return data_dir

    # extract from the raw data dir to the base data dir
    def extract(src: str):
        target_dir = destination_dir(src)
        os.makedirs(target_dir, exist_ok=True)

        if tarfile.is_tarfile(src):
            with tarfile.open(src, "r:*") as tf:
                tf.extractall(target_dir)
        else:
            shutil.copy2(src, os.path.join(target_dir, os.path.basename(src)))
    
    srcs = []
    for root, _, files in os.walk(raw_data_dir):
            for filename in files:
                srcs.append(os.path.join(root, filename))

    if parallel:
        # run basic parallel extraction
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(extract, src) for src in srcs]
            for i, future in enumerate(as_completed(futures)):
                future.result()
                print(f'Extracted {i+1}/{len(futures)} files')
            for future in as_completed(futures):
                future.result()
    else:
        for i, src in enumerate(srcs): 
            extract(src)
            print(f'Extracted {i+1}/{len(srcs)} files')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download QMugs wfn files from ETH Zurich collection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--start', type=int, default=0, help='Starting index of wfns files (inclusive)')
    parser.add_argument('--end', type=int, default=1, help='Ending index of wfns files (inclusive)')
    parser.add_argument('--parallel', action='store_true', help='Download wfns files in parallel in the same terminal')
    parser.add_argument('--max_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--download_wfns_only', action='store_true', help='Only download wfn files, skip other files')
    parser.add_argument('--extract_only', action='store_true', help='Only extract pre-loaded data, skip download')
    args = parser.parse_args()

    with open('utils/download_data.json', 'r') as f:
        download_config = json.load(f)
    data_dir = download_config.get('data_dir', 'dataset')
    raw_data_dir = os.path.join(data_dir, download_config.get('raw_data_subdir', 'raw'))
    os.makedirs(raw_data_dir, exist_ok=True)

    if not args.extract_only:
        # download non wfn files
        if not args.download_wfns_only:
            for key in ['summary.csv', 'structures.tar.gz', 'vibspectra.tar.gz', 'tarball_assignment.csv']:
                time_delta = download_file(
                    out_file = os.path.join(raw_data_dir, key), 
                    url = download_config['urls'][key]
                )
                print(f"Downloaded {key} in {time_delta:.4f} seconds")

        # download wfn files
        wfns_results = download_wfn_files(
            index_range=(args.start, args.end),
            download_config=download_config,
            parallel=args.parallel,
            max_workers=args.max_workers
        )
        print("Download times for wfn files:")
        print("\n".join(f"{res[0]} : {res[1]}" for res in wfns_results))
    
    # extract downloaded files
    extract_files(
        data_dir=data_dir,
        raw_data_dir=raw_data_dir,
        wfns_subdir=download_config.get('wfns_subdir', 'wfns'),
        parallel=args.parallel,
        max_workers=args.max_workers
    )
    