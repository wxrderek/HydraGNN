import os, json
import logging
import sys
import argparse
import subprocess

from typing import Tuple

def download_wfn_files(
    index_range: Tuple[int, int] = (0, 1),
    data_dir: str = 'dataset'
):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    if index_range[0] < 0 or index_range[1] > 99 or index_range[0] >= index_range[1]:
        raise ValueError("Indices out of range, must be 0-99 inclusive and start < end")

    for i in range(*index_range):
        index = '0' + str(i) if len(str(i))==1 else str(i)
        subprocess.run(
            ['bash', 'download_wfn_file.sh', index, data_dir],
            check=True,
        )
