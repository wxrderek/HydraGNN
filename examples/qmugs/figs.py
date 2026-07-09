import os, json
import logging
import sys
from mpi4py import MPI
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

from rdkit import Chem

try:
    import psi4
    psi4.core.be_quiet()
except ImportError:
    print('[WARNING] psi4 package not detected')

NUM_MOLECULES = 665911
NUM_CONFORMERS = 1992984

BOHR_TO_ANGSTROM = 0.52917721092


def run_figs():
    print('yeah')

if __name__ == "__main__":
    run_figs()