import os, json
import logging
import sys
from mpi4py import MPI
import argparse
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from openqdc.datasets import QMugs


def openQDC_QMugs():

    dataset = QMugs(
        array_format="numpy",
    )

    print(dataset[0])
    m0 = Chem.MolFromSmiles(dataset[0]['name'])
    Draw.MolToFile(mol=m0, filename='qmugs_m0.png', size=(300, 300), kekulize=True)


def inspect_wfn(path):
    wfns = np.load(path, allow_pickle=True).tolist()
    wfn = wfns[0]



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--OpenQDC",
        action="store_true",
        help="look at partial QMugs dataset with OpenQDC",
    )
    parser.add_argument(
        "--inspect_wfn",
        type=str,
        default=None,
        help="look at wfn file from QMugs",
    )
    args = parser.parse_args()

    if args.OpenQDC:
        openQDC_QMugs()
    if args.inspect_wfn:
        inspect_wfn(args.inspect_wfn)

