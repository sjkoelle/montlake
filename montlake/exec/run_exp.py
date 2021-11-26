# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/exec.run_exp.ipynb (unless otherwise specified).

__all__ = ['parse_args', 'subset_data']

# Cell
import numpy as np
import random
import argparse
import json
import sys
import scipy
from ..vendor.tfcompat.hparam import HParams
from ..mflasso.main import run_exp as run_exp_mflasso
from ..tslasso.main import run_exp as run_exp_tslasso
import os

def parse_args(args):
    parser= argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to JSON config (can override with cmd line args)")
    parser.add_argument("--raw_data", help="Path to raw data")
    parser.add_argument("--outdir", help="Path to save outputs")
    parser.add_argument("--nreps", help="Number of replicates", type = int)
    parser.add_argument("--name", help = "Name for saving")
    parser.add_argument("--tslasso", help = "run tslasso",  action='store_true')
    parser.add_argument("--mflasso", help = "run mflasso",  action='store_true')
    args = parser.parse_args(args)
    return args

def subset_data(hparams):

    nsub = hparams.n
    file_type = hparams.raw_data[-3:]
    if file_type == "mat":
        data = scipy.io.loadmat(hparams.raw_data)
        n = data['R'].shape[0]
        nsub = np.min((nsub, n))
        randomindices = np.random.choice(range(n),nsub, replace = False)
        positions = data['R'][randomindices,:,:]
    if file_type == "npz":
        data = np.load(hparams.raw_data)
        n = data['R'].shape[0]
        nsub = np.min((nsub, n))
        randomindices = np.random.choice(range(n),nsub, replace = False)
        positions = data['R'][randomindices,:,:]
    if file_type == 'npy':
        data = np.load(hparams.raw_data)
        n = data.shape[0]
        nsub = np.min((nsub, n))
        randomindices = np.random.choice(range(n),nsub, replace = False)
        positions = data[randomindices,:,:]
    print(nsub, ' points avaiable')
    return(positions, randomindices)

# Cell
import pdb

try:
    from nbdev.imports import IN_NOTEBOOK
except:
    IN_NOTEBOOK = False

if __name__ == "__main__" and not IN_NOTEBOOK:

    np.random.seed(1234)
    random.seed(1234)
    args = parse_args(sys.argv[1:])
    print(args)
    config = {}
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    #pdb.set_trace()
    config.update(vars(args))
    hparams = HParams(**config)
#    if hparams.data_sub == None:
    positions, randomindices = subset_data(hparams)
#     else:
#         positions = np.load(hparams.data_sub)

    if not os.path.exists(hparams.outdir):
        os.makedirs(hparams.outdir)

    np.save(hparams.outdir + '/positions' + hparams.name, positions)
    np.save(hparams.outdir + '/indices'+ hparams.name, randomindices)

    if hparams.mflasso:
        run_exp_mflasso(positions = positions, hparams = hparams)

    if hparams.tslasso:
        run_exp_tslasso(positions = positions, hparams = hparams)