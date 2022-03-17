# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/exec.plotting.ipynb (unless otherwise specified).

__all__ = ['parse_args', 'config', 'hparams', 'positions']

# Cell
from ..atomgeom.features import get_features,get_D_feats_feats
from ..atomgeom.utils import get_atoms_4
from ..simulations.rigidethanol import get_rigid_ethanol_data
from ..utils.utils import get_234_indices, get_atoms3_full, get_atoms4_full, data_stream_custom_range, get_cosines
from ..geometry.geometry import get_geom, get_wlpca_tangent_sel, get_rm_tangent_sel
from ..gradients.estimate import get_grads_pullback
from ..statistics.normalization import normalize_L212
from ..optimization.gradientgrouplasso import get_sr_lambda_parallel
from ..optimization.utils import get_selected_function_ids,get_selected_functions_lm2
from ..utils.replicates import Replicate, get_supports_brute
from ..plotting.manifolds import plot_manifold_2d,plot_manifold_featurespace
from ..plotting.flasso import plot_reg_path_ax_lambdasearch_customcolors_norm, plot_watch_custom
from megaman.embedding import SpectralEmbedding
from ..statistics.supportrecovery import get_min_min, get_mu_full_ind, get_kappa_s,get_gamma_max
from ..plotting.main import plot_experiment

import pandas as pd
import dill as pickle
import os
import sys
import numpy as np
import itertools
from itertools import permutations,combinations
from sklearn.decomposition import TruncatedSVD
import pathos
from pathos.multiprocessing import ProcessingPool as Pool

# Cell
import numpy as np
import random
import argparse
import json
import sys
import scipy
from ..vendor.tfcompat.hparam import HParams
import os

def parse_args(args):
    parser= argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to JSON config (can override with cmd line args)")
    #parser.add_argument("--raw_data", help="Path to raw data")
    #parser.add_argument("--outdir", help="Path to save outputs")
    args = parser.parse_args(args)
    return args


# Cell

try:
    from nbdev.imports import IN_NOTEBOOK
except:
    IN_NOTEBOOK = False

if __name__ == "__main__" and not IN_NOTEBOOK:

    np.random.seed(1234)
    random.seed(1234)
    args = parse_args(sys.argv[1:])
    config = {}
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    config.update(vars(args))
    hparams = HParams(**config)
    if not os.path.exists(hparams.outdir):
        os.makedirs(hparams.outdir)

    positions = np.load(hparams.position_file)
    for key in hparams.ground_truth.keys():
        if hparams.ground_truth[key] is None:
            hparams.ground_truth[key] = np.asarray([])
        hparams.ground_truth[key] = np.asarray(hparams.ground_truth[key], dtype = int)

    print('plotting',hparams.ground_truth)

    plot_experiment(result_file = hparams.result_file,
                    positions = positions,
                    d = hparams.d,
                    name = hparams.name,
                    ncord = hparams.ncord,
                    embedding = hparams.embedding,
                    ground_truth = hparams.ground_truth,
                    colors_gt = hparams.ground_truth_colors,
                    outdir = hparams.outdir,
                    color_counts_all = hparams.color_counts_all,
                   colors_id_all = hparams.colors_id_all,
                    names_gt_plot = hparams.names_gt_plot,
                    plot_gt = hparams.plot_gt,
                   n_components = hparams.n_components,
                   ptsize = hparams.ptsize,
                   alpha = hparams.alpha,
                    name_counts_all = hparams.name_counts_all,
                   gt_reg_color = hparams.gt_reg_color,
                   sel_reg_color = hparams.sel_reg_color,
                   plot_watch_full = hparams.plot_watch_full,
                   plot_watch_results = hparams.plot_watch_results,
                    cosine_color = hparams.cosine_color,
                    selected_cosines = hparams.selected_cosines,
                    cosine_cluster = hparams.cosine_cluster,
                    plot_set = hparams.plot_set,
                    wheel_font = hparams.wheel_font,
                   )


# Cell
import matplotlib.pyplot as plt
from ..plotting.manifolds import plot_manifold_2d, plot_manifold_3d,plot_manifold_featurespace,plot_manifold_3d_set
from ..plotting.plotting import plot_cosines, get_cmap,get_names, plot_cosines_cluster
from ..plotting.flasso import plot_reg_path_ax_lambdasearch_customcolors_norm,plot_watch_custom, plot_watch,plot_cos_boxes, plot_reg_path_ax_lambdasearch_customcolors
import dill as pickle
import pathos
from ..utils.utils import data_stream_custom_range, cosine_similarity
from pathos.multiprocessing import ProcessingPool as Pool
from ..atomgeom.features import get_features
from ..utils.utils import get_atoms4_full, get_index_matching, get_cosines
from ..utils.replicates import get_detected_values2d
from ..statistics.supportrecovery import get_min_min, get_mu_full_ind, get_kappa_s, get_gamma_max
import numpy as np
import itertools
import seaborn as sns

from matplotlib.patches import Rectangle

np.random.seed(1234)
random.seed(1234)

config = {}
with open("/Users/samsonkoelle/tunatostada/experiments/configs_plotting/jmlr/reth_diagram_mf_jmlr.json") as f:
    config.update(json.load(f))

#config.update(vars(args))
hparams = HParams(**config)
if not os.path.exists(hparams.outdir):
    os.makedirs(hparams.outdir)

positions = np.load(hparams.position_file)
for key in hparams.ground_truth.keys():
    if hparams.ground_truth[key] is None:
        hparams.ground_truth[key] = np.asarray([])
    hparams.ground_truth[key] = np.asarray(hparams.ground_truth[key], dtype = int)

print('plotting',hparams.ground_truth)

#     plot_experiment(result_file = hparams.result_file,
#                     positions = positions,
#                     d = hparams.d,
#                     name = hparams.name,
#                     ncord = hparams.ncord,
#                     embedding = hparams.embedding,
#                     ground_truth = hparams.ground_truth,
#                     colors_gt = hparams.ground_truth_colors,
#                     outdir = hparams.outdir,
#                     color_counts_all = hparams.color_counts_all,
#                    colors_id_all = hparams.colors_id_all,
#                     names_gt_plot = hparams.names_gt_plot,
#                     plot_gt = hparams.plot_gt,
#                    n_components = hparams.n_components,
#                    ptsize = hparams.ptsize,
#                    alpha = hparams.alpha,
#                     name_counts_all = hparams.name_counts_all,
#                    gt_reg_color = hparams.gt_reg_color,
#                    sel_reg_color = hparams.sel_reg_color,
#                    plot_watch_full = hparams.plot_watch_full,
#                    plot_watch_results = hparams.plot_watch_results,
#                     cosine_color = hparams.cosine_color,
#                     selected_cosines = hparams.selected_cosines,
#                     cosine_cluster = hparams.cosine_cluster
#                    )