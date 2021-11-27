# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/mflasso.main.ipynb (unless otherwise specified).

__all__ = ['run_exp']

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
from ..utils.replicates import Replicate, get_supports_brute,get_supports_lasso

from megaman.embedding import SpectralEmbedding

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

def run_exp(positions, hparams):

    d = hparams.d
    n_components = hparams.n_components
    atoms2_feat = hparams.atoms2_feat
    atoms3_feat = hparams.atoms3_feat
    atoms4_feat = hparams.atoms4_feat
    atoms2_dict = hparams.atoms2_dict
    atoms3_dict = hparams.atoms3_dict
    atoms4_dict = hparams.atoms4_dict
    diagram = hparams.diagram

    ii = np.asarray(hparams.ii)
    jj = np.asarray(hparams.jj)
    outfile = hparams.outdir + '/' + hparams.name + 'results_mflasso'
    print(ii)
    #load geometric features
    natoms = positions.shape[1]
    n = positions.shape[0]
    atoms2 = np.asarray(list(itertools.combinations(range(natoms), 2)))
    atoms2full = atoms2
    atoms3 = np.asarray(list(itertools.combinations(range(natoms), 3)))
    atoms4 = np.asarray(list(itertools.combinations(range(natoms), 4)))
    atoms3full = get_atoms3_full(atoms3)
    atoms4full = get_atoms4_full(atoms4)

    if atoms2_feat:
        atoms2_feats = atoms2full
    else:
        atoms2_feats = np.asarray([])

    if atoms3_feat:
        atoms3_feats = atoms3full
    else:
        atoms3_feats = np.asarray([])

    if atoms4_feat:
        atoms4_feats = atoms4full
    else:
        atoms4_feats = np.asarray([])

    #compute rotation/translation invariant featureization
    cores = pathos.multiprocessing.cpu_count() - 1
    pool = Pool(cores)
    print('feature dimensions',atoms2_feats.shape, atoms3_feats.shape,atoms4_feats.shape)
    #import pdb;pdb.set_trace
    results = pool.map(lambda i: get_features(positions[i],
                               atoms2 = atoms2_feats,
                               atoms3 = atoms3_feats,
                               atoms4 = atoms4_feats),
        data_stream_custom_range(list(range(n))))
    data = np.vstack([np.hstack(results[i]) for i in range(n)])
    data = data - np.mean(data, axis = 0)
    #apply SVD
    svd = TruncatedSVD(n_components=50)
    data_svd = svd.fit_transform(data)

    #compute geometry
    radius = hparams.radius
    n_neighbors = hparams.n_neighbors
    geom = get_geom(data_svd, radius, n_neighbors)

    #compute embedding
    spectral_embedding = SpectralEmbedding(n_components=n_components,eigen_solver='arpack',geom=geom)
    embed_spectral = spectral_embedding.fit_transform(data_svd)

    #obtain gradients
    if atoms2_dict:
        atoms2_dicts = atoms2full
    else:
        atoms2_dicts = np.asarray([])
    if atoms3_dict:
        atoms3_dicts = atoms3full
    else:
        atoms3_dicts = np.asarray([])
    if atoms4_dict and not diagram:
        atoms4_dicts = atoms4full
    elif atoms4_dict:
        atoms4_dicts= get_atoms_4(natoms, ii, jj)[0]
    else:
        atoms4_dicts = np.asarray([])

    p = len(atoms2_dicts) + len(atoms3_dicts) + len(atoms4_dicts)
    #run
    replicates = {}
    embedding = embed_spectral
    nreps = hparams.nreps
    nsel = hparams.nsel
    for r in range(nreps):
        #print(i)
        replicates[r] = Replicate(nsel = nsel, n = 10000)
        replicates[r].tangent_bases_M = get_wlpca_tangent_sel(data_svd, geom, replicates[r].selected_points, d)
        replicates[r].tangent_bases_phi = get_rm_tangent_sel(embedding, geom, replicates[r].selected_points, d)
        D_feats_feats = np.asarray([get_D_feats_feats(positions[replicates[r].selected_points[i]],
                   atoms2in = atoms2_feats,
                   atoms3in = atoms3_feats,
                   atoms4in = atoms4_feats,
                   atoms2out = atoms2_dicts,
                   atoms3out = atoms3_dicts,
                   atoms4out = atoms4_dicts) for i in range(nsel)])
        replicates[r].dg_x = np.asarray([svd.transform(D_feats_feats[i].transpose()).transpose() for i in range(nsel)])
        replicates[r].dg_x_normalized = normalize_L212(replicates[r].dg_x)
        replicates[r].dg_M = np.einsum('i b p, i b d -> i d p', replicates[r].dg_x_normalized, replicates[r].tangent_bases_M)
        replicates[r].dphispectral_M = get_grads_pullback(data_svd,  embedding, geom, replicates[r].tangent_bases_M, replicates[r].tangent_bases_phi, replicates[r].selected_points)
        replicates[r].dphispectral_M_normalized = normalize_L212(replicates[r].dphispectral_M)


    #run manifold lasso
    gl_itermax= hparams.gl_itermax
    reg_l2 = hparams.reg_l2
    max_search = hparams.max_search
    d = hparams.d
    tol = hparams.tol
    learning_rate = hparams.learning_rate
    for r in range(nreps):
        replicates[r].results = get_sr_lambda_parallel(replicates[r].dphispectral_M_normalized , replicates[r].dg_M, gl_itermax,reg_l2, max_search, d, tol,learning_rate)
        replicates[r].get_ordered_axes()
        replicates[r].sel_l = replicates[r].get_selection_lambda()

    #get manifold lasso support
    selected_functions_unique = np.asarray(np.unique(get_selected_function_ids(replicates,d)), dtype = int)
    supports_lasso = get_supports_lasso(replicates,p,d)

    #get two stage support
    selected_functions_lm2 = get_selected_functions_lm2(replicates)
    supports_brute = get_supports_brute(replicates,nreps,p,d,selected_functions_lm2)
    selected_functions_unique_twostage  = np.asarray(np.where(supports_brute > 0.)[0], dtype = int)

    pool.close()
    pool.restart()

    #compute function values for plotting... needs 'order234' for full computation
    print('computing selected function values lasso')
    selected_function_values = pool.map(
                    lambda i: get_features(positions[i],
                                           atoms2 = np.asarray([]),
                                           atoms3 = np.asarray([]),
                                           atoms4 = atoms4_dicts[selected_functions_unique]),
                    data_stream_custom_range(list(range(n))))

    selected_function_values_array = np.vstack([np.hstack(selected_function_values[i]) for i in range(n)])

    print('computing selected function values brute')
    selected_function_values_brute = pool.map(
                    lambda i: get_features(positions[i],
                                           atoms2 = np.asarray([]),
                                           atoms3 = np.asarray([]),
                                           atoms4 = atoms4_dicts[selected_functions_unique_twostage]),
                    data_stream_custom_range(list(range(n))))

    selected_function_values_array_brute = np.vstack([np.hstack(selected_function_values_brute[i]) for i in range(n)])

    #remove large gradient arrays
    print('getting cosines')
    cosine = get_cosines(np.swapaxes(replicates[0].dg_M,1,2))

    print('saving')
    replicates_small = {}
    replicates_small[0].cosine_abs = np.mean(np.abs(cosine), axis = 0)
    for r in range(nreps):
        replicates_small[r] = Replicate(nsel=nsel, n=n,
                                        selected_points=replicates[r].selected_points)
        replicates_small[r].dg_M = replicates[r].dg_M
        replicates_small[r].dphispectral_M = replicates[r].dphispectral_M
        cosine = get_cosines(np.swapaxes(replicates[r].dg_M,1,2))
        replicates_small[r].cosine_abs = np.mean(np.abs(cosine), axis = 0)
        replicates_small[r].cs_reorder = replicates[r].cs_reorder
        replicates_small[r].xaxis_reorder = replicates[r].xaxis_reorder

    #prepare to save
    results = {}
    results['replicates_small'] = replicates_small
    results['embed'] = embedding
    results['data'] = data_svd
    results['supports_brute'] = supports_brute
    results['supports_lasso'] = supports_lasso
    results['selected_function_values'] = selected_function_values
    results['selected_function_values_brute'] = selected_function_values_brute
    results['selected_functions_unique'] = selected_functions_unique
    results['selected_functions_unique_twostage'] = selected_functions_unique_twostage
    results['geom'] = geom

    #save
    with open(outfile,'wb') as output:
        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)