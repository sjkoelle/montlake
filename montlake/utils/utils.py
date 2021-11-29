# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/utils.utils.ipynb (unless otherwise specified).

__all__ = ['get_index_matching', 'get_234_indices', 'get_atoms4_full', 'get_atoms3_full', 'data_stream_custom_range',
           'get_cosines', 'cosine_similarity']

# Cell
import numpy as np
from einops import rearrange

def get_index_matching(probe, superset):

    probe_permuted = np.asarray([probe[[0, 1, 2, 3]],
                                 probe[[3,2,1,0]],
                                 # combos[c][[0,2,1,3]],
                                 probe[[0,2,1,3]],
                                 probe[[3,1,2,0]]])

    output = np.asarray([])
    for p in range(4):
        #print(p)
        output = np.append(output,np.where((superset==tuple(probe_permuted[p])).all(1))[0])

    return(int(output))

def get_234_indices(selected_indices, natoms4, natoms2, natoms3, order234 = [2,0,1]):
    '''
    Get indices in dictionary of each functions set ordered by order234'''
    lens = [natoms2, natoms3,natoms4]

    combostart = [0, lens[order234[0]], lens[order234[0]] + lens[order234[1]], lens[order234[0]] + lens[order234[1]]+ lens[order234[2]]]

    nsel = len(selected_indices)
    functionset_id = np.zeros(nsel)
    for j in range(nsel):
        for i in range(len(combostart) - 1):
            if selected_indices[j] > combostart[i] and selected_indices[j] < combostart[i+1] :
                #print('here')
                functionset_id[j] = i

    return(np.asarray(functionset_id , dtype = int),combostart)

def get_atoms4_full(atoms4):

    combos4 = np.asarray([[0, 1, 2,3],
                         [1,2,3,0],
                         [2,3,0,1],
                         [3,0,1,2],
                         [0, 1,3,2],
                         [1,0,2,3] ])

    atoms4full = np.asarray([atoms4[:,c] for c in combos4])
    atoms4full = rearrange(atoms4full,'i j k -> (j i) k')
    return(atoms4full)

def get_atoms3_full(atoms3):

    combos3 = np.asarray([[0, 1, 2],
                         [1,2,0],
                        [2,0,1]])

    atoms3full = np.asarray([atoms3[:,c] for c in combos3])
    atoms3full = rearrange(atoms3full,'i j k -> (j i ) k')
    return(atoms3full)

def data_stream_custom_range(selind):
    for i in range(len(selind)):
        yield i

# Cell
import numpy as np

# def get_cosines(dg):
#     n = dg.shape[0]
#     p = dg.shape[1]
#     d = dg.shape[2]
#     coses = np.zeros((n, p, p))
#     for i in range(n):
#         for j in range(p):
#             for k in range(p):
#                 coses[i, j, k] = cosine_similarity(dg[i, j, :], dg[i, k,:])  # sklearn.metrics.pairwise.cosine_similarity(X = np.reshape(dg[:,i,:], (1,d*n)),Y = np.reshape(dg[:,j,:], (1,d*n)))[0][0]

#     return (coses)
def get_cosines(dg):
    n = dg.shape[0]
    p = dg.shape[2]
    d = dg.shape[1]
    coses = np.zeros((n, p, p))
    for i in range(n):
        for j in range(p):
            for k in range(p):
                coses[i, j, k] = cosine_similarity(dg[i, :, j], dg[i, :,k])  # sklearn.metrics.pairwise.cosine_similarity(X = np.reshape(dg[:,i,:], (1,d*n)),Y = np.reshape(dg[:,j,:], (1,d*n)))[0][0]

    return (coses)

def cosine_similarity(a, b):
    output = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return (output)