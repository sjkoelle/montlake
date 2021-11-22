# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/atomgeom.features.ipynb (unless otherwise specified).

__all__ = ['data_stream_custom_range', 'position_to_torsion', 'position_to_distances', 'position_to_planarangle',
           'get_D_pos_feats', 'get_D_feats_feats', 'get_features']

# Cell

import torch
from ..geometry.geometry import RiemannianManifold, TangentBundle
import numpy as np
from einops import rearrange
from multiprocessing_on_dill.pool import Pool
import itertools

def data_stream_custom_range(selind):
    for i in range(len(selind)):
        yield i

def position_to_torsion(pos4, grad = True):
    #print(type(pos4))
    d1 = pos4[0]
    c1 = pos4[1]
    c2 = pos4[2]
    d2 = pos4[3]
    cc = c2 - c1
    ip = torch.einsum('i, i', (d1 - c1), (c2 - c1)) / (torch.sum((c2 - c1) ** 2))
    tilded1 = [d1[0] - ip * cc[0], d1[1] - ip * cc[1], d1[2] - ip * cc[2]]
    iq = torch.einsum('i,i',(d2 - c2), (c1 - c2)) / (torch.sum((c1 - c2) ** 2))
    cc2 = c1 - c2
    tilded2 = [d2[0] - iq * cc2[0], d2[1] - iq * cc2[1], d2[2] - iq * cc2[2]]
    tilded2star = [tilded2[0] + cc2[0], tilded2[1] + cc2[1], tilded2[2] + cc2[2]]
    ab = torch.sqrt(
        (tilded2star[0] - c1[0]) ** 2
        + (tilded2star[1] - c1[1]) ** 2
        + (tilded2star[2] - c1[2]) ** 2
    )
    bc = torch.sqrt(
        (tilded2star[0] - tilded1[0]) ** 2
        + (tilded2star[1] - tilded1[1]) ** 2
        + (tilded2star[2] - tilded1[2]) ** 2
    )
    ca = torch.sqrt(
        (tilded1[0] - c1[0]) ** 2
        + (tilded1[1] - c1[1]) ** 2
        + (tilded1[2] - c1[2]) ** 2
    )
    torsion = torch.acos((ab ** 2 - bc ** 2 + ca ** 2) / (2 * ab * ca))
    if grad == True:
        torsion.backward()
        torsion = torsion.detach().numpy()
    return(torsion)

def position_to_distances(pos2, grad = True):
    distance = torch.norm(pos2[0] - pos2[1])
    if grad == True:
        distance.backward()
        distance = distance.detach().numpy()
    return(distance)

def position_to_planarangle(pos3, grad = True):
    combos = torch.tensor([[0, 1], [1, 2], [2, 0]])
    ab = torch.norm(pos3[combos[0, 0], :] - pos3[combos[0, 1], :])
    bc = torch.norm(pos3[combos[1, 0], :] - pos3[combos[1, 1], :])
    ca = torch.norm(pos3[combos[2, 0], :] - pos3[combos[2, 1], :])
    planarangle = torch.acos((ab ** 2 - bc ** 2 + ca ** 2) / (2 * ab * ca))
    if grad == True:
        planarangle.backward()
        planarangle = planarangle.detach().numpy()
    return planarangle


def get_D_pos_feats(positions, atoms2, atoms3, atoms4):

    positions=torch.tensor(positions, requires_grad = True)
    natoms = positions.shape[0]
    natoms2 = len(atoms2)
    natoms3 = len(atoms3)
    natoms4 = len(atoms4)

    D_atompos_pairdist = np.zeros((natoms,3,natoms2))
    D_atompos_triang = np.zeros((natoms,3,natoms3))
    D_atompos_tetrator = np.zeros((natoms,3,natoms4))

    distances = np.zeros((natoms2))
    planarangles = np.zeros((natoms3))
    torsions = np.zeros((natoms4))

    for d in range(natoms2):
        atom2 = atoms2[d]
        pos2 = positions[atom2]
        #print(pos2)
        distances[d] = position_to_distances(pos2)
        D_atompos_pairdist[:,:,d] = positions.grad
        positions.grad.zero_()


    for p in range(natoms3):

        atom3 = atoms3[p]
        pos3 = positions[atom3]
        planarangles[p] = position_to_planarangle(pos3)
        D_atompos_triang[:,:,p] = positions.grad
        positions.grad.zero_()

    for t in range(natoms4):
        atom4 = atoms4[t]
        pos4 = positions[atom4]
        torsions[t] = position_to_torsion(pos4)
        D_atompos_tetrator[:,:,t] = positions.grad
        positions.grad.zero_()

    D_atomposvec_pairdist = rearrange(D_atompos_pairdist, 'a s t  -> (a s) (t)')
    D_atomposvec_triang = rearrange(D_atompos_triang, 'a s t  -> (a s) (t)')
    D_atomposvec_tetrator = rearrange(D_atompos_tetrator, ' a s t  ->  (a s) (t) ')
    D_pos_feats = np.concatenate([D_atomposvec_pairdist,D_atomposvec_triang,D_atomposvec_tetrator], axis = 1)

    return(D_pos_feats)

def get_D_feats_feats(positions,
               atoms2in = np.asarray([]),
               atoms3in = np.asarray([]),
               atoms4in = np.asarray([]),
               atoms2out = np.asarray([]),
               atoms3out = np.asarray([]),
               atoms4out = np.asarray([])):

    D_pos_feats_in = get_D_pos_feats(positions, atoms2in, atoms3in, atoms4in)
    D_pos_feats_out = get_D_pos_feats(positions, atoms2out, atoms3out, atoms4out)
    D_pos_feats_feats = np.einsum('b a, a c -> b c', np.linalg.pinv(D_pos_feats_in),D_pos_feats_out)
    return(D_pos_feats_feats)

# Cell
import torch
from .features import position_to_torsion,position_to_distances,position_to_planarangle
def get_features(positions,atoms2 = np.asarray([]), atoms3 = np.asarray([]), atoms4 = np.asarray([])):

    positions=torch.tensor(positions, requires_grad = False)

    combos = np.asarray([[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]])


    natoms2 = len(atoms2)
    natoms3 = len(atoms3)
    natoms4 = len(atoms4)

    distances = np.zeros((natoms2))
    planarangles = np.zeros((natoms3))
    torsions = np.zeros((natoms4))

    for d in range(natoms2):
        atom2 = atoms2[d]
        for e in range(1):
            pos2 = positions[atom2]
            #print(pos2)
            distances[d,1] = position_to_distances(pos2, grad = False)

    for p in range(natoms3):

        atom3 = atoms3[p,:]
        pos3 = positions[atom3]
        planarangles[p] = position_to_planarangle(pos3, grad = False)

    for t in range(natoms4):
        atom4 = atoms4[t,:]
        pos4 = positions[atom4]
        torsions[t] = position_to_torsion(pos4, grad = False)

    return(distances, planarangles, torsions)

