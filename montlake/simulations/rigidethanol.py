# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/simulations.rigidethanol.ipynb (unless otherwise specified).

__all__ = ['get_rigid_ethanol_data']

# Cell
import numpy as np
import math

def get_rigid_ethanol_data(xvar, noise = False):

    n = 10000
    natoms = 9
    cor = 0

    positions = np.zeros((n, natoms, 3))
    positions[0, 0, :] = np.asarray([0., 0., 0.])
    positions[0, 1, :] = np.asarray([-10., 0., np.sqrt(2) / 100])
    positions[0, 2, :] = np.asarray([0., 10., np.sqrt(3) / 100])
    positions[0, 8, :] = np.asarray([1., 10., np.sqrt(5) / 100])
    positions[0, 3, :] = np.asarray([np.sqrt(7) / 100, np.cos(2 / 3 * np.pi), np.sin(2 / 3 * np.pi)])
    positions[0, 4, :] = np.asarray([np.sqrt(11) / 100, np.cos(2 / 3 * np.pi), np.sin(4 / 3 * np.pi)])
    positions[0, 5, :] = np.asarray([-11., 1., np.sqrt(13) / 100])
    positions[0, 6, :] = np.asarray([-11., np.cos(2 / 3 * np.pi) + np.sqrt(17) / 1000, np.sin(2 / 3 * np.pi)])
    positions[0, 7, :] = np.asarray([-11., np.cos(2 / 3 * np.pi) + np.sqrt(19) / 100, np.sin(4 / 3 * np.pi)])

    angles1 = np.tile(np.linspace(start=0., stop=2 * math.pi, num=int(np.sqrt(n)), endpoint=False),
                      int(np.sqrt(n)))
    angles2 = np.repeat(np.linspace(start=0., stop=2 * math.pi, num=int(np.sqrt(n)), endpoint=False),
                        int(np.sqrt(n)))
    for i in range(1, n):
        rotationmatrix1 = np.zeros((3, 3))
        rotationmatrix1[1, 1] = 1
        rotationmatrix1[0, 0] = np.cos(angles1[i])
        rotationmatrix1[0, 2] = -np.sin(angles1[i])
        rotationmatrix1[2, 2] = np.cos(angles1[i])
        rotationmatrix1[2, 0] = np.sin(angles1[i])
        rotationmatrix2 = np.zeros((3, 3))
        rotationmatrix2[0, 0] = 1
        rotationmatrix2[1, 1] = np.cos(angles2[i])
        rotationmatrix2[1, 2] = -np.sin(angles2[i])
        rotationmatrix2[2, 2] = np.cos(angles2[i])
        rotationmatrix2[2, 1] = np.sin(angles2[i])
        positions[i, np.asarray([3, 4]), :] = positions[0, np.asarray([3, 4]), :]
        positions[i, np.asarray([2, 8]), :] = np.matmul(rotationmatrix1,
                                                        positions[0, np.asarray([2, 8]),
                                                        :].transpose()).transpose()
        positions[i, np.asarray([1, 5, 6, 7]), :] = np.matmul(rotationmatrix2,
                                                              positions[0, np.asarray([1, 5, 6, 7]),
                                                              :].transpose()).transpose()

    covariance = np.identity(natoms)
    for i in range(natoms):
        for j in range(natoms):
            if i != j:
                covariance[i, j] = cor
    covariance = xvar * covariance
    if noise == True:
        for i in range(n):
            for j in range(3):
                positions[i, :, j] = np.random.multivariate_normal(positions[i, :, j], covariance, size=1)

    return (positions)