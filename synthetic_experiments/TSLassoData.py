from __future__ import annotations

from numpy import ndarray

from utils import pickle_save, pickle_load

from ordered_set import OrderedSet
from typing import Sequence


class TSLassoData(object):

    def __init__(self,
                 data: ndarray,
                 gradients: ndarray,
                 true_idxs: Sequence[int],
                 d: int, D: int):

        assert data.shape[0] == gradients.shape[0]
        assert data.shape[1] == gradients.shape[2] == D

        self.n = data.shape[0]  # number of data points
        self.p = gradients.shape[1]  # number of dictionary functions
        self.D = D  # ambient number of dimensions
        self.d = d  # intrinsic dim

        self.true_indices = OrderedSet(sorted(true_idxs))
        self.fake_indices = OrderedSet(i for i in range(self.p) if i not in self.true_indices)

        self.data = data
        self.gradients = gradients

    def save_pkl(self, path: str) -> None:
        pickle_save(self, path)

    @staticmethod
    def load_pkl(path: str) -> TSLassoData:
        return pickle_load(path)
