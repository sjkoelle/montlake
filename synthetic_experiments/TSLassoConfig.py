from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils import pickle_save, pickle_load


@dataclass
class TSLassoConfig:

    apply_svd: Optional[bool] = False
    svd_D: Optional[int] = None

    geom_n_neighbors: Optional[int] = 200
    geom_eps: Optional[float] = None

    reps: Optional[int] = 25

    gl_n: Optional[int] = 500
    gl_maxsearch: Optional[int] = 30
    gl_maxiter: Optional[int] = 500
    gl_l2_reg: Optional[float] = 0.0
    gl_tol: Optional[float] = 1e-14
    gl_lr: Optional[float] = 100.0

    def save_pkl(self, path: str) -> None:
        pickle_save(self, path)

    @staticmethod
    def load_pkl(path: str) -> TSLassoConfig:
        return pickle_load(path)
