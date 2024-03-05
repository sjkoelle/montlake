import numpy as np

from megaman.geometry import Geometry
from megaman.utils.estimate_radius import run_estimate_radius

from typing import Optional, Any

import pickle
import os


def estimate_eps(data: np.ndarray, d: int,
                 rmin: Optional[float] = 0.25, rmax: Optional[float] = 5.0,
                 sample_pct: Optional[float] = 0.05, ntry: Optional[int] = 25) -> float:

    geom = Geometry(adjacency_method='brute', adjacency_kwds=dict(radius=rmax))

    geom.set_data_matrix(data)
    dist = geom.compute_adjacency_matrix()

    sample = np.random.permutation(data.shape[0])[:int(data.shape[0] * sample_pct)]

    distortion_vs_rad = run_estimate_radius(
        data, dist, sample=sample, d=d, rmin=rmin, rmax=rmax,
        ntry=ntry, run_parallel=True, search_space='logspace')

    radii = distortion_vs_rad[:, 0].astype(float)
    distortions = distortion_vs_rad[:, 1].astype(float)

    return float(radii[np.argmin(distortions)])


def pickle_save(obj: Any, path: str) -> None:

    if ".pkl" != path[-4:]:
        path = path + ".pkl"

    print("Saving data to", path)

    f = open(path, "wb")
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def pickle_load(path: str) -> Any:

    if ".pkl" != path[-4:]:
        path = path + ".pkl"

    assert os.path.exists(path)

    print("Loading data from", path)

    f = open(path, "rb")
    obj = pickle.load(f)
    f.close()

    return obj
