import numpy as np
from numpy import ndarray

from sortedcontainers import SortedDict
from typing import Optional, Dict, Any
import time

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from umap import UMAP

from megaman.embedding import spectral_embedding
from megaman.geometry import Geometry, RiemannMetric

from montlake.geometry.geometry import get_geom, get_wlpca_tangent_sel
from montlake.optimization.gradientgrouplasso import get_sr_lambda_parallel

from utils import pickle_save, estimate_eps

from ManifoldLassoConfig import ManifoldLassoConfig
from TSLassoData import TSLassoData


class ManifoldLasso(object):

    def __init__(self,
                 mlasso_config: ManifoldLassoConfig,
                 mlasso_data: TSLassoData):

        self.mlasso_config = mlasso_config

        self.raw_data = mlasso_data.data
        self.raw_gradients = mlasso_data.gradients

        self.n, self.d, self.D, self.p = mlasso_data.n, mlasso_data.d, mlasso_data.D, mlasso_data.p
        self.m = mlasso_config.m

        self._data_mean = None
        self._svd = None

        self._geom = None

        self._data = None
        self._gradients = None
        self._embeddings = None
        self._tangent_bundle = None
        self._rmetric = None

    @property
    def apply_svd(self) -> bool: return self.mlasso_config.apply_svd
    @property
    def svd_D(self) -> int: return self.mlasso_config.svd_D

    @property
    def geom_n_neighbors(self) -> int: return self.mlasso_config.geom_n_neighbors

    @property
    def geom_radius(self,
                    rmin: Optional[float] = 0.25, rmax: Optional[float] = 5.0,
                    sample_pct: Optional[float] = 0.05, ntry: Optional[int] = 25) -> float:

        if self.mlasso_config.geom_eps is None:
            self.mlasso_config.geom_eps = estimate_eps(self.data, self.d, rmin=rmin, rmax=rmax, sample_pct=sample_pct,
                                                       ntry=ntry)
        return self.mlasso_config.geom_eps

    @property
    def embedding_algo(self) -> str: return self.mlasso_config.embedding_algo
    @property
    def embedding_kwargs(self) -> Dict[str, Any]: return self.mlasso_config.embedding_kwargs if self.mlasso_config.embedding_kwargs else dict()

    @property
    def reps(self) -> int: return self.mlasso_config.reps

    @property
    def gl_n(self) -> int: return self.mlasso_config.gl_n if self.mlasso_config.gl_n else self.n
    @property
    def gl_maxsearch(self) -> int: return self.mlasso_config.gl_maxsearch
    @property
    def gl_maxiter(self) -> int: return self.mlasso_config.gl_maxiter
    @property
    def gl_l2_reg(self) -> float: return self.mlasso_config.gl_l2_reg
    @property
    def gl_tol(self) -> float: return self.mlasso_config.gl_tol
    @property
    def gl_lr(self) -> float: return self.mlasso_config.gl_lr

    @property
    def data_mean(self) -> ndarray:
        if self._data_mean is None:
            self._data_mean = np.mean(self.raw_data, axis=0)
        return self._data_mean

    @property
    def svd(self) -> TruncatedSVD:
        if self._svd is None:
            n_components = self.svd_D if self.svd_D else self.D
            self._svd = TruncatedSVD(n_components=n_components).fit(self.raw_data - self.data_mean)
        return self._svd

    @property
    def data(self) -> ndarray:
        if self._data is None:
            self._data = self.transform_data(self.raw_data, subtract_mean=True, apply_svd=self.apply_svd)
        return self._data

    @property
    def gradients(self) -> ndarray:
        if self._gradients is None:
            self._gradients = self.transform_data(self.raw_gradients, subtract_mean=False, apply_svd=self.apply_svd)
        return self._gradients

    @property
    def embeddings(self) -> ndarray:
        if self._embeddings is None:
            if self.embedding_algo == "spectral":
                self._embeddings, _, _ = spectral_embedding.spectral_embedding(geom=self.geom,
                                                                               eigen_solver='amg',
                                                                               n_components=self.m,
                                                                               **self.embedding_kwargs)
            elif self.embedding_algo == "umap":
                self._embeddings = UMAP(n_components=self.m, n_neighbors=self.geom_n_neighbors,
                                        metric="euclidean", **self.embedding_kwargs).fit_transform(self.data)
            else:
                raise KeyError
        return self._embeddings

    @property
    def geom(self) -> Geometry:
        if self._geom is None:
            self._geom = get_geom(self.data, radius=self.geom_radius, n_neighbors=self.geom_n_neighbors)
        return self._geom

    @property
    def tangent_bundle(self) -> ndarray:
        if self._tangent_bundle is None:
            self._tangent_bundle = get_wlpca_tangent_sel(self.data, self.geom, range(self.n), self.d)
        return self._tangent_bundle

    @property
    def rmetric(self) -> ndarray:
        if self._rmetric is None:
            self._rmetric = RiemannMetric(self.embeddings, self.geom.laplacian_matrix, self.d)
            self._rmetric.get_rmetric()
            self._rmetric = self._rmetric.Hvv.transpose((0, 2, 1))
        return self._rmetric

    def transform_mean(self, data: ndarray) -> ndarray: return data - self.data_mean

    def transform_svd(self, data: ndarray) -> ndarray:

        data_shape = data.shape

        if len(data_shape) > 2:
            data = data.reshape(-1, data_shape[-1])

        data = self.svd.transform(data)

        if len(data_shape) > 2:
            data = data.reshape(data_shape)

        return data

    def transform_data(self, data: ndarray,
                       subtract_mean: Optional[bool] = True,
                       apply_svd: Optional[bool] = True) -> ndarray:

        if subtract_mean:
            data = self.transform_mean(data)
        if apply_svd:
            data = self.transform_svd(data)
        return data

    def _compute_Y(self, rep_pts_idx: ndarray,
                   rep_tangent_bases_ambient: ndarray,
                   rep_tangent_bases_embedding: ndarray) -> ndarray:

        Y = np.zeros((rep_pts_idx.shape[0], self.d, self.m), dtype=float)

        for i, idx in enumerate(rep_pts_idx):

            neighbor_pts = self.geom.affinity_matrix[idx].indices

            Ai_t = ((self.data[neighbor_pts, :] - self.data[idx, :]) @ rep_tangent_bases_ambient[i])  # k_i x d
            Bproj_i = (self.embeddings[neighbor_pts, :] - self.embeddings[idx, :]) @ (rep_tangent_bases_embedding[i] @ rep_tangent_bases_embedding[i].T)  # k_i x m

            lr = LinearRegression(fit_intercept=False)
            lr.fit(Ai_t, Bproj_i)
            Y[i, :, :] = lr.coef_.T

        gammas = np.sqrt(np.sum(Y ** 2, axis=(0, 1), keepdims=True) / Y.shape[0])
        return Y / (gammas + np.finfo(float).eps)

    def _compute_X(self, rep_gradients_ambient: ndarray, rep_tangent_bases_ambient: ndarray) -> ndarray:

        gammas = np.sqrt(np.sum(rep_gradients_ambient ** 2, axis=(0, 2), keepdims=True) / rep_gradients_ambient.shape[0])
        return np.einsum('ipD,iDd->idp', rep_gradients_ambient / (gammas + np.finfo(float).eps),
                         rep_tangent_bases_ambient)

    def run(self,
            time_run: Optional[bool] = False,
            results_path: Optional[str] = None) -> None:

        sel_coeffs = []
        sel_lambs = []
        lamb_norms = []
        lambs = []
        times = []

        for _ in range(self.reps):

            # if we're timing the experiment, set all the precomputed things that we can reuse for separate
            # runs of group lasso to None so that we time all the steps.
            if time_run:

                self._geom = None

                self._data = None
                self._embeddings = None
                self._gradients = None
                self._tangent_bundle = None
                self._rmetric = None

            start_time = time.time()

            rep_pts_idx = np.random.choice(np.arange(self.n), self.gl_n, replace=False)

            rep_gradients_ambient = self.gradients[rep_pts_idx]
            rep_tangent_bases_ambient = self.tangent_bundle[rep_pts_idx]
            rep_tangent_bases_embedding = self.rmetric[rep_pts_idx]

            rep_X = self._compute_X(rep_gradients_ambient, rep_tangent_bases_ambient)
            rep_Y = self._compute_Y(rep_pts_idx, rep_tangent_bases_ambient, rep_tangent_bases_embedding)

            sel_lamb, coeff, norm = get_sr_lambda_parallel(rep_Y, rep_X, self.gl_maxiter, self.gl_l2_reg,
                                                           self.gl_maxsearch, self.d, self.gl_tol, self.gl_lr)

            rep_time = (time.time() - start_time) if time_run else None
            times.append(rep_time)

            norm = SortedDict(norm)

            try:
                sel_coeffs.append(coeff[sel_lamb])
            except KeyError:
                print("Did not converge")
                continue

            sel_lambs.append(sel_lamb)
            lamb_norms.append(SortedDict(norm))
            lambs.append(np.array(tuple(norm.keys())))

        results = dict(n=self.n, d=self.d, D=self.D, p=self.p,
                       config=self.mlasso_config,
                       sel_coeffs=(np.stack(sel_coeffs) if sel_coeffs else []),
                       sel_lambs=np.array(sel_lambs),
                       lamb_norms=tuple(lamb_norms),
                       lambs=tuple(lambs),
                       times=np.array(times))

        if results_path is not None:
            pickle_save(results, results_path)
