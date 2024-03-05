from __future__ import annotations

import numpy as np

from typing import Optional, Callable, Union, Sequence
from itertools import combinations, product

from sym_types import DomainInput
from sym_constants import inv_abs_tol, inv_rel_tol
from sym_utils import parse_domain, pickle_save, pickle_load
from vis_utils import plot_points


class ManifoldData(object):

    def __init__(self,
                 d: int, D: int,
                 local_data: Optional[bool] = False,
                 has_coord_pts: Optional[bool] = True,
                 has_amb_pts: Optional[bool] = True,
                 has_coord_jac: Optional[bool] = True,
                 has_inv_jac: Optional[bool] = True,
                 has_tan_plane: Optional[bool] = True):

        self.d = d
        self.D = D

        self.local_data = local_data

        if has_coord_pts:
            self.coord_pts = np.array([], dtype=float).reshape(0, self.d)
        if has_amb_pts:
            self.amb_pts = np.array([], dtype=float).reshape(0, self.D)
        if has_coord_jac and not self.local_data:
            self.coord_jac = np.array([], dtype=float).reshape(0, self.d, self.D)
        if has_inv_jac and not self.local_data:
            self.inv_jac = np.array([], dtype=float).reshape(0, self.D, self.d)
        if has_tan_plane and not self.local_data:
            self.tan_plane = np.array([], dtype=float).reshape(0, self.D, self.d)

        self.data_dict = {data_key: data_arr for data_key, data_arr in
                          zip(("coord_pts", "amb_pts", "coord_jac", "inv_jac", "tan_plane"),
                              (self.coord_pts, self.amb_pts, self.coord_jac, self.inv_jac, self.tan_plane))
                          if data_arr is not None}

    def __getitem__(self, item) -> np.ndarray:
        return self.data_dict.get(item, None)

    def _update_attr(self, data_key: str) -> None:
        if hasattr(self, data_key):
            setattr(self, data_key, self.data_dict[data_key])

    @property
    def num_points(self) -> int:
        if self.data_dict:
            return next(iter(self.data_dict.values())).shape[0]
        return 0

    @property
    def same_num_points(self) -> bool:
        shapes = tuple(data_arr.shape[0] for data_arr in self.data_dict.values())
        return all(shapes[0] == s for s in shapes[1:])

    @property
    def has_inf(self) -> bool:
        return any(np.any(np.isinf(data_arr)) for data_arr in self.data_dict.values())

    @property
    def has_nan(self) -> bool:
        return any(np.any(np.isnan(data_arr)) for data_arr in self.data_dict.values())

    @property
    def has_integrity(self):
        return self.same_num_points and not self.has_inf and not self.has_nan

    def reset_data(self,
                   data_key: str,
                   data_shape: Optional[tuple[int, ...]] = None) -> None:

        if data_shape is None:
            self.data_dict[data_key] = np.array([], dtype=float).reshape(0, *self.data_dict[data_key].shape[1:])
        else:
            self.data_dict[data_key] = np.array([], dtype=float).reshape(0, *data_shape)

        self._update_attr(data_key)

    def reset_all_data(self) -> None:
        for data_key in self.data_dict.keys():
            self.reset_data(data_key)

    def add_data(self,
                 data_key: str,
                 data: np.ndarray) -> None:

        self.data_dict[data_key] = np.concatenate([self.data_dict[data_key], data], axis=0)
        self._update_attr(data_key)

    def remove_data(self,
                    remove_flags: np.ndarray) -> None:

        assert self.same_num_points

        for data_key, data in self.data_dict.items():
            self.data_dict[data_key] = data[~remove_flags]
            self._update_attr(data_key)

    def remove_bad_values(self) -> None:

        for data in self.data_dict.values():

            bad_val_flags = np.zeros(self.num_points, dtype=bool)
            bad_val_flags += np.isnan(data.sum(axis=tuple(range(1, data.ndim))))
            bad_val_flags += np.isinf(data.sum(axis=tuple(range(1, data.ndim))))

            if bad_val_flags.any():
                self.remove_data(bad_val_flags)

    def bbox(self, data_key: [Optional[str]] = "amb_pts") -> tuple[np.ndarray, np.ndarray]:

        assert not (self.local_data and data_key in {"coord_jac", "inv_jac", "tan_plane"})
        return np.min(self.data_dict[data_key], axis=0), np.max(self.data_dict[data_key], axis=0)

    def add_noise(self,
                  data_key: str,
                  save_key: Optional[str] = None,
                  noise_func: Optional[Union[str, Callable]] = "normal",
                  noise_kwargs: Optional[dict] = None,
                  clip_bounds: Optional[DomainInput] = None,
                  clip_bbox: Optional[bool] = False,
                  bbox_tol: Optional[int] = 0.0) -> np.ndarray:

        assert not (self.local_data and data_key in {"coord_jac", "inv_jac", "tan_plane"})

        if noise_func == "normal":
            def noise_func(size, loc=0.0, scale=0.01, **_) -> np.ndarray:
                return np.random.normal(loc=loc, scale=scale, size=size)

        if noise_kwargs is None:
            noise_kwargs = dict()

        noisy_data = self.data_dict[data_key] + noise_func(self.data_dict[data_key].shape, **noise_kwargs)

        if clip_bounds is not None or clip_bbox:

            if clip_bbox:

                clip_bounds = self.bbox(data_key)
                clip_min, clip_max = clip_bounds

                if bbox_tol:
                    clip_min = clip_min - bbox_tol
                    clip_max = clip_max + bbox_tol

            else:
                clip_min, clip_max = clip_bounds

            noisy_data = np.clip(noisy_data, clip_min, clip_max)

        if save_key is not None:
            self.data_dict[save_key] = noisy_data

        return noisy_data

    def project(self,
                data: Optional[np.ndarray] = None,
                data_key: Optional[np.ndarray] = None,
                normalize: Optional[bool] = False) -> np.ndarray:

        assert self.tan_plane is not None

        if data is None:
            data = self.data_dict[data_key]

        if data.ndim == 2 and self.local_data:
            out_data = np.einsum('nD,Dd->nd', data, self.tan_plane)
        elif data.ndim == 2 and not self.local_data:
            out_data = np.einsum('nD,nDd->nd', data, self.tan_plane)
        elif data.ndim == 3 and self.local_data:
            out_data = np.einsum('npD,Dd->npd', data, self.tan_plane)
        elif data.ndim == 3 and not self.local_data:
            out_data = np.einsum('npD,nDd->npd', data, self.tan_plane)
        else:
            raise ValueError

        if normalize:
            out_data = out_data / np.linalg.norm(data, axis=-1, keepdims=True)

        return out_data

    def projected_amb_pts(self, normalize: Optional[bool] = False) -> np.ndarray:
        return self.project(data_key="amb_pts", normalize=normalize)

    def projected_coord_jac(self, normalize: Optional[bool] = False) -> np.ndarray:
        return self.project(data_key="coord_jac", normalize=normalize)

    def nu(self, agg_func: Optional[Callable] = np.max) -> float:

        assert not self.local_data and self.coord_jac is not None

        X_hat = self.projected_coord_jac(normalize=True)
        xTx = np.einsum('npd,ndq->npq', X_hat, X_hat)

        Gsq = np.zeros_like(xTx)
        Gsq[:, list(range(self.d)), list(range(self.d))] = np.linalg.norm(self.coord_jac, axis=2) ** 2

        return float(agg_func([np.linalg.norm(mat, ord=2) for mat in (np.linalg.inv(xTx) - Gsq)]))

    def mu(self,
           other_gradients_key: str,
           agg_func: Optional[Callable] = np.max,
           keep_correlations: Optional[bool] = False) -> Union[float, np.ndarray]:

        assert not self.local_data and self.coord_jac is not None

        X_hat = self.projected_coord_jac(normalize=True)
        other_gradients = self.project(self.data_dict[other_gradients_key], normalize=True)

        if keep_correlations:
            return agg_func(np.abs(np.einsum("npd,nqd->npq", X_hat, other_gradients)), axis=0)
        else:
            return float(agg_func(np.abs(np.einsum("npd,nqd->npq", X_hat, other_gradients))))

    def cond_num(self,
                 data_key: Optional[str] = "coord_jac",
                 agg_func: Optional[Callable] = np.max) -> float:

        assert not (self.local_data and data_key in {"coord_jac", "inv_jac", "tan_plane"})
        return float(agg_func(np.linalg.cond(self.data_dict[data_key]), axis=0))

    def visualize_pts(self,
                      data_key: Optional[str, Sequence[str]] = "amb_pts",
                      color_key: Optional[str, Sequence[str]] = "coord_pts",
                      data_dims: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
                      data_vis_dims: Optional[int] = None,
                      color_dims: Optional[Sequence[int]] = None,
                      **kwargs) -> None:

        if isinstance(data_key, str):
            data = self.data_dict[data_key]
        else:
            data = np.hstack([self.data_dict[dk] for dk in data_key])
        assert data.ndim == 2

        if data_vis_dims is None:
            data_vis_dims = min(data.shape[1], 3)

        if data_dims is None:
            data_dims = tuple(combinations(range(data.shape[1]), data_vis_dims))
        elif isinstance(data_dims[0], int):
            data_dims = tuple(combinations(data_dims, data_vis_dims))

        colors = None
        subfig_layout = None

        if color_key is not None:

            if isinstance(color_key, str):
                colors = self.data_dict[color_key]
            else:
                colors = np.hstack([self.data_dict[dk] for dk in color_key])

            assert colors.ndim == 2

            if color_dims is None:
                color_dims = tuple(range(colors.shape[1]))

            subplot_data_color_idxs = tuple(product(data_dims, color_dims))
            subfig_layout = (len(data_dims), len(color_dims))

        else:

            subplot_data_color_idxs = data_dims

        plot_points(datapoints=data, colors=colors,
                    subplot_data_color_idxs=subplot_data_color_idxs,
                    subfig_layout=subfig_layout,
                    **kwargs)

    def save_pkl(self, path: str) -> None:
        pickle_save(self, path)

    @staticmethod
    def load_pkl(path: str) -> ManifoldData:
        return pickle_load(path)


class CoordSystem(object):

    def __init__(self,
                 d: int, D: int,
                 coord_domain: DomainInput,
                 coord_func: Callable,
                 inv_func: Callable,
                 coord_jac: Optional[Callable] = None,
                 inv_jac: Optional[Callable] = None):

        self.d = d
        self.D = D

        self.coord_domain = parse_domain(coord_domain, (self.d, ))

        self.coord_func = coord_func
        self.inv_func = inv_func

        self.coord_jac = coord_jac
        self.inv_jac = inv_jac

        self.data = ManifoldData(d, D,
                                 local_data=False, has_coord_pts=True, has_amb_pts=True,
                                 has_coord_jac=coord_jac is not None,
                                 has_inv_jac=inv_jac is not None,
                                 has_tan_plane=coord_jac is not None)

    def generate_data(self,
                      sampling_func: Optional[Union[str, Callable]] = "uniform",
                      sampling_kwargs: Optional[dict] = None,
                      num_samples: Optional[int] = None,
                      up_to_num_samples: Optional[int] = None,
                      reset_data: Optional[bool] = False) -> None:

        if reset_data:
            self.data.reset_all_data()

        if num_samples is None:
            assert up_to_num_samples is not None
            num_samples = up_to_num_samples - self.data.num_points

        if num_samples <= 0:
            return

        if sampling_kwargs is None:
            sampling_kwargs = dict()

        if sampling_func == "uniform":
            def sampling_func(num: int, **_) -> np.ndarray:
                return np.random.uniform(*self.coord_domain, size=(num, self.d))

        coord_pts = sampling_func(num_samples, **sampling_kwargs)

        self.data.add_data("coord_pts", coord_pts)

        if self.coord_jac is not None:

            amb_pts = self.inv_func(coord_pts, batch_input=True)
            self.data.add_data("amb_pts", self.inv_func(coord_pts, batch_input=True))
            self.data.add_data("coord_jac", self.coord_jac(amb_pts, batch_input=True))

        if self.inv_jac is not None:

            inv_jac = self.inv_jac(coord_pts, batch_input=True)
            self.data.add_data("inv_jac", inv_jac)
            self.data.add_data("tan_plane", np.linalg.svd(inv_jac, full_matrices=False)[0])

    def check_correctness(self,
                          num_samples: Optional[int] = None,
                          save_data: Optional[bool] = False) -> None:

        if num_samples is None:
            assert self.data.num_points
        else:
            self.generate_data(num_samples=num_samples)

        assert self.data.has_integrity
        assert np.allclose(self.data.coord_pts, self.coord_func(self.data.amb_pts, batch_input=True),
                           atol=inv_abs_tol, rtol=inv_rel_tol)

        if self.inv_jac is not None and self.coord_jac is not None:

            assert np.allclose(self.data.coord_jac @ self.data.inv_jac,
                               np.repeat(np.identity(self.d)[np.newaxis, :], repeats=self.data.num_points, axis=0),
                               atol=inv_abs_tol, rtol=inv_rel_tol)
            assert np.allclose(self.data.tan_plane @ (self.data.tan_plane.transpose((0, 2, 1)) @ self.data.inv_jac),
                               self.data.inv_jac,
                               atol=inv_abs_tol, rtol=inv_rel_tol)

        if not save_data:
            self.data.reset_all_data()

    def save_pkl(self, path: str) -> None:
        pickle_save(self, path)

    @staticmethod
    def load_pkl(path: str) -> CoordSystem:
        return pickle_load(path)
