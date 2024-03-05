import glob
import os
from sortedcontainers import SortedDict

from utils import estimate_eps

from TSLassoData import TSLassoData

from TSLasso import TSLasso
from TSLassoConfig import TSLassoConfig
from ManifoldLasso import ManifoldLasso
from ManifoldLassoConfig import ManifoldLassoConfig


def estimate_eps_for_data_folder(data_folder, d):
    data_paths = glob.glob(os.path.join(data_folder, "*_data_*.pkl"))
    data_noise_scales = (float(path[:-4].split("_")[-2]) for path in data_paths)
    return SortedDict((scale, estimate_eps(TSLassoData.load_pkl(path).data, d, rmin=0.15, rmax=5.0, ntry=25))
                      for path, scale in zip(data_paths, data_noise_scales))


# estimate epsilon for the data manifolds
# print(estimate_eps_for_data_folder("m1_data_original", d=2))
# print(estimate_eps_for_data_folder("m2_data_original", d=2))
# print(estimate_eps_for_data_folder("m3_data_original", d=3))


def run_experiments_for_data_folder(data_folder, results_folder,
                                    algorithm, algorithm_config,
                                    noise_scales, geom_eps_dict, time_run,
                                    m=None):

    if m is not None:
        algorithm_config.m = m
    if geom_eps_dict is None:
        geom_eps_dict = dict()

    def run_experiment(data, result_path, geom_eps):
        if geom_eps:
            algorithm_config.geom_eps = geom_eps
        algorithm(algorithm_config, data).run(results_path=result_path, time_run=time_run)
        algorithm_config.geom_eps = None

    data_paths = glob.glob(os.path.join(data_folder, "*_data_*.pkl"))
    data_noise_scales = (float(os.path.basename(path)[:-4].split("_")[-2]) for path in data_paths)
    data_paths_dict = dict(zip(data_noise_scales, data_paths))

    for noise_scale in noise_scales:
        data_path = data_paths_dict[noise_scale]
        exp_result_path = os.path.join(results_folder, os.path.basename(data_path).replace("data", "results"))
        run_experiment(data=TSLassoData.load_pkl(data_path),
                       result_path=exp_result_path,
                       geom_eps=geom_eps_dict.get(noise_scale, None))


tslasso_config = TSLassoConfig(apply_svd=False, svd_D=None,
                               geom_eps=None, geom_n_neighbors=200,
                               reps=25,
                               gl_n=500, gl_maxiter=500, gl_maxsearch=12, gl_l2_reg=0.0,
                               gl_tol=1e-14, gl_lr=100.0)
mlasso_dm_config = ManifoldLassoConfig(m=None,
                                       apply_svd=False, svd_D=None,
                                       geom_eps=None, geom_n_neighbors=200,
                                       embedding_algo="spectral",
                                       embedding_kwargs=dict(diffusion_maps=True, diffusion_time=1.0),
                                       reps=25,
                                       gl_n=500, gl_maxiter=500, gl_maxsearch=12, gl_l2_reg=0.0,
                                       gl_tol=1e-14, gl_lr=100.0)
mlasso_umap_config = ManifoldLassoConfig(m=None,
                                         apply_svd=False, svd_D=None,
                                         geom_eps=None, geom_n_neighbors=200,
                                         embedding_algo="umap", embedding_kwargs=dict(min_dist=0.0),
                                         reps=25,
                                         gl_n=500, gl_maxiter=500, gl_maxsearch=12, gl_l2_reg=0.0,
                                         gl_tol=1e-14, gl_lr=100.0)


# best estimated values of epsilon(kernel width) for the paper data.
m1_geom_eps_dict = {0.0: 0.20558748363608462, 0.001: 0.20558748363608462, 0.0025: 0.20558748363608462,
                    0.005: 0.20558748363608462, 0.01: 0.3114280054523255, 0.025: 0.5989727857720543,
                    0.05: 1.2666732225166226, 0.1: 2.0869061661771418, 0.25: 3.438280109668161, 0.5: 4.413269219225256}
m2_geom_eps_dict = {0.0: 0.2690942285620823, 0.001: 0.2690942285620823, 0.0025: 0.3114280054523255,
                    0.005: 0.2690942285620823, 0.01: 0.3114280054523255, 0.025: 0.646582683866758,
                    0.05: 1.3424263706442396, 0.1: 2.080895725143908, 0.25: 5.000000000000001, 0.5: 2.7871277805089325}
m3_geom_eps_dict = {0.0: 0.48274469230281486, 0.001: 0.48274469230281486, 0.0025: 0.48274469230281486,
                    0.005: 0.5586898591988101, 0.01: 0.5586898591988101, 0.025: 0.7483027661820681,
                    0.05: 1.3424263706442396, 0.1: 2.408261257399385, 0.25: 2.080895725143908, 0.5: 3.7330468658382334}

m1_data_folder_path = "m1_data_original"
m2_data_folder_path = "m2_data_original"
m3_data_folder_path = "m3_data_original"

time_experiments = False
experiment_noise_scales = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]

experiment_kwargs = [  # TS Lasso Experiments
                     dict(algorithm=TSLasso, algorithm_config=tslasso_config,
                          data_folder=m1_data_folder_path, results_folder="m1_tslasso_wotime_exp",
                          geom_eps_dict=m1_geom_eps_dict, m=None),
                     dict(algorithm=TSLasso, algorithm_config=tslasso_config,
                          data_folder=m2_data_folder_path, results_folder="m2_tslasso_wotime_exp",
                          geom_eps_dict=m2_geom_eps_dict, m=None),
                     dict(algorithm=TSLasso, algorithm_config=tslasso_config,
                          data_folder=m3_data_folder_path, results_folder="m3_tslasso_wotime_exp",
                          geom_eps_dict=m3_geom_eps_dict, m=None),
                       # MLasso DM Experiments
                     dict(algorithm=ManifoldLasso, algorithm_config=mlasso_dm_config,
                          data_folder=m1_data_folder_path, results_folder="m1_mlasso_dm_wotime_exp",
                          geom_eps_dict=m1_geom_eps_dict, m=2),
                     dict(algorithm=ManifoldLasso, algorithm_config=mlasso_dm_config,
                          data_folder=m2_data_folder_path, results_folder="m2_mlasso_dm_wotime_exp",
                          geom_eps_dict=m2_geom_eps_dict, m=2),
                     dict(algorithm=ManifoldLasso, algorithm_config=mlasso_dm_config,
                          data_folder=m3_data_folder_path, results_folder="m3_mlasso_dm_wotime_exp",
                          geom_eps_dict=m3_geom_eps_dict, m=3),
                       # MLasso UMAP Experiments
                     dict(algorithm=ManifoldLasso, algorithm_config=mlasso_umap_config,
                          data_folder=m1_data_folder_path, results_folder="m1_mlasso_umap_wotime_exp",
                          geom_eps_dict=m1_geom_eps_dict, m=2),
                     dict(algorithm=ManifoldLasso, algorithm_config=mlasso_umap_config,
                          data_folder=m2_data_folder_path, results_folder="m2_mlasso_umap_wotime_exp",
                          geom_eps_dict=m2_geom_eps_dict, m=2),
                     dict(algorithm=ManifoldLasso, algorithm_config=mlasso_umap_config,
                          data_folder=m3_data_folder_path, results_folder="m3_mlasso_umap_wotime_exp",
                          geom_eps_dict=m3_geom_eps_dict, m=3)]


for exp_kwargs in experiment_kwargs:
    run_experiments_for_data_folder(**exp_kwargs,
                                    noise_scales=experiment_noise_scales,
                                    time_run=time_experiments)
