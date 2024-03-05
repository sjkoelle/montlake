import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import os
import glob
from sortedcontainers import SortedDict

from utils import pickle_load


def get_data_from_experiment_folders(exp_folders, output_folder):

    def _get_result_dicts_from_folder(results_folder):

        result_paths = glob.glob(os.path.join(results_folder, "*_results_*.pkl"))
        result_noise_scales = tuple(float(path[:-4].split("_")[-2]) for path in result_paths)
        return SortedDict((scale, pickle_load(path)) for path, scale in zip(result_paths, result_noise_scales))

    recovery_dict, reg_path_dict, times_dict = dict(), dict(), dict()

    for exp_folder in exp_folders:

        exp_folder_base = os.path.basename(exp_folder)

        if "mlasso" in exp_folder_base:
            algo_str = "_".join(exp_folder_base.split("_")[1:3])
        else:
            algo_str = exp_folder_base.split("_")[1]

        manifold_str = exp_folder_base.split("_")[0]

        results_dict = _get_result_dicts_from_folder(exp_folder)

        for noise_scale, noise_result_dict in results_dict.items():

            reps = noise_result_dict["config"].reps
            reps_converged = noise_result_dict["sel_lambs"].shape[0]
            d = noise_result_dict["d"]
            p = noise_result_dict["p"]

            for i in range(reps):

                sel_lamb, norms, selected_funcs, num_correct_recovered = None, None, None, None

                if i < reps_converged:
                    sel_lamb = noise_result_dict["sel_lambs"][i]
                    norms = noise_result_dict["lamb_norms"][i]
                    selected_funcs = np.argsort(norms[sel_lamb])[-d:]
                    num_correct_recovered = sum(int(idx < d) for idx in selected_funcs)

                if num_correct_recovered is not None:
                    recovery_dict[(algo_str, manifold_str, noise_scale, i)] = int(num_correct_recovered == d)
                else:
                    recovery_dict[(algo_str, manifold_str, noise_scale, i)] = 0

                times_dict[(algo_str, manifold_str, noise_scale, i)] = noise_result_dict["times"][i]

            if noise_result_dict["lamb_norms"]:
                x_max = max(x for norms in noise_result_dict["lamb_norms"] for x in norms.keys())
            else:
                x_max = 0.0

            x_bin_centers = x_max - np.logspace(np.log10(0.00001), np.log10(x_max), num=50, base=10.0, endpoint=True)[
                                    ::-1]
            x_bin_centers = np.concatenate([x_bin_centers, np.array([x_max])])
            x_bin_centers[0] = 0.0

            x_bin_counts = np.zeros(x_bin_centers.shape[0])
            y_vals = np.zeros((p, x_bin_centers.shape[0]))

            for norms in noise_result_dict["lamb_norms"]:
                for x, y in norms.items():
                    x_bin = np.argmin((x_bin_centers - x) ** 2)
                    x_bin_counts[x_bin] += 1.0
                    y_vals[:, x_bin] += y

            non_zeros = x_bin_counts > 0

            ys = y_vals[:, non_zeros]
            ys /= x_bin_counts[non_zeros]

            xs = x_bin_centers[non_zeros]

            for func, func_ys in enumerate(ys):
                for x, y in zip(xs, func_ys):
                    reg_path_dict[(algo_str, manifold_str, noise_scale, func, x)] = y

    def _format_table(df):

        df['algorithm'] = pd.Categorical(df["algorithm"], ["tslasso", "mlasso_umap", "mlasso_dm"])
        df = df.replace({"tslasso": "TSLasso", "mlasso_umap": "MLasso_UMAP", "mlasso_dm": "MLasso_DM"})

        return df

    recovery_df = pd.DataFrame({"recovered": recovery_dict}).reset_index()
    recovery_df = recovery_df.rename(columns={"level_0": "algorithm", "level_1": "manifold", "level_2": "sigma", "level_3": "run"})
    _format_table(recovery_df).to_pickle(os.path.join(output_folder, "recovery_df.pkl"))

    reg_path_df = pd.DataFrame({"magnitude": reg_path_dict}).reset_index()
    reg_path_df = reg_path_df.rename(columns={"level_0": "algorithm", "level_1": "manifold", "level_2": "sigma", "level_3": "dict_func", "level_4": "lambda"})
    _format_table(reg_path_df).to_pickle(os.path.join(output_folder, "reg_path_df.pkl"))

    times_df = pd.DataFrame({"time": times_dict}).reset_index()
    times_df = times_df.rename(columns={"level_0": "algorithm", "level_1": "manifold", "level_2": "sigma", "level_3": "run"})
    _format_table(times_df).to_pickle(os.path.join(output_folder, "times_df.pkl"))


def plot_recovery(recovery_df_path):

    recovery_df = pd.read_pickle(recovery_df_path)

    sns.set_theme()

    color_palette_1 = sns.color_palette("pastel", n_colors=10)
    color_palette_2 = sns.color_palette("deep", n_colors=10)

    g = sns.catplot(recovery_df, x="sigma", y="recovered", hue="algorithm", col="manifold", kind="bar",
                    estimator="mean", errorbar=("ci", 90), width=0.65, dodge=True, aspect=1.5, legend=False,
                    palette=[color_palette_2[2], color_palette_1[0], color_palette_1[1]])

    plt.tight_layout()

    for i, row in enumerate(g.axes):
        for j, ax in enumerate(row):

            title_text = ax.title._text

            if i == 0:
                ax.set_title(title_text.split("=")[1].strip())
            else:
                ax.set_title("")

            if j == 0:
                ax.set_ylabel("")

    g.set_ylabels("")
    plt.tight_layout()
    plt.legend(loc='upper right')
    g.figure.subplots_adjust(top=0.95, bottom=0.1)
    plt.show()


def plot_speedup(times_df_path):

    times_df = pd.read_pickle(times_df_path)
    times_df['time_inv'] = 1.0 / times_df["time"].to_numpy()

    mean_df = times_df[['algorithm', 'manifold', 'sigma', "time"]].groupby(['algorithm', 'manifold', 'sigma']).agg("mean")
    mean_df = mean_df.rename(columns={"time": "mean"})

    var_df = times_df[['algorithm', 'manifold', 'sigma', "time"]].groupby(['algorithm', 'manifold', 'sigma']).agg("var", ddof=0)
    var_df = var_df.rename(columns={"time": "var"})

    var_inv_df = times_df[['algorithm', 'manifold', 'sigma', "time_inv"]].groupby(['algorithm', 'manifold', 'sigma']).agg("var", ddof=0)
    var_inv_df = var_inv_df.rename(columns={"time_inv": "var_inv"})

    times_df = pd.concat([mean_df, var_df, var_inv_df], axis=1).reset_index()

    ts_lasso_df = times_df[times_df["algorithm"] == "TSLasso"]
    mlasso_umap_df = times_df[times_df["algorithm"] == "MLasso_UMAP"]
    mlasso_dm_df = times_df[times_df["algorithm"] == "MLasso_DM"]

    mlasso_umap_df["speedup_mean"] = mlasso_umap_df["mean"].to_numpy() / ts_lasso_df["mean"].to_numpy()
    mlasso_umap_df["speedup_std"] = np.sqrt(mlasso_umap_df["var"].to_numpy() * ts_lasso_df["var_inv"].to_numpy())

    mlasso_dm_df["speedup_mean"] = mlasso_dm_df["mean"].to_numpy() / ts_lasso_df["mean"].to_numpy()
    mlasso_dm_df["speedup_std"] = np.sqrt(mlasso_dm_df["var"].to_numpy() * ts_lasso_df["var_inv"].to_numpy())

    times_df = pd.DataFrame(np.concatenate([mlasso_umap_df.to_numpy(), mlasso_dm_df.to_numpy()]),
                            columns=mlasso_dm_df.columns)
    times_df = times_df[["algorithm", "manifold", "sigma", "speedup_mean", "speedup_std"]]
    times_df = times_df.rename(columns={"speedup_mean": "mean", "speedup_std": "std"})
    times_df["mean"] = times_df["mean"].clip(upper=1.96)
    times_df["mean"] -= 1.0

    sns.set_theme()

    color_palette = sns.color_palette("deep", n_colors=10)

    g = sns.catplot(times_df, x="sigma", y="mean", hue="algorithm", kind="bar",
                    col="manifold", bottom=1.0, width=0.65, dodge=True, aspect=1.5, legend=False,
                    palette=[color_palette[0], color_palette[1]])

    for i, row in enumerate(g.axes):
        for j, (ax, cur_f) in enumerate(zip(row, ["m1", "m2", "m3"])):

            x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
            y_coords = [p.get_height() + 1.0 for p in ax.patches]
            ax.errorbar(x=x_coords, y=y_coords, yerr=times_df.loc[times_df["manifold"] == cur_f, "std"],
                        fmt="none", c="black", elinewidth=2)

            title_text = ax.title._text

            if i == 0:
                ax.set_title(cur_f.upper())
            else:
                ax.set_title("")

            if j == 0:
                ax.set_ylabel(title_text.split("|")[0].strip().split("=")[1].strip())

    g.set_ylabels("")
    plt.tight_layout()
    plt.legend(loc='upper right')
    g.figure.subplots_adjust(top=0.95, bottom=0.1)
    plt.show()


def plot_reg_path(reg_path_df_path, manifold_name, d):

    reg_path_df = pd.read_pickle(reg_path_df_path)

    reg_path_df = reg_path_df[reg_path_df["manifold"] == manifold_name]
    reg_path_df = reg_path_df[reg_path_df["sigma"].isin([0.0, 0.025, 0.05])]
    reg_path_df["magnitude"] = reg_path_df["magnitude"].clip(upper=12.0)
    reg_path_df["is_true_func"] = reg_path_df["dict_func"] < d

    sns.set_theme()

    color_palette = sns.color_palette("deep", n_colors=10)

    g = sns.relplot(
        data=reg_path_df,
        x="lambda", y="magnitude", row="algorithm", col="sigma", kind="line",
        size="is_true_func", hue="dict_func", sizes=[0.6, 2.4], aspect=1., facet_kws=dict(sharex=False),
        palette=[color_palette[0]] * d + [color_palette[1]] * 36, legend=False)

    for i, row in enumerate(g.axes):
        for j, ax in enumerate(row):

            title_text = ax.title._text

            if i == 0:
                ax.set_title(title_text.split("|")[-1].strip(),  fontsize=12)
            else:
                ax.set_title("")

            if j == 0:
                ax.set_ylabel(title_text.split("|")[0].strip().split("=")[1].strip(), fontsize=12)

            if i < len(g.axes) - 1:
                ax.get_xaxis().set_ticklabels([])
            else:
                ax.tick_params(axis='x', labelsize=8)
                ax.set_xlabel("lambda", fontsize=12)

            ax.get_yaxis().set_ticklabels([])

    plt.tight_layout()
    g.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.06)
    plt.show()


# create pandas dfs from the tslasso and mlasso result folders. The dataframes thus created are expected as inputs
# for the visualizations below.

# time_experiment_folders = ["m1_mlasso_dm_time_exp", "m1_mlasso_umap_time_exp", "m1_tslasso_time_exp",
#                            "m2_mlasso_dm_time_exp", "m2_mlasso_umap_time_exp", "m2_tslasso_time_exp",
#                            "m3_mlasso_dm_time_exp", "m3_mlasso_umap_time_exp", "m3_tslasso_time_exp"]
#
# wotime_experiment_folders = ["m1_mlasso_dm_wotime_exp", "m1_mlasso_umap_wotime_exp", "m1_tslasso_wotime_exp",
#                              "m2_mlasso_dm_wotime_exp", "m2_mlasso_umap_wotime_exp", "m2_tslasso_wotime_exp",
#                              "m3_mlasso_dm_wotime_exp", "m3_mlasso_umap_wotime_exp", "m3_tslasso_wotime_exp"]
#
# get_data_from_experiment_folders(exp_folders=time_experiment_folders, output_folder="time_results_df")
# get_data_from_experiment_folders(exp_folders=wotime_experiment_folders, output_folder="wotime_results_df")

plot_recovery(recovery_df_path="wotime_results_df/recovery_df.pkl")
plot_speedup(times_df_path="time_results_df/times_df.pkl")

plot_reg_path(reg_path_df_path="wotime_results_df/reg_path_df.pkl", manifold_name="m1", d=2)
plot_reg_path(reg_path_df_path="wotime_results_df/reg_path_df.pkl", manifold_name="m2", d=2)
plot_reg_path(reg_path_df_path="wotime_results_df/reg_path_df.pkl", manifold_name="m3", d=3)
