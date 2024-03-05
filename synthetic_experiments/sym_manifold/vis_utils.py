import os
from typing import Sequence

import matplotlib.pyplot as plt


def plot_points(datapoints, subplot_data_color_idxs, colors=None,
                title=None, col_subtitles=None, axes_labels=None,
                figsize=None, subfigsize=(6, 6), subfig_layout=None,
                marker_size=0.75, dpi=96, aspect="auto",
                view_params=None, view_params_same_per_row=True,
                save_dir=None, save_path=None, save_separate=False,
                show_plot=True,
                hide_ticks=True, hide_axes=True, hide_grid=True):

    if subfig_layout is None:
        subfig_layout = (len(subplot_data_color_idxs), 1)

    subfig_rows, subfig_cols = subfig_layout
    num_subfigs = subfig_rows * subfig_cols

    if figsize is None:
        figsize = (subfigsize[0] * subfig_cols, subfigsize[1] * subfig_rows)
    fig = plt.figure(figsize=figsize, dpi=dpi)

    if title:
        fig.suptitle(title)

    if axes_labels is True:
        axes_labels = [tuple("y" + str(idx) for idx in (data_color_idx if colors is None else data_color_idx[0]))
                       for col_idxs in subplot_data_color_idxs]
    elif axes_labels is not None and isinstance(axes_labels[0], str):
        axes_labels = (axes_labels, ) * num_subfigs

    if save_dir:

        assert os.path.isdir(save_dir)

        if save_separate:
            if isinstance(save_path, Sequence):
                save_path = [os.path.join(save_dir, path) for path in save_path]
            else:
                assert title is not None
                save_path = [os.path.join(save_dir, title) + "_" + "".join(col_idxs if colors is None else col_idxs[0])
                             for col_idxs in subplot_data_color_idxs]
        else:
            if save_path:
                save_path = os.path.join(save_dir, save_path)
            else:
                save_path = os.path.join(save_dir, title)

    num_dims = None

    for i, data_color_idx in enumerate(subplot_data_color_idxs):

        if colors is not None:
            data_idx, color_idx = data_color_idx
        else:
            data_idx, color_idx = data_color_idx, None

        num_dims = len(data_idx)

        subfig_color = colors[:, color_idx] if color_idx is not None else None
        subfig_data = tuple(datapoints[:, di] for di in data_idx)

        if num_dims == 2:
            ax = fig.add_subplot(*subfig_layout, i + 1)
        else:
            ax = fig.add_subplot(*subfig_layout, i + 1, projection='3d')

        ax.tick_params(labelsize=6)

        if hide_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
            if num_dims == 3:
                ax.set_zticks([])

        if axes_labels is None and hide_axes:
            ax.set_axis_off()
        if hide_grid:
            ax.grid(False)
        if view_params:
            if isinstance(view_params, dict):
                ax.view_init(**view_params)
            else:
                vp_idx = (i // subfig_cols) if view_params_same_per_row else i
                ax.view_init(**view_params[vp_idx])

        ax.set_aspect(aspect)

        if axes_labels:
            ax.set_xlabel(axes_labels[i][0], fontsize=7.5, labelpad=-0.5)
            ax.set_ylabel(axes_labels[i][1], fontsize=7.5, labelpad=-0.5)
            if num_dims == 3:
                ax.set_zlabel(axes_labels[i][2], fontsize=7.5, labelpad=-2)

        if col_subtitles is not None and i < subfig_cols:
            ax.set_title("Colored by " + col_subtitles[i])

        ax.set_xlim(subfig_data[0].min() - 0.01, subfig_data[0].max() + 0.01)
        ax.set_ylim(subfig_data[1].min() - 0.01, subfig_data[1].max() + 0.01)
        if num_dims == 3:
            ax.set_zlim(subfig_data[2].min() - 0.01, subfig_data[2].max() + 0.01)

        p = ax.scatter(*subfig_data, s=marker_size, c=subfig_color, cmap="cividis")

        if subfig_color is not None:
            cbar = fig.colorbar(p, ax=ax, cax=ax.inset_axes([1.05, 0.25, 0.025, 0.5]), orientation="vertical")
            cbar.ax.tick_params(labelsize=7.5)

        if save_separate:
            extent = ax.get_tightbbox().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(save_path[i], bbox_inches=extent)

    fig.tight_layout()

    if num_dims == 2:
        fig.subplots_adjust(right=0.925, top=0.925, bottom=0.075, hspace=0.2)
    else:
        fig.subplots_adjust(right=0.925, top=0.925, bottom=0.075, hspace=0.05)

    if save_path and not save_separate:
        plt.savefig(save_path)

    if show_plot:
        plt.show()
