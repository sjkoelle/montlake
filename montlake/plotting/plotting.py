# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/plotting.plotting.ipynb (unless otherwise specified).

__all__ = ['width', 'plot_reg_path_ax_lambdasearch_customcolors_names', 'get_cmap', 'get_names', 'plot_cosines',
           'plot_cosines_cluster']

# Cell
import numpy as np
from matplotlib import rcParams
from pylab import rcParams
import matplotlib.pyplot as plt
import math
import seaborn as sns
rcParams['figure.figsize'] = 25, 10

def width(p,w):
    if p > 1.:
        output = 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)
    else:
        output = w
    return(output)

def plot_reg_path_ax_lambdasearch_customcolors_names(axes, coeffs, xaxis, fig, colors, names):
    p = coeffs.shape[3]
    q = coeffs.shape[1]
    gnames = np.asarray(list(range(p)), dtype=str)

    # xlabel = r"$\displaystyle \lambda$"
    # ylabel = r"$\displaystyle \|\hat \beta_{j}\|_2$"
    rcParams['axes.titlesize'] = 30
    plt.rc('text', usetex=True)

    # maxes = np.zeros(q)
    # for k in range(q):
    #     maxes[k] = np.linalg.norm(coeffs[:, k, :, :], axis=1).max()
    # normax = maxes.max()
    normax = np.sqrt(np.sum(np.sum(np.sum(coeffs ** 2, axis=1), axis=1), axis=1).max())

    for k in range(q):
        for j in range(p):
            toplot = np.linalg.norm(coeffs[:, k, :, j], axis=1)
            w = .15
            widths = np.asarray([width(xaxis[i], w) for i in range(len(xaxis))])
            # axes[k+1].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
            axes[k + 1].plot(xaxis, toplot, 'go--', linewidth=10, markersize=0, alpha=1.,
                             color=colors[j], label=gnames[j])
    for j in range(p):
        toplot = np.linalg.norm(np.linalg.norm(coeffs[:, :, :, j], axis=2), axis=1)
        # axes[0].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
        axes[0].plot(xaxis, toplot, 'go--', linewidth=10, markersize=0, alpha=.5,
                     color=colors[j], label=gnames[j])

    kkk = xaxis.copy()
    kkk.sort()

    # xupperindex = np.min(np.where(np.sum(np.sum(np.sum(coeffs**2, axis = 1), axis = 1), axis = 1) ==0)[0])

    for k in range(1 + q):
        axes[k].tick_params(labelsize=50)
        axes[k].set_xscale('symlog')
        axes[k].set_yscale('symlog')
        axes[k].set_ylim(bottom=0, top=normax)
        # axes[k].set_xlim(left = 0, right = xaxis[xupperindex])
        if (k == 0):
            tixx = np.hstack(
                [np.asarray([0]), 10 ** np.linspace(math.floor(np.log10(normax)), math.floor(np.log10(normax)) + 1, 2)])
        if k != 0:
            # axes[k].set_yticks(tixx)
            axes[k].set_yticklabels([])
        if k != q:
            axes[k + 1].set_title(names[k], fontsize=40)
            # axes[k + 1].set_title(r"$\phi_{{{}}}$.format(k)")
        if k == 0:
            axes[k].set_title("Combined", fontdict={'fontsize': 50})
    for k in range(1 + q):
        axes[k].grid(True, which="both", alpha=True)
        axes[k].set_xlabel(r"$\lambda$", fontsize=50)
        axes[k].set_xticklabels([])
        axes[k].set_xticks([])

    axes[0].set_ylabel(r"$||\beta_j||$", fontsize=50)


def get_cmap(subset):

    cmap = plt.get_cmap('rainbow',len(subset))
    colors_subset = np.zeros((len(subset),4))
    for s in range(len(subset)):
        colors_subset[s] = cmap(s)

    return(colors_subset)

def get_names(subset):

    names = np.zeros(len(subset), dtype = object)
    for s in range(len(subset)):
        names[s] = r"$g_{{{}}}$".format(subset[s])
    return(names)

def plot_cosines(cosines, ax, colors, names):
    p = cosines.shape[0]
    sns.heatmap(cosines, ax=ax, vmin=0., vmax=1.)
    #    ax = sns.heatmap(x, cmap=cmap)
    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=60)

    for xtick, color in zip(ax.get_xticklabels(), colors):
        xtick.set_color(color)
    for ytick, color in zip(ax.get_yticklabels(), colors):
        ytick.set_color(color)
    #ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=500 / p)
    #ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=500 / p)
    ax.set_xticklabels(names, fontsize=500 / p)
    #ax.set_yticklabels(names, fontsize=500 / p, rotation = 90)
    ax.set_yticklabels(names, fontsize=500 / p, rotation = 0)

    ax.set_ylabel(r"$g_{j'}$", fontsize=90)
    ax.set_xlabel(r"$g_{j}$", fontsize=90)
    # ax.set_title(r"$\text{hi}$")
    ax.set_title(
        r"$\frac{1}{n'} \sum_{i = 1}^{n'} \frac{ |\langle grad_{\mathcal M} g_j (\xi_i) ,grad_{\mathcal M} g_{j'} (\xi_i)\rangle|}{\|grad_{\mathcal M} g_j (\xi_i) \| \| grad_{\mathcal M} g_{j'}(\xi_i) \|}$",
        fontsize=80)

# Cell
def plot_cosines_cluster(cos):

    clustermap = sns.clustermap(cos)
    clustermap.ax_row_dendrogram.set_visible(False)
    clustermap.ax_col_dendrogram.set_visible(False)

    sns.set(font_scale=2)
    f, axarr = plt.subplots(1,1, figsize=(15, 15))
    sns.heatmap(cos[clustermap.dendrogram_col.reordered_ind][:,clustermap.dendrogram_col.reordered_ind], ax = axarr, vmin = 0., vmax = 1.)
    axarr.set_title(r"$\frac{1}{n'} \sum_{i = 1}^{n'} \frac{ | \langle grad_{\mathcal M} g_j (\xi_i) ,grad_{\mathcal M} g_{j'} (\xi_i)\rangle}{\|grad_{\mathcal M} g_i (\xi_i) \| \| grad_{\mathcal M} g_j(\xi_i) \|} $" ,
                    fontsize = 80,pad = 50)
    axarr.set_xticklabels([])
    axarr.set_yticklabels([])
    axarr.set_xticks([])
    axarr.set_yticks([])
    axarr.set_xlabel(r'$g_j$', fontsize= 90)
    axarr.set_ylabel(r"$g_{j'}$", fontsize= 90)
    cbar = axarr.collections[0].colorbar
    cbar.ax.tick_params(labelsize=60)
    plt.tight_layout()
    #plt.savefig('/Users/samsonkoelle/Downloads/manigrad-100818/mani-samk-gradients/Figures/figure_for_jmlr/eth_fulldict_cosines')
