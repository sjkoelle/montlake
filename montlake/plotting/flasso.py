# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/plotting.flasso.ipynb (unless otherwise specified).

__all__ = ['width', 'plot_cos_boxes', 'plot_reg_path_ax_lambdasearch_customcolors_names',
           'plot_reg_path_ax_lambdasearch_customcolors_tslasso', 'plot_reg_path_ax_lambdasearch_customcolors_norm',
           'plot_watch_custom', 'plot_watch', 'plot_reg_path_ax_lambdasearch_customcolors']

# Cell
import numpy as np
from matplotlib import rcParams
from pylab import rcParams
import matplotlib.pyplot as plt
import math
import seaborn as sns
from collections import OrderedDict
from matplotlib.patches import Rectangle
rcParams['figure.figsize'] = 25, 10

def width(p,w):
    if p > 1.:
        output = 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)
    else:
        output = w
    return(output)

def plot_cos_boxes(sup_sel, names, col, sel, d , nreps, axarr):


    sns.heatmap(col, yticklabels = names, xticklabels = names, ax = axarr, vmin = 0., vmax = 1.)
    axarr.set_xticklabels(axarr.get_xmajorticklabels(), fontsize = 30)
    axarr.set_yticklabels(axarr.get_ymajorticklabels(), fontsize = 30)

    if d == 2:
        for r in range(nreps):
            pos1 = np.where(sel == sup_sel[r,1])[0]
            pos2 =  np.where(sel == sup_sel[r,0])[0]
            axarr.add_patch(Rectangle((pos1, pos2), 1, 1,facecolor = [0,1,0,0.], hatch = '/',fill= True, edgecolor='blue', lw=1))
            axarr.add_patch(Rectangle((pos2, pos1), 1, 1,facecolor = [0,1,0,0.], hatch = '/',fill= True, edgecolor='blue', lw=1))


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


def plot_reg_path_ax_lambdasearch_customcolors_tslasso(axes, coeffs, xaxis, fig, colors, names):
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

    for j in range(p):
        toplot = np.linalg.norm(np.linalg.norm(coeffs[:, :, :, j], axis=2), axis=1)
        # axes[0].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
        axes.plot(xaxis, toplot, 'go--', linewidth=10, markersize=0, alpha=.5,
                     color=colors[j], label=gnames[j])

    kkk = xaxis.copy()
    kkk.sort()

    # xupperindex = np.min(np.where(np.sum(np.sum(np.sum(coeffs**2, axis = 1), axis = 1), axis = 1) ==0)[0])

    axes.tick_params(labelsize=50)
    axes.set_xscale('symlog')
    axes.set_yscale('symlog')
    axes.set_ylim(bottom=0, top=normax)
    # axes[k].set_xlim(left = 0, right = xaxis[xupperindex])

    tixx = np.hstack(
        [np.asarray([0]), 10 ** np.linspace(math.floor(np.log10(normax)), math.floor(np.log10(normax)) + 1, 2)])


    axes.set_title("Combined", fontdict={'fontsize': 50})

    axes.grid(True, which="both", alpha=True)
    axes.set_xlabel(r"$\lambda$", fontsize=50)
#     axes.set_xticklabels([])
#     axes.set_xticks([])

    axes.set_ylabel(r"$||\beta_j||$", fontsize=50)


def plot_reg_path_ax_lambdasearch_customcolors_norm(ax, coeffs, xaxis, fig, colors):
    p = coeffs.shape[3]
    q = coeffs.shape[1]
    gnames = np.asarray(list(range(p)), dtype=str)

    rcParams['axes.titlesize'] = 30
    plt.rc('text', usetex=True)

    normax = np.sqrt(np.sum(np.sum(np.sum(coeffs ** 2, axis=1), axis=1), axis=1).max())

    for j in range(p):
        toplot = np.linalg.norm(np.linalg.norm(coeffs[:, :, :, j], axis=2), axis=1)
        # axes[0].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
        ax.plot(xaxis, toplot, 'go--', linewidth=5, markersize=0, alpha=1.,
                color=colors[j], label=gnames[j])

    kkk = xaxis.copy()
    kkk.sort()
    ax.tick_params(labelsize=50)
    ax.set_yscale('symlog')
    ax.set_ylim(bottom=0, top=normax)
    tixx = np.hstack(
        [np.asarray([0]), 10 ** np.linspace(math.floor(np.log10(normax)), math.floor(np.log10(normax)) + 1, 2)])

    ax.grid(True, which="both", alpha=True)


def plot_watch_custom(to_plot, p, ax, colors,nreps, names = None, s=.1, fontsize = 70):

    if names is None:
        names = np.asarray(list(range(p)), dtype = str)

    theta = np.linspace(0, 2 * np.pi, 10000)
    cmap = plt.get_cmap('twilight_shifted', p)
    angles = np.linspace(0, 2 * np.pi, p + 1)
    radius = 1.
    a = radius * np.cos(theta)
    b = radius * np.sin(theta)

    ax.scatter(a, b, color='gray', s = .2, alpha=.1)
    if len(to_plot.shape) > 1:
        totes = np.sum(to_plot, axis=0)
    else:
        totes = to_plot

    for j in range(p):
        nm = names[j]
        #print(np.cos(angles[j]), np.sin(angles[j]))  # r'$test \frac{1}{}$'.format(g)
        ax.scatter(np.cos(angles[j]), np.sin(angles[j]), color=cmap.colors[j], marker='x')
        ax.text(x=1.1 * np.cos(angles[j]),
                y=1.1 * np.sin(angles[j]),
                s=r"$g_{{{}}}$".format(nm), color=colors[j],  # cmap.colors[j],
                fontdict={'fontsize': fontsize},
                horizontalalignment='center',
                verticalalignment='center')

#         ax.text(x=.9 * np.cos(angles[j]), y=.9 * np.sin(angles[j]), s=str(totes[j] / nreps), fontdict={'fontsize': 100 * s},
#                 horizontalalignment='center',
#                 verticalalignment='center')

    for j in range(p):
        ax.scatter(np.cos(angles[j]), np.sin(angles[j]), color=colors[j], marker='o', s= s* 500 * totes[j])

    if len(to_plot.shape) > 1:
        for i in range(p):
            for j in range(p):
                x_values = [np.cos(angles[j]), np.cos(angles[i])]
                y_values = [np.sin(angles[j]), np.sin(angles[i])]
                ax.plot(x_values, y_values, linewidth=to_plot[i, j] * 8*s, color='black')
#                 if to_plot[i, j] > 0:
#                     ax.text(x=np.mean(x_values),
#                             y=np.mean(y_values),
#                             s=str(to_plot[i, j] / nreps),
#                             fontdict={'fontsize': 40})  # ,

    ax.set_aspect(1)
    ax.set_axis_off()
    #ax.set_title(r"$\omega = 25$")


def plot_watch(to_plot, names, colors, ax,nreps):

    p = to_plot.shape[0]
    theta = np.linspace(0, 2 * np.pi, 10000)
    angles = np.linspace(0, 2 * np.pi, p + 1)
    radius = 1.

    a = radius * np.cos(theta)
    b = radius * np.sin(theta)
    ax.scatter(a, b, color='gray', s=.2,
               alpha=.1)

    if len(to_plot.shape) > 1:
        totes = np.sum(to_plot, axis=0)
    else:
        totes = to_plot

    for j in range(p):
        ax.scatter(np.cos(angles[j]), np.sin(angles[j]), color=colors[j], marker='x')
        ax.text(x=1.1 * np.cos(angles[j]),
                y=1.1 * np.sin(angles[j]),
                s=names[j], color=colors[j],
                fontdict={'fontsize': 60},
                horizontalalignment='center',
                verticalalignment='center')

        ax.text(x=.9 * np.cos(angles[j]), y=.9 * np.sin(angles[j]), s=str(totes[j] / nreps), fontdict={'fontsize': 30},
                horizontalalignment='center',
                verticalalignment='center')

    for j in range(p):
        ax.scatter(np.cos(angles[j]), np.sin(angles[j]), color=colors[j], marker='o', s=100 * totes[j])

    if len(to_plot.shape) > 1:
        for i in range(p):
            for j in range(p):
                x_values = [np.cos(angles[j]), np.cos(angles[i])]
                y_values = [np.sin(angles[j]), np.sin(angles[i])]
                ax.plot(x_values, y_values, linewidth=to_plot[i, j], color='black')

                if to_plot[i, j] > 0:
                    ax.text(x=np.mean(x_values),
                            y=np.mean(y_values),
                            s=str(to_plot[i, j] / nreps),
                            fontdict={'fontsize': 20})  # ,

    ax.set_aspect(1)
    ax.set_axis_off()


def plot_reg_path_ax_lambdasearch_customcolors(axes, coeffs, xaxis,fig, colors,gnames):
    p = coeffs.shape[3]
    q = coeffs.shape[1]
    #gnames = np.asarray(list(range(p)), dtype=str)

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
            axes[k + 1].plot(xaxis, toplot, 'go--', linewidth=5, markersize=0, alpha=1.,
                             color=colors[j], label=gnames[j])
    for j in range(p):
        toplot = np.linalg.norm(np.linalg.norm(coeffs[:, :, :, j], axis=2), axis=1)
        # axes[0].boxplot(toplot, positions=xaxis, showfliers=False, vert=True, widths=widths,medianprops=dict(linestyle=''))
        axes[0].plot(xaxis, toplot, 'go--', linewidth=5, markersize=0, alpha=1.,
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
            axes[k+1].set_title(r"$\phi_{{{}}}$".format(k+1), fontsize = 50)
            #axes[k + 1].set_title(r"$\phi_{{{}}}$.format(k)")
        if k == 0:
            axes[k].set_title("Combined", fontdict={'fontsize': 50})
    for k in range(1 + q):
        axes[k].grid(True, which="both", alpha=True)
        axes[k].set_xlabel(r"$\lambda$", fontsize = 50)

    axes[0].set_ylabel(r"$\|\beta\|$", fontsize = 50)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    # fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=50)
    # fig.text(0.05, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=60)
    fig.subplots_adjust(right=0.75)
    leg_ax = fig.add_axes([.8, 0.15, 0.05, 0.7])
    leg_ax.axis('off')
    leg = leg_ax.legend(by_label.values(), gnames, prop={'size': 300 / p})
    # leg.set_title('Torsion', prop={'size': Function})
    for l in leg.get_lines():
        l.set_alpha(1)
    leg_ax.set_title("$g_{j}$", fontsize = 1000/p)
    # fig.savefig(filename + 'beta_paths_n' + str(n) + 'nsel' + str(nsel) + 'nreps' + str(
    #    nreps))
