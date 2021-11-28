# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/plotting.main.ipynb (unless otherwise specified).

__all__ = ['plot_experiment']

# Cell
import matplotlib.pyplot as plt

# Cell
def plot_experiment(results,  positions, d,name, ncord = 6, embedding= True, ground_truth = None, colors = None, ground_truth_colors = None):

    print('loading data')
    infile = '/Users/samsonkoelle/thesis_data/processed_data_2/rigidethanol/re_nonoise_diagram_mfresults_mflasso'
    with open(infile,'rb') as inp:
        results = pickle.load(inp, pickle.HIGHEST_PROTOCOL)

    print('compute ground truth values for comparison')
    if ground_truth is not None:
        n = positions.shape[0]
        results = pool.map(lambda i: get_features(positions[i],
                                   atoms2 = ground_truth['atoms2'],
                                   atoms3 = ground_truth['atoms3'],
                                   atoms4 = ground_truth['atoms4']),
            data_stream_custom_range(list(range(n))))

    print('getting colors and names of selected functions')
    selected_lasso = results['selected_lasso']
    colors = get_cmap(selected_lasso)
    names = get_names(selected_lasso)
    selected_ts = results['selected_ts']
    colors_brute = get_cmap(selected_ts)
    names_brute = get_names(selected_ts)

    print('plotting top coordinates in feature space')
    data = results['data']
    title = name + ' top PCA'
    plot_manifold_featurespace(data,title,ncord)
    #plt.savefig('/Users/samsonkoelle/Downloads/manigrad-100818/mani-samk-gradients/Figures/figure_for_jmlr/tol30_replicate')

    print('plotting sample regularization path')
    fig, axes_all = plt.subplots(figsize=(15, 10))
    plot_reg_path_ax_lambdasearch_customcolors_norm(axes_all, results['replicates_small'][0].cs_reorder, results['replicates_small'][0].xaxis_reorder / results['replicates_small'][0].xaxis_reorder.max() , fig,colors)#axes_all[0].imshow(asdf)
    axes_all.set_title('Regularization path (single replicate)', fontsize = 40)
    axes_all.set_ylabel(r'$||\beta_j||$', fontsize = 40)
    axes_all.set_xticklabels([])
    plt.tight_layout()
    #plt.savefig('/Users/samsonkoelle/Downloads/manigrad-100818/mani-samk-gradients/Figures/figure_for_jmlr/tol30_replicate')

    if embedding:
        print('plotting selected function values')
        for s in range(len(selected_functions)):
            c = results['selected_function_values'][:,s]
            embed = results['embed']
            if n_components == 3:
                plot_manifold_3d(embed, s, alpha, c, title, title_color)
            if n_components == 2:
                plot_manifold_2d(embed, s, alpha, c, title, title_color)
            #plt.savefig('/Users/samsonkoelle/Downloads/manigrad-100818/mani-samk-gradients/Figures/figure_for_jmlr/tol30_replicate')
        for s in range(len(selected_functions_brute)):
            c = results['selected_function_brute values'][:,s]
            embed = results['embed']
            if n_components == 3:
                plot_manifold_3d(embed, s, alpha, c, title, title_color)
            if n_components == 2:
                plot_manifold_2d(embed, s, alpha, c, title, title_color)
            #plt.savefig('/Users/samsonkoelle/Downloads/manigrad-100818/mani-samk-gradients/Figures/figure_for_jmlr/tol30_replicate')
        print('plotting ground truth function values')

    print("plotting watches")
    fig, axes_all = plt.subplots(figsize=(15, 10))
    plot_watch_custom(results['supports_lasso'], p, axes_all,colors, nreps)
    axes_all.set_title('Estimated Support', fontsize = 40)
    plt.tight_layout()

    if d > 1:
        print("plotting full cosine matrix")
        cosines_full= results['replicates_small'].cosines
        plot_cosines(cosines_full, ax, colors)
        #plt.savefig('/Users/samsonkoelle/Downloads/manigrad-100818/mani-samk-gradients/Figures/figure_for_jmlr/tol30_replicate')

        print("plotting cosines of ground truth and selected")
        selected_lasso_gt = np.unique(np.concatenate(selected_lasso,ground_truth['atoms4'])) #add 234
        cuz_l = np.abs(get_cosines(np.swapaxes(results['replicates_small'][0].dg_M[:,:,subset_l0_plusgt], 1,2)))
        cuz_l0 = np.mean(cuz_l, axis = 0)
        sns.heatmap(cosines_selected, yticklabels = names_l0, xticklabels = names, ax = axarr, vmin = 0., vmax = 1.)
        axarr.set_xticklabels(axarr.get_xmajorticklabels(), fontsize = 30)
        axarr.set_yticklabels(axarr.get_ymajorticklabels(), fontsize = 30)
        for d in range(detected_values.shape[1]):
            axarr.add_patch(Rectangle((detected_values[1,d], detected_values[0,d]), 1, 1,facecolor = [0,1,0,0.], hatch = '/',fill= True, edgecolor='blue', lw=1))
        for xtick, color in zip(axarr.get_xticklabels(), colors_l0_plusgt):
            xtick.set_color(color)
        for ytick, color in zip(axarr.get_yticklabels(), colors_l0_plusgt):
            ytick.set_color(color)
        axarr.set_title(r"$\frac{1}{n'} \sum_{i = 1}^{n'} \frac{|\langle grad_{\mathcal M} g_j (\xi_i) ,grad_{\mathcal M} g_{j'} (\xi_i)\rangle|}{\|grad_{\mathcal M} g_j (\xi_i) \| \| grad_{\mathcal M} g_{j'}(\xi_i) \|} $" ,
                        fontsize = 30)
        plt.tight_layout()
        plt.yticks(rotation= 0)

        print("plotting cosines of ground truth and selected")
        selected_lasso_gt = np.unique(np.concatenate(selected_ts,ground_truth['atoms4'])) #add 234
        cuz_l = np.abs(get_cosines(np.swapaxes(results['replicates_small'][0].dg_M[:,:,subset_l0_plusgt], 1,2)))
        cuz_l0 = np.mean(cuz_l, axis = 0)
        sns.heatmap(cosines_selected, yticklabels = names_l0, xticklabels = names, ax = axarr, vmin = 0., vmax = 1.)
        axarr.set_xticklabels(axarr.get_xmajorticklabels(), fontsize = 30)
        axarr.set_yticklabels(axarr.get_ymajorticklabels(), fontsize = 30)
        for d in range(detected_values.shape[1]):
            axarr.add_patch(Rectangle((detected_values[1,d], detected_values[0,d]), 1, 1,facecolor = [0,1,0,0.], hatch = '/',fill= True, edgecolor='blue', lw=1))
        for xtick, color in zip(axarr.get_xticklabels(), colors_l0_plusgt):
            xtick.set_color(color)
        for ytick, color in zip(axarr.get_yticklabels(), colors_l0_plusgt):
            ytick.set_color(color)
        axarr.set_title(r"$\frac{1}{n'} \sum_{i = 1}^{n'} \frac{|\langle grad_{\mathcal M} g_j (\xi_i) ,grad_{\mathcal M} g_{j'} (\xi_i)\rangle|}{\|grad_{\mathcal M} g_j (\xi_i) \| \| grad_{\mathcal M} g_{j'}(\xi_i) \|} $" ,
                        fontsize = 30)
        plt.tight_layout()
        plt.yticks(rotation= 0)

    if d == 2:
        print("getting correlations with ground truth")
        coses = np.zeros((nreps,d,d))
        for r in range(nreps):
            print(r)
            rep = replicates[r]

            j1 = get_index_matching(ground_truth[0], superset)
            j1 = get_index_matching(ground_truth[1], superset)
            j3 = sel[r][0]
            j4 = sel[r][1]

            coses[r,0,0] = np.sum(np.abs(np.asarray([cosine_similarity(replicates[r].dg_M[i,:,j1], replicates[r].dg_M[i,:,j3]) for i in range(nsel)]))) / nsel
            coses[r,0,1] = np.sum(np.abs(np.asarray([cosine_similarity(replicates[r].dg_M[i,:,j1], replicates[r].dg_M[i,:,j4]) for i in range(nsel)]))) / nsel
            coses[r,1,0] = np.sum(np.abs(np.asarray([cosine_similarity(replicates[r].dg_M[i,:,j2], replicates[r].dg_M[i,:,j3]) for i in range(nsel)]))) / nsel
            coses[r,1,1] = np.sum(np.abs(np.asarray([cosine_similarity(replicates[r].dg_M[i,:,j2], replicates[r].dg_M[i,:,j4]) for i in range(nsel)]))) / nsel