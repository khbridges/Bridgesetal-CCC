import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats


def remove_recompute(adata):
    # del adata.obsm['X_umap'], adata.obsm['X_pca'], adata.obsp, adata.uns, adata.obsm['X_diffmap']
    sc.tl.pca(adata, svd_solver='auto')
    sc.pp.neighbors(adata)  # using with default parameters
    sc.tl.umap(adata)
    return adata


def write_deres(excel_path, adata, group, de_key):
    # writing results to excel file
    res_cat = ['names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj']

    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')

    for g in group:
        d = {res_cat[0]: adata.uns[de_key][res_cat[0]][str(g)].tolist(),
             res_cat[1]: adata.uns[de_key][res_cat[1]][str(g)].tolist(),
             res_cat[2]: adata.uns[de_key][res_cat[2]][str(g)].tolist(),
             res_cat[3]: adata.uns[de_key][res_cat[3]][str(g)].tolist(),
             res_cat[4]: adata.uns[de_key][res_cat[4]][str(g)].tolist()
             }
        df = pd.DataFrame(data=d)
        df.to_excel(writer, sheet_name=str(g))

    writer.save()


def plot_genes_bootstrap(adata, cc_geneset, cat_order, meta_slot, bar_colors, y_max):
    for gene_name in cc_geneset:
        dat = np.array(adata[:, gene_name].X.todense()).flatten()
        dat_stat = np.zeros((len(np.unique(adata.obs[meta_slot])), 3))
        dat_stat2 = np.zeros((len(np.unique(adata.obs[meta_slot])), 3))
        b = 0

        for g in cat_order:
            i = np.where(adata.obs[meta_slot] == g)[0]
            ci_info = bs.bootstrap(dat[i], stat_func=bs_stats.mean)
            dat_stat[b, 0] = ci_info.value
            dat_stat2[b, 0] = ci_info.value
            dat_stat[b, 1] = dat_stat[b, 0] - ci_info.lower_bound
            dat_stat2[b, 1] = ci_info.lower_bound
            dat_stat[b, 2] = ci_info.upper_bound - dat_stat[b, 0]
            dat_stat2[b, 2] = ci_info.upper_bound
            b = b + 1

        # plotting results as bar graphs
        fig, ax = plt.subplots(figsize=(4, 5))
        bar_ind = cat_order
        barlist = ax.bar(bar_ind, dat_stat[:, 0], yerr=[dat_stat[:, 1], dat_stat[:, 2]], align='center', ecolor='black',
                         capsize=10, color=bar_colors)
        plt.title(gene_name)
        ax.set_ylabel('ln[mRNA counts + 1]')
        ax.set_xticks(np.arange(len(bar_ind)))
        ax.set_xticklabels(bar_ind, rotation=75)
        plt.ylim([0, y_max])
        # plt.tight_layout()

        return dat_stat2


def comb_rep(adata, slot_name):
    adata.obs['sample'] = [adata.obs[slot_name][h].split()[0] for h in range(adata.shape[0])]
    return adata


def find_clust(data, clustermap, num_clust, color_pal, row_or_col):
    if row_or_col == 'row':
        rc_ind = 0
        dendro_link = clustermap.dendrogram_row.linkage
    else:
        rc_ind = 1
        dendro_link = clustermap.dendrogram_col.linkage

    cluster_iter = np.zeros((data.shape[rc_ind], data.shape[rc_ind]))
    cluster_iter[:, 0] = np.arange(data.shape[rc_ind])

    for g in np.arange(1, data.shape[rc_ind], 1):
        cluster_iter[:, g] = cluster_iter[:, g - 1]
        pair = dendro_link[g - 1, :2]
        ind_0 = np.where(cluster_iter[:, g] == pair[0])[0]
        ind_1 = np.where(cluster_iter[:, g] == pair[1])[0]
        for y in ind_0:
            cluster_iter[y, g] = g + data.shape[rc_ind] - 1
        for z in ind_1:
            cluster_iter[z, g] = g + data.shape[rc_ind] - 1

    # renumber corr # clusters & create dictionary to feed to sns clustermap
    network_pal = sns.color_palette(color_pal, len(np.unique(cluster_iter[:, data.shape[rc_ind] - num_clust])))
    network_lut = dict(zip(np.unique(cluster_iter[:, data.shape[rc_ind] - num_clust]), network_pal))
    network_colors = pd.Series(cluster_iter[:, data.shape[rc_ind] - num_clust],
                               index=cluster_iter[:, data.shape[rc_ind] - num_clust]).map(network_lut)

    return network_lut, network_colors
