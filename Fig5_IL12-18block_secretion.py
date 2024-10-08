import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from useful_func import find_clust

sc.set_figure_params(figsize=(4, 4), fontsize=20)

# reading in concatenated data (incl. both single and double blocks)
filename = '/Users/katebridges/Downloads/GAP-blockingEXPR-Eve-Jan2024_gapedits.xlsx'
data = pd.read_excel(filename, sheet_name='FINAL-KB', index_col=1)
data = data.iloc[:19, :]

# only using average rows (2-3 reps per condition - some reps excluded for low signal)
data = data.loc[data.index[data.index.str.contains('Average')], :]

# limiting to Tx-activated C/Cs
cc = ['GM-CSF', 'IL-10', 'IL-9', 'IL-15', 'LIF', 'IL-7', 'IL-1b', 'M-CSF', 'IL-12p40', 'VEGF', 'IL-4',
      'CXCL2', 'CCL2', 'CXCL1', 'CCL20', 'TNFa', 'CCL19', 'G-CSF', 'IFNg', 'CCL4', 'CCL5', 'CX3CL1', 'IL-11',
      'IL-6', 'IL-5', 'CXCL10', 'IL-1a', 'CXCL9', 'CCL11', 'IL-12p70', 'CXCL5']
data_lim = data.loc[:, cc]

# function for cluster assignment from dendrogram


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


# heatmap visualization (z-score)
sns.set(font_scale=1.5)
cg = sns.clustermap(data_lim, cmap='RdYlBu_r', center=0, linewidths=0.1, linecolor='black', z_score=1, xticklabels=cc)

# viz with 6 clusters highlighted
network_lut, network_colors = find_clust(data_lim, cg, 6, "husl", 'col')
sns.clustermap(data_lim, cmap='RdYlBu_r', center=0, z_score=1, xticklabels=cc, col_colors=network_colors.values)

