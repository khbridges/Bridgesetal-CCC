import scanpy as sc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import milopy

from da_func import encode_replicates
from da_func import build_samplerep
from useful_func import comb_rep
from da_func import run_milo
from da_func import plot_durable_clusters
from da_func import cluster_nhoods
from da_func import get_sc_louvain
from useful_func import write_deres
from da_func import highlight_ind


# figure params & color scheme for plotting
sc.set_figure_params(figsize=(4, 4), fontsize=20)
celltype_dict = {'Basophil': sns.color_palette('tab20', 20)[0],
                 'CCR7+ DC': sns.color_palette('tab20', 20)[2],
                 'CD4+ T cell': sns.color_palette('tab20', 20)[4],
                 'CD8+ T cell': sns.color_palette('tab20', 20)[6],
                 'Fibroblast': sns.color_palette('tab20', 20)[8],
                 'Macrophage': sns.color_palette('tab20', 20)[10],
                 'cDC2': sns.color_palette('tab20b', 20)[4],
                 'NK cell': sns.color_palette('tab20', 20)[12],
                 'Neutrophil': sns.color_palette('tab20', 20)[16],
                 'Poorly classified': 'xkcd:light grey',
                 'Treg': sns.color_palette('tab20', 20)[1],
                 'Tumor cell': sns.color_palette('tab20', 20)[3],
                 'XCR1+ DC': sns.color_palette('tab20', 20)[5]}

# read ICB lo + CD40ag d8 v d10 data in from checkpoint
d8v10 = sc.read('/Users/katebridges/Downloads/20221201_cpicd40_d8-10.h5ad')

# encoding replicates for stat analysis
d8v10.obs['replicate'] = encode_replicates(d8v10, None)

# encoding condition & replicate info for downstream analyses
dose_dict = {'BD6': 1,
             'BD7': 3}
d8v10.obs['cond_continuous'] = d8v10.obs["sample"].map(dose_dict).astype(int)
d8v10.obs['sample_rep'] = build_samplerep(d8v10, 'cond_continuous', 'replicate')

# plotting data in UMAP space by treatment, color by celltype (S5B)
for b in ['BD6', 'BD7']:
    sc.pl.umap(d8v10[d8v10.obs['sample'].str.contains(b)], color='celltype', palette=celltype_dict, s=25)
    plt.xlim([np.min(d8v10.obsm['X_umap'], axis=0)[0]-1, np.max(d8v10.obsm['X_umap'], axis=0)[0]+1])
    plt.ylim([np.min(d8v10.obsm['X_umap'], axis=0)[1]-1, np.max(d8v10.obsm['X_umap'], axis=0)[1]+1])

# limiting object to macrophages, DCs, and T cells for NICHES network generation (in R)
celltype_map = {'Basophil': 'Basophil',
                'CCR7+ DC': 'DC',
               'CD4+ T cell': 'T cell',
                'CD8+ T cell': 'T cell',
                'Fibroblast': 'Fibroblast',
                'Macrophage': 'Macrophage',
                'NK cell': 'NK cell',
                'Neutrophil': 'Neutrophil',
                'Poorly classified': 'Poorly classified',
                'Treg': 'T cell',
                'Tumor cell': 'Tumor cell',
                'XCR1+ DC': 'DC',
                'cDC2': 'DC'}
d8v10.obs['grouping'] = d8v10.obs['celltype'].map(celltype_map)
d8v10_lim = d8v10[d8v10.obs['grouping'].str.contains('Macro') | d8v10.obs['grouping'].str.contains('DC') |
                  d8v10.obs['grouping'].str.contains('T cell')]

# write to file for NICHES network generation (done in R - see run-niches.R)
d8v10_lim.write('/Users/katebridges/Downloads/ICB-CD40ag_d8v10-LIM.h5ad')

# reading in NICHES network from checkpoint & viz (6A)
combd10 = sc.read('/Users/katebridges/niches_alra_combd8-d10.h5ad')
combd10 = comb_rep(combd10, 'Condition')

sample_corr = {'1': 'BD6', '3': 'BD7'}
combd10.obs['sample_corr'] = combd10.obs['sample'].map(sample_corr)
for b in ['BD6', 'BD7']:
    sc.pl.umap(combd10[combd10.obs['sample'].str.contains(b)], color='VectorType', s=30)
    plt.xlim([np.min(combd10.obsm['X_umap'], axis=0)[0]-1, np.max(combd10.obsm['X_umap'], axis=0)[0]+1])
    plt.ylim([np.min(combd10.obsm['X_umap'], axis=0)[1]-1, np.max(combd10.obsm['X_umap'], axis=0)[1]+1])

# differential abundance testing of NICHES networks with Milo & viz (6B)
day_dict = {'BD6': 1, 'BD7': 3}
combd10.obs['cond_continuous'] = combd10.obs["sample_corr"].map(day_dict).astype(int)
combd10 = run_milo(combd10)
milopy.plot.plot_nhood_graph(combd10, alpha=0.1, min_size=5)

# cluster and visualize results for "durable" clusters (spatialFDR > 0.1, abs(logFC) < 1) (6C-D)
partition5 = cluster_nhoods(combd10, 2, 1.1)
louvain_lab0, louvain_pal0 = plot_durable_clusters(combd10, list(partition5.values()), 'Louvain cluster', alpha=0.1, beta=1, min_size=5)
combd10.uns['nhood_adata'].obs['louvain'] = louvain_lab0

# writing Milo and updated NICHES networks results to file
combd10.obs['sc_louvain'] = get_sc_louvain(combd10)
combd10.obs['sc_louvain'] = combd10.obs['sc_louvain'].astype('category')
combd10.obs['louvain_str'] = list(map(str, combd10.obs['sc_louvain'].values))
combd10.write('/Users/katebridges/niches_alra_combd8-d10_MILO.h5ad')

# exploration of differentially predicted L-R axes
sc.tl.rank_genes_groups(combd10, 'louvain_str', method='wilcoxon', key_added='louvain-wilc')
write_deres('/Users/katebridges/Downloads/20221202_COMB-d8vd10_wilc.xlsx', combd10, np.unique(combd10.obs['sc_louvain']), 'louvain-wilc')

# visualization of L-R axes of interest in highlighted DA clusters (6E)
# for renumbered clusters 5 (3) and 27 (11)
highlight_clust3 = ['5', '27', '-1']
d10_highlight = highlight_ind(highlight_clust3, combd10)

lr_lim2 = ['Il18—Il18r1', 'Cxcl10—Cxcr3', 'Cxcl9—Cxcr3',
           'Il27—Il27ra', 'Cd86—Cd28', 'Cd80—Cd28', 'Il15—Il2rb', 'Il15—Il2rg', 'Cxcl16—Cxcr6',
           'Il12b—Il12rb1', 'Il12b—Il12rb2', 'Tnfsf4—Tnfrsf4']

sc.pl.matrixplot(d10_highlight, lr_lim2, groupby='louvain_str', dendrogram=False, swap_axes=True,
                 standard_scale='var', cmap='Reds',
                 categories_order=['5', '27', '-1'])

