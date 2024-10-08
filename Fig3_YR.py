import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import milopy

from da_func import encode_replicates
from da_func import build_samplerep
from useful_func import comb_rep
from da_func import run_milo
from da_func import cluster_nhoods
from da_func import plot_nhood_clusters
from da_func import get_sc_louvain
from useful_func import write_deres
from da_func import highlight_ind
from da_func import highlight_NICHEScluster

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

# read ctrl vs CD40ag data in from checkpoint
cd40 = sc.read('/Users/katebridges/Downloads/20220915_cd40only.h5ad')

# encoding replicates for stat analysis
cd40.obs['replicate'] = encode_replicates(cd40, None)

# encoding condition & replicate info for downstream analyses
dose_dict = {'Ctrl d8': 0,
             'CD40ag d8': 1}
cd40.obs['cond_continuous'] = cd40.obs["sample_cond"].map(dose_dict).astype(int)
cd40.obs['sample_rep'] = build_samplerep(cd40, 'cond_continuous', 'replicate')

# plotting data in UMAP space by treatment, color by celltype (3A)
for b in ['BD2', 'BD3']:
    sc.pl.umap(cd40[cd40.obs['sample'].str.contains(b)], color='celltype', palette=celltype_dict, s=25)
    plt.xlim([np.min(cd40.obsm['X_umap'], axis=0)[0]-1, np.max(cd40.obsm['X_umap'], axis=0)[0]+1])
    plt.ylim([np.min(cd40.obsm['X_umap'], axis=0)[1]-1, np.max(cd40.obsm['X_umap'], axis=0)[1]+1])

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
cd40.obs['grouping'] = cd40.obs['celltype'].map(celltype_map)
cd40_lim = cd40[cd40.obs['grouping'].str.contains('Macro') | cd40.obs['grouping'].str.contains('DC') |
                cd40.obs['grouping'].str.contains('T cell')]

# write to file for NICHES network generation (done in R - see run-niches.R)
cd40_lim.write('/Users/katebridges/Downloads/cd40only-LIM.h5ad')

# reading in NICHES network from checkpoint
cd40_comm = sc.read('/Users/katebridges/niches_alra_cd40d8_iter2.h5ad')

# combining replicates into each condition for metadata slot
cd40_comm = comb_rep(cd40_comm, 'Condition')

# visualization in UMAP space (3A)
for c in ['BD2', 'BD3']:
    sc.pl.umap(cd40_comm[cd40_comm.obs['sample'].str.contains(c)], color='VectorType', s=30)
    plt.xlim([np.min(cd40_comm.obsm['X_umap'], axis=0)[0]-1, np.max(cd40_comm.obsm['X_umap'], axis=0)[0]+1])
    plt.ylim([np.min(cd40_comm.obsm['X_umap'], axis=0)[1]-1, np.max(cd40_comm.obsm['X_umap'], axis=0)[1]+1])

# sc.pl.umap(cd40_comm, color='celltype.Sending', s=30)
# sc.pl.umap(cd40_comm, color='celltype.Receiving', s=30)

# differential abundance testing of NICHES networks with Milo
dose_dict = {'BD2': 0,
             'BD3': 1}
cd40_comm.obs['cond_continuous'] = cd40_comm.obs["sample"].map(dose_dict).astype(int)

cd40_comm = run_milo(cd40_comm)
# sns.histplot(cd40.uns['nhood_adata'].obs['Nhood_size'])
# plt.axvline(np.median(cd40.uns['nhood_adata'].obs['Nhood_size']), color='k', linestyle='--')

# visualization of differential abundance result (3B)
milopy.plot.plot_nhood_graph(cd40, alpha=0.1, min_size=5)

# grouping differentially abundant nhoods for downstream analyses (3C)
partition1 = cluster_nhoods(cd40_comm, 2, 2.2)
louvain_lab0, louvain_pal0 = plot_nhood_clusters(cd40_comm, list(partition1.values()), 'Louvain cluster', alpha=0.1, min_size=5)

# writing Milo and updates NICHES networks results to file
nhood_cd40 = cd40_comm.uns['nhood_adata']
nhood_cd40.write('/Users/katebridges/Downloads/20221109_cd40niches_milores_iter2.h5ad')

cd40_comm.obs['sc_louvain'] = get_sc_louvain(cd40_comm)
cd40_comm.obs['sc_louvain'] = cd40_comm.obs['sc_louvain'].astype('category')
cd40_comm.obs['louvain_str'] = list(map(str, cd40_comm.obs['sc_louvain'].values))
cd40_comm.write('/Users/katebridges/niches_alra_cd40d8_iter2_MILO.h5ad')

# exploration of differentially predicted L-R axes
sc.tl.rank_genes_groups(cd40_comm, 'louvain_str', method='wilcoxon', key_added='louvain-wilc')
write_deres('/Users/katebridges/Downloads/20221109_CD40niches_wilc.xlsx', cd40_comm, np.unique(cd40_comm.obs['sc_louvain']), 'louvain-wilc')

# visualization of L-R axes of interest in highlighted DA clusters (3E)
highlight_clust0 = ['25', '20', '23', '28', '-1']
cd40_highlight = highlight_ind(highlight_clust0, cd40_comm)
lr_lim = ['Il10—Il10ra', 'Il18—Il18r1',
          'Cxcl16—Cxcr6', 'Il12b—Il12rb2', 'Il12b—Il12rb1', 'Il15—Il2rb', 'Il15—Il2rg',
          'Pdcd1lg2—Pdcd1', 'Cd274—Pdcd1',
          'Ccl22—Ccr4']

sc.pl.matrixplot(cd40_highlight, lr_lim, groupby='louvain_str', dendrogram=False, swap_axes=True,
                 categories_order=highlight_clust0, standard_scale='var', cmap='Reds')

# patterns in expression of genes downstream of predicted signaling (3F)
# focus on T cells predicted to receive IL18 signal in cluster 20
cd40_t = cd40[cd40.obs['celltype'].str.contains('T cell') | cd40.obs['celltype'].str.contains('Treg')]
cd40_t.layers["scaled"] = sc.pp.scale(cd40_t, copy=True).X

# reading in cytokine signature genes from Immune Dictionary
cd8_genes = pd.read_excel('/Users/katebridges/Downloads/41586_2023_6816_MOESM5_ESM.xlsx', sheet_name='T_cell_CD8')
il18_genes = cd8_genes[cd8_genes['Cytokine'] == 'IL-18']

# plotting mean scaled expr of top 5 IL18 genes
sc.pl.matrixplot(cd40_t, il18_genes['Gene'][:5], groupby='cluster20_receiving', swap_axes=True,
                 layer="scaled", vcenter=0, cmap='RdYlBu_r')

# calculating signature score using top 25 and plotting
sc.tl.score_genes(cd40_t, il18_genes['Gene'][:25], score_name='IL18resp')
sns.violinplot(data=cd40_t.obs, x='cluster20_receiving', y='IL18resp')

# highlighting cells in gene expression space which are predicted to participate in clusters of interest (3H)
cd40 = highlight_NICHEScluster(cd40_comm, cd40, 25)
cd40 = highlight_NICHEScluster(cd40_comm, cd40, 20)
cd40.obs['20s_25r'] = ['{} + {}'.format(cd40.obs['cluster20_sending'][j], cd40.obs['cluster25_receiving'][j]) for j in np.arange(cd40.shape[0])]
cd40.obs['20r_25s'] = ['{} + {}'.format(cd40.obs['cluster25_sending'][j], cd40.obs['cluster20_receiving'][j]) for j in np.arange(cd40.shape[0])]

map_map = {'Highlight + Highlight': 'xkcd:violet',
           'Highlight + Other': 'xkcd:light red',
           'Other + Highlight': 'xkcd:bright blue',
           'Other + Other':  'xkcd:light grey'}
sc.pl.umap(cd40, color=['20s_25r'], palette=map_map, s=45, sort_order=True, groups=['Other + Highlight', 'Highlight + Other', 'Highlight + Highlight'])
sc.pl.umap(cd40, color=['20r_25s'], palette=map_map, s=45, sort_order=True, groups=['Other + Highlight', 'Highlight + Other', 'Highlight + Highlight'])

