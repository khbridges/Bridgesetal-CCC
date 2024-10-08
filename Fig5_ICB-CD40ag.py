import scanpy as sc
import numpy as np
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

# read ctrl vs ICB lo vs ICB lo + CD40ag data in from checkpoint
icb_cd40 = sc.read('/Users/katebridges/Downloads/20230809_ctrl-cpi-comb.h5ad')

# viz data in gene expression space (5A)
for m in ['BD2', 'BD4', 'BD6']:
    sc.pl.umap(icb_cd40[icb_cd40.obs['sample'].str.contains(m)], color='VectorType', s=30)
    plt.xlim([np.min(icb_cd40.obsm['X_umap'], axis=0)[0] - 1, np.max(icb_cd40.obsm['X_umap'], axis=0)[0] + 1])
    plt.ylim([np.min(icb_cd40.obsm['X_umap'], axis=0)[1] - 1, np.max(icb_cd40.obsm['X_umap'], axis=0)[1] + 1])

# encoding replicates for stat analysis
icb_cd40.obs['replicate'] = encode_replicates(icb_cd40, None)

# encoding condition & replicate info for downstream analyses
dose_dict = {'BD2': 0,
             'BD5': 0,
             'BD6': 1}
icb_cd40.obs['cond_continuous'] = icb_cd40.obs["sample"].map(dose_dict).astype(int)
icb_cd40.obs['sample_rep'] = build_samplerep(icb_cd40, 'cond_continuous', 'replicate')

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
icb_cd40.obs['grouping'] = icb_cd40.obs['celltype'].map(celltype_map)
icb_cd40_lim = icb_cd40[icb_cd40.obs['grouping'].str.contains('Macro') | icb_cd40.obs['grouping'].str.contains('DC') |
                        icb_cd40.obs['grouping'].str.contains('T cell')]

# write to file for NICHES network generation (done in R - see run-niches.R)
icb_cd40_lim.write('/Users/katebridges/Downloads/icb-cd40-LIM.h5ad')

# reading in NICHES network from checkpoint & initial viz (5A)
comb = sc.read('/Users/katebridges/niches_alra_comb_cpid8.h5ad')
comb = comb_rep(comb, 'Condition')
sc.pl.umap(comb, color='VectorType', s=30)
plt.xlim([np.min(comb.obsm['X_umap'], axis=0)[0]-1, np.max(comb.obsm['X_umap'], axis=0)[0]+1])
plt.ylim([np.min(comb.obsm['X_umap'], axis=0)[1]-1, np.max(comb.obsm['X_umap'], axis=0)[1]+1])

# differential abundance testing of NICHES networks with Milo
dose_dict = {'BD2': 0,
             'BD5': 0,
             'BD6': 1}
comb.obs['cond_continuous'] = comb.obs["sample"].map(dose_dict).astype(int)
comb = run_milo(comb)
milopy.plot.plot_nhood_graph(comb, alpha=0.1, min_size=5)

# grouping differentially abundant nhoods for downstream analyses (5A)
partition3 = cluster_nhoods(comb, 2, 1.25)
louvain_lab0, louvain_pal0 = plot_nhood_clusters(comb, list(partition3.values()), 'Louvain cluster', alpha=0.1, min_size=5)

# writing Milo and updates NICHES networks results to file
nhood_cpicomb = comb.uns['nhood_adata']
nhood_cpicomb.write('/Users/katebridges/Downloads/20221117_CPIcombniches_milores.h5ad')

comb.obs['sc_louvain'] = get_sc_louvain(comb)
comb.obs['sc_louvain'] = comb.obs['sc_louvain'].astype('category')
comb.obs['louvain_str'] = list(map(str, comb.obs['sc_louvain'].values))
comb.write('/Users/katebridges/niches_alra_comb_cpid8_MILO.h5ad')

# exploration of differentially predicted L-R axes
sc.tl.rank_genes_groups(comb, 'louvain_str', method='wilcoxon', key_added='louvain-wilc')
write_deres('/Users/katebridges/Downloads/20221117_COMB-CPIniches_wilc.xlsx', comb, np.unique(comb.obs['sc_louvain']), 'louvain-wilc')

# visualization of L-R axes of interest in highlighted DA clusters (5B)
highlight_clust2 = ['36', '47', '46', '44', '4', '-1']
combcpi_highlight = highlight_ind(highlight_clust2, comb)

lr_lim1 = ['Il10—Il10ra', 'Csf1—Csf1r', 'Csf2—Csf2ra', 'Csf2—Csf2rb',
           'Ifng—Ifngr1', 'Ifng—Ifngr2',
           'Cd80—Ctla4', 'Ccl22—Ccr4', 'Tnfsf4—Tnfrsf4', 'Il12b—Il12rb1', 'Il12b—Il12rb2', 'Il6—Il6ra',
           'Il15—Il2rb', 'Il15—Il2rg',
           'Il18—Il18r1',
           'Cxcl9—Cxcr3', 'Cxcl10—Cxcr3',
           'Il27—Il27ra', 'Cxcl16—Cxcr6']

sc.pl.matrixplot(combcpi_highlight, lr_lim1, groupby='louvain_str', dendrogram=False, swap_axes=True,
                 categories_order=['47', '46', '44', '4', '-1'], standard_scale='var', cmap='Reds')

# highlighting cells in gene expression space which are predicted to participate in clusters of interest (5C-D)
# considering renumbered clusters 13 (44), 15 (47), 1 (4), 14 (46), and 9 (36)
icb_cd40_lim = highlight_NICHEScluster(comb, icb_cd40_lim, 44)
icb_cd40_lim = highlight_NICHEScluster(comb, icb_cd40_lim, 47)
icb_cd40_lim = highlight_NICHEScluster(comb, icb_cd40_lim, 4)
icb_cd40_lim = highlight_NICHEScluster(comb, icb_cd40_lim, 46)
icb_cd40_lim = highlight_NICHEScluster(comb, icb_cd40_lim, 36)

map_map = {'Highlight + Highlight': 'xkcd:violet',
           'Highlight + Other': 'xkcd:light red',
           'Other + Highlight': 'xkcd:bright blue',
           'Other + Other':  'xkcd:light grey'}
icb_cd40_lim.obs['4s_46r'] = ['{} + {}'.format(icb_cd40_lim.obs['cluster4_sending'][j], icb_cd40_lim.obs['cluster46_receiving'][j])
                              for j in np.arange(icb_cd40_lim.shape[0])]
icb_cd40_lim.obs['4s_46r'] = icb_cd40_lim.obs['4s_46r'].cat.reorder_categories(['Other + Other', 'Other + Highlight', 'Highlight + Other', 'Highlight + Highlight'], ordered=True)

icb_cd40_lim.obs['46s_4r'] = ['{} + {}'.format(icb_cd40_lim.obs['cluster46_sending'][j], icb_cd40_lim.obs['cluster4_receiving'][j])
                              for j in np.arange(icb_cd40_lim.shape[0])]
icb_cd40_lim.obs['46s_4r'] = icb_cd40_lim.obs['46s_4r'].cat.reorder_categories(['Other + Other', 'Other + Highlight', 'Highlight + Other', 'Highlight + Highlight'], ordered=True)

# let's consider overlap between 4 and 46 sending/receiving (loop? - 5D)
sc.pl.umap(icb_cd40_lim, color=['4s_46r'], palette=map_map, s=45, sort_order=True, groups=['Other + Highlight', 'Highlight + Other', 'Highlight + Highlight'])
sc.pl.umap(icb_cd40_lim, color=['46s_4r'], palette=map_map, s=45, sort_order=True, groups=['Other + Highlight', 'Highlight + Other', 'Highlight + Highlight'])

# just viz cluster 9 (36) sending/receiving (5D)
sc.pl.umap(icb_cd40_lim, color=['cluster36_sending', 'cluster36_receiving'], groups='Highlight')
