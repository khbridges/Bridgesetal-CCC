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
from da_func import nhood_expression_mapping
from da_func import highlight_NICHEScluster
from useful_func import plot_genes_bootstrap

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

# read ICB dosing data in from checkpoint
cpi = sc.read('/Users/katebridges/Downloads/cpi-20220809.h5ad')

# encoding replicates for stat analysis
cpi.obs['replicate'] = encode_replicates(cpi, None)

# encoding condition & replicate info for downstream analyses
dose_dict = {'Ctrl d8': 0,
             'CPI lo d8': 25,
             'CPI hi d8': 100}
cpi.obs['cond_continuous'] = cpi.obs["sample_cond"].map(dose_dict).astype(int)
cpi.obs['sample_rep'] = build_samplerep(cpi, 'cond_continuous', 'replicate')

# plotting data in UMAP space by treatment, color by celltype (S4C)
for b in ['BD2', 'BD5', 'BD4']:
    sc.pl.umap(cpi[cpi.obs['sample'].str.contains(b)], color='celltype', palette=celltype_dict, s=25)
    plt.xlim([np.min(cpi.obsm['X_umap'], axis=0)[0]-1, np.max(cpi.obsm['X_umap'], axis=0)[0]+1])
    plt.ylim([np.min(cpi.obsm['X_umap'], axis=0)[1]-1, np.max(cpi.obsm['X_umap'], axis=0)[1]+1])

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
cpi.obs['grouping'] = cpi.obs['celltype'].map(celltype_map)
cpi_lim = cpi[cpi.obs['grouping'].str.contains('Macro') | cpi.obs['grouping'].str.contains('DC') |
               cpi.obs['grouping'].str.contains('T cell')]

# write to file for NICHES network generation (done in R - see run-niches.R)
cpi_lim.write('/Users/katebridges/Downloads/ICBdoses-LIM.h5ad')

# reading in NICHES network from checkpoint & viz (4A)
cpi_comm = sc.read('/Users/katebridges/niches_alra_cpid8_iter2.h5ad')
cpi_comm = comb_rep(cpi_comm, 'Condition')

for k in ['BD2', 'BD5', 'BD4']:
    sc.pl.umap(cpi_comm[cpi_comm.obs['sample'].str.contains(k)], color='VectorType', s=30)
    plt.xlim([np.min(cpi_comm.obsm['X_umap'], axis=0)[0]-1, np.max(cpi.obsm['X_umap'], axis=0)[0]+1])
    plt.ylim([np.min(cpi_comm.obsm['X_umap'], axis=0)[1]-1, np.max(cpi.obsm['X_umap'], axis=0)[1]+1])

# differential abundance testing of NICHES networks with Milo & viz (4B)
dose_dict = {'BD2': 0,
             'BD5': 25,
             'BD4': 100}
cpi_comm.obs['cond_continuous'] = cpi_comm.obs["sample"].map(dose_dict).astype(float)
cpi_comm = run_milo(cpi_comm)
milopy.plot.plot_nhood_graph(cpi_comm, alpha=0.1, min_size=5)

# grouping differentially abundant nhoods for downstream analyses (4C)
partition = cluster_nhoods(cpi, 4, 0.0155)
louvain_lab0, louvain_pal0 = plot_nhood_clusters(cpi_comm, list(partition.values()), 'Louvain cluster', alpha=0.05, min_size=5)

# writing Milo and updates NICHES networks results to file
nhood_cpi = cpi_comm.uns['nhood_adata']
nhood_cpi.write('/Users/katebridges/Downloads/20230809_cpiniches_milores.h5ad')

cpi_comm.obs['sc_louvain'] = get_sc_louvain(cpi_comm)
cpi_comm.obs['sc_louvain'] = cpi_comm.obs['sc_louvain'].astype('category')
cpi_comm.obs['louvain_str'] = list(map(str, cpi_comm.obs['sc_louvain'].values))
cpi.write('/Users/katebridges/niches_alra_cpid8_iter3_MILO_20230809.h5ad')

# exploration of differentially predicted L-R axes
sc.tl.rank_genes_groups(cpi_comm, 'louvain_str', method='wilcoxon', key_added='louvain-wilc')
write_deres('/Users/katebridges/Downloads/20230809_CPIniches_wilc.xlsx', cpi_comm, np.unique(cpi_comm.obs['sc_louvain']), 'louvain-wilc')

# viz of il10 expression across DA nhoods (4F)
il10_map = nhood_expression_mapping(cpi_comm, 'Il10')
nhood_cpi.obs['Il10'] = nhood_cpi.obs['louvain'].map(il10_map)
sc.pl.embedding(nhood_cpi, "X_milo_graph", size=nhood_cpi.obs["Nhood_size"] * 10, color="Il10",
                edges=False, neighbors_key="nhood", frameon=False, cmap="PiYG_r", vmax=2, vmin=-2)

# viz of downstream signaling genes (4G) - cluster 33 was renumbered to 8 for viz
cpi = highlight_NICHEScluster(cpi_comm, cpi, 33)
sc.pl.umap(cpi, color='cluster33_sending', groups='Highlight')

# isolating macrophages only
cpi_macs = cpi[cpi.obs['celltype'] == 'Macrophage']
cpi_macs.layers["scaled"] = sc.pp.scale(cpi_macs, copy=True).X

# viz of z-scored expression of IL-10-inducible genes by cluster 8 macs vs other macs
sc.pl.matrixplot(cpi_macs, ['Dusp1', 'Ddit4', 'Mtor','Rheb','Akt1','Rptor'], groupby='cluster33_receiving', layer="scaled", vcenter=0, cmap='RdYlBu_r')
