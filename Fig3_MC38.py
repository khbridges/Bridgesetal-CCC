import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from da_func import encode_replicates
from da_func import build_samplerep
from useful_func import remove_recompute

# figure params & color scheme for plotting
sc.set_figure_params(figsize=(4, 4), fontsize=20)
celltype_dict = {'mregDC': sns.color_palette('tab20', 20)[2],
                 'CD4+ T cell': sns.color_palette('tab20', 20)[4],
                 'T cells': sns.color_palette('tab20', 20)[6],
                 'TAM 1': sns.color_palette('tab20', 20)[10],
                 'Monocyte': 'xkcd:light brown',
                 'TAM 2': 'xkcd:dark grey',
                 'cDC': sns.color_palette('tab20b', 20)[4],
                 'NK cells': sns.color_palette('tab20', 20)[12],
                 'Neutrophils': sns.color_palette('tab20', 20)[16],
                 'Poorly classified': 'xkcd:light grey',
                 'Treg': sns.color_palette('tab20', 20)[1]}

# reading in counts and metadata from GEO (GSE224400)
adatas = []
pools = np.array([1, 2, 3, 4])
for j in pools:
    pool = sc.read('/Users/katebridges/Downloads/GSE224400_RAW/GSM70217{}_pool{}_raw_counts.tsv.gz'.format(j+27, j))
    adatas.append(pool.copy())
adata = adatas[0].concatenate(adatas[1:], index_unique=None, batch_key='pool')

# reading in metadata
metadata = pd.read_csv('/Users/katebridges/Downloads/GSE224400_barcodes_and_metadata.csv', index_col=0)
adata.obs = metadata

# normalize and logarithmize (dealing with raw counts)
adata.raw = adata
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# dimensionality reduction
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# removal of cell types without matching populations in YR dataset
adata = adata[adata.obs['major_cell_type'] != 'B cells']
adata = adata[adata.obs['major_cell_type'] != 'pDC']
adata = adata[adata.obs['major_cell_type'] != 'Basophils']

adata = remove_recompute(adata)

# revision of myeloid annotation - clustering and manual label
mac_dc = adata[adata.obs['major_cell_type'] == 'MoMacDC']
mac_dc = remove_recompute(mac_dc)
sc.tl.leiden(mac_dc, resolution=0.5)

leiden_map = {'0': 'TAM 1',
              '1': 'TAM 1',
              '2': 'TAM 1',
              '3': 'TAM 1',
              '5': 'TAM 1',
              '7': 'TAM 1',
              '4': 'TAM 2',
              '6': 'Monocyte',
              '9': 'Monocyte',
              '8': 'mregDC',
              '10': 'cDC'}
mac_dc.obs['myeloid_revised'] = mac_dc.obs['leiden'].map(leiden_map)
celltype_revised = adata.obs['major_cell_type'].copy()
celltype_revised = celltype_revised.cat.add_categories(np.unique(mac_dc.obs['myeloid_revised']))

for h in mac_dc.obs['myeloid_revised'].index:
    celltype_revised[h] = mac_dc.obs['myeloid_revised'][h]
celltype_revised = celltype_revised.cat.remove_categories(['MoMacDC'])
adata.obs['celltype_revised'] = celltype_revised

# visualization of data by revised celltype annotation (S2B)
sc.pl.umap(adata[adata.obs['sample_condition'].str.contains('MC38')], color='celltype_revised', palette=celltype_dict, s=25)
plt.xlim([np.min(adata.obsm['X_umap'], axis=0)[0]-1, np.max(adata.obsm['X_umap'], axis=0)[0]+1])
plt.ylim([np.min(adata.obsm['X_umap'], axis=0)[1]-1, np.max(adata.obsm['X_umap'], axis=0)[1]+1])

sc.pl.umap(adata[adata.obs['sample_condition'].str.contains('aCD40')], color='celltype_revised', palette=celltype_dict, s=25)
plt.xlim([np.min(adata.obsm['X_umap'], axis=0)[0]-1, np.max(adata.obsm['X_umap'], axis=0)[0]+1])
plt.ylim([np.min(adata.obsm['X_umap'], axis=0)[1]-1, np.max(adata.obsm['X_umap'], axis=0)[1]+1])

# visualization of Cd40 expression (S2C)
sc.pl.umap(adata, color='Cd40', cmap='magma')

# limit dataset to macrophages, T cells, DCs and write out for NICHES network generation
adata_lim = adata[adata.obs['celltype_revised'].str.contains('TAM') | adata.obs['celltype_revised'].str.contains('T cell') | adata.obs['celltype_revised'].str.contains('DC')]
adata_lim = remove_recompute(adata_lim)
group_map = {'TAM 1': 'Macrophage',
             'TAM 2': 'Macrophage',
             'T cells': 'T cell',
             'mregDC': 'DC',
             'cDC': 'DC'}
adata_lim.obs['grouping'] = adata_lim.obs['celltype_revised'].map(group_map)
adata_lim.write('/Users/katebridges/Downloads/GSE224400_LIM-20231102.h5ad')

# reading in NICHES networks from checkpoint
mc38 = sc.read('/Users/katebridges/niches_alra_MC38.h5ad')
mc38_map = {'MC38_aCD40_rep1': 'CD40ag',
            'MC38_aCD40_rep2': 'CD40ag',
            'MC38_rep1': 'MC38'}
mc38.obs['sample'] = mc38.obs['Condition'].map(mc38_map)

# visualization of networks by cell type pairs (3F)
sc.pl.umap(mc38, color='VectorType', s=30)
