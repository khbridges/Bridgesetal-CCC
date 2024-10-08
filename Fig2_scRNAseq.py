import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# read all d8 data in from checkpoint
all_d8 = sc.read('/Users/katebridges/Downloads/20220915_cd40only.h5ad')

# plotting data in UMAP space by treatment, color by celltype (2B)
for j in ['BD2', 'BD5', 'BD4', 'BD3', 'BD6']:
    sc.pl.umap(all_d8[all_d8.obs['sample'].str.contains(j)], color='celltype', palette=celltype_dict, s=25)
    plt.xlim([np.min(all_d8.obsm['X_umap'], axis=0)[0]-1, np.max(all_d8.obsm['X_umap'], axis=0)[0]+1])
    plt.ylim([np.min(all_d8.obsm['X_umap'], axis=0)[1]-1, np.max(all_d8.obsm['X_umap'], axis=0)[1]+1])

# viz of data by Cd40 expression (2C)
sc.pl.umap(all_d8, color='Cd40', s=25, cmap='magma')

# investigation of macrophage gene expression heterogeneity in response to CD40ag (2D)
# isolating macrophages only
mac_all = all_d8[all_d8.obs['celltype'].str.contains('Macro')]

# isolating analyses to highly variable genes only
sc.pp.highly_variable_genes(mac_all, max_mean=np.inf)
mac_lim = mac_all[:, mac_all.var['highly_variable']]

# for control and CD40ag-inclusive regimens only:
# calculating mean and fano factor for all highly variable genes
bd2 = pd.DataFrame({'gene': mac_lim.var_names,
                    'mean': np.array(mac_lim[mac_lim.obs['sample'].str.contains('BD2')].X.mean(axis=0)).flatten(),
                    'var': np.array(mac_lim[mac_lim.obs['sample'].str.contains('BD2')].X.todense().var(axis=0)).flatten()})
bd2['fano'] = bd2['var']/bd2['mean']
bd2['fano'] = bd2['fano'].fillna(value=0)

bd3 = pd.DataFrame({'gene': mac_lim.var_names,
                    'mean': np.array(mac_lim[mac_lim.obs['sample'].str.contains('BD3')].X.mean(axis=0)).flatten(),
                    'var': np.array(mac_lim[mac_lim.obs['sample'].str.contains('BD3')].X.todense().var(axis=0)).flatten()})
bd3['fano'] = bd3['var'] / bd3['mean']
bd3['fano'] = bd3['fano'].fillna(value=0)

bd6 = pd.DataFrame({'gene': mac_lim.var_names,
                    'mean': np.array(mac_lim[mac_lim.obs['sample'].str.contains('BD6')].X.mean(axis=0)).flatten(),
                    'var': np.array(mac_lim[mac_lim.obs['sample'].str.contains('BD6')].X.todense().var(axis=0)).flatten()})
bd6['fano'] = bd6['var'] / bd6['mean']
bd6['fano'] = bd6['fano'].fillna(value=0)

plot_df = pd.DataFrame({'gene': mac_lim.var_names,
                        'bd3-bd2_mean': np.log2(bd3['mean']/bd2['mean']),
                        'bd3-bd2_fano': np.log2(bd3['fano']/bd2['fano']),
                        'bd6-bd2_mean': np.log2(bd6['mean'] / bd2['mean']),
                        'bd6-bd2_fano': np.log2(bd6['fano'] / bd2['fano'])})

# CD40 mono v ctrl
sns.scatterplot(data=plot_df, x='bd3-bd2_mean', y='bd3-bd2_fano')
plt.xlim([-7,5])
plt.ylim([-3,2])

# ICB + CD40 v ctrl
sns.scatterplot(data=plot_df, x='bd6-bd2_mean', y='bd6-bd2_fano')
plt.xlim([-7,5])
plt.ylim([-3,2])
