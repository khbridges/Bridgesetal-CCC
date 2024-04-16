import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from da_func import encode_replicates
from da_func import build_samplerep

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

# plotting data in UMAP space by treatment, color by celltype (2B)
sc.pl.umap(cd40[cd40.obs['sample'].str.contains('BD2')], color='celltype', palette=celltype_dict, s=25)
plt.xlim([np.min(cd40.obsm['X_umap'], axis=0)[0]-1, np.max(cd40.obsm['X_umap'], axis=0)[0]+1])
plt.ylim([np.min(cd40.obsm['X_umap'], axis=0)[1]-1, np.max(cd40.obsm['X_umap'], axis=0)[1]+1])

sc.pl.umap(cd40[cd40.obs['sample'].str.contains('BD3')], color='celltype', palette=celltype_dict, s=25)
plt.xlim([np.min(cd40.obsm['X_umap'], axis=0)[0]-1, np.max(cd40.obsm['X_umap'], axis=0)[0]+1])
plt.ylim([np.min(cd40.obsm['X_umap'], axis=0)[1]-1, np.max(cd40.obsm['X_umap'], axis=0)[1]+1])

# plotting data in UMAP space by expression of CD40 (2C)
sc.pl.umap(cd40, color='Cd40', s=25, cmap='magma')

# mean & fano factor calculation for ctrl vs CD40ag-treated macrophages (S1A)
# isolating macrophages
macro = cd40[cd40.obs['celltype'].str.contains('Macro')]
macro.X = macro.layers['counts']

# cc of interest
cc = ['Tnf', 'Cxcl1', 'Ccl5', 'Ccl3', 'Il6', 'Il12b',  'Il10', 'Chil3', 'Ccl2']

# calculate Fano factor for CCs of interest from raw counts (stored in cd40.layers['counts'])
mac_calc = np.zeros((len(cd40.obs['sample_rep'].unique()), len(cc)))  # for fano
mac_calc2 = np.zeros((len(cd40.obs['sample_rep'].unique()), len(cc)))  # for mean

i = 0
for f in cd40.obs['sample_rep'].unique():
    macro_rep = pd.DataFrame(macro[macro.obs['sample_rep'].str.contains(f)][:, cc].X.todense())
    mac_calc[i, :] = (macro_rep.var()/macro_rep.mean()).values
    mac_calc2[i, :] = macro_rep.mean()
    i = i + 1

# writing results to file - for visualization in GraphPad Prism
mac_calc = pd.DataFrame(mac_calc)
mac_calc.index = cd40.obs['sample_rep'].unique()
mac_calc.columns = cc
mac_calc.to_excel('/Users/katebridges/Downloads/macroONLY_cd40-perrep_fano.xlsx')

mac_calc2 = pd.DataFrame(mac_calc2)
mac_calc2.index = cd40.obs['sample_rep'].unique()
mac_calc2.columns = cc
mac_calc2.to_excel('/Users/katebridges/Downloads/macroONLY_cd40-perrep_mean.xlsx')

# limiting object to macrophages, DCs, and T cells for NICHES network generation (in R)

