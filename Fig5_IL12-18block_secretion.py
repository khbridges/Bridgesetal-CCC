import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from useful_func import find_clust

sc.set_figure_params(figsize=(4, 4), fontsize=20)

# reading in serum cytokine data (Eve tech)
filename = '/Users/katebridges/Downloads/GAP-blockingEXPR-Eve-Jan2024.xlsx'
# excluding first column (sample names) and using treatment + rep column as index labels
data = pd.read_excel(filename, sheet_name='FINAL-KB', index_col=0, usecols=np.arange(1, 45))

sample_map = {'Untreated_1': 'Untreated', 'Untreated_2': 'Untreated',
              'Tx_1': 'Tx', 'Tx_2': 'Tx', 'Tx_3': 'Tx',
              'Tx_db-block_1': 'Tx_db-block', 'Tx_db-block_2': 'Tx_db-block', 'Tx_db-block_3': 'Tx_db-block'}


# viz in heatmap (5G)
sns.set(font_scale=1.5)
cg = sns.clustermap(data, cmap='seismic', center=0, linewidths=0.1, linecolor='black', rasterized=False, z_score=1, xticklabels=data.columns)
network_lut, network_colors = find_clust(data, cg, 11, "husl", 'col')
sns.clustermap(data, cmap='seismic', center=0, linewidths=0.1, linecolor='black', rasterized=False, z_score=1, xticklabels=data.columns, col_colors=network_colors.values)

# plotting norm trends across treatments for clusters of interest
data_norm = data.copy()/data.max(axis=0)
data_melt = pd.melt(data_norm)
data_melt['sample_id'] = np.tile(data_norm.index, 43)
data_melt['treat'] = data_melt['sample_id'].map(sample_map)
data_melt['cluster'] = np.repeat(network_colors.astype('category').cat.codes.values, 8)

sns.set_style("whitegrid")
sns.lineplot(x="treat",
             y="value",
             hue="cluster",
             data=data_melt[data_melt['cluster'] == 10],
             palette={9: list(network_colors)[np.where(data.columns == 'IFNg')[0][0]],
                      5: list(network_colors)[np.where(data.columns == 'CXCL9')[0][0]]})
plt.legend([],[], frameon=False)
plt.tight_layout()
plt.xlabel('')
plt.ylabel('')
