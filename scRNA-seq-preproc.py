# preprocessing pipeline from https://scanpy-tutorials.readthedocs.io/en/multiomics/cite-seq/pbmc5k.html

# importing necessary modules & setting figure parameters
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.colors import rgb2hex

sc.logging.print_versions()
sc.set_figure_params(figsize=(4, 4), fontsize=20)

# INITIAL READ & PREPROCESSING
# loading all files, splitting into rna & protein, do some basic preprocessing
paths = ['/Users/katebridges/Downloads/20211004_scRNAseq/BD1/',
         '/Users/katebridges/Downloads/20211004_scRNAseq/BD2/',
         '/Users/katebridges/Downloads/20211004_scRNAseq/BD3/',
         '/Users/katebridges/Downloads/20211004_scRNAseq/BD4/',
         '/Users/katebridges/Downloads/20211004_scRNAseq/BD5/',
         '/Users/katebridges/Downloads/20211004_scRNAseq/BD6/',
         '/Users/katebridges/Downloads/20211004_scRNAseq/BD7/']
samples = ['BD1', 'BD2', 'BD3', 'BD4', 'BD5', 'BD6', 'BD7']

# storing both RNA and protein data (CITE-seq)
adatas_rna = []
adatas_protein = []

# setting filtering parameters - curr only using minimum params for downstream filtering
filt_param = {key: '' for key in samples}
filt_param['BD1'] = {'min_counts': 1000, 'max_counts': 70000, 'min_genes': 200, 'max_genes': 7000}

filt_param['BD2'] = {'min_counts': 1000, 'max_counts': 60000, 'min_genes': 200, 'max_genes': 7000}

filt_param['BD3'] = {'min_counts': 1000, 'max_counts': 60000, 'min_genes': 200, 'max_genes': 6500}

filt_param['BD4'] = {'min_counts': 1000, 'max_counts': 70000, 'min_genes': 200, 'max_genes': 8000}

filt_param['BD5'] = {'min_counts': 1000, 'max_counts': 60000, 'min_genes': 200, 'max_genes': 7000}

filt_param['BD6'] = {'min_counts': 1000, 'max_counts': 60000, 'min_genes': 200, 'max_genes': 6500}

filt_param['BD7'] = {'min_counts': 1000, 'max_counts': 60000, 'min_genes': 200, 'max_genes': 6500}

for path1, sample in zip(paths, samples):
    adata = sc.read_10x_h5(path1 + 'filtered_feature_bc_matrix.h5', gex_only=False)
    adata.var_names_make_unique()
    adata.layers['counts'] = adata.X.copy()

    # split into protein & rna objects for ease of preprocessing
    protein = adata[:, adata.var["feature_types"] == "Antibody Capture"].copy()
    rna = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()

    # now onto RNA preprocessing
    rna.var["mito"] = rna.var_names.str.startswith("mt-")  # lowercase for mouse data
    sc.pp.calculate_qc_metrics(rna, qc_vars=["mito"], inplace=True)

    # QC plots
    # sc.pl.highest_expr_genes(rna, n_top=20)  # , save=True)
    # sc.pl.violin(rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mito'], multi_panel=True)  # , save='_QC_basic.pdf')
    # sc.pl.violin(rna, 'total_counts', log=True, cut=0)  # , save='_QC_counts.pdf')
    # sc.pl.scatter(rna, 'total_counts', 'n_genes_by_counts', color='pct_counts_mito')  # , save='_QC_countsvgenes.pdf')
    # sc.pl.violin(rna, 'pct_counts_mito')
    # plt.axhline(0.055, color='orange')
    # plt.savefig(fig_dir + 'violin_percentmito.pdf'), plt.close()

    # plt.figure()
    # sns.distplot(rna.obs['total_counts'], kde=False, bins=60)
    # plt.savefig(fig_dir + 'distribution_n_counts.pdf'), plt.close()

    # plt.figure()
    # sns.distplot(rna.obs['total_counts'][rna.obs['total_counts'] < 30000], kde=False, bins=60)
    # plt.axvline(filt_param[sample]['min_counts'])
    # plt.savefig(fig_dir + 'distribution_zoom_n_counts.pdf'), plt.close()

    # plt.figure()
    # p3 = sns.distplot(rna.obs['n_genes_by_counts'], kde=False, bins=60)
    # plt.axvline(filt_param[sample]['min_genes'])
    # plt.axvline(filt_param[sample]['max_genes'])
    # plt.savefig(fig_dir + 'distribution_n_genes.pdf'), plt.close()

    # plt.figure()
    # sns.distplot(rna.obs['n_genes_by_counts'][rna.obs['n_genes_by_counts'] < 4000], kde=False, bins=60)
    # plt.axvline(filt_param[sample]['min_genes'])
    # plt.savefig(fig_dir + 'distribution_zoom_n_genes.pdf'), plt.close()

    # let's do the filtering & normalization
    print('Filter sample: {}'.format(sample))
    print('Number of cells before filters: {:d}'.format(rna.n_obs))

    # minimum filters cells not sequenced deeply enough
    sc.pp.filter_cells(rna, min_counts=filt_param[sample]['min_counts'])

    # filtering out high mito genes (dead) and cells within gene range
    rna = rna[rna.obs['pct_counts_mito'] < 15, :]
    sc.pp.filter_cells(rna, min_genes=filt_param[sample]['min_genes'])
    print('Number of cells after filters: {:d}'.format(rna.n_obs))

    # ####################### Normalization ################
    # per cell normalization to support subsetting samples in future analyses
    sc.pp.normalize_per_cell(rna, counts_per_cell_after=1e6)
    sc.pp.log1p(rna)

    # Replot QC parameters after normalization
    sc.pl.scatter(rna, 'n_counts', 'n_genes', color='pct_counts_mito')  # , save='_QC_postnorm_countsvsgenes.pdf')

    adatas_protein.append(protein[rna.obs_names, :].copy())
    adatas_rna.append(rna.copy())

del rna, protein
adata_rna = adatas_rna[0].concatenate(adatas_rna[1:], batch_key='sample', batch_categories=samples)
adata_protein = adatas_protein[0].concatenate(adatas_protein[1:], batch_key='sample', batch_categories=samples)
del adatas_rna, adatas_protein

# combining into one dataset
adata_rna.obsm["protein"] = adata_protein.to_df()

# and a few final metrics before we write to file...
print('Final dataset:')
print(adata_rna.obs['sample'].value_counts())

# write checkpoint to file
adata_rna.write('/Users/katebridges/PycharmProjects/test/20211021_citeseq_preprocessed.h5ad')

# HASHTAGGING ANALYSIS
# fitting distribution of each hashtag with GMMs to avoid arbitrary threshold setting
adata = adata_rna.copy()
hash_onoff = np.zeros((adata.obsm['protein'].shape[0], 6))

for j in np.arange(1, 7):
    hash1 = np.log10(adata.obsm['protein'][adata.obsm['protein'].columns[j]]+1).values
    # let's fit GMM to distribution of each hashtag antibody, should help us distinguish OFF/ON
    N = np.arange(1, 5)  # fitting models with 1-5 components (not > 5 for simplicity's sake)
    models = [None for p in range(len(N))]
    # fitting a GMM for each possible component combo
    for k in range(len(N)):
        models[k] = GaussianMixture(N[k]).fit(hash1.reshape(-1, 1))
    # compute the AIC and the BIC
    BIC = [m.bic(hash1.reshape(-1, 1)) for m in models]  # Lauffenburger paper chooses best GMM by minimizing BIC
    # finding cluster labels
    M_best = models[np.argmin(BIC)]
    gmm_labels = M_best.predict(hash1.reshape(-1, 1))
    # plotting results
    plt.figure()
    sns.distplot(hash1, kde=True, bins=60)
    for k in np.unique(gmm_labels):
        plt.axvline(np.mean(hash1[np.where(gmm_labels == k)]), color='orange')

    # now we need to label "high" cluster as 1, and "low" cluster(s) as 0
    ind_max = np.argmax(hash1)
    on_marker = np.where(gmm_labels == gmm_labels[ind_max])[0]
    for b in on_marker:
        hash_onoff[b, j-1] = 1

# writing this info to metadata
adata.obsm['hash'] = hash_onoff

# now let's remove cells labeled with zero or more than one hashtag
adata_clean = adata[np.where(np.sum(adata.obsm['hash'], axis=1) == 1)[0], :]

# adding additional metadata slot so I can highlight treatment + day
hash_num = []
for d in np.arange(adata_clean.obsm['hash'].shape[0]):
    if np.sum(adata_clean.obsm['hash'][d, :]) > 0:
        hash_num.append(adata_clean.obs['sample'][d] + ' ' + str(np.where(adata_clean.obsm['hash'][d, :] == 1)[0][0] + 1))
    else:
        hash_num.append(adata_clean.obs['sample'][d] + ' ' + str(0))

sample_dict = {'BD1 0': 'Ctrl day unknown',
               'BD1 1': 'Ctrl d1',
               'BD1 2': 'Ctrl d1',
               'BD1 3': 'Ctrl d3',
               'BD1 4': 'Ctrl d3',
               'BD1 5': 'Ctrl d6',
               'BD1 6': 'Ctrl d6',
               'BD2': 'Ctrl d8',
               'BD3': 'CD40ag d8',
               'BD4': 'CPI hi d8',
               'BD5': 'CPI lo d8',
               'BD6': 'TTx d8',
               'BD7': 'TTx d10'
               }


sample_info = []
for z in hash_num:
    for g in np.arange(len(sample_dict)):
        cat = list(sample_dict.items())[g][0]
        if z.find(cat) != -1:
            sample_info.append(list(sample_dict.items())[g][1])

adata_clean.obs['sample_cond'] = sample_info

# write to file for easy recall
adata_clean.write('/Users/katebridges/PycharmProjects/test/20211105_citeseq_hashed.h5ad')


