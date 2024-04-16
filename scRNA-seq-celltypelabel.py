import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pipeline_functions import training_data_select
from pipeline_functions import viz_training_data
from pipeline_functions import one_hot_encode
from pipeline_functions import cell_type_classifier
from pipeline_functions import process_label
from useful_func import remove_recompute

# figure params
sc.set_figure_params(figsize=(4, 4), fontsize=20)

# reading in preprocessed data
file_dir = '/Users/katebridges/PycharmProjects/test/20211105_citeseq_hashed.h5ad'
adata = sc.read(file_dir)

# cell type labeling pipeline
marker = ['Cd3d', 'Cd3e', 'Cd3g', 'Cd4', 'Foxp3', 'Il2ra', 'Cd8a',
          'Ncr1', 'Klri2',
          'Csf1r', 'Itgam', 'Cd68',
          'Fscn1', 'Cacnb3', 'Ccr7',
          'Xcr1', 'Clec9a',
          'Cxcr2', 'Lcn2', 'Hdc',
          'Ms4a2', 'Cebpa',
          'Col1a1', 'Dcn', 'Ptprc',
          'Lgals7', 'Aqp1'
          ]

cell_types = ['CD4+ T cell', 'CD8+ T cell', 'Treg', 'NK cell', 'Macrophage', 'CCR7+ DC', 'XCR1+ DC', 'Neutrophil',
              'Basophil', 'Fibroblast', 'Tumor cell']

celltypes = np.zeros((len(cell_types), len(marker)))
celltypes[0, :7] = [1, 1, 1, 1, -1, -1, -1]  # CD4+
celltypes[1, :7] = [1, 1, 1, -1, -1, -1, 1]  # CD8+
celltypes[2, :7] = [1, 1, 1, 1, 1, 0, -1]  # Treg
celltypes[3, :3] = [-1, -1, -1]  # NK
celltypes[3, 7:9] = [1, 1]
# lymphoid lineage cells need to be off for myeloid macrophages
celltypes[:4, 9:12] = -1*np.ones((4, 3))
celltypes[4, 9:12] = [1, 1, 1]  # macrophage
celltypes[4, :3] = [-1, -1, -1]
celltypes[5, 12:15] = [1, 1, 1]  # CCR7+ DC
celltypes[6, 15:17] = [1, 1]  # XCR1+ DC
celltypes[7, 17:20] = [1, 1, 1]  # neut
celltypes[8, 20:22] = [1, 1]  # basophil
celltypes[9, 22:25] = [1, 1, -1]  # cancer-associated fibroblast
celltypes[10, 24:] = [-1, 1, 1]  # tumor cell

tot_lab, tot_ideal_ind, tot_traindata, tot_testdata = training_data_select(adata, marker, celltypes, cell_types,
                                                                           np.arange(len(cell_types)))

# FEEDFORWARD NEURAL NETWORK FOR CELL TYPE ANNOTATION, VISUALIZATION
learning_rate = 0.025  # altering learning rate to change how much neural net can adjust during each training epoch
training_epochs = 500
batch_size = 100
display_step = 5

# using aggregate data for training to bolster cell type abundances in training sets
tot_lab_onehot = one_hot_encode(tot_lab)
all_train_ind = np.array([])
ideal_ = np.argmax(tot_lab_onehot, axis=1)
train_split = 0.5
for k in np.unique(ideal_):
    all_ind = np.where(ideal_ == k)[0]  # randomly select half for training, other half goes to validation
    train_ind = np.random.choice(all_ind, round(train_split*len(all_ind)), replace=False)
    all_train_ind = np.concatenate((all_train_ind, train_ind))

total_predicted_lab, tot_prob, colorm, pred = cell_type_classifier(tot_lab_onehot, tot_traindata,
                                                                   tot_testdata,
                                                                   all_train_ind,
                                                                   learning_rate, training_epochs, batch_size,
                                                                   display_step)

# reordering cell type labels and filtering by probability
total_lab, total_prob = process_label(tot_prob, tot_lab, total_predicted_lab, tot_ideal_ind, adata, 0.8)

# write results as metadata
cluster2annotation = {0: 'CD4+ T cell',
                      1: 'CD8+ T cell',
                      2: 'Treg',
                      3: 'NK cell',
                      4: 'Macrophage',
                      5: 'CCR7+ DC',
                      6: 'XCR1+ DC',
                      7: 'Neutrophil',
                      8: 'Basophil',
                      9: 'Fibroblast',
                      10: 'Tumor cell',
                      -1: 'Poorly classified'
                      }
adata.obs['celltype'] = pd.Series(total_lab.astype('int')).map(cluster2annotation).values
adata.write('/Users/katebridges/PycharmProjects/test/20211118_citeseq_annotated.h5ad')

# preparing objects for downstream analysis (by replicate & in pairs for comparison)
adata.obs['hashing'] = np.argmax(adata.obsm['hash'], axis=1)

# ctrl vs. CD40ag
cd40 = adata[adata.obs['sample'].str.contains('BD2') | adata.obs['sample'].str.contains('BD3')]
cd40 = remove_recompute(cd40)
cd40.write('/Users/katebridges/Downloads/20220915_cd40only.h5ad')

# ctrl vs. ICB lo vs. ICB hi
cpi = adata[adata.obs['sample'].str.contains('BD2') | adata.obs['sample'].str.contains('BD5') | adata.obs['sample'].str.contains('BD4')]
cpi = remove_recompute(cpi)
cpi.write('/Users/katebridges/Downloads/cpi-20220809.h5ad')

# ctrl vs. ICB lo vs. ICB lo + CD40ag
combination = adata[adata.obs['sample'].str.contains('BD2') | adata.obs['sample'].str.contains('BD5') | adata.obs['sample'].str.contains('BD6')]
combination = remove_recompute(combination)
combination.write('/Users/katebridges/Downloads/20230809_ctrl-cpi-comb.h5ad')
