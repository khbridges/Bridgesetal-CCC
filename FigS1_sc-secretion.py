import numpy as np
import pandas as pd

# reading in data from file
file_dir = '/Users/katebridges/Downloads/AA_YR_sc-sec/'
replicates = ['2019.08.20/', '2020.01.31/', '2020.03.20/']

yr_ctrl = pd.DataFrame({})
yr_lps = pd.DataFrame({})

ctrl_counts = np.array([])
lps_counts = np.array([])

for g in replicates:
    ctrl_rep = pd.read_csv(file_dir + g + 'Chip 2 YUMMER No Stim 16hr asinhTransformed.csv')
    yr_ctrl = pd.concat([yr_ctrl, ctrl_rep])
    ctrl_counts = np.concatenate((ctrl_counts, [g] * ctrl_rep.shape[0]))

    lps_rep = pd.read_csv(file_dir + g + 'Chip 4 YUMMER LPS 100ng 16hr asinhTransformed.csv')
    yr_lps = pd.concat([yr_lps, lps_rep])
    lps_counts = np.concatenate((lps_counts, [g] * lps_rep.shape[0]))

yr_ctrl['replicate'] = ctrl_counts
yr_ctrl['condition'] = ['ctrl'] * yr_ctrl.shape[0]
yr_lps['replicate'] = lps_counts
yr_lps['condition'] = ['lps'] * yr_lps.shape[0]

all_dat = pd.concat([yr_ctrl, yr_lps])

# limit to c/c panel of interest
cc = ['TNF', 'CCL5', 'CCL3', 'IL_6', 'IL_10', 'Chi3L3', 'CCL2']

# calculating fano factor
ctrl_var = pd.DataFrame({})
lps_var = pd.DataFrame({})

for h in replicates:
    ctrl_var[h] = yr_ctrl[yr_ctrl['replicate'].str.contains(h)].var()/yr_ctrl[yr_ctrl['replicate'].str.contains(h)].mean()
    lps_var[h] = yr_lps[yr_lps['replicate'].str.contains(h)].var()/yr_lps[yr_lps['replicate'].str.contains(h)].mean()

# write results to excel for visualization (GraphPad Prism)
ctrl_var.to_excel(file_dir + 'fano-factor_ctrl.xlsx')
lps_var.to_excel(file_dir + 'fano-factor_lps.xlsx')

