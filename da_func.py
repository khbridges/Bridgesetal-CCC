import numpy as np
import scanpy as sc
import milopy  # had to revert markupsafe (2.0.1) for updated version issue
import milopy.core as milo
import itertools
from distinctipy import distinctipy
import networkx as nx
from community import community_louvain  # from python-louvain, do not 'pip install community'


def encode_replicates(adata, rep_dict):
    rep = [None] * adata.shape[0]
    for z in np.arange(adata.shape[0]):
        if np.sum(adata.obsm['hash'][z]) > 0:
            z_arg = np.argmax(adata.obsm['hash'][z])
            if adata.obs['sample'][z] == 'BD1':
                rep[z] = rep_dict[z_arg]
            else:
                rep[z] = 'R{}'.format(z_arg+1)
        else:
            rep[z] = 'R0'
    return rep


def build_samplerep(adata, sample_slot, replicate_slot):
    sample_rep = [None] * adata.shape[0]
    for r in np.arange(adata.shape[0]):
        sample_rep[r] = str(adata.obs[sample_slot][r]) + ' ' + str(adata.obs[replicate_slot][r])
    return sample_rep


def run_milo(adata):
    sc.pp.neighbors(adata)
    milo.make_nhoods(adata)
    # Count cells from each sample (just cond, no rep) in each nhood
    adata.obs['rep_code'] = adata.obs['Condition'].cat.codes
    milo.count_nhoods(adata, sample_col="rep_code")
    # Test for differential abundance between conditions
    # need to convert to "continuous" encoding to describe time
    milo.DA_nhoods(adata, design="~ cond_continuous")
    # build graph for viz
    milopy.utils.build_nhood_graph(adata)

    # annotation of nhoods by joint celltypes of interacting pairs
    milopy.utils.annotate_nhoods(adata, anno_col='celltype.Joint')
    adata.uns['nhood_adata'].obs.loc[adata.uns['nhood_adata'].obs["nhood_annotation_frac"] < 0.5, "nhood_annotation"] = "Mixed"

    return adata


def group_nhoods(adata, min_connect, max_difflfc):
    adj_nhood = np.zeros((adata.obsm['nhoods'].shape[1], adata.obsm['nhoods'].shape[1]))

    # only considering single cells belonging to more than one
    overlap_ind = np.where(np.sum(adata.obsm['nhoods'], axis=1) > 1)[0]
    for g in overlap_ind:
        nhood_ind = np.where(adata.obsm['nhoods'][g, :].todense() == 1)[1]
        ij = list(itertools.permutations(nhood_ind, 2))
        for q in ij:
            adj_nhood[q] = adj_nhood[q] + 1

    # still need to filter adj matrix entries to zero by connections (< 3) and LFC match (diff > 0.25?)
    nonzero_ind = np.where(adj_nhood > 0)
    logFC = adata.uns['nhood_adata'].obs['logFC']
    for f in np.arange(len(nonzero_ind[0])):
        if adj_nhood[nonzero_ind[0][f], nonzero_ind[1][f]] < min_connect or abs(logFC[nonzero_ind[0][f]] - logFC[nonzero_ind[1][f]]) > max_difflfc:
            adj_nhood[nonzero_ind[0][f], nonzero_ind[1][f]] = 0

    return adj_nhood


def cluster_nhoods(adata, min_connect, max_difflfc):
    test_adj = group_nhoods(adata, min_connect, max_difflfc)
    G_test = nx.from_numpy_matrix(test_adj)
    partition2 = community_louvain.best_partition(G_test)
    print(np.max(list(partition2.values())))
    return partition2


def plot_nhood_clusters(adata, cluster_labels, title, alpha=0.1, min_size=10, plot_edges=False):
    nhood_adata = adata.uns["nhood_adata"].copy()

    nhood_adata.obs["graph_color"] = cluster_labels
    nhood_adata.obs["graph_color"] = nhood_adata.obs["graph_color"].astype('category')

    clust_col = distinctipy.get_colors(len(np.unique(nhood_adata.obs["graph_color"])))
    clust_pal = {np.unique(nhood_adata.obs["graph_color"])[i]: clust_col[i] for i in range(len(clust_col))}

    nhood_adata.obs.loc[nhood_adata.obs["SpatialFDR"] > alpha, "graph_color"] = np.nan

    # plotting order
    ordered = nhood_adata.obs.sort_values('SpatialFDR', na_position='last').index[::-1]
    nhood_adata = nhood_adata[ordered]

    sc.pl.embedding(nhood_adata, "X_milo_graph",
                    color="graph_color", palette=clust_pal,
                    size=adata.uns["nhood_adata"].obs["Nhood_size"] * min_size,
                    edges=plot_edges, neighbors_key="nhood",
                    frameon=False,
                    title=title
                    )

    return nhood_adata.obs["graph_color"], clust_pal


def plot_durable_clusters(adata, cluster_labels, title, alpha=0.1, beta=0.5, min_size=10, plot_edges=False):
    nhood_adata = adata.uns["nhood_adata"].copy()

    nhood_adata.obs["graph_color"] = cluster_labels
    nhood_adata.obs["graph_color"] = nhood_adata.obs["graph_color"].astype('category')

    clust_col = distinctipy.get_colors(len(np.unique(nhood_adata.obs["graph_color"])))
    clust_pal = {np.unique(nhood_adata.obs["graph_color"])[i]: clust_col[i] for i in range(len(clust_col))}

    nhood_adata.obs.loc[nhood_adata.obs["SpatialFDR"] < alpha, "graph_color"] = np.nan
    nhood_adata.obs.loc[nhood_adata.obs["logFC"] > beta, "graph_color"] = np.nan
    nhood_adata.obs.loc[nhood_adata.obs["logFC"] < -beta, "graph_color"] = np.nan

    sc.pl.embedding(nhood_adata, "X_milo_graph",
                    color="graph_color", palette=clust_pal,
                    size=adata.uns["nhood_adata"].obs["Nhood_size"] * min_size,
                    edges=plot_edges, neighbors_key="nhood",
                    frameon=False,
                    title=title
                    )

    return nhood_adata.obs["graph_color"], clust_pal


def get_sc_louvain(adata, cluster_slot='louvain'):
    louvain_onehot = np.zeros((adata.uns['nhood_adata'].obs[cluster_slot].shape[0], (np.unique(adata.uns['nhood_adata'].obs[cluster_slot])[-2] + 1).astype('int')))
    for c in adata.uns['nhood_adata'].obs[cluster_slot].index:
        if adata.uns['nhood_adata'].obs[cluster_slot][c] < float('inf'):
            louvain_onehot[int(c), adata.uns['nhood_adata'].obs[cluster_slot][c].astype('int')] = 1

    # get single-cell louvain neighborhood cluster labels
    sc_onehot = adata.obsm['nhoods']*louvain_onehot
    sc_louvain = np.zeros(sc_onehot.shape[0])
    for t in np.arange(sc_onehot.shape[0]):
        if np.sum(sc_onehot[t, :]) == 0:
            sc_louvain[t] = -1
        else:
            sc_louvain[t] = np.argmax(sc_onehot[t, :])

    return sc_louvain.astype('int')


def highlight_ind(clust, adata):
    highlight_ind_ = []
    for g in clust:
        i = np.where(adata.obs['louvain_str'] == g)[0]
        highlight_ind_.append(i)
    adata_highlight = adata[np.array(list(itertools.chain(*highlight_ind_)))]
    return adata_highlight


def highlight_NICHEScluster(niches_adata, adata, cluster_no):
    codes_1 = niches_adata[niches_adata.obs['sc_louvain'] == cluster_no].obs.index

    v = np.zeros(adata.shape[0])
    w = np.zeros(adata.shape[0])

    for n in codes_1:
        ind = np.where(adata.obs.index.str.contains("-".join(n.split("-", 2)[:1])))
        if len(ind[0]) > 0:
            v[ind[0]] = 1

    for n in codes_1:
        ind = np.where(adata.obs.index.str.contains("-".join(n.split('—')[1].split("-", 2)[:1])))
        if len(ind[0]) > 0:
            w[ind[0]] = 1

    adata.obs['cluster{}_sending'.format(cluster_no)] = v
    adata.obs['cluster{}_receiving'.format(cluster_no)] = w
    highlight_map = {0: 'Other', 1: 'Highlight'}
    adata.obs['cluster{}_sending'.format(cluster_no)] = adata.obs['cluster{}_sending'.format(cluster_no)].map(highlight_map)
    adata.obs['cluster{}_receiving'.format(cluster_no)] = adata.obs['cluster{}_receiving'.format(cluster_no)].map(highlight_map)

    return adata


def nhood_expression_mapping(adata, gene_oi):
    avg_expr = [np.mean(adata[adata.obs['sc_louvain'] == b][:, gene_oi].X).tolist() for b in np.unique(adata.obs['sc_louvain'])]
    expr_nhood_map = {np.unique(adata.obs['sc_louvain'])[c]: avg_expr[c] for c in np.arange(len(np.unique(adata.obs['sc_louvain'])))}

    return expr_nhood_map

