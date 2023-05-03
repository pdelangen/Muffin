# %%
import sys
sys.path.append("./")
import numpy as np
from scipy.io import mmread
from settings import settings, paths
import pandas as pd
import scanpy as sc
sc.set_figure_params(dpi=500)
# %%
import lib.countModelerScanpyWrapper as cm
dataset = sc.read_10x_mtx(paths.scRNAseqGenes)
# %%
sc.set_figure_params(dpi=500, color_map = 'viridis')
dataset.var['mt'] = dataset.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(dataset, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(dataset, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)
sc.pl.scatter(dataset, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(dataset, x='total_counts', y='n_genes_by_counts')
dataset = dataset[dataset.obs.pct_counts_mt < 15, :]
dataset = dataset[dataset.obs.n_genes_by_counts < 4000, :]
dataset = dataset[dataset.obs.n_genes_by_counts > 1000, :]
dataset.X = dataset.X.toarray().astype("int32")
# %%
design = np.ones((dataset.X.shape[0], 1))
cm.set_design_matrix(dataset, design)
cm.compute_size_factors(dataset)
kept = cm.trim_low_counts(dataset)
dataset = dataset[:, kept].copy()
cm.fit_mv_trendline(dataset)
cm.computeResiduals(dataset, clip=np.sqrt(9+len(dataset)/4))
hv = cm.feature_selection_elbow(dataset)
cm.compute_PA_PCA(dataset,feature_mask=hv, max_rank=100, plot=True)
# %%
cm.compute_UMAP(dataset)
cm.cluster_rows_leiden(dataset)
# %%
sc.pl.umap(dataset, color='leiden', legend_loc='on data',
            legend_fontsize=5, legend_fontoutline=0.1, s=10.0,
            title='10k PBMCs gene-centric scRNA-seq', palette='tab20')
dataset.obs["log_sf"] = np.log(dataset.obs["size_factors"])
sc.pl.umap(dataset, color='log_sf', legend_loc='on data',
            legend_fontsize=5, legend_fontoutline=0.1, s=10.0,
            title='10k PBMCs gene-centric scRNA-seq', palette='tab20')
# %%

from sklearn.preprocessing import StandardScaler
dataset.layers["scaled"] = StandardScaler().fit_transform(dataset.layers["residuals"])
sc.tl.rank_genes_groups(dataset, 'leiden', use_raw=False, layer="scaled",
                        method='logreg', class_weight="balanced")
sc.pl.rank_genes_groups(dataset, sharey=False)
# %%
sc.pl.rank_genes_groups_heatmap(dataset, layer="residuals", 
                                use_raw=False, vmin=-3, vmax=3, 
                                cmap='viridis')

# %%
# Load Pol 2 atlas scRNA-seq dataset and match cell barcodes with gene-centric analysis
matrix = mmread(paths.scRNAseqPol2 + "matrix.mtx.gz").astype("int32").toarray().T
cells = pd.read_csv(paths.scRNAseqPol2 + "barcodes.tsv.gz", sep="\t", header=None)
features = pd.read_csv(paths.scRNAseqPol2 + "features.tsv.gz", sep="\t", header=None)[0]
matchingCells = dataset.obs_names[np.isin(dataset.obs_names, cells.values.ravel())]
matchedCells = pd.Series(np.arange(len(cells)), index=cells.values.ravel()).loc[matchingCells].values
cells = cells.loc[matchedCells]
matrix = matrix[matchedCells]
# %%
dataset_pol2 = cm.dataset_from_arrays(matrix, 
                                               cells[0].values, 
                                               features)
matchingCellLabel = pd.Series(np.arange(len(dataset.obs_names)), 
                              index=dataset.obs_names)
matchingIndex = matchingCellLabel.loc[dataset.obs_names].values
# %%
design = np.ones((dataset_pol2.X.shape[0], 1))
cm.set_design_matrix(dataset_pol2, design)
cm.set_size_factors(dataset_pol2, dataset.obs["size_factors"][matchingIndex])
kept = cm.trim_low_counts(dataset_pol2)
dataset_pol2 = dataset_pol2[:, kept].copy()
cm.fit_mv_trendline(dataset_pol2)
cm.computeResiduals(dataset_pol2, clip=np.sqrt(9+len(dataset_pol2)/4))
hv = cm.feature_selection_elbow(dataset_pol2)
cm.compute_PA_PCA(dataset_pol2,feature_mask=hv, max_rank=100, plot=True)
# %%
cm.compute_UMAP(dataset_pol2)
# %%
dataset_pol2.obs["leiden"] = dataset.obs["leiden"][matchingIndex]
sc.pl.umap(dataset_pol2, color='leiden', legend_loc='on data',
            legend_fontsize=5, legend_fontoutline=0.1, s=10.0,
            title='10k PBMCs gene-centric scRNA-seq', palette='tab20')
sc.pl.pca(dataset_pol2, color='leiden', legend_loc='on data',
            legend_fontsize=5, legend_fontoutline=0.1, s=10.0,
            title='10k PBMCs gene-centric scRNA-seq', palette='tab20')
# %%
dataset_pol2.layers["scaled"] = StandardScaler().fit_transform(dataset_pol2.layers["residuals"])
sc.tl.rank_genes_groups(dataset_pol2, 'leiden', use_raw=False, layer="scaled",
                        method='logreg', class_weight="balanced")
sc.pl.rank_genes_groups(dataset_pol2, sharey=False)
# %%
sc.pl.rank_genes_groups_heatmap(dataset_pol2, layer="residuals", 
                                use_raw=False, vmin=-3, vmax=3, 
                                cmap='viridis')

# %%
