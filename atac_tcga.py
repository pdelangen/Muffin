# %%
import sys
sys.path.append("./")
from settings import settings, paths
import pandas as pd
import muffin as sp
import numpy as np
dataset_tsv = pd.read_csv("/shared/projects/pol2_chipseq/data_newPkg/tcga_atac/atac_table.txt", sep="\t")
sequencing_metadata = pd.read_csv("/shared/projects/pol2_chipseq/data_newPkg/tcga_atac/sequencing_stats.csv", 
                                  sep="\t", index_col=0)
dataset = sp.load.dataset_from_arrays(dataset_tsv.iloc[:, 7:].values.T, row_names=dataset_tsv.columns[7:],
                                 col_names=dataset_tsv["name"])
dataset.var = dataset_tsv.iloc[:, :7]
dataset.obs["label"] = [s[:4] for s in dataset.obs_names]
dataset.obs["FRIP"] = sequencing_metadata["FRIP"].values
dataset.obs["subtype"] = sequencing_metadata["BRCA_pam50"].values
dataset.obs["subtype"][dataset.obs["subtype"]=="Normal"] = np.nan
dataset.var.rename(columns={"seqnames":"Chromosome","start":"Start","end":"End"}, inplace=True)
# %%
# %%
import pyranges as pr
pygreat = sp.grea.pyGREAT(paths.GOfile, paths.gencode, paths.chromsizes)
dataset.var_names = pygreat.label_by_nearest_gene(dataset.var[["Chromosome","Start","End"]])
# %%
design = np.ones((dataset.X.shape[0], 1))
# design = np.concatenate([design, dataset.obs["FRIP"].values.reshape(-1,1)], axis=1)
sp.load.set_design_matrix(dataset, design)
sp.tools.compute_size_factors(dataset, method="deseq")
kept = sp.tools.trim_low_counts(dataset)
dataset = dataset[:, kept].copy()
sp.tools.computeResiduals(dataset)
hv = sp.tools.feature_selection_elbow(dataset)
sp.tools.compute_PA_PCA(dataset, feature_mask=hv, max_rank=100, plot=True)
# %%
sp.tools.compute_UMAP(dataset, umap_params={"n_neighbors":30, "min_dist":0.5})
# %%
sp.plots.plot_reduced_dim(dataset, which="X_umap", points_labels=dataset.obs["label"], 
                            label_type="categorical")
sp.plots.plot_reduced_dim(dataset, which="X_pca", points_labels=dataset.obs["label"], 
                            label_type="categorical")
sp.plots.plot_reduced_dim(dataset, which="X_pca", points_labels=dataset.obs["FRIP"], 
                            label_type="numeric")
# %%
import scanpy as sc
from matplotlib.pyplot import rc_context
sc.set_figure_params(dpi=500, color_map = 'viridis')
with rc_context({'figure.figsize': (5, 5)}):
    sc.pl.umap(dataset, color='label', legend_loc='on data',
                legend_fontsize=5, legend_fontoutline=0.1, s=10.0,
                title='TCGA-ATAC UMAP', palette='tab20')
    sc.pl.umap(dataset, color='FRIP', legend_loc='on data',
                legend_fontsize=5, legend_fontoutline=0.1, s=10.0,
                title='TCGA-ATAC UMAP', palette='tab20')

# %% 
sp.plots.mega_heatmap(dataset[:, hv], layer="residuals", label_col="label", vmin=-3, vmax=3)
# %%
import scanpy as sc
from matplotlib.pyplot import rc_context
sc.set_figure_params(dpi=500)
with rc_context({'figure.figsize': (5, 5)}):
    sc.pl.umap(dataset, color='subtype', legend_loc='on data',
                legend_fontsize=5, legend_fontoutline=0.1, s=10.0,
                title='TCGA-ATAC UMAP', palette='tab20')
# %%
from muffin.utils.diff_expr import DESeq2, t_test
lumA = dataset.obs["subtype"] == "LumB"
lumB = dataset.obs["subtype"] == "LumA"
subset = dataset[~dataset.obs["subtype"].isna()]
# %%
from sklearn.preprocessing import StandardScaler
dataset.layers["scaled"] = StandardScaler().fit_transform(dataset.layers["residuals"])
sc.tl.rank_genes_groups(subset, 'subtype', use_raw=False, layer="scaled",
                        method='logreg', class_weight="balanced")
# %%
import matplotlib.pyplot as plt
with plt.rc_context({'figure.figsize': (5, 5)} ):
    sc.pl.rank_genes_groups_heatmap(subset, layer="residuals", use_raw=False)
