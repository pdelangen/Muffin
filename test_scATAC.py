# %%
import sys
sys.path.append("./")
import pandas as pd
import numpy as np
from settings import settings, paths
import collections
import scipy.sparse as sp_sparse
import tables
# %%
CountMatrix = collections.namedtuple('CountMatrix', ['feature_ref', 'barcodes', 'matrix'])
def get_matrix_from_h5(filename):
    with tables.open_file(filename, 'r') as f:
        mat_group = f.get_node(f.root, 'matrix')
        barcodes = f.get_node(mat_group, 'barcodes').read()
        data = getattr(mat_group, 'data').read()
        indices = getattr(mat_group, 'indices').read()
        indptr = getattr(mat_group, 'indptr').read()
        shape = getattr(mat_group, 'shape').read()
        matrix = sp_sparse.csc_array((data, indices, indptr), shape=shape)
        feature_ref = {}
        feature_group = f.get_node(mat_group, 'features')
        feature_ids = getattr(feature_group, 'id').read()
        feature_names = getattr(feature_group, 'name').read()
        feature_types = getattr(feature_group, 'feature_type').read()
        feature_ref['id'] = feature_ids
        feature_ref['name'] = feature_names
        feature_ref['feature_type'] = feature_types
        tag_keys = getattr(feature_group, '_all_tag_keys').read()
        for key in tag_keys:
            key = key.decode("utf-8")
            feature_ref[key] = getattr(feature_group, key).read()
         
        return CountMatrix(feature_ref, barcodes, matrix)
mat = get_matrix_from_h5(paths.scAtacHD5)


# %%
from lib.utils.pyGREATglm import pyGREAT
import pyranges as pr
gsea_obj = pyGREAT(paths.GOfile, paths.gencode, paths.chromsizes)
# %%
positions = pd.DataFrame([x.split(':')[0:1] + x.split(':')[1].split('-') for x in mat.feature_ref["name"].astype(str)],
                         columns=['Chromosome', 'Start', 'End'])
names = gsea_obj.labelByNearest(pr.PyRanges(positions))
# %%
import lib.countModelerScanpyWrapper as cm
dataset = cm.dataset_from_arrays(mat.matrix.astype("int32").toarray().T, 
                                               mat.barcodes.astype(str), 
                                               names)
for k in mat.feature_ref.keys():
    if k == "id":
        continue
    dataset.var[k] = mat.feature_ref[k].astype(str)
dataset.var[positions.columns] = positions.values
# %%
mappingTab = pd.read_csv(paths.scAtacMapqual, index_col="barcode").loc[mat.barcodes.astype(str)]
FRiP = np.sum(dataset.X, axis=1) / mappingTab["total"].values
# %%
design = np.ones((dataset.X.shape[0], 1))
design = np.concatenate([design, FRiP.reshape(-1,1)], axis=1)
cm.set_design_matrix(dataset, design)
cm.compute_size_factors(dataset)
kept = cm.trim_low_counts(dataset)
dataset = dataset[:, kept].copy()
cm.fit_mv_trendline(dataset)
cm.computeResiduals(dataset)
hv = cm.feature_selection_elbow(dataset)
cm.compute_PA_PCA(dataset,feature_mask=hv, max_rank=100, plot=True)
# %%
cm.compute_UMAP(dataset)
cm.cluster_rows_leiden(dataset)
# %%
cm.plot_reduced_dim(dataset, which="X_umap", points_labels=dataset.obs["leiden"], 
                            label_type="categorical")
cm.plot_reduced_dim(dataset, which="X_pca", points_labels=np.log(dataset.obs["size_factors"]), 
                            label_type="numeric")
# %%
import scanpy as sc
sc.set_figure_params(dpi=500)
from sklearn.preprocessing import StandardScaler
dataset.layers["scaled"] = StandardScaler().fit_transform(dataset.layers["residuals"])
sc.tl.rank_genes_groups(dataset, 'leiden', use_raw=False, layer="scaled",
                        method='logreg', class_weight="balanced")
# %%
sc.pl.rank_genes_groups_heatmap(dataset, layer="residuals", use_raw=False, 
                                vmin=-3, vmax=3, cmap='viridis',
                                n_genes=2)

# %%
