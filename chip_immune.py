# %%
import sys
sys.path.append("./")
import pandas as pd
import numpy as np
from settings import settings, paths
import os
import muffin as sp
metadata_chip = pd.read_csv(paths.immuneChipPath + "chip_meta.tsv", sep="\t", index_col="File accession")
metadata_input = pd.read_csv(paths.immuneChipPath + "input_meta.tsv", sep="\t", index_col="File accession")
metadata_peaks = pd.read_csv(paths.immuneChipPath + "peak_meta.tsv", sep="\t", index_col="File accession")
bam_files = [paths.immuneChipPath + "chip/" + f + ".bam" for f in metadata_chip.index]
input_files = [paths.immuneChipPath + "input/" + f + ".bam" for f in metadata_input.index]
peak_files = [paths.immuneChipPath + "peak/" + f + ".bigBed" for f in metadata_peaks.index]
# %%
import pyBigWig
def read_bigBed_encode(path):
    bb = pyBigWig.open(path)
    chroms = bb.chroms()
    entries_list = []
    for chrom, length in chroms.items():
        entries = bb.entries(chrom, 0, length)
        for entry in entries:
            entries_list.append({
                'chrom': chrom,
                'start': int(entry[0]),
                'end': int(entry[1]),
                'name': entry[2],
            })
    df = pd.DataFrame(entries_list)
    df[["name", "Score", "Strand", "FC", "Pval", "FDR", "Summit"]] = df["name"].str.split("\t", expand=True)
    df[["FC", "Pval", "FDR"]] = df[["FC", "Pval", "FDR"]].astype("float")
    df["Summit"] = df["Summit"].astype(int)
    return df
beds = [read_bigBed_encode(f) for f in peak_files]
chromSizes = pd.read_csv(paths.chromsizes, sep="\t", header=None, index_col=0).iloc[:,0].to_dict()
# %%
consensus_peaks = sp.peakMerge.mergePeaks(beds, chromSizes)
consensus_peaks.to_csv("tempaaaa/consensuses_h3k27me3.bed", sep="\t", index=None, header=None)
# %%
featureCountParams = {"nthreads":54, "allowMultiOverlap":True, "minOverlap":1,
                      "countMultiMappingReads":False}
dataset = sp.load.dataset_from_bam(bam_files, genomic_regions_path="tempaaaa/consensuses_h3k27me3.bed",
                                 input_bam_paths=input_files, isBEDannotation=True, 
                                 featureCounts_params=featureCountParams,chromsizes=chromSizes)
# %%
import pyranges as pr
gsea_obj = sp.grea.pyGREAT(paths.GOfile, paths.gencode, paths.chromsizes)
dataset.var_names = gsea_obj.label_by_nearest_gene(dataset.var[["Chromosome","Start","End"]]).astype(str)
# %%
design = np.ones((dataset.X.shape[0],1))
sp.load.set_design_matrix(dataset, design)
sp.tools.rescale_input_center_scale(dataset, plot=True)
detectable = sp.tools.trim_low_counts(dataset)
dataset = dataset[:, detectable]
dataset.uns["pouet"]=1
sp.tools.computeResiduals(dataset)
peaks = sp.tools.pseudo_peak_calling(dataset)
hv = sp.tools.feature_selection_elbow(dataset)
sp.tools.compute_PA_PCA(dataset, feature_mask=peaks&hv, plot=True)
# %%
sp.tools.compute_UMAP(dataset)
# %%
dataset.obs["Cell type detailed"] = metadata_chip["Biosample term name"].values
dataset.obs["Cell type"] = metadata_chip["Biosample cell type"].values
# %%
import scanpy as sc
sc.set_figure_params(dpi=500)
sc.pl.umap(dataset, color='Cell type', legend_loc='on data',
                legend_fontsize=5, legend_fontoutline=0.1, s=10.0,
                palette='Paired')
# %% 
sp.plots.mega_heatmap(dataset[:, peaks & hv], layer="residuals", label_col="Cell type", vmin=-3, vmax=3)
# %% 
dataset.layers["lfc_norm"] = np.log2(np.maximum(dataset.X, 0.5)/dataset.layers["normalization_factors"])
sp.plots.mega_heatmap(dataset[:, hv & peaks], layer="lfc_norm", label_col="Cell type", vmin=0.0, vmax=5.0)
# %% 
dataset.layers["lfc_library_size"] = np.log2(np.maximum(dataset.X, 0.5)*dataset.obs["size_factors_input"][:, None]/dataset.obs["size_factors"][:, None]/np.maximum(dataset.layers["input"], 0.5))
sp.plots.mega_heatmap(dataset[:, hv & peaks], layer="lfc_library_size", label_col="Cell type", vmin=0.0, vmax=5.0)
# %%
from sklearn.preprocessing import StandardScaler
dataset.layers["scaled"] = StandardScaler().fit_transform(dataset.layers["residuals"])
noNK = dataset[dataset.obs["Cell type"]!="NK cell"]
sc.tl.rank_genes_groups(noNK, 'Cell type', use_raw=False, layer="scaled",
                        method='logreg', class_weight="balanced")
# %%
import scanpy as sc
sc.set_figure_params(dpi=500)
sc.pl.rank_genes_groups_heatmap(noNK, n_genes=10, layer="lfc_norm",
                                use_raw=False, cmap='viridis')

# %%
