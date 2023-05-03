# %%
import sys
sys.path.append("./")
import pandas as pd
import numpy as np
from settings import settings, paths
import os
# %%
fOrder = os.listdir(paths.chipBAM)
bamFiles = [paths.chipBAM + p for p in fOrder if p.endswith(".bam")]
fOrder = [f[:-4] for f in fOrder if f.endswith(".bam")]
metadataAll = pd.read_csv(paths.chipmetadata, sep="\t", index_col="File accession")
metadata = metadataAll.loc[fOrder]
# %%
exp = metadata["Experiment accession"]
metadata_exp = pd.read_csv(paths.chipmetadata_exp, index_col="Accession", sep="\t", skiprows=1)
metadata_chip = metadata_exp.loc[exp]
#%%
biosample = metadata_chip["Biosample accession"]
metadata_exp["Accession"] = metadata_exp.index
metadata_exp.index = metadata_exp["Biosample accession"]
metadata_input = metadata_exp[metadata_exp["Assay title"]=="Control ChIP-seq"].loc[biosample]
# %%
metadata_cpy = metadataAll.copy()
metadata_cpy["File acc"] = metadata_cpy.index
metadata_cpy.set_index("Experiment accession", inplace=True)
metadata_cpy = metadata_cpy[metadata_cpy["Output type"]=="alignments"]
metadata_cpy = metadata_cpy[metadata_cpy["Assay"]=="Control ChIP-seq"].loc[metadata_input["Accession"]]
# %%
inputBAM = [paths.inputBAM+f+".bam" for f in metadata_cpy["File acc"]]
# %%
age = np.array([int(s.split(" ")[0]) for s in metadata_chip["Biosample age"]]).astype("float")
age = (age - age.mean())/age.std()
pipelineVer = metadata["File analysis title"]
sex = ["female" in l for l in metadata_chip["Biosample summary"]]
label = metadata_chip["Biosample term name"]
bamFiles = np.array(bamFiles)
inputBAM = np.array(inputBAM)
# %%
chromSizes = pd.read_csv(paths.chromsizes, sep="\t", header=None, index_col=0).iloc[:,0].to_dict()
# %%
import superPackage as sp
featureCountParams = {"nthreads":28, "allowMultiOverlap":True, "minOverlap":1,
                      "countMultiMappingReads":False}
dataset = sp.load.dataset_from_bam(bamFiles, genomic_regions_path=paths.ctcf_remap_nr,
                                 input_bam_paths=inputBAM, isBEDannotation=True, 
                                 featureCounts_params=featureCountParams,chromsizes=chromSizes)
# %%
import pyranges as pr
gsea_obj = sp.grea.pyGREAT(paths.GOfile, paths.gencode, paths.chromsizes)
dataset.var_names = gsea_obj.labelByNearest(dataset.var[["Chromosome","Start","End"]]).astype(str)
# %%
design = np.ones((dataset.X.shape[0],1))
s = pd.get_dummies(sex, drop_first=True).values
sp.load.set_design_matrix(dataset, design)
sp.tools.rescale_input(dataset, plot=True)
detectable = sp.tools.trim_low_counts(dataset)
dataset = dataset[:, detectable]
dataset.obs["sex"] = np.array(["H", "F"])[np.array(sex).astype(int)]
dataset.obs["label"] = label.values
sp.tools.computeResiduals(dataset)
peaks = sp.tools.pseudo_peak_calling(dataset)
hv = sp.tools.feature_selection_elbow(dataset)
sp.tools.compute_PA_PCA(dataset, feature_mask=hv&peaks, plot=True)
# %%
sp.tools.compute_UMAP(dataset)
# %%
import scanpy as sc
sc.set_figure_params(dpi=500)
sc.pl.umap(dataset, color='label', legend_loc='on data',
                legend_fontsize=5, legend_fontoutline=0.1, s=10.0,
                palette='Set1')
sc.pl.pca(dataset, color='est_signal_to_noise', legend_loc='on data',
                legend_fontsize=5, legend_fontoutline=0.1, s=10.0,
                palette='Set1')
sc.pl.pca(dataset, color='sex', legend_loc='on data',
                legend_fontsize=5, legend_fontoutline=0.1, s=10.0,
                palette='Set1')
# %%
peakData = dataset[:, peaks]
peakData.uns["test"]=1
sp.tools.differential_expression_A_vs_B(peakData, "sex", "H", method="deseq")
# %%
import scanpy as sc
sc.set_figure_params(dpi=500)
sc.pl.rank_genes_groups_heatmap(peakData, n_genes=20, layer="residuals",
                                use_raw=False, cmap='viridis')

# %%
all_reg = pr.PyRanges(dataset.var[["Chromosome","Start","End","Strand"]])
DE_reg = pr.PyRanges(peakData.var[["Chromosome","Start","End","Strand"]][peakData.varm["DE_results"]["padj"]<0.05])
results = gsea_obj.findEnriched(query=DE_reg, background=all_reg)
# %%
gsea_obj.clusterTreemap(results)
# %%
gsea_obj.findGenesForGeneSet()
# %%
