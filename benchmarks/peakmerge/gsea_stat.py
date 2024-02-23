# %%
import scanpy as sc
import numpy as np
import pandas as pd
import muffin
import sys
sys.path.append("./")
sys.path.append("../..")
from settings import settings, paths

dataset = sc.read_h5ad("h3k4me3_results/dataset.h5ad")
# %%
selected = dataset[(dataset.obs["Cell type"] == "B cell") | (dataset.obs["Cell type"] == "T cell")]
muffin.tools.differential_expression_A_vs_B(selected, category="Cell type", 
                                        ref_category="T cell")
# %%
gsea_obj = muffin.great.pyGREAT(paths.gencode, paths.chromsizes, paths.GOfile)
# %%
DE_indexes = (selected.varm["DE_results"]["padj"] < 0.05) & (np.abs(selected.varm["DE_results"]["log2FoldChange"]) > 1.0)
all_regions = selected.var[["Chromosome", "Start", "End"]]
results = gsea_obj.find_enriched(all_regions[DE_indexes], all_regions, cores=16)
# %%
np.random.seed(42)
query_random = all_regions[np.random.permutation(DE_indexes.values)]
# %%
results_random_reg = gsea_obj.find_enriched(query_random, all_regions, cores=16)
# %%
from copy import deepcopy
gsea_obj_random = deepcopy(gsea_obj)
goterm_matrix = np.apply_along_axis(np.random.permutation, 1, gsea_obj_random.mat.values)
# %%
from scipy.sparse import csr_array
sp_go = csr_array(goterm_matrix)
sparse_df = pd.DataFrame.sparse.from_spmatrix(sp_go, 
                                  index=gsea_obj_random.mat.index, 
                                  columns=gsea_obj_random.mat.columns,)
# %%
gsea_obj_random.mat = sparse_df
results_go_permuted = gsea_obj_random.find_enriched(all_regions[DE_indexes], all_regions, cores=16)
# %%
results_poisson_go_permuted = gsea_obj_random.__poisson_glm__(all_regions[DE_indexes], 
                                                            all_regions, cores=16,
                                                            yes_really=True)
# %%
results_poisson_random_reg = gsea_obj.__poisson_glm__(query_random, 
                                                        all_regions, cores=16,
                                                        yes_really=True)

# %%
results_poisson = gsea_obj.__poisson_glm__(all_regions[DE_indexes], 
                                            all_regions, cores=16,
                                            yes_really=True)
# %%
num_pos_poisson_true = (results_poisson["BH corrected p-value"] < 0.05).sum()
num_pos_NB_true = (results["BH corrected p-value"] < 0.05).sum()
num_pos_poisson_random_reg = (results_poisson_random_reg["BH corrected p-value"] < 0.05).sum()
num_pos_NB_random_reg = (results_random_reg["BH corrected p-value"] < 0.05).sum()
num_pos_poisson_go_permuted = (results_poisson_go_permuted["BH corrected p-value"] < 0.05).sum()
num_pos_NB_go_permuted = (results_go_permuted["BH corrected p-value"] < 0.05).sum()

# %%
import pyranges as pr
import matplotlib.pyplot as plt
from scipy.stats import hypergeom

all_genes_query_random = pr.PyRanges(query_random).join(pr.PyRanges(gsea_obj.geneRegulatory)).as_df()["gene_name"].unique()
all_genes_query_true = pr.PyRanges(all_regions[DE_indexes]).join(pr.PyRanges(gsea_obj.geneRegulatory)).as_df()["gene_name"].unique()
all_genes = pr.PyRanges(all_regions).join(pr.PyRanges(gsea_obj.geneRegulatory)).as_df()["gene_name"].unique()

genes_per_go_tot = gsea_obj.mat.loc[:, all_genes].values.sum(axis=1)
genes_per_go_obs = gsea_obj.mat.loc[:, all_genes_query_true].values.sum(axis=1)
genes_per_go_random = gsea_obj.mat.loc[:, all_genes_query_random].values.sum(axis=1)

genes_per_go_tot_permuted = gsea_obj_random.mat.loc[:, all_genes].values.sum(axis=1)
genes_per_go_obs_permuted = gsea_obj_random.mat.loc[:, all_genes_query_true].values.sum(axis=1)
genes_per_go_random_permuted = gsea_obj_random.mat.loc[:, all_genes_query_random].values.sum(axis=1)



pvals_true = hypergeom(len(all_genes), genes_per_go_tot, len(all_genes_query_true)).sf(genes_per_go_obs-1)
pvals_random_reg = hypergeom(len(all_genes), genes_per_go_tot, len(all_genes_query_random)).sf(genes_per_go_random-1)
pvals_random_go = hypergeom(len(all_genes), genes_per_go_tot_permuted, len(all_genes_query_true)).sf(genes_per_go_obs_permuted-1)

# %%
from statsmodels.stats.multitest import fdrcorrection
num_pos_hyper_true = fdrcorrection(pvals_true)[0].sum()
num_pos_hyper_random_reg = fdrcorrection(pvals_random_reg)[0].sum()
num_pos_hyper_go_permuted = fdrcorrection(pvals_random_go)[0].sum()

# %%
results_binom = gsea_obj.__binomial_glm__(all_regions[DE_indexes], 
                                            all_regions, cores=16,
                                            yes_really=True)
results_binom_random_reg = gsea_obj.__binomial_glm__(query_random, 
                                                        all_regions, cores=16,
                                                        yes_really=True)
# %%
results_binom_go_permuted = gsea_obj_random.__binomial_glm__(all_regions[DE_indexes], 
                                                        all_regions, cores=16,
                                                        yes_really=True)
# %%
num_pos_binom_true = (results_binom["BH corrected p-value"] < 0.05).sum()
num_pos_binom_random_reg = (results_binom_random_reg["BH corrected p-value"] < 0.05).sum()
num_pos_binom_go_permuted = (results_binom_go_permuted["BH corrected p-value"] < 0.05).sum()
# %%
table = [[num_pos_poisson_true, num_pos_binom_true, num_pos_hyper_true, num_pos_NB_true],
        [num_pos_poisson_random_reg, num_pos_binom_random_reg, num_pos_hyper_random_reg, num_pos_NB_random_reg],
        [num_pos_poisson_go_permuted, num_pos_binom_go_permuted, num_pos_hyper_go_permuted, num_pos_NB_go_permuted]]

# %%
num_pos_great_true = (fdrcorrection(pvals_true)[0][np.isin(gsea_obj.mat.index, results_binom.index)] & (results_binom["BH corrected p-value"] < 0.05)).sum()
num_pos_great_random_reg = (fdrcorrection(pvals_random_reg)[0][np.isin(gsea_obj.mat.index, results_binom_random_reg.index)] & (results_binom_random_reg["BH corrected p-value"] < 0.05)).sum()
num_pos_great_go_permuted = (fdrcorrection(pvals_random_go)[0][np.isin(gsea_obj.mat.index, results_binom_go_permuted.index)] & (results_binom_go_permuted["BH corrected p-value"] < 0.05)).sum()

# %%
table = [[num_pos_poisson_true, num_pos_binom_true, num_pos_hyper_true, num_pos_great_true, num_pos_NB_true],
        [num_pos_poisson_random_reg, num_pos_binom_random_reg, num_pos_hyper_random_reg, num_pos_great_random_reg, num_pos_NB_random_reg],
        [num_pos_poisson_go_permuted, num_pos_binom_go_permuted, num_pos_hyper_go_permuted, num_pos_great_go_permuted, num_pos_NB_go_permuted]]
# %%
table = pd.DataFrame(table, 
             index=["True DE regions", "Randomized DE regions", "True DE regions, randomized genesets"], 
             columns=["Poisson", "Binomial", "Hypergeometric", "Hypergeometric & Binomial", "Negative Binomial (Ours)"])
# %%
table
# %%
