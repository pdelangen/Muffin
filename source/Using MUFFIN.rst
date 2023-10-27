Installation
------------

Get MUFFIN via conda :

.. code-block:: bash

   conda install -c bioconda muffin


Loading your data
-----------------

MUFFIN offers two methods to load in your data : via BAM files and genomic regions (in bed, SAF or GTF format) 
or manually via python arrays. Note that MUFFIN only accepts raw, non-normalized counts. Datasets are stored in the AnnData format. 
For more information on how it works, see the `AnnData documentation <https://anndata.readthedocs.io/en/latest/index.html>`_ .
Additionnally, it should be compatible with Scanpy data loading methods if you convert the count tables to dense arrays.

This is the most complicated part if your dataset is not in an usual format.

.. code-block:: python

   import muffin
   # Load from a python array (see the API ref for all details)
   dataset = muffin.load.dataset_from_arrays(array, row_names=sample_names,
                                             col_names=feature_names)

   # Using one of Scanpy's loader
   import scanpy as sc
   dataset = sc.read_10x_mtx(path)
   dataset.X = dataset.X.astype(np.int32).toarray()

   # With BAM files. bam_files are a list of paths.
   # You can use either genomic_regions_path or genomic_regions to specify which regions to sample the signal from.
   # The former is a path to the genome annotation (in BED, SAF or GTF format) 
   # while the latter is already stored in a pandas dataframe.
   dataset = muffin.load.dataset_from_bam(bam_files, genomic_regions_path=path)

   # With BAM files and Input BAM file. 
   # Here chromSizes is mandatory and is a dictionnary of the form {"chr1":size, "chr2":size}
   dataset = muffin.load.dataset_from_bam(bam_files, genomic_regions=regions,
                                          input_bam_paths=input_files,
                                          chromsizes=chromSizes)


You can correct for unwanted sources of variations by inputing a design matrix. 
If you do not want to correct for confounding factors, just keep a column vector of ones as in the example.

.. code-block:: python

   muffin.load.set_design_matrix(np.ones(len(dataset)))


Setting normalization factors
-----------------------------
Depending on your dataset, you should use a different type of normalization.

.. code-block:: python

   # Recommended with deep sequencing
   muffin.tools.compute_size_factors(dataset, method="deseq")
   # Recommended with small counts and large number of samples
   muffin.tools.compute_size_factors(dataset, method="scran")
   # Recommended with small counts and small number of samples
   # This is the default as it works with most datasets
   muffin.tools.compute_size_factors(dataset, method="top_fpkm")
   # Datasets with input
   muffin.tools.rescale_input_center_scale(dataset)

Alternatively, you can provide your own normalization factors.

.. code-block:: python

   # Per observation normalization vector
   muffin.load.set_size_factors(dataset, your_size_factors)
   # Per observation, per variable normalization matrix
   muffin.load.set_normalization_factors(dataset, your_normalization_factors)

Removing unused variables
-------------------------
It is a MANDATORY step to remove all-zeroes variables that do not carry any signal.
By default Muffin removes variables that do not have at least 1 count in at least 3 experiments.

.. code-block:: python
   
   nonzero = muffin.tools.trim_low_counts(dataset)
   dataset = dataset[:, nonzero]


Count Modelling and transformation
----------------------------------
At the core of muffin is its count modelling method based on a Negative Binomial (NB) model. 
This step transforms counts to residuals of a regularized NB model. 
You can think of this as something similar to a z-score of logCPM values, but more robust and flexible. 
However residuals give more weight to sufficiently expressed variables and to those with large variability.
The results are stored in dataset.layers["residuals"] .

.. code-block:: python
   
   muffin.tools.compute_residuals(dataset)


Feature Selection
-----------------
This is a facultative step that helps to remove variables with low expression or low variability across samples, which are carrying not a lot of information.
Do not erase the original dataset as it can still be used when performing Differential Expression !

.. code-block:: python

   # Conservative approach (recommended)
   selected = muffin.tools.feature_selection_elbow(dataset)
   # For dataset with input sequencing we provide a tool to remove variables with low fold change over input
   peaks = muffin.tools.pseudo_peak_calling(dataset)


Dimensionnality reduction
-------------------------
We use provide a UMAP wrapper, and implement PCA with optimal number of component selection using Parallel Analysis (or jackstraw).
By default PCA will be run on residuals, and UMAP on the PCA representation.
Depending on your dataset, we recommend different approaches: 
- With a dataset with a large number of observations, perform PCA then UMAP
- If there is not a lot of observations, perform either only PCA or UMAP
As in Scanpy, these representations are stored in .obsm["X_pca"] and .obsm["X_umap"]

.. code-block:: python

   # PCA. We provide the selected features computed previously in order to not erase the dataset !
   muffin.tools.compute_pa_pca(dataset, feature_mask=selected, max_rank=100, plot=True)
   # UMAP
   muffin.tools.compute_umap(dataset, umap_params={"min_dist":0.5, "n_neighbors":30})
   # UMAP, directly on residuals
   muffin.tools.compute_umap(dataset, on="features", which="residuals", feature_mask=hv, 
                             umap_params={"min_dist":0.5, "n_neighbors":30})

Downstream analyses
-------------------
Clustering
**********
This is a crucial step of most scRNA-seq pipelines. We implement a custom graph clustering method, but you can also use the one from Scanpy.

.. code-block:: python

   muffin.tools.cluster_rows_leiden(dataset)

Differential expression
***********************
We provide a wrapper to DESeq2 to perform a two-categories differential expression. 
Note that we pass the design matrix to DESeq2.
Results will be stored in dataset.varm["DE_results"],
and for compatibility with scanpy, in dataset.uns["rank_genes_groups"].

.. code-block:: python

   # Here, category is a column name in dataset.obsm . 
   # ref_category is the reference category from which log fold changes will be computed.
   # If more than two uniques value are present in the column, an error will be raised !
   muffin.tools.differential_expression_A_vs_B(dataset, category, ref_category)

In the case of multi-categories differential expression, we recommend using Scanpy's logistic regression method :

.. code-block:: python

   from sklearn.preprocessing import StandardScaler
   dataset.layers["scaled"] = StandardScaler().fit_transform(dataset.layers["residuals"])
   sc.tl.rank_genes_groups(dataset, 'Subtype', use_raw=False, layer="scaled",
                           method='logreg', class_weight="balanced")
   # Ugly hack to solve an issue with scanpy logreg that does not output all fields for its visualization tools
   dataset.uns["rank_genes_groups"]["logfoldchanges"] = dataset.uns["rank_genes_groups"]["scores"]
   dataset.uns["rank_genes_groups"]["pvals"] = dataset.uns["rank_genes_groups"]["scores"]
   dataset.uns["rank_genes_groups"]["pvals_adj"] = dataset.uns["rank_genes_groups"]["scores"]


Gene Set Enrichment Analysis of genomic regions
***********************************************
If you are working with genomic regions instead of genes, we provide tools to link your genomic regions to genes and functional annotations.
This particularly important for assays such as ATAC-seq or ChIP-seq. Our method supposes that your regions of interest are a subset of background regions
(for example, all regions considered for DE testing and DE regions).
We recommend you to check the ATAC-seq and ChIP-seq examples for more details.

.. code-block:: python

   # Initialize the GSEA object
   # A gmt file is in the format :
   # term_id1 \t term_name1 \t gene1 \t gene2...\n
   # term_id2 \t term_name2 \t gene1 \t gene2...\n
   gsea_obj = muffin.grea.pyGREAT(geneset_gmt_file, gtf_file, chromSizes_file)
   # Link to genes
   dataset.var_names = gsea_obj.label_by_nearest_gene(dataset.var[["Chromosome","Start","End"]]).astype(str)
   # Assume we performed differential expression and want to see the affected gene sets.
   # Retrieve DE regions
   DE_indexes = (dataset.varm["DE_results"]["padj"] < 0.05) & (np.abs(dataset.varm["DE_results"]["log2FoldChange"]) > 1.0)
   all_regions = dataset.var[["Chromosome", "Start", "End"]]
   query = all_regions[DE_indexes]
   # Perform GREA (Genomic Region Enrichment Analysis)
   gsea_results = gsea_obj.find_enriched(query, all_regions, cores=16)
   # Visualize clustered enrichments
   gsea_obj.cluster_treemap(gsea_results)


Interfacing with the Scanpy ecosystem
*************************************
Outputs of MUFFIN are stored in the AnnData format, and mimics the data slots that Scanpy uses internally, which makes the use of Scanpy functions seamless. 
If you want to visualize the expression levels across different conditions or clusters, residuals are stored in the .layers["residuals"] data slot.