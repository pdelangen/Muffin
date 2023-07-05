"""
Functions to create a dataset from arrays or bam files.
"""
import anndata as ad
import numpy as np
import pandas as pd
import pyranges as pr
from .utils import utils
import warnings
import muffin

def dataset_from_arrays(count_table, row_names=None, col_names=None, input_count_table=None,
                        random_loc_count_table=None, random_loc_input_table=None):
    """
    Import count matrices from python array.

    Parameters
    ----------
    count_table : integer ndarray
        (sample, genomic feature) RAW, NON NORMALIZED COUNT data. To save time and memory
        you can use an int32 format.
    row_annot : None or dict
        If set to None, will give an integer id to each row (1->n).
        If set to a dict, expects a dict of the form {annotation_name:annotations},
        where annotations are ordered according to the count table.
        By default None.
    col_annot : None or dict
        If set to None, will give an integer id to each column (1->n).
        If set to a dict, expects a dict of the form {annotation_name:annotations},
        where annotations are ordered according to the count table.
        By default None.
    input_count_table : integer or float ndarray or None, optional
        For sequencing data with input only !
        (sample, genomic feature) Raw, non normalized input counts (int) or 
        input with custom normalization. If your input counts are already normalized,
        you have to set input_already_normalized to True.
        To save time and memory you can use a 32bits format.
        By default None
    random_loc_count_table : integer ndarray, optional
        For sequencing data with input only ! Mandatory if your input count table
        is not already normalized.
        (sample, n_random_intervals) Raw, non normalized counts (int) sampled
        at random genomic intervals (preferably at least 5,000). 
        To save time and memory you can use an int32 format.
        By default None
    random_loc_input_table : integer ndarray, optional
        For sequencing data with input only ! Mandatory if your input count table
        is not already normalized.
        (sample, n_random_intervals) Raw, non normalized counts (int) sampled
        at random genomic intervals (preferably at least 5,000). 
        To save time and memory you can use an int32 format.
        By default None

    Returns
    -------
    dataset : AnnData
        Annotated data matrix
    """
    dataset = ad.AnnData(count_table, dtype=count_table.dtype)
    if row_names is None:
        dataset.obs_names = np.arange(len(count_table)).astype(str)
    else:
        dataset.obs_names = row_names.astype(str)
    if col_names is None:
        dataset.var_names = np.arange(len(count_table)).astype(str)
    else:
        dataset.var_names = col_names.astype(str)
    if input_count_table is not None:
        dataset.layers["input"] = input_count_table
        dataset.uns["counts_random"] = random_loc_count_table
        dataset.uns["input_random"] = random_loc_input_table
    return dataset

def dataset_from_bam(bam_paths, genomic_regions=None, genomic_regions_path=None, row_names=None, col_names=None,
                    featureCounts_params=None, isBEDannotation=True, input_bam_paths=None, 
                    n_random_input=10000, chromsizes=None):
    """
    Initialize count matrices using BAM files and query random regions.

    Parameters
    ----------
    bam_paths : str array-like 
        Paths to BAM files.
    genomic_regions: pandas Dataframe
        Genomic annotation in bed format
    genomic_regions_path : str
        Path to a bed or SAF genome annotation file
    row_annot : None or dict, optional
        If set to None, will give file name to each row (1->n).
        If set to a dict, expects a dict of the form {annotation_name:annotations},
        where annotations are ordered according to the BAM file order.
        By default None.
    col_annot : None or dict, optional
        If set to None, will give use the given genomic regions as annotations.
        If set to a dict, expects a dict of the form {annotation_name:annotations},
        where annotations are ordered according to the genomic_regions_path.
        By default None.
    featureCounts_params : None or dict, optional
        Dictionary of keyword arguments to be passed to featureCounts, see featureCounts
        R package documentation, by default None
    isBEDannotation : bool, optional
        If set to true assumes the genomic_regions_path is in .bed format with strand
        information otherwise, it assumes a SAF file (see featureCounts doc), by default True.
    input_bam_paths : str, optional
        For sequencing data with input only !
        Paths to input BAM files, path order has to match the signal bam_paths.
        By default None.
    n_random_input : int, optional
        For sequencing data with input only !
        Number of random regions to sample to estimate centering and scaling factors.
        By default 10000.
    chromsizes : dict or None, optional
        Mandatory for sequencing data with input !
        Set of chromosome that will be used to generate random sampling locations.
        Dict with format : {"chr1":3000000, "chr2":456455412}.

    Returns
    -------
    dataset
        _description_
    """        
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
    subread = importr("Rsubread")
    utils.createDir(muffin.params["temp_dir"])
    # Append random regions to bed
    if input_bam_paths is not None:
        if chromsizes is None:
            raise ValueError("Missing chromosome size dictionary !")
        np.random.seed(42)
        randomRegions = pr.random(n=n_random_input, length=500, chromsizes=chromsizes, strand=False, int64=True).as_df()
        randomRegions.columns = ["Chr", "Start", "End"]
        randomRegions.insert(0, "GeneID", ["r"+str(i) for i in range(n_random_input)])
        randomRegions.insert(3, "Strand", ["." for i in range(n_random_input)])
    # Convert bed file to SAF
    if isBEDannotation:
        if genomic_regions is None:
            bedFile = pd.read_csv(genomic_regions_path, sep="\t", header=None)
        else:
            bedFile = genomic_regions
        ncols = bedFile.shape[1]
        if ncols >= 6:
            bedFile = bedFile.iloc[:, [0,1,2,5]]
            bedFile.insert(0, "GeneID", ["r"+str(i) for i in range(len(bedFile))])
            bedFile.columns = ["GeneID", "Chr", "Start", "End", "Strand"]
        else:
            bedFile = bedFile.iloc[:, [0,1,2]]
            bedFile.columns = ["Chr", "Start", "End"]
            bedFile.insert(0, "GeneID", ["r"+str(i) for i in range(len(bedFile))])
            bedFile.insert(3, "Strand", ["." for i in range(len(bedFile))])
            bedFile.columns = ["GeneID", "Chr", "Start", "End", "Strand"]
        if col_names is None:
            col_names = np.arange(len(bedFile))
        bedFile.loc[:, "GeneID"] = col_names
        bedFile = pd.concat([bedFile, randomRegions])
        bedFile.to_csv(muffin.params["temp_dir"] + "/ann.saf", sep="\t", index=None)
        genomic_regions_path = muffin.params["temp_dir"] + "/ann.saf"
    else:
        genomic_regions.to_csv(muffin.params["temp_dir"] + "/ann.saf", sep="\t", index=None)
        genomic_regions_path = muffin.params["temp_dir"] + "/ann.saf"
    paramsDict = {"files":ro.vectors.StrVector(bam_paths)}
    # Count reads over genomic regions
    paramsDict["annot.ext"] = genomic_regions_path
    if featureCounts_params is not None:
        paramsDict = dict(paramsDict, **featureCounts_params)
    # Sanitize arguments for rpy2
    paramsDictRpy2 = {}
    for k in paramsDict:
        paramsDictRpy2[k.replace(".", "_")] = paramsDict[k]
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        res = subread.featureCounts(**paramsDictRpy2)
        mapping_stats = pd.DataFrame(res["stat"]).set_index("Status")
        if input_bam_paths is None:
            dataset = ad.AnnData(res["counts"].T.astype(np.int32), dtype=np.int32)
        else:
            dataset = ad.AnnData(res["counts"].T.astype(np.int32)[:, :-n_random_input], dtype=np.int32)
            dataset.uns["counts_random"] = res["counts"].T.astype(np.int32)[:, -n_random_input:]
        dataset.uns["tot_mapped_counts"] = mapping_stats.sum(axis=0).values
    # Same but for input files
    if input_bam_paths is not None:
        # Take care of the case where some files may share the same input
        # Which causes an issue with featureCounts
        unique_paths = list(set(input_bam_paths))
        paramsDict = {"files":ro.vectors.StrVector(unique_paths)}
        if genomic_regions_path is not None:
            paramsDict["annot.ext"] = genomic_regions_path
        if featureCounts_params is not None:
            paramsDict = dict(paramsDict, **featureCounts_params)
        # Sanitize arguments for rpy2
        paramsDictRpy2 = {}
        for k in paramsDict:
            paramsDictRpy2[k.replace(".", "_")] = paramsDict[k]
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            res = subread.featureCounts(**paramsDictRpy2)
            if len(unique_paths) != len(input_bam_paths):
                warnings.warn("Multiple files share the same input (this can be intended)")
                unique_counts = res["counts"].T.astype(np.int32)
                dataset.layers["input"] = np.zeros((len(input_bam_paths), unique_counts.shape[1]-n_random_input), dtype="int32")
                dataset.uns["input_random"] = np.zeros((len(input_bam_paths), n_random_input), dtype="int32")
                for i, val in enumerate(input_bam_paths):
                    index = unique_paths.index(val)
                    dataset.layers["input"][i,:] = unique_counts[index,:-n_random_input]
                    dataset.uns["input_random"][i,:] = unique_counts[index, -n_random_input:]
            else:
                dataset.layers["input"] = res["counts"].T.astype(np.int32)[:, :-n_random_input]
                dataset.uns["input_random"] = res["counts"].T.astype(np.int32)[:, -n_random_input:]
                mapping_stats_input = pd.DataFrame(res["stat"]).set_index("Status")
                dataset.uns["tot_mapped_input"] = mapping_stats_input.sum(axis=0).values
                fracMapped = mapping_stats.loc["Assigned"]/dataset.uns["tot_mapped_counts"]
                fracMappedInput = mapping_stats_input.loc["Assigned"]/mapping_stats_input.sum(axis=0)
                dataset.obs["est_signal_to_noise"] = fracMapped.values/fracMappedInput.values
    # Annotate
    if row_names is None:
        dataset.obs_names = np.array([f.split("/")[-1] for f in bam_paths])
    else:        
        dataset.obs_names = row_names.astype(str)
    if col_names is None:
        dataset.var_names = np.arange(dataset.X.shape[1]).astype(str)
    else:
        dataset.var_names = col_names.astype(str)
    dataset.var[["Chromosome", "Start", "End", "Strand"]] = bedFile.values[:-n_random_input, 1:]
    dataset.var[["Start", "End"]] = dataset.var[["Start", "End"]].astype("int")
    dataset.var["Chromosome"] = dataset.var["Chromosome"].astype("category")
    return dataset

def set_design_matrix(dataset, design):
    """
    Set design matrix for statistical modelling.
    If you do not want to remove any confounding factors, set it 
    to a column vector of ones.
    E.g. : np.ones((model.dataset.layers["input"].shape[0],1))

    Parameters
    ----------
    design : ndarray
        Design matrix.

    Returns
    -------
    self
    """
    dataset.obsm["design"] = design

def set_size_factors(dataset, values):
    """
    Load custom size factors for the model.

    Parameters
    ----------
    values : float ndarray
        Size factors, e.g. np.sum(model.matrices["counts"], axis=1).
        They will be standardized to unit mean to ensure numerical
        stability when fitting models.
    """
    dataset.obs["size_factors"] = (values.astype("float64") / np.mean(values.astype("float64"))).astype("float32")

def set_normalization_factors(dataset, normalization_factors):
    """
    Load normalization factors from a numpy array

    Parameters
    ----------
    dataset : _type_
        _description_
    normalization_factors : float ndarray (observations x variables)
        Normalization factors per observation, per variable
    """    
    dataset.obs["normalization_factors"] = normalization_factors