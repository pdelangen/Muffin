"""
Functions for data processing (normalization factors, deviance residuals, PCA with optimal number of components, UMAP...).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from statsmodels.nonparametric.smoothers_lowess import lowess
import umap
from statsmodels.stats.multitest import fdrcorrection
import scipy as sc
import kneed
from scipy.sparse import csr_array
from .utils import FA_models, cluster, normalization, diff_expr, stats




def rescale_input(dataset, plot=False):
    """
    For data with input only !
    Centers and scales fold changes by modifying the input count matrix.
    Novel scaled input is stored in dataset.layers["normalization_factors"]
    """
    # Scale factors (only used for visualisation)
    dataset.obs["size_factors"] = np.sum(dataset.X, axis=1)
    dataset.obs["size_factors"] = dataset.obs["size_factors"]/dataset.obs["size_factors"].mean()
    dataset.obs["size_factors_input"] = np.sum(dataset.layers["input"], axis=1)
    dataset.obs["size_factors_input"] = dataset.obs["size_factors_input"]/dataset.obs["size_factors_input"].mean()
    mean_lfc = np.zeros(len(dataset.obsm["design"]))
    ipStrength = np.zeros(len(dataset.obsm["design"]))
    # Find reference point for fold change over input
    for i in range(len(dataset.X)):
        order1 = np.argsort(dataset.uns["input_random"][i])
        order = np.argsort(dataset.uns["counts_random"][i][order1], kind="stable")
        sortedCounts = dataset.uns["counts_random"][i][order1][order]
        sortedInput = dataset.uns["input_random"][i][order1][order]
        sumCounts = np.cumsum(sortedCounts)
        sumInput =  np.cumsum(sortedInput)
        pj = sumCounts/np.sum(sortedCounts)
        qj = sumInput/np.sum(sortedInput)
        amax = np.argmax(qj-pj)
        mean_lfc[i] = np.log(sumCounts[amax]/sumInput[amax])
        maxEnrich = dataset.layers["input"][i].sum()/dataset.X[i].sum()
        pi0 = maxEnrich*np.exp(mean_lfc[i])
        ipStrength[i] = pi0
    dataset.obs["centered_lfc"] = mean_lfc
    dataset.obs["pi0"] = 1.0/ipStrength
    # Fill input
    dataset.layers["normalization_factors"] = dataset.layers["input"].astype("float32")
    for i in range(dataset.layers["normalization_factors"].shape[1]):
        meanInput = np.mean(dataset.layers["normalization_factors"][:, i] / dataset.obs["size_factors_input"].values) * dataset.obs["size_factors_input"].values
        if meanInput.sum() < 1e-10:
            # In last resort assume there is one input read
            # Since there is not a lot of signal here it does not
            # matter much
            meanInput = 1.0 / len(dataset.X) * dataset.obs["size_factors_input"].values
        dataset.layers["normalization_factors"][:, i] = np.where(dataset.layers["normalization_factors"][:, i] < 0.5, 
                                                        meanInput, dataset.layers["normalization_factors"][:, i])
    dataset.layers["normalization_factors"] *= np.exp(mean_lfc)[:, None]
    ipStrength = ipStrength/np.median(ipStrength)
    # Re-scale input to correct LFC curve
    if plot:
        plt.figure(dpi=500)
    for i in range(len(dataset.X)):
        fc = (np.maximum(dataset.X[i], 1e-1)/dataset.layers["normalization_factors"][i])
        scale_pos = ipStrength[i] * (fc - 1) + 1
        scale_neg = 1.0 / (ipStrength[i] * (1/fc - 1) + 1)
        scale = np.where(fc > 1, scale_pos, scale_neg)
        dataset.layers["normalization_factors"][i] = dataset.layers["normalization_factors"][i]/(scale/fc)
        lfc = np.log(np.maximum(1e-1,dataset.X[i])/dataset.layers["normalization_factors"][i])
        if plot:
            plt.plot((np.sort(lfc)), linewidth=0.5, c=[dataset.obs["est_signal_to_noise"][i]/12.0,0,0])
    if plot:
        plt.xlim(plt.xlim()[0], plt.xlim()[1])
        plt.ylabel("Normalized log(Fold Change)")
        plt.xlabel("Log(Fold Change) rank")
        plt.hlines(0.0, plt.xlim()[0], plt.xlim()[1])
        plt.show()
    

def trim_low_counts(dataset, min_exp=3, min_counts=1, min_mean=0.0):
    """
    Returns a boolean array of variables with at least 
    min_counts in min_exp rows, and with mean normalized 
    value above min_mean. 

    Parameters
    ----------
    min_exp : int, optional
        Minimum number of experiment to have at least min_counts, by default 3, at least 1
    min_counts : int, optional
        Minimum number of counts, by default 1
    min_mean : float, optional
        Mean normalized count threshold, by default 0.0

    Returns
    -------
    kept_features: boolean ndarray
        Variables that satisfies above criterions.
    """
    if min_exp < 1 or min_counts < 1:
        raise ValueError(f"min_exp and min_counts have to be 1 or greater.")
    dataset.var["means"] = stats.computeMeans(dataset.X, dataset.obs["size_factors"].values)
    dropped_features = stats.computeDropped(dataset.X, dataset.var["means"].values, 
                                                   min_exp, min_counts, min_mean)
    return ~dropped_features
    
def remove_low_peaks(dataset, minExp=3):
    """_summary_

    Parameters
    ----------
    dataset : _type_
        _description_
    minExp : int, optional
        _description_, by default 3
    """    
    pass

def pseudo_peak_calling(dataset, alpha=0.05, minFC=1.0, minExp=3):
    """
    Performs a pseudo-peak calling using the fitted NB model
    and normalized input.
    Returns a boolean array of features with significant 
    enrichment over input in at least minExp rows.

    Parameters
    ----------
    alpha : float, optional
        Peak fdr threshold, by default 0.05
    minFC : float, optional
        Fold change over input threshold, by default 1.0
    minExp : int, optional
        Minimal number of samples/cells with significant 
        enrichment over input, by default 3.

    Returns
    -------
    kept_features: boolean ndarray
        Boolean array of features with significant 
        enrichment over input in at least minExp rows.
    """        
    numEnrich = np.zeros(dataset.X.shape[1], dtype="int32")
    for i in range(len(dataset.X)):
        norm_i = dataset.X[i]
        input_i = np.maximum(dataset.layers["input"][i], 1) * np.exp(dataset.obs["centered_lfc"].iloc[i])
        n = 1 / dataset.var["reg_alpha"].values
        p = input_i / (input_i + dataset.var["reg_alpha"].values * (input_i**2))
        peak_p = sc.stats.nbinom(n, p).sf(norm_i-1)
        fc = norm_i / input_i
        fdr = fdrcorrection(peak_p)[1]
        numEnrich += ((fdr < alpha) & (fc > minFC)).astype(int)
    kept = numEnrich >= minExp
    return kept

def feature_selection_chisquare(dataset, alpha=0.05):
    """_summary_

    Parameters
    ----------
    dataset : AnnData
        _description_
    alpha : float, optional
        _description_, by default 0.05

    Returns
    -------
    _type_
        _description_
    """     
    ssr = np.sum(np.square(dataset.layers["residuals"]),axis=0)
    pval = sc.stats.chi2(dataset.obsm["design"].shape[0]-dataset.obsm["design"].shape[1]).sf(ssr)
    return fdrcorrection(pval, alpha=alpha)[0]

def feature_selection_topK(dataset, k_features):
    """
    Returns a boolean array of the k features with the 
    largest sum of squared residuals.

    Parameters
    ----------
    k_features : int
        Number of features to keep

    Returns
    -------
    boolean ndarray
        Boolean array of highly variable features.
    """        

    ssr = np.sum(np.square(dataset.layers["residuals"]),axis=0)
    order = np.argsort(ssr)[::-1]
    bool_array = np.zeros(len(ssr), dtype=bool)
    bool_array[order[:k_features]] = True
    return bool_array


def feature_selection_elbow(dataset, plot:bool=True, subsample=10000):
    """
    Performs feature selection by finding a conservative ankle point of the
    ranked sum of squared residuals curve.

    Parameters
    ----------
    plot : bool, optional
        Whether to plot the ranked SSR curve with the ankle point,
        by default True

    Returns
    -------
    boolean ndarray
        Boolean array of highly variable features.
    """
    ssr = np.sum(np.square(dataset.layers["residuals"]),axis=0)
    subsample_ssr = np.sort(ssr)[np.linspace(0, len(ssr)-1, subsample, dtype=int)]
    elbow_locator = kneed.KneeLocator(np.arange(subsample), subsample_ssr, 
                                        curve="convex", interp_method="polynomial",
                                        online=True)
    min_ssr = elbow_locator.elbow_y
    if plot:
        elbow_locator.plot_knee()
        plt.ylabel("Sum of squared residuals")
        plt.xlabel("Feature rank")
        plt.show()
    return ssr >= min_ssr

def compute_size_factors(dataset, method="top_fpkm"):
    if method == "top_fpkm":
        values = normalization.top_detected_sum_norm(dataset.X)
    elif method == "scran":
        values = normalization.scran_norm(dataset.X)
    elif method == "deseq":
        values = normalization.median_of_ratios(dataset.X)
    elif method == "fpkm":
        values = normalization.sum_norm(dataset.X)
    elif method == "fpkm_uq":
        values = normalization.fpkm_uq(dataset.X)
    else:
        raise ValueError(f"Invalid method : {method}, use either top_fpkm, scran, deseq, fpkm, fpkm_uq")
    dataset.obs["size_factors"] = (values / np.mean(values)).astype("float32")
    return dataset

def computeResiduals(dataset, residuals="deviance", clip="auto", subSampleEst=2000, maxThreads=-1, verbose=True, plot=True):
    """
    Compute residuals from the regularized NB model for each feature.

    Parameters
    ----------
    residuals : str, optional
        Whether to compute "deviance" residuals or "pearson" residuals,
        by default "deviance"
    maxThreads : int, optional
        Number of threads to use, by default -1 (all)
    verbose : bool, optional
        Verbosity, by default True
    plot : bool, optional
        Whether to plot feature importance or not on the mean-variance graph,
        by default True

    Returns
    -------
    self
    """
    np.random.seed(42)
    if "means" not in dataset.var.keys():
        dataset.var["means"] = np.apply_along_axis(stats.normAndMean, 0, dataset.X, 
                                                        dataset.obs["size_factors"].values)
    dataset.var["variances"] = np.apply_along_axis(stats.normAndVar, 0, dataset.X, 
                                                        dataset.obs["size_factors"].values)
    logMean = np.log(dataset.var["means"])
    # Estimate regularized variance in function of mean expression
    nLowess = min(subSampleEst, dataset.X.shape[1])
    indices = np.linspace(0, dataset.X.shape[1]-1, nLowess, dtype=int)
    meanOrder = np.argsort(dataset.var["means"])
    subset = meanOrder[indices]
    alphas = np.zeros(nLowess)
    if "input" in dataset.layers.keys():
        if not "normalization_factors" in dataset.layers.keys():
            raise Exception("Input counts have NOT been scaled!")
        with Parallel(n_jobs=maxThreads, verbose=verbose, batch_size=128, max_nbytes=None) as pool:
            alphas = pool(delayed(stats.fit_alpha_input)(dataset.X[:, subset[i]], 
                                                                dataset.obsm["design"], 
                                                                dataset.layers["normalization_factors"][:, subset[i]]) for i in range(nLowess))
    else:
        with Parallel(n_jobs=maxThreads, verbose=verbose, batch_size=128, max_nbytes=None) as pool:
            alphas = pool(delayed(stats.fit_alpha)(dataset.obs["size_factors"], 
                                                          dataset.X[:, subset[i]], 
                                                          dataset.obsm["design"]) for i in range(nLowess))
    # Kill workers or they keep being active even if the program is shut down
    get_reusable_executor().shutdown(wait=False, kill_workers=True)
    alphas = np.array(alphas)
    validAlphas = (alphas > 1e-3) & (alphas < 1e5)
    logAlpha = lowess(np.log(alphas[validAlphas]), indices[validAlphas], 
                            xvals=indices, frac=0.1, return_sorted=False)
    logAlpha = np.clip(logAlpha, 
                        np.log(alphas[validAlphas]).min(),
                        np.log(alphas[validAlphas]).max())
    dataset.var["reg_alpha"] = np.exp(sc.interpolate.interp1d(indices, logAlpha)(np.argsort(meanOrder)))
    # Dispatch accross multiple processes
    sf = dataset.obs["size_factors"].values
    design = dataset.obsm["design"]
    regAlpha = dataset.var["reg_alpha"].values
    if "normalization_factors" not in dataset.layers.keys():
        with Parallel(n_jobs=maxThreads, verbose=verbose, batch_size=128, max_nbytes=None) as pool:
            residuals = pool(delayed(stats.compute_residuals)(regAlpha[i],
                                                                 sf, 
                                                                 dataset.X[:, i], 
                                                                 design, 
                                                                 residuals) for i in range(dataset.X.shape[1]))
    else:
        with Parallel(n_jobs=maxThreads, verbose=verbose, batch_size=128, max_nbytes=None) as pool:
            residuals = pool(delayed(stats.compute_residuals_input)(regAlpha[i], 
                                                                        dataset.X[:, i], 
                                                                        design, 
                                                                        residuals, 
                                                                        dataset.layers["normalization_factors"][:, i]) for i in range(dataset.X.shape[1]))
    # Kill workers or they keep being active even if the program is shut down
    get_reusable_executor().shutdown(wait=False, kill_workers=True)
    if clip == "auto":
        clip = np.sqrt(9+len(dataset)/4)
    dataset.layers["residuals"] = np.clip(np.array(residuals, copy=False).T, -clip, clip)
    if plot:
        # Plot mean/variance relationship and selected probes
        v = dataset.var["variances"]
        m = dataset.var["means"]
        plt.figure(dpi=500)
        w = np.std(dataset.layers["residuals"], axis=0)
        w = w / w.max()
        plt.scatter(m, v, s = 2.0*(100000/len(m)), linewidths=0, c=w, alpha=1.0)
        plt.scatter(m, m+m*m*dataset.var["reg_alpha"], s = 1.0, linewidths=0, c=[0.0,1.0,0.0])
        pts = np.geomspace(max(m.min(), 1.0/len(dataset.obs["size_factors"])), m.max())
        plt.plot(pts, pts)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Feature mean")
        plt.ylabel("Feature variance")
        plt.show()
        plt.close()
    return dataset
    
def compute_PA_PCA(dataset, layer="residuals", feature_mask=None, perm=3, alpha=0.01, 
                    solver="randomized", whiten=True,
                    max_rank=None, mincomp=2, plot=False):
    """
    Permutation Parallel Analysis to find the optimal number of PCA components.
    Stored under self.reduced_dims["PCA"].

    Parameters
    ----------
    feature_name: str, optional
        On which feature to compute PCA on (e.g. "residuals")
    
    feature_mask : boolean ndarray or None, optional
        Subset of features to use for Umap, by default None.
    

    perm: int (default 3)
        Number of permutations. On large matrices (row * col > 1,000,000) the eigenvalues are 
        very stable, so there is no need to use a large number of permutations,
        and only one permutation can be a reasonable choice on large matrices.
        On smaller matrices the number of permutations should be increased.

    alpha: float (default 0.01)
        Permutation p-value threshold.
    
    solver: "arpack" or "randomized"
        Chooses the SVD solver. Randomized is much faster but very slightly less accurate.
    
    whiten: bool (default True)
        If set to true, each component is transformed to have unit variance.

    max_rank: int or None (default None)
        Maximum number of principal components to compute. Must be strictly less 
        than the minimum of n_features and n_samples. If set to None, computes
        up to min(n_samples, n_features)-1 components.
    
    mincomp: int (default 2)
        Number of components to return

    plot: bool (default None)
        Whether to plot the randomized eigenvalues of not
    """
    if feature_mask is None:
        data = dataset.layers[layer]
    else:
        data = dataset.layers[layer][:, feature_mask]
    dataset.obsm["X_pca"], model = FA_models.permutationPA_PCA(data, perm=perm, 
                                                            alpha=alpha, solver=solver, whiten=whiten,
                                                            max_rank=max_rank, mincomp=mincomp, 
                                                            plot=plot, returnModel=True)
    if feature_mask is not None:
        dataset.varm["PCs"] = np.zeros((dataset.obsm["X_pca"].shape[1], dataset.X.shape[1])).T
        dataset.varm["PCs"][feature_mask] = model.components_[:dataset.obsm["X_pca"].shape[1]].T
    else:
        dataset.varm["PCs"] = model.components_[:dataset.obsm["X_pca"].shape[1]].T
    dataset.uns["pca"] = dict()
    dataset.uns["pca"]['variance_ratio'] = model.explained_variance_ratio_[:dataset.obsm["X_pca"].shape[1]]
    dataset.uns['pca']['variance'] = model.explained_variance_[:dataset.obsm["X_pca"].shape[1]]

def compute_UMAP(dataset, on="reduced_dims", which="X_pca", feature_mask=None, umap_params={}):
    """
    Compute UMAP and stores it under dataset.reduced_dims["X_umap"].

    Parameters
    ----------
    on : str, optional
        On which data representation to perform clustering. Use
        "reduced_dims" or "features", by default "reduced_dims".
    which : str, optional
        Which reduced_dims or feature to use. I.e. "PCA" or "residuals", 
        by default "PCA".
    feature_mask : boolean ndarray, optional
        Subset of features to use for Umap (works with PCA as well), 
        by default None
    umap_params : dict, optional
        Dictionnary of keyword arguments for UMAP, see UMAP documentation, 
        by default if not provided metric is set to euclidean with 10 or less input dimensions,
        and to correlation otherwise. Random state is also fixed and should not 
        be provided.

    Returns
    -------
    dataset
    """
    if on == "reduced_dims":
        try:
            if feature_mask is None:
                data = dataset.obsm[which]
            else:
                data = dataset.obsm[which][:, feature_mask]
        except KeyError:
            raise KeyError(f"Invalid 'which' parameter : {which}, make sure you have initialized the key or computed pca beforehand.")
    elif on == "features":
        try:
            if feature_mask is None:
                data = dataset.layers[which]
            else:
                data = dataset.layers[which][:, feature_mask]
        except KeyError:
            raise KeyError(f"Invalid 'which' parameter : {which}, make sure you have initialized the key or computed residuals beforehand.")
    else:
        raise ValueError("Invalid 'on' parameter, use either 'reduced_dims' or 'features'")
    if not "metric" in umap_params.keys():
        if data.shape[1] > 10:
            umap_params["metric"] = "correlation"
        else:
            umap_params["metric"] = "euclidean"
    umap_params["random_state"] = 42
    dataset.obsm["X_umap"] = umap.UMAP(**umap_params).fit_transform(data)
    return dataset

def cluster_rows_leiden(dataset, on="reduced_dims", which="X_pca", feature_mask=None,
                    metric="correlation", k="auto", r=1.0, restarts=10):  
    """
    Computes Shared Nearest Neighbor graph clustering.

    Parameters
    ----------
    on : str, optional
        On which data representation to perform clustering. Use
        "reduced_dims" or "features", by default "reduced_dims".
    which : str, optional
        Which reduced_dims or feature to use. I.e. "PCA" or "residuals", 
        by default "PCA".
    feature_mask : boolean ndarray, optional
        Subset of features to use for Umap (works with PCA as well), 
        by default None
    metric : str, optional
        Metric to use for kNN search, by default "correlation"
    k : str, optional
        Number of nearest neighbors to find, 
        by default "auto" uses 5*nFeatures^0.2 as a rule of thumb.
    r : float, optional
        Resolution parameter of the graph clustering, by default 1.0
    restarts : int, optional
        Number of times to restart the graph clustering before
        keeping the best partition, by default 10
    
    Returns
    -------
    self
    """        
    if on == "reduced_dims":
        try:
            if feature_mask is None:
                data = dataset.obsm[which]
            else:
                data = dataset.obsm[which][:, feature_mask]
        except KeyError:
            raise KeyError(f"Invalid 'which' parameter : {which}, make sure you have initialized the key or computed pca beforehand.")
    elif on == "features":
        try:
            if feature_mask is None:
                data = dataset.layers[which]
            else:
                data = dataset.layers[which][:, feature_mask]
        except KeyError:
            raise KeyError(f"Invalid 'which' parameter : {which}, make sure you have initialized the key or computed residuals beforehand.")
    else:
        raise ValueError("Invalid 'on' parameter, use either 'reduced_dims' or 'features'")
    dataset.obs["leiden"] = cluster.graphClustering(data, metric=metric, 
                                                         k=k, r=r, restarts=restarts).astype(str)
    return dataset

def differential_expression_A_vs_B(dataset, category, ref_category, alternative="two-sided",
                                   method="auto"):
    """

    Parameters
    ----------
    dataset : _type_
        _description_
    category : _type_
        _description_
    ref_category : _type_
        _description_
    method : str, optional
        _description_, by default "auto"

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    if method == "auto":
        if len(dataset) > 50:
            print("Using t-test due to large dataset")
            method = "t-test"
        else:
            method = "deseq"
    factors, comp = pd.factorize(dataset.obs[category])
    test = (comp != ref_category).astype(int)
    factors = np.array(["A", "B"])[test][(factors!=0).astype(int)]
    if len(comp) != 2:
        raise ValueError(f"Invalid number of categories, found {len(comp)} unique values in {category} : {comp}")
    print(f"Comparing {comp[test][1]} to (reference) {comp[test][0]}")
    if method == "deseq":
        if "input" in dataset.layers.keys():
            if not "normalization_factors" in dataset.layers.keys():
                raise Exception("Input counts have NOT been scaled!")
        if "normalization_factors" not in dataset.layers.keys():
            print("Using DESeq2 with normalization factors per row")
            res = diff_expr.DESeq2(dataset.X, dataset.obs["size_factors"],
                                   factors, dataset.obsm["design"])
        else:
            print("Using DESeq2 with normalization factors per row, per gene")
            res = diff_expr.DESeq2_input(dataset.X, dataset.layers["normalization_factors"],
                                   factors, dataset.obsm["design"])
        res = res.loc[:, ["z-score", "log2FoldChange", "pvalue", "padj"]]
    elif method == "t-test":
        res = diff_expr.t_test(dataset.layers["residuals"],
                               dataset.X,
                               dataset.obs["size_factors"],
                               factors)
    elif method == "wilcoxon":
        res = diff_expr.wilcoxon(dataset.layers["residuals"],
                               dataset.X,
                               dataset.obs["size_factors"],
                               factors)
    res.index = dataset.var_names
    dataset.varm["DE_results"] = res
    ordered = res.sort_values("z-score", ascending=False)
    reverseOrdered = res.sort_values("z-score", ascending=True)
    dataset.uns["rank_genes_groups"] = dict()
    dataset.uns["rank_genes_groups"]["params"] = {"groupby":category,
                                                  "reference":ref_category,
                                                  "method":method,
                                                  "use_raw":False,
                                                  "layer":"residuals",
                                                  "corr_method":"benjamini-hochberg"}
    dataset.uns["rank_genes_groups"]["names"] = np.recarray((len(res),), 
                                                            dtype=[(comp[test][1], 'O'), (comp[test][0], 'O'),])  
    dataset.uns["rank_genes_groups"]["names"][comp[test][1]] = ordered.index
    dataset.uns["rank_genes_groups"]["names"][comp[test][0]] = reverseOrdered.index
    dataset.uns["rank_genes_groups"]["scores"] = np.recarray((len(res),), 
                                                            dtype=[(comp[test][1], 'float'), (comp[test][0], 'float'),])  
    dataset.uns["rank_genes_groups"]["scores"][comp[test][1]] = ordered["z-score"]
    dataset.uns["rank_genes_groups"]["scores"][comp[test][0]] = -reverseOrdered["z-score"]
    dataset.uns["rank_genes_groups"]["pvals"] = np.recarray((len(res),), 
                                                            dtype=[(comp[test][1], 'float'), (comp[test][0], 'float'),])  
    dataset.uns["rank_genes_groups"]["pvals"][comp[test][1]] = ordered["pvalue"]
    dataset.uns["rank_genes_groups"]["pvals"][comp[test][0]] = reverseOrdered["pvalue"]
    dataset.uns["rank_genes_groups"]["pvals_adj"] = np.recarray((len(res),), 
                                                            dtype=[(comp[test][1], 'float'), (comp[test][0], 'float'),])  
    dataset.uns["rank_genes_groups"]["pvals_adj"][comp[test][1]] = ordered["padj"]
    dataset.uns["rank_genes_groups"]["pvals_adj"][comp[test][0]] = reverseOrdered["padj"]
    dataset.uns["rank_genes_groups"]["logfoldchanges"] = np.recarray((len(res),), 
                                                            dtype=[(comp[test][1], 'float'), (comp[test][0], 'float'),])  
    dataset.uns["rank_genes_groups"]["logfoldchanges"][comp[test][1]] = ordered["log2FoldChange"]
    dataset.uns["rank_genes_groups"]["logfoldchanges"][comp[test][0]] = -reverseOrdered["log2FoldChange"]


def mega_heatmap(dataset, layer, ):
    pass