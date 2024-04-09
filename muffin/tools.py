"""
Functions for data processing (normalization factors, deviance residuals, PCA with optimal number of components, UMAP...).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.nonparametric.smoothers_lowess import lowess
import umap
from statsmodels.stats.multitest import fdrcorrection
import scipy as sc
import kneed
from scipy.sparse import csr_array
from .utils import FA_models, cluster, normalization, diff_expr, stats
import warnings
import muffin

def rescale_input_quantile(dataset, plot=False):
    """
    For data with input only !
    Centers and scales fold changes by modifying the input count matrix.
    Novel scaled input is stored in dataset.layers["normalization_factors"]

    Parameters
    ----------
    dataset : AnnData
        Dataset in AnnData format, with input layer filled
    plot : bool, optional
        Whether to plot or not, by default False
    """
    # Scale factors (only used for visualisation)
    dataset.obs["size_factors"] = np.sum(dataset.X, axis=1)
    dataset.obs["size_factors"] = dataset.obs["size_factors"]/dataset.obs["size_factors"].mean()
    dataset.obs["size_factors_input"] = np.sum(dataset.layers["input"], axis=1)
    dataset.obs["size_factors_input"] = dataset.obs["size_factors_input"]/dataset.obs["size_factors_input"].mean()
    mean_lfc = np.zeros(len(dataset))
    ipStrength = np.zeros(len(dataset))
       # Find reference point for fold change over input
    for i in range(len(dataset.X)):
        order1 = np.argsort(dataset.uns["input_random"][i])[::-1]
        order = np.argsort(dataset.uns["counts_random"][i][order1], kind="stable")
        sortedCounts = dataset.uns["counts_random"][i][order1][order]
        sortedInput = dataset.uns["input_random"][i][order1][order]
        sumCounts = np.cumsum(sortedCounts)
        sumInput =  np.cumsum(sortedInput)
        pj = sumCounts/np.sum(sortedCounts)
        qj = sumInput/np.sum(sortedInput)
        amax = np.argmax(qj-pj)
        if sumInput[amax] <= 10 or sumCounts[amax] <= 10:
            # take full sum when the method does not work well
            # and takes the first few value
            # usually it happens on bad samples with poor ip
            warnings.warn(f"Sample {dataset.obs_names[i]} appears to have very low enrichment vs input.")
            mean_lfc[i] = np.log(sumCounts[-1]/sumInput[-1])
        else:
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
    # Create a reference metasample for quantile normalization
    metasample = np.zeros(dataset.shape[1])
    metasampleOrdered = np.zeros(dataset.shape[1])
    for i in range(len(dataset.X)):
        lfc_i = np.log(np.maximum(dataset.X[i],1e-1)/dataset.layers["normalization_factors"][i])
        metasample += np.sort(lfc_i)/len(dataset.X)
        metasampleOrdered += lfc_i/len(dataset.X)
    # Create quantile interpolators for positive and negative lfcs
    metasample = np.exp(metasample)
    positive = metasample >= 1.0
    avgPos = positive.mean()
    avgNeg = (~positive).mean()
    positive_interpolator = sc.interpolate.interp1d(np.linspace(0.0, 1.0, positive.sum()), metasample[positive],
                                                    fill_value="extrapolate")
    negative_interpolator = sc.interpolate.interp1d(np.linspace(0.0, 1.0, (~positive).sum()), metasample[~positive],
                                                    fill_value="extrapolate")
    interpolator = sc.interpolate.interp1d(np.linspace(0.0, 1.0, len(metasample)), metasample,
                                                    fill_value="extrapolate")
    # Re-scale input to correct LFC curve
    plt.figure(dpi=muffin.params["figure_dpi"])
    for i in range(len(dataset.X)):
        fc = np.maximum(dataset.X[i],1e-1)/dataset.layers["normalization_factors"][i]
        positive = fc >= 1.0
        rankPos = sc.stats.rankdata(fc[positive]) / (positive.sum())
        rankNeg = sc.stats.rankdata(fc[~positive]) / ((~positive).sum())
        sfPos = positive_interpolator(rankPos)
        sfNeg = negative_interpolator(rankNeg)
        scale = np.zeros_like(fc)
        scale[positive] = sfPos
        scale[~positive] = sfNeg
        scale = metasample[np.argsort(np.argsort(fc))]
        dataset.layers["normalization_factors"][i] = dataset.layers["normalization_factors"][i]/(scale/fc)
        # ri_scaled = self.matrices["input_random"][i]/(scale/fc)
        lfc = np.log(np.maximum(1e-1, dataset.X[i])/dataset.layers["normalization_factors"][i])
        plt.plot(np.sort(lfc), linewidth=0.8)
    plt.xlim(plt.xlim()[0], plt.xlim()[1])
    plt.hlines(0.0, plt.xlim()[0], plt.xlim()[1])
    if muffin.params["autosave_plots"] is not None:
        plt.savefig(muffin.params["autosave_plots"]+"/norm_input_quantile"+muffin.params["autosave_format"],
                    bbox_inches="tight")
    plt.show()

def rescale_input_center_scale(dataset, plot=True):
    """
    For data with input only !
    Centers and scales fold changes by modifying the input count matrix.
    Novel scaled input is stored in dataset.layers["normalization_factors"]

    Parameters
    ----------
    dataset : AnnData
        Dataset in AnnData format, with input layer filled
    plot : bool, optional
        Whether to plot or not, by default False
    """
    # Scale factors (only used for visualisation)
    dataset.obs["size_factors"] = np.sum(dataset.X, axis=1)
    dataset.obs["size_factors"] = dataset.obs["size_factors"]/dataset.obs["size_factors"].mean()
    dataset.obs["size_factors_input"] = np.sum(dataset.layers["input"], axis=1)
    dataset.obs["size_factors_input"] = dataset.obs["size_factors_input"]/dataset.obs["size_factors_input"].mean()
    mean_lfc = np.zeros(len(dataset))
    ipStrength = np.zeros(len(dataset))
    # Find reference point for fold change over input
    for i in range(len(dataset.X)):
        order1 = np.argsort(dataset.uns["input_random"][i])[::-1]
        order = np.argsort(dataset.uns["counts_random"][i][order1], kind="stable")
        sortedCounts = dataset.uns["counts_random"][i][order1][order]
        sortedInput = dataset.uns["input_random"][i][order1][order]
        sumCounts = np.cumsum(sortedCounts)
        sumInput =  np.cumsum(sortedInput)
        pj = sumCounts/np.sum(sortedCounts)
        qj = sumInput/np.sum(sortedInput)
        amax = np.argmax(qj-pj)
        if sumInput[amax] <= 10 or sumCounts[amax] <= 10:
            # take full sum when the method does not work well
            # and takes the first few value
            # usually it happens on bad samples with poor ip
            warnings.warn(f"Sample {dataset.obs_names[i]} appears to have very low enrichment vs input.")
            mean_lfc[i] = np.log(sumCounts[-1]/sumInput[-1])
        else:
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
    # Raw plot
    if plot:
        plt.figure(dpi=muffin.params["figure_dpi"])
    for i in range(len(dataset.X)):
        lfc = np.log(np.maximum(1e-1,dataset.X[i])/dataset.layers["normalization_factors"][i])
        if plot:
            plt.plot((np.sort(lfc)), linewidth=0.5)
    if plot:
        plt.xlim(plt.xlim()[0], plt.xlim()[1])
        plt.ylabel("Normalized log(Fold Change)")
        plt.xlabel("Log(Fold Change) rank")
        plt.hlines(0.0, plt.xlim()[0], plt.xlim()[1])
        if muffin.params["autosave_plots"] is not None:
            plt.savefig(muffin.params["autosave_plots"]+"/norm_raw"+muffin.params["autosave_format"],
                        bbox_inches="tight")
        plt.show()
    dataset.layers["normalization_factors"] *= np.exp(mean_lfc)[:, None]
    ipStrength = ipStrength/np.median(ipStrength)
    # Center-only plot
    if plot:
        plt.figure(dpi=muffin.params["figure_dpi"])
    for i in range(len(dataset.X)):
        lfc = np.log(np.maximum(1e-1,dataset.X[i])/dataset.layers["normalization_factors"][i])
        if plot:
            plt.plot((np.sort(lfc)), linewidth=0.5)
    if plot:
        plt.xlim(plt.xlim()[0], plt.xlim()[1])
        plt.ylabel("Normalized log(Fold Change)")
        plt.xlabel("Log(Fold Change) rank")
        plt.hlines(0.0, plt.xlim()[0], plt.xlim()[1])
        if muffin.params["autosave_plots"] is not None:
            plt.savefig(muffin.params["autosave_plots"]+"/norm_input_center"+muffin.params["autosave_format"],
                        bbox_inches="tight")
        plt.show()
    # Re-scale input to correct LFC curve
    if plot:
        plt.figure(dpi=muffin.params["figure_dpi"])
    for i in range(len(dataset.X)):
        fc = (np.maximum(dataset.X[i], 1e-1)/dataset.layers["normalization_factors"][i])
        scale_pos = ipStrength[i] * (fc - 1) + 1
        scale_neg = 1.0 / (ipStrength[i] * (1/fc - 1) + 1)
        scale = np.where(fc > 1, scale_pos, scale_neg)
        dataset.layers["normalization_factors"][i] = dataset.layers["normalization_factors"][i]*fc/scale
        lfc = np.log(np.maximum(1e-1,dataset.X[i])/dataset.layers["normalization_factors"][i])
        if plot:
            plt.plot((np.sort(lfc)), linewidth=0.5)
    if plot:
        plt.xlim(plt.xlim()[0], plt.xlim()[1])
        plt.ylabel("Normalized log(Fold Change)")
        plt.xlabel("Log(Fold Change) rank")
        plt.hlines(0.0, plt.xlim()[0], plt.xlim()[1])
        if muffin.params["autosave_plots"] is not None:
            plt.savefig(muffin.params["autosave_plots"]+"/norm_input_center_scale"+muffin.params["autosave_format"],
                        bbox_inches="tight")
        plt.show()

    

def trim_low_counts(dataset, min_exp=3, min_counts=1):
    """
    Returns a boolean array of variables with at least 
    min_counts in min_exp rows. 

    Parameters
    ----------
    dataset : AnnData
        Dataset in AnnData format.
    min_exp : int, optional
        Minimum number of experiment to have at least min_counts, by default 3, at least 1
    min_counts : int, optional
        Minimum number of counts, by default 1

    Returns
    -------
    kept_features: boolean ndarray
        Variables that satisfies above criterions.
    """
    if min_exp < 1 or min_counts < 1:
        raise ValueError(f"min_exp and min_counts have to be 1 or greater.")
    dropped_features = stats.computeDropped(dataset.X, min_exp, min_counts)
    return ~dropped_features
    

def pseudo_peak_calling(dataset, alpha=0.05, minFC=0.5, minExp=2):
    """
    Performs a pseudo-peak calling using the fitted NB model
    and normalized input.
    Returns a boolean array of features with significant 
    enrichment over input in at least minExp rows.

    Parameters
    ----------
    dataset : AnnData
        Dataset in AnnData format.
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
    """
    Returns a boolean array of the features which do not passes the sum of
    squared residuals chisquared goodness of fit test. It is usually too
    stringent.

    Parameters
    ----------
    dataset : AnnData
        Dataset in AnnData format.
    alpha : float, optional
        FDR, by default 0.05

    Returns
    -------
    boolean ndarray
        Boolean array of highly variable features.
    """     
    ssr = np.sum(np.square(dataset.layers["residuals"]),axis=0)
    pval = sc.stats.chi2(dataset.obsm["design"].shape[0]-dataset.obsm["design"].shape[1]).sf(ssr)
    return fdrcorrection(pval, alpha=alpha)[0]

def feature_selection_top_k(dataset, k_features):
    """
    Returns a boolean array of the k features with the 
    largest sum of squared residuals.

    Parameters
    ----------
    dataset : AnnData
        Dataset in AnnData format.
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

def feature_selection_elbow(dataset, plot:bool=True, subsample=20000):
    """
    Performs feature selection by finding a conservative ankle point of the
    ranked standard deviation of residuals curve. We find the ankle point of a third
    degree polynomial fit, which yields a quite conservative feature selection, but
    typically removes at least half of the features.

    Parameters
    ----------
    dataset : AnnData
        Dataset in AnnData format.
    plot : bool, optional
        Whether to plot the ranked SSR curve with the ankle point,
        by default True

    Returns
    -------
    boolean ndarray
        Boolean array of highly variable features.
    """
    sds = np.sum(np.square(dataset.layers["residuals"]),axis=0)
    subsample = min(subsample, len(sds))
    subsample_ssr = np.sort(sds)[np.linspace(0, len(sds)-1, subsample, dtype=int)]
    elbow_locator = kneed.KneeLocator(np.arange(subsample), subsample_ssr, 
                                        curve="convex", interp_method="polynomial",
                                        online=True, polynomial_degree=3)
    min_ssr = elbow_locator.elbow_y
    if plot:
        plt.figure(dpi=muffin.params["figure_dpi"])
        elbow_locator.plot_knee()
        p = np.polyfit(np.arange(subsample), subsample_ssr, 3)
        plt.plot(np.arange(subsample), np.polyval(p, np.arange(subsample)))
        plt.ylabel("Sum of squared residuals")
        plt.xlabel("Feature rank")
        if muffin.params["autosave_plots"] is not None:
            plt.savefig(muffin.params["autosave_plots"]+"/feature_selection_elbow"+muffin.params["autosave_format"],
                        bbox_inches="tight")
        plt.show()
        # Plot mean/variance relationship and selected probes
        v = dataset.var["variances"]
        m = dataset.var["means"]
        plt.figure(dpi=muffin.params["figure_dpi"])
        w = sds >= min_ssr
        plt.scatter(m, v, s = 2.0*(100000/len(m)), linewidths=0, c=w, alpha=1.0, rasterized=True)
        plt.scatter(m, m+m*m*dataset.var["reg_alpha"], s = 1.0, linewidths=0, c=[0.0,1.0,0.0], rasterized=True)
        pts = np.geomspace(max(m.min(), 1.0/len(dataset.obs["size_factors"])), m.max())
        plt.plot(pts, pts, rasterized=True)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Feature mean")
        plt.ylabel("Feature variance")
        if muffin.params["autosave_plots"] is not None:
            plt.savefig(muffin.params["autosave_plots"]+"/feature_selection_elbow_mv_plot"+muffin.params["autosave_format"],
                        bbox_inches="tight")
        plt.show()
    return sds >= min_ssr

def compute_size_factors(dataset, method="top_fpkm"):
    """
    Compute size factors.

    Parameters
    ----------
    dataset : AnnData
        Dataset in AnnData format.
    method : str, optional
        Method to use, by default "top_fpkm". Available methods:

        - "top fpkm" : Selects top 5% most detectable variables and 
                       computes sum of counts normalization.
        - "fpkm" : Computes sum of counts normalization.
        - "scran" : Selects top 5% most detectable variables and 
                    computes scran pooling and deconvolution normalization.
        - "deseq" : Applies median of ratios, works well with deeply sequenced
                    datasets. Will raise an error if not suitable.
        - "fpkm_uq" : Computes Upper Quartile normalization.

    Returns
    -------
    dataset : AnnData
    """    
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

def compute_residuals(dataset, residuals="quantile", clip=np.inf, subSampleEst=2000, maxThreads=-1, verbose=True, plot=True):
    """
    Compute residuals from a regularized NB model for each variable.

    Parameters
    ----------
    residuals : str, optional
        Whether to compute "anscombe", "deviance", "quantile", "rqr" or "pearson" residuals, by
        default "anscombe"
    clip : float, optional
        Value to clip residuals to (+-value), you can also provide "auto" to
        clip at +-sqrt(9+len(dataset)/4), default np.inf
    subSampleEst : int, optional
        Number of samples to use to estimate mean-overdispersion relationship.
    maxThreads : int, optional
        Number of threads to use, by default -1 (all)
    verbose : bool, optional
        Verbosity, by default True
    plot : bool, optional
        Whether to plot the mean-variance graph, by default True

    Returns
    -------
    dataset : AnnData
    """
    np.random.seed(42)
    if "normalization_factors" not in dataset.layers.keys():
        dataset.var["means"] = stats.computeMeans(dataset.X, 
                                                  dataset.obs["size_factors"].values)
    else:
        dataset.var["means"] = stats.computeMeanNormFactors(dataset.X, 
                                                            dataset.layers["normalization_factors"])
    if "normalization_factors" not in dataset.layers.keys():
        dataset.var["variances"] = stats.computeVar(dataset.X, 
                                                    dataset.obs["size_factors"].values)
    else:
        dataset.var["variances"] = stats.computeVarNormFactors(dataset.X, 
                                                            dataset.layers["normalization_factors"])
    # Estimate regularized variance in function of mean expression
    nLowess = min(subSampleEst, dataset.X.shape[1])
    indices = np.linspace(0, dataset.X.shape[1]-1, nLowess, dtype=int)
    meanOrder = np.argsort(dataset.var["means"].values)
    subset = meanOrder[indices]
    alphas = np.zeros(nLowess)
    if "input" in dataset.layers.keys():
        if not "normalization_factors" in dataset.layers.keys():
            raise Exception("Input counts have NOT been scaled!")
        with Parallel(n_jobs=maxThreads, verbose=verbose, batch_size=512, max_nbytes=None) as pool:
            alphas = pool(delayed(stats.fit_alpha_input)(dataset.X[:, subset[i]], 
                                                                dataset.obsm["design"], 
                                                                dataset.layers["normalization_factors"][:, subset[i]]) for i in range(nLowess))
    else:
        with Parallel(n_jobs=maxThreads, verbose=verbose, batch_size=512, max_nbytes=None) as pool:
            alphas = pool(delayed(stats.fit_alpha)(dataset.obs["size_factors"], 
                                                          dataset.X[:, subset[i]], 
                                                          dataset.obsm["design"]) for i in range(nLowess))
    # Kill workers or they keep being active even if the program is shut down
    get_reusable_executor().shutdown(wait=False, kill_workers=True)
    alphas = np.array(alphas)
    validAlphas = (alphas > 1e-3) & (alphas < 1e3)
    # Take rolling median
    regAlpha = np.array(pd.Series(alphas[validAlphas]).rolling(200, center=True, min_periods=1).median())
    # Interpolate between values with lowess
    dataset.var["reg_alpha"] = np.exp(lowess(np.log(regAlpha), indices[validAlphas], frac=0.05, xvals=np.argsort(meanOrder)))
    # Dispatch accross multiple processes
    sf = dataset.obs["size_factors"].values
    design = dataset.obsm["design"]
    regAlpha = dataset.var["reg_alpha"].values
    with Parallel(n_jobs=maxThreads, verbose=verbose, batch_size=512) as pool:
        if "normalization_factors" not in dataset.layers.keys():
            residuals = pool(delayed(stats.compute_residuals)(regAlpha[i],
                                                              sf, 
                                                              dataset.X[:, i], 
                                                              design, 
                                                              residuals) for i in range(dataset.X.shape[1]))
        else:
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
    
    # Standardize to unit norm
    ref_std = np.std(dataset.layers["residuals"])
    dataset.layers["residuals"] /= np.linalg.norm(dataset.layers["residuals"], axis=1)[:, None]
    dataset.layers["residuals"] /= np.std(dataset.layers["residuals"]) / ref_std
    if plot:
        # Plot mean/variance relationship and selected probes
        v = dataset.var["variances"]
        m = dataset.var["means"]
        plt.figure(dpi=muffin.params["figure_dpi"])
        w = np.std(dataset.layers["residuals"], axis=0)
        w = w / w.max()
        plt.scatter(m, v, s = 2.0*(100000/len(m)), linewidths=0, c=w, alpha=1.0, rasterized=True)
        plt.scatter(m, m+m*m*dataset.var["reg_alpha"], s = 1.0, linewidths=0, c=[0.0,1.0,0.0], rasterized=True)
        pts = np.geomspace(max(m.min(), 1.0/len(dataset.obs["size_factors"])), m.max())
        plt.plot(pts, pts, rasterized=True)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Feature mean")
        plt.ylabel("Feature variance")
        if muffin.params["autosave_plots"] is not None:
            plt.savefig(muffin.params["autosave_plots"]+"/mv_trendline"+muffin.params["autosave_format"],
                        bbox_inches="tight")
        plt.show()
        plt.close()
    return dataset
    
def compute_pa_pca(dataset, layer="residuals", feature_mask=None, perm=3, alpha=0.01, 
                    solver="arpack", whiten=False,
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
        Number of permutations. On large matrices (row * col > 100,000) the eigenvalues are 
        very stable, so there is no need to use a large number of permutations,
        and only one permutation can be a reasonable choice on large matrices.
        On smaller matrices the number of permutations should be increased.

    alpha: float (default 0.01)
        Permutation p-value threshold.
    
    solver: "arpack" or "randomized"
        Chooses the SVD solver. Randomized is much faster but less accurate and yields
        different results machine-to-machine.
    
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
    dataset.obsm["X_pca"], model, dstat_obs, dstat_null, pvals = FA_models.permutationPA_PCA(data, perm=perm, 
                                                            alpha=alpha, solver=solver, whiten=whiten,
                                                            max_rank=max_rank, mincomp=mincomp)
    if feature_mask is not None:
        dataset.varm["PCs"] = np.zeros((dataset.obsm["X_pca"].shape[1], dataset.X.shape[1])).T
        dataset.varm["PCs"][feature_mask] = model.components_[:dataset.obsm["X_pca"].shape[1]].T
    else:
        dataset.varm["PCs"] = model.components_[:dataset.obsm["X_pca"].shape[1]].T
    dataset.uns["pca"] = dict()
    dataset.uns["pca"]['variance_ratio'] = model.explained_variance_ratio_[:dataset.obsm["X_pca"].shape[1]]
    dataset.uns['pca']['variance'] = model.explained_variance_[:dataset.obsm["X_pca"].shape[1]]
    r_est = dataset.obsm["X_pca"].shape[1]
    if plot:
        plt.figure(dpi=muffin.params["figure_dpi"])
        plt.plot(np.arange(len(dstat_obs))+1, dstat_obs)
        for i in range(len(dstat_null)):
            plt.plot(np.arange(len(dstat_obs))+1, dstat_null[i], linewidth=0.2)
        plt.xlabel("PCA rank")
        plt.ylabel("Explained variance (log scale)")
        plt.yscale("log")
        plt.legend(["Observed eigenvalues", "Randomized eigenvalues"])
        plt.xlim(1,r_est*1.2)
        plt.ylim(np.min(dstat_null[:, :int(r_est*1.2)])*0.95, dstat_obs.max()*1.05)
        if muffin.params["autosave_plots"] is not None:
            plt.savefig(muffin.params["autosave_plots"]+"/pca_num_comps"+muffin.params["autosave_format"],
                        bbox_inches="tight")
        plt.show()

def compute_umap(dataset, on="reduced_dims", which="X_pca", feature_mask=None, umap_params={}):
    """
    Compute UMAP and stores it under dataset.reduced_dims["X_umap"].

    Parameters
    ----------
    on : str, optional
        On which data representation to perform umap. Use "reduced_dims"
        or "features", by default "reduced_dims".
    which : str, optional
        Which reduced_dims or feature to use. I.e. "PCA" or "residuals", by
        default "PCA".
    feature_mask : boolean ndarray, optional
        Subset of features to use for Umap (works with PCA as well), by default
        None
    umap_params : dict, optional
        Dictionnary of keyword arguments for UMAP, see UMAP documentation, by
        default if not provided metric is set to euclidean. Random state is also
        fixed and should not be provided.

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
                    metric="auto", k="auto", r=1.0, restarts=10):  
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
        Metric to use for kNN search, by default "auto". If set to auto
        Pearson correlation is used as the metric when there are more than 
        10 input dimensions; otherwise, the Euclidean distance is used.
    k : "auto" or int, optional
        Number of nearest neighbors to find, 
        by default "auto" uses 4*nFeatures^0.2 as a rule of thumb.
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
    if metric == "auto":
        if data.shape[1] > 10:
            metric = "correlation"
        else:
            metric = "euclidean"
    dataset.obs["leiden"] = cluster.graphClustering(data, metric=metric, 
                                                         k=k, r=r, restarts=restarts).astype(str)
    return dataset

def differential_expression_A_vs_B(dataset, category, ref_category, alternative="two-sided",
                                   method="auto"):
    """
    Performs differential expression between two categories.
    Results will be stored in dataset.varm["DE_results"],
    and for parity with scanpy, in dataset.uns["rank_genes_groups"].

    Parameters
    ----------
    dataset : AnnData
        Dataset in AnnData format.
    category : str
        Name of the obs column to use for grouping. If more than two unique values
        are present, will raise an error.
    ref_category : str
        Name of the reference category used for log fold change computations.
    method : str, optional
        Method to be used for differential expression, by default "auto".
        Available methods:
        - "auto" : Uses deseq if dataset size > 50, t-test otherwise.
        - "deseq": Uses DESeq2 with the supplied design matrix beforehand.
        - "t-test": Performs welsh t-test on residuals.
        - "wilcoxon": Performs mann-whitney u-test on residuals (prefer using t-test).
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
    # Same format as scanpy
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



def cca(dataset1, dataset2, n_comps=30, layer="residuals"):
    """
    Perform Canonical Correspondance Analysis as in seurat.
    Results will be stored in the .obsm["X_cca"] slot of each dataset.
    
    Parameters
    ----------
    dataset1 : anndata
        First dataset to integrate (order does not matter).
    dataset2 : _type_
        Second dataset to integrate (order does not matter).
    n_comps : int, optional
        Number of CCA components to use, by default 30.
    layer : str, optional
        .layers key present in both datasets, by default "residuals".
    """    
    var_names = dataset1.var_names.intersection(dataset2.var_names).unique()
    if len(var_names) < dataset1.shape[1]:
        warnings.warn(f"Warning, not all variables are shared between both datasets, {len(var_names)} in common left")
    Zx, Zy = FA_models.seurat_cca(dataset1[:, var_names].layers[layer],
                                  dataset2[:, var_names].layers[layer],
                                  n_comps)
    dataset1.obsm["X_cca"] = Zx
    dataset2.obsm["X_cca"] = Zy



def transfer_categorical_labels(dataset_ref, dataset_target, label, 
                                representation_common, representation_target="X_pca",
                                k_smoothing="auto", metric="auto"):
    """
    Transfer a categorical label from observations of dataset_ref to
    dataset_target. e.g clustering info from scRNA-seq to scATAC. 
    Uses a RF classifier fitted on reference dataset to predict labels on
    the target dataset. Predictions are then smoothed using kNN, and 
    untransferrable points are detected using Local Outlier Factor.

    Parameters
    ----------
    dataset_ref : anndata
        Dataset with reference labels
    dataset_target : anndata
        Dataset to which predict labels.
    label : str
        Column name in dataset_ref.obs
    representation_common : str
        .obsm key present in both datasets. e.g. joint embedding produced by
        cca, then batch corrected by a tool such as harmony.
    representation_target : str, optional
        .obsm key present in dataset_target, that will be used to smooth
        transferred labels to nearest neighbors, by default "X_pca"
    k_smoothing : "auto" or int
        Number of nearest neighbors used for label smoothing.
    """
    predictor_1 = RandomForestClassifier(criterion="log_loss", random_state=42)
    labelencoder = LabelEncoder()
    int_cat = labelencoder.fit_transform(dataset_ref.obs[label])
    predictor_1.fit(dataset_ref.obsm[representation_common], int_cat)
    predicted = predictor_1.predict(dataset_target.obsm[representation_common])

    if metric == "auto":
        if dataset_ref.obsm[representation_common].shape[1] > 10:
            metric = "correlation"
        else:
            metric = "euclidean"

    if k_smoothing == "auto":
        k_smoothing = int(np.power(dataset_target.shape[0], 0.2)*4)
    predicted = cluster.approx_knn_predictor(dataset_target.obsm[representation_target], predicted, metric, 
                                             k_smoothing)
    
    dataset_target.obs[label + "_transferred"] = labelencoder.inverse_transform(predicted)
    
    outlierdetector = LocalOutlierFactor(n_neighbors=50, metric=metric, novelty=True)
    outlierdetector.fit(dataset_ref.obsm[representation_common])
    outliers = outlierdetector.predict(dataset_target.obsm[representation_common]) < 0.0
    dataset_target.obs.loc[outliers,label + "_transferred"] = "Untransferrable"
    