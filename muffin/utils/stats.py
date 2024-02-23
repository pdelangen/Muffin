import numpy as np
from statsmodels.genmod.families.family import NegativeBinomial, Poisson
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.api import NegativeBinomial as nbfit
from statsmodels.api import Poisson
import statsmodels.api as sm
from scipy.stats import nbinom, norm
import numba as nb
import warnings
import pandas as pd

@nb.njit()
def add_segments(array, segments):
    for start, end in segments:
        array[start:end] += 1
    return array

def median_absolute_deviation(data):
    """
    Computes the median absolute deviation of a dataset.

    Parameters:
    -----------
    data : list, numpy array
        The input data.

    Returns:
    --------
    mad : float
        The median absolute deviation of the input data.
    """
    import numpy as np
    
    # Compute median
    median = np.median(data)

    # Compute absolute deviations from median
    abs_deviations = np.abs(data - median)

    # Compute median of absolute deviations
    mad = np.median(abs_deviations)

    return mad

def normAndMean(counts, sf):
    return (counts/sf).mean()

def normAndVar(counts, sf):
    return (counts/sf).var()

@nb.njit()
def computeMeans(counts, sf):
    means = np.zeros(counts.shape[1], dtype="float32")
    for i in range(counts.shape[1]):
        means[i] = np.mean(counts[:, i] / sf)
    return means


@nb.njit()
def computeVar(counts, sf):
    vars = np.zeros(counts.shape[1], dtype="float32")
    for i in range(counts.shape[1]):
        vars[i] = np.var(counts[:, i] / sf)
    return vars

@nb.njit()
def computeMeanNormFactors(counts, nf):
    means = np.zeros(counts.shape[1], dtype="float32")
    s = np.mean(nf)
    for i in range(counts.shape[1]):
        means[i] = np.mean(counts[:, i] / (nf[:, i] / s))
    return means

@nb.njit()
def computeVarNormFactors(counts, nf):
    vars = np.zeros(counts.shape[1], dtype="float32")
    s = np.mean(nf)
    for i in range(counts.shape[1]):
        vars[i] = np.var(counts[:, i] / (nf[:, i] / s))
    return vars

def computeDropped(counts, min_exp, min_counts):
    toDrop = np.zeros(counts.shape[1], dtype="bool")
    for i in range(len(toDrop)):
        toDrop[i] = (counts[:, i] >= min_counts).sum() < min_exp
    return toDrop

def fit_alpha(exposure, counts, design):
    warnings.filterwarnings("error")
    init = [np.log(np.mean(counts/exposure)/design.shape[1])]*design.shape[1] + [10.0]
    try:
        model = nbfit(counts, design, exposure=exposure).fit(init, method="nm", 
                                                            ftol=1e-9, 
                                                            maxiter=500, 
                                                            disp=False, 
                                                            skip_hessian=True)
    except:
        return -1
    warnings.filterwarnings("default")
    return model.params[-1]

def fit_alpha_input(counts, design, input_counts):
    warnings.filterwarnings("error") 
    init = [np.log(np.mean(counts/input_counts)/design.shape[1])]*design.shape[1] + [10.0]
    try:
        model = nbfit(counts, design, exposure=input_counts).fit(init, method="nm", 
                                                         ftol=1e-9, 
                                                         maxiter=500, 
                                                         disp=False, 
                                                         skip_hessian=True)
    except:
        return -1
    warnings.filterwarnings("default")
    return model.params[-1]

def nb_rqr(x, m, alpha):
    # Randomized Quantile Residuals
    n = 1/alpha
    p = m / (m + alpha * (m**2))
    q = nbinom(n,p).sf(x-1) - np.random.random(x.shape) * nbinom(n,p).pmf(x)
    # Clip for numerical stability
    q = np.clip(q, 1e-323, 1-6e-17)
    return norm.isf(q)

def nb_mqr(x, m, alpha):
    # Middle point Quantile residuals
    n = 1/alpha
    p = m / (m + alpha * (m**2))
    q = nbinom(n,p).sf(x-1) - 0.5 * nbinom(n,p).pmf(x)
    # Clip for numerical stability
    q = np.clip(q, 1e-323, 1-6e-17)
    return norm.isf(q)

def compute_residuals(alpha, exposure, counts, design, res_type):
    alpha = np.clip(alpha, 1e-5, 1e5)
    distrib = NegativeBinomial(alpha=alpha)
    model = GLM(counts, design, family=distrib, exposure=exposure).fit(
                full_output=False, tol=0.0)
    predicted = model.predict()
    if res_type == "deviance":
        residuals = distrib.resid_dev(counts, predicted)
    elif res_type == "anscombe":
        residuals = distrib.resid_anscombe(counts, predicted)
    elif res_type == "quantile":
        residuals = nb_mqr(counts, predicted, alpha)
    elif res_type == "rqr":
        residuals = nb_rqr(counts, predicted, alpha)
    else:
        residuals = (counts - predicted) / np.sqrt(predicted + alpha * predicted**2)
    return (residuals - residuals.mean()).astype("float32") 

def compute_residuals_input(alpha, counts, design, res_type, 
                            input_counts):
    alpha = np.clip(alpha, 1e-5, 1e5)
    distrib = NegativeBinomial(alpha=alpha)
    try:
        model = GLM(counts, design, family=distrib, exposure=input_counts).fit(
                    full_output=False, tol=0.0)
    except:
        print("Regression error")
        return np.zeros_like(counts).astype("float32")
    predicted = model.predict()
    if res_type == "deviance":
        residuals = distrib.resid_dev(counts, predicted)
    elif res_type == "anscombe":
        residuals = distrib.resid_anscombe(counts, predicted)
    elif res_type == "quantile":
        residuals = nb_mqr(counts, predicted, alpha)
    elif res_type == "rqr":
        residuals = nb_rqr(counts, predicted, alpha)
    else:
        residuals = (counts - predicted) / np.sqrt(predicted + alpha * predicted**2)
    return residuals.astype("float32") 

# Used by GSEA
def fitNBinomModel(hasAnnot, observed, expected, goTerm, idx):
    df = pd.DataFrame(np.array([hasAnnot.T.astype(float), np.ones_like(expected)]).T, 
                                columns=["GS", "Intercept"], index=idx)
    warnings.filterwarnings("ignore") 
    model = nbfit(observed, df, exposure=expected)
    model = model.fit([0.0,0.0,10.0], disp=False)
    warnings.filterwarnings("default") 
    beta = model.params["GS"]
    waldP = model.pvalues["GS"]
    # Get one sided pvalues
    if beta >= 0:
        pvals = waldP/2.0
    else:
        pvals = 1.0-waldP/2.0
    return (goTerm, pvals, beta)

def fitPoissonModel(hasAnnot, observed, expected, goTerm, idx):
    df = pd.DataFrame(np.array([hasAnnot.T.astype(float), np.ones_like(expected)]).T, 
                                columns=["GS", "Intercept"], index=idx)
    warnings.filterwarnings("ignore") 
    model = Poisson(observed, df, exposure=expected)
    model = model.fit(disp=False)
    warnings.filterwarnings("default") 
    beta = model.params["GS"]
    waldP = model.pvalues["GS"]
    # Get one sided pvalues
    if beta >= 0:
        pvals = waldP/2.0
    else:
        pvals = 1.0-waldP/2.0
    return (goTerm, pvals, beta)

def fitBinomialModel(hasAnnot, observed, expected, goTerm, idx):
    y = np.array([observed.values.ravel(), expected.values-observed.values.ravel()]).T
    df = pd.DataFrame(np.array([hasAnnot.T.astype(float), np.ones_like(expected)]).T, 
                                columns=["GS", "Intercept"], index=idx)
    # Fitting the binomial GLM
    warnings.filterwarnings("ignore")
    model = sm.GLM(y, df, family=sm.families.Binomial())
    results = model.fit()
    warnings.filterwarnings("default")
    
    # Extracting the coefficient (beta) for 'hasAnnot' and its p-value
    beta = results.params["GS"]
    waldP = results.pvalues["GS"]
    # Adjusting the p-value based on the direction of the effect (beta)
    if beta >= 0:
        pval_adjusted = waldP / 2.0
    else:
        pval_adjusted = 1.0 - waldP / 2.0
    
    return (goTerm, pval_adjusted, beta)

def gauss_kernel(distances, sigma=1.0):
    return np.exp(-distances*sigma)
