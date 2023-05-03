import numpy as np
from statsmodels.genmod.families.family import NegativeBinomial
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.api import NegativeBinomial as nbfit
import numba as nb
import warnings
import pandas as pd


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


def computeDropped(counts, means, min_exp, min_counts, min_mean):
    toDrop = np.zeros(counts.shape[1], dtype="bool")
    for i in range(len(toDrop)):
        toDrop[i] = (means[i] <= min_mean) | ((counts[:, i] >= min_counts).sum() < min_exp)
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

def compute_residuals(alpha, exposure, counts, design, res_type):
    alpha = np.clip(alpha, 1e-5, 1e5)
    distrib = NegativeBinomial(alpha=alpha)
    model = GLM(counts, design, family=distrib, exposure=exposure).fit(
                full_output=False, tol=0.0)
    predicted = model.predict()
    if res_type == "deviance":
        residuals = distrib.resid_dev(counts, predicted)
    else:
        residuals = (counts - predicted) / np.sqrt(predicted + alpha * predicted**2)
        residuals = np.clip(residuals, -np.sqrt(9+len(exposure)/4), np.sqrt(9+len(exposure)/4))
    return residuals.astype("float32") 

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
    else:
        residuals = (counts - predicted) / np.sqrt(predicted + alpha * predicted**2)
        residuals = np.clip(residuals, -np.sqrt(9+len(counts)/4), np.sqrt(9+len(counts)/4))
    return residuals.astype("float32") 

# Used by GSEA
def fitNBinomModel(hasAnnot, observed, expected, goTerm, idx):
    df = pd.DataFrame(np.array([hasAnnot.T.astype(float), np.ones_like(expected)]).T, 
                                columns=["GS", "Intercept"], index=idx)
    model = nbfit(observed, df, exposure=expected, loglike_method="nb1")
    model = model.fit([0.0,0.0,10.0], method="lbfgs", disp=False)
    beta = model.params["GS"]
    waldP = model.pvalues["GS"]
    # Get one sided pvalues
    if beta >= 0:
        pvals = waldP/2.0
    else:
        pvals = (1.0-waldP/2.0)
    return (goTerm, pvals, beta)
