import numpy as np
from scipy.stats import gmean
import warnings
import numba as nb

def top_features(counts, pct=0.05):
    detected = [np.sum(counts > i, axis=0) for i in range(0,25,5)][::-1]
    mostDetected = np.lexsort(detected)[::-1]
    return mostDetected[:int(counts.shape[1]*pct+1)]

def scran_norm(counts):
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
    scran = importr("scran")
    mostDetected = top_features(counts)
    with localconverter(ro.default_converter + numpy2ri.converter):
        sf = scran.calculateSumFactors(counts.T[mostDetected])
    return sf

@nb.njit()
def mor(counts, avg):
    sf = np.zeros(len(counts))
    for i in range(len(counts)):           
        sf[i] = np.nanmedian(counts[i][avg.nonzero()]/avg[avg.nonzero()])
    return sf

def median_of_ratios(counts):
    avg = gmean(counts, axis=0)
    sf = mor(counts, avg)
    if np.any(sf == 0.0):
        raise ValueError("One observation returned a 0 size factor (caused by too many zeroes), \
                         consider using another normalization approach.")
    return sf

def top_detected_sum_norm(counts):
    mostDetected = top_features(counts)
    return np.sum(counts[:, mostDetected], axis=1)

def sum_norm(counts):
    return np.sum(counts, axis=1)

def fpkm_uq(counts):
    return np.percentile(counts, 75, axis=1)