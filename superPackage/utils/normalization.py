import numpy as np
from scipy.stats import gmean
import warnings


def top_features(counts, pct=0.05):
    detected = [np.sum(counts > i, axis=0) for i in range(5)][::-1]
    mostDetected = np.lexsort(detected)[::-1]
    # Avoid features with extremely large expression
    means = counts.mean(axis=0)[mostDetected]
    outliers = means > (np.percentile(means, 95)*2)
    return mostDetected[~outliers][:int(counts.shape[1]*pct+1)]

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

def median_of_ratios(counts):
    avg = gmean(counts, axis=0)
    ratios = counts / avg
    pctNan = np.mean(ratios < 1e-9) + np.mean(avg < 1e-3)
    if pctNan > 0.25:
        warnings.warn(f"Median of ratios detected > 25% of zeros features ({pctNan*100}), consider using another approach")
    return np.nanmedian(ratios, axis=1)

def top_detected_sum_norm(counts):
    mostDetected = top_features(counts)
    return np.sum(counts[:, mostDetected], axis=1)

def sum_norm(counts):
    return np.sum(counts, axis=1)

def fpkm_uq(counts):
    return np.percentile(counts, 75, axis=1)