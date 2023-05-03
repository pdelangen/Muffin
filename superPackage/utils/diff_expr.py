import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, norm
from statsmodels.stats.multitest import fdrcorrection
import warnings


def DESeq2(counts, sf, labels, to_regress):
    # Import here for people that do not want R
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
    deseq = importr("DESeq2")
    countTable = pd.DataFrame(counts.T, columns=np.arange(len(counts)).astype(str))
    # Detect constant columns and remove them, as deseq automatically adds an intercept
    non_intercepts = np.std(to_regress/(1e-8+to_regress.mean(axis=0)), axis=0) > 1e-9
    infos = pd.DataFrame(labels, index=np.arange(len(counts)).astype(str), columns=["Category"])
    infos["sizeFactor"] = sf.ravel()
    if np.sum(non_intercepts) > 0:
        names = ["V"+str(i) for i in range(non_intercepts.sum())]
        infos[names] = to_regress[:, non_intercepts]
        formula = "~1+Category+" + "+".join(["V"+str(i) for i in range(non_intercepts.sum())])
    else:
        formula = "~1+Category"
    print(formula)
    print(infos)
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        dds = deseq.DESeqDataSetFromMatrix(countData=countTable, colData=infos, design=ro.Formula(formula))
        deseq.DESeq(dds, test="wald")
        print(deseq.resultsNames(dds))
        try:
            res = deseq.lfcShrink(dds, coef=2)
        except:
            res = deseq.results(dds)
            warnings.warn("LFC shrinkage failed!")
    res = pd.DataFrame(res.slots["listData"], index=res.slots["listData"].names).T
    res["padj"] = np.nan_to_num(res["padj"], nan=1.0)
    # Add pseudo z-score for consistency with scanpy
    res["z-score"] = np.abs(norm().isf(res["pvalue"].values*0.5+1e-300)) * np.sign(res["log2FoldChange"].values)
    return res


def DESeq2_input(counts, input_mat, labels, to_regress):
    # Import here for people that do not want R
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
    deseq = importr("DESeq2")
    countTable = pd.DataFrame(counts.T, columns=np.arange(len(counts)).astype(str))
    # Detect constant columns and remove them, as deseq automatically adds an intercept
    non_intercepts = np.std(to_regress/(1e-8+to_regress.mean(axis=0)), axis=0) > 1e-9
    infos = pd.DataFrame(labels, index=np.arange(len(counts)).astype(str), columns=["Category"])
    if np.sum(non_intercepts) > 0:
        names = ["V"+str(i) for i in range(non_intercepts.sum())]
        infos[names] = to_regress[:, non_intercepts]
        formula = "~1+Category+" + "+".join(["V"+str(i) for i in range(non_intercepts.sum())])
    else:
        formula = "~1+Category"
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        dds = deseq.DESeqDataSetFromMatrix(countData=countTable, colData=infos, design=ro.Formula(formula))
    a = input_mat.T
    r_matrix = ro.r['matrix'](ro.FloatVector(a.flatten()), nrow=a.shape[0], ncol=a.shape[1], byrow=True)
    rfunc = ro.r('function(dds, a) {normalizationFactors(dds)<-a; return(dds)}')
    dds = rfunc(dds, r_matrix)
    dds = deseq.DESeq(dds)
    try:
        res = deseq.lfcShrink(dds, coef=2)
    except:
        res = deseq.results(dds)
        warnings.warn("LFC shrinkage failed!")
    res = pd.DataFrame(res.slots["listData"], index=res.slots["listData"].names).T
    res["padj"] = np.nan_to_num(res["padj"], nan=1.0)
    # Add pseudo z-score for consistency with scanpy
    res["z-score"] = np.abs(norm().isf(np.clip(res["pvalue"].values*0.5,1e-300,1.0))) * np.sign(res["log2FoldChange"].values)
    return res


def t_test(values, raw, sf, groups):
    categories = np.unique(groups)
    stats, pvals = ttest_ind(values[groups == categories[1]], 
                             values[groups == categories[0]],
                             equal_var=False)
    avgA = np.apply_along_axis(lambda a: np.mean(a/sf[groups == categories[0]]), 
                               axis=0, arr=raw[groups == categories[0]])
    avgB = np.apply_along_axis(lambda a: np.mean(a/sf[groups == categories[1]]), 
                               axis=0, arr=raw[groups == categories[1]])
    lfcs = np.log2(np.maximum(1/len(sf), avgB) / np.maximum(1/len(sf), avgA))
    padj = fdrcorrection(pvals)[1]
    res = pd.DataFrame(np.array([stats, lfcs, pvals, padj]).T, 
                       columns=["z-score", "log2FoldChange", "pvalue", "padj"])
    return res


def wilcoxon(values, raw, sf, groups):
    categories = np.unique(groups)
    stats, pvals = mannwhitneyu(values[groups == categories[0]], 
                             values[groups == categories[1]],
                             equal_var=False)
    avgA = np.apply_along_axis(lambda a: np.mean(a*sf[groups == categories[1]]), 
                               axis=0, arr=raw[groups == categories[0]])
    avgB = np.apply_along_axis(lambda a: np.mean(a*sf[groups == categories[0]]), 
                               axis=0, arr=raw[groups == categories[1]])
    lfcs = np.log2(np.maximum(1/len(sf), avgB) / np.maximum(1/len(sf), avgA))
    padj = fdrcorrection(pvals)[1]
    res = pd.DataFrame(np.array([stats, lfcs, pvals, padj]).T, 
                       columns=["z-score", "log2FoldChange", "pvalue", "padj"])
    return res