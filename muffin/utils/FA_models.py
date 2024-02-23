import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.spatial import distance

def permutationPA_PCA(X, perm=3, alpha=0.01, solver="randomized", whiten=False,
                      max_rank=None, mincomp=2,):
    """
    Permutation Parallel Analysis to find the optimal number of PCA components.

    Parameters
    ----------
    X: ndarray
        obs, features matrix to compute PCA on

    perm: int (default 3)
        Number of permutations. On large matrices the eigenvalues are 
        very stable, so there is no need to use a large number of permutations,
        and only one permutation can be a reasonable choice on large matrices.
        On smaller matrices the number of permutations should be increased.

    alpha: float (default 0.01)
        Permutation p-value threshold.
    
    solver: "arpack" or "randomized"
        Chooses the SVD solver. Randomized is faster but less accurate.
    
    whiten: bool (default True)
        If set to true, each component is transformed to have unit variance.

    max_rank: int or None (default None)
        Maximum number of principal components to compute. Must be strictly less 
        than the minimum of n_features and n_samples. If set to None, computes
        up to min(n_samples, n_features)-1 components.
    
    mincomp: int (default 0)
        Number of components to return

    returnModel: bool (default None)
        Whether to return the fitted PCA model or not. (The full model computed up to max_rank)

    Returns
    -------
    decomp: ndarray of shape (n obs, k components)
        PCA decomposition with optimal number of components

    model: sklearn PCA object
        Returned only if returnModel is set to true
    """
    # Compute eigenvalues of observed data
    ref = PCA(max_rank, whiten=whiten, svd_solver=solver, random_state=42)
    decompRef = ref.fit_transform(X)
    dstat_obs = ref.explained_variance_
    # Compute permutation eigenvalues
    dstat_null = np.zeros((perm, len(dstat_obs)))
    np.random.seed(42)
    for b in range(perm):
        X_perm = np.apply_along_axis(np.random.permutation, 0, X)
        perm = PCA(max_rank, whiten=whiten, svd_solver=solver, random_state=42)
        perm.fit(X_perm)
        dstat_null[b, :] = perm.explained_variance_

    # Compute p values
    pvals = np.ones(len(dstat_obs))
    delta = np.zeros(len(dstat_obs))
    for i in range(len(dstat_obs)):
        pvals[i] = norm(loc=np.mean(dstat_null[:, i]), scale=np.std(dstat_null[:, i])).sf(dstat_obs[i])
    for i in range(1, len(dstat_obs)):
        pvals[i] = 1.0-(1.0-pvals[i - 1])*(1.0-pvals[i])
    # estimate rank
    r_est = max(sum(pvals <= alpha),mincomp)
    if r_est == max_rank:
        print("""WARNING, estimated number of components is equal to maximal number of computed components !\n Try to rerun with higher max_rank.""")
    return decompRef[:, :r_est], ref, dstat_obs, dstat_null, pvals


def seurat_cca(x, y, comps=None):
    """Performs CCA similarly to what is described in the seurat paper.

    Parameters
    ----------
    x : ndarray
        samples, variables matrix
    y : ndarray
        samples, variables matrix
    comps : int
        number of components to use

    Returns
    -------
    Zx, Zy : float ndarray
        Reduced dimensionnality representation respectively for x and y
    """    
    XYT = np.dot(x, y.T)
    model = TruncatedSVD(comps)
    U = model.fit_transform(XYT)
    # Rescale to unit norm to have comparable latent spaces
    Zx = U / np.linalg.norm(U, axis=0)
    Zy = model.components_[:U.shape[1]].T
    return Zx, Zy