import numpy as np
import pynndescent
import igraph
import leidenalg
from sklearn.metrics import balanced_accuracy_score
import umap
from sklearn.cluster import MiniBatchKMeans
from fastcluster import linkage_vector, linkage
import scipy.cluster.hierarchy as hierarchy
from sklearn.neighbors import KNeighborsTransformer
from scipy.spatial.distance import pdist

def graphClustering(matrix, metric, k="auto", r=1.0, snn=True, 
                    approx=True, restarts=1):
    """
    Performs graph based clustering on the matrix.

    Parameters
    ----------
    metric: string
        Metric used for nn query. 
        See the pynndescent documentation for a list of available metrics.

    r: float (optional, default 1.0)
        Resolution parameter of the graph partitionning algorithm. Lower values = less clusters.

    k: "auto" or integer (optional, default "auto")
        Number of nearest neighbors used to build the NN graph.
        If set to auto uses 5*numPoints^0.2 neighbors as a rule of thumb, as too few 
        NN with a lot of points can create disconnections in the graph.

    snn: Boolean (optional, default True)
        If set to True, it will perform the Shared Nearest Neighbor Graph 
        clustering variant, where the edges of the graph are weighted according 
        to the number of shared nearest neighbors between two nodes. Otherwise,
        all edges are equally weighted. SNN usually produces a more refined clustering 
        but it can also hallucinate some clusters.
    
    approx: Boolean (optional, default True)
        Whether to use approximate nearest neighbors using nearest neighbor descent
        or exact nearest neighbors. The exact method will take very long on a large number of
        points (>15000-20000).

    restarts: integer (optional, default 1)
        The number of times to restart the graph partitionning algorithm, before keeping 
        the best partition according to the quality function.
    
    Returns
    -------
    labels: ndarray
        Index of the cluster each sample belongs to.
    """
    # Create NN graph
    if k == "auto":
        k = int(np.power(matrix.shape[0], 0.2)*4)
    if approx:
        # Add a few extra NNs to compute in order to get more accurate ANNs
        extraNN = 20
        index = pynndescent.NNDescent(matrix, n_neighbors=k+extraNN+1, metric=metric, 
                                    low_memory=False, random_state=42)
        nnGraph = index.neighbor_graph[0][:, 1:k+1]
    else:
        graph = KNeighborsTransformer(mode="connectivity", n_neighbors=k+1, 
                        metric=metric, n_jobs=-1).fit_transform(matrix)
        nnGraph = np.array(graph.nonzero()).T
        nnGraph = nnGraph[:, 1].reshape(len(matrix), -1)[:, 1:k+1]
    edges = np.zeros((nnGraph.shape[0]*nnGraph.shape[1], 2), dtype='int64')
    if snn:
        weights = np.zeros((nnGraph.shape[0]*nnGraph.shape[1]), dtype='float')
    for i in range(len(nnGraph)):
        for j in range(nnGraph.shape[1]):
            if nnGraph[i, j] > -0.5:    # Pynndescent may fail to find nearest neighbors in some cases
                link = nnGraph[i, j]
                edges[i*nnGraph.shape[1]+j] = [i, link]
                if snn:
                    # Weight the edges based on the number of shared nearest neighbors between two nodes
                    weights[i*nnGraph.shape[1]+j] = len(np.intersect1d(nnGraph[i], nnGraph[link]))
    graph = igraph.Graph(n=len(nnGraph), edges=edges, directed=True)
    # Restart clustering multiple times and keep the best partition
    best = -np.inf
    partitions = None
    for i in range(restarts):
        if snn:
            part = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, 
                                            seed=i, resolution_parameter=r, weights=weights, n_iterations=-1)
        else:
            part = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, 
                                            seed=i, resolution_parameter=r, n_iterations=-1)
        if part.quality() > best:
            partitions = part
            best = part.quality()
    # Map partitions to per object assignments
    clustered = np.zeros(len(nnGraph), dtype="int")
    for i, p in enumerate(partitions):
        clustered[p] = i
    return clustered


def twoStagesHClinkage(matrix, kMetaSamples=10000, method="ward", metric="euclidean",):
    """
    Three steps Hierachical clustering. UMAP -> K-Means -> Ward HC on clusters
    centroids.

    Parameters
    ----------
    matrix : array-like
        Data matrix
    
    metric : string
        Metric used for nn query. It is recommended to use Pearson correlation
        for float values and Dice similarity for binary data.
        See the pynndescent documentation for a list of available metrics.
    
    kMetaSamples : int, optional (default 10000)
        Number of K-Means clusters, or groups of samples used by HC.

    method : string, optional (default "ward")
        HC method
    """
    fast = method in ('single', 'centroid', 'median', 'ward')
    fast = (method == "single") or (fast and metric == "euclidean")
    if fast:
        clusterFunc = linkage_vector
    else:
        clusterFunc = linkage
    # Aggregrate samples via K-means in order to scale to large datasets
    if len(matrix) > kMetaSamples:
        clustering = MiniBatchKMeans(n_clusters=kMetaSamples, init="random", random_state=42, 
                                    n_init=1)
        assignedClusters = clustering.fit_predict(matrix)
        Kx = clustering.cluster_centers_
        # Perform HC
        link = clusterFunc(Kx, method=method, metric=metric)
        Korder = hierarchy.leaves_list(link)
        order = np.array([], dtype="int")
        for c in Korder:
            order = np.append(order, np.where(c == assignedClusters)[0])
        return order, link
    else:
        link = clusterFunc(matrix, method=method, metric=metric)
        return hierarchy.leaves_list(link), link


def HcOrderRow(mat, method="ward", metric="euclidean"):
    link = linkage_vector(mat, method=method, metric=metric)
    rowOrder = hierarchy.leaves_list(link)
    return rowOrder

def HcOrder(mat, method="ward", metric="euclidean"):
    link = linkage_vector(mat, method=method, metric=metric)
    rowOrder = hierarchy.leaves_list(link)
    link = linkage_vector(mat.T, method=method, metric=metric)
    colOrder = hierarchy.leaves_list(link)
    return rowOrder, colOrder

def looKnnCV(X, Y, metric, k):
    '''
    Leave-one-out KNN classification
    '''
    index = pynndescent.NNDescent(X, n_neighbors=min(30+k, len(X)-1), 
                                metric=metric, low_memory=False, random_state=42)
    # Exclude itself and select NNs (equivalent to leave-one-out cross-validation)
    # Pynndescent is ran with a few extra neighbors for a better accuracy on ANNs
    nnGraph = index.neighbor_graph[0][:, 1:k+1]
    pred = []
    annProp = np.bincount(Y)
    for nns in Y[nnGraph]:
        # Find the most represented label in the k nearest neighbors,
        # Weighted by the proportion of the labels
        pred.append(np.argmax(np.bincount(nns, minlength=len(annProp))/annProp))
    score = balanced_accuracy_score(Y, pred)
    return score


def approx_knn_predictor(X, Y, metric, k):
    index = pynndescent.NNDescent(X, n_neighbors=min(30+k, len(X)-1), 
                                metric=metric, low_memory=False, random_state=42)
    # Exclude itself and select NNs (equivalent to leave-one-out cross-validation)
    # Pynndescent is ran with a few extra neighbors for a better accuracy on ANNs
    nnGraph = index.neighbor_graph[0][:, 0:k]
    pred = []
    annProp = np.sqrt(np.bincount(Y) + 1)
    for nns in Y[nnGraph]:
        # Find the most represented label in the k nearest neighbors,
        # Weighted by the proportion of the labels
        pred.append(np.argmax(np.bincount(nns, minlength=len(annProp))/annProp))
    return pred

