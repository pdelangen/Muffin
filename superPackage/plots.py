import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import plot_utils


def plot_reduced_dim(dataset, which="Umap", components=[0,1], points_labels=None, 
                        label_type="categorical", palette="auto", highlight=None):
    """

    Parameters
    ----------
    which : str, optional
        Which reduced dimensionality representation to use, 
        by default "X_umap"
    components : integer list of length 2, optional
        Which components of the reduced dimensionality representation to plot, 
        by default [0,1]
    points_labels : ndarray, optional
        Label for each point, by default None
    label_type : "categorical" or "numeric", optional
        If set to "categorical" assumes, by default "categorical"
    palette : "auto" or matplotlib colormap or dict, optional
        If set "auto", will automatically choose the color palette
        according to label type and number of labels.
        If label_type is set to "numeric", it should be a matplotlib colormap.
        If label_type is set to "categorical" should be a dictionnary
        {category:color} with color being an hexadecimal str or rgb array-like (r,g,b).
        By default "auto".
    highlight : boolean ndarray, optional
        Points to highlight on the graph, by default None.

    Returns
    -------
    self
    """        
    plt.figure(dpi=500)
    # Add per point annotation
    try:
        data = dataset.obsm[which][:, components]
    except KeyError:
        raise KeyError(f"Invalid 'which' parameter : {which}, make sure you have initialized the key or computed pca beforehand.")
    if points_labels is not None:
        if label_type=="categorical":
            factorized, factors = pd.factorize(points_labels)
            palette, c = plot_utils.getPalette(factorized)
        elif label_type=="numeric":
            if palette == "auto":
                palette = "viridis"
            rescaled_labels = (points_labels - points_labels.min())
            rescaled_labels /= rescaled_labels.max()
            c = sns.color_palette(palette, as_cmap=True)(rescaled_labels)
        else:
            raise ValueError(f"Invalid label_type argument : {label_type}, use 'categorical' or 'numeric'.")
    else:
        c="b"
    # Add highlight on a subset of points
    if highlight is not None:
        s = highlight.astype(float)
    else:
        s = 0.0
    base_dot_size = np.clip(9000./len(data), 0.5, 50.0)
    plt.scatter(data[:, 0], data[:, 1], c=c, 
                s=base_dot_size * (1+s*4), 
                linewidths=np.sqrt(base_dot_size)*0.5*s, edgecolors="k")
    plt.xlabel(f"{which} {components[0]+1}")
    plt.ylabel(f"{which} {components[1]+1}")
    plt.show()
    return dataset

def mega_heatmap(dataset):
    pass