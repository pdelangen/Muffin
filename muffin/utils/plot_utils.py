import numpy as np
import seaborn as sns
import pandas as pd 
import numba as nb
import matplotlib.pyplot as plt
from . import cluster
import matplotlib as mpl
from skimage.transform import resize
import scipy.cluster.hierarchy as hierarchy
from matplotlib.patches import Patch

def applyPalette(annot, avail, palettePath, ret_labels=False):
    annotPalette = pd.read_csv(palettePath, sep=",", index_col=0)
    palette = annotPalette.loc[avail][["r","g","b"]]
    colors = annotPalette.loc[annot][["r","g","b"]].values
    if ret_labels:
        return palette.index, palette.values, colors
    return palette.values, colors
    
def getPalette(labels, palette=None):
    """
    Selects a proper palette according to the annotation count.
    Uses "paired" from seaborn if the number of labels is under 12.
    Uses "tab20" without grey from seaborn if the number of labels is under 18.
    Uses a quasi-random sequence otherwise.
    Then applies palette according to integer labels

    Parameters
    ----------
    labels: ndarray of integers
        Label for each point.

    Returns
    -------
    palette: ndarray of shape (n_labels,3)
        0-1 rgb color values for each label

    colors: ndarray of shape (n_points,3)
        0-1 rgb color values for each point
    """
    numLabels = np.max(labels)
    if numLabels < 10:
        palette = np.array(sns.color_palette())
        colors = palette[labels]
    elif numLabels < 12:
        palette = np.array(sns.color_palette("Paired"))
        colors = palette[labels]
    elif numLabels < 18:
        # Exclude gray
        palette = sns.color_palette("tab20")
        palette = np.array(palette[:14] + palette[16:])
        colors = palette[labels]
    else:  
        # Too many labels, use random colors
        # Quasi-Random Sequence (has better rgb coverage than random)
        g = 1.22074408460575947536
        v = np.array([1/g, 1/g/g, 1/g/g/g])[:, None]
        palette = np.mod(v * (np.arange(numLabels + 1) + 1), 1.0).T
        colors = palette[labels]
    return palette, colors

@nb.njit()
def raster_matrix(matrix, row_order, col_order, res_x=4000, res_y=4000):
    rasterSize = (min(res_x, matrix.shape[0]),min(res_y, matrix.shape[1]))
    fig = np.zeros(rasterSize, dtype="float32")
    sums = np.zeros(rasterSize,  dtype="float32")
    for i, xc in enumerate(row_order):
        xCoord = rasterSize[0] * i / len(matrix)
        xCoordInt = int(xCoord)
        for j, yc in enumerate(col_order):
            yCoord = rasterSize[1] * j / matrix.shape[1]
            yCoordInt = int(yCoord)
            # Bilinear interpolation
            w = 1.0 - np.abs(xCoordInt - xCoord) * np.abs(yCoordInt - yCoord)
            fig[xCoordInt, yCoordInt] += matrix[xc, yc] * w
            sums[xCoordInt, yCoordInt] += w
            if yCoordInt < rasterSize[1]-1:
                w = 1.0 - np.abs(xCoordInt - xCoord) * np.abs(1 + yCoordInt - yCoord)
                fig[xCoordInt, 1+yCoordInt] += matrix[xc, yc] * w
                sums[xCoordInt, 1+yCoordInt] += w
            if xCoordInt < rasterSize[0]-1:
                w = 1.0 - np.abs(xCoordInt + 1 - xCoord) * np.abs(yCoordInt - yCoord)
                fig[xCoordInt+1, yCoordInt] += matrix[xc, yc] * w
                sums[xCoordInt+1, yCoordInt] += w
            if (xCoordInt < rasterSize[0]-1) and (yCoordInt < rasterSize[1]-1):
                w = 1.0 - np.abs(xCoordInt + 1 - xCoord) * np.abs(1+yCoordInt - yCoord)
                fig[xCoordInt+1, 1+yCoordInt] += matrix[xc, yc] * w
                sums[xCoordInt+1, 1+yCoordInt] += w
    fig = fig / (1e-9+sums)
    return fig

@nb.njit()
def raster_matrix_3axes(matrix, row_order, col_order, res_x=4000, res_y=4000):
    rasterSize = (min(res_x, matrix.shape[0]),min(res_y, matrix.shape[1]), matrix.shape[2])
    fig = np.zeros(rasterSize, dtype="float32")
    sums = np.zeros(rasterSize,  dtype="float32")
    for i, xc in enumerate(row_order):
        xCoord = rasterSize[0] * i / len(matrix)
        xCoordInt = int(xCoord)
        for j, yc in enumerate(col_order):
            yCoord = rasterSize[1] * j / matrix.shape[1]
            yCoordInt = int(yCoord)
            # Bilinear interpolation
            w = 1.0 - np.abs(xCoordInt - xCoord) * np.abs(yCoordInt - yCoord)
            fig[xCoordInt, yCoordInt] += matrix[xc, yc] * w
            sums[xCoordInt, yCoordInt] += w
            if yCoordInt < rasterSize[1]-1:
                w = 1.0 - np.abs(xCoordInt - xCoord) * np.abs(1 + yCoordInt - yCoord)
                fig[xCoordInt, 1+yCoordInt] += matrix[xc, yc] * w
                sums[xCoordInt, 1+yCoordInt] += w
            if xCoordInt < rasterSize[0]-1:
                w = 1.0 - np.abs(xCoordInt + 1 - xCoord) * np.abs(yCoordInt - yCoord)
                fig[xCoordInt+1, yCoordInt] += matrix[xc, yc] * w
                sums[xCoordInt+1, yCoordInt] += w
            if (xCoordInt < rasterSize[0]-1) and (yCoordInt < rasterSize[1]-1):
                w = 1.0 - np.abs(xCoordInt + 1 - xCoord) * np.abs(1+yCoordInt - yCoord)
                fig[xCoordInt+1, 1+yCoordInt] += matrix[xc, yc] * w
                sums[xCoordInt+1, 1+yCoordInt] += w
    fig = fig / (1e-9+sums)
    return fig
@nb.njit()
def mean_squared(matrix, centers):
    msr = np.zeros(matrix.shape[1])
    for j in range(matrix.shape[1]):
            squaredVals = np.square(matrix[:,j]-centers[j])
            msr[j] = np.mean(squaredVals)
    return msr

def frac_barplot(matrix, labels, categories, palette, order, res_x=4000, res_y=4000):
    ssr_per_cat = np.zeros((len(categories), matrix.shape[1]))
    centers = np.min(matrix, axis=0)
    for i, c in enumerate(categories):
        inCat = labels == c
        ssr_per_cat[i] = mean_squared(matrix[inCat], centers)
    ssr_per_cat = ssr_per_cat / ssr_per_cat.sum(axis=0)
    rasterRes = (res_x,min(res_y, matrix.shape[1]))
    ssr_per_cat_per_bucket = np.zeros((len(ssr_per_cat), rasterRes[1]))
    for j, yc in enumerate(order):
        yCoord = rasterRes[1] * j / matrix.shape[1]
        yCoordInt = int(yCoord)
        ssr_per_cat_per_bucket[:, yCoordInt] += ssr_per_cat[:, yc]
    ssr_per_cat_per_bucket /= ssr_per_cat_per_bucket.sum(axis=0)
    runningSum = np.zeros(ssr_per_cat_per_bucket.shape[1])
    barPlot = np.zeros((rasterRes[0], ssr_per_cat_per_bucket.shape[1], 3))
    fractCount = ssr_per_cat_per_bucket*rasterRes[0]
    for i, c in enumerate(fractCount):
        for j, f in enumerate(c):
            positions = np.array([runningSum[j]+0.5,runningSum[j]+f+0.5]).astype(np.int64)
            barPlot[positions[0]:positions[1], j] = palette[i]
            runningSum[j] += f
    return barPlot



def mega_clustermap(matrix, rowOrder=None, colOrder=None, rowLink=None, colLink=None, labels=None, 
                    resolution=4000, vmin=None, vmax=None, 
                    show_dendrogram=True, show_bar=True,
                    max_dendro=5000, cmap="vlag", dpi=300,
                    figsize=(3,3)):
    if rowOrder is None:
        rowOrder = np.arange(matrix.shape[0])
        show_dendrogram = False
    if colOrder is None:
        colOrder = np.arange(matrix.shape[1])
        show_dendrogram = False
    if vmin is None:
        vmin = np.min(matrix)
    if vmax is None:
        vmax = np.max(matrix)
    # Get colors and palettes
    if labels is not None:
        idx, eq = pd.factorize(labels)
        palette, colors = getPalette(idx)
    res_x = resolution
    res_y = resolution
    # Raster heatmap
    fig_img = raster_matrix(matrix, rowOrder, colOrder, res_x=res_x, res_y=res_y)
    if fig_img.shape[1] != res_y:
        fig_img = resize(fig_img, 
                            (fig_img.shape[0], res_y), anti_aliasing=True, order=int(fig_img.shape[1]>res_y))
    if fig_img.shape[0] != res_x:
        fig_img = resize(fig_img, 
                            (res_x, res_y), anti_aliasing=True, order=int(fig_img.shape[0]>res_x))
    fig_img = np.clip((fig_img - vmin) / (vmax-vmin), 0.0, 1.0)
    if type(cmap) is str:
        cmap = sns.color_palette(cmap, as_cmap=True)
    fig_img = cmap(fig_img.ravel())
    fig_img = fig_img.reshape(res_x, res_y, -1)[:,:,:3]
    # Render barplot
    if labels is not None:
        barplot = frac_barplot(matrix, labels, eq, palette, colOrder, res_x=res_x, res_y=res_y)
        barplot = resize(barplot, 
                        (int(res_x*0.1), res_y), anti_aliasing=True)
        # Draw category info
        labelCol = colors[rowOrder].reshape(len(colors),1,3)
        labelCol = resize(labelCol, 
                        (res_x, int(res_y*0.02)), anti_aliasing=True, order=int(labelCol.shape[0]>res_x))
    def color_func(*args):
        return '#555555'
    # Define the width and height ratios for the subplots
    width_ratios = [1.0, 1.0, 40.0, 4]
    height_ratios = [1.0, 40.0, 4.0]
    fig, ax = plt.subplots(3, 4, figsize=figsize, dpi=dpi,
                        gridspec_kw={'width_ratios': width_ratios, 'height_ratios': height_ratios,
                                        "hspace":0.005, "wspace":0.005})
    if show_dendrogram:
        hierarchy.dendrogram(rowLink, p=max_dendro, truncate_mode="lastp", color_threshold=-1, ax=ax[1,0],
                            orientation="left", link_color_func=color_func, show_leaf_counts=False,
                            no_labels=True)
        hierarchy.dendrogram(colLink, p=max_dendro, truncate_mode="lastp", color_threshold=-1, ax=ax[0,2],
                            orientation="top", link_color_func=color_func, show_leaf_counts=False,
                            no_labels=True)
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            if not (i == 1 and j == 3):
                ax[i,j].axis("off")

    if labels is not None:
        ax[1,1].imshow(labelCol[::-1], 
                interpolation="nearest")
        if show_bar:
            ax[2,2].imshow(barplot, 
                    interpolation="nearest")
    ax[1,2].imshow(fig_img[::-1], 
            interpolation="nearest")
    ax[1,2].set_aspect(fig.get_size_inches()[1]/fig.get_size_inches()[0]*1.03)
    ax[2,2].set_aspect(fig.get_size_inches()[1]/fig.get_size_inches()[0])
    cbar = mpl.colorbar.ColorbarBase(ax[1,3], cmap=cmap, orientation = 'vertical',
                              norm=mpl.colors.Normalize(vmin, vmax),
                              ticks=np.linspace(vmin, vmax, 5))
    cbar.ax.tick_params(labelsize=6)
    ax[1,3].set_aspect(7.5)
    if labels is not None:
        patches = []
        for i in range(len(eq)):
            legend = Patch(color=palette[i], label=eq[i])
            patches.append(legend)
        fig.legend(handles=patches, prop={'size': 6}, 
                    loc="upper center", ncol=5)
    return fig, ax