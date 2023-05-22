# %%
import numpy as np

randomTest = np.random.normal(size=(52, 17810))
randomTest[:10, :500] += 2.0
randomTest[45:50, 1000:1200] += 2.0
labels = np.array(["A"]*10 + ["B"]*35 + ["C"]*5 + ["B"]*2)
# %%
from muffin.utils.plot_utils import mega_clustermap
from muffin.utils.cluster import twoStagesHClinkage
import scipy.cluster.hierarchy as hierarchy
rowOrder, rowLink = twoStagesHClinkage(randomTest)
colOrder, colLink = twoStagesHClinkage(randomTest.T)
# %%
mega_clustermap(randomTest, rowOrder=rowOrder, colOrder=colOrder, rowLink=rowLink, 
                colLink=colLink, labels=labels, vmin=-3, vmax=3, resolution=4000, cmap="viridis")
# %%
