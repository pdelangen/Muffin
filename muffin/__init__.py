"""
Functions for generic comparisons of regulatory sequencing datasets.
"""

params = {"autosave_plots":None,
          "autosave_format":".pdf",
          "figure_dpi":300}

from . import load
from .peakMerge import mergePeaks
from . import grea
from . import plots
from . import recipes
from . import tools