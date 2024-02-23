params = {"autosave_plots":None,
          "autosave_format":".pdf",
          "figure_dpi":300,
          "temp_dir":"tmp_muffin/"}

from . import load
from .peakMerge import merge_peaks
from . import great
from . import plots
from . import tools