# %%
import pandas as pd
import pyranges as pr
from collections import OrderedDict
from copy import deepcopy
import sys
sys.path.append("./")
from settings import paths, settings
import numpy as np
import muffin
import os
files = [paths.ctcf_remap_bed, paths.cage_peaks, paths.histone_peaks]
out_files = ["benchmarks/peakmerge/benchmark_results/density_peakmerge_ctcf.bed", 
       "benchmarks/peakmerge/benchmark_results/density_peakmerge_cage.bed", 
       "benchmarks/peakmerge/benchmark_results/density_peakmerge_histone.bed"]
output_bedgraph = ["benchmarks/peakmerge/benchmark_results/bedgraphs/density_peakmerge_ctcf.wig", 
       "benchmarks/peakmerge/benchmark_results/bedgraphs/density_peakmerge_cage.wig",
       "benchmarks/peakmerge/benchmark_results/bedgraphs/density_peakmerge_histone.wig"]
# %%
i = 0
f = files[i]
out = out_files[i]
bed_all = pr.read_bed(f).as_df()
bed_all.dtypes
# %%
merged = muffin.peakMerge.merge_peaks(bed_all, paths.chromsizes, fileFormat="bed", 
                                      output_bedgraph=output_bedgraph[i])
merged.to_csv(out, sep="\t", header=None, index=None)

# %%

i = 1
f = files[i]
out = out_files[i]
cage_peak_files = os.listdir(f)
all_beds = [pr.read_bed(f+file).as_df() for file in cage_peak_files]
# %%
pd.concat(all_beds).iloc[:, [0,1,2,3,4,5]].to_csv("benchmarks/peakmerge/benchmark_results/cage_all.bed", sep="\t", header=None, index=None)
merged = muffin.peakMerge.merge_peaks(pd.concat(all_beds), paths.chromsizes, fileFormat="bed", 
                                      output_bedgraph=output_bedgraph[i], inferCenter=True)
merged.to_csv(out, sep="\t", header=None, index=None)
# %%

i = 2
f = files[i]
out = out_files[i]
cage_peak_files = os.listdir(f)
all_beds = [pr.read_bed(f+file).as_df() for file in cage_peak_files]
# %%
pd.concat(all_beds).iloc[:, [0,1,2,3,4,5]].to_csv("benchmarks/peakmerge/benchmark_results/histone_all.bed", sep="\t", header=None, index=None)
merged = muffin.peakMerge.merge_peaks(pd.concat(all_beds), paths.chromsizes, fileFormat="narrowPeak", 
                                      output_bedgraph=output_bedgraph[i], inferCenter=True)
merged.to_csv(out, sep="\t", header=None, index=None)
# %%
