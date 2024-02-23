# %%
import pandas as pd
import pyranges as pr
from collections import OrderedDict
from copy import deepcopy
import sys
sys.path.append("./")
sys.path.append("../..")
from settings import paths, settings
import os
def naive_merge(bed, strand=False):
    print(bed.as_df().dtypes)
    merged = bed.merge(strand=strand, count=True).as_df()
    return merged[merged["Count"] >= 2]

# %%
f = paths.ctcf_remap_bed
out = "benchmarks/peakmerge/benchmark_results/naive_peakmerge_ctcf.bed"
bed_all = pr.read_bed(f)
merged = bed_all.merge(count=True).as_df()
merged = merged[merged["Count"] >= 2]
pr.PyRanges(merged).to_bed(out)
# %%
f = paths.cage_peaks
out = "benchmarks/peakmerge/benchmark_results/naive_peakmerge_cage.bed"
cage_peak_files = os.listdir(f)
all_beds = [pr.read_bed(f+file).as_df() for file in cage_peak_files]
bed_all = pd.concat(all_beds).iloc[:, [0,1,2,3,4,5]]
merged = naive_merge(pr.PyRanges(bed_all), True)
merged.to_csv(out, sep="\t", header=None, index=None)
# %%
f = paths.histone_peaks
out = "benchmarks/peakmerge/benchmark_results/naive_peakmerge_histone.bed"
cage_peak_files = os.listdir(f)
all_beds = [pr.read_bed(f+file).as_df() for file in cage_peak_files]
bed_all = pd.concat(all_beds).iloc[:, [0,1,2,3,4,5]]
merged = naive_merge(pr.PyRanges(bed_all))
merged.to_csv(out, sep="\t", header=None, index=None)
# %%
