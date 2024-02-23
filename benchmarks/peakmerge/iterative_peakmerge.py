# %%
import pandas as pd
import pyranges as pr
from collections import OrderedDict
from copy import deepcopy
import sys
sys.path.append("./")
sys.path.append("../..")
from settings import paths, settings
import numpy as np


# %%
def iterative_merge(bed, min_overlap=2, infer_center=False):
    if infer_center:
        bed["ThickStart"] = bed["Start"] * 0.5 + bed["End"] * 0.5 
    bed["Strand"] = bed["Strand"].astype("str")
    df = dict([(k, x) for k, x in bed.groupby("Strand")])
    merged = []
    for s in df:
        posPerChr = dict([(k, x) for k, x in df[s].groupby("Chromosome")])
        for chrom in posPerChr:
            curr_chrom = posPerChr[chrom]
            strength_ordered = curr_chrom.sort_values("Score", ascending=False).copy()
            strength_ordered.index = np.arange(len(strength_ordered))
            summit_order = np.argsort(strength_ordered["ThickStart"].values)
            previousPeak = np.roll(strength_ordered.iloc[summit_order].index, 1)
            previousPeak[0] = -1
            previousPeak = previousPeak[np.argsort(summit_order)]
            nextPeak = np.roll(strength_ordered.iloc[summit_order].index, -1)
            nextPeak[-1] = -1
            nextPeak = nextPeak[np.argsort(summit_order)]
            strength_ordered["Next Peak"] = nextPeak
            strength_ordered["Previous Peak"] = previousPeak
            strength_ordered = strength_ordered.to_dict("index")
            strength_ordered_all = deepcopy(strength_ordered)
            for j in strength_ordered_all:
                if j in strength_ordered:
                    peak_data = strength_ordered.pop(j)
                    num_peaks = 1
                    start, end = peak_data["Start"], peak_data["End"] 
                    # Search overlaps left side
                    k = peak_data["Previous Peak"]
                    if k != -1:
                        peak_data_overlap = strength_ordered_all[k]
                        while k >= 0 and peak_data_overlap["ThickStart"] < end and peak_data_overlap["ThickStart"] > start:
                            if k in strength_ordered:
                                strength_ordered.pop(k)
                                num_peaks += 1
                            k = peak_data_overlap["Previous Peak"]
                            if k == -1:
                                break
                            peak_data_overlap = strength_ordered_all[k]
                    # Search overlaps right side
                    k = peak_data["Next Peak"]
                    if k != -1:
                        peak_data_overlap = strength_ordered_all[k]
                        while k >= 0 and peak_data_overlap["ThickStart"] < end and peak_data_overlap["ThickStart"] > start:
                            if k in strength_ordered:
                                strength_ordered.pop(k)
                                num_peaks += 1
                            k = peak_data_overlap["Next Peak"]
                            if k == -1:
                                break
                            peak_data_overlap = strength_ordered_all[k]
                    if num_peaks >= min_overlap:
                        merged.append(peak_data)
    return merged

# %%
bed_all = pr.read_bed(paths.ctcf_remap_bed).as_df()
merged = iterative_merge(bed_all)
pd.DataFrame(merged).to_csv("benchmarks/peakmerge/benchmark_results/iterative_peakmerge_cage.bed", sep="\t", header=None, index=None)# %%
# %%
import os
f = paths.cage_peaks
out = "benchmarks/peakmerge/benchmark_results/iterative_peakmerge_cage.bed"
cage_peak_files = os.listdir(f)
all_beds = [pr.read_bed(f+file).as_df() for file in cage_peak_files]
# %%
bed_all = pd.concat(all_beds)
merged = iterative_merge(bed_all, infer_center=True)
pd.DataFrame(merged).iloc[:, [0,1,2,3,4,5]].to_csv(out, sep="\t", header=None, index=None)
# %%
import os
f = paths.histone_peaks
out = "benchmarks/peakmerge/benchmark_results/iterative_peakmerge_histone.bed"
cage_peak_files = os.listdir(f)
all_beds = [pr.read_bed(f+file).as_df() for file in cage_peak_files]
# %%
bed_all = pd.concat(all_beds)
merged = iterative_merge(bed_all, infer_center=True)
pd.DataFrame(merged).iloc[:, [0,1,2,3,4,5]].to_csv(out, sep="\t", header=None, index=None)
# %%
