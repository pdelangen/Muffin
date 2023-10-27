'''
Generate consensus peaks.
'''
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, oaconvolve
from scipy.signal.windows import gaussian

def merge_peaks(beds, chrom_sizes, fileFormat="narrowPeak", inferCenter=False, forceUnstranded=False, 
                  sigma="auto", perPeakDensity=False, perPeakMultiplier=0.5,
                  minOverlap=2):
    """
    Read peak called files, generate consensuses and the matrix.

    Parameters
    ----------
    beds: list of pandas dataframes
        Each dataframe should be formatted in bed format.
        Summit is assumed to be 7th column (=thickStart) for bed format,
        and 9th column for narrowPeak format.

    chrom_sizes: str
        Path to tab separated (chromosome, length) annotation file.

    fileFormat: "bed" or "narrowPeak"
        Format of the files being read. 
        Bed file format assumes the max signal position to be at the 6th column 
        (0-based) in absolute coordinates.
        The narrowPeak format assumes the max signal position to be the 9th column 
        with this position being relative to the start position.

    inferCenter: boolean (optional, default False)
        If set to true will use the position halfway between start and end 
        positions. Enable this only if the summit position is missing.

    forceUnstranded: Boolean (optional, default False)
        If set to true, assumes all peaks are not strand-specific even 
        if strand specific information was found.

    sigma: float or "auto" (optional, default "auto")
        Size of the gaussian filter (lower values = more separation).
        Only effective if perPeakDensity is set to False. 
        "auto" automatically selects the filter width at 
        (average peak size)/8. 
    
    perPeakDensity: Boolean (optional, default False)
        If set to false will perform a gaussian filter along the genome (faster),
        assuming all peaks have roughly the same size.
        If set to true will create the density curve per peak based on each peak
        individual size. This is much more slower than the filter method.
        May be useful if peaks are expected to have very different sizes. Can
        also be faster when the number of peaks is small.

    perPeakMultiplier: float (optional, default 0.25)
        Only effective if perPeakDensity is set to True. Adjusts the width of 
        the gaussian fitted to each peak (lower values = more separation).
    
    minOverlap: integer (optional, default 2)
        Minimum number of peaks required at a consensus. 2 Indicates that a peak
        must be replicated at least once.
    """
    alltabs = []
    for tab in beds:
        fmt = fileFormat
        if not fmt in ["bed", "narrowPeak"]:
            raise TypeError(f"Unknown file format : {fmt}")
        # Read bed format
        if fmt == "bed":
            if inferCenter:
                usedCols = [0,1,2,5]
            else:
                usedCols = [0,1,2,5,6]
            tab = tab.iloc[:, usedCols].copy()
            tab[5000] = 1
            tab.columns = np.arange(len(tab.columns))
            tab[0] = tab[0].astype("str", copy=False)
            tab[3].fillna(value=".", inplace=True)
            if inferCenter:
                tab[5] = tab[4]
                tab[4] = ((tab[1]+tab[2])*0.5).astype(int)
            tab[5] = [1]*len(tab)
            alltabs.append(tab)
        elif fmt == "narrowPeak":
            if inferCenter:
                usedCols = [0,1,2,5]
            else:
                usedCols = [0,1,2,5,9]
            tab = tab.iloc[:, usedCols].copy()
            tab[5000] = 1
            tab.columns = np.arange(len(tab.columns))
            tab[0] = tab[0].astype("str", copy=False)
            tab[3].fillna(value=".", inplace=True)
            if inferCenter:
                tab[5] = tab[4]
                tab[4] = ((tab[1]+tab[2])*0.5).astype(int, copy=False)
            else:
                tab[4] = (tab[1] + tab[4]).astype(int, copy=False)
            alltabs.append(tab)    
    # Concatenate files
    df = pd.concat(alltabs)
    numElements = len(df)
    avgPeakSize = np.median(df[2] - df[1])
    # Check strandedness
    if forceUnstranded == True:
        df[3] = "."
        strandCount = 1
    else:
        # Check if there is only stranded or non-stranded elements
        strandValues = np.unique(df[3])
        strandCount = len(strandValues)
        if strandCount > 2:
            raise ValueError("More than two strand directions !")
        elif strandCount == 2 and "." in strandValues:
            raise ValueError("Unstranded and stranded values !")
    # Split per strand
    df = dict([(k, x) for k, x in df.groupby(3)])
    ########### Peak separation step ########### 
    # Compute sigma if automatic setting
    if sigma == "auto":   
        sigma = avgPeakSize/8
    else:
        sigma = float(sigma)
    if perPeakDensity:
        sigma = perPeakMultiplier
    windowSize = int(8*sigma)+1
    sepPerStrand = {}
    sepIdxPerStrand = {}
    # Iterate for each strand
    consensuses = []
    j = 0
    for s in df.keys():
        # Split peaks per chromosome
        df[s].sort_values(by=[0, 4], inplace=True)
        posPerChr = dict([(k, x.values[:, [1,2,4]].astype(int)) for k, x in df[s].groupby(0)])
        infoPerChr = dict([(k, x.values) for k, x in df[s].groupby(0)])
        # Iterate over all chromosomes
        sepPerStrand[s] = {}
        sepIdxPerStrand[s] = {}
        for chrName in posPerChr.keys():
            # Place peak on the genomic array
            try:
                currentLen = chrom_sizes[str(chrName)]
            except KeyError:
                print(f"Warning: chromosome {str(chrName)} is not in genome annotation and will be removed")
                continue
            array = np.zeros(currentLen, dtype="float32")
            peakIdx = posPerChr[chrName]
            np.add.at(array, peakIdx[:, 2],1)
            if not perPeakDensity:
                # Smooth peak density
                smoothed = oaconvolve(array, gaussian(windowSize, sigma), "same")
                # Split consensuses
                separators = argrelextrema(smoothed, np.less_equal)[0]      # Get local minimas
            else:
                smoothed = np.zeros(currentLen, dtype="float32")
                for i in range(len(peakIdx)):
                    peakSigma = (peakIdx[i, 1] - peakIdx[i, 0])*sigma
                    windowSize = int(8*peakSigma)+1
                    start = max(peakIdx[i, 2] - int(windowSize/2), 0)
                    end = min(peakIdx[i, 2] + int(windowSize/2) + 1, currentLen)
                    diffStart = max(-peakIdx[i, 2] + int(windowSize/2), 0)
                    diffEnd = windowSize + min(currentLen - peakIdx[i, 2] - int(windowSize/2) - 1, 0)
                    smoothed[start:end] += gaussian(windowSize, peakSigma)[diffStart:diffEnd]
                separators = argrelextrema(smoothed, np.less_equal)[0]      # Get local minimas
            separators = separators[np.where(np.ediff1d(separators) != 1)[0]+1]    # Removes consecutive separators (because less-equal comparison)
            separators = np.insert(separators, [0,len(separators)], [0, currentLen])        # Add start and end points
            # Assign peaks to separators
            # Not the most optimized but fast enough
            smallest_bin = np.digitize(peakIdx[:,0], separators)
            largest_bin = np.digitize(peakIdx[:,1], separators)
            bin_to_segments = dict()
            for seg_id, (smallest, largest) in enumerate(zip(smallest_bin, largest_bin)):
                seg_start, seg_end = peakIdx[seg_id, :2]
                seg_length = seg_end - seg_start
                for bin_id in range(smallest, largest+1):
                    bin_start = separators[bin_id-1]  # np.digitize's output is 1-indexed
                    bin_end = separators[bin_id]
                    bin_length = bin_end - bin_start

                    overlap_start = max(bin_start, seg_start)
                    overlap_end = min(bin_end, seg_end)
                    overlap_length = max(0, overlap_end - overlap_start)
                    
                    if overlap_length / bin_length > 0.5 or overlap_length / seg_length > 0.5:
                        if bin_id in bin_to_segments:
                            bin_to_segments[bin_id].append(seg_id)
                        else:
                            bin_to_segments[bin_id] = [seg_id]
            # Format consensus peaks
            for k in bin_to_segments.keys():
                currentConsensus = infoPerChr[chrName][bin_to_segments[k]]
                # Exclude consensuses that are too small
                if len(currentConsensus) < minOverlap:
                    continue
                currentSep = separators[k-1:k+1]
                # Setup consensuses coordinates
                consensusStart = max(np.min(currentConsensus[:,1]), currentSep[0])
                consensusEnd = min(np.max(currentConsensus[:,2]), currentSep[1])
                # Discard abnormally small consensus peaks
                if consensusEnd-consensusStart < avgPeakSize*0.125:
                    continue
                inSep = (currentConsensus[:,4] > currentSep[0]) & (currentConsensus[:,4] < currentSep[1]) 
                if inSep.sum() >= 1:
                    consensusCenter = int(np.mean(currentConsensus[inSep,4]))
                else:
                    consensusCenter = int(consensusStart*0.5+consensusEnd*0.5)
                # Mean value of present features
                meanScore = len(currentConsensus)
                # Add consensus to the genomic locations
                data = [chrName, consensusStart, consensusEnd, j, 
                        meanScore, s, consensusCenter, consensusCenter + 1]
                consensuses.append(data)
                j += 1
    return pd.DataFrame(consensuses)
