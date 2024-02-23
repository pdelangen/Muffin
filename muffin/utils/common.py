import subprocess
import pickle
import os
import scipy as sc
import pandas as pd
import textwrap


def sanitize_array(array):
    if sc.sparse.issparse(array):
        return array.toarray()
    elif isinstance(array, pd.DataFrame) or isinstance(array, pd.Series):
        return array.values
    else:
        return array

def createDir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        print(f"Directory {path} already exists !")


def customwrap(s,width=20,max_lines=5):
    return "<br>".join(textwrap.wrap(s,width=width, max_lines=max_lines)).capitalize()

def capTxtLen(txt, maxlen):
    try:
        if len(txt) < maxlen:
            return txt
        else:
            return txt[:maxlen] + '...'
    except:
        return "N/A"

def runScript(script, argumentList, outFile=None):
    # Runs the command as a standard bash command
    # script is command name without path
    # argumentList is the list of argument that will be passed to the command
    if outFile == None:
        subprocess.run([script] + argumentList)
    else:
        with open(outFile, "wb") as outdir:
            subprocess.run([script] + argumentList, stdout=outdir)


def dump(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def read_bigBed(path):
    import pyBigWig
    bb = pyBigWig.open(path)
    chroms = bb.chroms()
    entries_list = []
    for chrom, length in chroms.items():
        entries = bb.entries(chrom, 0, length)
        for entry in entries:
            entries_list.append({
                'chrom': chrom,
                'start': entry[0],
                'end': entry[1],
            })
    return pd.DataFrame(entries_list)