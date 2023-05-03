import KDEpy
import numpy as np

def findMode(arr):
    # Finds the modal value of a continuous sample
    pos, fitted = KDEpy.FFTKDE(bw="silverman").fit(arr).evaluate(100000)
    return pos[np.argmax(fitted)]