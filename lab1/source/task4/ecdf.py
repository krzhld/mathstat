import numpy as np
import matplotlib.pyplot as plt


def ecdf(selection):
    x, counts = np.unique(selection, return_counts=True)
    cusum = np.cumsum(counts)
    x = np.insert(x, 0, -4)
    cusum = np.insert(cusum, 0, 0.)
    x = np.append(x, 4)
    cusum = np.append(cusum, cusum[-1])
    return x, cusum / cusum[-1]


def ecdf_pois(selection):
    x, counts = np.unique(selection, return_counts=True)
    cusum = np.cumsum(counts)
    x = np.insert(x, 0, -4)
    cusum = np.insert(cusum, 0, 0.)
    return x, cusum / cusum[-1]
