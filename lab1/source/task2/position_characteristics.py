import numpy as np
import math


def sample_average(selection):
    result = np.mean(selection)
    return result


def sample_median(selection):
    # selection.sort()
    size = len(selection)
    return (selection[size // 2 - (size % 2)] + selection[size // 2]) / 2


def half_sum_extreme_s_elem(selection):
    return (selection[0] + selection[-1]) / 2


def half_sum_quartiles(selection):
    size = len(selection)
    first_quartile = selection[math.ceil(size * 0.25) - 1]
    third_quartile = selection[math.ceil(size * 0.75) - 1]
    return (first_quartile + third_quartile) / 2


def trunc_mean(selection):
    size = len(selection)
    r = math.floor(size / 4)
    return np.mean(selection[r:size-r])
