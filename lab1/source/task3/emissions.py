import numpy as np


def count_emissions(selection):
    size = len(selection)
    Q1 = np.quantile(selection, 0.25)
    Q3 = np.quantile(selection, 0.75)
    X1 = Q1 - 1.5 * (Q3 - Q1)
    X2 = Q3 + 1.5 * (Q3 - Q1)
    count = 0
    for x in selection:
        if x < X1 or x > X2:
            count += 1
    return count
