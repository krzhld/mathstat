import numpy as np
import position_characteristics as pc
from scipy.stats import cauchy


def cauchy_char(a, b):
    iterations = [10, 100, 1000]
    E = []
    D = []

    for cur_n in iterations:
        x_mean = []
        x_med = []
        z_R = []
        z_Q = []
        z_tr = []
        for i in range(0, 1000):
            selection = cauchy.rvs(loc=a, scale=b, size=cur_n)
            selection.sort()
            x_mean = np.append(x_mean, pc.sample_average(selection))
            x_med = np.append(x_med, pc.sample_median(selection))
            z_R = np.append(z_R, pc.half_sum_extreme_s_elem(selection))
            z_Q = np.append(z_Q, pc.half_sum_quartiles(selection))
            z_tr = np.append(z_tr, pc.trunc_mean(selection))

        cur_char_mean = np.mean(x_mean)
        E = np.append(E, cur_char_mean)
        cur_char_squared_mean = np.mean([i ** 2 for i in x_mean])
        D = np.append(D, cur_char_squared_mean - cur_char_mean ** 2)

        cur_char_mean = np.mean(x_med)
        E = np.append(E, cur_char_mean)
        cur_char_squared_mean = np.mean([i ** 2 for i in x_med])
        D = np.append(D, cur_char_squared_mean - cur_char_mean ** 2)

        cur_char_mean = np.mean(z_R)
        E = np.append(E, cur_char_mean)
        cur_char_squared_mean = np.mean([i ** 2 for i in z_R])
        D = np.append(D, cur_char_squared_mean - cur_char_mean ** 2)

        cur_char_mean = np.mean(z_Q)
        E = np.append(E, cur_char_mean)
        cur_char_squared_mean = np.mean([i ** 2 for i in z_Q])
        D = np.append(D, cur_char_squared_mean - cur_char_mean ** 2)

        cur_char_mean = np.mean(z_tr)
        E = np.append(E, cur_char_mean)
        cur_char_squared_mean = np.mean([i ** 2 for i in z_tr])
        D = np.append(D, cur_char_squared_mean - cur_char_mean ** 2)

    return E, D
