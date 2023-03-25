import numpy as np
import position_characteristics as pc
from scipy.stats import poisson


def poisson_char(mu):
    iterations = {10, 100, 1000}
    E = []
    D = []

    for cur_n in iterations:
        x_mean = []
        x_med = []
        z_R = []
        z_Q = []
        z_tr = []
        for i in range(0, 1000):
            selection = poisson.rvs(mu=mu, size=cur_n)
            selection.sort()
            x_mean = np.append(x_mean, pc.sample_average(selection))
            x_med = np.append(x_med, pc.sample_median(selection))
            z_R = np.append(z_R, pc.half_sum_extreme_s_elem(selection))
            z_Q = np.append(z_Q, pc.half_sum_quartiles(selection))
            z_tr = np.append(z_tr, pc.trunc_mean(selection))
            characteristics = [x_mean, x_med, z_R, z_Q, z_tr]
        for cur_char in characteristics:
            cur_char_mean = np.mean(cur_char)
            E = np.append(E, cur_char_mean)
            cur_char_squared_mean = np.mean([i ** 2 for i in cur_char])
            D = np.append(D, cur_char_squared_mean - cur_char_mean ** 2)

    return E, D
