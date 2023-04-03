import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt


def uniform_distr(a, b):
    start_point = a
    length = b - a
    iterations = {10: 15, 50: 15, 1000: 15}
    x = np.linspace(uniform.ppf(0.001, loc=start_point, scale=length), uniform.ppf(0.999, loc=start_point, scale=length), 1000)


    for cur_n in iterations:
        plt.figure()
        plt.title(f"Выборка из n={cur_n} элементов для равномерного распределения (a={a}, b={b})")
        plt.xlabel("Uniform numbers")
        plt.ylabel("Density")
        plt.plot(x, uniform.pdf(x, loc=start_point, scale=length), 'r-')

        s = uniform.rvs(loc=start_point, scale=length, size=cur_n)
        plt.hist(s, density=True, color='white', edgecolor='black', bins=iterations[cur_n])

    plt.show()
