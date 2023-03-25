import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt


def cauchy_distr(a, b):
    iterations = {10: 10, 50: 15, 1000: 10}
    x = np.linspace(cauchy.ppf(0.01, loc=a, scale=b), cauchy.ppf(0.99, loc=a, scale=b), 1000)

    for cur_n in iterations:
        plt.figure()
        plt.title(f"Выборка из n={cur_n} элементов для распределения Коши (a={a}, b={b})")
        plt.xlabel("Cauchy numbers")
        plt.ylabel("Density")
        plt.plot(x, cauchy.pdf(x), 'r-')

        s = cauchy.rvs(loc=a, scale=b, size=cur_n)
        plt.hist(s, density=True, color='white', edgecolor='black', bins=iterations[cur_n])

    plt.show()
