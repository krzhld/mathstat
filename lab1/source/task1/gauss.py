import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def gauss_distr(mu, sigma):
    iterations = {10: 10, 50: 15, 1000: 20}
    x = np.linspace(norm.ppf(0.001, loc=mu, scale=sigma), norm.ppf(0.999, loc=mu, scale=sigma), 1000)

    for cur_n in iterations:
        plt.figure()
        plt.title(f"Выборка из n={cur_n} элементов для стандартного нормального распределения")
        plt.xlabel("Normal numbers")
        plt.ylabel("Density")
        plt.plot(x, norm.pdf(x), 'r-')

        s = norm.rvs(loc=mu, scale=sigma, size=cur_n)
        plt.hist(s, density=True, color='white', edgecolor='black', bins=iterations[cur_n])

    plt.show()
