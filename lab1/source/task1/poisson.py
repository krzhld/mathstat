import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt


def poisson_disrt(mu):
    iterations = {10: 5, 50: 15, 1000: 20}
    x = np.arange(poisson.ppf(0.001, mu=mu), poisson.ppf(0.999, mu=mu))

    for cur_n in iterations:
        plt.figure()
        plt.title(f"Выборка из n={cur_n} элементов для распределения Пуассона (μ={mu})")
        plt.xlabel("Poisson numbers")
        plt.ylabel("Density")
        plt.plot(x, poisson.pmf(x, mu=mu), 'r-')

        s = poisson.rvs(mu=mu, size=cur_n)
        plt.hist(s, density=True, color='white', edgecolor='black', bins=iterations[cur_n])

    plt.show()
