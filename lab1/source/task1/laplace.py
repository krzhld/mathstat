import numpy as np
from scipy.stats import laplace
import matplotlib.pyplot as plt


def laplace_disrt(alpha, beta):
    iterations = {10: 10, 50: 15, 1000: 20}
    x = np.linspace(laplace.ppf(0.001, loc=alpha, scale=beta), laplace.ppf(0.999, loc=alpha, scale=beta), 1000)

    for cur_n in iterations:
        plt.figure()
        plt.title(f"Выборка из n={cur_n} элементов для распределения Лапласа (α={alpha}, β={beta})")
        plt.xlabel("Laplace numbers")
        plt.ylabel("Density")
        plt.plot(x, laplace.pdf(x), 'r-')

        s = laplace.rvs(loc=alpha, scale=beta, size=cur_n)
        plt.hist(s, density=True, color='white', edgecolor='black', bins=iterations[cur_n])

    plt.show()
