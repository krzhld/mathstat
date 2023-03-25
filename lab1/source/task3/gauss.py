import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import emissions as em


def gauss_boxplot(mu, sigma):
    size1 = 20
    size2 = 100

    plt.figure()
    selection1 = norm.rvs(loc=mu, scale=sigma, size=size1)
    selection2 = norm.rvs(loc=mu, scale=sigma, size=size2)
    plt.boxplot([selection1, selection2], vert=False, labels=[size1, size2])
    plt.title(f"Стандартное нормальное распределение")
    plt.xlabel("Normal numbers")
    plt.ylabel("n")
    plt.show()


def gauss_experiment(mu, sigma):
    iterations = {20, 100}
    result = []
    for size in iterations:
        emissions_mean = 0
        for i in range(0, 1000):
            selection = norm.rvs(loc=mu, scale=sigma, size=size)
            emissions_mean += em.count_emissions(selection)
        emissions_mean /= (size * 1000)
        result = np.append(result, emissions_mean)
    return result


def gauss_calc(mu, sigma):
    Q1 = norm.ppf(0.25, loc=mu, scale=sigma)
    Q3 = norm.ppf(0.75, loc=mu, scale=sigma)
    X1 = Q1 - 1.5 * (Q3 - Q1)
    X2 = Q3 + 1.5 * (Q3 - Q1)
    result = norm.cdf(X1, mu, sigma) + (1 - norm.cdf(X2, mu, sigma))
    print("& %.3f & %.3f & %.3f & %.3f & %.3f " % (Q1, Q3, X1, X2, result))
    return result
