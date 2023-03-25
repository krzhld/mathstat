import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import emissions as em


def poisson_boxplot(mu):
    size1 = 20
    size2 = 100

    plt.figure()
    selection1 = poisson.rvs(mu=mu, size=size1)
    selection2 = poisson.rvs(mu=mu, size=size2)
    plt.boxplot([selection1, selection2], vert=False, labels=[size1, size2])
    plt.title(f"Распределение Пуассона (μ={mu})")
    plt.xlabel("Poisson numbers")
    plt.ylabel("n")
    plt.show()


def poisson_experiment(mu):
    iterations = {20, 100}
    result = []
    for size in iterations:
        emissions_mean = 0
        for i in range(0, 1000):
            selection = poisson.rvs(mu=mu, size=size)
            emissions_mean += em.count_emissions(selection)
        emissions_mean /= (size * 1000)
        result = np.append(result, emissions_mean)
    return result


def poisson_calc(mu):
    Q1 = poisson.ppf(0.25, mu=mu)
    Q3 = poisson.ppf(0.75, mu=mu)
    X1 = Q1 - 1.5 * (Q3 - Q1)
    X2 = Q3 + 1.5 * (Q3 - Q1)
    result = (poisson.cdf(X1, mu=mu) - poisson.pmf(X1, mu=mu)) + (1 - poisson.cdf(X2, mu=mu))
    print("& %.3f & %.3f & %.3f & %.3f & %.3f " % (Q1, Q3, X1, X2, result))
    return result
