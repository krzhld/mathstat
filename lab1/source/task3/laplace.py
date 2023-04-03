import numpy as np
from scipy.stats import laplace
import matplotlib.pyplot as plt
import emissions as em


def laplace_boxplot(alpha, beta):
    size1 = 20
    size2 = 100

    plt.figure()
    selection1 = laplace.rvs(loc=alpha, scale=beta, size=size1)
    selection2 = laplace.rvs(loc=alpha, scale=beta, size=size2)
    plt.boxplot([selection1, selection2], vert=False, labels=[size1, size2])
    plt.title(f"Распределение Лапласа (a={alpha}, b={beta})")
    plt.xlabel("Laplace numbers")
    plt.ylabel("n")
    plt.show()


def laplace_experiment(alpha, beta):
    iterations = {20, 100}
    result = []
    for size in iterations:
        emissions_mean = 0
        for i in range(0, 1000):
            selection = laplace.rvs(loc=alpha, scale=beta, size=size)
            emissions_mean += em.count_emissions(selection)
        emissions_mean /= (size * 1000)
        result = np.append(result, emissions_mean)
    return result


def laplace_calc(alpha, beta):
    Q1 = laplace.ppf(0.25, loc=alpha, scale=beta)
    Q3 = laplace.ppf(0.75, loc=alpha, scale=beta)
    X1 = Q1 - 1.5 * (Q3 - Q1)
    X2 = Q3 + 1.5 * (Q3 - Q1)
    result = laplace.cdf(X1, alpha, beta) + (1 - laplace.cdf(X2, alpha, beta))
    print("& %.3f & %.3f & %.3f & %.3f & %.3f " % (Q1, Q3, X1, X2, result))
    return result
