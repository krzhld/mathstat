import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import emissions as em


def cauchy_boxplot(a, b):
    size1 = 20
    size2 = 100

    plt.figure()
    selection1 = cauchy.rvs(loc=a, scale=b, size=size1)
    selection2 = cauchy.rvs(loc=a, scale=b, size=size2)
    plt.boxplot([selection1, selection2], vert=False, labels=[size1, size2])
    plt.title(f"Распределение Коши (a={a}, b={b})")
    plt.xlabel("Cauchy numbers")
    plt.ylabel("n")
    plt.show()


def cauchy_experiment(a, b):
    iterations = {20, 100}
    result = []
    for size in iterations:
        emissions_mean = 0
        for i in range(0, 1000):
            selection = cauchy.rvs(loc=a, scale=b, size=size)
            emissions_mean += em.count_emissions(selection)
        emissions_mean /= (size * 1000)
        result = np.append(result, emissions_mean)
    return result


def cauchy_calc(a, b):
    Q1 = cauchy.ppf(0.25, loc=a, scale=b)
    Q3 = cauchy.ppf(0.75, loc=a, scale=b)
    X1 = Q1 - 1.5 * (Q3 - Q1)
    X2 = Q3 + 1.5 * (Q3 - Q1)
    result = cauchy.cdf(X1, a, b) + (1 - cauchy.cdf(X2, a, b))
    print("& %.3f & %.3f & %.3f & %.3f & %.3f " % (Q1, Q3, X1, X2, result))
    return result
