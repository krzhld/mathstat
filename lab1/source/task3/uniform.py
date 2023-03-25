import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt
import emissions as em


def uniform_boxplot(a, b):
    start_point = a
    length = b - a
    size1 = 20
    size2 = 100

    plt.figure()
    selection1 = uniform.rvs(loc=start_point, scale=length, size=size1)
    selection2 = uniform.rvs(loc=start_point, scale=length, size=size2)
    plt.boxplot([selection1, selection2], vert=False, labels=[size1, size2])
    plt.title(f"Равномерное распределение (a={a}, b={b})")
    plt.xlabel("Uniform numbers")
    plt.ylabel("n")
    plt.show()


def uniform_experiment(a, b):
    start_point = a
    length = b - a
    iterations = {20, 100}
    result = []
    for size in iterations:
        emissions_mean = 0
        for i in range(0, 1000):
            selection = uniform.rvs(loc=start_point, scale=length, size=size)
            emissions_mean += em.count_emissions(selection)
        emissions_mean /= (size * 1000)
        result = np.append(result, emissions_mean)
    return result


def uniform_calc(a, b):
    start_point = a
    length = b - a
    Q1 = uniform.ppf(0.25, loc=start_point, scale=length)
    Q3 = uniform.ppf(0.75, loc=start_point, scale=length)
    X1 = Q1 - 1.5 * (Q3 - Q1)
    X2 = Q3 + 1.5 * (Q3 - Q1)
    result = uniform.cdf(X1, start_point, length) + (1 - uniform.cdf(X2, start_point, length))
    print("& %.3f & %.3f & %.3f & %.3f & %.3f " % (Q1, Q3, X1, X2, result))
    return result
