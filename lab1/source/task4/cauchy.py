import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import ecdf
import seaborn as sns


def cauchy_ecdf(a, b):
    sizes = {20, 60, 100}

    for size in sizes:
        selection = cauchy.rvs(loc=a, scale=b, size=size)
        x, y = ecdf.ecdf(selection)
        plt.figure()
        plt.title(f"Э.ф.р для р-я Коши (a={a}, b={b}; выборка из {len(selection)} элементов)")
        plt.xlabel("x")
        plt.ylabel("F(x)")
        plt.plot(x, y, drawstyle='steps-post')

        x = np.linspace(cauchy.ppf(0.01, loc=a, scale=b), cauchy.ppf(0.99, loc=a, scale=b), 1000)
        y = cauchy.cdf(x, loc=a, scale=b)
        plt.plot(x, y, 'r-')
        plt.xlim(-4, 4)
        plt.show()
        cauchy_kde(a, b, selection, size)


def cauchy_kde(a, b, selection, size):
    koef = {0.5, 1, 2}
    for cur_koef in koef:
        plt.figure()
        plt.title(f"Ядерная оценка плотности р-я Коши (a={a}, b={b}, n={size}, h={cur_koef}h_n)")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        sns.kdeplot(data=selection, bw_method='silverman', bw_adjust=cur_koef, fill=True, common_norm=False, alpha=0,
                    linewidth=2)
        x = np.linspace(-4, 4, 1000)
        plt.plot(x, cauchy.pdf(x, loc=a, scale=b), 'r-')
        plt.xlim(-4, 4)
    plt.show()
