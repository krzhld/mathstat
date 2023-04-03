import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt
import ecdf
import seaborn as sns


def uniform_ecdf(a, b):
    start_point = a
    length = b - a
    sizes = {20, 60, 100}

    for size in sizes:
        selection = uniform.rvs(loc=start_point, scale=length, size=size)
        x, y = ecdf.ecdf(selection)
        plt.figure()
        plt.title(f"Э.ф.р для равномерного р-я (a={a}, b={b}; выборка из {len(selection)} элементов)")
        plt.xlabel("x")
        plt.ylabel("F(x)")
        plt.plot(x, y, drawstyle='steps-post')

        x = np.linspace(uniform.ppf(0.001, loc=start_point, scale=length), uniform.ppf(0.999, loc=start_point, scale=length), 1000)
        y = uniform.cdf(x, loc=start_point, scale=length)
        plt.plot(x, y, 'r-')
        plt.show()
        uniform_kde(a, b, selection, size)


def uniform_kde(a, b, selection, size):
    start_point = a
    length = b - a
    koef = {0.5, 1, 2}
    for cur_koef in koef:
        plt.figure()
        plt.title(f"Ядерная оценка плотности равномерного р-я (a={a}, b={b}, n={size}, h={cur_koef}h_n)")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        sns.kdeplot(data=selection, bw_method='silverman', bw_adjust=cur_koef, fill=True, common_norm=False,
                    alpha=0,
                    linewidth=2)
        x = np.linspace(-4, 4, 1000)
        plt.plot(x, uniform.pdf(x, loc=start_point, scale=length), 'r-')
        plt.xlim(-4, 4)
    plt.show()
