import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import ecdf
import seaborn as sns


def gauss_ecdf(mu, sigma):
    sizes = {20, 60, 100}

    for size in sizes:
        selection = norm.rvs(loc=mu, scale=sigma, size=size)
        x, y = ecdf.ecdf(selection)
        plt.figure()
        plt.title(f"Э.ф.р для стандартного нормального р-я (выборка из {len(selection)} элементов)")
        plt.xlabel("x")
        plt.ylabel("F(x)")
        plt.plot(x, y, drawstyle='steps-post')

        x = np.linspace(norm.ppf(0.0001, loc=mu, scale=sigma), norm.ppf(0.9999, loc=mu, scale=sigma), 1000)
        y = norm.cdf(x, loc=mu, scale=sigma)
        plt.plot(x, y, 'r-')
        plt.xlim(-4, 4)
        plt.show()
        gauss_kde(mu, sigma, selection, size)


def gauss_kde(mu, sigma, selection, size):
    koef = {0.5, 1, 2}
    for cur_koef in koef:
        plt.figure()
        plt.title(f"Ядерная оценка плотности стандартного нормального р-я (n={size}, h={cur_koef}h_n)")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        sns.kdeplot(data=selection, bw_method='silverman', bw_adjust=cur_koef, fill=True, common_norm=False, alpha=0,
                    linewidth=2)
        x = np.linspace(-4, 4, 1000)
        plt.plot(x, norm.pdf(x, loc=mu, scale=sigma), 'r-')
        plt.xlim(-4, 4)
    plt.show()
