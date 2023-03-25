import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import ecdf
import seaborn as sns


def poisson_ecdf(mu):
    sizes = {20, 60, 100}

    for size in sizes:
        selection = poisson.rvs(mu=mu, size=size)
        x, y = ecdf.ecdf_pois(selection)
        plt.figure()
        plt.title(f"Э.ф.р для р-я Пуассона (μ={mu}) (выборка из {len(selection)} элементов)")
        plt.xlabel("x")
        plt.ylabel("F(x)")
        plt.plot(x, y, drawstyle='steps-post')

        x = np.arange(poisson.ppf(0.001, mu=mu), poisson.ppf(0.999, mu=mu))
        y = poisson.cdf(x, mu=mu)
        plt.plot(x, y, 'r-')
        plt.show()
        poisson_kde(mu, selection, size)


def poisson_kde(mu, selection, size):
    koef = {0.5, 1, 2}
    for cur_koef in koef:
        plt.figure()
        plt.title(f"Ядерная оценка плотности р-я Пуассона (μ={mu}, n={size}, h={cur_koef}h_n)")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        sns.kdeplot(data=selection, bw_method='silverman', bw_adjust=cur_koef, fill=True, common_norm=False,
                    alpha=0,
                    linewidth=2)
        x = np.arange(poisson.ppf(0.001, mu=mu), poisson.ppf(0.999, mu=mu))
        plt.plot(x, poisson.pmf(x, mu=mu), 'r-')
        plt.xlim(6, 14)
    plt.show()
