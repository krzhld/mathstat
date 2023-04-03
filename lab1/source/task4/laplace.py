import numpy as np
from scipy.stats import laplace
import matplotlib.pyplot as plt
import ecdf
import seaborn as sns


def laplace_ecdf(alpha, beta):
    sizes = {20, 60, 100}

    for size in sizes:
        selection = laplace.rvs(loc=alpha, scale=beta, size=size)
        x, y = ecdf.ecdf(selection)
        plt.figure()
        plt.title(f"Э.ф.р для р-я Лапласа (α={alpha}, β={beta}) (выборка из {len(selection)} элементов)")
        plt.xlabel("x")
        plt.ylabel("F(x)")
        plt.plot(x, y, drawstyle='steps-post')

        x = np.linspace(laplace.ppf(0.001, loc=alpha, scale=beta), laplace.ppf(0.999, loc=alpha, scale=beta), 1000)
        y = laplace.cdf(x, loc=alpha, scale=beta)
        plt.plot(x, y, 'r-')
        plt.xlim(-4, 4)
        plt.show()
        laplace_kde(alpha, beta, selection, size)


def laplace_kde(alpha, beta, selection, size):
    koef = {0.5, 1, 2}
    for cur_koef in koef:
        plt.figure()
        plt.title(f"Ядерная оценка плотности р-я Лапласа (α={alpha}, β={beta}, n={size}, h={cur_koef}h_n)")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        sns.kdeplot(data=selection, bw_method='silverman', bw_adjust=cur_koef, fill=True, common_norm=False,
                    alpha=0, linewidth=2)
        x = np.linspace(-4, 4, 1000)
        plt.plot(x, laplace.pdf(x, loc=alpha, scale=beta), 'r-')
        plt.xlim(-4, 4)
    plt.show()
