import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import path
import scipy.optimize as opt
from scipy.optimize import linprog


def minimization(A, y, eps):
    [m, n] = A.shape

    c = np.concatenate((np.zeros((n, 1)), np.ones((m, 1))), axis=0)
    c = np.ravel(c)

    diag = np.diag(np.full(m, -eps))

    M_1 = np.concatenate((-A, diag), axis=1)
    M_2 = np.concatenate((A, diag), axis=1)
    M = np.concatenate((M_1, M_2), axis=0)

    v = np.concatenate((-y, y), axis=0)

    l_b = np.concatenate((np.full(n, None), np.full(m, 1)), axis=0)
    u_b = np.full(n + m, None)

    bounds = [(l_b[i], u_b[i]) for i in range(len(l_b))]

    opt = linprog(c=c, A_ub=M, b_ub=v, bounds=bounds)
    y = opt.x

    coefs = y[0:n]
    w = y[n:n+m]

    return [coefs, w]


def parser():
    data1 = pd.read_csv('data/Channel_1_700nm_2mm.csv', sep=';', encoding='cp1251')
    data2 = pd.read_csv('data/Channel_2_700nm_2mm.csv', sep=';', encoding='cp1251')

    data1_mv = np.ravel(data1.drop('мА', axis=1).to_numpy())
    data2_mv = np.ravel(data2.drop('мА', axis=1).to_numpy())

    data1_n = np.arange(1, len(data1_mv) + 1, 1)
    data2_n = np.arange(1, len(data2_mv) + 1, 1)

    data1_eps = 1e-4
    data2_eps = 1e-4

    data1_X = np.stack((np.ones(len(data1_mv)), data1_n))
    data1_X = np.transpose(data1_X)
    [data1_tau, data1_w] = minimization(data1_X, data1_mv, data1_eps)

    data2_X = np.stack((np.ones(len(data2_mv)), data2_n))
    data2_X = np.transpose(data2_X)
    [data2_tau, data2_w] = minimization(data2_X, data2_mv, data2_eps)

    with open('data/Ch1.txt', 'w') as f:
        print(f'{data1_tau[0]} {data1_tau[1]}', file=f)
        for temp in data1_w:
            print(temp, file=f)

    with open('data/Ch2.txt', 'w') as f:
        print(f'{data2_tau[0]} {data2_tau[1]}', file=f)
        for temp in data2_w:
            print(temp, file=f)


data1_fixed_int = []
data2_fixed_int = []


def count_Jaccar(R):
    data1_new = [[data1_fixed_int[i][0] * R, data1_fixed_int[i][1] * R] for i in range(len(data1_fixed_int))]
    all_data = data1_new + data2_fixed_int
    min_inc = list(all_data[0])
    max_inc = list(all_data[0])
    for interval in all_data:
        min_inc[0] = max(min_inc[0], interval[0])
        min_inc[1] = min(min_inc[1], interval[1])
        max_inc[0] = min(max_inc[0], interval[0])
        max_inc[1] = max(max_inc[1], interval[1])
    JK = (min_inc[1] - min_inc[0]) / (max_inc[1] - max_inc[0])
    return JK


def load_processed(filename):
    A = 0
    B = 0
    w = []
    with open(filename) as f:
        A, B = [float(t) for t in f.readline().split()]
        for line in f.readlines():
            w.append(float(line))
    return A, B, w


if __name__ == '__main__':
    data1 = pd.read_csv('data/Channel_1_700nm_2mm.csv', sep=';', encoding='cp1251')
    data2 = pd.read_csv('data/Channel_2_700nm_2mm.csv', sep=';', encoding='cp1251')

    data1 = data1['мВ']
    data2 = data2['мВ']

    data1.plot(color='steelblue')
    plt.title('Experiment data: channel 1 700nm 2mm')
    plt.xlabel('n')
    plt.ylabel('мВ')
    plt.savefig(path.PATH + 'report/resources/input_PR1.png', dpi=1000)
    plt.close()
    data2.plot(color='yellowgreen')
    plt.title('Experiment data: channel 2 700nm 2mm')
    plt.xlabel('n')
    plt.ylabel('мВ')
    plt.savefig(path.PATH + 'report/resources/input_PR2.png', dpi=1000)
    plt.close()

    eps = 1e-4
    plt.fill_between(data1.index + 1, data1 - eps, data1 + eps, color='steelblue')
    plt.xlabel('n')
    plt.ylabel('мВ')
    plt.title('Interval data: channel 1 700nm 2mm')
    plt.savefig(path.PATH + 'report/resources/intervals_PR1.png', dpi=1000)
    plt.close()
    plt.fill_between(data2.index + 1, data2 - eps, data2 + eps, color='yellowgreen')
    plt.xlabel('n')
    plt.ylabel('мВ')
    plt.title('Interval data: channel 2 700nm 2mm')
    plt.savefig(path.PATH + 'report/resources/intervals_PR2.png', dpi=1000)
    plt.close()

    parser()
    A1, B1, w1 = load_processed('data/Ch1.txt')
    A2, B2, w2 = load_processed('data/Ch2.txt')

    for i in data1.index:
        plt.vlines(i + 1, data1[i] + w1[i] * eps, data1[i] - w1[i] * eps, color='steelblue')
    plt.plot(np.arange(1, 201), A1 + B1 * (np.arange(1, 201)), label='lsm', color='yellowgreen')
    plt.xlabel('n')
    plt.ylabel('мВ')
    plt.title('Interval data ch 1')
    plt.savefig(path.PATH + 'report/resources/lr_PR1.png', dpi=1000)
    plt.close()
    for i in data2.index:
        plt.vlines(i + 1, data2[i] + w2[i] * eps, data2[i] - w2[i] * eps, color='yellowgreen')
    plt.plot(np.arange(1, 201), A2 + B2 * (np.arange(1, 201)), label='lsm', color='steelblue')
    plt.xlabel('n')
    plt.ylabel('мВ')
    plt.title('Interval data ch 2')
    plt.savefig(path.PATH + 'report/resources/lr_PR2.png', dpi=1000)
    plt.close()

    plt.hist(w1, color='steelblue')
    plt.title('w1 hist')
    plt.savefig(path.PATH + 'report/resources/whyst_PR1.png', dpi=1000)
    plt.close()
    plt.hist(w2, color='yellowgreen')
    plt.title('w2 hist')
    plt.savefig(path.PATH + 'report/resources/whyst_PR2.png', dpi=1000)
    plt.close()

    data1_fixed = [y - (i + 1) * B1 for i, y in enumerate(data1)]
    for i in data1.index:
        plt.vlines(i + 1, data1_fixed[i] + w1[i] * eps, data1_fixed[i] - w1[i] * eps, color='steelblue')
    plt.plot(np.arange(1, 201), [A1] * 200, label='lsm', color='yellowgreen')
    plt.xlabel('n')
    plt.ylabel('мВ')
    plt.title('Data without linear drifting ch 1')
    plt.savefig(path.PATH + 'report/resources/fixed_PR1.png', dpi=1000)
    plt.close()
    data2_fixed = [y - (i + 1) * B2 for i, y in enumerate(data2)]
    for i in data2.index:
        plt.vlines(i + 1, data2_fixed[i] + w2[i] * eps, data2_fixed[i] - w2[i] * eps, color='yellowgreen')
    plt.plot(np.arange(1, 201), [A2] * 200, label='lsm', color='steelblue')
    plt.xlabel('n')
    plt.ylabel('мВ')
    plt.title('Data without linear drifting ch 2')
    plt.savefig(path.PATH + 'report/resources/fixed_PR2.png', dpi=1000)
    plt.close()

    plt.hist(data1_fixed, color='steelblue')
    plt.title('l1 hist')
    plt.savefig(path.PATH + 'report/resources/fhyst_PR1.png', dpi=1000)
    plt.close()
    plt.hist(data2_fixed, color='yellowgreen')
    plt.title('l2 hist')
    plt.savefig(path.PATH + 'report/resources/fhyst_PR2.png', dpi=1000)
    plt.close()


    data1_fixed_int = [[y - w1[i] * eps, y + w1[i] * eps] for i, y in enumerate(data1_fixed)]
    data2_fixed_int = [[y - w2[i] * eps, y + w2[i] * eps] for i, y in enumerate(data2_fixed)]


    R_interval = np.linspace(1, 1.1, 1000)
    Jaccars = []
    for R in R_interval:
        Jaccars.append(count_Jaccar(R))

    optimal_x = opt.fmin(lambda x: -count_Jaccar(x), 1.068531955)
    min1 = opt.root(count_Jaccar, 1.068531955 + eps)
    max1 = opt.root(count_Jaccar, 1.068531955 - eps)

    plt.plot(R_interval, Jaccars, label="Jaccard", zorder=1)
    plt.scatter(optimal_x[0], count_Jaccar(optimal_x[0]), label=f"optimal point at R={round(optimal_x[0], 9)}",
                color="r")
    plt.scatter(min1.x, count_Jaccar(min1.x), label=f"$min_R$={str(min1.x[0])}", color="b", zorder=2)
    plt.scatter(max1.x, count_Jaccar(max1.x), label=f"$max_R$={str(max1.x[0])}", color="b", zorder=2)
    plt.legend()
    plt.xlabel('$R_{21}$')
    plt.ylabel('Jaccard')
    plt.title('Jaccard vs R')
    plt.savefig(path.PATH + 'report/resources/jakkar.png', dpi=1000)
    plt.close()

    data1_new = [[data1_fixed_int[i][0] * optimal_x[0], data1_fixed_int[i][1] * optimal_x[0]] for i in
                 range(len(data1_fixed_int))]
    all_data = data1_new + data2_fixed_int
    plt.hist([(inter[0] + inter[1]) / 2 for inter in all_data], label="Combined with optimal R")
    plt.legend()
    plt.title('Histogram of combined data with optimal R21')
    plt.savefig(path.PATH + 'report/resources/jakkar_combined_hist.png', dpi=1000)
    plt.close()
