import gauss as g
import cauchy as c
import uniform as u
import laplace as l
import poisson as p
import numpy as np
import math as m


def print_table(E, D, name):
    print('\hline')
    k = 0
    print(name + ' n = 10 & $\overline{x} $ & $med\:x$ & $z_{R}$ & $z_{Q}$ & $z_{tr}$ \\\\')
    print('\hline\hline')
    print(f'$E(z)$ & {E[k + 0]:.3f} & {E[k + 1]:.3f} & {E[k + 2]:.3f} & {E[k + 3]:.3f} & {E[k + 4]:.3f} \\\\')
    print('\hline')
    print(f'$D(z)$ & {D[k + 0]:.3f} & {D[k + 1]:.3f} & {D[k + 2]:.3f} & {D[k + 3]:.3f} & {D[k + 4]:.3f}  \\\\')
    print('\hline')
    print('$E(z) \pm \sqrt{D(z)}$ & ' + f'[{(E[k + 0] - np.sqrt(D[k + 0])):.3f};{(E[k + 0] + np.sqrt(D[k + 0])):.3f}] & [{(E[k + 1] - np.sqrt(D[k + 1])):.3f};{(E[k + 1] + np.sqrt(D[k + 1])):.3f}] & [{(E[k + 2] - np.sqrt(D[k + 2])):.3f};{(E[k + 2] + np.sqrt(D[k + 2])):.3f}] & [{(E[k + 3] - np.sqrt(D[k + 3])):.3f};{(E[k + 3] + np.sqrt(D[k + 3])):.3f}] & [{(E[k + 4] - np.sqrt(D[k + 4])):.3f};{(E[k + 4] + np.sqrt(D[k + 4])):.3f}]   \\\\')
    print('\hline')
    print('$\hat{E}(z)$ ' + f' & {m.trunc(E[k + 0])} & {m.trunc(E[k + 1])} & {m.trunc(E[k + 2])} & {m.trunc(E[k + 3])} & {m.trunc(E[k + 4])}  \\\\')
    print('\hline\hline')
    k = 5
    print(name + ' n = 100 & $\overline{x} $ & $med\:x$ & $z_{R}$ & $z_{Q}$ & $z_{tr}$ \\\\')
    print('\hline\hline')
    print(f'$E(z)$ & {E[k + 0]:.3f} & {E[k + 1]:.3f} & {E[k + 2]:.3f} & {E[k + 3]:.3f} & {E[k + 4]:.3f} \\\\')
    print('\hline')
    print(f'$D(z)$ & {D[k + 0]:.3f} & {D[k + 1]:.3f} & {D[k + 2]:.3f} & {D[k + 3]:.3f} & {D[k + 4]:.3f}  \\\\')
    print('\hline')
    print(
        '$E(z) \pm \sqrt{D(z)}$ & ' + f'[{(E[k + 0] - np.sqrt(D[k + 0])):.3f};{(E[k + 0] + np.sqrt(D[k + 0])):.3f}] & [{(E[k + 1] - np.sqrt(D[k + 1])):.3f};{(E[k + 1] + np.sqrt(D[k + 1])):.3f}] & [{(E[k + 2] - np.sqrt(D[k + 2])):.3f};{(E[k + 2] + np.sqrt(D[k + 2])):.3f}] & [{(E[k + 3] - np.sqrt(D[k + 3])):.3f};{(E[k + 3] + np.sqrt(D[k + 3])):.3f}] & [{(E[k + 4] - np.sqrt(D[k + 4])):.3f};{(E[k + 4] + np.sqrt(D[k + 4])):.3f}]   \\\\')
    print('\hline')
    print(
        '$\hat{E}(z)$ ' + f' & {m.trunc(E[k + 0])} & {m.trunc(E[k + 1])} & {m.trunc(E[k + 2])} & {m.trunc(E[k + 3])} & {m.trunc(E[k + 4])}  \\\\')
    print('\hline\hline')
    k = 10
    print(name + ' n = 1000 & $\overline{x} $ & $med\:x$ & $z_{R}$ & $z_{Q}$ & $z_{tr}$ \\\\')
    print('\hline\hline')
    print(f'$E(z)$ & {E[k + 0]:.3f} & {E[k + 1]:.3f} & {E[k + 2]:.3f} & {E[k + 3]:.3f} & {E[k + 4]:.3f} \\\\')
    print('\hline')
    print(f'$D(z)$ & {D[k + 0]:.3f} & {D[k + 1]:.3f} & {D[k + 2]:.3f} & {D[k + 3]:.3f} & {D[k + 4]:.3f}  \\\\')
    print('\hline')
    print(
        '$E(z) \pm \sqrt{D(z)}$ & ' + f'[{(E[k + 0] - np.sqrt(D[k + 0])):.3f};{(E[k + 0] + np.sqrt(D[k + 0])):.3f}] & [{(E[k + 1] - np.sqrt(D[k + 1])):.3f};{(E[k + 1] + np.sqrt(D[k + 1])):.3f}] & [{(E[k + 2] - np.sqrt(D[k + 2])):.3f};{(E[k + 2] + np.sqrt(D[k + 2])):.3f}] & [{(E[k + 3] - np.sqrt(D[k + 3])):.3f};{(E[k + 3] + np.sqrt(D[k + 3])):.3f}] & [{(E[k + 4] - np.sqrt(D[k + 4])):.3f};{(E[k + 4] + np.sqrt(D[k + 4])):.3f}]   \\\\')
    print('\hline')
    print(
        '$\hat{E}(z)$ ' + f' & {m.trunc(E[k + 0])} & {m.trunc(E[k + 1])} & {m.trunc(E[k + 2])} & {m.trunc(E[k + 3])} & {m.trunc(E[k + 4])}  \\\\')
    print('\hline')
    print('\n\n\n')


E, D = g.gauss_char(0, 1)
print_table(E, D, 'Gauss')
E, D = c.cauchy_char(0, 1)
print_table(E, D, 'Cauchy')
E, D = u.uniform_char(-(3 ** 0.5), 3 ** 0.5)
print_table(E, D, 'Uniform')
E, D = l.laplace_char(0, 1 / 2 ** 0.5)
print_table(E, D, 'Laplace')
E, D = p.poisson_char(10)
print_table(E, D, 'Poisson')
