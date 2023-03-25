import gauss as g
import cauchy as c
import uniform as u
import laplace as l
import poisson as p


def print_table(E, D, name):
    print(f"{name}:")
    print("E:")
    for temp in E:
        print(f"& %.3f " % temp, end='')
    print()
    print("D:")
    for temp in D:
        print(f"& %.3f " % temp, end='')
    print(end='\n\n')


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
