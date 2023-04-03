import gauss as g
import cauchy as c
import uniform as u
import laplace as l
import poisson as p


g.gauss_boxplot(0, 1)
print("Gauss")
print(f"Experiment: {g.gauss_experiment(0, 1)}")
print(f"Theory: {g.gauss_calc(0, 1)}")
print()

c.cauchy_boxplot(0, 1)
print("Cauchy")
print(f"Experiment: {c.cauchy_experiment(0, 1)}")
print(f"Theory: {c.cauchy_calc(0, 1)}")
print()

u.uniform_boxplot(-(3 ** 0.5), 3 ** 0.5)
print("Uniform")
print(f"Experiment: {u.uniform_experiment(-(3 ** 0.5), 3 ** 0.5)}")
print(f"Theory: {u.uniform_calc(-(3 ** 0.5), 3 ** 0.5)}")
print()

l.laplace_boxplot(0, 1 / 2 ** 0.5)
print("Laplace")
print(f"Experiment: {l.laplace_experiment(0, 1 / 2 ** 0.5)}")
print(f"Theory: {l.laplace_calc(0, 1 / 2 ** 0.5)}")
print()

p.poisson_boxplot(10)
print("Poisson")
print(f"Experiment: {p.poisson_experiment(10)}")
print(f"Theory: {p.poisson_calc(10)}")
print()
