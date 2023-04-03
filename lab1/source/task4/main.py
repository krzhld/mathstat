import gauss as g
import cauchy as c
import uniform as u
import laplace as l
import poisson as p


g.gauss_ecdf(0, 1)
c.cauchy_ecdf(0, 1)
u.uniform_ecdf(-(3 ** 0.5), 3 ** 0.5)
l.laplace_ecdf(0, 1 / 2 ** 0.5)
p.poisson_ecdf(10)
