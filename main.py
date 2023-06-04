import numpy as np
from functions import SquaredNormOfLinear
from methods import GradientProjectionMethod
from constraints import SphereConstraint


a = np.array([
    [1, 2],
    [3, 4],
], dtype=float)
f = np.array([-5, 4], dtype=float)
u = np.array([-2, 4], dtype=float)
zero = np.zeros(2, dtype=float)

funcs = [
    SquaredNormOfLinear(a=a, f=f)
]

cons = [
    SphereConstraint(1, -4)
]

task = GradientProjectionMethod(max_iter=100, functions=funcs, constraints=cons)

res = task.solve(zero)

print(np.linalg.norm(res))
