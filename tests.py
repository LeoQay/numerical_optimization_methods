import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog

import methods
import functions
import constraints

import matplotlib.pyplot as plt


class TestProject(unittest.TestCase):
    pass


class TestMethod(TestProject):
    pass


class TestProjectionMethod(TestMethod):
    def test_1(self):
        rng = np.random.default_rng(10000)
        n = 10

        func_alpha = rng.uniform(1, 2)
        func_c = rng.uniform(-10, 10, n)
        funcs = [functions.SquaredDot(alpha=func_alpha, c=func_c)]

        cons_f = rng.uniform(-10, 10, n)
        cons = [constraints.SphereConstraint(1, -0.0001, f=cons_f)]

        method = methods.GradientProjectionMethod(
            alpha=0.00001, max_iter=10000,
            funcs=funcs, constraints=cons
        )

        start = rng.uniform(-0.01, 0.01, n)

        my_res = method.solve(start)

        plt.figure(figsize=(7, 4))
        plt.plot(range(len(method.vals)), method.vals)
        plt.grid(True)
        plt.show()

        res = minimize(
            lambda x: funcs[0](x),
            start,
            jac=lambda x: funcs[0].gradient(x),
            constraints={
                'type': 'eq',
                'fun': lambda x: cons[0](x),
                'jac': lambda x: cons[0].gradient(x)
            },
            options={'maxiter': 10000}
        )

        my_answer = funcs[0](my_res)
        scipy_answer = funcs[0](res.x)
        print(my_answer, scipy_answer)

    def test_2(self):
        rng = np.random.default_rng(100)
        n = 100

        func_alpha = rng.uniform(1, 2)
        func_c = rng.uniform(-10, 10, n)
        funcs = [functions.SquaredDot(alpha=func_alpha, c=func_c)]

        con_alpha = rng.uniform(1, 2)
        con_beta = rng.uniform(-10, 10)
        con_c = rng.uniform(-10, 10, n)
        cons = [constraints.DotEqualConstraint(alpha=con_alpha, beta=con_beta, c=con_c)]

        method = methods.GradientProjectionMethod(
            alpha=0.00001, max_iter=10000,
            funcs=funcs, constraints=cons
        )

        start = rng.uniform(-0.01, 0.01, n)

        my_res = method.solve(start)

        plt.figure(figsize=(7, 4))
        plt.plot(range(len(method.vals)), method.vals)
        plt.grid(True)
        plt.show()

        res = minimize(
            lambda x: funcs[0](x),
            start,
            jac=lambda x: funcs[0].gradient(x),
            constraints={
                'type': 'eq',
                'fun': lambda x: cons[0](x),
                'jac': lambda x: cons[0].gradient(x)
            },
            options={'maxiter': 10000}
        )

        my_answer = funcs[0](my_res)
        scipy_answer = funcs[0](res.x)
        print(my_answer, scipy_answer)

    def test_3(self):
        rng = np.random.default_rng(1090)
        n = 100

        a = rng.uniform(-1, 1, (n, n))
        f = rng.uniform(-1, 1, n)
        funcs = [functions.SquaredNormOfLinear(a=a, f=f)]
        cons = [constraints.SphereConstraint(1, -4)]

        method = methods.GradientProjectionMethod(
            alpha=0.001, max_iter=10000, tol=0,
            funcs=funcs, constraints=cons
        )

        start = rng.uniform(-10, 10, n)

        my_res = method.solve(start)

        plt.figure(figsize=(7, 4))
        plt.plot(range(len(method.vals)), method.vals)
        plt.grid(True)
        plt.show()

        res = minimize(
            lambda x: funcs[0](x),
            start,
            jac=lambda x: funcs[0].gradient(x),
            constraints={
                'type': 'eq',
                'fun': lambda x: cons[0](x),
                'jac': lambda x: cons[0].gradient(x)
            },
            options={'maxiter': 10000}
        )

        my_answer = funcs[0](my_res)
        scipy_answer = funcs[0](res.x)
        print(my_answer, scipy_answer)

    def test_4(self):
        rng = np.random.default_rng(1055)
        n = 10

        a = rng.uniform(-1, 1, (n, n))
        f = rng.uniform(-1, 1, n)
        funcs = [functions.NormOfLinear(a=a, f=f)]
        cons = [constraints.SphereConstraint(1, -4)]

        method = methods.GradientProjectionMethod(
            alpha=0.001, max_iter=15000, tol=0,
            funcs=funcs, constraints=cons
        )

        start = rng.uniform(-10, 10, n)

        my_res = method.solve(start)

        plt.figure(figsize=(7, 4))
        plt.plot(range(len(method.vals)), method.vals)
        plt.grid(True)
        plt.show()

        res = minimize(
            lambda x: funcs[0](x),
            start,
            jac=lambda x: funcs[0].gradient(x),
            constraints={
                'type': 'eq',
                'fun': lambda x: cons[0](x),
                'jac': lambda x: cons[0].gradient(x)
            },
            options={'maxiter': 10000}
        )

        my_answer = funcs[0](my_res)
        scipy_answer = funcs[0](res.x)
        print(my_answer, scipy_answer)


class TestNewtonMethod(TestMethod):
    def test_1(self):
        rng = np.random.default_rng(10000)
        n = 100

        func_alpha = rng.uniform(1, 2)
        func_c = rng.uniform(-10, 10, n)
        funcs = [functions.SquaredDot(alpha=func_alpha, c=func_c)]

        cons_f = rng.uniform(4, 10, n)
        cons = [constraints.BallConstraint(1, -4, f=cons_f)]

        method = methods.NewtonMethod(
            alpha=1,
            sub_task_params={
                'max_iter': 10,
                'alpha': 0.001,
                'tol': 0.0
            },
            max_iter=100,
            funcs=funcs, constraints=cons
        )

        start = cons_f + rng.uniform(-0.01, 0.01, n)

        my_res = method.solve(start)

        plt.figure(figsize=(7, 4))
        plt.plot(range(len(method.vals)), method.vals)
        plt.grid(True)
        plt.show()

        res = minimize(
            lambda x: funcs[0](x),
            start,
            jac=lambda x: funcs[0].gradient(x),
            constraints={
                'type': 'ineq',
                'fun': lambda x: -cons[0](x),
                'jac': lambda x: -cons[0].gradient(x)
            },
            options={'maxiter': 10000}
        )

        my_answer = funcs[0](my_res)
        scipy_answer = funcs[0](res.x)
        print(np.linalg.norm(my_res - cons_f), np.linalg.norm(res.x - cons_f))
        print(my_answer, scipy_answer)

    def test_2(self):
        rng = np.random.default_rng(10000)
        n = 100

        func_alpha = rng.uniform(1, 2)
        func_a = rng.uniform(-10, 10, (n, n))
        func_f = rng.uniform(-10, 10, n)
        funcs = [functions.NormOfLinear(a=func_a, f=func_f, alpha=func_alpha)]

        cons_f = rng.uniform(4, 10, n)
        cons = [constraints.SphereConstraint(1, -4, f=cons_f)]

        method = methods.NewtonMethod(
            alpha=1,
            sub_task_params={
                'max_iter': 20,
                'alpha': 0.01,
                'tol': 0.0
            },
            max_iter=100,
            funcs=funcs, constraints=cons
        )

        start = cons_f + rng.uniform(-0.01, 0.01, n)

        my_res = method.solve(start)

        plt.figure(figsize=(7, 4))
        plt.plot(range(len(method.vals)), method.vals)
        plt.grid(True)
        plt.show()

        res = minimize(
            lambda x: funcs[0](x),
            start,
            jac=lambda x: funcs[0].gradient(x),
            constraints={
                'type': 'eq',
                'fun': lambda x: cons[0](x),
                'jac': lambda x: cons[0].gradient(x)
            },
            options={'maxiter': 10000}
        )

        my_answer = funcs[0](my_res)
        scipy_answer = funcs[0](res.x)
        print(np.linalg.norm(my_res - cons_f), np.linalg.norm(res.x - cons_f))
        print(my_answer, scipy_answer)


class TestFunction(TestProject):
    @staticmethod
    def check_func_val_grad_hess(func, point, val, grad, hess):
        point = np.array(point)
        assert np.isclose(func(point), np.array(val, dtype=float), atol=1e-10)
        assert np.isclose(func.gradient(point), np.array(grad, dtype=float), atol=1e-10)
        assert np.isclose(func.hessian(point), np.array(hess, dtype=float), atol=1e-10)


class TestNormOfLinear(TestFunction):
    def test_1(self):
        self.check_func_val_grad_hess(
            functions.NormOfLinear(),
            [0], 0, [0], [[0]]
        )

    def test_2(self):
        self.check_func_val_grad_hess(
            functions.NormOfLinear(),
            [1], 1, [1], [[0]]
        )


class TestSimplexMethod(TestMethod):
    def test_1(self):
        a = np.array([
            [1.0, 1.0, 3.0, 1.0],
            [1.0, -1, 1, 2]
        ])
        b = np.array([3, 1.0])
        c = np.array([2.0, -1, 2, 3])
        method = methods.CanonicalSimplexMethod(a=a, b=b, c=c)
        result = method.solve()

        print(result)

        assert (result == np.array([0.0, 0.0, 1.0, 0.0])).all()

    def test_2(self):
        a = np.array([
            [2, 3, 6],
            [4, 2, 4],
            [4, 6, 8]
        ], dtype=float)
        b = np.array([240, 200, 160], dtype=float)
        c = -np.array([4, 5, 4], dtype=float)

        res = linprog(c, A_ub=a, b_ub=b)

        print(res.x)
        print(np.dot(res.x, c))
        print((res.x >= -1e-10).all())
        print(np.isclose((a @ res.x), b, atol=1e10).all())

        method = methods.SimplexMethod(c=c, a_low=a, b_low=b)
        result = method.solve()

        print(result)
        print(np.dot(result, c))
        print((result >= -1e-10).all())
        print(np.isclose((a @ result), b, atol=1e10).all())

        assert np.isclose(result, [40, 0, 0], atol=1e-10).all()

    def test_3(self):
        rng = np.random.default_rng(1000)
        n = 100
        m = 150
        a = rng.uniform(3, 30, (n, m))
        b = rng.uniform(20, 25, n)
        c = -rng.uniform(1, 2, m)

        res = linprog(c, A_ub=a, b_ub=b)

        print(np.dot(res.x, c))
        print((res.x >= -1e-10).all())
        print(np.isclose((a @ res.x), b, atol=1e10).all())

        print(((a @ res.x) <= b).all())

        method = methods.SimplexMethod(c=c, a_low=a, b_low=b)
        result = method.solve()

        print(np.dot(result, c))
        print((result >= -1e-10).all())
        print(np.isclose((a @ result), b, atol=1e10).all())

    def test_4(self):
        a = -np.array([
            [0, 1, 1],
            [2, 1, 2],
            [2, -1, 2]
        ], dtype=float)
        b = -np.array([4, 6, 2], dtype=float)
        c = np.array([3, 2, 1], dtype=float)

        res = linprog(c, A_eq=a, b_eq=b)

        print(res.x)

        print(np.dot(res.x, c))
        print((res.x >= -1e-10).all())
        print(np.isclose((a @ res.x), b, atol=1e10).all())

        method = methods.CanonicalSimplexMethod(a=a, b=b, c=c)
        result = method.solve()

        print(result)

        print(np.dot(result, c))
        print((result >= -1e-10).all())
        print(np.isclose((a @ result), b, atol=1e10).all())


if __name__ == '__main__':
    unittest.main()
