import numpy as np

import functions


class OptimizationMethod:
    def __init__(self, funcs=None, constraints=None):
        self.functions = [] if funcs is None else list(funcs)
        self.constraints = [] if constraints is None else list(constraints)

    def solve(self, *args, **kwargs):
        raise NotImplementedError

    def add_functions(self, *funcs):
        self.functions.extend(funcs)

    def add_constraint(self, *cons):
        self.constraints.extend(cons)

    def calculate_functions(self, u):
        return sum((f(u) for f in self.functions))

    def calculate_gradients(self, u):
        return sum((f.gradient(u) for f in self.functions))

    def calculate_hessians(self, u):
        return sum((f.hessian(u) for f in self.functions))


class GradientProjectionMethod(OptimizationMethod):
    """
        Ограничения считаются как ИЛИ
    """

    def __init__(self, alpha=0.0001, max_iter=1000, tol=1e-9, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.vals = []

    def solve(self, u_0):
        u_k = u_0.ravel()

        current = self.calculate_functions(u_k)

        self.vals = []

        for i in range(1, self.max_iter + 1):
            temp = u_k - self.alpha * self.calculate_gradients(u_k)
            projections = [con.projection(temp) for con in self.constraints]
            dists = np.array([np.linalg.norm(temp - p) for p in projections])
            u_k = projections[dists.argmin()]

            new = self.calculate_functions(u_k)
            if np.abs(new - current) < self.tol:
                break
            self.vals.append(new)
            current = new

        return u_k


class NewtonMethod(OptimizationMethod):
    def __init__(self, alpha=1, max_iter=100, sub_task_params=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.max_iter = max_iter
        self.sub_task_params = {} if sub_task_params is None else sub_task_params
        self.vals = []

    def solve(self, u_0, eps=0.01):
        u_k = u_0.ravel()

        self.vals = []

        for i in range(1, self.max_iter + 1):
            gradient = self.calculate_gradients(u_k)
            hessian = self.calculate_hessians(u_k)
            method = GradientProjectionMethod(
                **self.sub_task_params,
                funcs=[
                    functions.Dot(c=gradient - 0.5 * ((hessian.T @ u_k[:, None]).ravel() + u_k)),
                    functions.DotSpecial(a=hessian, alpha=0.5)
                ],
                constraints=self.constraints
            )

            result = method.solve(u_k + np.random.uniform(-eps, eps, u_k.shape[0]))

            u_k += self.alpha * (result - u_k)

            self.vals.append(self.calculate_functions(u_k))

        return u_k


class CanonicalSimplexMethod(OptimizationMethod):
    """
    <c, u> ---> inf
    u >= 0
    Au = b
    """

    def __init__(self, a, b, c):
        super().__init__()
        self.a = np.array(a)
        self.b = np.array(b.ravel())
        self.c = np.array(c.ravel())

    @staticmethod
    def _solve_tab(tab, basis):
        deltas = tab[-1, 1:]
        a = tab[:-1, 1:]

        r_row = np.arange(a.shape[1])
        r_col = np.arange(a.shape[0])

        while True:
            if deltas.max() <= 0:
                break

            not_basis = ~basis

            basis_locations_save = a[:, basis].argmax(axis=0)

            not_basis_and_pos = not_basis & (deltas > 0)

            current = a[:, not_basis_and_pos]
            not_pos = current <= 0
            not_pos_column = not_pos.all(axis=0)

            if not_pos_column.any():
                raise ValueError('-inf')

            pos_column = r_row[not_basis_and_pos][
                deltas[not_basis_and_pos][~not_pos_column].argmax()
            ]

            temp1 = np.array(a[:, pos_column])
            temp2 = np.array(tab[:-1, 0])
            mask = temp1 > 0
            temp2[mask] /= temp1[mask]
            pos_row = r_col[mask][temp2[mask].argmin()]

            main_row = np.array(tab[pos_row]) / a[pos_row, pos_column]
            tab -= main_row[None, :] * tab[:, pos_column + 1][:, None]
            tab[pos_row] = main_row

            basis_after_step = \
                np.isclose(a[:, basis].sum(axis=0), 1, atol=1e-10) & \
                np.isclose(a[:, basis].max(axis=0), 1.0, atol=1e-10) & \
                (a[:, basis].argmax(axis=0) == basis_locations_save)

            if not basis_after_step.all():
                basis[r_row[basis][basis_after_step.argmin()]] = False
                basis[pos_column] = True

        result = np.zeros(basis.shape[0], dtype=float)
        result[basis] = tab[:-1, 0][a[:, basis].argmax(axis=0)]

        return result

    def solve(self):
        mask = self.b < 0
        self.a[mask, :] *= -1
        self.b[mask] *= -1

        a = np.hstack([self.a, np.eye(self.a.shape[0])])
        M = max(
            np.abs(self.a).max(),
            np.abs(self.b).max(),
            np.abs(self.c).max()
        ) * self.a.shape[0] * self.a.shape[1] * 1e5
        c = np.hstack([self.c, np.full(self.a.shape[0], M)])

        basis = np.zeros(a.shape[1], dtype=bool)
        basis[-a.shape[0]:] = True

        tab = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
        tab[:-1, 1:] = a
        tab[:-1, 0] = self.b
        tab[-1, 1:] = np.dot(c[basis], a).ravel() - c
        tab[-1, 0] = np.dot(c[basis], self.b)

        result = CanonicalSimplexMethod._solve_tab(tab, basis)

        return result[:-self.a.shape[0]]


class SimplexMethod(CanonicalSimplexMethod):
    """
    <c, u> ---> inf

    u >= 0
    A_eq @ u = b_eq
    A_low @ u <= b_low
    A_high @ u >= b_high
    """

    def __init__(self, c, a_eq=None, b_eq=None, a_low=None, b_low=None, a_high=None, b_high=None):
        a = None
        to_b = []

        if a_eq is not None and b_eq is not None:
            a = a_eq
            to_b.append(b_eq)

        if a_low is not None and b_low is not None:
            to_b.append(b_low)
            if a is None:
                a = np.hstack([a_low, np.eye(a_low.shape[0])])
            else:
                a = np.vstack([
                    np.hstack([a, np.zeros((a.shape[0], a.shape[0]), dtype=float)]),
                    np.hstack([a_low, np.eye(a_low.shape[0])])
                ])

        if a_high is not None and b_high is not None:
            to_b.append(b_high)
            if a is None:
                a = np.hstack([a_high, -np.eye(a_high.shape[0])])
            else:
                a = np.vstack([
                    np.hstack([a, np.zeros((a.shape[0], a.shape[0]), dtype=float)]),
                    np.hstack([a_high, -np.eye(a_high.shape[0])])
                ])

        if not to_b:
            raise ValueError

        b = np.hstack(to_b)
        c_ = np.zeros(a.shape[1], dtype=float)
        c_[:c.shape[0]] = c

        self.shape = c.shape[0]

        super().__init__(a, b, c_)

    def solve(self):
        result = super().solve()
        return result[:self.shape]
