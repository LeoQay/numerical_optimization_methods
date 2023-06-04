import numpy as np


class Function:
    def __init__(self):
        pass

    def __call__(self, u):
        raise NotImplementedError

    def gradient(self, u):
        raise NotImplementedError

    def hessian(self, u):
        raise NotImplementedError


class SquaredNormOfLinear(Function):
    """
    Func(u) = alpha * ||Au - f||^2

    Grad(u) = 2 * alpha * A^T (Au - f)

    Default values:
        alpha = 1
        A = I
        f = 0
    """

    def __init__(self, a=None, f=None, alpha=None):
        super().__init__()

        self.a = a
        self.f = f if f is None else f.reshape(-1, 1)
        self.alpha = alpha

        if self.a is None and self.f is None:
            self.func = self._all_is_none
            self.grad = self._grad_all_is_none
            self.hess = self._hessian_n_unknown
        elif self.f is None:
            self.func = self._f_is_none
            self.grad = self._grad_f_is_none
            self.hess = self._hessian_n_known
            if self.alpha is None:
                self._hessian = self.a.T @ self.a
            else:
                self._hessian = self.alpha * (self.a.T @ self.a)
        elif self.a is None:
            self.func = self._a_is_none
            self.grad = self._grad_a_is_none
            self.hess = self._hessian_n_known
            if self.alpha is None:
                self._hessian = np.eye(self.f.shape[0])
            else:
                self._hessian = self.alpha * np.eye(self.f.shape[0])
        else:
            self.func = self._all_known
            self.grad = self._grad_all_known
            self.hess = self._hessian_n_known
            if self.alpha is None:
                self._hessian = self.a.T @ self.a
            else:
                self._hessian = self.alpha * (self.a.T @ self.a)

    def __call__(self, u):
        res = self.func(u.reshape(-1, 1)).ravel()
        return res if self.alpha is None else self.alpha * res

    def gradient(self, u):
        res = self.grad(u.reshape(-1, 1)).ravel()
        return res if self.alpha is None else self.alpha * res

    def hessian(self, u):
        return self.hess(u.ravel())

    def _hessian_n_known(self, u):
        return self._hessian

    def _hessian_n_unknown(self, u):
        if self.alpha is None:
            self._hessian = np.eye(u.shape[0])
        else:
            self._hessian = self.alpha * np.eye(u.shape[0])
        self.hess = self._hessian_n_known
        return self._hessian

    def _all_known(self, u):
        return np.linalg.norm(self.a @ u - self.f) ** 2

    def _grad_all_known(self, u):
        return 2 * (self.a.T @ (self.a @ u - self.f))

    def _f_is_none(self, u):
        return np.linalg.norm(self.a @ u) ** 2

    def _grad_f_is_none(self, u):
        return 2 * (self.a.T @ (self.a @ u))

    def _a_is_none(self, u):
        return np.linalg.norm(u - self.f) ** 2

    def _grad_a_is_none(self, u):
        return 2 * (u - self.f)

    def _all_is_none(self, u):
        return np.linalg.norm(u) ** 2

    def _grad_all_is_none(self, u):
        return 2 * u


class NormOfLinear(Function):
    """
    Func(u) = alpha * ||Au - f||

    Grad(u) = ( alpha / ||Au - f|| ) * A^T (Au - f)

    Default values:
        alpha = 1
        A = I
        f = 0
    """

    def __init__(self, a=None, f=None, alpha=None):
        super().__init__()

        self.a = a
        self.f = f if f is None else f.reshape(-1, 1)
        self.alpha = alpha

        if self.a is None and self.f is None:
            self.func = self._all_is_none
            self.grad = self._grad_all_is_none
            self.hess = self._hessian_all_is_none
        elif self.f is None:
            self.func = self._f_is_none
            self.grad = self._grad_f_is_none
            self.hess = self._hessian_f_is_none
        elif self.a is None:
            self.func = self._a_is_none
            self.grad = self._grad_a_is_none
            self.hess = self._hessian_a_is_none
        else:
            self.func = self._all_known
            self.grad = self._grad_all_known
            self.hess = self._hessian_all_known

    def __call__(self, u):
        res = self.func(u.reshape(-1, 1)).ravel()
        return res if self.alpha is None else self.alpha * res

    def gradient(self, u):
        res = self.grad(u.reshape(-1, 1)).ravel()
        return res if self.alpha is None else self.alpha * res

    def hessian(self, u):
        res = self.hess(u.ravel())
        if self.alpha is None:
            return res
        return self.alpha * res

    def _all_known(self, u):
        return np.linalg.norm(self.a @ u - self.f)

    def _grad_all_known(self, u):
        temp = self.a @ u - self.f
        norm = np.linalg.norm(temp)
        if np.isclose(norm, 0, atol=1e-10):
            return np.zeros(u.shape[0])
        return self.a.T @ temp / norm

    def _hessian_all_known(self, u):
        temp1 = self.a.T @ self.a
        temp2 = self.a @ u - self.f
        temp3 = self.a.T @ temp2
        norm = np.linalg.norm(temp2)
        if np.isclose(norm, 0, atol=1e-10):
            return np.zeros((u.shape[0], u.shape[0]))
        return temp1 / (norm ** 2) - (temp3 @ temp3.T) / (norm ** 3)

    def _f_is_none(self, u):
        return np.linalg.norm(self.a @ u)

    def _grad_f_is_none(self, u):
        temp = self.a @ u
        norm = np.linalg.norm(temp)
        if np.isclose(norm, 0, atol=1e-10):
            return np.zeros(u.shape[0])
        return self.a.T @ temp / norm

    def _hessian_f_is_none(self, u):
        temp1 = self.a.T @ self.a
        temp2 = temp1 @ u
        norm = np.linalg.norm(self.a @ u)
        if np.isclose(norm, 0, atol=1e-10):
            return np.zeros((u.shape[0], u.shape[0]))
        return temp1 / (norm ** 2) - (temp2 @ temp2.T) / (norm ** 3)

    def _a_is_none(self, u):
        return np.linalg.norm(u - self.f)

    def _grad_a_is_none(self, u):
        temp = u - self.f
        norm = np.linalg.norm(temp)
        if np.isclose(norm, 0, atol=1e-10):
            return np.zeros(u.shape[0])
        return temp / norm

    def _hessian_a_is_none(self, u):
        temp = u - self.f
        norm = np.linalg.norm(temp)
        if np.isclose(norm, 0, atol=1e-10):
            return np.zeros((u.shape[0], u.shape[0]))
        return np.eye(u.shape[0]) / (norm ** 2) - (temp @ temp.T) / (norm ** 3)

    def _all_is_none(self, u):
        return np.linalg.norm(u)

    def _grad_all_is_none(self, u):
        norm = np.linalg.norm(u)
        if np.isclose(norm, 0, atol=1e-10):
            return np.zeros(u.shape[0])
        return u / norm

    def _hessian_all_is_none(self, u):
        norm = np.linalg.norm(u)
        if np.isclose(norm, 0, atol=1e-10):
            return np.zeros((u.shape[0], u.shape[0]))
        return np.eye(u.shape[0]) / (norm ** 2) - (u @ u.T) / (norm ** 3)


class Dot(Function):
    """
    Func(u) = alpha * <c, u>

    Default value:
        alpha = 1
    """
    def __init__(self, c, alpha=None):
        super().__init__()
        self.c = c.ravel()
        self.alpha = alpha
        self._hessian = np.zeros((self.c.shape[0], self.c.shape[0]))

    def __call__(self, u):
        res = np.dot(self.c, u.ravel())
        return res if self.alpha is None else self.alpha * res

    def gradient(self, u):
        res = self.c
        return res if self.alpha is None else self.alpha * res

    def hessian(self, u):
        return self._hessian


class SquaredDot(Function):
    """
    Func(u) = alpha * <c, u>^2

    Default value:
        alpha = 1
    """
    def __init__(self, c, alpha=None):
        super().__init__()
        self.c = c.ravel()
        self.alpha = alpha

        if self.alpha is None:
            self._hessian = 2 * (self.c[:, None] @ self.c[None, :])
        else:
            self._hessian = (2 * self.alpha) * (self.c[:, None] @ self.c[None, :])

    def __call__(self, u):
        res = np.dot(self.c, u.ravel()) ** 2
        return res if self.alpha is None else self.alpha * res

    def gradient(self, u):
        res = (2 * np.dot(self.c, u.ravel())) * self.c
        return res if self.alpha is None else self.alpha * res

    def hessian(self, u):
        return self._hessian


class DotSpecial(Function):
    """
    Func(u) = <Au, u> = u^T * A * u

    Default values:
         A = I
         alpha = 1
    """

    def __init__(self, a=None, alpha=None):
        super().__init__()
        self.a = a
        self.alpha = alpha
        if self.a is not None:
            self._hessian = self.a + self.a.T
            if self.alpha is not None:
                self._hessian *= self.alpha

    def __call__(self, u):
        u = u.ravel()
        if self.a is None:
            res = np.dot(u, u)
        else:
            res = (u[None, :] @ self.a @ u[:, None])[0, 0]
        if self.alpha is not None:
            res *= self.alpha
        return res

    def gradient(self, u):
        u = u.ravel()
        if self.a is None:
            if self.alpha is None:
                return 2 * u
            return (2 * self.alpha) * u
        return (self._hessian @ u[:, None]).ravel()

    def hessian(self, u):
        if self.a is None:
            if self.alpha is None:
                return 2 * np.eye(u.ravel().shape[0])
            return (2 * self.alpha) * np.eye(u.ravel().shape[0])
        return self._hessian
