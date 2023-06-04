import numpy as np


class Constraint:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def gradient(self, *args, **kwargs):
        raise NotImplementedError

    def projection(self, h):
        raise NotImplementedError


class DotEqualConstraint(Constraint):
    """
    alpha * <c,u> + beta == 0
    """

    def __init__(self, alpha, beta, c):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.c = c.ravel()

        if self.alpha == 0:
            raise ValueError("Alpha mustn't be equal to null")

    def __call__(self, u):
        u = u.ravel()
        return self.alpha * np.dot(u, self.c) + self.beta

    def gradient(self, u):
        u = u.ravel()
        return self.alpha * self.c

    def projection(self, h):
        h = h.ravel()
        return h - (
                (np.dot(self.c, h) + self.beta / self.alpha) /
                np.dot(self.c, self.c)
        ) * self.c


class DotLowerConstraint(Constraint):
    """
    alpha * <c,u> + beta <= 0
    """

    def __init__(self, alpha, beta, c):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.c = c.ravel()

        if self.alpha == 0:
            raise ValueError("Alpha mustn't be equal null")

    def __call__(self, u):
        u = u.ravel()
        return self.alpha * np.dot(u, self.c) + self.beta

    def gradient(self, u):
        u = u.ravel()
        return self.alpha * self.c

    def projection(self, h):
        h = h.ravel()
        if self.__call__(h) <= 0:
            return h
        return h - (
                (np.dot(self.c, h) + self.beta / self.alpha) /
                np.dot(self.c, self.c)
        ) * self.c


class SphereConstraint(Constraint):
    """
    alpha * ||u - f||^2 + beta == 0

    Default value:
        f = 0
    """

    EPS_CLOSE = 1e-10
    EPS_ADD = 1e-5

    def __init__(self, alpha, beta, f=None):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.f = f if f is None else f.ravel()

        if self.alpha == 0 or self.beta == 0:
            raise ValueError("parameters must be not null")

        if self.alpha / self.beta > 0:
            raise ValueError("alpha / beta must be < 0")

    def __call__(self, u):
        u = u.ravel()
        if self.f is None:
            return self.alpha * (np.linalg.norm(u) ** 2) + self.beta
        return self.alpha * (np.linalg.norm(u - self.f) ** 2) + self.beta

    def gradient(self, u):
        u = u.ravel()
        if self.f is None:
            return (2 * self.alpha) * u
        return (2 * self.alpha) * (u - self.f)

    def projection(self, h):
        h = h.ravel()

        if self.f is None:
            norm = np.linalg.norm(h)
            if np.abs(norm) < self.EPS_CLOSE:
                h += np.random.uniform(-self.EPS_ADD, self.EPS_ADD, h.shape[0])
            c = np.sqrt(-self.beta / self.alpha) / np.linalg.norm(h)
            return c * h

        norm = np.linalg.norm(h - self.f)
        if np.abs(norm) < self.EPS_CLOSE:
            h += np.random.uniform(-self.EPS_ADD, self.EPS_ADD, h.shape[0])
        c = np.sqrt(-self.beta / self.alpha) / np.linalg.norm(h - self.f)
        return c * h + (1 - c) * self.f


class BallConstraint(Constraint):
    """
    alpha * ||u - f||^2 + beta <= 0

    Default value:
        f = 0
    """

    EPS_CLOSE = 1e-10
    EPS_ADD = 1e-5

    def __init__(self, alpha, beta, f=None):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.f = f if f is None else f.ravel()

        if self.alpha == 0 or self.beta == 0:
            raise ValueError("parameters must be not null")

        if self.alpha / self.beta > 0:
            raise ValueError("alpha / beta must be < 0")

    def __call__(self, u):
        u = u.ravel()
        if self.f is None:
            return self.alpha * (np.linalg.norm(u) ** 2) + self.beta
        return self.alpha * (np.linalg.norm(u - self.f) ** 2) + self.beta

    def gradient(self, u):
        u = u.ravel()
        if self.f is None:
            return (2 * self.alpha) * u
        return (2 * self.alpha) * (u - self.f)

    def projection(self, h):
        h = h.ravel()

        if self.__call__(h) <= 0:
            return h

        if self.f is None:
            norm = np.linalg.norm(h)
            if np.abs(norm) < self.EPS_CLOSE:
                h += np.random.uniform(-self.EPS_ADD, self.EPS_ADD, h.shape[0])
            c = np.sqrt(-self.beta / self.alpha) / np.linalg.norm(h)
            return c * h

        norm = np.linalg.norm(h - self.f)
        if np.abs(norm) < self.EPS_CLOSE:
            h += np.random.uniform(-self.EPS_ADD, self.EPS_ADD, h.shape[0])
        c = np.sqrt(-self.beta / self.alpha) / np.linalg.norm(h - self.f)
        return c * h + (1 - c) * self.f
