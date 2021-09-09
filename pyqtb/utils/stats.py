import numpy as np
from scipy.stats import norm as norm
from scipy.linalg import cholesky as chol


def set_state(state):
    if type(state) is tuple:
        QRNG.set_state(state)
    else:
        QRNG.seed(state)


def get_state():
    return QRNG.get_state()


def rand(sz=None):
    if sz is None:
        return QRNG.rand()
    else:
        return np.reshape(QRNG.rand((np.prod(sz),)), sz, order="F")


def randn(sz=None):
    return norm.ppf(rand(sz))


def binornd(n, p, sz=None):
    n = int(n)
    sz = (n,) if sz is None else (n,)+sz
    return np.sum(rand(sz) < p, axis=0)


def mnrnd(n, p):
    n = int(n)
    edges = np.insert(np.cumsum(np.array(p) / np.sum(p)), 0, 0)
    return np.histogram(rand((n,)), bins=edges)[0]


def mvnrnd(mu, sigma):
    t = chol(sigma+1e-8)
    return t.T @ randn((len(mu),)) + mu


def randi(m, sz=None):
    if sz is None:
        return int(np.ceil(rand()*m))
    else:
        return list(np.ceil(rand(sz)*m).astype("int"))


def sample(p, n):
    if n > 1e4:  # normal approximation for performance
        mu = p*n
        sigma = (-np.outer(p, p) + np.diag(p)) * n
        k = np.round(mvnrnd(mu, sigma))
        k[np.where(k < 0)[0]] = 0
        if sum(k) > n:
            k[np.argmax(k)] -= sum(k) - n
        else:
            k[-1] = n-sum(k[:-1])
    else:
        if len(p) == 2:
            k = np.empty((2,))
            k[0] = binornd(n, p[0])
            k[1] = n - k[0]
        else:
            k = mnrnd(n, p)
    return k


class QRNG:
    rng = np.random.RandomState()

    @classmethod
    def seed(cls, n):
        cls.rng.seed(n)

    @classmethod
    def rand(cls, sz=()):
        return cls.rng.rand(*sz)

    @classmethod
    def get_state(cls):
        return cls.rng.get_state()

    @classmethod
    def set_state(cls, state):
        cls.rng.set_state(state)
