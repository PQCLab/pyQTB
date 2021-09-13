"""
The module contains routines for random number generation (RNG)

These generators are useful to synchronize from various QTB libraries and make the results repeatable.
"""
import numpy as np

from typing import Union, Tuple, Optional, List
from scipy.stats import norm as norm
from scipy.linalg import cholesky as chol


def set_state(state: Union[int, Tuple]) -> None:
    """Sets the state of RNG

    If input is an integer it is considered as the RNG seed.
    If input is a tuple, it is considered as the RNG state in the form of numpy.random.set_state input.

    :param state: RNG seed or state
    """
    if type(state) is tuple:
        RNG.set_state(state)
    else:
        RNG.seed(state)


# noinspection PyTypeChecker
def get_state() -> Tuple:
    """Returns the RNG state

    The output format is the same as of numpy.random.get_state

    :return: RNG state
    """
    return RNG.get_state()


def rand(sz: Optional[Tuple[int, ...]] = None) -> Union[float, np.ndarray]:
    """Generates uniform random numbers from 0 to 1

    If array shape is not provided, the result if a single random variable.

    :param sz: Array shape, optional
    :return: Sample of random variable
    """
    if sz is None:
        return RNG.rand()
    else:
        return np.reshape(RNG.rand((np.prod(sz),)), sz, order="F")


def randn(sz: Optional[Tuple[int, ...]] = None) -> Union[float, np.ndarray]:
    """Generates standard normal random numbers

    If array shape is not provided, the result if a single random variable.

    :param sz: Array shape, optional
    :return: Sample of random variable
    """
    return norm.ppf(rand(sz))


def binornd(n: int, p: float, sz: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """Generates binomial random numbers

    If array shape is not provided, the result if a single random variable.

    :param n: Number of tries
    :param p: Probability of success
    :param sz: Array shape, optional
    :return: Sample of random variable
    """
    sz = (n, ) if sz is None else (n, ) + sz
    return np.sum(rand(sz) < p, axis=0)


def mnrnd(n: int, p: np.ndarray) -> np.ndarray:
    """Generates multinomial random numbers

    :param n: Number of tries
    :param p: Array of outcomes probabilities
    :return: Sample of random variables
    """
    edges = np.insert(np.cumsum(np.array(p) / np.sum(p)), 0, 0)
    return np.histogram(rand((n, )), bins=edges)[0]


def mvnrnd(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Generates multivariate normal random numbers

    :param mu: Vector of mean values
    :param sigma: Covariance matrix
    :return: Sample of random variables
    """
    t = chol(sigma + 1e-8)
    return t.T @ randn((len(mu),)) + mu


def randi(m: float, sz: Optional[Tuple[int, ...]] = None) -> Union[int, List[int]]:
    """Generates uniform integer random numbers from 0 to ``m``

    If array shape is not provided, the result if a single random variable.

    :param m: Maximum value
    :param sz: Array shape, optional
    :return: Sample of random variable
    """
    if sz is None:
        return int(np.ceil(rand() * m))
    else:
        return np.ceil(rand(sz) * m).astype("int")


def sample(p: np.ndarray, n: int) -> np.ndarray:
    """Samples clicks from discrete probability distribution

    :param p: Probability distribution
    :param n: Number of tries
    :return: Array of observed counts
    """
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


class RNG:
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
