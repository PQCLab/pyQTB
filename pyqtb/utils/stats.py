"""
The module contains routines for random number generation (RNG)

These generators are useful to synchronize from various QTB libraries and make the results repeatable.
"""
import numpy as np
from itertools import product

from typing import Union, Tuple, Optional, List, NamedTuple
from scipy.stats import norm as norm
from scipy.linalg import cholesky as chol
from scipy.stats.distributions import chi2


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
        mu = p * n
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
            k[0] = binornd(int(n), p[0])
            k[1] = n - k[0]
        else:
            k = mnrnd(int(n), p)
    return k


def medcouple(x: np.ndarray) -> float:
    """Calculates medcouple for a vector array

    Medcouple is a robust estimation for the data skewness.
    See https://en.wikipedia.org/wiki/Medcouple.

    :param x: Data array
    :return: Medcouple value
    """
    x = np.sort(x)
    len_x = len(x)
    xm = (x[int(np.floor((len_x - 1) / 2))] + x[int(np.ceil((len_x - 1) / 2))]) / 2

    x_positive, x_negative = list(x[x >= xm]), list(x[x <= xm])
    len_xp, len_xn = len(x_positive), len(x_negative)
    h = []
    for jp, jn in product(range(len_xp), range(len_xn)):
        xp, xn = x_positive[jp], x_negative[jn]
        if xp > xn:
            h.append(((xp - xm) - (xm - xn)) / (xp - xn))
        else:
            h.append(np.sign(len_xp - 1 - jp - jn))

    h = np.sort(h)
    len_h = len(h)
    return (h[int(np.floor((len_h - 1) / 2))] + h[int(np.ceil((len_h - 1) / 2))]) / 2


class WhiskerBox(NamedTuple):
    """Whisker box data

    Attributes:
        median      Median
        q25         Lower quantile
        q75         Upper quantile
        iqr         Inter-quantile range
        mc          Medcouple
        w1          Lower bound of whisker box
        w2          Upper bound of whisker box
        data        Data array
        is_outlier  Boolean array that shows outliers of data array
    """
    median: float
    q25: float
    q75: float
    iqr: float
    mc: float
    w1: float
    w2: float
    data: np.ndarray
    is_outlier: np.ndarray


def adjusted_whisker_box(data: np.ndarray) -> WhiskerBox:
    """Calculates adjusted whisker box

    See https://en.wikipedia.org/wiki/Box_plot#Variations

    :param data: Data array
    :return: Whisker box data
    """
    q = np.quantile(data, [.25, .5, .75], interpolation="midpoint")
    iqr = q[2] - q[0]
    mc = medcouple(data)
    w1 = (q[0] - 1.5 * iqr * np.exp(-4 * mc)) if mc > 0 else (q[0] - 1.5 * iqr * np.exp(-3 * mc))
    w2 = (q[2] + 1.5 * iqr * np.exp(+3 * mc)) if mc > 0 else (q[0] - 1.5 * iqr * np.exp(+4 * mc))
    return WhiskerBox(
        median=q[1], q25=q[0], q75=q[2], iqr=iqr,
        mc=mc, w1=w1, w2=w2,
        data=data.copy(), is_outlier=np.logical_or(data < w1, data > w2)
    )


def get_bound(
    x: Union[float, np.ndarray], dim: int, rank: int,
    bound_type: str = "mean", quantile: float = None
) -> Union[float, np.ndarray]:
    """Calculates theoretical bound for state tomography infidelity 1-F or its inverse.

    If the input is the total sample size the function returns the corresponding bound for infidelity.
    If the input is the infidelity bound the function returns corresponding total sample size.
    See details in https://arxiv.org/abs/2012.15656.

    :param x: Total QT sample size or infidelity bound
    :param dim: Total system dimension
    :param rank: State rank
    :param bound_type: Bound type (default: mean), optional
    :param quantile: Quantile value (for ``bound_type="quantile"``), optional
    :param inverse: If True the function returns inverse bound (default: False), optional
    :return: Infidelity bound
    """
    nu = (2 * dim - rank) * rank - 1
    if bound_type == "mean":
        return nu ** 2 / (4 * x * (dim - 1))
    elif bound_type == "std":
        return nu ** 2 / (4 * x * (dim - 1)) * np.sqrt(2 * nu)
    elif bound_type == "quantile":
        return chi2.ppf(quantile, nu) * nu / (4 * x * (dim - 1))
    else:
        raise ValueError("QTB Error: bound type unknown")
