"""The module contains the collection of built-in tests

See details in https://arxiv.org/abs/2012.15656
"""
import sys
import numpy as np

from pyqtb import Dimension, Test
from pyqtb.utils.state import haar_random, random_noisy_prepared
from pyqtb.utils.helpers import standard_measurements


def rps_test(dim: Dimension) -> Test:
    """Random pure states test

    Every experiment run generates a Haar random pure state

    See details in https://arxiv.org/abs/2012.15656

    :param dim: System dimension
    :return: Test specification
    """
    return Test(
        dim=dim,
        fun_state=lambda d: haar_random(d, rank=1),
        fun_sim=standard_measurements(),
        rank=1,
        name="Random pure states", title="RPS",
        seed=161,
        nsample=[10 ** (p + max(0, len(dim) - 3)) for p in [2, 3, 4, 5, 6]],
        nexp=1000
    )


def rmspt2_test(dim: Dimension) -> Test:
    """Random mixed states by partial tracing test (rank-2)

    Every experiment run generates a rank-2 mixed state by taking partial trace of a Haar random pure state
    in the extended system of dimension 2 * dim.full

    See details in https://arxiv.org/abs/2012.15656

    :param dim: System dimension
    :return: Test specification
    """
    return Test(
        dim=dim,
        fun_state=lambda d: haar_random(d, rank=2),
        fun_sim=standard_measurements(),
        rank=2,
        name="Random mixed states by partial tracing: rank-2", title="RMSPT-2",
        seed=1312,
        nsample=[10 ** (p + max(0, len(dim) - 2)) for p in [2, 3, 4, 5, 6]],
        nexp=1000
    )


def rmsptd_test(dim: Dimension) -> Test:
    """Random mixed states by partial tracing test (rank-d)

    Every experiment run generates a fully mixed state by taking partial trace of a Haar random pure state
    in the extended system of dimension dim.full ** 2

    See details in https://arxiv.org/abs/2012.15656

    :param dim: System dimension
    :return: Test specification
    """
    return Test(
        dim=dim,
        fun_state=lambda d: haar_random(d, rank=d.full),
        fun_sim=standard_measurements(),
        rank=dim.full,
        name="Random mixed states by partial tracing: rank-d", title="RMSPT-d",
        seed=117218,
        nsample=[10 ** (p + max(0, len(dim) - 1)) for p in [2, 3, 4, 5, 6]],
        nexp=1000
    )


def rnp_test(dim: Dimension) -> Test:
    """Random noisy preparation state

    Every experiment run generates a mixed state being the result of some noisy state preparation procedure.
    The state has full rank but most of the eigenvalues are low.

    See details in https://arxiv.org/abs/2012.15656

    :param dim: System dimension
    :return: Test specification
    """
    return Test(
        dim=dim,
        fun_state=lambda d: random_noisy_prepared(d, init_error_limits=(0, 0.05), depolarization_limits=(0, 0.01)),
        fun_sim=standard_measurements(),
        rank=int(np.prod(dim)),
        name="Random noisy preparation", title="RNP",
        seed=758942,
        nsample=[10 ** (p + max(0, len(dim) - 1)) for p in [2, 3, 4, 5, 6]],
        nexp=1000
    )


def get_test(test_code: str, dim: Dimension) -> Test:
    """Returns a test by its code

    :param test_code: String code of the test (rps/rmspt2/rmsptd/rnp)
    :param dim: System dimension
    :return: Test specification
    """
    try:
        return getattr(sys.modules[__name__], test_code.lower() + "_test")(dim)
    except KeyError:
        raise ValueError(f"QTB Error: test with code {test_code} not found")
