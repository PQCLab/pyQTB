import numpy as np

from pyqtb import Dimension, Test, StateGeneratorHandler
from pyqtb.utils.state import state
from pyqtb.utils.helpers import standard_measurements


def rps_test(dim: Dimension) -> Test:
    return Test(
        dim=dim,
        fun_state=StateGeneratorHandler(lambda d: state(d, "haar_dm", rank=1)),
        fun_meas=standard_measurements(),
        rank=1,
        name="Random pure states", title="RPS",
        seed=161,
        nsample=[10 ** (p + max(0, len(dim) - 3)) for p in [2, 3, 4, 5, 6]],
        nexp=1000
    )


def rmspt2_test(dim: Dimension) -> Test:
    return Test(
        dim=dim,
        fun_state=StateGeneratorHandler(lambda d: state(d, "haar_dm", rank=2)),
        fun_meas=standard_measurements(),
        rank=2,
        name="Random mixed states by partial tracing: rank-2", title="RMSPT-2",
        seed=1312,
        nsample=[10 ** (p + max(0, len(dim) - 2)) for p in [2, 3, 4, 5, 6]],
        nexp=1000
    )


def rmsptd_test(dim: Dimension) -> Test:
    return Test(
        dim=dim,
        fun_state=StateGeneratorHandler(lambda d: state(d, "haar_dm", rank=dim.full)),
        fun_meas=standard_measurements(),
        rank=dim.full,
        name="Random mixed states by partial tracing: rank-d", title="RMSPT-d",
        seed=117218,
        nsample=[10 ** (p + max(0, len(dim) - 1)) for p in [2, 3, 4, 5, 6]],
        nexp=1000
    )


def rnp_test(dim: Dimension) -> Test:
    return Test(
        dim=dim,
        fun_state=StateGeneratorHandler(lambda d: state(d, "haar_dm", rank=1, init_err=("unirnd", 0, 0.05), depol=("unirnd", 0, 0.01))),
        fun_meas=standard_measurements(),
        rank=int(np.prod(dim)),
        name="Random noisy preparation", title="RNP",
        seed=758942,
        nsample=[10 ** (p + max(0, len(dim) - 1)) for p in [2, 3, 4, 5, 6]],
        nexp=1000
    )


def get_test(test_code: str, dim: Dimension) -> Test:
    try:
        return locals()[test_code + "_test"](dim)
    except KeyError:
        raise ValueError(f"QTB Error: test with code {test_code} not found")
