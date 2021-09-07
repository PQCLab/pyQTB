import numpy as np
from typing import List

from pyqtb import Test
from pyqtb.utils.state import state
from pyqtb.utils.helpers import standard_meas


def rps_test(dim: List[int]) -> Test:
    return Test(
        dim=dim,
        fun_state=lambda: state(dim, "haar_dm", rank=1),
        fun_meas=standard_meas,
        rank=1,
        code="rps", title="RPS", name="Random pure states",
        seed=161,
        nsample=[10 ** (p + max(0, len(dim) - 3)) for p in [2, 3, 4, 5, 6]],
        nexp=1000
    )


def rmspt2_test(dim: List[int]) -> Test:
    return Test(
        dim=dim,
        fun_state=lambda: state(dim, "haar_dm", rank=2),
        fun_meas=standard_meas,
        rank=2,
        code="rmspt2", title="RMSPT-2", name="Random mixed states by partial tracing: rank-2",
        seed=1312,
        nsample=[10 ** (p + max(0, len(dim) - 2)) for p in [2, 3, 4, 5, 6]],
        nexp=1000
    )


def rmsptd_test(dim: List[int]) -> Test:
    d = int(np.prod(dim))
    return Test(
        dim=dim,
        fun_state=lambda: state(dim, "haar_dm", rank=d),
        fun_meas=standard_meas,
        rank=d,
        code="rmsptd", title="RMSPT-d", name="Random mixed states by partial tracing: rank-d",
        seed=117218,
        nsample=[10 ** (p + max(0, len(dim) - 1)) for p in [2, 3, 4, 5, 6]],
        nexp=1000
    )


def rnp_test(dim: List[int]) -> Test:
    return Test(
        dim=dim,
        fun_state=lambda: state(dim, "haar_dm", rank=1, init_err=("unirnd", 0, 0.05), depol=("unirnd", 0, 0.01)),
        fun_meas=standard_meas,
        rank=int(np.prod(dim)),
        code="rnp", title="RNP", name="Random noisy preparation",
        seed=758942,
        nsample=[10 ** (p + max(0, len(dim) - 1)) for p in [2, 3, 4, 5, 6]],
        nexp=1000
    )


def get_test(test_code: str, dim: List[int]) -> Test:
    try:
        return locals()[test_code + "_test"](dim)
    except KeyError:
        raise ValueError(f"QTB: test with code {test_code} not found")
