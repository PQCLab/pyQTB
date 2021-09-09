import numpy as np
import time

from typing import Callable, List, Union, Any, Tuple

from pyqtb import Dimension, Measurement, ProtocolHandler, DataSimulatorHandler
from pyqtb.utils.stats import randn


def simulate_experiment(
    dm: np.ndarray,
    ntot: int,
    fun_proto: ProtocolHandler,
    fun_sim: DataSimulatorHandler,
    dim: Dimension
) -> Tuple[List[Any], List[Measurement], float, bool]:

    data, meas = [], []
    time_proto, sm_flag = 0, True
    jn = 0
    while jn < ntot:
        tc = time.time()
        meas_curr = fun_proto(jn, ntot, meas, data, dim)
        time_proto += time.time() - tc

        assert jn + meas_curr.nshots <= ntot, "QTB Error: Number of measurements exceeds available sample size"

        sm_flag = sm_flag and is_product(meas_curr.map, dim)
        data.append(fun_sim(dm, meas_curr))
        meas.append(meas_curr)
        jn += meas_curr.nshots

    return data, meas, time_proto, sm_flag


def check_dm(dm, tol: float = 1e-8) -> None:
    assert np.linalg.norm(dm - dm.conj().T) < tol, "QTB Error: Density matrix should be Hermitian"
    assert np.sum(np.linalg.eigvals(dm) < -tol) == 0, "QTB Error: Density matrix should be non-negative"
    assert np.abs(np.trace(dm) - 1) < tol, "QTB Error: Density matrix should have a unit trace"


def is_product(a: Union[np.ndarray, List[np.ndarray]], dim: Dimension) -> bool:
    if type(a) is list:
        return all([is_product(aj, dim) for aj in a])

    md = len(dim)
    if md == 1:
        return True
    f = True
    for js in range(md):
        dim_left = np.prod(dim[0:js]) if js > 0 else 1
        dim_center = dim[js]
        dim_right = np.prod(dim[(js + 1):]) if js < md - 1 else 1
        ap = np.reshape(a, (dim_right, dim_center, dim_left, dim_right, dim_center, dim_left), order='F')
        ap = np.transpose(ap, (1, 4, 0, 3, 2, 5))
        ap = np.reshape(ap, (dim_center ** 2, -1), order='F')
        f = f and (np.linalg.matrix_rank(ap) == 1)
        if ~f:
            break
    return f


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.trace(a)
    b = b / np.trace(b)
    v, w, _ = np.linalg.svd(a)
    sqa = v @ np.diag(np.sqrt(w)) @ v.conj().T
    bab = sqa.dot(b).dot(sqa)
    f = np.real(np.sum(np.sqrt(np.linalg.eigvals(bab)))**2)
    if f > 1:  # fix computation inaccuracy
        f = 2-f
    return f


def principal(H, K=1):
    v, u = np.linalg.eigh(H)
    idx = np.argsort(abs(v))[::-1]
    idx = idx[0:K]
    return u[:, idx]


def rand_unitary(dim: int) -> np.ndarray:
    q, r = np.linalg.qr(randn((dim, dim)) + 1j * randn((dim, dim)))
    r = np.diag(r)
    return q * (r / abs(r))


def complement_basis(u: np.ndarray) -> np.ndarray:
    if len(u.shape) == 1:
        u = np.reshape(u, (u.shape[0], 1))

    d, m = u.shape
    if m >= d:
        return u

    [q, r] = np.linalg.qr(np.hstack((u, randn((d, d - m)) + 1j * randn((d, d-m)))))
    r = np.diag(r)
    return q * (r / abs(r))
