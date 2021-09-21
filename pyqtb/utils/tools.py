"""The module contains secondary functions"""
import numpy as np
import time

from itertools import product
from typing import List, Union, Any, Tuple

from pyqtb import Dimension, Measurement, ProtocolHandler, DataSimulatorHandler
from pyqtb.utils.stats import randn


def simulate_experiment(
    dm: np.ndarray,
    ntot: int,
    fun_proto: ProtocolHandler,
    fun_sim: DataSimulatorHandler,
    dim: Dimension
) -> Tuple[List[Any], List[Measurement], float, bool]:
    """Simulates a single QT experiment data

    :raises AssertionError: If protocol handler returns a measurement with number of samples exceeding available sample size

    :param dm: Input density matrix
    :param ntot: Total available sample size
    :param fun_proto: Measurement protocol function handler
    :param fun_sim: Measurement data simulator handler
    :param dim: System dimension
    :return: List of measurement results, list of conducted measurements, total protocol calculation time (sec),
    and flag that indicates whether the measurement protocol was using separable measurements only
    """
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


def check_dm(dm: np.ndarray, tol: float = 1e-6) -> None:
    """Checks whether the input is a valid density matrix

    :raises AssertionError: if input is not a valid density matrix

    :param dm: Density matrix
    :param tol: Tolerance (default: 1e-8), optional
    """
    assert np.linalg.norm(dm - dm.conj().T) < tol, "QTB Error: Density matrix should be Hermitian"
    assert np.sum(np.linalg.eigvals(dm) < -tol) == 0, "QTB Error: Density matrix should be non-negative"
    assert np.abs(np.trace(dm) - 1) < tol, "QTB Error: Density matrix should have a unit trace"


def is_product(a: Union[np.ndarray, List[np.ndarray]], dim: Dimension) -> bool:
    """Checks whether the input is a product matrix with respect to specific dimensions of subsystems

    For example, if ``dim = Dimension([2, 2])`` the function checks if the input could be expressed as tensor product
    of two 2x2 matrices.

    If input is a list of matrices, the function checks whether all of them are product matrices.

    :param a: Input matrix ot list of matrices
    :param dim: System dimension
    :return: True if the input is a product matrix (or matrices)
    """
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
    """Calculates the Uhlmann's fidelity between two density matrices

    :param a: First density matrix
    :param b: Second density matrix
    :return: Fidelity
    """
    a = a / np.trace(a)
    b = b / np.trace(b)
    v, w, _ = np.linalg.svd(a)
    sqa = v @ np.diag(np.sqrt(w)) @ v.conj().T
    bab = sqa.dot(b).dot(sqa)
    f = np.real(np.sum(np.sqrt(np.linalg.eigvals(bab)))**2)
    if f > 1:  # fix computation inaccuracy
        f = 2-f
    return f


def principal(h: np.ndarray, k: int = 1) -> np.ndarray:
    """Returns principal eigenvectors of a hermitian matrix

    :param h: Input matrix
    :param k: Number of principal components (default: 1), optional
    :return: Matrix of principal components
    """
    v, u = np.linalg.eigh(h)
    idx = np.argsort(abs(v))[::-1]
    idx = idx[0:k]
    return u[:, idx]


def rand_unitary(dim: int) -> np.ndarray:
    """Generates a Haar random unitary matrix

    :param dim: Hilbert space dimension
    :return: Unitary matrix
    """
    q, r = np.linalg.qr(randn((dim, dim)) + 1j * randn((dim, dim)))
    r = np.diag(r)
    return q * (r / abs(r))


def complement_basis(u: np.ndarray) -> np.ndarray:
    """Adds random vectors to an incomplete basis

    :param u: A single basis vector or orthogonal matrix with columns being basis vectors
    :return: Unitary matrix
    """
    if len(u.shape) == 1:
        u = np.reshape(u, (u.shape[0], 1))

    d, m = u.shape
    if m >= d:
        return u

    [q, r] = np.linalg.qr(np.hstack((u, randn((d, d - m)) + 1j * randn((d, d-m)))))
    r = np.diag(r)
    return q * (r / abs(r))


def basis2povm(u: np.ndarray) -> List[np.ndarray]:
    """Converts a basis in the for of unitary matrix to a set of POVM operators

    :param u: Unitary matrix where columns are the basis vectors
    :return: List of POVM operators matrices
    """
    return [np.outer(u[:, j], u[:, j].conj()) for j in range(u.shape[1])]


def kron_list(a: List[np.ndarray], b: List[np.ndarray]) -> List[np.ndarray]:
    """Returns a Kronecker all-to-all product of two lists of matrices

    :param a: First list of matrices
    :param b: Second list of matrices
    :return: Resulting list of matrices
    """
    return [np.kron(aj, bj) for aj, bj in product(a, b)]
