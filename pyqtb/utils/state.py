"""
The module contains routines for random states generation

All the random variables are sampled using pyqtb.utils.stats

See details in https://arxiv.org/abs/2012.15656
"""
import numpy as np
from typing import Union, Tuple

from pyqtb import Dimension
from pyqtb.utils.stats import randn, rand
from pyqtb.utils.tools import complement_basis


def haar_random(dim: Dimension, rank: int, return_eig: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Returns a random density matrix of a specific rank using purification and partial tracing

    See details in https://arxiv.org/abs/2012.15656

    :param dim: System dimension
    :param rank: State rank
    :param return_eig: If True, the function returns eigen-decomposition of the density matrix (default: False), optional
    :returns: Density matrix (if ``return_eig == False``) or (if ``return_eig == True``) its eigen-decomposition
    """
    dim = dim.full
    if rank is None:
        rank = dim
    psi = randn((dim, rank)) + 1j * randn((dim, rank))
    psi = psi / np.linalg.norm(np.reshape(psi, (-1,)))
    
    if return_eig:
        u, w, _ = np.linalg.svd(psi, full_matrices=False)
        return u, w ** 2
    else:
        return psi @ psi.conj().T


def random_noisy_prepared(
    dim: Dimension,
    init_error_limits: Tuple[float, float],
    depolarization_limits: Tuple[float, float]
) -> np.ndarray:
    """Simulates the noisy preparation procedure of a random pure state

    The output density matrix is the simulation of a noisy U|0> operation.
    The simulation includes a random classical initialization error of each subsystem.
    The U operation is also considered to be noisy with random depolarization error rate.

    See details in https://arxiv.org/abs/2012.15656

    :param dim: System dimension
    :param init_error_limits: Limits of the values of initialization error random variable
    :param depolarization_limits: Limits of the values of depolarization error rate random variable
    :return: Output density matrix
    """
    u, w = haar_random(dim, 1, True)

    # Fill basis with zero eigenvalues
    u = complement_basis(u)
    if len(w) < dim.full:
        w = np.pad(w, (0, dim.full - len(w)), "constant")

    # Include initialization error
    e0 = rand() * (init_error_limits[1] - init_error_limits[0]) + init_error_limits[0]
    for d in dim.list:
        w = np.kron(w, np.array([1 - e0, e0] + [0] * (d - 2)))

    # Include depolarizing noise
    p = rand() * (depolarization_limits[1] - depolarization_limits[0]) + depolarization_limits[0]
    w = (1 - p) * w + p / dim.full

    return (u * w) @ u.conj().T
