"""Factorized orthogonal (FO) tomography protocol

Adaptive tomography protocol.
Each measurement is conducted in the factorized basis orthogonal to eigen-basis of current estimator.

See details in https://arxiv.org/abs/2012.15656
"""
import numpy as np
from typing import List

from pyqtb import Dimension, Measurement, ProtocolHandler, EstimatorHandler

import pyqtb.utils.tools as tools
import pyqtb.utils.stats as stats
from scipy.optimize import minimize


def proto_fo(fun_est: EstimatorHandler) -> ProtocolHandler:
    """Returns protocol handler for FO tomography protocol

    :param fun_est: Function handler that returns current estimator based on data observed
    :return: Protocol handler
    """
    def handler(jn: int, ntot: int, meas: List[Measurement], data: List[np.ndarray], dim: Dimension) -> Measurement:
        assert dim.list != [2], "QTB Error: Single-qubit case not supported for FO protocol"
        extras = {"type": "povm"}

        if jn == 0:
            basis = np.eye(dim.full)
        else:
            dm = fun_est(meas, data, dim)
            num_vectors = stats.randi(sum(dim.list) - len(dim))
            sub_basis = tools.principal(dm, num_vectors)
            orthogonal_basis_vectors = get_orthogonal_basis(sub_basis, dim)
            basis = 1
            for phi in orthogonal_basis_vectors:
                basis = np.kron(basis, tools.complement_basis(phi))

            if jn > 1e4:
                extras.update({"dm": dm})

        nshots = max(100, np.floor(jn / 30))
        nshots = min(ntot - jn, nshots)
        return Measurement(nshots=nshots, map=tools.basis2povm(basis), extras=extras)

    return handler


def get_orthogonal_basis(sub_basis: np.ndarray, dim: Dimension):
    num_vectors = sub_basis.shape[1]
    assert num_vectors <= sum(dim) - len(dim), "QTB Error: Too many vectors, system could not be resolved"

    def parametrize(x: np.ndarray, normalize: bool):
        sub_vectors, norms, vector = [], [], 1
        for d in dim.list:
            phi = x[0:d] + 1j * x[d:2 * d]
            sub_vectors.append(phi)
            norms.append(np.linalg.norm(phi))
            if normalize:
                phi = phi / norms[-1]
            vector = np.kron(vector, phi)
            x = x[2 * d:]
        return sub_vectors, vector, np.array(norms)

    def fun(x: np.ndarray):
        _, vector, norms = parametrize(x, False)
        return sum(abs(vector.conj() @ sub_basis) ** 2) + sum(norms + 1 / norms) - 2 * len(norms)

    result, fun_value = None, 1
    while fun_value > 1e-5:
        x0 = stats.randn((2 * sum(dim),))
        result = minimize(fun, x0, method="BFGS")
        fun_value = result.fun

    # noinspection PyUnresolvedReferences
    return parametrize(result.x, True)[0]
