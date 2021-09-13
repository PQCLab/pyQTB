"""
Projected pseudo-inversion (PPI) estimator by POVM measurements results. Density matrix is estimated by linear
inversion of measurements results and eigenvalues projection on canonical simplex.
"""
import numpy as np
from pyqtb import Dimension, Measurement, List, EstimatorHandler


def est_ppi():
    """PPI estimator

    :return: Estimator handler
    """
    def handler(meas: List[Measurement], data: List[np.ndarray], dim: Dimension) -> np.ndarray:
        dim = dim.full

        operators = [
            np.concatenate(tuple([np.expand_dims(operator, axis=0) for operator in m.map]), axis=0) for m in meas
        ]
        operators = np.concatenate(tuple(operators), axis=0)
        meas_matrix = np.reshape(operators, (operators.shape[0], -1))

        probabilities = [clicks / np.sum(clicks) for clicks in data]
        probabilities = np.concatenate(tuple(probabilities))

        dm = np.reshape(np.linalg.pinv(meas_matrix) @ probabilities, (dim, dim), order="F")
        dm = (dm + dm.conj().T)/2
        dm = dm / np.trace(dm)
        w, v = np.linalg.eig(dm)
        return (v*project_probabilities(w)) @ v.conj().T

    return handler


def project_probabilities(p):
    a = 0
    ind = np.argsort(p)
    for ji, i in enumerate(ind):
        irest = ind[ji:]
        nrest = len(irest)
        if p[i]+a/nrest >= 0:
            p[irest] += a/nrest
            break
        a += p[i]
        p[i] = 0
    return p
