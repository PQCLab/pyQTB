"""
Full rank maximum likelihood (FRML) estimator for POVM measurements results. Density matrix is estimated by
maximization of likelihood function. The maximization is done by numerical solving of likelihood equation for the
square root of a density matrix.

This estimator implementation requires installing pyRootTomography package:
https://github.com/PQCLab/pyRootTomography
"""

import numpy as np
from pyqtb import Dimension, Measurement, List, EstimatorHandler

from root_tomography.entity import State
from root_tomography.experiment import Experiment
from root_tomography.estimator import reconstruct_state


def est_frml() -> EstimatorHandler:
    """FRML estimator

    :return: Estimator handler
    """
    def handler(meas: List[Measurement], data: List[np.ndarray], dim: Dimension) -> np.ndarray:
        proto = [
            np.concatenate(tuple([np.expand_dims(operator, axis=0) for operator in m.map]), axis=0) for m in meas
        ]
        e = Experiment(dim.full, State, "polynomial").set_data(
            proto=proto,
            clicks=data,
            nshots=[np.sum(k) for k in data]
        )
        init = meas[-1].extras["dm"] if (type(meas[-1]) is dict and "dm" in meas[-1].extras) else "pinv"
        return reconstruct_state(e, rank="full", tol=1e-10, init=init).dm

    return handler
