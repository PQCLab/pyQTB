"""
Full rank maximum likelihood (FRML) estimator for POVM measurements results. Density matrix is estimated by
maximization of likelihood function. The maximization is done by numerical solving of likelihood equation for the
square root of a density matrix.

This estimator implementation requires installing pyRootTomography package:
https://github.com/PQCLab/pyRootTomography
"""

import numpy as np
from root_tomography.entity import State
from root_tomography.experiment import Experiment
from root_tomography.estimator import reconstruct_state


def est_frml():
    """FRML estimator

    :return: Estimator handler
    """
    def handler(meas, data, dim) -> np.ndarray:
        proto = [
            np.concatenate(tuple([np.expand_dims(operator, axis=0) for operator in m.elem]), axis=0) for m in meas
        ]
        e = Experiment(int(np.prod(dim)), State, "polynomial").set_data(
            proto=proto,
            clicks=data,
            nshots=[np.sum(k) for k in data]
        )
        init = meas[-1]["dm"] if "dm" in meas[-1] else "pinv"
        return reconstruct_state(e, rank="full", tol=1e-10, init=init).dm

    return handler
