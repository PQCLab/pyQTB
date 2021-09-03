"""
Adequate rank maximum likelihood (ARML) estimator for POVM measurements results. Density matrix is estimated by
maximization of likelihood function. The maximization is done by numerical solving of likelihood equation for the
square root of a density matrix. Quantum state rank is chosen automatically by means of chi-squared test.

This estimator implementation requires installing pyRootTomography package:
https://github.com/PQCLab/pyRootTomography
"""

import numpy as np
from root_tomography.entity import State
from root_tomography.experiment import Experiment
from root_tomography.estimator import reconstruct_state


def est_arml(sl: float = 0.05):
    """ARML estimator for the specific significance level of the chi-squared test

    :param sl: Significance level
    :return: Estimator handler
    """
    def handler(meas, data, dim, alpha) -> np.ndarray:
        proto = [m["povm"] for m in meas]
        e = Experiment(int(np.prod(dim)), State, "polynomial").set_data(
            proto=proto,
            clicks=data,
            nshots=[np.sum(k) for k in data]
        )
        init = meas[-1]["dm"] if "dm" in meas[-1] else "pinv"
        return reconstruct_state(e, significance_level=alpha, tol=1e-10, init=init).dm

    return lambda meas, data, dim: handler(meas, data, dim, sl)
