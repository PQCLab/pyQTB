"""The module contains wrappers to simplify the package usage"""
import numpy as np
from copy import copy
from types import SimpleNamespace
from typing import List, Protocol, Optional, Any

from pyqtb import Dimension, Measurement, ProtocolHandler, DataSimulatorHandler, Result
import pyqtb.utils.stats as stats


def standard_measurements() -> DataSimulatorHandler:
    """Measurement data simulator handler for ideal measurements

    The handler supports different types of measurements. The particular type depend on the output ``m: Measurement``
    of protocol handler for a given QT method.
    Below we consider ``dm: np.ndarray`` to be the input state density matrix.

    **Positive Operator-Valued Measure (POVM) measurement**

    * ``m.extras['type'] == 'povm'`` (or empty)
    * ``m.map: List[np.ndarray]`` -- list of POVM operators matrices such that

    Here ``np.trace(dm @ m.map[j])`` is the probability to get ``j``-th measurement result.

    **Operator measurement**

    * ``m.extras['type'] == 'operator'``
    * ``m.map: np.ndarray`` -- measurement operator matrix

    Here ``np.trace(dm @ m.map)`` is the probability to observe a count in the measurement.

    **Observable measurement**

    * ``m.extras['type'] == 'observable'``
    * ``m.map: np.ndarray`` -- matrix of an observable

    Here ``np.trace(dm @ m.map)`` is the observable expected value.

    :return: Measurement data simulator handler
    """
    def result_by_povm(dm: np.ndarray, meas: Measurement):
        tol = 1e-8
        probabilities = np.real(
            np.array([operator.flatten() for operator in meas.map]) @ np.reshape(dm, (-1,), order="F")
        )

        if np.any(probabilities < 0):
            if np.any(probabilities < -tol):
                raise ValueError("Measurement operators are not valid: negative probabilities exist")
            probabilities[probabilities < 0] = 0

        total = np.sum(probabilities)
        if abs(1 - total) > tol:
            raise ValueError("Measurement operators are not valid: total probability is not equal to 1")

        return stats.sample(probabilities / total, meas.nshots)

    def result_by_operator(dm: np.ndarray, meas: Measurement):
        return result_by_povm(dm, Measurement(
            nshots=meas.nshots,
            map=[meas.map, np.eye(meas.map.shape[0], dtype=complex) - meas.map]
        ))[0]

    def result_by_observable(dm: np.ndarray, meas: Measurement):
        w, v = np.linalg.eig(meas.map)
        clicks = result_by_povm(dm, Measurement(
            nshots=meas.nshots,
            map=[np.outer(v[:, j], v[:, j].conj()) for j in range(v.shape[1])]
        ))
        return np.sum(clicks * w) / meas.nshots

    def handler(dm: np.ndarray, meas: Measurement):
        if "type" not in meas.extras or meas.extras["type"] == "povm":
            return result_by_povm(dm, meas)
        elif meas.extras["type"] == "operator":
            return result_by_operator(dm, meas)
        elif meas.extras["type"] == "observable":
            return result_by_observable(dm, meas)
        else:
            raise ValueError("Unknown measurement type")

    return handler


def static_protocol(protocol: List[Measurement]) -> ProtocolHandler:
    """Returns the protocol handler for static (non-adaptive) measurements

    Non-adaptive QT methods rely on a set of measurements that are all independent of each other.
    The function input is the list of measurements to be perform.

    The relative values of field ``nshots`` for each measurement specification are used to set the absolute values
    of sample size for each measurement.
    For example, consider a tomography experiment with 1000 total sample size.
    ``static_protocol([Measurement(nshots=1, ...), Measurement(nshots=1, ...)])`` will return a protocol handler for
    two measurements with sample size 500 each.
     ``static_protocol([Measurement(nshots=1, ...), Measurement(nshots=3, ...)])`` means that the second measurement
     will have 3 times more samples. So the first measurement is conducted 250 times and the second one -- 750 times.

    :param protocol: List of protocol measurements
    :return: Protocol function handler
    """
    ratio = [m.nshots for m in protocol]
    cdf = np.cumsum(np.array(ratio) / np.sum(ratio))

    def handler(jn: int, ntot: int, *_) -> Measurement:
        idx = np.where(cdf >= (jn + 1) / ntot)[0][0]
        return Measurement(
            nshots=int((ntot - jn) if idx + 1 == len(cdf) else np.floor(cdf[idx] * ntot) - jn),
            map=protocol[idx].map,
            extras=protocol[idx].extras
        )

    return handler


class IterProtocolHandler(Protocol):
    """Iteration measurement protocol handler data type

    The handler serves iterative_protocol.
    It is basically the same as pyqtb.ProtocolHandler but includes extra first argument ``iteration: int`` for iteration
    number and should return a list of measurements.
    """
    def __call__(
        self,
        iteration: int,
        jn: int,
        ntot: int,
        meas: List[Measurement],
        data: List[Any],
        dim: Dimension
    ) -> List[Measurement]:
        ...


def iterative_protocol(iteration_protocol: IterProtocolHandler) -> ProtocolHandler:
    """Returns the protocol handler for iterative adaptive measurements

    Each iteration contains a protocol being a list of measurements.

    :param iteration_protocol: Function handler that returns the protocol for a given iteration
    :return: Protocol function handler
    """
    class Iteration(SimpleNamespace):
        number: int
        start: int
        current: int
        length: Optional[int] = None

    def handler(jn: int, ntot: int, meas: List[Measurement], data: List[Any], dim: Dimension) -> Measurement:
        if meas:
            iteration = copy(meas[-1].extras["iteration"])
            iteration.current += 1
            if iteration.current == iteration.length:
                iteration = Iteration(number=iteration.number + 1, start=len(meas), current=0)
        else:
            iteration = Iteration(number=1, start=0, current=0)

        if iteration.length is None:
            protocol = iteration_protocol(iteration.number, jn, ntot, meas, data, dim)
            assert sum([m.nshots for m in protocol]) <= (ntot - jn),\
                "QTB Error: Iteration protocol total sample size exceeds available number of measurements"
            meas_current = protocol[0]
            meas_current.extras.update({"iteration_protocol": protocol})
            iteration.length = len(protocol)
        else:
            meas_current = meas[iteration.start].extras["iteration_protocol"][iteration.current]

        meas_current.extras.update({"iteration": iteration})
        return meas_current

    return handler


def qubits_qt_collect(n: int, proto_name: str, est_name: str, test_code: str, filename: str = None, **kwargs) -> Result:
    """Wrapper to run QTB for a set of qubits

    If ``filename`` argument is not provided, it is generated automatically and printed out.

    :param n: Number of qubits
    :param proto_name: String protocol name
    :param est_name: String estimator name
    :param test_code: String test code
    :param filename: Name of the file to save the results
    :param kwargs: Additional arguments to pass into pyqtb.collect function
    :return:
    """
    dim = Dimension([2] * n)
    proto_name = proto_name.lower()
    est_name = est_name.lower()
    test_code = test_code.lower()

    if est_name == "ppi":
        from pyqtb.methods.est_ppi import est_ppi
        est_fun = est_ppi()
    elif est_name == "frml":
        from pyqtb.methods.est_frml import est_frml
        est_fun = est_frml()
    elif est_name == "arml":
        from pyqtb.methods.est_arml import est_arml
        est_fun = est_arml()
    else:
        raise ValueError("Unknown estimator name")

    if proto_name == "fmub":
        from pyqtb.methods.proto_mub import proto_mub
        proto_fun = proto_mub(dim)
    elif proto_name == "fmub":
        from pyqtb.methods.proto_fmub import proto_fmub
        proto_fun = proto_fmub(dim)
    elif proto_name == "amub":
        from pyqtb.methods.proto_amub import proto_amub
        proto_fun = proto_amub(dim, est_fun)
    elif proto_name == "fo":
        from pyqtb.methods.proto_fo import proto_fo
        proto_fun = proto_fo(est_fun)
    elif proto_name == "fomub":
        from pyqtb.methods.proto_fomub import proto_fomub
        proto_fun = proto_fomub(dim, est_fun)
    else:
        raise ValueError("Unknown protocol name")

    if filename is None:
        filename = f"q{n}_{test_code}_{proto_name}-{est_name}.pickle"
        print(f"Generated filename: {filename}")

    if "name" not in kwargs:
        kwargs.update({"name": proto_name.upper() + "-" + est_name.upper()})

    from pyqtb.analyze import collect
    from pyqtb.tests import get_test
    return collect(dim, proto_fun, est_fun, get_test(test_code, dim), filename=filename, **kwargs)
