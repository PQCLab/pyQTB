"""
The module contains the function for performing tests on quantum tomography (QT) method.

***************
Dimension array
***************
The dimension array specifies the partition of the Hilbert space. Examples:

* ``dim = [2]`` -- single qubit system
* ``dim = [2,2,2]`` -- three qubits system
* ``dim = [2,3]`` -- qubit + qutrit system

Note that it is always preferable to specify partition.
E.g. use ``dim = [2,2]`` instead of ``dim = [4]`` for the two-qubit system.

*************************
Measurement specification
*************************
A single measurement is described by a dictionary ``m``.
Its items depend on a particular measurement type (see below).
The ``m['nshots']: int`` item is the measurement sample size (default: 1).
One may also store auxiliary items to access them in QT method.

===================================================
Positive operator-valued measure (POVM) measurement
===================================================
The measurement is described by a set of POVM operators.
The item ``m['povm']: List[numpy.ndarray]`` contains the list of POVM operators matrices.
The measurement result is a ``numpy.ndarray`` vector of integers that sums up to ``m['nshots']``.

======================
Observable measurement
======================
The measurements of the average value of an observable (e.g. a Pauli observables).
The item ``m['observable']: numpy.ndarray`` contains the matrix of the observable.
The measurement results is a float number.

====================
Operator measurement
====================
Only a single POVM operator is considered (e.g. when one projects a polarizing qubit onto a state using the polarizer).
The ``m['operator']: numpy.ndarray`` contains the matrix of the measurement operator.
The measurement result is an integer.

***********************
QT method specification
***********************
A QT method is specified by the set of measurements performed over quantum system and by the quantum state estimator.

============================
Measurement protocol handler
============================
User must provide the function handler that specifies QT method measurements. The function takes the following inputs:

* ``jn: int`` -- the number of samples being measured so far,
* ``ntot: int`` -- total sample size,
* ``meas: List[dict]`` -- the list of previous measurements (see Measurement specification section),
* ``data: List[Union[numpy.ndarray, float]]`` -- the list of previous measurement results (see Measurement specification section),
* ``dim: List[int]`` -- dimension array.

The function output ``m: dict`` must specify the next measurement to be performed
(see Measurement specification section).

------------------------------
Example: POVM protocol handler
------------------------------
::

    def povm_protocol_handler():
        def handler(jn, ntot, data, meas, dim)
            # process inputs
            return {'povm': ..., 'nshots': ...}
        return handler

----------------------------------
Example: Storing additional fields
----------------------------------
::

    def povm_protocol_handler():
        def handler(jn, ntot, data, meas, dim)
            # process inputs
            # the field foo will be also available in meas in future measurements
            return {'povm': ..., 'nshots': ..., 'foo': ...}
        return handler

------------------------------
Example: Suppressing arguments
------------------------------
::

    def povm_protocol_handler():
        def handler(jn, ntot)
            # process inputs
            return {'povm': ..., 'nshots': ...}
        return handler

-------------------------------------
Example: Using ``helpers.static_proto``
-------------------------------------
::

    from helpers.static_proto import static_proto

    def povm_protocol_handler():
        return static_proto({
            'mtype': 'povm',  # measurement type
            'elems': [[...], [...], ...],  # list of POVM operators sets
        })

=================
Estimator handler
=================
User must provide the function handler that specifies QT method estimator. The function takes the following inputs:

* ``meas: List[dict]`` -- the list of previous measurements (see Measurement specification section),
* ``data: List[Union[numpy.ndarray, float]]`` -- the list of previous measurement results (see Measurement specification section),
* ``dim: List[int]`` -- dimension array.

The function output ``dm: numpy.ndarray`` must be an estimated density matrix.

--------------------------
Example: estimator handler
--------------------------
::

    def estimator_handler():
        def handler(data, meas, dim)
            # process inputs
            return dm
        return handler

------------------------------
Example: Suppressing arguments
------------------------------
::

    def estimator_handler():
        def handler(data, meas)
            # process inputs
            return dm
        return handler

"""

import numpy as np
import time
from typing import List, Union, Optional, Callable

import utils.qtb_tools as tools
import utils.qtb_stats as stats
from qtb_tests import qtb_tests
from utils.qtb_state import qtb_state
from utils.qtb_result import Result


def qtb_analyze(
        fun_proto: Callable[[int, int, List[dict], List[Union[np.ndarray, float]], List[int]], dict],
        fun_est: Callable[[List[dict], List[Union[np.ndarray, float]], List[int]], np.ndarray],
        dim: List[int],
        tests: List[str],
        mtype: str = "povm",
        name: str = "Untitled QT-method",
        max_nsample: float = np.inf,
        display: bool = True,
        filename: Optional[str] = None,
        savefreq: int = 1,
        par_job_id: Optional[int] = None,
        par_job_num: Optional[int] = None
) -> Result:
    """Runs tests to analyze the QT method.

    Given the argument ``filename`` is set, the program stores the intermediate analysis results in the file.
    If the file already exists the program tries to load existing results and update them. The frequency of file update
    is set by ``savefreq`` argument. For example, value 10 makes the program to save results in a file after
    every 10-th experiment. The best practice is to set value to 1000 for very fast QT methods as this would make
    analysis faster. For slow QT methods it is better to set 1.

    Note that is is always **recommended** to specify ``filename`` so the long computations could be continued
    in case of interruption.

    The function supports parallel mode. In this mode all experiments are divided into ``par_job_num`` batches and
    the batch ``par_job_id`` is processed. When all jobs are finished, user must run ``python par_finish.py filename``.
    This command combines the results over all jobs and removes the temporal directory. Note that the parallel mode
    requires specifying the ``filename`` argument.


    :param fun_proto: Measurement protocol handler
    :param fun_est: Estimator handler
    :param dim: Subsystems dimensions
    :param tests: List of test codes to be performed (see qtb_tests for available test codes)
    :param mtype: Type of the measurements performed by the QT method (default: 'POVM'), optional
    :param name: QT method name that would be displayed in reports, optional
    :param max_nsample: Maximum sample size that the method could handle (infinite by default), optional
    :param display: Display analysis status in the command window (default: True), optional
    :param filename: Path to file where the analysis results would be stored, optional
    :param savefreq: Frequency of file saving (default: 1), optional
    :param par_job_id: Job index in parallel mode (default: None), optional
    :param par_job_num: Total number of jobs in parallel mode (default: None), optional
    :return: Analysis result object
    """
    if display:
        print("Initialization...")

    result = Result(filename, dim, display)
    if par_job_id is not None and par_job_num is not None:
        result.par_init(par_job_id, par_job_num)
    result.load()
    result.set_name(name)
    result.set_cpu()

    # Prepare tests
    test_desc = qtb_tests(dim)
    test_codes = tests
    for tcode in test_codes:
        test = test_desc[tcode]
        test["nsample"] = list(filter(lambda ntot: ntot <= max_nsample, test["nsample"]))
        result.init_test(tcode, test)
    result.save()

    # Perform tests
    nb = 0
    for j_test, tcode in enumerate(test_codes):
        test = result.tests[tcode]

        if display:
            print("===> Running test {}/{}: {} ({})".format(j_test+1, len(test_codes), test["name"], test["code"]))
            nb = tools.uprint("")

        for exp_id, experiment in enumerate(result.experiments(tcode)):
            if not np.isnan(experiment["fidelity"][0]):  # experiment results loaded
                continue
            seed = test["seed"] + experiment["exp_num"]
            stats.set_state(seed)
            dm = qtb_state(dim, **test["generator"])
            state = stats.get_state()
            for ntot_id, ntot in enumerate(test["nsample"]):
                if display:
                    nb = tools.uprint("Experiment {}/{}, nsamples = 1e{:.0f}".format(
                        experiment["exp_num"], test["nexp"], np.round(np.log10(ntot))
                    ), nb)
                stats.set_state(state)
                data, meas, experiment["time_proto"][ntot_id], sm_flag = conduct_experiment(
                    dm, ntot, fun_proto, dim, mtype
                )
                tc = time.time()
                dm_est = tools.call(fun_est, meas, data, dim)
                experiment["time_est"][ntot_id] = time.time()-tc
                f, msg = tools.isdm(dm_est)
                if not f:
                    raise ValueError("Estimator error: {}".format(msg))
                experiment["nmeas"][ntot_id] = len(meas)
                experiment["fidelity"][ntot_id] = tools.fidelity(dm, dm_est)
                experiment["sm_flag"] = experiment["sm_flag"] and sm_flag

            result.update(tcode, exp_id, experiment)
            if (exp_id+1) % savefreq == 0 or experiment["is_last"]:
                result.save()

        if display:
            tools.uprint("Done", nb, end="\n")

    return result


def conduct_experiment(dm, ntot, fun_proto, dim, mtype="povm"):
    data, meas = [], []
    time_proto, sm_flag = 0, True
    jn = 0
    while jn < ntot:
        tc = time.time()
        meas_curr = tools.call(fun_proto, jn, ntot, meas, data, dim)
        time_proto += time.time() - tc

        if "nshots" not in meas_curr:
            meas_curr["nshots"] = 1

        if jn + meas_curr["nshots"] > ntot:
            raise ValueError("Number of measurements exceeds available sample size")

        sm_flag = sm_flag and np.all(tools.isprod(meas_curr[mtype], dim))

        data.append(get_data(dm, meas_curr, mtype))
        meas.append(meas_curr)

        jn += meas_curr["nshots"]

    return data, meas, time_proto, sm_flag


def get_data(dm, meas_curr, mtype="povm"):
    tol = 1e-8

    if mtype == "povm":
        probabilities = np.real(
            np.array([operator.flatten() for operator in meas_curr["povm"]]) @ np.reshape(dm, (-1,), order="F")
        )

        if np.any(probabilities < 0):
            if np.any(probabilities < -tol):
                raise ValueError("Measurement operators are not valid: negative probabilities exist")
            probabilities[probabilities < 0] = 0

        total = np.sum(probabilities)
        if abs(1 - total) > tol:
            raise ValueError("Measurement operators are not valid: total probability is not equal to 1")

        return stats.sample(probabilities / total, meas_curr["nshots"])

    elif mtype == "operator":
        meas_curr["povm"] = [
            meas_curr["operator"],
            np.eye(meas_curr["operator"].shape[0], dtype=complex) - meas_curr["operator"]
        ]
        return get_data(dm, meas_curr, mtype="povm")[0]

    elif mtype == "observable":
        w, v = np.linalg.eig(meas_curr["observable"])
        meas_curr["povm"] = [np.outer(v[:, j], v[:, j].conj()) for j in range(v.shape[1])]
        clicks = get_data(dm, meas_curr, mtype="povm")
        return np.sum(clicks * w) / meas_curr["nshots"]

    else:
        raise ValueError("Unknown measurement type")
