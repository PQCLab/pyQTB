"""The module contains the function for performing tests on quantum tomography (QT) method.

A QT method is specified by the set of measurements performed over quantum system and by the quantum state estimator.
To specify the QT method one should define ProtocolHandler and EstimatorHandler.
Detailed explanation of these handlers if given in ``pyqtb`` module.

See details in https://arxiv.org/abs/2012.15656
"""
import time
import numpy as np
from typing import Optional

from pyqtb import Dimension, ProtocolHandler, EstimatorHandler, Test, Result

import pyqtb.utils.tools as tools
import pyqtb.utils.stats as stats


def collect(
    dim: Dimension,
    fun_proto: ProtocolHandler,
    fun_est: EstimatorHandler,
    test: Test,
    name: str = "Untitled QT method",
    max_nsample: float = np.inf,
    display: bool = True,
    filename: Optional[str] = None,
    savefreq: int = 1,
    par_job_id: Optional[int] = None,
    par_job_num: Optional[int] = None
) -> Result:
    """Runs tests to collect the QT method.

    Given the argument ``filename`` is set, the program stores the intermediate analysis results in the file.
    If the file already exists the program tries to load existing results and update them.
    The frequency of file update is set by ``savefreq`` argument.
    For example, value 10 makes the program to save results in a file after every 10-th experiment.
    The best practice is to set value to 1000 for very fast QT methods as this would make analysis faster.
    For slow QT methods it is better to set 1.

    Note that is is always **recommended** to specify ``filename`` so the long computations could be continued
    in case of interruption.

    The function supports parallel mode.
    In this mode all experiments are divided into ``par_job_num`` batches and the batch ``par_job_id`` is processed.
    When all jobs are finished, user must run ``python -m pyqtb.par_finish filename``.
    This command combines the results over all jobs and removes the temporal directory.
    Note that the parallel mode requires specifying the ``filename`` argument.

    :param dim: System dimension
    :param fun_proto: Measurement protocol handler
    :param fun_est: Estimator handler
    :param test: The test to be performed
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

    result = Result(dim=dim, filename=filename, verbose=display)
    if par_job_id is not None and par_job_num is not None:
        result.par_init(par_job_id, par_job_num)
    result.set_name(name)
    result.set_cpu()
    result.load()

    if not np.isinf(max_nsample):
        test._replace(nsample=list(filter(lambda n: n <= max_nsample, test.nsample)))
    result.init_test(test)

    if display:
        print(f"===> Running test {test.name} ({test.title})")

    for idx, experiment in enumerate(result.experiments):
        if experiment.fidelity:  # experiment results loaded
            continue

        seed = test.seed + experiment.number
        stats.set_state(seed)

        dm = test.fun_state(dim)
        qrng_state = stats.get_state()

        for ntot in test.nsample:
            stats.set_state(qrng_state)
            if display:
                print(f"Experiment {experiment.number}/{test.nexp}, sample size = 1e{np.round(np.log10(ntot)):.0f}")

            data, meas, time_proto, sm_flag = tools.simulate_experiment(dm, ntot, fun_proto, test.fun_sim, dim)

            tc = time.time()
            dm_est = fun_est(meas, data, dim)
            time_est = time.time() - tc
            tools.check_dm(dm_est)

            experiment.time_proto.append(time_proto)
            experiment.time_est.append(time_est)
            experiment.nmeas.append(len(meas))
            experiment.fidelity.append(tools.fidelity(dm, dm_est))
            experiment.sm_flag = experiment.sm_flag and sm_flag

        if (idx + 1) % savefreq == 0 or experiment == result.experiments[-1]:
            result.save()

    if display:
        print("DONE")

    return result
