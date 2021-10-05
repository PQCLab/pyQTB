"""The module contains functions for performing and reporting tests on quantum tomography (QT) method.

A QT method is specified by the set of measurements performed over quantum system and by the quantum state estimator.
To specify the QT method one should define ProtocolHandler and EstimatorHandler.
Detailed explanation of these handlers if given in ``pyqtb`` module.

See details in https://arxiv.org/abs/2012.15656
"""
import time
import numpy as np
from typing import Optional, List, NamedTuple
from warnings import warn
from tabulate import tabulate

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from pyqtb import Dimension, ProtocolHandler, EstimatorHandler, Test, Result, ExperimentResult

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
    """Runs tests to collect the QT method data.

    See details in https://arxiv.org/abs/2012.15656.

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


class BenchmarkValues(NamedTuple):
    """Benchmark values

    See details in https://arxiv.org/abs/2012.15656.

    Attributes:
        title               Title of the QT method
        quantile            Quantile that was used to calculate benchmark values
        error_rate          Error rate that was used to calculate benchmark values
        num_samples         Sample size
        extrapolated        Boolean flag if sample size was obtained by extrapolation
        num_measurements    Number of different measurements
        time_protocol       Total time of measurement protocol calculation, sec
        time_estimation     Time of state estimation, sec
        efficiency          QT method efficiency
        outliers_ratio      Ratio of number of outliers
        factorized          Boolean flag that shows if QT method uses factorized measurements only
    """
    title: str
    quantile: float
    error_rate: float
    num_samples: int
    extrapolated: Optional[bool]
    num_measurements: Optional[int]
    time_protocol: Optional[float]
    time_estimation: Optional[float]
    efficiency: Optional[float]
    outliers_ratio: Optional[float]
    factorized: Optional[bool]


def get_experiments(result: Result) -> List[ExperimentResult]:
    n_exp = len(result.experiments)
    experiments = []
    for e in result.experiments:
        if len(e.fidelity) > 0:
            experiments.append(e)

    if len(experiments) < len(result.experiments):
        raise warn(
            f"QTB Warning: Statistics is incomplete, {n_exp - len(experiments)}/{n_exp} results are empty"
        )

    return experiments


def report(
    result: Result,
    error_rates: Optional[List[float]] = None,
    quantile: float = .95
) -> List[BenchmarkValues]:
    """Generates the report over QT method benchmark values

    See details in https://arxiv.org/abs/2012.15656.

    :param result: Result of data collection
    :param error_rates: List of benchmark error rates (default: [1e-1, 1e-2, 1e-3, 1e-4]), optional
    :param quantile: Quantile value for data interpolation (default: .95), optional
    :return: List of benchmark values for
    """
    if error_rates is None:
        error_rates = [1e-1, 1e-2, 1e-3, 1e-4]

    experiments = get_experiments(result)
    n_exp = len(experiments)

    df = np.array([(1 - np.array(e.fidelity)) for e in experiments])
    nmeas = np.array([np.array(e.nmeas) for e in experiments])
    time_proto = np.array([np.array(e.time_proto) for e in experiments])
    time_est = np.array([np.array(e.time_est) for e in experiments])

    outliers_ratio = np.array([
        np.sum(stats.adjusted_whisker_box(df[:, j]).is_outlier) / n_exp for j in range(df.shape[1])
    ])
    sm_flag = all([e.sm_flag for e in experiments])

    df_mean = np.mean(df, axis=0)
    df_quantile = np.quantile(df, quantile, axis=0, interpolation="midpoint")
    nmeas_quantile = np.quantile(nmeas, quantile, axis=0, interpolation="midpoint")
    time_proto_quantile = np.quantile(time_proto, quantile, axis=0, interpolation="midpoint")
    time_est_quantile = np.quantile(time_est, quantile, axis=0, interpolation="midpoint")

    benchmarks = []
    log_n = np.log10(result.test.nsample)
    for error_rate in error_rates:
        log_nb = interp1d(np.log10(df_quantile), log_n, fill_value="extrapolate")(np.log10(error_rate))
        nb = int(10 ** log_nb)
        extrapolated = not (min(result.test.nsample) <= nb <= max(result.test.nsample))

        efficiency = (
                stats.get_bound(nb, result.dim.full, result.test.rank, "mean") /
                (10 ** interp1d(log_n, np.log10(df_mean))(log_nb))
        ) if not extrapolated else None

        benchmarks.append(BenchmarkValues(
            title=result.name,
            quantile=quantile,
            error_rate=error_rate,
            num_samples=nb,
            extrapolated=extrapolated,
            num_measurements=int(interp1d(log_n, nmeas_quantile)(log_nb)) if not extrapolated else None,
            time_protocol=float(interp1d(log_n, time_proto_quantile)(log_nb)) if not extrapolated else None,
            time_estimation=float(interp1d(log_n, time_est_quantile)(log_nb)) if not extrapolated else None,
            efficiency=efficiency,
            outliers_ratio=float(interp1d(log_n, outliers_ratio)(log_nb)) if not extrapolated else None,
            factorized=sm_flag
        ))

    return benchmarks


def as_table(benchmarks: List[BenchmarkValues], display_fields: List[str] = None) -> str:
    """Returns a string formatted table of benchmark values

    :param benchmarks: List of benchmark values
    :param display_fields: Fields to display
    :return: String table
    """
    field_names = {
        "title": "QT Method",
        "quantile": "Quantile",
        "error_rate": "Error rate, %",
        "num_samples": "Sample size",
        "num_measurements": "Number of measurements",
        "time_protocol": "Protocol time, sec",
        "time_estimation": "Estimation time, sec",
        "efficiency": "Efficiency, %",
        "outliers_ratio": "Outliers ratio",
        "factorized": "Factorized measurements"
    }

    fields = field_names.keys() if display_fields is None else display_fields

    def to_string(field: str, bv: BenchmarkValues) -> str:
        value = getattr(bv, field)
        if value is None:
            return "-"
        elif field == "error_rate" or field == "efficiency":
            return f"{value * 100:.2f}"
        elif field == "num_samples":
            return f"{value:,}".replace(",", " ") + ("*" if bv.extrapolated else "")
        elif field == "time_protocol" or field == "time_estimation":
            return f"{value:.8f}"
        elif field == "factorized":
            return "Y" if value else "N"
        else:
            return str(value)

    rows = []
    for b in benchmarks:
        rows.append([to_string(field, b) for field in fields])

    return tabulate(rows, headers=[field_names[field] for field in fields], colalign="left")


def plot(
    result: Result,
    parameter: str = "infidelity",
    color: str = "tab:blue",
    view: str = "quantile",
    quantile: float = .95
):
    def get_values(e: ExperimentResult) -> np.ndarray:
        if parameter == "infidelity":
            return 1 - np.array(e.fidelity)
        else:
            return np.array(getattr(e, parameter))

    experiments = get_experiments(result)
    data = np.array([get_values(e) for e in experiments])

    if view == "stats":
        pass  # todo
    elif view == "errorbar":
        pass  # todo
    elif view == "mean":
        pass  # todo
    elif view == "quantile":
        data_quantile = np.quantile(data, quantile, axis=0, interpolation="midpoint")
        return plt.plot(result.test.nsample, data_quantile, color=color)
    else:
        raise ValueError("QTB Error: unknown plot view")


def compare(
    results: List[Result],
    error_rate: float = 1e-3,
    quantile: float = .95,
    show_bound: bool = True,
    titles: List[str] = None
) -> List[BenchmarkValues]:

    benchmarks = []

    for jb, result in enumerate(results):
        b = report(result, error_rates=[error_rate], quantile=quantile)[0]
        if titles is not None:
            b = b._replace(title=titles[jb])
        benchmarks.append(b)

    if show_bound:
        dim, rank = results[0].test.dim.full, results[0].test.rank
        benchmarks.insert(0, BenchmarkValues(
            title="Lower bound",
            quantile=quantile,
            error_rate=error_rate,
            num_samples=int(stats.get_bound(error_rate, dim, rank, "quantile", quantile)),
            extrapolated=False,
            num_measurements=None,
            time_protocol=None,
            time_estimation=None,
            efficiency=1.,
            outliers_ratio=None,
            factorized=None
        ))

    return benchmarks
