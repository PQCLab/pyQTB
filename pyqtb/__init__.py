"""The module contains main data types of the package"""
import os
import numpy as np
import dill as pickle

from typing import NamedTuple, List, Any, Optional, Dict, Protocol

from threading import Thread
from types import SimpleNamespace
from cpuinfo import get_cpu_info


class Dimension:
    """System dimension

    The class instance specifies the partition of the Hilbert space. Examples:

    * ``Dimension([2])`` -- single qubit system
    * ``Dimension([2,2,2])`` -- three qubits system
    * ``Dimension([2,3])`` -- qubit + qutrit system
    """
    def __init__(self, spec: List[int]):
        self._spec = spec

    @property
    def list(self) -> List[int]:
        """List of subsystems dimensions"""
        return self._spec

    @property
    def full(self) -> int:
        """Total system dimension"""
        return int(np.prod(self.list))

    def __len__(self) -> int:
        """Returns the number of subsystems"""
        return len(self._spec)

    def __getitem__(self, item: int) -> int:
        """Returns the dimension of subsystem by its index"""
        return self.list[item]


class Measurement(SimpleNamespace):
    """Measurement specification

    Attributes:
        nshots  Integer sample size
        map     An object that can map a density matrix into the measurement result (e.g. list of POVM operators)
        extras  Extra measurement information that could be accessed within QT method handlers (default: {}), optional
    """
    nshots: int
    map: Any
    extras: Dict[str, Any]
    
    def __init__(self, **kwargs):
        if "extras" not in kwargs:
            kwargs.update({"extras": {}})
        super(Measurement, self).__init__(**kwargs)


class ProtocolHandler(Protocol):
    """Measurement protocol handler data type

    Handlers of types ProtocolHandler and EstimatorHandler together specify the QT method.
    Consider the function ``fun`` to be the protocol handler.
    It takes the information from all the previous measurements and returns the next measurement to perform.

    ``m = fun(jn, ntot, meas, data, dim)``

    Function arguments:
        * ``jn: int``                     Number of samples being measured so far,
        * ``ntot: int``                   Total sample size,
        * ``meas: List[Measurement]``     List of previous measurements,
        * ``data: List[Any]``             List of previous measurement results,
        * ``dim: Dimension``              System dimension

    Function returns:
        * ``m: Measurement``              Measurement to perform next

    **Example**
    ::

        from pyqtb import Measurement, ProtocolHandler
        def protocol_handler():
            def handler(jn, ntot, data, meas, dim):
                # process inputs
                return Measurement(nshots=100, map=...)
            return ProtocolHandler(handler)


    **Example: Storing additional fields**
    ::

        from pyqtb import Measurement, ProtocolHandler
        def protocol_handler():
            def handler(jn, ntot, data, meas, dim):
                # process inputs
                very_important_list = ['it was hard to calculate', 'so i better store it']
                # the ``very_important_list`` will be available in ``meas[-1]`` in the next handler access
                return Measurement(nshots=100, map=..., extras=very_important_list)
            return ProtocolHandler(handler)


    **Example: Using static_protocol helper**
    ::

        from pyqtb.utils.helpers import static_protocol
        def povm_protocol_handler():
            return static_protocol({
                'mtype': 'povm',  # POVM measurement type
                'maps': [[...], [...], ...],  # list of POVM operators sets
            })
    """
    def __call__(self, jn: int, ntot: int, meas: List[Measurement], data: List[Any], dim: Dimension) -> Measurement:
        ...


class EstimatorHandler(Protocol):
    """Estimator handler data type

    Handlers of types ProtocolHandler and EstimatorHandler together specify the QT method.
    Consider the function ``fun`` to be the estimator handler.
    It takes the information from all measurements and returns the estimated density matrix.

    ``dm = fun(meas, data, dim)``

    Function arguments:
        * ``meas: List[Measurement]``     List of all measurements,
        * ``data: List[Any]``             List of all measurement results,
        * ``dim: Dimension``              System dimension

    Function returns:
        * ``dm: np.ndarray``              Density matrix

    **Example**
    ::

        from pyqtb import EstimatorHandler
        def estimator_handler():
            def handler(data, meas, dim):
                # process inputs
                return dm
            return EstimatorHandler(handler)
    """
    def __call__(self, meas: List[Measurement], data: List[Any], dim: Dimension) -> np.ndarray:
        ...


class StateGeneratorHandler(Protocol):
    """State generator handler data type

    The handler determines the class of quantum states that are covered by a specific test.
    Consider the function ``fun`` to be the state generator handler.
    It takes the system dimension as the input and returns a random density matrix.
    **Note** that, in order to tests to be fully reproducible, ``fun`` should use RNG from ``pyqtb.utils.stats``.

    ``dm = fun(dim)``

    Function arguments:
        * ``dim: Dimension``    System dimension

    Function returns:
        * ``dm: np.ndarray``    Density matrix

    **Example: random |0><0| and |1><1| mixture**
    ::

        import numpy as np
        from pyqtb import StateGeneratorHandler
        from pyqtb.utils.stats import rand
        def state_handler():
            def handler(dim):
                e = rand()
                return np.diag([1 - e, e] + [0] * (dim.full - 2))
            return StateGeneratorHandler(handler)
    """
    def __call__(self, dim: Dimension) -> np.ndarray:
        ...


class DataSimulatorHandler(Protocol):
    """Measurement data simulator handler data type

    The handler determines the way the data is simulated within a specific test.
    Consider the function ``fun`` to be the data simulator handler.
    It takes the density matrix and the measurement specification as the input and returns a random data.
    **Note** that, in order to tests to be fully reproducible, ``fun`` should use RNG from ``pyqtb.utils.stats``.

    ``data = fun(dm, m)``

    Function arguments:
        * ``dm: np.ndarray``    Density matrix
        * ``m: Measurement``    Measurement specification

    Function returns:
        * ``data: Any``         Measurement results data

    **Example: POVM measurements data simulator**
    ::

        import numpy as np
        from pyqtb import DataSimulatorHandler
        from pyqtb.utils.stats import mnrnd
        def povm_data_handler():
            def handler(dm, m):
                probabilities = [np.trace(dm @ p) for p in m.map]
                return mnrnd(probabilities, m.nshots)
            return DataSimulatorHandler(handler)
    """
    def __call__(self, dm: np.ndarray, m: Measurement) -> Any:
        ...


class Test(NamedTuple):
    """Test specification

    Attributes:
        dim         System dimension
        fun_state   Function handler that generates a random state density matrix for a specific dimension
        fun_sim     Function handler that generates a measurement result for a specific density matrix and measurement
        nsample     List of total sample sizes
        nexp        Number of experiments simulation
        seed        Random number generator seed
        rank        Maximal rank of generated states density matrices
        name        Full test name
        title       Short test title to display in reports
    """
    dim: Dimension
    fun_state: StateGeneratorHandler
    fun_sim: DataSimulatorHandler
    nsample: List[int]
    nexp: int
    seed: int
    rank: int
    name: str
    title: str


class ExperimentResult(SimpleNamespace):
    """Result of a single experiment in test

    Attributes:
        number      Experiment number among all the experiments in test
        time_proto  List of protocol computation times (sec) for each value of sample size in test
        time_est    List of estimator computation times (sec) for each value of sample size in test
        nmeas       List of total measurement counts for each value of sample size in test
        fidelity    List of fidelity values for each value of sample size in test
        sm_flag     True if all measurements are separable for each value of sample size in test, False otherwise
    """
    number: int
    time_proto: List[float]
    time_est: List[float]
    nmeas: List[int]
    fidelity: List[float]
    sm_flag: bool


class Result:
    """A class to store QT method analysis results data

    Attributes:
        lib                 Name and version of the current library
        extension           Results filename extension
        data_fields         List of the class instance attributes that are stored in the result file

        dim                 System dimension
        name                Name of the QT method
        cpu                 String information about the CPU that processed the data
        verbose             Display information in command window
        filename            Results filename
        test                Test specification
        experiments_range   A range of experiments indices that are processed
        experiments         List of experiments results
        par_job_id          Job id in parallel mode
        par_job_num         Total number of jobs in parallel mode
        par_filename        Results filename in parallel mode
    """
    lib = "pyQTB v0.1"
    extension = ".pickle"
    data_fields = [
        "name", "dim", "cpu", "lib",
        "test", "experiments_range", "experiments"
    ]

    def __init__(self, dim: Optional[Dimension] = None, filename: Optional[str] = None, verbose: bool = True):
        """
        :param dim: Dimension array, optional
        :param filename: Results filename, optional
        :param verbose: Display information in command window (default: True), optional
        """
        self.dim = dim
        self.name = ""
        self.cpu = ""
        self.verbose = verbose

        if filename is not None:
            el = len(self.extension)
            if len(filename) <= el or filename[-el:] != self.extension:
                filename += self.extension
        self.filename = filename

        self.test = None
        self.experiments_range = None
        self.experiments: List[ExperimentResult] = []

        # parallel mode
        self._par_job_id = None
        self._par_job_num = None
        self._par_filename = None

    def load(self) -> "Result":
        """Load data from file if filename specified"""
        if self.is_par_mode():
            if not self._load(self.par_filename) and self._load(self.filename):
                self.experiments_range = self.get_experiments_range(self.test.nexp)
                self.experiments = [self.experiments[j] for j in self.experiments_range]
            return self

        self._load(self.filename)
        return self

    def _load(self, filename: str) -> bool:
        """Loads data from file if exists

        :param filename: Filename
        :return: True if file is loaded, False otherwise
        """
        if not filename or not os.path.isfile(filename):
            return False

        with open(filename, "rb") as handle:
            data = pickle.load(handle)
            self.set_data(data)

        if self.verbose:
            print(f"Results file {filename} loaded")

        return True

    def save(self) -> "Result":
        """Save data to file if filename specified"""
        filename = self.par_filename if self.is_par_mode() else self.filename
        if filename is not None:
            safe_dump(filename, self.data)
        return self

    def set_dim(self, dim: Dimension) -> "Result":
        """Sets the system dimension

        :param dim: Dimension object
        """
        self.dim = dim
        return self

    def set_name(self, name: str) -> "Result":
        """Sets the QT method name

        :param name: QT method name
        """
        self.name = name
        return self

    def set_cpu(self, cpu: Optional[str] = None) -> "Result":
        """Sets the CPU information

        :param cpu: CPU string name (default: value provided by cpuinfo package), optional
        """
        self.cpu = get_cpu_info()["brand_raw"] if cpu is None else cpu
        return self

    def set_data(self, data: Dict[str, Any]) -> "Result":
        """Sets the result instance data from dictionary

        :param data: Data dictionary. The keys must correspond to the data_fields attribute.
        """
        for field in self.data_fields:
            setattr(self, field, data[field])
        return self

    @property
    def data(self) -> Dict[str, Any]:
        """Result instance data

        :return: Data dictionary. The keys correspond to the data_fields attribute.
        """
        data = {field: getattr(self, field) for field in self.data_fields}
        return data

    def init_test(self, test: Test) -> "Result":
        """Initializes the test

        :param test: Test specification
        """
        if self.test is not None:
            return self

        self.test = test
        self.experiments_range = self.get_experiments_range(test.nexp)
        for experiment_id in range(len(self.experiments_range)):
            self.experiments.append(ExperimentResult(
                number=experiment_id + self.experiments_range[0] + 1,
                time_est=[],
                time_proto=[],
                nmeas=[],
                fidelity=[],
                sm_flag=True
            ))
        return self

    @property
    def par_job_id(self) -> Optional[int]:
        return self._par_job_id

    @property
    def par_job_num(self) -> Optional[int]:
        return self._par_job_num

    @property
    def par_filename(self) -> Optional[str]:
        return self._par_filename

    @property
    def par_dir(self) -> str:
        """Temporal directory that stores the computation results in parallel mode

        :return: Path to directory
        """
        return os.path.join(
            os.path.dirname(self.filename),
            "tmp_" + os.path.basename(self.filename),
        )

    def par_init(self, par_job_id: int, par_job_num: int) -> "Result":
        """Initializes the parallel mode

        :param par_job_id: Parallel job id
        :param par_job_num: Total number of jobs
        """
        assert self.filename, "QTB Error: Filename not specified"
        assert 0 <= par_job_id < par_job_num, "QTB Error: Invalid job id"

        try:
            os.mkdir(self.par_dir)
        except FileExistsError:
            pass

        self._par_job_id = par_job_id
        self._par_job_num = par_job_num
        self._par_filename = os.path.join(
            self.par_dir,
            "job_" + str(par_job_id).zfill(len(str(par_job_num-1))) + self.extension
        )
        return self

    def is_par_mode(self) -> bool:
        """Checks if the parallel mode is initialized

        :return: True for parallel mode, False otherwise
        """
        return self.par_job_id is not None and self.par_job_num is not None

    def get_experiments_range(self, n_exp: int) -> range:
        """Returns the range of experiments to compute

        The results differs for normal and parallel modes.
        In the latter case the range is a batch defined by par_job_id and par_job_num

        :param n_exp: Total number of experiments
        :return: Range of experiments
        """
        if self.is_par_mode():
            n_per_job = int(np.ceil(n_exp / self.par_job_num))
            return range(n_per_job * self.par_job_id, min(n_per_job * (self.par_job_id + 1), n_exp))
        else:
            return range(n_exp)

    def par_finish(self) -> "Result":
        """Finishes the parallel mode by merging all the results and deleting the temporal directory"""
        assert self.filename, "QTB Error: Filename not specified"
        assert os.path.isdir(self.par_dir), "QTB Error: Parallel results directory does not exist"
        self.load()

        _, _, files = next(os.walk(self.par_dir))
        for filename in files:
            print(f"Merging job file {filename}...")
            par_result = Result(filename=os.path.join(self.par_dir, filename), verbose=False).load()
            if self.test is None:
                self.set_dim(par_result.dim)
                self.set_name(par_result.name)
                self.set_cpu(par_result.cpu)
                self.init_test(par_result.test)

            for experiment_id, experiment in zip(par_result.experiments_range, par_result.experiments):
                self.experiments[experiment_id] = experiment
        self.save()

        print(f"All results are successfully merged into {self.filename}")
        if any([np.any(np.isnan(experiment.fidelity)) for experiment in self.experiments]):
            print("Some results are missed")

        for filename in files:
            os.remove(os.path.join(self.par_dir, filename))
        os.rmdir(self.par_dir)

        return self


def safe_dump(filename: str, data: Dict[str, Any], protect: bool = True) -> None:
    """Safe file save using pickle and threading

    :param filename: Path to save the data
    :param data: Data dictionary
    :param protect: True to protect file from interruption during saving, False otherwise (default: True), optional
    """
    if protect:
        th = Thread(target=safe_dump, args=(filename, data, False))
        th.start()
        th.join()
    else:
        with open(filename, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
