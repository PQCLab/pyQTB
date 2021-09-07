import os
from threading import Thread
from typing import NamedTuple, List, Callable, Any, Optional, Dict
from types import FunctionType, SimpleNamespace

import dill as pickle
import numpy as np
import inspect
import re

from cpuinfo import get_cpu_info


class Measurement(NamedTuple):
    mtype: str
    nshots: int
    elem: Any


class Test(NamedTuple):
    dim: List[int]
    fun_state: Callable[[List[int]], np.ndarray]
    fun_meas: Callable[[np.ndarray, Measurement], Any]
    nsample: List[int]
    nexp: int
    seed: int
    rank: int
    code: str
    title: str
    name: str

    def __str__(self):
        str_values = []
        for attr in ["dim", "fun_state", "fun_meas", "nsample", "nexp", "seed"]:
            value = getattr(self, attr)
            if isinstance(value, FunctionType):
                str_value = re.sub(r"[\n\t\s]*", "", inspect.getsourcelines(value)[0][0])
            else:
                str_value = str(value)
            str_values.append(f"{attr}={str_value}")
        return "::".join(str_values)


class Experiment(SimpleNamespace):
    number: int
    time_est: np.ndarray
    time_proto: np.ndarray
    nmeas: np.ndarray
    fidelity: np.ndarray
    sm_flag: bool


class Result:

    lib = "pyQTB"
    extension = ".pickle"
    data_fields = [
        "name", "dim", "cpu", "lib",
        "test", "experiments_range", "experiments"
    ]

    def __init__(self, dim: Optional[List[int]] = None, filename: Optional[str] = None, verbose: bool = True):
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
        self.experiments: List[Experiment] = []

        # parallel mode
        self.par_job_id: Optional[int] = None
        self.par_job_num: Optional[int] = None
        self.par_filename: Optional[str] = None

    def load(self) -> "Result":
        if self.is_par_mode():
            if not self._load(self.par_filename) and self._load(self.filename):
                self.experiments_range = self.get_experiments_range(self.test.nexp)
                self.experiments = [self.experiments[j] for j in self.experiments_range]
            return self

        self._load(self.filename)
        return self

    def _load(self, filename) -> bool:
        if not filename or not os.path.isfile(filename):
            return False

        with open(filename, "rb") as handle:
            data = pickle.load(handle)
            self.set_data(data)

        if self.verbose:
            print(f"Results file {filename} loaded")

        return True

    def save(self) -> "Result":
        filename = self.par_filename if self.is_par_mode() else self.filename
        if filename is not None:
            safe_dump(filename, self.data)
        return self

    def set_dim(self, dim: List[int]) -> "Result":
        self.dim = dim
        return self

    def set_name(self, name: str) -> "Result":
        self.name = name
        return self

    def set_cpu(self, cpu: Optional[str] = None) -> "Result":
        self.cpu = get_cpu_info()["brand_raw"] if cpu is None else cpu
        return self

    def set_data(self, data: Dict[str, Any]) -> "Result":
        for field in self.data_fields:
            setattr(self, field, data[field])
        return self

    @property
    def data(self) -> Dict[str, Any]:
        data = {field: getattr(self, field) for field in self.data_fields}
        return data

    def init_test(self, test: Test) -> "Result":
        if self.test is not None:
            return self

        self.test = test
        self.experiments_range = self.get_experiments_range(test.nexp)
        for experiment_id in range(len(self.experiments_range)):
            self.experiments.append(Experiment(
                number=experiment_id + self.experiments_range[0] + 1,
                time_est=np.full((len(test.nsample),), np.nan),
                time_proto=np.full((len(test.nsample),), np.nan),
                nmeas=np.full((len(test.nsample),), np.nan),
                fidelity=np.full((len(test.nsample),), np.nan),
                sm_flag=True
            ))
        return self

    @property
    def par_dir(self) -> str:
        return os.path.join(
            os.path.dirname(self.filename),
            "tmp_" + os.path.basename(self.filename),
        )

    def par_init(self, par_job_id: int, par_job_num: int) -> "Result":
        assert self.filename, "QTB Error: Filename not specified"
        assert 0 <= par_job_id < par_job_num, "QTB Error: Invalid job id"

        try:
            os.mkdir(self.par_dir)
        except FileExistsError:
            pass

        self.par_job_id = par_job_id
        self.par_job_num = par_job_num
        self.par_filename = os.path.join(
            self.par_dir,
            "job_" + str(par_job_id).zfill(len(str(par_job_num-1))) + self.extension
        )
        return self

    def is_par_mode(self) -> bool:
        return self.par_job_id is not None and self.par_job_num is not None

    def get_experiments_range(self, n_exp: int) -> range:
        if self.is_par_mode():
            n_per_job = int(np.ceil(n_exp / self.par_job_num))
            return range(n_per_job * self.par_job_id, min(n_per_job * (self.par_job_id + 1), n_exp))
        else:
            return range(n_exp)

    def par_finish(self) -> "Result":
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

        print("All results are successfully merged")
        if any([np.any(np.isnan(experiment.fidelity)) for experiment in self.experiments]):
            print("Some results are missed")

        for filename in files:
            os.remove(os.path.join(self.par_dir, filename))
        os.rmdir(self.par_dir)

        return self


def safe_dump(filename: str, data: Dict[str, Any], protect: bool = True) -> None:
    if protect:
        th = Thread(target=safe_dump, args=(filename, data, False))
        th.start()
        th.join()
    else:
        with open(filename, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
