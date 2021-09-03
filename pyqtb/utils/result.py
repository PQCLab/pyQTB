import numpy as np
from cpuinfo import get_cpu_info
import pickle
import os.path
from dict_hash import sha256
from threading import Thread
from copy import deepcopy


class Result:

    name = ""
    dim = None
    cpu = ""
    lib = "pyQTB"
    extension = ".pickle"
    filename = ""
    tests = None
    verbose = True
    par_result = None
    par_job_id = None
    par_job_num = None
    data_fields = ["name", "dim", "cpu", "lib"]

    def __init__(self, filename, dim=None, verbose=True):
        self.dim = dim
        self.tests = {}
        self.verbose = verbose
        if filename is not None:
            el = len(self.extension)
            if len(filename) <= el or filename[-el:] != self.extension:
                filename += self.extension
            self.filename = filename

    def load(self):
        if not self.filename or not os.path.isfile(self.filename):
            return self
        with open(self.filename, "rb") as handle:
            data = pickle.load(handle)
            self.set_data(data)
        if self.verbose:
            print("Results file loaded with tests:")
            for test in self.tests.values():
                print("* {} ({})".format(test["name"], test["code"]))
        if self.is_par_mode():
            self.par_result.load()
        return self

    def save(self):
        if self.is_par_mode():
            self.par_result.save()
        elif self.filename is not None:
            data = self.get_data()
            safe_dump(self.filename, data)
        return self

    def set_dim(self, dim):
        if self.is_par_mode():
            self.par_result.set_dim(dim)
        self.dim = dim
        return self

    def set_name(self, name):
        if self.is_par_mode():
            self.par_result.set_name(name)
        self.name = name
        return self

    def set_cpu(self, cpu=None):
        if cpu is None:
            cpu = get_cpu_info()["brand_raw"]
        if self.is_par_mode():
            self.par_result.set_cpu(cpu)
        self.cpu = cpu
        return self

    def set_data(self, data):
        if self.dim and data["dim"] != self.dim:
            raise ValueError("Failed to set data: dimensions mismatch")
        for field in self.data_fields:
            setattr(self, field, data[field])
        for tcode, test in data.items():
            if tcode not in self.data_fields:
                self.tests[tcode] = test
        return self

    def get_data(self):
        data = {}
        for field in self.data_fields:
            data.update({field: getattr(self, field)})
        for tcode, test in self.tests.items():
            data.update({tcode: test})
        return data

    def init_test(self, tcode, test):
        thash = sha256(test)
        new_test = True
        if tcode in self.tests:
            if self.tests[tcode]["hash"] == thash:
                new_test = False
            elif self.verbose:
                print("Failed to update results for test {} (fingerprint mismatch).\n"
                      "These results will be overwritten\n".format(test["code"]))
        if new_test:
            testr = deepcopy(test)
            testr.update({
                "hash": thash,
                "time_proto": np.full((test["nexp"], len(test["nsample"])), np.nan),
                "time_est": np.full((test["nexp"], len(test["nsample"])), np.nan),
                "nmeas": np.full((test["nexp"], len(test["nsample"])), np.nan),
                "fidelity": np.full((test["nexp"], len(test["nsample"])), np.nan),
                "sm_flag": True
            })
            self.tests.update({tcode: testr})

        if self.is_par_mode() and tcode not in self.par_result.tests:
            par_test = deepcopy(test)
            par_range = self.par_get_range(par_test["nexp"])
            par_test.update({
                "parent": test,
                "start": par_range[0],
                "nexp": len(par_range),
                "time_proto": self.tests[tcode]["fidelity"][par_range, :],
                "time_est": self.tests[tcode]["time_est"][par_range, :],
                "nmeas": self.tests[tcode]["nmeas"][par_range, :],
                "fidelity": self.tests[tcode]["fidelity"][par_range, :],
                "sm_flag": self.tests[tcode]["sm_flag"]
            })
            self.par_result.tests.update({tcode: par_test})

        return self

    def experiments(self, tcode):
        if self.is_par_mode():
            return self.par_result.experiments(tcode)

        test = self.tests[tcode]
        if "start" in test:
            start_ind = test["start"]
        else:
            start_ind = 0

        experiments = []
        for exp_id in range(test["nexp"]):
            experiments.append({
                "exp_num": start_ind + exp_id + 1,
                "time_est": test["time_est"][exp_id, :],
                "time_proto": test["time_proto"][exp_id, :],
                "nmeas": test["nmeas"][exp_id, :],
                "fidelity": test["fidelity"][exp_id, :],
                "sm_flag": test["sm_flag"],
                "is_last": False
            })

        experiments[-1]["is_last"] = True
        return experiments

    def update(self, tcode, exp_id, experiment):
        if self.is_par_mode():
            self.par_result.update(tcode, exp_id, experiment)
            return self
        for field in ["time_est", "time_proto", "nmeas", "fidelity"]:
            self.tests[tcode][field][exp_id, :] = experiment[field]
        self.tests[tcode]["sm_flag"] = self.tests[tcode]["sm_flag"] and experiment["sm_flag"]
        return self

    def par_init(self, par_job_id, par_job_num):
        if not self.filename:
            raise ValueError("One must specify filename to use parallel mode")
        if par_job_id < 0 or par_job_id >= par_job_num:
            raise ValueError("Invalid job id")
        par_dir = "tmp_" + self.filename
        try:
            os.mkdir(par_dir)
        except FileExistsError:
            pass
        self.par_job_id = par_job_id
        self.par_job_num = par_job_num
        self.par_result = Result("{}/job_id{}".format(par_dir, str(par_job_id).zfill(len(str(par_job_num-1)))), self.dim, verbose=False)
        return self

    def is_par_mode(self):
        return self.par_job_id is not None

    def par_get_range(self, n):
        n_per_job = int(np.ceil(n / self.par_job_num))
        return range(n_per_job * self.par_job_id, min(n_per_job * (self.par_job_id + 1), n))

    def par_finish(self):
        if not self.filename:
            raise ValueError("One must specify filename to use parallel mode")
        self.load()
        par_dir = "tmp_" + self.filename
        par_result = None
        _, _, files = next(os.walk(par_dir))
        for filename in files:
            print("Merging job file {}...".format(filename))
            par_result = Result(par_dir + "/" + filename, verbose=False)
            par_result.load()
            for tcode, test in par_result.tests.items():
                self.init_test(tcode, test["parent"])
                start_ind = par_result.tests[tcode]["start"]
                for exp_id, experiment in enumerate(par_result.experiments(tcode)):
                    self.update(tcode, start_ind + exp_id, experiment)

        if par_result is not None:
            self.set_dim(par_result.dim)
            self.set_name(par_result.name)
            self.set_cpu(par_result.cpu)
        self.save()
        print("All results are successfully merged")
        for test in self.tests.values():
            if np.any(np.isnan(test["fidelity"])):
                print("- Some results are missed in the test {} ({})".format(test["name"], test["code"]))
        for filename in files:
            os.remove(par_dir + "/" + filename)
        os.rmdir(par_dir)
        return self


def safe_dump(filename, data, protect=True):
    if protect:
        th = Thread(target=safe_dump, args=(filename, data, False))
        th.start()
        th.join()
    else:
        with open(filename, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
