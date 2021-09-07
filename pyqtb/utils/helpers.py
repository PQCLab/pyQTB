import numpy as np

from pyqtb import Measurement
from pyqtb.analyze import analyze
from pyqtb.utils.tools import call

import pyqtb.utils.stats as stats


def static_proto(proto):
    mtype, elems = proto["mtype"], proto["elems"]
    ratio = proto["ratio"] if "ratio" in proto else [1] * len(elems)
    cdf = np.cumsum(np.array(ratio) / np.sum(ratio))

    def handler(jn, ntot) -> Measurement:
        idx = np.where(cdf >= (jn + 1) / ntot)[0][0]
        return Measurement(
            mtype=mtype,
            elem=elems[idx],
            nshots=(ntot - jn) if idx + 1 == len(cdf) else np.floor(cdf[idx] * ntot) - jn
        )

    return handler


def iterative_proto(fun_meas_set, *args):
    def handler(jn, ntot, meas):
        new_iter, iter_len = False, 0
        if not len(meas):
            new_iter = True
            niter = 1
            start = 0
            current = 0
        else:
            niter = meas[-1]["niter"]
            start = meas[-1]["iter_start"]
            current = meas[-1]["iter_current"] + 1
            iter_len = meas[-1]["iter_length"]
            if current >= iter_len:
                new_iter = True
                niter += 1
                start = len(meas)
                current = 0
                iter_len = 0

        if new_iter:
            meas_set = call(fun_meas_set, niter, jn, ntot, meas, *args)
            if type(meas_set) is dict:
                meas_set = [meas_set]
            measurement = meas_set[0]
            measurement["meas_set"] = meas_set
            iter_len = len(meas_set)
        else:
            measurement = meas[start]["meas_set"][current]

        measurement["niter"] = niter
        measurement["iter_start"] = start
        measurement["iter_current"] = current
        measurement["iter_length"] = iter_len
        return measurement

    return handler


def standard_meas(dm: np.ndarray, meas: Measurement):
    if meas.mtype == "povm":
        tol = 1e-8
        probabilities = np.real(
            np.array([operator.flatten() for operator in meas.elem]) @ np.reshape(dm, (-1,), order="F")
        )

        if np.any(probabilities < 0):
            if np.any(probabilities < -tol):
                raise ValueError("Measurement operators are not valid: negative probabilities exist")
            probabilities[probabilities < 0] = 0

        total = np.sum(probabilities)
        if abs(1 - total) > tol:
            raise ValueError("Measurement operators are not valid: total probability is not equal to 1")

        return stats.sample(probabilities / total, meas.nshots)

    elif meas.mtype == "operator":
        return standard_meas(dm, Measurement(
            mtype="povm",
            elem=[meas.elem, np.eye(meas.elem.shape[0], dtype=complex) - meas.elem],
            nshots=meas.nshots
        ))[0]

    elif meas.mtype == "observable":
        w, v = np.linalg.eig(meas.elem)
        clicks = standard_meas(dm, Measurement(
            mtype="povm",
            elem=[np.outer(v[:, j], v[:, j].conj()) for j in range(v.shape[1])],
            nshots=meas.nshots
        ))
        return np.sum(clicks * w) / meas.nshots

    else:
        raise ValueError("Unknown measurement type")


def qn_state_analyze(n: int, proto_name: str, est_name: str, test_code: str, **kwargs):

    proto_name = proto_name.lower()
    est_name = est_name.lower()
    test_code = test_code.lower()
    dim = [2] * n

    if est_name == "ppi":
        from pyqtb.methods.est_ppi import est_ppi
        est_fun = est_ppi()
        est_mtype = "povm"
    elif est_name == "frml":
        from pyqtb.methods.est_frml import est_frml
        est_fun = est_frml()
        est_mtype = "povm"
    elif est_name == "arml":
        from pyqtb.methods.est_arml import est_arml
        est_fun = est_arml()
        est_mtype = "povm"
    else:
        raise ValueError("Unknown estimator name")

    if proto_name == "fmub":
        from pyqtb.methods.proto_fmub import proto_fmub
        proto_fun = proto_fmub(dim)
        proto_mtype = "povm"
    elif proto_name == "amub":
        from pyqtb.methods.proto_amub import proto_amub
        proto_fun = proto_amub(np.prod(dim), est_fun)
        proto_mtype = "povm"
    elif proto_name == "fo":
        from pyqtb.methods.proto_fo import proto_fo
        proto_fun = proto_fo(est_fun)
        proto_mtype = "povm"
    elif proto_name == "fomub":
        from pyqtb.methods.proto_fomub import proto_fomub
        proto_fun = proto_fomub(dim, est_fun)
        proto_mtype = "povm"
    else:
        raise ValueError("Unknown protocol name")

    if est_mtype != proto_mtype:
        raise ValueError("Protocol {} and estimator {} are incompatible".format(proto_name.upper(), est_name.upper()))

    if "filename" not in kwargs:
        if test_code == "all":
            kwargs.update({"filename": "q{}_{}-{}.pickle".format(n, proto_name, est_name)})
        else:
            kwargs.update({"filename": "q{}_{}_{}-{}.pickle".format(n, test_code, proto_name, est_name)})

    if "name" not in kwargs:
        kwargs.update({"name": proto_name.upper() + "-" + est_name.upper()})

    analyze(proto_fun, est_fun, dim, [test_code], mtype=proto_mtype, **kwargs)
