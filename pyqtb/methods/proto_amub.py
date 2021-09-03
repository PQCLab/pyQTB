import numpy as np
import pyqtb.utils.tools as qtb_tools
from pyqtb.utils import protocols
from pyqtb.helpers.iterative_proto import iterative_proto
from copy import deepcopy


def proto_amub(d, fun_est):
    proto = protocols("mub" + str(d))
    base_elems = proto["elems"]
    vh = proto["vectors"][0].conj().T
    return iterative_proto(get_measset, base_elems, vh, fun_est)


def get_measset(iter, jn, ntot, meas, data, dim, base_elems, vh, fun_est):
    proto = deepcopy(base_elems)
    m = len(proto)
    if iter > 1:
        dm = qtb_tools.call(fun_est, meas, data, dim)
        _, u = np.linalg.eigh(dm)
        u = u.dot(vh)
        for elem in proto:
            for k in range(elem.shape[0]):
                elem[k, :, :] = u.dot(elem[k, :, :]).dot(u.conj().T)

    nshots = [max(100, np.floor(jn/30))]*m
    nleft = ntot - jn
    if nleft < sum(nshots):
        nshots = [np.floor(nleft/m)]*m
        nshots[-1] = nleft - sum(nshots[0:-1])

    measset = [{"povm": povm, "nshots": n} for povm, n in zip(proto, nshots)]
    if iter > 1 and jn > 1e4:
        measset[-1].update({"dm": dm})
    return measset
