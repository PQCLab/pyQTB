import numpy as np
import pyqtb.utils.tools as qtb_tools
import pyqtb.utils.stats as qtb_stats
from pyqtb.utils import protocols
from .proto_fo import get_suborth
from pyqtb.helpers.iterative_proto import iterative_proto
from copy import deepcopy


def proto_fomub(dim, fun_est):
    base_elems = []
    vh = []
    for d in dim:
        proto = protocols("mub" + str(d))
        base_elems.append(proto["elems"])
        vh.append(proto["vectors"][0].conj().T)
    return iterative_proto(get_measset, base_elems, vh, fun_est)


def get_measset(iter, jn, ntot, meas, data, dim, base_elems, vh, fun_est):
    if iter > 1:
        dm = qtb_tools.call(fun_est, meas, data, dim)
        K = qtb_stats.randi(sum(dim) - len(dim))
        psi = qtb_tools.principal(dm, K)
        phis = get_suborth(psi, dim)
        proto = []
        for elems, phi_j, vh_j in zip(deepcopy(base_elems), phis, vh):
            u = qtb_tools.complete_basis(phi_j).dot(vh_j)
            for elem in elems:
                for k in range(elem.shape[0]):
                    elem[k, :, :] = u.dot(elem[k, :, :]).dot(u.conj().T)
            proto = qtb_tools.listkron(proto, elems) if proto else elems
    else:
        proto = []
        for elems in base_elems:
            proto = qtb_tools.listkron(proto, elems) if proto else elems

    m = len(proto)
    nshots = [max(100, np.floor(jn / 30))] * m
    nleft = ntot - jn
    if nleft < sum(nshots):
        nshots = [np.floor(nleft / m)] * m
        nshots[-1] = nleft - sum(nshots[0:-1])

    measset = [{"povm": povm, "nshots": n} for povm, n in zip(proto, nshots)]
    if iter > 1 and jn > 1e4:
        measset[-1].update({"dm": dm})
    return measset
