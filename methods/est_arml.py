import sys, os
sys.path.append(os.path.abspath("../pyRootTomography"))
from rt_dm_reconstruct import rt_dm_reconstruct


def est_arml(sl=0.05):
    return lambda meas, data: handler(meas, data, sl)


def handler(meas, data, sl):
    proto = [m["povm"] for m in meas]
    init = meas[-1]["dm"] if "dm" in meas[-1] else "pinv"
    dm, _ = rt_dm_reconstruct(data, proto, significanceLevel=sl, tol=1e-10, init=init)
    return dm