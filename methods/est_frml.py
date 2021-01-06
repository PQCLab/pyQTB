import sys, os
sys.path.append(os.path.abspath("../pyRootTomography"))
from rt_dm_reconstruct import rt_dm_reconstruct


def est_frml(meas, data):
    proto = [m["povm"] for m in meas]
    init = meas[-1]["dm"] if "dm" in meas[-1] else "pinv"
    dm, _ = rt_dm_reconstruct(data, proto, rank="full", tol=1e-10, init=init)
    return dm