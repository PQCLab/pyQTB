import numpy as np


def static_proto(proto):
    mtype, elems = proto["mtype"], proto["elems"]
    ratio = proto["ratio"] if "ratio" in proto else [1] * len(elems)
    cdf = np.cumsum(np.array(ratio) / np.sum(ratio))

    def handler(jn, ntot):
        idx = np.where(cdf >= (jn + 1) / ntot)[0][0]
        elem = elems[idx]
        nshots = (ntot - jn) if idx + 1 == len(cdf) else np.floor(cdf[idx] * ntot) - jn
        return {mtype: elem, "nshots": nshots}
    
    return handler
