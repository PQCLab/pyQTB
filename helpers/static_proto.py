import numpy as np

def static_proto(proto):
    mtype = proto["mtype"]
    elems = proto["elems"]
    ratio = proto["ratio"] if "ratio" in proto else [1]*len(elems)
    cdf = np.cumsum(np.array(ratio)/np.sum(ratio))
    
    return lambda jn, ntot : handler(jn,ntot,elems,mtype,cdf)

def handler(jn, ntot, elems, mtype, cdf):
    ind = np.where(cdf >= (jn+1)/ntot)[0][0]
    elem = elems[ind]
    nshots = ntot-jn if ind+1 == len(cdf) else np.floor(cdf[ind]*ntot)-jn
    return {mtype: elem, "nshots": nshots}