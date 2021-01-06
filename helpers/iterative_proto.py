from utils.qtb_tools import call


def iterative_proto(fun_measset, *args):
    return lambda jn, ntot, meas, data, dim: handler(fun_measset, jn, ntot, meas, data, dim, *args)


def handler(fun_measset, jn, ntot, meas, *args):
    newiter = False
    itlen = 0
    if not len(meas):
        newiter = True
        niter = 1
        start = 0
        current = 0
    else:
        niter = meas[-1]["niter"]
        start = meas[-1]["iter_start"]
        current = meas[-1]["iter_current"]+1
        itlen = meas[-1]["iter_length"]
        if current >= itlen:
            newiter = True
            niter += 1
            start = len(meas)
            current = 0
            itlen = 0
    
    if newiter:
        measset = call(fun_measset, niter, jn, ntot, meas, *args)
        if type(measset) is dict:
            measset = [measset]
        measurement = measset[0]
        measurement["measset"] = measset
        itlen = len(measset)
    else:
        measurement = meas[start]["measset"][current]
    
    measurement["niter"] = niter
    measurement["iter_start"] = start
    measurement["iter_current"] = current
    measurement["iter_length"] = itlen
    return measurement
