import numpy as np
from qtb_analyze import qtb_analyze


def qn_state_analyze(n: int,
                     proto_name: str,
                     est_name: str,
                     test: str,
                     **kwargs):

    proto_name = proto_name.lower()
    est_name = est_name.lower()
    test = test.lower()
    dim = [2] * n

    if est_name == "ppi":
        from methods.est_ppi import est_ppi
        est_fun = est_ppi
        est_mtype = "povm"
    elif est_name == "frml":
        from methods.est_frml import est_frml
        est_fun = est_frml
        est_mtype = "povm"
    elif est_name == "arml":
        from methods.est_arml import est_arml
        est_fun = est_arml()
        est_mtype = "povm"
    else:
        raise ValueError("Unknown estimator name")

    if proto_name == "fmub":
        from methods.proto_fmub import proto_fmub
        proto_fun = proto_fmub(dim)
        proto_mtype = "povm"
    elif proto_name == "amub":
        from methods.proto_amub import proto_amub
        proto_fun = proto_amub(np.prod(dim), est_fun)
        proto_mtype = "povm"
    elif proto_name == "fo":
        from methods.proto_fo import proto_fo
        proto_fun = proto_fo(est_fun)
        proto_mtype = "povm"
    elif proto_name == "fomub":
        from methods.proto_fomub import proto_fomub
        proto_fun = proto_fomub(dim, est_fun)
        proto_mtype = "povm"
    else:
        raise ValueError("Unknown protocol name")

    if est_mtype != proto_mtype:
        raise ValueError("Protocol {} and estimator {} are incompatible".format(proto_name.upper(), est_name.upper()))

    if "filename" not in kwargs:
        if test == "all":
            kwargs.update({"filename": "q{}_{}-{}.pickle".format(n, proto_name, est_name)})
        else:
            kwargs.update({"filename": "q{}_{}_{}-{}.pickle".format(n, test, proto_name, est_name)})

    if "name" not in kwargs:
        kwargs.update({"name": proto_name.upper() + "-" + est_name.upper()})

    qtb_analyze(proto_fun, est_fun, dim, test, mtype=proto_mtype, **kwargs)
