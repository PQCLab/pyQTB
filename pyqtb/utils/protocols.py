import numpy as np
from pyqtb.utils.tools import vec2povm, listkronpower
import pickle
import os


def qtb_proto(ptype, nsub=1):
    if ptype == "pauli":
        elems = []
        elems.append(np.array([[1, 0], [0, 1]]))
        elems.append(np.array([[0, 1], [1, 0]]))
        elems.append(np.array([[0, -1j], [1j, 0]]))
        elems.append(np.array([[1, 0], [0, -1]]))
        if nsub > 1:
            elems = listkronpower(elems, nsub)
        return {"mtype": "observable", "elems": elems}

    elif ptype in ["mub2", "mub3", "mub4", "mub8"]:
        fname = os.path.dirname(__file__) + "/mubs.pickle"
        with open(fname, "rb") as handle:
            vectors = pickle.load(handle)[ptype]
            elems = [vec2povm(u) for u in vectors]
        if nsub > 1:
            vectors = listkronpower(elems, nsub)
            elems = listkronpower(elems, nsub)
        return {"mtype": "povm", "elems": elems, "vectors": vectors}
    else:
        raise ValueError("Unknown protocol")

