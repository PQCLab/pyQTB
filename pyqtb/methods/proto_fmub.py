from pyqtb.utils import protocols
from pyqtb.utils import listkron
from pyqtb.helpers.static_proto import static_proto


def proto_fmub(dim):
    elems = []
    for d in dim:
        proto = protocols("mub" + str(d))
        elems = listkron(elems, proto["elems"]) if elems else proto["elems"]
    proto["elems"] = elems
    return static_proto(proto)
