from utils.qtb_proto import qtb_proto
from utils.qtb_tools import listkron
from helpers.static_proto import static_proto


def proto_fmub(dim):
    elems = []
    for d in dim:
        proto = qtb_proto("mub"+str(d))
        elems = listkron(elems, proto["elems"]) if elems else proto["elems"]
    proto["elems"] = elems
    return static_proto(proto)
