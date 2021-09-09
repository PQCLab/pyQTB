from pyqtb import Dimension, ProtocolHandler
from pyqtb.utils.protocols import factorized_mub
from pyqtb.utils.helpers import static_proto


def proto_fmub(dim: Dimension) -> ProtocolHandler:
    return static_proto(factorized_mub(dim.list))
