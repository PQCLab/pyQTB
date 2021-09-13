from pyqtb import Dimension, ProtocolHandler
from pyqtb.utils.protocols import factorized_mub
from pyqtb.utils.helpers import static_protocol


def proto_fmub(dim: Dimension) -> ProtocolHandler:
    return static_protocol(factorized_mub(dim.list))
