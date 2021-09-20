"""Mutually unbiased bases (MUB) protocol

POVM elements are MUB bases

See details in https://arxiv.org/abs/2012.15656
"""
from pyqtb import Dimension, ProtocolHandler
from pyqtb.utils.protocols import mub
from pyqtb.utils.helpers import static_protocol


def proto_mub(dim: Dimension) -> ProtocolHandler:
    """MUB protocol handler

    :param dim: System dimension
    :return: Protocol handler
    """
    return static_protocol(mub(dim.full))
