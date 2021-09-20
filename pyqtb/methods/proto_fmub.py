"""Factorized mutually unbiased bases (MUB) protocol

POVM elements are tensor products of MUB bases for each subsystem

See details in https://arxiv.org/abs/2012.15656
"""
from pyqtb import Dimension, ProtocolHandler
from pyqtb.utils.protocols import factorized_mub
from pyqtb.utils.helpers import static_protocol


def proto_fmub(dim: Dimension) -> ProtocolHandler:
    """Factorized MUB protocol handler

    :param dim: System dimension
    :return: Protocol handler
    """
    return static_protocol(factorized_mub(dim.list))
