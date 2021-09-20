"""Factorized orthogonal (FO) mutually unbiased bases (MUB) tomography protocol

Adaptive tomography protocol.
Each measurement is conducted in the rotated factorized MUB bases such that one of the total POVM
operators is orthogonal to the current estimator.

See details in https://arxiv.org/abs/2012.15656
"""
import numpy as np
from typing import List

from pyqtb import Dimension, Measurement, ProtocolHandler, EstimatorHandler
from pyqtb.utils.protocols import mub
from pyqtb.utils.helpers import iterative_protocol
from pyqtb.methods.proto_amub import proto_amub
from pyqtb.methods.proto_fo import get_orthogonal_basis

import pyqtb.utils.tools as tools
import pyqtb.utils.stats as stats


def proto_fomub(dim: Dimension, fun_est: EstimatorHandler) -> ProtocolHandler:
    """Returns protocol handler for factorized orthogonal MUB tomography protocol

    :param dim: System dimension
    :param fun_est: Function handler that returns current estimator based on data observed
    :return: Protocol handler
    """
    if len(dim) == 1:
        return proto_amub(dim, fun_est)

    protocol_base_sub, povm0_base_sub, num_bases = [], [], 1
    for d in dim.list:
        protocol_base_sub.append(mub(d))
        povm0_base_sub.append(protocol_base_sub[-1][0].extras["basis"].conj().T)
        num_bases *= len(protocol_base_sub[-1])

    def handler(
            iteration: int, jn: int, ntot: int, meas: List[Measurement], data: List[np.ndarray], dim: Dimension
    ) -> List[Measurement]:
        if iteration == 1:
            dm = None
            povm_sub = []
            for protocol_j in protocol_base_sub:
                povm_sub.append([m.map for m in protocol_j])
        else:
            dm = fun_est(meas, data, dim)
            num_vectors = stats.randi(sum(dim.list) - len(dim))
            sub_basis = tools.principal(dm, num_vectors)
            orthogonal_basis_vectors = get_orthogonal_basis(sub_basis, dim)

            povm_sub = []
            for phi, protocol_j, povm0_base in zip(orthogonal_basis_vectors, protocol_base_sub, povm0_base_sub):
                povm0_basis = tools.complement_basis(phi) @ povm0_base
                povm_sub.append([
                    [povm0_basis @ operator @ povm0_basis.conj().T for operator in m.map]
                    for m in protocol_j
                ])

        nshots = [max(100, np.floor(jn / 30))] * num_bases
        nleft = ntot - jn
        if nleft < sum(nshots):
            nshots = [np.floor(nleft / num_bases)] * num_bases
            nshots[-1] = nleft - sum(nshots[0:-1])

        povm = povm_sub[0]
        for j in range(1, len(povm_sub)):
            povm = tools.kron_list(povm, povm_sub[j])

        protocol = [Measurement(nshots=n, map=list(p), extras={"type": "povm"}) for p, n in zip(povm, nshots)]
        if jn > 1e4:
            protocol[-1].extras.update({"dm": dm})
        return protocol

    return iterative_protocol(handler)
