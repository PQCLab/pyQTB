"""Factorized mutually unbiased bases (MUB) protocol

POVM elements are tensor products of MUB bases for each subsystem

See details in https://arxiv.org/abs/2012.15656
"""
import numpy as np
from typing import List

from pyqtb import Dimension, Measurement, ProtocolHandler, EstimatorHandler
from pyqtb.utils.protocols import mub
from pyqtb.utils.helpers import iterative_protocol


def proto_amub(dim: Dimension, fun_est: EstimatorHandler) -> ProtocolHandler:
    protocol_base = mub(dim.full)
    povm0_base = protocol_base[0].extras["basis"].conj().T
    num_bases = len(protocol_base)

    def iteration_handler(
            iteration: int, jn: int, ntot: int, meas: List[Measurement], data: List[np.ndarray], _: Dimension
    ) -> List[Measurement]:
        nshots = [max(100, int(np.floor(jn / 30)))] * num_bases
        n_left = ntot - jn
        if n_left < sum(nshots):
            nshots = [int(np.floor(n_left / num_bases))] * num_bases
            nshots[-1] = n_left - sum(nshots[0:-1])

        if iteration == 1:
            protocol = [
                Measurement(
                    nshots=n,
                    map=m.map,
                    extras={"type": "povm"}
                )
                for m, n in zip(protocol_base, nshots)
            ]
        else:
            dm = fun_est(meas, data, dim)
            _, dm_basis = np.linalg.eigh(dm)
            povm0_basis = dm_basis @ povm0_base
            protocol = [
                Measurement(
                    nshots=n,
                    map=[povm0_basis @ operator @ povm0_basis.conj().T for operator in m.map],
                    extras={"type": "povm"}
                )
                for m, n in zip(protocol_base, nshots)
            ]

            if jn > 1e4:
                protocol[-1].extras.update({"dm": dm})

        return protocol

    return iterative_protocol(iteration_handler)
