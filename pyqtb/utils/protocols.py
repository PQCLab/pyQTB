"""The module contains QT protocols generators"""
import numpy as np
import dill as pickle
import os

from typing import List

from pyqtb import Measurement
import pyqtb.utils.tools as tools


def pauli(num_qubits: int = 1) -> List[Measurement]:
    """Multi-qubit Pauli measurements protocol

    Each element is a tensor product of 2-level Pauli matrices.
    The protocol contains ``4 ** num_qubits`` elements

    :param num_qubits: Number of qubits
    :return: List of measurements
    """
    s = [
        np.array([[1, 0], [0, 1]], dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex)
    ]
    elements = s
    for _ in range(num_qubits - 1):
        elements = tools.kron_list(elements, s)
    return [Measurement(nshots=1, map=e, extras={"type": "observable"}) for e in elements]


def mub(dim: int) -> List[Measurement]:
    """Mutually unbiased bases (MUB) measurement protocol

    Each element is a set of POVM operators.

    :param dim: System dimension
    :return: List of measurements
    """
    assert dim in [2, 3, 4, 8], "QTB Error: Only MUB dimensions 2, 3, 4, 8 are currently supported"
    with open(os.path.dirname(__file__) + "/mubs.pickle", "rb") as handle:
        bases = [u.astype(complex) for u in pickle.load(handle)["mub" + str(dim)]]
        return [Measurement(nshots=1, map=tools.basis2povm(u), extras={"type": "povm", "basis": u}) for u in bases]


def factorized_mub(dim: List[int]) -> List[Measurement]:
    """Factorized mutually unbiased bases (MUB) measurement protocol

    The measurement bases are obtained by tensor products of subsystems bases.
    Each element is a set of POVM operators.

    :param dim: List of dimensions of each sub-system
    :return: List of measurements
    """
    bases = [m.extras["basis"] for m in mub(dim[0])]
    if len(dim) > 1:
        for d in dim[1:]:
            bases = tools.kron_list(bases, [m.extras["basis"] for m in mub(d)])
    return [Measurement(nshots=1, map=tools.basis2povm(u), extras={"type": "povm", "basis": u}) for u in bases]
