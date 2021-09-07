import numpy as np
from typing import List
from itertools import product

import pickle
import os


def pauli(num_subsystems: int = 1) -> dict:
    s = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[1, 0], [0, -1]])
    ]
    elems = s
    for _ in range(num_subsystems - 1):
        elems = kron_list(elems, s)
    return {"mtype": "observable", "elems": elems}


def mub(dim: int) -> dict:
    assert dim in [2, 3, 4, 8], "QTB Error: Only MUB dimensions 2, 3, 4, 8 are currently supported"
    with open(os.path.dirname(__file__) + "/mubs.pickle", "rb") as handle:
        bases = pickle.load(handle)["mub" + str(dim)]
        return {
            "mtype": "povm",
            "bases": bases,
            "elems": [basis2povm(u) for u in bases]
        }


def factorized_mub(dim: List[int]) -> dict:
    bases = mub(dim[0])["bases"]
    if len(dim) > 1:
        for d in dim[1:]:
            bases = kron_list(bases, mub(d)["bases"])
    return {
        "mtype": "povm",
        "bases": bases,
        "elems": [basis2povm(u) for u in bases]
    }


def basis2povm(u: np.ndarray) -> List[np.ndarray]:
    return [np.outer(u[:, j], u[:, j].conj()) for j in range(u.shape[1])]


def kron_list(a: List[np.ndarray], b: List[np.ndarray]) -> List[np.ndarray]:
    return [np.kron(aj, bj) for aj, bj in product(a, b)]
