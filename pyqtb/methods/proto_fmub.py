from typing import List, Callable
from pyqtb.utils.protocols import factorized_mub
from pyqtb.utils.helpers import static_proto


def proto_fmub(dim: List[int]) -> Callable:
    return static_proto(factorized_mub(dim))
