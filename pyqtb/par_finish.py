"""Finish parallel computation

The command merges all the parallel jobs into a single result file.

``python -m pyqtb.par_finish FILENAME``
"""
import sys
from pyqtb import Result

if __name__ == "__main__":
    Result(filename=sys.argv[1]).par_finish()
