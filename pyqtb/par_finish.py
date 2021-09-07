import sys
import os
from pyqtb import Result

if __name__ == "__main__":
    name = sys.argv[1]
    if os.path.isfile(name):
        pass
    elif os.path.isdir(name):
        name = name.rstrip("/")
        file = os.path.basename(name)
        if file[0:4] == "tmp_":
            directory = os.path.dirname(name)
            if directory:
                directory += "/"
            name = directory + file[4:]
        else:
            name = ""
    else:
        name = ""

    if name:
        result = Result(filename=name)
        result.par_finish()
