import sys
import os
from utils.qtb_result import Result

if __name__ == "__main__":
    name = sys.argv[1]
    if os.path.isfile(name):
        pass
    elif os.path.isdir(name):
        name = name.rstrip("/")
        file = os.path.basename(name)
        if file[0:4] == "tmp_":
            dir = os.path.dirname(name)
            if dir:
                dir += "/"
            name = dir + file[4:]
        else:
            name = ""
    else:
        name = ""

    if name:
        result = Result(name)
        result.par_finish()
