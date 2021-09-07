import argparse
import glob
import scipy.io
import os.path
from pyqtb import Result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--informat", help="Input files format (matlab/python)", required=True)
    parser.add_argument("-o", "--outformat", help="Output files format (matlab/python)", required=True)
    parser.add_argument("files", nargs="+", help="Input files")
    args = parser.parse_args()
    method = ""
    if args.informat == "python" and args.outformat == "matlab":
        scipy.io.matlab.mio5_params.NP_TO_MXTYPES['i4'] = scipy.io.matlab.mio5_params.mxDOUBLE_CLASS
        method = "pickle2mat"
    elif args.informat == "matlab" and args.outformat == "python":
        method = "mat2pickle"
    
    for file_glob in args.files:
        for file in glob.glob(file_glob):
            bname, ext = os.path.splitext(file)
            try:
                if method == "pickle2mat" and ext == ".pickle":
                    result = Result(file, verbose=False)
                    result.load()
                    scipy.io.savemat(bname+".mat", {"data": result.get_data()}, appendmat=False)
                elif method == "mat2pickle" and ext == ".pickle":
                    data = scipy.io.loadmat(file)
                    Result(bname+".pickle").set_data(data).save()
                else:
                    continue
                print("File {}: SUCCESS".format(file))
            except:
                print("File {}: FAILURE".format(file))
