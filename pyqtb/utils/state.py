import numpy as np

from pyqtb import Dimension
from pyqtb.utils.stats import randn, rand
from pyqtb.utils.tools import rand_unitary, complement_basis


def state(dim: Dimension, stype, rank=np.nan, init_err=None, depol=None):
    if depol is None:
        depol = []
    if init_err is None:
        init_err = []
    Dim = dim.full
    if np.isnan(rank):
        rank = Dim
    
    if stype == "haar_vec":
        x = randn((Dim,))+1j*randn((Dim,))
        return x/np.linalg.norm(x)
    elif stype == "haar_dm":
        psi = state(Dim * rank, "haar_vec")
        psi = np.reshape(psi, (Dim,rank), order="F")
        u,w,_ = np.linalg.svd(psi, full_matrices=False)
        w = w**2
    elif stype == "bures_dm":
        G = randn((Dim,Dim)) + 1j*randn((Dim,Dim))
        U = rand_unitary(Dim)
        A = (np.eye(Dim)+U).dot(G)
        u,w,_ = np.linalg.svd(A, full_matrices=False)
        w = w**2
        w = w/np.sum(w)
    else:
        raise ValueError("Unknown state type: {}".format(stype))
    
    u = complement_basis(u)
    if len(w) < Dim: w = np.pad(w, (0,Dim-len(w)), "constant")
    
    if init_err:
        e0 = param_generator(*init_err)
        w = 1
        for d in dim:
            w = np.kron(w, np.array([1-e0,e0] + [0]*(d-2)))
    
    if depol:
        p = param_generator(*depol)
        w = (1-p)*w + p/Dim
    
    return (u*w).dot(u.conj().T)


def param_generator(ptype, x1, x2=1, lims=None):
    if lims is None:
        lims = [np.nan, np.nan]
    if ptype == "fixed":
        p = x1
    elif ptype == "unirnd":
        p = rand()*(x2-x1) + x1
    elif ptype == "normrnd":
        p = randn()*x2 + x1
        if lims[0] is not np.nan and p < lims[0]:
            p = lims[0]
        elif lims[1] is not np.nan and p > lims[1]:
            p = lims[1]
    else:
        raise ValueError("Unknown depolarization type: {}".format(ptype))
    return p
