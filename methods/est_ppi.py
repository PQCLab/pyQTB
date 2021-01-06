import numpy as np


def est_ppi(meas, data, dim):
    Dim = np.prod(dim)
    
    M = [ m["povm"] for m in meas ]
    M = np.concatenate(tuple(M), axis=0)
    B = np.reshape(M, (M.shape[0],-1))
    
    prob = [ kj/np.sum(kj) for kj in data ]
    prob = np.concatenate(tuple(prob))
    
    dm = np.reshape(np.linalg.pinv(B).dot(prob), (Dim,Dim), order="F")
    dm = (dm+dm.conj().T)/2
    dm = dm/np.trace(dm)
    w, v = np.linalg.eig(dm)
    return (v*project_probabilities(w)).dot(v.conj().T)


def project_probabilities(p):
    a = 0
    ind = np.argsort(p)
    for ji, i in enumerate(ind):
        irest = ind[ji:]
        nrest = len(irest)
        if p[i]+a/nrest >= 0:
            p[irest] += a/nrest
            break
        a += p[i]
        p[i] = 0
    return p
