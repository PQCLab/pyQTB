import numpy as np
from inspect import signature
from .qtb_stats import randn


def isdm(dm, tol=1e-8):
    if np.linalg.norm(dm-dm.conj().T) > tol:
        return False, "Density matrix should be Hermitian"
    
    if np.sum(np.linalg.eigvals(dm)<-tol) > 0:
        return False, "Density matrix shold be non-negative"
    
    if np.abs(np.trace(dm)-1) > tol:
        return False, "Density matrix should have a unit trace"
    
    return True, ""


def isprod(A: np.array, dim):
    if len(A.shape) == 3:
        f = [ isprod(A[j, :, :], dim) for j in range(A.shape[0]) ]
    else:
        md = len(dim)
        if md == 1:
            return True
        
        f = True
        for js in range(md):
            diml = np.prod(dim[0:js]) if js > 0 else 1
            dims = dim[js]
            dimr = np.prod(dim[(js+1):]) if js < md-1 else 1
            Ap = np.reshape(A, (dimr,dims,diml,dimr,dims,diml), order='F')
            Ap = np.transpose(Ap,(1,4,0,3,2,5))
            Ap = np.reshape(Ap, (dims**2,-1), order='F')
            f = f and (np.linalg.matrix_rank(Ap) == 1)
            if ~f:
                break
    return f


def uprint(text, nb=0, end=""):
    textsp = text + " "*max(0, nb-len(text))
    print("\r" + textsp, end=end, flush=True)
    return len(textsp)


def fidelity(a, b):
    a = a/np.trace(a)
    b = b/np.trace(b)
    v, w, _ = np.linalg.svd(a)
    sqa = v.dot(np.diag(np.sqrt(w))).dot(v.conj().T)
    A = sqa.dot(b).dot(sqa)
    f = np.real(np.sum(np.sqrt(np.linalg.eigvals(A)))**2)
    if f > 1:  # fix computation inaccuracy
        f = 2-f
    return f


def call(fun, *args):
    n = len(signature(fun).parameters)
    return fun(*args[0:n])


def supkron(A, B):
    sa = A.shape
    sb = B.shape
    if len(sa) < len(sb):
        A = np.reshape((1,) * (len(sb) - len(sa)) + sa, A, order="F")
    elif len(sa) > len(sb):
        B = np.reshape((1,) * (len(sa) - len(sb)) + sb, B, order="F")
    return np.kron(A, B)


def listkron(A, B):
    C = []
    for ja in range(len(A)):
        for jb in range(len(B)):
            C.append(supkron(A[ja], B[jb]))
    return C


def listkronpower(A0, N):
    A = A0
    for j in range(1, N):
        A = listkron(A, A0)
    return A


def principal(H, K=1):
    v, u = np.linalg.eigh(H)
    idx = np.argsort(abs(v))[::-1]
    idx = idx[0:K]
    return u[:, idx]


def randunitary(Dim):
    q, r = np.linalg.qr(randn((Dim, Dim))+1j*randn((Dim, Dim)))
    r = np.diag(r)
    return q*(r/abs(r))


def complete_basis(u):
    sh = u.shape
    if len(sh) == 1:
        u = np.reshape(u, (sh[0], 1))
    d, m = u.shape
    if m >= d:
        return u
    [q, r] = np.linalg.qr(np.hstack((u, randn((d, d-m)) + 1j*randn((d, d-m)))))
    r = np.diag(r)
    return q*(r/abs(r))


def vec2povm(psi):
    d = psi.shape[0]
    m = psi.shape[1]
    povm = np.empty((m, d, d), dtype=complex)
    for j in range(m):
        povm[j, :, :] = np.outer(psi[:, j], psi[:, j].conj())
    return povm

