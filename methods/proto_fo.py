import numpy as np
import utils.qtb_tools as qtb_tools
import utils.qtb_stats as qtb_stats
from scipy.optimize import minimize


def proto_fo(fun_est):
    return lambda jn, ntot, meas, data, dim: handler(jn, ntot, meas, data, dim, fun_est)


def handler(jn, ntot, meas, data, dim, fun_est):
    if dim == [2]:
        raise ValueError("Single-qubit case not supported")

    if jn > 0:
        dm = qtb_tools.call(fun_est, meas, data, dim)
        K = qtb_stats.randi(sum(dim) - len(dim))
        psi = qtb_tools.principal(dm, K)
        phis = get_suborth(psi, dim)
        u = 1
        for phi in phis:
            u = np.kron(u, qtb_tools.complete_basis(phi))
    else:
        u = np.eye(int(np.prod(dim)))

    nshots = max(100, np.floor(jn / 30))
    nshots = min(ntot - jn, nshots)

    measurement = {"povm": qtb_tools.vec2povm(u), "nshots": nshots}
    if jn > 1e4:
        measurement.update({"dm": dm})
    return measurement


def get_suborth(psi, dim):
    K = psi.shape[1]
    if K > (sum(dim) - len(dim)):
        raise ValueError("Too many vectors. System could not be resolved")

    res = None
    fval = 1
    while fval > 1e-5:
        x0 = qtb_stats.randn((2 * sum(dim),))
        res = minimize(_suborth_fun, x0, args=(psi, dim), method="BFGS")
        fval = res.fun

    phis, _, _ = _suborth_param(res.x, dim)
    return phis


def _suborth_fun(x, psi, dim):
    _, phi, norms = _suborth_param(x, dim, normalize=False)
    return sum(abs(phi.conj().T.dot(psi)) ** 2) + sum(norms + 1 / norms) - 2 * len(norms)


def _suborth_param(x, dim, normalize=True):
    phis = []
    norms = []
    phi = 1
    for d in dim:
        phi_j = x[0:d] + 1j * x[d:2 * d]
        phis.append(phi_j)
        norms.append(np.linalg.norm(phi_j))
        if normalize:
            phi_j = phi_j / norms[-1]
        phi = np.kron(phi, phi_j)
        x = x[2 * d:]
    return phis, phi, np.array(norms)
