import numpy as np

def qtb_tests(dim):
    nsub = len(dim)
    Dim = int(np.prod(dim))
    return {
        "rps": {
            "code": "RPS",
            "name": 'Random pure states',
            "seed": 161,
            "nsample": np.power(10, np.array([2,3,4,5,6]) + max(0,nsub-3)),
            "nexp": 1000,
            "rank": 1,
            "generator": {"stype": "haar_dm", "rank": 1}
        },
        "rmspt_2": {
            "code": "RMSPT-2",
            "name": 'Random mixed states by partial tracing: rank-2',
            "seed": 1312,
            "nsample": np.power(10, np.array([2,3,4,5,6]) + max(0,nsub-2)),
            "nexp": 1000,
            "rank": 2,
            "generator": {"stype": "haar_dm", "rank": 2}
        },
        "rmspt_d": {
            "code": "RMSPT-d",
            "name": 'Random mixed states by partial tracing: rank-d',
            "seed": 117218,
            "nsample": np.power(10, np.array([2,3,4,5,6]) + (nsub-1)),
            "nexp": 1000,
            "rank": Dim,
            "generator": {"stype": "haar_dm", "rank": Dim}
        },
        "rnp": {
            "code": "RNP",
            "name": 'Random noisy preparation',
            "seed": 758942,
            "nsample": np.power(10, np.array([2,3,4,5,6]) + (nsub-1)),
            "nexp": 1000,
            "rank": Dim,
            "generator": {"stype": "haar_dm", "rank": 1, "init_err": ("unirnd", 0, 0.05), "depol": ("unirnd", 0, 0.01)}
        }
    }