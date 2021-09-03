import numpy as np
import time
from qtb_tests import qtb_tests
from utils.qtb_state import qtb_state
import utils.qtb_tools as tools
import utils.qtb_stats as stats
from utils.qtb_result import Result


def qtb_analyze(fun_proto, fun_est, dim: list, tests="all", mtype="povm",
                name="Untitled QT-method", max_nsample=np.inf,
                display=True, filename="none", savefreq=1,
                par_job_id=None, par_job_num=None):

    if display:
        print("Initialization...")

    result = Result(filename, dim, display)
    if par_job_id is not None:
        result.par_init(par_job_id, par_job_num)
    result.load()
    result.set_name(name)
    result.set_cpu()

    # Prepare tests
    test_desc = qtb_tests(dim)
    test_codes = [*test_desc] if tests == "all" else tests
    if type(test_codes) is str:
        test_codes = [test_codes]

    for tcode in test_codes:
        test = test_desc[tcode]
        test["nsample"] = list(filter(lambda ntot: ntot <= max_nsample, test["nsample"]))
        result.init_test(tcode, test)
    result.save()

    # Perform tests
    nb = 0
    for j_test, tcode in enumerate(test_codes):
        test = result.tests[tcode]

        if display:
            print("===> Running test {}/{}: {} ({})".format(j_test+1, len(test_codes), test["name"], test["code"]))
            nb = tools.uprint("")

        for exp_id, experiment in enumerate(result.experiments(tcode)):
            if not np.isnan(experiment["fidelity"][0]):  # experiment results loaded
                continue
            seed = test["seed"] + experiment["exp_num"]
            stats.set_state(seed)
            dm = qtb_state(dim, **test["generator"])
            state = stats.get_state()
            for ntot_id, ntot in enumerate(test["nsample"]):
                if display:
                    nb = tools.uprint(
                        "Experiment {}/{}, nsamples = 1e{:.0f}"
                        .format(experiment["exp_num"], test["nexp"], np.round(np.log10(ntot))),
                        nb)
                stats.set_state(state)
                data, meas, experiment["time_proto"][ntot_id], sm_flag = conduct_experiment(dm, ntot, fun_proto, dim, mtype)
                tc = time.time()
                dm_est = tools.call(fun_est, meas, data, dim)
                experiment["time_est"][ntot_id] = time.time()-tc
                f, msg = tools.isdm(dm_est)
                if not f:
                    raise ValueError("Estimator error: {}".format(msg))
                experiment["nmeas"][ntot_id] = len(meas)
                experiment["fidelity"][ntot_id] = tools.fidelity(dm, dm_est)
                experiment["sm_flag"] = experiment["sm_flag"] and sm_flag

            result.update(tcode, exp_id, experiment)
            if (exp_id+1) % savefreq == 0 or experiment["is_last"]:
                result.save()

        if display:
            tools.uprint("Done", nb, end="\n")

    return result


def conduct_experiment(dm, ntot, fun_proto, dim, mtype="povm"):
    data = []
    meas = []
    time_proto = 0
    sm_flag = True
    jn = 0
    while jn < ntot:
        tc = time.time()
        meas_curr = tools.call(fun_proto, jn, ntot, meas, data, dim)
        time_proto += time.time()-tc
        if not "nshots" in meas_curr:
            meas_curr["nshots"] = 1
        if jn + meas_curr["nshots"] > ntot:
            raise ValueError("Number of measurements exceeds available sample size")
        sm_flag = sm_flag and np.all(tools.isprod(meas_curr[mtype], dim))
        data.append(get_data(dm, meas_curr, mtype))
        meas.append(meas_curr)
        jn += meas_curr["nshots"]
    return data, meas, time_proto, sm_flag


def get_data(dm, meas_curr, mtype="povm"):
    tol = 1e-8
    if mtype == "povm":
        m = meas_curr["povm"].shape[0]
        prob = np.empty((m,))
        for j_povm in range(m):
            prob[j_povm] = np.real(np.trace(dm.dot(meas_curr["povm"][j_povm,:,:])))
        extraop = False
        probsum = np.sum(prob)
        if np.any(prob < 0):
            if np.any(prob < -tol):
                raise ValueError("Measurement operators are not valid: negative eigenvalues exist")
            prob[prob<0] = 0
            probsum = np.sum(prob)
        if probsum > 1+tol:
            raise ValueError("Measurement operators are not valid: total probability is greater than 1")
        if probsum < 1-tol:
            extraop = True
            prob = np.append(prob, 1-probsum)

        clicks = stats.sample(prob, meas_curr["nshots"])
        if extraop:
            clicks = clicks[0:-1]

        return clicks

    elif mtype == "operator":
        meas_curr["povm"] = meas_curr["operator"]
        clicks = get_data(dm, meas_curr, mtype="povm")
        return clicks[0]

    elif mtype == "observable":
        w, v = np.linalg.eig(meas_curr["observable"])
        M = [ np.outer(v[:,i],v[:,i].conj()) for i in range(v.shape[1]) ]
        meas_curr["povm"] = np.stack(tuple(M), axis=0)
        clicks = get_data(dm, meas_curr, mtype="povm")
        return np.sum(clicks*w)/meas_curr["nshots"]

    else:
        raise ValueError("Unknown measurement type")
