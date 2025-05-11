import numpy as np


def cb_safety_ctrl(
    states_q_ctrl,
    states_q,
    states_qd,
    check_q=True,
    check_dq=True,
    check_tau=True,
    # check_tau=False,
    check_tau_power=True,
    # check_tau_power=False,
    # check_tau=False,
    q_min=None,
    q_max=None,
    halt=True,
    # halt=False,
    tau_limit=32,
    qd_limit=1.6,
    power_limit=1200,
    power_max_breaks=2**2,
    power_halt=1500,
    power_window=2**3,
    Kp=None,
    Kd=None,
    dq_coef=None,
):
    from unicon.utils import pp_arr
    power_mean = 0
    num_breaks = 0
    print('q_min', q_min)
    print('q_max', q_max)
    cmd_q = states_q_ctrl
    q = states_q
    qd = states_qd
    if check_dq and dq_coef is None:
        m = 1.0
        dq_coef = np.sqrt(Kp / m)
        print('dq_coef', dq_coef)

    def cb():
        nonlocal num_breaks, power_mean
        safe = 1
        if check_q:
            if np.any(cmd_q > q_max) or np.any(cmd_q < q_min):
                print('safety cmd q')
                print('q_min', pp_arr(q_min))
                print('q_max', pp_arr(q_max))
                print('cond', np.logical_or(cmd_q > q_max, cmd_q < q_min))
                safe = 0 if int(check_q) == 1 else safe
        if check_dq:
            dq = dq_coef * (q - cmd_q)
            # print('dq', dq)
            cond = np.abs(dq) > qd_limit
            if np.any(cond):
                print('safety qd')
                print('cond', np.nonzero(cond))
                print('qd_limit', qd_limit)
                print('dq', pp_arr(dq))
                safe = 0 if int(check_dq) == 1 else safe
        if check_tau or check_tau_power:
            tau = Kp * (cmd_q - q) + Kd * (-qd)
            # print('Kp', pp_arr(Kp))
            # print('tau', pp_arr(tau))
            # print('cmd_q', pp_arr(cmd_q))
            # print('q', pp_arr(q))
            # print('qd', pp_arr(qd))
            cond = np.abs(tau) > tau_limit
            if check_tau and np.any(cond):
                print('safety tau')
                print('cond', np.nonzero(cond))
                print('tau_limit', tau_limit)
                print('tau', pp_arr(tau))
                safe = 0 if int(check_tau) == 1 else safe
        if check_tau_power:
            power = np.dot(np.abs(tau), np.abs(qd))
            # print('tau pow', frameno, power, power_mean)
            if power > power_halt:
                num_breaks += 1
                print('safety tau power break', num_breaks, power, power_halt)
                if num_breaks > power_max_breaks:
                    print('safety tau power halt', power, power_halt)
                    safe = 0 if int(check_tau_power) == 1 else safe
            power_mean = (power + power_mean * (power_window - 1)) / power_window
            if power_mean > power_limit:
                print('safety tau power mean', power, power_mean, power_limit)
                safe = 0 if int(check_tau_power) == 1 else safe
            # if power > power_limit:
            # num_breaks += 1
            # print(frameno, 'safety tau power', power, power_limit)
            # if num_breaks > power_max_breaks:
            # return True if halt else False
            # else:
            # num_breaks = num_breaks - 1 if num_breaks else 0
        if safe:
            return None
        print('cmd_q', pp_arr(cmd_q))
        print('q', pp_arr(q))
        return (True if halt else False)

    return cb


def cb_safety_states(
    states_rpy,
    states_qd,
    states_tau=None,
    check_rpy=True,
    # check_power=True,
    check_qd=False,
    check_power=False,
    qd_limit=None,
    power_limit=900,
    max_roll=1.2,
    max_pitch=1.2,
    halt=True,
):
    from unicon.utils import pp_arr
    max_rp = np.array([max_roll, max_pitch])
    rpy = states_rpy
    qd = states_qd
    tau_est = states_tau
    check_power = check_power and states_tau is not None

    def cb():
        safe = 1
        if check_rpy and np.any(np.abs(rpy[0:2]) > max_rp):
            print('safety rpy', rpy)
            safe = 0 if int(check_rpy) == 1 else safe
        if check_qd:
            cond = np.abs(qd) > qd_limit
            if np.any(cond):
                print('safety qd')
                print('cond', np.where(cond))
                print('qd', pp_arr(qd))
                print('qd_limit', pp_arr(qd_limit))
                safe = 0 if int(check_qd) == 1 else safe
        if check_power:
            power = np.dot(np.abs(tau_est), np.abs(qd))
            # print('pow', power)
            if power > power_limit:
                print('safety power', power, power_limit)
                safe = 0 if int(check_power) == 1 else safe
        return None if safe else (True if halt else False)

    return cb


def cb_safety_integrity_check(
    states_q,
    states_qd,
    states_q_ctrl,
    dof_map=None,
    eps=1e-7,
    halt=False,
    # lock_q_ctrl=True,
    lock_q_ctrl=False,
):
    import numpy as np
    last_v = None
    onset = False
    pt = 0
    states_v = states_qd
    # states_v = states_q
    last_inds = None

    def cb():
        nonlocal last_v, onset, pt, last_inds
        pt += 1
        v = states_v[dof_map]
        if last_v is None:
            last_v = v.copy()
            return
        cond = np.abs(v - last_v) < eps
        last_v[:] = v
        if np.any(cond):
            inds = tuple(np.where(cond)[0])
            if last_inds != inds:
                print('\ncb_safety_integrity_check inds', inds)
            last_inds = inds
            if onset:
                if lock_q_ctrl:
                    states_q_ctrl[dof_map] = states_q[dof_map]
                print('*' if lock_q_ctrl else '+', end='', flush=True)
                return
            onset = True
            print('cb_safety_integrity_check onset', pt)
            if halt:
                return True
        elif onset:
            print('\ncb_safety_integrity_check offset', pt)
            onset = False
            last_inds = None

    return cb


def cb_safety_verify_imu(
    states_rpy,
    states_ang_vel,
    states_quat,
    tol=1e-3,
):
    from unicon.utils import quat2rpy_np

    def cb():
        if np.sum(np.abs(states_quat)) > 0.1:
            rpy2 = quat2rpy_np(states_quat)
            err = np.sum(np.square(rpy2 - states_rpy))
            if err > tol:
                print('cb_safety_verify_imu', rpy2, rpy, err, tol)

    return cb
