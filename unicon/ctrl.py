import numpy as np


def cb_ctrl_q_from_target_id(
    states_q_ctrl,
    states_q_target,
):

    def cb():
        states_q_ctrl[:] = states_q_target

    return cb


def cb_ctrl_q_mask(
    states_q_ctrl,
    default_mask=None,
    ctrl_mask=None,
    default_q_ctrl=None,
):
    default_q_ctrl = np.zeros(len(states_q_ctrl)) if default_q_ctrl is None else default_q_ctrl
    default_mask = ~ctrl_mask if default_mask is None else default_mask

    def cb():
        states_q_ctrl[default_mask] = default_q_ctrl[default_mask]

    return cb


def cb_ctrl_q_from_target_lerp(
    states_q_ctrl,
    states_q,
    states_q_target,
    max_steps=200,
    verbose=False,
    # verbose=True,
    cycle=True,
    fixed_src=True,
    dof_map=None,
    from_q_ctrl=True,
    t_fn=None,
):
    from unicon.utils import pp_arr
    ts = 0
    step_t = 1 / max_steps
    q_start = None
    src = states_q_ctrl if from_q_ctrl else states_q

    t_fns = {
        'exp': lambda t: 1 - np.exp(-10 * t),
        'hyp': lambda t: 1 - 1 / (30 * t + 1),
        'atan': lambda t: (2 / np.pi) * np.arctan(10 * t),
    }
    t_fn = t_fns[t_fn] if isinstance(t_fn, str) else t_fn

    def cb():
        nonlocal ts, q_start
        if q_start is None:
            q_start = src.copy() if fixed_src else src
        if ts >= max_steps:
            if verbose:
                print('q', pp_arr(states_q))
                print('target q', pp_arr(states_q_target))
            if cycle:
                ts = 0
                # print('q_cur', pp_arr(q_cur))
                q_start = src.copy() if fixed_src else q_start
            return True
        t = min(ts * step_t, 1)
        t = t if t_fn is None else t_fn(t)
        q_ctrl_lerp = q_start * (1 - t) + t * states_q_target
        if verbose:
            print('ts', ts)
            print('q_start', pp_arr(q_start))
            print('states_q_target', pp_arr(states_q_target))
            print('q_ctrl_lerp', pp_arr(q_ctrl_lerp))
        ts += 1
        states_q_ctrl[dof_map] = q_ctrl_lerp[dof_map]

    return cb


def cb_ctrl_tau_gen(
    states_tau_ctrl,
    states_q_ctrl,
    states_q,
    states_qd,
    states_qd_ctrl=None,
    states_qdd_ctrl=None,
    Kp=None,
    Kd=None,
    tau_limit=None,
    # gear=1.0,
    gain_q=None,
    gain_qd=None,
    gain_qdd=None,
    bias_q=None,
    bias_qd=None,
    bias_qdd=None,
):
    tau_limit = np.array(tau_limit, dtype=np.float32) if tau_limit is not None else None
    gain_q = Kp if Kp is not None else gain_q
    bias_q = -Kp if Kp is not None else bias_q
    bias_qd = -Kd if Kd is not None else bias_qd

    def cb():
        tau = states_q_ctrl * gain_q
        if states_qd_ctrl is not None:
            tau += states_qd_ctrl * gain_qd
        if states_qdd_ctrl is not None:
            tau += states_qd_ctrl * gain_qdd
        tau += states_q * bias_q + states_qd * bias_qd
        if bias_qdd is not None:
            tau += bias_qdd
        # cur = gear * np.stack([ones, states_q, states_qd], axis=-1)
        # tau = gear * ((gain * cur).sum(axis=-1) * q_ctrl + (bias * cur).sum(axis=-1))
        if tau_limit is not None:
            tau = np.clip(tau, -tau_limit, tau_limit)
        states_tau_ctrl[:] = tau

    return cb


def cb_ctrl_tau_q_ctrl_pd(
    states_tau_ctrl,
    states_q_ctrl,
    states_q,
    states_qd,
    Kp,
    Kd,
    tau_limit=None,
):
    tau_limit = np.array(tau_limit, dtype=np.float32) if tau_limit is not None else None

    def cb():
        tau = Kp * (states_q_ctrl - states_q) + Kd * (-states_qd)
        if tau_limit is not None:
            tau = np.clip(tau, tau_limit[0], tau_limit[1])
        states_tau_ctrl[:] = tau

    return cb
