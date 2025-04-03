import numpy as np


def cb_fake_recv_send_close(
    states_q_ctrl,
    states_rpy,
    states_ang_vel,
    states_quat,
    states_q,
    states_qd,
    kp=None,
    kd=None,
    qdd_coef=5.0,
    use_qdd=False,
    q_ctrl_min=-3.0,
    q_ctrl_max=3.0,
    qd_limit=10.0,
    q_ctrl_mix=0.6,
    robot_def=None,
    dt=None,
):
    qd_limit = robot_def.get('QD_LIMIT', qd_limit)
    import time
    from unicon.utils import rpy2quat_np
    num_dofs = len(states_q)
    _send_ts = time.time()
    _states_q = np.zeros(num_dofs)
    _states_qd = np.zeros(num_dofs)
    _states_q_ctrl = np.zeros(num_dofs)
    _states_rpy = np.zeros(3)
    _states_ang_vel = np.zeros(3)
    _qdd = np.zeros(num_dofs) if use_qdd else None
    print('fake sys', use_qdd, qdd_coef, q_ctrl_mix)

    def cb_send():
        nonlocal _send_ts, _qdd
        _send_ts = time.monotonic()
        _states_q_ctrl[:] = states_q_ctrl
        if use_qdd:
            _qdd = qdd_coef * ((_states_q_ctrl - _states_q) * kp + (-_states_qd) * kd)

    def cb_recv():
        ts = time.monotonic()
        _dt = ts - _send_ts if dt is None else dt
        if use_qdd:
            _states_q[:] = _states_q + _states_qd * dt
            _states_qd[:] = _states_qd + _qdd * dt
        else:
            q_new = _states_q * (1 - q_ctrl_mix) + _states_q_ctrl * q_ctrl_mix
            _states_qd[:] = (q_new - _states_q) / dt
            _states_q[:] = q_new
        _states_qd[:] = np.clip(_states_qd, -qd_limit, qd_limit)
        _states_q[:] = np.clip(_states_q, q_ctrl_min, q_ctrl_max)
        states_q[:] = _states_q
        states_qd[:] = _states_qd
        states_rpy[:] = _states_rpy
        states_ang_vel[:] = _states_ang_vel
        quat = rpy2quat_np(states_rpy)
        states_quat[:] = quat

    return cb_recv, cb_send, None
