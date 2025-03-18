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
    inv_mass=5.0,
    use_qdd=True,
    mean=[
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        1.,
    ],
    std=[0.001] * 10,
    std_dof=0.001,
    q_min=-3.0,
    q_max=3.0,
    qd_min=-10.0,
    qd_max=10.0,
    q_ctrl_mix=0.8,
):
    import time
    mean = np.array(mean)
    num_dofs = len(states_q)
    _send_ts = time.time()
    _states_q = np.zeros(num_dofs)
    _states_qd = np.zeros(num_dofs)
    _states_q_ctrl = np.zeros(num_dofs)
    _qdd = None

    def cb_send():
        nonlocal _send_ts, _qdd
        _send_ts = time.time()
        _states_q_ctrl[:] = states_q_ctrl
        if use_qdd:
            _qdd = inv_mass * ((_states_q_ctrl - _states_q) * kp + (-_states_qd) * kd)

    def cb_recv():
        ts = time.time()
        dt = ts - _send_ts
        if _qdd is not None:
            _states_q[:] = _states_q + _states_qd * dt
            _states_qd[:] = _states_qd + _qdd * dt
        else:
            q_new = _states_q * (1 - q_ctrl_mix) + _states_q_ctrl * q_ctrl_mix
            _states_qd[:] = (q_new - _states_q) / dt
            _states_q[:] = q_new
        _states_qd[:] = np.clip(_states_qd, qd_min, qd_max)
        _states_q[:] = np.clip(_states_q, q_min, q_max)
        noise = np.random.randn(num_dofs * 2 + 10)
        states_q[:] = _states_q + noise[:num_dofs] * std_dof
        states_qd[:] = _states_qd + noise[num_dofs:num_dofs * 2] * std_dof
        states_rpy[:] = mean[:3] + noise[num_dofs * 2:num_dofs * 2 + 3] * std[:3]
        states_ang_vel[:] = mean[3:6] + noise[num_dofs * 2 + 3:num_dofs * 2 + 6] * std[3:6]
        quat = mean[6:10] + noise[num_dofs * 2 + 6:num_dofs * 2 + 10] * std[6:10]
        quat = quat / (np.linalg.norm(quat) + 1e-6)
        states_quat[:] = quat

    return cb_recv, cb_send, None
