def cb_teleop_q(
    states_q_ctrl,
    states_q,
    states_input,
    axis_names=None,
    cmd_key_q_ctrl_next='BTN_Y',
    cmd_key_axis_next='BTN_B',
    cmd_key_q_ctrl_prev='BTN_A',
    cmd_key_axis_prev='BTN_X',
    # use_q=True,
    use_q=False,
    use_q_ctrl=False,
    axis_delta_q=0.01,
    q_ctrl_inds=None,
    eps=0.1,
    input_keys=None,
):
    axis_names = [
        'ABS_X',
        'ABS_Y',
        'ABS_RX',
        'ABS_RY',
    ] if axis_names is None else axis_names
    input_keys = __import__('unicon.inputs').inputs._default_input_keys if input_keys is None else input_keys
    import numpy as np
    last_cmd = np.zeros(len(states_input))

    def is_cmd_rising(idx):
        return last_cmd[idx] <= 0 and states_input[idx] > 0

    axis_pt = 0
    axis_inds = [input_keys.index(n) for n in axis_names]
    num_axes = len(axis_inds)
    q_ctrl_inds = [1, 0, 0, 0] if q_ctrl_inds is None else q_ctrl_inds

    num_dofs = len(states_q)
    num_dofs_p = num_dofs + 1
    q_ctrl_cur = np.zeros(num_dofs, dtype=np.float32)
    if use_q:
        q_cur = states_q
    elif use_q_ctrl:
        q_cur = states_q_ctrl
    else:
        q_cur = q_ctrl_cur

    _q_ctrl_inds = []
    _axes_inds = []

    cmd_idx_q_ctrl_next = input_keys.index(cmd_key_q_ctrl_next)
    cmd_idx_axis_next = input_keys.index(cmd_key_axis_next)
    cmd_idx_q_ctrl_prev = input_keys.index(cmd_key_q_ctrl_prev)
    cmd_idx_axis_prev = input_keys.index(cmd_key_axis_prev)

    def cb():
        nonlocal q_ctrl_inds, axis_pt
        if is_cmd_rising(cmd_idx_q_ctrl_next):
            q_ctrl_inds[axis_pt] = (q_ctrl_inds[axis_pt] + 1) % num_dofs_p
            print(axis_pt, 'axes joint inds', q_ctrl_inds)
        if is_cmd_rising(cmd_idx_axis_next):
            axis_pt = (axis_pt + 1) % num_axes
            print('setting axis idx', axis_pt)
            print('q_ctrl_cur', np.round(q_ctrl_cur.astype(np.float64), 2).tolist())
        if is_cmd_rising(cmd_idx_q_ctrl_prev):
            q_ctrl_inds[axis_pt] = (q_ctrl_inds[axis_pt] - 1 + num_dofs_p) % num_dofs_p
            print(axis_pt, 'axes joint inds', q_ctrl_inds)
        if is_cmd_rising(cmd_idx_axis_prev):
            axis_pt = (axis_pt - 1 + num_axes) % num_axes
            print('setting axis idx', axis_pt)
            print('q_ctrl_cur', np.round(q_ctrl_cur.astype(np.float64), 2).tolist())
        axes = states_input[axis_inds]
        _q_ctrl_inds.clear()
        _axes_inds.clear()
        for i, x in enumerate(q_ctrl_inds):
            if x > 0:
                _q_ctrl_inds.append(x - 1)
                _axes_inds.append(i)
        _axes = axes[_axes_inds]
        if np.any(np.abs(_axes) > eps) and len(_q_ctrl_inds):
            ctrl = q_cur[_q_ctrl_inds] + axis_delta_q * _axes
            print(
                'cb_teleop_q',
                _q_ctrl_inds,
                np.round(states_q_ctrl[_q_ctrl_inds].astype(np.float64), 2).tolist(),
                np.round(ctrl.astype(np.float64), 2).tolist(),
            )
            q_ctrl_cur[_q_ctrl_inds] = ctrl
        states_q_ctrl[:] = q_ctrl_cur
        last_cmd[:] = states_input

    return cb
