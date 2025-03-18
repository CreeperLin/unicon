def cb_teleop_q(
    states_q_ctrl,
    states_q,
    states_input,
    axis_inds=[0, 1, 2, 3],
    cmd_idx_q_ctrl_inds=4,
    cmd_idx_axis_inds=5,
    # use_q=True,
    use_q=False,
    axis_delta_q=0.01,
    q_ctrl_inds=None,
):
    import numpy as np
    last_cmd = np.zeros(len(states_input))

    def is_cmd_rising(idx):
        return last_cmd[idx] <= 0 and states_input[idx] > 0

    axis_pt = 0
    num_axes = len(axis_inds)
    q_ctrl_inds = list(range(num_axes)) if q_ctrl_inds is None else q_ctrl_inds

    def cb():
        nonlocal q_ctrl_inds, axis_pt
        if is_cmd_rising(cmd_idx_q_ctrl_inds):
            q_ctrl_inds[axis_pt] = (q_ctrl_inds[axis_pt] + 1) % len(states_q_ctrl)
            print(axis_pt, 'axes joint inds', q_ctrl_inds)
        if is_cmd_rising(cmd_idx_axis_inds):
            axis_pt = (axis_pt + 1) % num_axes
            print('setting axis idx', axis_pt)
        axes = states_input[axis_inds]
        ctrl = (states_q if use_q else states_q_ctrl)[q_ctrl_inds] + axis_delta_q * axes
        states_q_ctrl[q_ctrl_inds] = ctrl
        last_cmd[:] = states_input

    return cb
