import numpy as np


def cb_cmd_vel(
    states_input,
    states_cmd,
    range_lin_vel_x=[[-.5, .5], [-.75, .75], [-1., 1.]],
    range_lin_vel_y=[[-.25, .25], [-.5, .5], [-.6, .6]],
    range_ang_vel_yaw=[[-.5, .5], [-.75, .75], [-1., 1.]],
    # step_vel=True,
    step_vel=False,
    step_vel_size=0.01,
    eps=0.01,
    input_keys=None,
    cmd=[0.0, 0.0, 0.0],
    init_range_pt=0,
    num_ranges=4,
    max_scale=1.2,
    min_vel=0.15,
    lin_vel_x=None,
    lin_vel_y=None,
    ang_vel_yaw=None,
    num_modes=5,
    init_mode=0,
    env_cfg=None,
    # init_mode=-1,
):
    if lin_vel_x is None and env_cfg is not None:
        ranges = env_cfg['commands']['ranges']
        print('env_cfg cmd ranges', ranges)
        lin_vel_x = ranges['lin_vel_x']
        lin_vel_y = ranges['lin_vel_y']
        ang_vel_yaw = ranges['ang_vel_yaw']
    import numpy as np
    input_keys = __import__('unicon.inputs').inputs._default_input_keys if input_keys is None else input_keys
    used_keys = ['ABS_X', 'ABS_Y', 'ABS_RX', 'BTN_A', 'BTN_B', 'BTN_X', 'BTN_Y', 'BTN_TL', 'BTN_TR']
    idx_abs_x, idx_abs_y, idx_abs_rx, idx_btn_a, idx_btn_b, idx_btn_x, idx_btn_y, idx_btn_tl, idx_btn_tr = [
        input_keys.index(k) for k in used_keys
    ]

    assert lin_vel_x is not None
    lin_vel_x = np.array(lin_vel_x).astype(np.float64)
    lin_vel_y = np.array(lin_vel_y).astype(np.float64)
    ang_vel_yaw = np.array(ang_vel_yaw).astype(np.float64)
    # scales = [(i+2)/(num_ranges+1) for i in range(num_ranges)]
    scales = [max_scale * (i + 1) / num_ranges for i in range(num_ranges)]
    range_lin_vel_x = [[s * x for x in lin_vel_x] for s in scales]
    range_lin_vel_y = [[s * x for x in lin_vel_y] for s in scales]
    range_ang_vel_yaw = [[s * x for x in ang_vel_yaw] for s in scales]

    print('range_lin_vel_x', range_lin_vel_x)
    print('range_lin_vel_y', range_lin_vel_y)
    print('range_ang_vel_yaw', range_ang_vel_yaw)

    import time
    import numpy as np
    ctrl_w = None
    range_pt = init_range_pt
    num_ranges = len(range_lin_vel_x)
    last_chg = 0
    num_vel_cmds = 3

    def update_ctrl_wb():
        nonlocal ctrl_w
        ranges = [r[range_pt] for r in [range_lin_vel_x, range_lin_vel_y, range_ang_vel_yaw]]
        print('cb_cmd_vel ranges', ranges)
        mins = [x[0] for x in ranges]
        mins = np.array(mins, dtype=np.float32)
        # min_lin_vel_x, min_lin_vel_y, min_ang_vel_yaw = mins
        maxs = [x[1] for x in ranges]
        maxs = np.array(maxs, dtype=np.float32)
        # max_lin_vel_x, max_lin_vel_y, max_ang_vel_yaw = maxs
        span = maxs - mins
        # span = 2 * torch.max(torch.stack([mins, maxs], dim=-1).abs(), dim=0)
        ctrl_span = 2.
        ctrl_w = span / ctrl_span

    cmd_const = np.zeros(num_vel_cmds, dtype=np.float32)
    cmd_const[:len(cmd)] = cmd
    cmd = np.zeros(num_vel_cmds, dtype=np.float32)

    ctrl_lin_vel_x = 0
    ctrl_lin_vel_y = 0
    ctrl_ang_vel_yaw = 0

    update_ctrl_wb()

    if len(states_cmd) > 3:
        init_mode = num_modes + init_mode if init_mode < 0 else init_mode
        states_cmd[3] = init_mode

    def clamp(x, lo=0, hi=1):
        return min(x, max(x, lo), hi)

    def cb():
        nonlocal cmd, ctrl_lin_vel_x, ctrl_lin_vel_y, ctrl_ang_vel_yaw, range_pt, last_chg
        abs_x = states_input[idx_abs_x]
        abs_y = states_input[idx_abs_y]
        abs_rx = states_input[idx_abs_rx]
        abs_x, abs_y, abs_rx = [0 if abs(x) < eps else x for x in (abs_x, abs_y, abs_rx)]
        if step_vel:
            ctrl_lin_vel_x = ctrl_lin_vel_x + step_vel_size * abs_y
            ctrl_lin_vel_y = ctrl_lin_vel_y + step_vel_size * abs_rx
            ctrl_ang_vel_yaw = ctrl_ang_vel_yaw + step_vel_size * abs_x
            if states_input[idx_btn_a] == 1:
                ctrl_lin_vel_x = ctrl_lin_vel_y = ctrl_ang_vel_yaw = 0
        else:
            ctrl_lin_vel_x = abs_y
            ctrl_lin_vel_y = abs_rx
            ctrl_ang_vel_yaw = abs_x
        ctrl_lin_vel_x, ctrl_lin_vel_y, ctrl_ang_vel_yaw = [
            clamp(x) for x in [ctrl_lin_vel_x, ctrl_lin_vel_y, ctrl_ang_vel_yaw]
        ]
        cmd[0] = -ctrl_lin_vel_x
        cmd[1] = -ctrl_lin_vel_y
        cmd[2] = -ctrl_ang_vel_yaw
        cmd *= ctrl_w
        out_cmd = cmd + cmd_const
        out_cmd[np.abs(out_cmd) < min_vel] = 0
        states_cmd[:num_vel_cmds] = out_cmd
        chg = time.time() - last_chg > 1
        if (states_input[idx_btn_a] == 1 or states_input[idx_btn_b] == 1) and chg:
            d = 1 if states_input[idx_btn_a] else -1
            range_pt = (range_pt + d + num_ranges) % num_ranges
            update_ctrl_wb()
            last_chg = time.time()
        if len(states_cmd) > 3:
            if (states_input[idx_btn_x] == 1 or states_input[idx_btn_y] == 1) and chg:
                d = 1 if states_input[idx_btn_x] else -1
                mode_pt = (states_cmd[3] + d + num_modes) % num_modes
                print('mode_pt', mode_pt)
                states_cmd[3] = mode_pt
                last_chg = time.time()

    return cb


def cb_cmd_const(
    states_input,
    states_cmd,
    input_keys=None,
    # cmd=[1.0, 0.0, 0.0],
    cmd=[0.5, 0.0, 0.0],
    # cmd=[0.0, 0.0, 0.0],
):

    def cb():
        states_cmd[:len(cmd)] = cmd

    return cb


_default_cmd_list = [
    [+1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, +1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, +1.0],
    [0.0, 0.0, -1.0],
]


def cb_cmd_list(
    states_input,
    states_cmd,
    input_keys=None,
    cmd_list=None,
    cmd_steps=150,
    cycle=True,
    verbose=True,
):
    cmd_list = _default_cmd_list if cmd_list is None else cmd_list
    cmd_list = np.array(cmd_list)
    pt = -1
    cmd_pt = -1
    cmd = None

    def cb():
        nonlocal pt, cmd_pt, cmd
        pt += 1
        if pt % cmd_steps == 0:
            cmd_pt += 1
            if cmd_pt == len(cmd_list):
                cmd_pt = 0 if cycle else len(cmd_list) - 1
            cmd = cmd_list[cmd_pt]
            if verbose:
                print('cb_cmd_list next', cmd_pt, cmd)
        states_cmd[:len(cmd)] = cmd

    return cb


def cb_cmd_replay(
    states_input,
    states_cmd,
    input_keys=None,
    frames=None,
    cycle=False,
    exit_on_end=True,
    verbose=True,
    init_pt=0,
):
    init_pt = min(init_pt, len(frames) // 2)
    pt = init_pt - 1

    def cb():
        nonlocal pt
        pt += 1
        if pt == len(frames):
            if verbose:
                print('cb_cmd_replay end', pt)
            if cycle:
                pt = 0
            elif exit_on_end:
                return True
            else:
                states_cmd[:] = 0
        cmd = frames[pt]
        states_cmd[:len(cmd)] = cmd

    return cb
