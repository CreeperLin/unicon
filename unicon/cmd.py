import numpy as np


def cb_cmd_vel(
    states_input,
    states_cmd,
    range_lin_vel_x=None,
    range_lin_vel_y=None,
    range_ang_vel_yaw=None,
    # step_vel=True,
    step_vel=False,
    step_vel_size=0.01,
    eps=0.03,
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
    enable_gait_modes=None,
    cmd_ranges=None,
    cmd_keys=None,
    cmd_orig_values=None,
    enable_extra_commands=None,
    num_vel_cmds=3,
    use_dpad=True,
    dpad_step=0.02,
    no_vy_cmd=True,
):
    inp_min = -1.
    inp_max = 1.
    import numpy as np
    import time
    _default_ranges = {
        'frequency': [0.5, 0.5],
        'phase': [0.5, 0.5],
        'duration': [0.5, 0.5],
        'foot_trajectory': [0.18, 0.18],
    }
    _default_cmd_keys = [
        'lin_vel_x',
        'lin_vel_y',
        'ang_vel_yaw',
        'frequency',
        'phase',
        'duration',
        'foot_trajectory',
        'body_height',
        'body_pitch',
        'waist_yaw',
        'waist_roll',
        'waist_pitch',
    ]
    _default_cmd_orig_values = {
        'body_height': 0.,
        'body_pitch': 0.,
        'frequency': 1.2,
    }
    _def_range = [-0.42, 0.42]
    cmd_keys = _default_cmd_keys if cmd_keys is None else cmd_keys
    cmd_orig_values = _default_cmd_orig_values if cmd_orig_values is None else cmd_orig_values
    num_cmd = len(states_cmd)
    num_commands = num_cmd
    if env_cfg is not None:
        env_cfg_env = env_cfg['env']
        observe_frequency = env_cfg_env.get('observe_frequency', True)
        observe_phase = env_cfg_env.get('observe_phase', True)
        observe_duration = env_cfg_env.get('observe_duration', True)
        observe_foot_height = env_cfg_env.get('observe_foot_height', True)
        observe_body_height = env_cfg_env.get('observe_body_height',)
        observe_body_pitch = env_cfg_env.get('observe_body_pitch',)
        observe_waist_roll = env_cfg_env.get('observe_waist_roll',)
        if not observe_phase:
            cmd_keys.pop(cmd_keys.index('phase'))
        if not observe_duration:
            cmd_keys.pop(cmd_keys.index('duration'))
        if not observe_foot_height:
            cmd_keys.pop(cmd_keys.index('foot_trajectory'))
        ranges = env_cfg['commands']['ranges']
        print('env_cfg cmd ranges', ranges)
        lin_vel_x = ranges.get('limit_vel_x', ranges['lin_vel_x'])
        lin_vel_y = ranges.get('limit_vel_y', ranges['lin_vel_y'])
        ang_vel_yaw = ranges.get('limit_vel_yaw', ranges['ang_vel_yaw'])
        cmd_ranges = [ranges[k] if k in ranges else _default_ranges.get(k, _def_range) for k in cmd_keys]
        cmd_min = [r[0] for r in cmd_ranges]
        cmd_max = [r[1] for r in cmd_ranges]
        cmd_min = np.array(cmd_min)
        cmd_max = np.array(cmd_max)
        cmd_span = cmd_max - cmd_min
        num_commands = env_cfg['commands'].get('num_commands', num_cmd)
        inp_span = inp_max - inp_min
        w_cmd = cmd_span / inp_span
        b_cmd = (cmd_min * inp_span - inp_min * cmd_span) / inp_span
        orig_ids = [cmd_keys.index(k) for k in cmd_orig_values]
        orig_vals = [v for v in cmd_orig_values.values()]
        print('orig_ids', orig_ids, orig_vals)
        b_cmd[orig_ids] = orig_vals
        w_cmd[orig_ids] *= 2
        w_cmd = w_cmd[:num_commands]
        b_cmd = b_cmd[:num_commands]
        print('cmd_min', cmd_min.tolist())
        print('cmd_max', cmd_max.tolist())
        print('w_cmd', w_cmd.tolist())
        print('b_cmd', b_cmd.tolist())
    print('cmd_keys', list(enumerate(cmd_keys)))
    print('num_cmd', num_cmd)
    print('num_commands', num_commands)
    input_keys = __import__('unicon.inputs').inputs._default_input_keys if input_keys is None else input_keys

    idx_vx = input_keys.index('ABS_Y')
    idx_vy = input_keys.index('ABS_X')
    idx_vyaw = input_keys.index('ABS_RX')
    idx_btn_a = input_keys.index('BTN_A')
    idx_btn_b = input_keys.index('BTN_B')
    idx_btn_x = input_keys.index('BTN_X')
    idx_btn_y = input_keys.index('BTN_Y')

    if range_lin_vel_x is None:
        scales = [max_scale * (i + 1) / num_ranges for i in range(num_ranges)]
        # scales = [(i+2)/(num_ranges+1) for i in range(num_ranges)]
        vel_ranges = lin_vel_x, lin_vel_y, ang_vel_yaw
        rngs = []
        for rng in vel_ranges:
            rng = [-1, 1] if rng is None else rng
            rng = np.array(rng).astype(np.float64)
            rng = [[s * x for x in rng] for s in scales]
            rngs.append(rng)
        range_lin_vel_x, range_lin_vel_y, range_ang_vel_yaw = rngs

    print('range_lin_vel_x', range_lin_vel_x)
    print('range_lin_vel_y', range_lin_vel_y)
    print('range_ang_vel_yaw', range_ang_vel_yaw)

    cmd_const = np.zeros(num_commands, dtype=np.float32)
    cmd_const[:len(cmd)] = cmd
    print('cmd_const', cmd_const)
    cmd = np.zeros(num_vel_cmds, dtype=np.float32)

    enable_gait_modes = (num_cmd == num_commands + 1) if enable_gait_modes is None else enable_gait_modes
    if enable_gait_modes:
        init_mode = (num_modes + init_mode) if init_mode < 0 else init_mode
        states_cmd[-1] = init_mode
        print('gait_modes', init_mode, num_modes)

    if enable_extra_commands is None:
        enable_extra_commands = num_cmd > num_vel_cmds and num_cmd == num_commands + (1 if enable_gait_modes else 0)
    if enable_extra_commands:
        if use_dpad:
            idx_extra_cmd = input_keys.index('ABS_HAT0Y')
            idx_extra_next = input_keys.index('ABS_HAT0X')
            idx_extra_prev = input_keys.index('ABS_HAT0X')
        else:
            idx_extra_cmd = input_keys.index('ABS_RY')
            idx_extra_prev = input_keys.index('BTN_Y')
            idx_extra_next = input_keys.index('BTN_X')
        num_extra_cmds = num_commands - num_vel_cmds
        extra_cmd_pt = -1
        input_extra = np.zeros(num_extra_cmds, dtype=np.float32)
        extra_beg = num_vel_cmds
        extra_end = num_vel_cmds + num_extra_cmds
        extra_inds = slice(extra_beg, extra_end)
        w_extra = w_cmd[extra_beg:]
        b_extra = b_cmd[extra_beg:]
        print('num_extra_cmds', num_extra_cmds, extra_cmd_pt, idx_extra_cmd, w_extra.shape)
        states_cmd[extra_inds] = input_extra * w_extra + b_extra + cmd_const[extra_beg:]

    ctrl_w = None
    range_pt = init_range_pt
    num_ranges = len(range_lin_vel_x)
    last_chg = 0

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

    ctrl_lin_vel_x = 0
    ctrl_lin_vel_y = 0
    ctrl_ang_vel_yaw = 0

    states_input_hist = np.zeros(len(states_input), dtype=np.int32)

    update_ctrl_wb()

    def clamp(x, lo=0, hi=1):
        return min(x, max(x, lo), hi)

    def cb():
        states_input_hist[:] = np.clip(states_input_hist / 2 + states_input * 2, -3, 3)
        nonlocal cmd, ctrl_lin_vel_x, ctrl_lin_vel_y, ctrl_ang_vel_yaw, range_pt, last_chg
        inp_vx = states_input[idx_vx]
        inp_vy = states_input[idx_vy]
        inp_vyaw = states_input[idx_vyaw]
        inp_vx, inp_vy, inp_vyaw = [0 if abs(x) < eps else x for x in (inp_vx, inp_vy, inp_vyaw)]
        inp_vy = 0 if no_vy_cmd else inp_vy
        if step_vel:
            ctrl_lin_vel_x = ctrl_lin_vel_x + step_vel_size * inp_vx
            ctrl_lin_vel_y = ctrl_lin_vel_y + step_vel_size * inp_vy
            ctrl_ang_vel_yaw = ctrl_ang_vel_yaw + step_vel_size * inp_vyaw
            if states_input[idx_btn_a] == 1:
                ctrl_lin_vel_x = ctrl_lin_vel_y = ctrl_ang_vel_yaw = 0
        else:
            ctrl_lin_vel_x = inp_vx
            ctrl_lin_vel_y = inp_vy
            ctrl_ang_vel_yaw = inp_vyaw
        ctrl_lin_vel_x, ctrl_lin_vel_y, ctrl_ang_vel_yaw = [
            clamp(x) for x in [ctrl_lin_vel_x, ctrl_lin_vel_y, ctrl_ang_vel_yaw]
        ]
        cmd[0] = -ctrl_lin_vel_x
        cmd[1] = -ctrl_lin_vel_y
        cmd[2] = -ctrl_ang_vel_yaw
        cmd *= ctrl_w
        out_cmd = cmd + cmd_const[:num_vel_cmds]
        out_cmd[np.abs(out_cmd) < min_vel] = 0
        states_cmd[:num_vel_cmds] = out_cmd
        # chg = time.time() - last_chg > 1
        chg = True
        if (states_input_hist[idx_btn_a] == 2 or states_input_hist[idx_btn_b] == 2) and chg:
            d = 1 if states_input_hist[idx_btn_a] else -1
            range_pt = (range_pt + d + num_ranges) % num_ranges
            update_ctrl_wb()
            last_chg = time.time()
        if enable_gait_modes:
            if (states_input_hist[idx_btn_x] == 2 or states_input_hist[idx_btn_y] == 2) and chg:
                d = 1 if states_input_hist[idx_btn_x] else -1
                mode_pt = (states_cmd[-1] + d + num_modes) % num_modes
                print('mode_pt', mode_pt)
                states_cmd[-1] = mode_pt
                last_chg = time.time()
        if enable_extra_commands:
            nonlocal extra_cmd_pt
            if (states_input_hist[idx_extra_prev] == -2 or states_input_hist[idx_extra_next] == 2) and chg:
                d = 1 if states_input_hist[idx_extra_next] > 0 else -1
                extra_cmd_pt = (extra_cmd_pt + d + num_extra_cmds) % num_extra_cmds
                cmd_pt = extra_beg + extra_cmd_pt
                print('extra_cmd_pt', extra_cmd_pt, idx_extra_cmd, cmd_keys[cmd_pt], cmd_min[cmd_pt], cmd_max[cmd_pt],
                      input_extra[extra_cmd_pt])
                last_chg = time.time()
                if extra_cmd_pt == 0:
                    print('input_extra cleared')
                    input_extra[:] = 0
                    states_cmd[extra_inds] = np.clip(b_extra + cmd_const[extra_inds], cmd_min[extra_inds],
                                                     cmd_max[extra_inds])
            if extra_cmd_pt == -1:
                return
            cmd_pt = extra_beg + extra_cmd_pt
            inp_extra = -1 * states_input[idx_extra_cmd]
            inp_extra = 0 if abs(inp_extra) < eps else inp_extra
            if inp_extra == 0:
                return
            if use_dpad:
                inp_extra = np.clip(input_extra[extra_cmd_pt] + inp_extra * dpad_step, inp_min, inp_max)
                # inp_extra = input_extra[extra_cmd_pt] + inp_extra * 0.02
            input_extra[extra_cmd_pt] = inp_extra
            states_cmd[extra_inds] = np.clip(input_extra * w_extra + b_extra + cmd_const[extra_inds],
                                             cmd_min[extra_inds], cmd_max[extra_inds])
            if int(inp_extra * 100) % 10 == 0:
                print('input_extra', extra_cmd_pt, round(inp_extra, 3), cmd_keys[cmd_pt], round(states_cmd[cmd_pt], 3))

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
