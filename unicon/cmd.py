import numpy as np


def cb_cmd_vel(
    states_input,
    states_cmd,
    states_rpy,
    range_lin_vel_x=None,
    range_lin_vel_y=None,
    range_ang_vel_yaw=None,
    # step_vel=True,
    step_vel=False,
    step_vel_size=0.01,
    eps=0.05,
    input_keys=None,
    cmd=[0.0, 0.0, 0.0],
    init_range_pt=0,
    num_ranges=4,
    max_scale=1.2,
    min_vel=0.1,
    # num_gait_modes=5,
    num_gait_modes=2,
    init_gait_mode=0,
    # init_gait_mode=1,
    # env_cfg=None,
    cmd_def_vals=None,
    cmd_keys=None,
    cmd_ranges=None,
    enable_gait_modes=None,
    # cmd_ranges=None,
    # cmd_keys=None,
    enable_extra_commands=None,
    extra_cmd_w_coef=0.8,
    num_vel_cmds=3,
    use_dpad=True,
    dpad_step=0.02,
    # no_vy_cmd=True,
    no_vy_cmd=False,
    use_pyaw_cmd=False,
    pyaw_cmd_kp=1.,
    pyaw_cmd_init=None,
):
    from unicon.utils import coalesce, get_ctx
    ctx = get_ctx()
    inp_min = -1.
    inp_max = 1.
    import numpy as np
    import time
    cmd_keys = coalesce(cmd_keys, ctx.get('env_command_keys'))
    cmd_ranges = coalesce(cmd_ranges, ctx.get('env_command_ranges'))
    cmd_def_vals = coalesce(cmd_def_vals, ctx.get('env_command_def_vals'))
    num_cmd = len(states_cmd)
    num_commands = len(cmd_keys)
    cmd_ranges = [[-1, 1] for _ in range(num_commands)] if cmd_ranges is None else cmd_ranges
    cmd_min = [r[0] for r in cmd_ranges]
    cmd_max = [r[1] for r in cmd_ranges]
    cmd_min = np.array(cmd_min)
    cmd_max = np.array(cmd_max)
    cmd_span = cmd_max - cmd_min
    inp_span = inp_max - inp_min
    w_cmd = cmd_span / inp_span
    b_cmd = (cmd_min * inp_span - inp_min * cmd_span) / inp_span
    orig_ids = [cmd_keys.index(k) for k in cmd_def_vals if k in cmd_keys]
    orig_vals = [v for k, v in cmd_def_vals.items() if k in cmd_keys]
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
    input_keys = coalesce(ctx.get('input_keys'), input_keys)

    idx_vx = input_keys.index('ABS_Y')
    idx_vy = input_keys.index('ABS_X')
    idx_vyaw = input_keys.index('ABS_RX')
    idx_pyaw = input_keys.index('ABS_RY')
    idx_btn_a = input_keys.index('BTN_A')
    idx_btn_b = input_keys.index('BTN_B')
    idx_btn_x = input_keys.index('BTN_X')
    idx_btn_y = input_keys.index('BTN_Y')
    idx_btn_select = input_keys.index('BTN_SELECT')
    # idx_btn_start = input_keys.index('BTN_START')

    idx_btn_pyaw = idx_btn_select

    if range_lin_vel_x is None:
        rng_lin_vel_x = cmd_ranges[cmd_keys.index('lin_vel_x')]
        rng_lin_vel_y = cmd_ranges[cmd_keys.index('lin_vel_y')]
        rng_ang_vel_yaw = cmd_ranges[cmd_keys.index('ang_vel_yaw')]
        scales = [max_scale * (i + 1) / num_ranges for i in range(num_ranges)]
        # scales = [(i+2)/(num_ranges+1) for i in range(num_ranges)]
        vel_ranges = rng_lin_vel_x, rng_lin_vel_y, rng_ang_vel_yaw
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
        init_gait_mode = (num_gait_modes + init_gait_mode) if init_gait_mode < 0 else init_gait_mode
        # states_cmd[-1] = init_gait_mode
        mode_pt = init_gait_mode
        print('gait_modes', init_gait_mode, num_gait_modes)

    if enable_extra_commands is None:
        _dim = 1 if enable_gait_modes else 0
        enable_extra_commands = num_commands > num_vel_cmds and num_cmd == num_commands + _dim
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
        w_extra = w_cmd[extra_beg:] * extra_cmd_w_coef
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

    if use_pyaw_cmd:
        # pyaw_cmd = states_rpy[2] if pyaw_cmd_init is None else pyaw_cmd_init
        pyaw_cmd = pyaw_cmd_init
        print('use_pyaw_cmd', pyaw_cmd, states_rpy)

    states_input_hist = np.zeros(len(states_input), dtype=np.int32)

    update_ctrl_wb()

    def wrap(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    def cb():
        states_input_hist[:] = np.clip(states_input_hist / 2 + states_input * 2, -3, 3)
        nonlocal cmd, ctrl_lin_vel_x, ctrl_lin_vel_y, ctrl_ang_vel_yaw, range_pt, last_chg, mode_pt
        nonlocal pyaw_cmd, pyaw_cmd_init
        inp_vx = states_input[idx_vx]
        inp_vy = states_input[idx_vy]
        inp_vyaw = states_input[idx_vyaw]
        inp_vx, inp_vy, inp_vyaw = [0 if abs(x) < eps else x for x in (inp_vx, inp_vy, inp_vyaw)]
        inp_vy = 0 if no_vy_cmd else inp_vy
        if step_vel:
            ctrl_lin_vel_x = ctrl_lin_vel_x + step_vel_size * -inp_vx
            ctrl_lin_vel_y = ctrl_lin_vel_y + step_vel_size * -inp_vy
            ctrl_ang_vel_yaw = ctrl_ang_vel_yaw + step_vel_size * -inp_vyaw
            if states_input[idx_btn_a] == 1:
                ctrl_lin_vel_x = ctrl_lin_vel_y = ctrl_ang_vel_yaw = 0
        else:
            ctrl_lin_vel_x = -inp_vx
            ctrl_lin_vel_y = -inp_vy
            ctrl_ang_vel_yaw = -inp_vyaw
        cmd[0] = ctrl_lin_vel_x
        cmd[1] = ctrl_lin_vel_y
        cmd[2] = ctrl_ang_vel_yaw
        if use_pyaw_cmd and pyaw_cmd is not None and abs(inp_vyaw) < 0.8:
            inp_pyaw = states_input[idx_pyaw]
            if abs(inp_pyaw) > eps and states_input[idx_btn_pyaw] > 0:
                pyaw_cmd = wrap(inp_pyaw * step_vel_size + pyaw_cmd)
                if int(pyaw_cmd * 100) % 10 == 0:
                    print('pyaw_cmd', pyaw_cmd)
            cur_yaw = states_rpy[2]
            yaw_error = wrap(pyaw_cmd - wrap(cur_yaw))
            vyaw_cmd = pyaw_cmd_kp * yaw_error
            cmd[2] += vyaw_cmd

        cmd[:] = np.clip(cmd, -1., 1.) * ctrl_w
        out_cmd = cmd + cmd_const[:num_vel_cmds]
        out_cmd[np.abs(out_cmd) < min_vel] = 0
        states_cmd[:num_vel_cmds] = out_cmd
        # chg = time.time() - last_chg > 1
        chg = True
        if (states_input_hist[idx_btn_a] == 2 or states_input_hist[idx_btn_b] == 2) and chg:
            d = 1 if states_input_hist[idx_btn_a] else -1
            range_pt = (range_pt + d + num_ranges) % num_ranges
            update_ctrl_wb()
            if use_pyaw_cmd and range_pt == num_ranges - 1:
                pyaw_cmd_init = states_rpy[2] if pyaw_cmd_init is None else pyaw_cmd_init
                pyaw_cmd = wrap(pyaw_cmd_init)
                print('pyaw_cmd reset', pyaw_cmd, pyaw_cmd_init)
            last_chg = time.time()
        if enable_gait_modes:
            if (states_input_hist[idx_btn_x] == 2 or states_input_hist[idx_btn_y] == 2) and chg:
                d = 1 if states_input_hist[idx_btn_x] else -1
                mode_pt = (mode_pt + d + num_gait_modes) % num_gait_modes
                print('gait mode_pt', mode_pt)
                last_chg = time.time()
            states_cmd[-1] = mode_pt
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
            inp_extra = 0
            _inp_extra = 0
            if extra_cmd_pt != -1:
                cmd_pt = extra_beg + extra_cmd_pt
                _inp_extra = -1 * states_input[idx_extra_cmd]
                _inp_extra = 0 if abs(_inp_extra) < eps else _inp_extra
                # if inp_extra == 0:
                #     return
                if use_dpad:
                    inp_extra = np.clip(input_extra[extra_cmd_pt] + _inp_extra * dpad_step, inp_min, inp_max)
                else:
                    inp_extra = _inp_extra
                    # inp_extra = input_extra[extra_cmd_pt] + inp_extra * 0.02
                input_extra[extra_cmd_pt] = inp_extra
            states_cmd[extra_inds] = np.clip(input_extra * w_extra + b_extra + cmd_const[extra_inds],
                                             cmd_min[extra_inds], cmd_max[extra_inds])
            if _inp_extra != 0 and int(round(inp_extra, 2) * 100) % 10 == 0:
                print('input_extra', extra_cmd_pt, round(inp_extra, 2), cmd_keys[cmd_pt], round(states_cmd[cmd_pt], 3))
        # print('states_cmd', states_cmd)

    return cb


def cb_cmd_wb(
    states_input,
    states_cmd,
    axis_names=None,
    axis_cmd_keys=None,
    axis_cmd_inds=None,
    axis_dir=None,
    cmd_def_vals=None,
    cmd_keys=None,
    cmd_ranges=None,
    input_keys=None,
    input_key_reset='BTN_A',
    input_key_gait_next='BTN_X',
    dpad_step=0.02,
    num_gait_modes=2,
    init_gait_mode=0,
    enable_gait_mode=True,
):
    from unicon.utils import coalesce, get_ctx, is_edge
    ctx = get_ctx()

    input_keys = coalesce(get_ctx().get('input_keys'), input_keys)
    num_cmds = len(states_cmd)
    if axis_names is None:
        axis_names = [
            'ABS_X',
            'ABS_Y',
            'ABS_RX',
            'ABS_RY',
            'ABS_HAT0X',
            'ABS_HAT0Y',
        ]
    cmd_keys = coalesce(cmd_keys, ctx.get('env_command_keys'))
    cmd_ranges = coalesce(cmd_ranges, ctx.get('env_command_ranges'))
    cmd_def_vals = coalesce(cmd_def_vals, ctx.get('env_command_def_vals'))
    num_commands = len(cmd_keys)
    if axis_cmd_keys is None:
        axis_cmd_keys = ['ang_vel_yaw', 'lin_vel_x', 'body_roll', 'body_pitch', 'body_yaw', 'body_height']
    if axis_cmd_inds is None:
        axis_cmd_inds = [(cmd_keys.index(x) if x in cmd_keys else None) for x in axis_cmd_keys]
    axis_names = [n for n, x in zip(axis_names, axis_cmd_inds) if x is not None]
    axis_cmd_inds = [x for x in axis_cmd_inds if x is not None]
    num_inputs = len(states_input)
    axis_inds = [input_keys.index(n) for n in axis_names]
    num_axes = len(axis_names)
    axis_dir = ([-1] * num_inputs) if axis_dir is None else axis_dir
    cmd_w = np.zeros((num_commands, num_axes))
    cmd_b = np.zeros(num_commands)
    inp_min = -1
    inp_max = +1
    inp_span = inp_max - inp_min
    if cmd_ranges is None:
        cmd_ranges = [[-1, 1] for _ in range(num_commands)]
    cmd_ranges = np.array(cmd_ranges)
    cmd_min = cmd_ranges[:, 0]
    cmd_max = cmd_ranges[:, 1]
    cmd_span = cmd_max - cmd_min
    cmd_b[:] = cmd_min + cmd_span * 0.5
    if cmd_def_vals is not None:
        cmd_def_zero_inds = [cmd_keys.index(n) for n in cmd_def_vals if n in cmd_keys]
        cmd_b[cmd_def_zero_inds] = list(cmd_def_vals.values())
        # cmd_w[cmd_def_zero_inds] *= 2
    input_states = np.zeros_like(states_input)
    last_input = np.zeros_like(states_input)
    hat_inds = [input_keys.index(n) for n in input_keys if 'HAT' in n]
    inp_reset = input_keys.index(input_key_reset)
    inp_gait_next = input_keys.index(input_key_gait_next)
    gait_mode = init_gait_mode

    enable_gait_mode = enable_gait_mode and (num_commands < num_cmds)
    print('cb_cmd_wb', num_inputs, num_axes, num_cmds, num_commands, enable_gait_mode)

    def update_wb():
        # print('axis_cmd_inds', axis_cmd_inds)
        # print('axis_inds', axis_inds)
        for wi, (ci, ai) in enumerate(zip(axis_cmd_inds, axis_inds)):
            w = cmd_span[ci] / inp_span * axis_dir[ai]
            cmd_w[ci, wi] = w
            # cmd_b[ci] = (cmd_min[ci] * inp_span - inp_min * cmd_span[ci]) / inp_span
        print(f'cmd_w {cmd_w.shape}\n{cmd_w}')
        print(f'cmd_b {cmd_b.shape}\n{cmd_b}')

    update_wb()

    def cb():
        s_hat = input_states[hat_inds]
        input_states[:] = states_input
        input_states[hat_inds] = np.clip(states_input[hat_inds] * dpad_step + s_hat, -1, 1)
        cmd = cmd_w @ input_states[axis_inds] + cmd_b
        cmd = np.clip(cmd, cmd_min, cmd_max)
        states_cmd[:len(cmd)] = cmd

        if is_edge(last_input, states_input, inp_reset) > 0:
            input_states[:] = 0.
            print('input_states reset')

        if is_edge(last_input, states_input, inp_gait_next) > 0:
            nonlocal gait_mode
            gait_mode = (gait_mode + 1) % num_gait_modes
            print('gait_mode', gait_mode)
        if enable_gait_mode:
            states_cmd[-1] = gait_mode

        last_input[:] = states_input

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
    loop=False,
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
            if loop:
                pt = 0
            elif exit_on_end:
                return True
            else:
                states_cmd[:] = 0
        cmd = frames[pt]
        states_cmd[:len(cmd)] = cmd

    return cb


def cb_cmd_rec(
    states_input,
    states_cmd,
    input_keys=None,
    rec_mean=True,
    loop=True,
    verbose=True,
    # rec_inds=None,
    rec_inds=[0, 1, 2],
    key_rec='BTN_TR',
    key_play='BTN_START',
):
    from unicon.utils import list2slice, is_edge, coalesce, get_ctx
    input_keys = coalesce(get_ctx().get('input_keys'), input_keys)
    is_rec = False
    is_play = False
    rec_inds = list2slice(rec_inds)
    max_frames = 2**16
    frames = np.zeros((max_frames, len(states_cmd[rec_inds])), dtype=states_cmd.dtype)
    rec_pt = -1
    play_pt = -1
    idx_rec = input_keys.index(key_rec)
    # key_play = 'BTN_TR'
    idx_play = input_keys.index(key_play)

    def cb():
        nonlocal play_pt, rec_pt, is_rec, is_play
        rec_p = is_edge(states_input, idx_rec) > 0
        if rec_p:
            is_play = False
            is_rec = not is_rec
            if rec_pt > -1:
                if is_rec:
                    rec_pt = -1
                elif rec_mean:
                    mean = np.mean(frames[:rec_pt + 1], axis=0)
                    frames[0] = mean
                    print('rec_mean', rec_pt, mean)
                    rec_pt = 0
            print('rec_p', is_rec, rec_pt)
        if is_rec:
            rec_pt += 1
            rec_pt = min(max_frames - 1, rec_pt)
            if rec_pt % 100 == 0:
                print('rec_pt', rec_pt)
            frames[rec_pt] = states_cmd[rec_inds]
            return
        play_p = is_edge(states_input, idx_play) > 0
        if play_p:
            is_rec = False
            is_play = not is_play
            if is_play and play_pt > -1:
                play_pt = -1
            print('play_p', is_play, play_pt, rec_pt)
        if is_play and rec_pt > -1:
            play_pt += 1
            if rec_mean:
                cmd = frames[0]
            elif play_pt > rec_pt:
                if verbose:
                    print('cb_cmd_rec play end', play_pt)
                if loop:
                    play_pt = 0
                else:
                    is_play = False
                    return
                cmd = frames[play_pt]
            if (play_pt + 1) % 100 == 0:
                print('play_pt', play_pt, cmd)
            states_cmd[rec_inds] += cmd

    return cb
