def parse_env_cfg_h0(env_cfg):
    from unicon.utils import get_ctx
    ctx = get_ctx()
    robot_def = ctx['robot_def']
    NAME = robot_def.get('NAME')
    DOF_NAMES = robot_def.get('DOF_NAMES')
    env_cfg_env = env_cfg['env']
    num_acts = env_cfg_env['num_actions']
    num_obs = env_cfg_env.get('num_partial_obs', env_cfg_env.get('num_observations'))
    env_dof_names = env_cfg.get('dof_names')
    print('env dof_names', env_dof_names)
    if env_dof_names is not None:
        dof_names = env_dof_names
    default_joint_angles = env_cfg['init_state']['default_joint_angles']
    env_kps = env_cfg['control']['stiffness']
    env_kds = env_cfg['control']['damping']
    env_torque_limits = env_cfg['control'].get('torque_limits', None)
    dof_names_std = None
    if 'robot_params' in env_cfg:
        params = env_cfg['robot_params'].get(NAME, None)
        if params is not None:
            default_joint_angles = params.get('default_joint_angles', None)
            print('robot_params', NAME, default_joint_angles)
            env_kps = params.get('stiffness', None)
            env_kds = params.get('damping', None)
            env_torque_limits = params.get('torque_limits', None)
            dof_names_std = params.get('dof_names_std', None)
        elif all(n in default_joint_angles for n in DOF_NAMES):
            print('warning using env_cfg values for robot_params')
        else:
            print('no robot_params', NAME)
            default_joint_angles = None
            env_kps = {}
            env_kds = {}
            env_torque_limits = None
    ctrl_dt = env_cfg['sim']['dt'] * env_cfg['control']['decimation']
    env_cfg_args = env_cfg.get('cmd_args', env_cfg.get('argv', None))
    print('num_acts', num_acts, 'num_obs', num_obs, 'num_dofs', len(dof_names))
    print('env_cfg_args', env_cfg_args)
    env_cfg_commands = env_cfg['commands']
    env_num_commands = env_cfg_commands['num_commands']
    env_cfg_env = env_cfg['env']
    observe_clock_only = env_cfg.get('observe_clock_only', False)
    observe_gait_commands = env_cfg_env.get('observe_gait_commands', True) and (not observe_clock_only)
    observe_frequency = env_cfg_env.get('observe_frequency', observe_gait_commands)
    observe_phase = env_cfg_env.get('observe_phase', observe_gait_commands)
    observe_duration = env_cfg_env.get('observe_duration', observe_gait_commands)
    observe_foot_height = env_cfg_env.get('observe_foot_height', observe_gait_commands)
    observe_body_height = env_cfg_env.get('observe_body_height',)
    observe_body_roll = env_cfg_env.get('observe_body_roll',)
    observe_body_pitch = env_cfg_env.get('observe_body_pitch',)
    observe_body_yaw = env_cfg_env.get('observe_body_yaw',)
    env_command_keys = [
        'lin_vel_x',
        'lin_vel_y',
        'ang_vel_yaw',
        ('gait_frequency' if observe_frequency else None),
        ('phase' if observe_phase else None),
        ('duration' if observe_duration else None),
        ('foot_swing_height' if observe_foot_height else None),
        ('body_height' if observe_body_height else None),
        ('body_roll' if observe_body_roll else None),
        ('body_pitch' if observe_body_pitch else None),
        ('body_yaw' if observe_body_yaw else None),
    ]
    cmd_range_keys = {
        'lin_vel_x': 'limit_vel_x',
        'lin_vel_y': 'limit_vel_y',
        'ang_vel_yaw': 'limit_vel_yaw',
    }
    env_command_keys = [x for x in env_command_keys if x is not None]
    print('env_num_commands', env_num_commands)
    print('env_command_keys', env_command_keys)
    assert len(env_command_keys) == env_num_commands
    ranges = env_cfg_commands['ranges']
    print('env_cfg ranges', ranges)
    def_ranges = {
        'gait_frequency': [env_cfg_commands.get('gait_frequency')] * 2,
        'foot_swing_height': [env_cfg_commands.get('foot_swing_height')] * 2,
        'phase': [0.5] * 2,
        'duration': [env_cfg_commands.get('duty_cycle')] * 2,
    }
    env_range_keys = [cmd_range_keys.get(k, k) for k in env_command_keys]
    env_command_ranges = [
        ranges.get(k, ranges.get(kk, def_ranges.get(k))) for k, kk in zip(env_range_keys, env_command_keys)
    ]
    print('env_command_ranges', env_command_ranges)
    cmd_def_zero_keys = [
        'lin_vel_x',
        'lin_vel_y',
        'ang_vel_yaw',
        'body_height',
        'body_roll',
        'body_pitch',
        'body_yaw',
    ]
    env_command_def_vals = {k: 0 for k in cmd_def_zero_keys}
    print('env_command_def_vals', env_command_def_vals)
    ret = {
        'env_command_keys': env_command_keys,
        'env_command_ranges': env_command_ranges,
        'env_command_def_vals': env_command_def_vals,
        'env_kps': env_kps,
        'env_kds': env_kds,
        'env_torque_limits': env_torque_limits,
        'ctrl_dt': ctrl_dt,
        'num_obs': num_obs,
        'num_acts': num_acts,
        'dof_names': dof_names,
        'env_cfg_args': env_cfg_args,
        # 'dof_names_std': dof_names_std,
        'default_joint_angles': default_joint_angles,
    }
    if dof_names_std is not None:
        ret['dof_names_std'] = dof_names_std
    return ret


def cb_infer_h0(
    states_rpy,
    states_ang_vel,
    states_quat,
    states_q,
    states_qd,
    states_cmd,
    states_q_ctrl,
    policy_fn,
    policy_reset_fn,
    states_infer_acts=None,
    states_infer_obs=None,
    states_rpy2=None,
    env_cfg=None,
    dof_map=None,
    # default_dof_pos=None,
    up_axis_idx=2,
    device='cpu',
    dtype=None,
    use_rpy=None,
    flatten_hist=True,
    # flatten_hist=False,
    retained_obs=False,
    observe_gait_commands=None,
    enable_gait_modes=None,
    adaptive_gait_frequency=None,
    stack_history_obs=None,
    flatten_obs=False,
    # gait_alt0=False,
    gait_alt0=True,
    dof_names=None,
    gravity_from_rpy=True,
):
    from unicon.utils import get_ctx
    from unicon.utils import get_print_fn, expect
    print = get_print_fn()
    ctx = get_ctx()
    robot_def = ctx['robot_def']
    NAME = robot_def['NAME']
    DOF_NAMES = robot_def['DOF_NAMES']
    dof_names_std = ctx.get('dof_names_std', robot_def['DOF_NAMES_STD'])
    import torch
    import numpy as np
    from unicon.utils.torch import to_tensor
    from unicon.utils import get_axis_params, quat_rotate_inverse_np, rpy2mat_np, list2slice
    dtype = torch.float if dtype is None else dtype
    np_dtype = np.float32
    use_rpy = env_cfg.get('use_rpy', True) if use_rpy is None else use_rpy
    flatten_hist = env_cfg['env'].get('flatten_hist', flatten_hist)
    num_actions = env_cfg['env']['num_actions']
    print('use_rpy', use_rpy)
    print('gravity_from_rpy', gravity_from_rpy)
    print('flatten_hist', flatten_hist)
    print('dof_names', dof_names)
    print('num_actions', num_actions)

    apply_dof_dir = env_cfg.get('apply_dof_dir', False)
    no_leg_def_pos = env_cfg.get('no_leg_def_pos', False)
    no_leg_def_pos = no_leg_def_pos and all(x not in NAME for x in ['a1', 'go2'])
    use_base_states = env_cfg.get('use_base_states', False)
    mask_obs_action = env_cfg.get('mask_obs_action', False)

    print('apply_dof_dir', apply_dof_dir)
    print('no_leg_def_pos', no_leg_def_pos)
    print('use_base_states', use_base_states)
    print('mask_obs_action', mask_obs_action)

    robot_params = env_cfg.get('robot_params', {}).get(NAME, None)

    if apply_dof_dir:
        dof_dir_asset = None if robot_params is None else robot_params.get('dof_dir')
        dof_dir_asset = {} if dof_dir_asset is None else robot_params
        dof_dir_asset = [dof_dir_asset.get(k, 1) for k in dof_names]
        dof_dir = np.array(dof_dir_asset, dtype=np_dtype)

    if mask_obs_action:
        dof_names_glb = dof_names
        dof_mask_asset = [n not in dof_names_std for n in dof_names_glb]
        dof_mask_asset = np.array(dof_mask_asset, dtype=np_dtype)
        dof_mask_inv = 1 - dof_mask_asset

    default_joint_angles = env_cfg['init_state']['default_joint_angles']
    print('default_joint_angles', default_joint_angles)
    if 'robot_params' in env_cfg:
        if robot_params is None:
            if all(n in default_joint_angles for n in DOF_NAMES):
                init_joint_angles = default_joint_angles
                print('using default_joint_angles')
            else:
                init_joint_angles = robot_def.get('Q_RESET', {})
                print('using Q_RESET')
        else:
            init_joint_angles = robot_params.get('default_joint_angles', {})
        print('mix env', NAME, init_joint_angles)
    else:
        init_joint_angles = default_joint_angles
    if isinstance(init_joint_angles, dict):
        if no_leg_def_pos:
            for k in init_joint_angles.keys():
                std_name = dof_names_std[k]
                if no_leg_def_pos and any(x in std_name for x in ['hip', 'knee', 'ankle']):
                    init_joint_angles[k] = 0.
        default_dof_pos = np.array([init_joint_angles.get(n, 0.) for n in dof_names])
    else:
        # expect(not no_leg_def_pos)
        if no_leg_def_pos:
            init_joint_angles[:12] = 0.
        default_dof_pos = init_joint_angles[dof_map]
    dof_map_action_dest = dof_map
    clip_actions = env_cfg['normalization']['clip_actions']
    action_scale = env_cfg['control']['action_scale']
    obs_scales = env_cfg['normalization']['obs_scales']
    print('obs_scales', obs_scales)
    scales_lin_vel = obs_scales['lin_vel']
    scales_ang_vel = obs_scales['ang_vel']
    scales_dof_pos = obs_scales['dof_pos']
    scales_dof_vel = obs_scales['dof_vel']
    gait_freq_cmd = obs_scales['gait_freq_cmd']
    gait_phase_cmd = obs_scales['gait_phase_cmd']
    footswing_height_cmd = obs_scales['footswing_height_cmd']
    if observe_gait_commands is None:
        observe_gait_commands = env_cfg['env'].get('observe_gait_commands', True)
    print('observe_gait_commands', observe_gait_commands)
    observe_clock_only = env_cfg.get('observe_clock_only', False)
    print('observe_clock_only', observe_clock_only)
    include_history_steps = env_cfg['env']['include_history_steps']
    if adaptive_gait_frequency is None:
        adaptive_gait_frequency = env_cfg['commands'].get('adaptive_gait_frequency', False)
        print('adaptive_gait_frequency', adaptive_gait_frequency)
    if stack_history_obs is None:
        stack_history_obs = env_cfg['env']['stack_history_obs']
    print('stack_history_obs', stack_history_obs)
    cmds_scale = [
        scales_lin_vel,
        scales_lin_vel,
        scales_ang_vel,
    ]
    default_commands = [
        0.,
        0.,
        0.,
    ]
    cmd_frequency = 1.2
    cmd_phase = 0.5
    env_cfg_env = env_cfg['env']
    observe_frequency = env_cfg_env.get('observe_frequency', observe_gait_commands)
    observe_phase = env_cfg_env.get('observe_phase', observe_gait_commands)
    observe_duration = env_cfg_env.get('observe_duration', observe_gait_commands)
    observe_foot_height = env_cfg_env.get('observe_foot_height', True)
    observe_body_height = env_cfg_env.get('observe_body_height',)
    observe_body_roll = env_cfg_env.get('observe_body_roll',)
    observe_body_pitch = env_cfg_env.get('observe_body_pitch',)
    observe_body_yaw = env_cfg_env.get('observe_body_yaw',)
    observe_waist_roll = env_cfg_env.get('observe_waist_roll',)
    print('observe_frequency', observe_frequency)
    print('observe_phase', observe_phase)
    print('observe_duration', observe_duration)
    print('observe_foot_height', observe_foot_height)
    print('observe_body_height', observe_body_height)
    print('observe_body_roll', observe_body_roll)
    print('observe_body_pitch', observe_body_pitch)
    print('observe_body_yaw', observe_body_yaw)
    print('observe_waist_roll', observe_waist_roll)
    if observe_frequency:
        cmds_scale.append(obs_scales['gait_freq_cmd'])
        cmd_frequency = 1.2
        default_commands.append(cmd_frequency)
    if observe_phase:
        cmds_scale.append(obs_scales['gait_phase_cmd'])
        cmd_phase = 0.5
        default_commands.append(cmd_phase)
    if observe_duration:
        cmds_scale.append(obs_scales.get('gait_duration_cmd', gait_phase_cmd))
        default_commands.append(0.5)
    if observe_foot_height:
        cmds_scale.append(obs_scales['footswing_height_cmd'])
        default_commands.append(0.12)
    if observe_body_height:
        cmds_scale.append(obs_scales['body_height_cmd'])  # Body Height Cmd (1 dim)
        default_commands.append(0.)
        # default_commands.append(0.1)
    if observe_body_roll:
        cmds_scale.append(obs_scales['body_roll_cmd'])
        default_commands.append(0.)
    if observe_body_pitch:
        cmds_scale.append(obs_scales['body_pitch_cmd'])  # Body Pitch Cmd (1 dim)
        default_commands.append(0.)
    if observe_body_yaw:
        cmds_scale.append(obs_scales['body_yaw_cmd'])
        default_commands.append(0.)
    if observe_waist_roll:
        cmds_scale.append(obs_scales['waist_roll_cmd'])  # Waist Roll (1 dim)
        default_commands.append(0.)
    if env_cfg['env'].get('interrupt_in_cmd'):
        cmds_scale.append(1)  # Interrupt Flag (1 dim)
        default_commands.append(0.)
    default_commands.extend([0] * (12 - len(default_commands)))
    cmds_scale = np.array(cmds_scale, dtype=np_dtype)
    print('cmds_scale', cmds_scale)
    gravity_vec = np.array(get_axis_params(-1., up_axis_idx), dtype=np_dtype)
    print('gravity_vec', gravity_vec)
    default_dof_pos_act = np.array(default_dof_pos, dtype=np_dtype)
    default_dof_pos = np.array(default_dof_pos, dtype=np_dtype)
    # default_dof_pos_tensor = to_tensor(default_dof_pos)
    retained_action_inds = env_cfg.get('retained_action_inds', env_cfg['env'].get('retained_actions_indxs'))
    dof_map_obs = dof_map
    retained_actions = env_cfg['env'].get('retained_actions', True)
    retained_actions_only = env_cfg['env'].get('retained_actions_only', False)
    retained_obs = env_cfg.get('retained_obs', retained_obs)
    retained_action_inds = retained_action_inds if retained_actions else None
    print('retained_actions', retained_actions)
    print('retained_actions_only', retained_actions_only)
    print('retained_obs', retained_obs)
    masked_action_inds = None
    if retained_action_inds is not None:
        dof_mask_inv = dof_mask_inv[retained_action_inds]
        _dof_map = list(range(dof_map.start, dof_map.stop)) if isinstance(dof_map, slice) else dof_map
        print('retained_action_inds', len(retained_action_inds), retained_action_inds)
        # default_dof_pos_tensor = default_dof_pos_tensor[dof_map_action_dest]
        if len(retained_action_inds) == num_actions:
            dof_map_action_dest = [_dof_map[i] for i in retained_action_inds]
            # default_dof_pos_tensor = default_dof_pos_tensor[retained_action_inds]
            default_dof_pos_act = default_dof_pos[retained_action_inds]
            # dof_map_action_src = list(range(num_actions))
        else:
            masked_action_inds = [i for i in range(num_actions) if i not in retained_action_inds]
            masked_action_inds = list2slice(masked_action_inds)
            # dof_map_action_src = retained_action_inds
        # print('dof_map_action_src', dof_map_action_src)
        # dof_map_action_src = list2slice(dof_map_action_src)
        # print('default_dof_pos_tensor', default_dof_pos_tensor)
        if retained_obs:
            dof_map_obs = dof_map_action_dest
            # default_dof_pos = default_dof_pos[dof_map_obs]
            default_dof_pos = default_dof_pos[retained_action_inds]
    dof_map_action_dest = list2slice(dof_map_action_dest)
    dof_map_obs = list2slice(dof_map_obs)
    default_dof_pos_act_tensor = to_tensor(default_dof_pos_act, device=device)
    print('masked_action_inds', masked_action_inds)
    print('dof_map_obs', dof_map_obs)
    print('dof_map_action_dest', dof_map_action_dest)
    print('clip_actions', clip_actions)
    print('obs_scales', obs_scales)
    print('action_scale', action_scale)
    print('default_dof_pos', np.round(default_dof_pos.astype(np.float64), decimals=3).tolist())
    # last_actions = torch.zeros(num_actions, dtype=dtype, device=device)
    last_actions = np.zeros(num_actions, dtype=np_dtype)
    last_actions_tensor = torch.zeros(num_actions, device=device)
    hist_buf = None
    env_cfg_env = env_cfg['env']
    num_partial_obs = env_cfg_env.get('num_partial_obs', env_cfg_env.get('num_partial_observations'))
    if stack_history_obs:
        num_partial_obs_full = False
        num_partial_obs = int(num_partial_obs // include_history_steps) if num_partial_obs_full else num_partial_obs
        print('include_history_steps', include_history_steps)
        print('num_partial_obs', num_partial_obs)
        hist_buf = torch.zeros(1, include_history_steps, num_partial_obs, device=device)

    # dt = 0.02
    dt = env_cfg['sim']['dt'] * env_cfg['control']['decimation']
    print('dt', dt)

    print(env_cfg['commands'])
    num_commands = env_cfg['commands'].get('num_commands', 7)
    # num_commands = max(num_commands, 5)
    print('num_commands', num_commands)
    num_cmds = len(states_cmd)
    enable_gait_modes = (num_cmds == num_commands + 1) if enable_gait_modes is None else enable_gait_modes
    print('enable_gait_modes', enable_gait_modes)

    if observe_gait_commands:
        commands = np.zeros(num_commands, dtype=np_dtype)
        if enable_gait_modes:
            keys = [
                None,
                None,
                None,
                'gait_frequency' if observe_frequency else None,
                'phase' if observe_phase else None,
                None,
                'foot_swing_height' if observe_foot_height else None,
            ]
            for i, k in enumerate(keys):
                if k is None:
                    continue
                rng = env_cfg['commands']['ranges'].get(k)
                if rng is None:
                    continue
                default_commands[i] = 0.5 * (rng[0] + rng[1])
            # default_commands = [0., 0., 0., 1., 0.5, 0.5, 0.2, 0., 0., 0., 0., ]
        commands[:num_commands] = default_commands[:num_commands]
        gait_indices = np.zeros(1)
        print('commands', commands)
        if num_cmds == num_commands + (1 if enable_gait_modes else 0):
            print('using states_cmd')
            commands = states_cmd

    num_extra_obs = env_cfg.get('num_extra_obs', 0)
    if num_extra_obs > 0:
        asset_extra_obs = env_cfg['asset_extra_obs']
        print('asset_extra_obs', asset_extra_obs.keys(), NAME)
        extra_obs = asset_extra_obs[NAME]
        extra_obs = np.frombuffer(extra_obs, dtype=np.np_dtype)
        extra_obs = torch.from_numpy(extra_obs).view(1, -1)
        print('extra_obs', num_extra_obs, extra_obs.shape)
        print(
            torch.min(extra_obs),
            torch.max(extra_obs),
        )
        assert len(extra_obs[0]) == num_extra_obs
        assert flatten_hist

    if enable_gait_modes:
        clock_scale, last_gait, jump_switching = 0, 0, 0

    s2w_step = 0.05
    # gait_ind_init = 0.5
    gait_ind_init = 0.

    num_obs = ctx.get('num_obs')
    num_dofs = len(ctx.get('dof_names'))
    # obs_pt = 3 + (export_size if exp_front else 0)
    # last_acts_pt = obs_pt + num_obs_prop - 3 - num_actions
    last_acts_pt = 6 + num_dofs * 2
    obs_buf = torch.zeros(num_obs, dtype=dtype, device=device)

    def torch_step(obs):

        obs[:, last_acts_pt:last_acts_pt + num_actions] = last_actions_tensor
        # obs_buf[obs_pt:last_acts_pt] = obs
        # hist = obs
        # hist_buf[1:] = hist_buf[:-1].clone()
        # hist_buf[0] = hist
        # obs_buf[hist_pt:hist_pt + hist_export_size] = hist_buf.flatten()

        if stack_history_obs:
            # print(obs.shape, hist.shape)
            hist_buf[:, :-1, :] = hist_buf[:, 1:, :].clone()
            hist_buf[:, -1, :] = obs
            obs = hist_buf.view(1, -1) if flatten_hist else hist_buf
            # obs = hist
        if num_extra_obs:
            obs = torch.cat([obs, extra_obs], dim=-1)
        if flatten_obs:
            obs = obs.flatten()
        # print('obs', obs.shape)
        actions = policy_fn(obs)
        acts = torch.clamp(actions, -clip_actions, clip_actions).view(1, -1)[0]
        # last_actions[:] = acts
        last_acts = acts
        if mask_obs_action:
            last_acts = acts * dof_mask_inv
        # obs_buf[last_acts_pt:last_acts_pt + num_actions] = last_actions
        last_actions_tensor[:] = last_acts
        # acts = acts * action_scale + default_dof_pos_tensor
        if masked_action_inds is not None:
            acts[masked_action_inds] = 0.
        acts = acts * action_scale + default_dof_pos_act_tensor

        # actions = policy_fn(obs_buf.view(1, -1))
        # actions = actions.flatten()
        # actions = self.policy(obs_buf)
        # t_tot += time.perf_counter() - t0
        # n_s += 1
        # print('acts', t_tot / n_s)
        # print('actions', actions.cpu().tolist())
        # t0 = time.perf_counter()
        # acts = torch.clip(actions, -clip_actions, clip_actions)
        # last_actions[:] = acts
        # obs_buf[:, last_acts_pt:last_acts_pt+num_actions] = acts
        # obs_buf[last_acts_pt:last_acts_pt + num_actions] = acts
        # acts = acts * action_scale + default_dof_pos_tensor
        # acts = acts[dof_map_tensor]
        # return acts, actions
        return acts

    def step_fn():
        # t0 = time.time()
        # quat = np.roll(states_quat, -1)
        # dof_pos = states_q + dof_q_offset
        dof_pos = states_q[dof_map_obs]
        dof_vel = states_qd[dof_map_obs]
        # dof_vel = np.clip(states_qd, -QD_LIMIT, QD_LIMIT)[dof_map_obs]
        base_ang_vel = states_ang_vel
        if use_rpy:
            rot_info = [states_rpy[:2], [0]]
        else:
            if gravity_from_rpy:
                mat = rpy2mat_np(states_rpy)
                projected_gravity = mat.T @ gravity_vec
            else:
                projected_gravity = quat_rotate_inverse_np(states_quat, gravity_vec)
            rot_info = [projected_gravity]
        # actions = np.array(last_actions)
        # actions = last_actions.numpy()
        actions = last_actions
        if mask_obs_action:
            actions = actions * dof_mask_inv
        obs_list = [
            base_ang_vel * scales_ang_vel,
            # projected_gravity,
            *rot_info,
            (dof_pos - default_dof_pos) * scales_dof_pos,
            dof_vel * scales_dof_vel,
            actions,
            states_cmd[:3] * cmds_scale[:3],
        ]
        if observe_gait_commands:
            # nonlocal gait_indices
            frequencies = cmd_frequency
            phases = cmd_phase
            # durations = commands[:, 5]
            gait_indices[:] = np.modf(gait_indices + dt * frequencies)[0]
            # print(gait_indices)
            if enable_gait_modes:
                gait_idx = states_cmd[-1]
                nonlocal clock_scale, last_gait, jump_switching
                if gait_idx == 0:  # Standing
                    clock_scale = max(0, clock_scale - s2w_step)
                else:  # Non-Standing
                    clock_scale = min(1, clock_scale + s2w_step)
                # if gait_idx == 1:
                if last_gait == 0 and gait_idx == 1:
                    print('gait_indices reset')
                    gait_indices[:] = gait_ind_init
                # phases = cmd_phase
                if gait_idx == 2:
                    phases = 0
                left_phase, right_phase = [gait_indices + phases, gait_indices + 0]
                if gait_idx != 3 and gait_idx != 4:
                    jump_switching = 0
                elif gait_idx == 3:  # Jumping, Left in the air
                    if last_gait == 4:
                        jump_switching = 1
                    jump_switching = max(0, jump_switching - 0.1)
                    left_phase[:] = 0.75 * (1 - jump_switching) + 0.5 * jump_switching
                    right_phase = 0.75 * jump_switching + right_phase * (1 - jump_switching)
                elif gait_idx == 4:  # Jumping, Right in the air
                    if last_gait == 3:
                        jump_switching = 1
                    jump_switching = max(0, jump_switching - 0.1)
                    left_phase = 0.75 * jump_switching + left_phase * (1 - jump_switching)
                    right_phase[:] = 0.75 * (1 - jump_switching) + 0.5 * jump_switching
                last_gait = gait_idx
                foot_phases = [left_phase, right_phase]
                # foot_phases = [clock_scale * left_phase, clock_scale * right_phase]
                # foot_phases = [clock_scale * left_phase + 0.25 * (1-clock_scale), clock_scale * right_phase + 0.25 * (1-clock_scale)]
            else:
                foot_phases = [gait_indices + phases, gait_indices]
            if enable_gait_modes and gait_idx == 0 and gait_alt0:
                # clock_inputs = [np.sin([x]) for x in [1, 1]]
                clock_inputs = [[1., 1.]]
            else:
                clock_inputs = [np.sin(2 * np.pi * x) for x in foot_phases]

            if adaptive_gait_frequency:
                v_level = 1.0 * np.linalg.norm(states_cmd[2]) + 0.5 * np.abs(states_cmd[2])
                velocity2frequency = (1.8 - 1.0) / (2.5 - 0.6)
                freq = (1.0 + velocity2frequency * (v_level - 0.6)).clip(min=1.0, max=1.8)
                # print(freq)
                commands[3] = freq
            if not observe_clock_only:
                obs_list.append(commands[3:num_commands] * cmds_scale[3:num_commands])
            # print('clock_inputs', foot_phases, clock_inputs)
            obs_list.extend(clock_inputs)
        obs = np.concatenate(obs_list)
        # print('commands', commands)
        # print('gait_indices', gait_indices)
        # for i, x in enumerate(obs_list):
        #     print(i, [len(x)], np.round(np.array(x).astype(np.float64), decimals=2).tolist())
        # print(obs.shape, [len(o) for o in obs_list])
        if states_infer_obs is not None:
            states_infer_obs[:] = obs
        obs = to_tensor(obs, dtype=dtype, device=device).view(1, -1)
        # print(obs[:, -13:])
        # print('obs', time.time() - t0)
        if stack_history_obs:
            # print(obs.shape, hist.shape)
            hist_buf[:, :-1, :] = hist_buf[:, 1:, :].clone()
            hist_buf[:, -1, :] = obs
            obs = hist_buf.view(1, -1) if flatten_hist else hist_buf
            # obs = hist
        if num_extra_obs:
            obs = torch.cat([obs, extra_obs], dim=-1)
        if flatten_obs:
            obs = obs.flatten()
        # print('obs', obs.shape)
        actions = policy_fn(obs)
        acts = torch.clamp(actions, -clip_actions, clip_actions).view(1, -1)[0]
        acts_np = acts.cpu().numpy()
        if states_infer_acts is not None:
            states_infer_acts[:] = acts_np
        last_actions[:] = acts_np
        # acts = acts * action_scale + default_dof_pos_tensor
        if masked_action_inds is not None:
            acts_np[masked_action_inds] = 0.
        acts = acts_np * action_scale + default_dof_pos_act
        # extras = (obs, actions)
        # acts = acts.cpu().numpy()
        # states_q_ctrl[dof_map_action_dest] = acts[dof_map_action_src]
        states_q_ctrl[dof_map_action_dest] = acts
        # print('acts', time.time() - t0)

    def reset_fn():
        nonlocal clock_scale, last_gait, jump_switching
        if stack_history_obs:
            hist_buf[:] = 0.
        last_actions[:] = 0.
        gait_indices[:] = 0.
        if enable_gait_modes:
            clock_scale, last_gait, jump_switching = 0, 0, 0
        if policy_reset_fn is not None:
            policy_reset_fn()

    return step_fn, reset_fn
