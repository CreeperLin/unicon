def cb_infer_gr1(
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
    env_cfg=None,
    dof_map=None,
    # default_dof_pos=None,
    up_axis_idx=2,
    device='cpu',
    dtype=None,
    # use_rpy=None,
    use_rpy=False,
    flatten_hist=True,
    # flatten_hist=False,
    retained_obs=False,
    observe_gait_commands=None,
    enable_gait_modes=None,
    adaptive_gait_frequency=None,
    # stack_history_obs=None,
    # flatten_obs=False,
    # gait_alt0=False,
    stack_history_obs=False,
    flatten_obs=True,
    gait_alt0=True,
    dof_names=None,
    gravity_from_rpy=True,
    states_rpy2=None,
    states_ang_vel2=None,
    states_left_target=None,
    states_right_target=None,
    states_reach_mask=None,
    double_policy=False,
    # planner_type='',
):
    from unicon.utils import get_ctx, import_obj
    robot_def = get_ctx()['robot_def']
    import torch
    import numpy as np
    from unicon.utils.torch import to_tensor
    from unicon.utils import get_axis_params, quat_rotate_inverse_np, rpy2mat_np, list2slice
    # from unicon.sensors.planners import planner_3dim

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
    if any(['mix' in x for x in env_cfg.get('argv', [])]):
        NAME = robot_def['NAME']
        NAME = 's42' if NAME == 's45' else NAME
        params = env_cfg['robot_params'].get(NAME, None)
        if params is None:
            init_joint_angles = robot_def.get('Q_RESET', {})
        else:
            init_joint_angles = params.get('default_joint_angles', {})
        print('mix env', NAME, init_joint_angles)
    else:
        init_joint_angles = env_cfg['init_state']['default_joint_angles']
    if isinstance(init_joint_angles, dict):
        default_dof_pos = np.array([init_joint_angles.get(n, 0.) for n in dof_names])
    else:
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
    default_dof_pos_np = np.array(default_dof_pos, dtype=np_dtype)
    default_dof_pos = np.array(default_dof_pos, dtype=np_dtype)
    default_dof_pos_tensor = to_tensor(default_dof_pos)
    retained_action_inds = env_cfg.get('retained_action_inds', env_cfg['env'].get('retained_actions_indxs'))
    dof_map_obs = dof_map
    retained_actions = env_cfg['env'].get('retained_actions', True)
    retained_obs = env_cfg.get('retained_obs', retained_obs)
    retained_action_inds = retained_action_inds if retained_actions else None
    print('retained_actions', retained_actions)
    print('retained_obs', retained_obs)
    masked_action_inds = None
    if retained_action_inds is not None:
        _dof_map = list(range(dof_map.start, dof_map.stop)) if isinstance(dof_map, slice) else dof_map
        print('retained_action_inds', len(retained_action_inds), retained_action_inds)
        # default_dof_pos_tensor = default_dof_pos_tensor[dof_map_action_dest]
        if len(retained_action_inds) == num_actions:
            dof_map_action_dest = [_dof_map[i] for i in retained_action_inds]
            default_dof_pos_tensor = default_dof_pos_tensor[retained_action_inds]
            default_dof_pos_np = default_dof_pos[retained_action_inds]
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
    print('masked_action_inds', masked_action_inds)
    print('dof_map_obs', dof_map_obs)
    print('dof_map_action_dest', dof_map_action_dest)
    print('clip_actions', clip_actions)
    print('obs_scales', obs_scales)
    print('action_scale', action_scale)
    print('default_dof_pos', np.round(default_dof_pos.astype(np.float64), decimals=3).tolist())
    # last_actions = torch.zeros(num_actions, dtype=dtype, device=device)
    last_actions = np.zeros(num_actions, dtype=np_dtype)
    hist = None
    if stack_history_obs:
        num_partial_obs_full = False
        env = env_cfg['env']
        num_partial_obs = env.get('num_partial_obs', env.get('num_partial_observations'))
        num_partial_obs = int(num_partial_obs // include_history_steps) if num_partial_obs_full else num_partial_obs
        print('include_history_steps', include_history_steps)
        print('num_partial_obs', num_partial_obs)
        hist = torch.zeros(1, include_history_steps, num_partial_obs, device=device)

    # dt = 0.02
    dt = env_cfg['sim']['dt'] * env_cfg['control']['decimation']
    print('dt', dt)

    print(env_cfg['commands'])
    num_commands = env_cfg['commands'].get('num_commands', 7)
    num_commands = max(num_commands, 5)
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
        extra_obs = np.frombuffer(extra_obs, dtype=np.float32)
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
    last_gait = 0

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

        rot_info2 = None
        if states_rpy2 is not None:
            if use_rpy:
                rot_info2 = [states_rpy2[:2], [0]]
            else:
                if gravity_from_rpy:
                    mat = rpy2mat_np(states_rpy2)
                    projected_gravity2 = mat.T @ gravity_vec
                else:
                    projected_gravity2 = quat_rotate_inverse_np(states_quat, gravity_vec)
                rot_info2 = [projected_gravity2]

        base_ang_vel2 = None
        if states_ang_vel2 is not None:
            base_ang_vel2 = states_ang_vel2

        # if len(planner_type) > 0:
        #     # planner_fn = import_obj(f"planner_{planner_type}", default_name_prefix='planner', default_mod_prefix='unicon.sensors.planners')
        #     planner_fn = planner_3dim
        #     planner_info = planner_fn(
        #         states_left_target, 
        #         states_right_target,
        #         states_cmd[:3],
        #         states_reach_mask
        #     )

        #     states_cmd[:3] = planner_info['commands']
        #     states_cmd[3:num_commands] = [1.2, 0.5, 0.5, 0, 0, 0]
        #     states_reach_mask[:] = planner_info['reach_mask']

        # actions = np.array(last_actions)
        # actions = last_actions.numpy()
        actions = last_actions
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
            # gait_indices[:] = np.modf(gait_indices + dt * frequencies)[0]
            gait_indices[:] = (gait_indices + dt * frequencies) % 1.0
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
                if gait_idx == 3:  # Jumping, Left in the air
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
                gait_idx = 1 - int(-0.05 < states_cmd[0] < 0.05 and -0.05 < states_cmd[1] < 0.05 and -0.1 < states_cmd[2] < 0.1) # 0 for standing and 1 for walking
                print("GAIT idx", gait_idx, states_cmd[:3], gait_indices)

                # left_phase, right_phase = [gait_indices + phases, gait_indices]
                left_phase = gait_indices + phases
                right_phase = gait_indices + 0.
                # if gait_idx == 1:
                #     mask_l = (left_phase > 0.15 and left_phase < 0.35)
                #     left_phase[mask_l] = 0.25
                #     right_phase[mask_l] = 0.25
                #     mask_r = (right_phase > 0.15 and right_phase < 0.35)
                #     right_phase[mask_r] = 0.25
                #     left_phase[mask_r] = 0.25
                if (gait_idx == 1) and (last_gait == 0): # standing2walking
                    left_phase[:] = 0.0
                    right_phase[:] = 0.0
                
                last_gait = gait_idx
                foot_phases = [left_phase, right_phase]
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

        obs_list.append(base_ang_vel2 * scales_ang_vel)
        obs_list.extend(rot_info2)

        # print(commands, clock_inputs, enable_gait_modes)
        # print(base_ang_vel, rot_info, base_ang_vel2, rot_info2)

        obs_list.append(states_left_target)
        obs_list.append(states_right_target)
        # print('states_left_target', states_left_target)
        # print('states_right_target', states_right_target)

        obs_list.append(states_reach_mask)

        obs = np.concatenate(obs_list)
        # print("One step observation", obs.shape)
        # for i in range(0, len(obs), 10):
        #     line = ','.join(f'{x:.4f}' for x in obs[i:i+10])
        #     print(line)

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
            hist[:, :-1, :] = hist[:, 1:, :].clone()
            hist[:, -1, :] = obs
            obs = hist.view(1, -1) if flatten_hist else hist
            # obs = hist
        if num_extra_obs:
            obs = torch.cat([obs, extra_obs], dim=-1)

        # print(obs.shape)
        # print(extra_obs.shape if num_extra_obs else None)

        if flatten_obs:
            obs = obs.flatten()
        # print('obs', obs.shape)
        if double_policy:
            choice = 1 if np.linalg.norm(states_cmd[:2]) < 0.01 and np.abs(states_cmd[2]) < 0.02 else 0
            print("Choice", choice, states_cmd[:])
            print("Commands", obs[6+29*3:17+29*3])
            actions = policy_fn(obs, choice)
        else:
            actions = policy_fn(obs)
        acts = torch.clamp(actions, -clip_actions, clip_actions).view(1, -1)[0]
        acts_np = acts.cpu().numpy()
        if states_infer_acts is not None:
            states_infer_acts[:] = acts_np
        last_actions[:] = acts_np
        # acts = acts * action_scale + default_dof_pos_tensor
        if masked_action_inds is not None:
            acts_np[masked_action_inds] = 0.
        acts = acts_np * action_scale + default_dof_pos_np
        # extras = (obs, actions)
        # acts = acts.cpu().numpy()
        # states_q_ctrl[dof_map_action_dest] = acts[dof_map_action_src]
        states_q_ctrl[dof_map_action_dest] = acts
        # print('acts', time.time() - t0)

    def reset_fn():
        nonlocal clock_scale, last_gait, jump_switching
        if stack_history_obs:
            hist[:] = 0.
        last_actions[:] = 0.
        gait_indices[:] = 0.
        if enable_gait_modes:
            clock_scale, last_gait, jump_switching = 0, 0, 0
        if policy_reset_fn is not None:
            policy_reset_fn()

    return step_fn, reset_fn
