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
    use_rpy=None,
    flatten_hist=True,
    # flatten_hist=False,
    retained_obs=True,
    observe_gait_commands=None,
    enable_gait_modes=None,
    adaptive_gait_frequency=None,
    stack_history_obs=None,
    flatten_obs=False,
    # gait_alt0=False,
    gait_alt0=True,
    dof_names=None,
    robot_def=None,
    gravity_from_rpy=True,
):
    import torch
    import numpy as np
    from unicon.utils.torch import to_tensor
    from unicon.utils import get_axis_params, quat_rotate_inverse_np, rpy2mat_np
    dtype = torch.float if dtype is None else dtype
    np_dtype = np.float32
    use_rpy = env_cfg.get('use_rpy', True) if use_rpy is None else use_rpy
    flatten_hist = env_cfg['env'].get('flatten_hist', flatten_hist)
    print('use_rpy', use_rpy)
    print('gravity_from_rpy', gravity_from_rpy)
    print('flatten_hist', flatten_hist)
    num_actions = env_cfg['env']['num_actions']
    if any(['mix' in x for x in env_cfg.get('argv', [])]):
        NAME = robot_def['NAME']
        NAME = 's42' if NAME == 's45' else NAME
        params = env_cfg['robot_params'].get(NAME, {})
        init_joint_angles = params.get('default_joint_angles', {})
        print('mix env', NAME, init_joint_angles)
    else:
        init_joint_angles = env_cfg['init_state']['default_joint_angles']
    default_dof_pos = np.array([init_joint_angles.get(n, 0.) for n in dof_names])
    dof_map_action = dof_map
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
        observe_gait_commands = env_cfg['env'].get('observe_gait_commands')
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
    use_clock = env_cfg['env'].get('use_clock', True)
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
    observe_frequency = env_cfg_env.get('observe_frequency', True)
    observe_phase = env_cfg_env.get('observe_phase', True)
    observe_duration = env_cfg_env.get('observe_duration', True)
    observe_foot_height = env_cfg_env.get('observe_foot_height', True)
    observe_body_height = env_cfg_env.get('observe_body_height',)
    observe_body_pitch = env_cfg_env.get('observe_body_pitch',)
    observe_waist_roll = env_cfg_env.get('observe_waist_roll',)
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
    if observe_body_pitch:
        cmds_scale.append(obs_scales['body_pitch_cmd'])  # Body Pitch Cmd (1 dim)
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
    retained_action_inds = retained_action_inds if retained_actions else None
    if retained_action_inds is not None:
        _dof_map = list(range(dof_map.start, dof_map.stop)) if isinstance(dof_map, slice) else dof_map
        print('retained_action_inds', len(retained_action_inds), retained_action_inds)
        dof_map_action = [_dof_map[i] for i in retained_action_inds]
        print('dof_map_action', dof_map_action)
        # default_dof_pos_tensor = default_dof_pos_tensor[dof_map_action]
        default_dof_pos_tensor = default_dof_pos_tensor[retained_action_inds]
        default_dof_pos_np = default_dof_pos[retained_action_inds]
        # print('default_dof_pos_tensor', default_dof_pos_tensor)
        if env_cfg.get('retained_obs', retained_obs):
            dof_map_obs = dof_map_action
            print('dof_map_obs', dof_map_obs)
            # default_dof_pos = default_dof_pos[dof_map_obs]
            default_dof_pos = default_dof_pos[retained_action_inds]
    print('clip_actions', clip_actions)
    print('obs_scales', obs_scales)
    print('action_scale', action_scale)
    print('default_dof_pos', default_dof_pos.tolist())
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
    elif use_clock:
        gait_time = 0
    # import time

    if enable_gait_modes:
        clock_scale, last_gait, jump_switching = 0, 0, 0

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
            nonlocal gait_indices
            frequencies = cmd_frequency
            phases = cmd_phase
            # durations = commands[:, 5]
            gait_indices = np.modf(gait_indices + dt * frequencies)[0]
            # print(gait_indices)
            if enable_gait_modes:
                gait_idx = states_cmd[-1]
                nonlocal clock_scale, last_gait, jump_switching
                if gait_idx == 0:  # Standing
                    clock_scale = max(0, clock_scale - 0.1)
                else:  # Non-Standing
                    clock_scale = min(1, clock_scale + 0.1)
                # if gait_idx == 1:
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
                foot_phases = [clock_scale * left_phase, clock_scale * right_phase]
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
        elif use_clock:
            nonlocal gait_time
            gait_time += 1
            freq = 1.2
            phases = gait_time * (dt * freq)
            clock_sin = np.sin(phases).reshape(1)
            clock_cos = np.cos(phases).reshape(1)
            obs_list.extend([
                clock_sin,
                clock_cos,
            ])
        obs = np.concatenate(obs_list)
        # for i, x in enumerate(obs_list):
        # print(i, np.round(x, decimals=2))
        # print(obs.shape, [len(o) for o in obs_list])
        if states_infer_obs is not None:
            states_infer_obs[:] = obs
        obs = to_tensor(obs, dtype=dtype, device=device).view(1, -1)
        # print(obs[:, -13:])
        # print('obs', time.time() - t0)
        if stack_history_obs:
            hist[:, :-1, :] = hist[:, 1:, :].clone()
            hist[:, -1, :] = obs
            obs = hist.view(1, -1) if flatten_hist else hist
            # obs = hist
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
        acts = acts_np * action_scale + default_dof_pos_np
        # extras = (obs, actions)
        # acts = acts.cpu().numpy()
        states_q_ctrl[dof_map_action] = acts
        # print('acts', time.time() - t0)

    def reset_fn():
        if stack_history_obs:
            hist[:] = 0.
        last_actions[:] = 0.
        gait_indices[:] = 0.
        if policy_reset_fn is not None:
            policy_reset_fn()

    return step_fn, reset_fn
