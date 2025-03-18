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
    default_dof_pos=None,
    up_axis_idx=2,
    device='cpu',
    dtype=None,
    use_rpy=True,
    flatten_hist=True,
):
    import torch
    import numpy as np
    from unicon.utils.torch import to_tensor
    from unicon.utils import get_axis_params, quat_rotate_inverse_np
    dtype = torch.float if dtype is None else dtype
    np_dtype = np.float32
    use_rpy = env_cfg.get('use_rpy', use_rpy)
    flatten_hist = env_cfg['env'].get('flatten_hist', flatten_hist)
    print('use_rpy', use_rpy)
    print('flatten_hist', flatten_hist)
    num_actions = env_cfg['env']['num_actions']
    # num_actions = len(default_dof_pos)
    dof_map_action = dof_map
    clip_actions = env_cfg['normalization']['clip_actions']
    action_scale = env_cfg['control']['action_scale']
    obs_scales = env_cfg['normalization']['obs_scales']
    scales_lin_vel = obs_scales['lin_vel']
    scales_ang_vel = obs_scales['ang_vel']
    scales_dof_pos = obs_scales['dof_pos']
    scales_dof_vel = obs_scales['dof_vel']
    gait_freq_cmd = obs_scales['gait_freq_cmd']
    gait_phase_cmd = obs_scales['gait_phase_cmd']
    footswing_height_cmd = obs_scales['footswing_height_cmd']
    observe_gait_commands = env_cfg['env']['observe_gait_commands']
    observe_clock_only = env_cfg.get('observe_clock_only', False)
    include_history_steps = env_cfg['env']['include_history_steps']
    stack_history_obs = env_cfg['env']['stack_history_obs']
    use_clock = env_cfg['env'].get('use_clock', True)
    cmds_scale = [
        scales_lin_vel,
        scales_lin_vel,
        scales_ang_vel,
        gait_freq_cmd,
        gait_phase_cmd,
        gait_phase_cmd,
        footswing_height_cmd,
    ]
    if env_cfg['env'].get('observe_body_height'):
        cmds_scale.append(obs_scales['body_height_cmd'])  # Body Height Cmd (1 dim)
    if env_cfg['env'].get('observe_body_pitch'):
        cmds_scale.append(obs_scales['body_pitch_cmd'])  # Body Pitch Cmd (1 dim)
    if env_cfg['env'].get('observe_waist_roll'):
        cmds_scale.append(obs_scales['waist_roll_cmd'])  # Waist Roll (1 dim)
    if env_cfg['env'].get('interrupt_in_cmd'):
        cmds_scale.append(1)  # Interrupt Flag (1 dim)
    cmds_scale = np.array(cmds_scale, dtype=np_dtype)
    gravity_vec = np.array(get_axis_params(-1., up_axis_idx), dtype=np_dtype)
    default_dof_pos_np = np.array(default_dof_pos, dtype=np_dtype)
    default_dof_pos = np.array(default_dof_pos, dtype=np_dtype)
    default_dof_pos_tensor = to_tensor(default_dof_pos)
    retained_action_inds = env_cfg.get('retained_action_inds')
    dof_map_obs = dof_map
    if retained_action_inds is not None:
        _dof_map = list(range(dof_map.start, dof_map.stop)) if isinstance(dof_map, slice) else dof_map
        print('retained_action_inds', retained_action_inds)
        dof_map_action = [_dof_map[i] for i in retained_action_inds]
        print('dof_map_action', dof_map_action)
        # default_dof_pos_tensor = default_dof_pos_tensor[dof_map_action]
        default_dof_pos_tensor = default_dof_pos_tensor[retained_action_inds]
        default_dof_pos_np = default_dof_pos[retained_action_inds]
        # print('default_dof_pos_tensor', default_dof_pos_tensor)
        if env_cfg.get('retained_obs', False):
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
        num_partial_obs = env_cfg['env']['num_partial_obs']
        num_partial_obs = int(num_partial_obs // include_history_steps) if num_partial_obs > 100 else num_partial_obs
        print('include_history_steps', include_history_steps)
        print('num_partial_obs', num_partial_obs)
        hist = torch.zeros(1, include_history_steps, num_partial_obs, device=device)

    # dt = 0.02
    dt = env_cfg['sim']['dt'] * env_cfg['control']['decimation']
    print('dt', dt)

    enable_gait_idx = len(states_cmd) > 3

    if observe_gait_commands:
        print(env_cfg['commands'])
        num_commands = env_cfg['commands'].get('num_commands', 7)
        num_commands = max(num_commands, 7)
        print('num_commands', num_commands)
        commands = np.zeros(num_commands, dtype=np_dtype)
        default_commands = [
            0.,
            0.,
            0.,
            1.2,
            0.5,
            0.5,
            0.12,
            0.,
            0.,
            0.,
            0.,
            0,
        ]
        if enable_gait_idx:
            keys = [
                None,
                None,
                None,
                'gait_frequency',
                'phase',
                None,
                'foot_swing_height',
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
    elif use_clock:
        gait_time = 0
    # import time

    if enable_gait_idx:
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
            quat = states_quat
            projected_gravity = quat_rotate_inverse_np(quat, gravity_vec)
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
            frequencies = commands[3]
            phases = commands[4]
            # durations = commands[:, 5]
            gait_indices = np.modf(gait_indices + dt * frequencies)[0]
            # print(gait_indices)
            if enable_gait_idx:
                gait_idx = states_cmd[3]
                nonlocal clock_scale, last_gait, jump_switching
                if gait_idx == 0:  # Standing
                    clock_scale = max(0, clock_scale - 0.1)
                else:  # Non-Standing
                    clock_scale = min(1, clock_scale + 0.1)
                if gait_idx == 1:
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
            clock_inputs = [np.sin(2 * np.pi * x) for x in foot_phases]
            if not observe_clock_only:
                obs_list.append(commands[3:] * cmds_scale[3:])
            obs_list.extend(clock_inputs)
        elif use_clock:
            nonlocal gait_time
            gait_time += 1
            freq = 1.2
            phases = gait_time * (dt * freq)
            clock_sin = np.sin(phases)
            clock_cos = np.cos(phases)
            obs_list.extend([
                clock_sin,
                clock_cos,
            ])
        # print([len(o) for o in obs_list])
        obs = np.concatenate(obs_list)
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
        actions = policy_fn(obs)
        acts = torch.clamp(actions, -clip_actions, clip_actions)[0]
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
        policy_reset_fn()

    return step_fn, reset_fn
