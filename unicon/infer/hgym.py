def cb_infer_hgym(
    states_rpy,
    states_ang_vel,
    states_quat,
    states_q,
    states_qd,
    states_cmd,
    states_q_ctrl,
    policy_fn,
    policy_reset_fn,
    env_cfg=None,
    dof_map=None,
    default_dof_pos=None,
    device='cpu',
    dtype=None,
):
    import torch
    import numpy as np
    from unicon.utils.torch import to_tensor
    dtype = torch.float if dtype is None else dtype
    np_dtype = np.float32
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
    scales_quat = obs_scales['quat']
    frame_stack = env_cfg['env']['frame_stack']
    num_single_obs = env_cfg['env']['num_single_obs']
    cmds_scale = np.array([
        scales_lin_vel,
        scales_lin_vel,
        scales_ang_vel,
    ], dtype=np_dtype)
    default_dof_pos = np.array(default_dof_pos, dtype=np_dtype)
    default_dof_pos_tensor = to_tensor(default_dof_pos)
    dof_map_obs = dof_map
    print('clip_actions', clip_actions)
    print('obs_scales', obs_scales)
    print('action_scale', action_scale)
    print('default_dof_pos', default_dof_pos.tolist())
    last_actions = torch.zeros(num_actions, dtype=dtype, device=device)
    hist = None
    if frame_stack:
        print('frame_stack', frame_stack, num_single_obs)
        hist = torch.zeros(1, frame_stack, num_single_obs, device=device)

    cycle_time = env_cfg['rewards']['cycle_time']
    gait_step = 0
    dt = env_cfg['sim']['dt'] * env_cfg['control']['decimation']
    print('dt', dt)
    pi = np.pi

    def step_fn():
        nonlocal gait_step
        gait_step += 1
        phase = gait_step * dt / cycle_time
        sin_pos = np.sin(2 * pi * phase)
        cos_pos = np.cos(2 * pi * phase)
        dof_pos = states_q[dof_map_obs]
        dof_vel = states_qd[dof_map_obs]
        base_ang_vel = states_ang_vel
        base_euler_xyz = states_rpy
        actions = last_actions.numpy()
        obs_list = [
            [sin_pos, cos_pos],
            states_cmd[:3] * cmds_scale[:3],
            (dof_pos - default_dof_pos) * scales_dof_pos,
            dof_vel * scales_dof_vel,
            actions,
            base_ang_vel * scales_ang_vel,
            base_euler_xyz * scales_quat,
        ]
        obs = np.concatenate(obs_list)
        # print(obs.tolist())
        obs = to_tensor(obs, dtype=dtype, device=device).view(1, -1)
        # print('obs', time.time() - t0)
        if frame_stack:
            hist[:, :-1, :] = hist[:, 1:, :].clone()
            hist[:, -1, :] = obs
            obs = hist.view(1, -1)
        actions = policy_fn(obs)
        acts = torch.clamp(actions, -clip_actions, clip_actions).to(device)[0]
        last_actions[:] = acts
        acts = acts * action_scale + default_dof_pos_tensor
        states_q_ctrl[dof_map_action] = acts
        # print('acts', time.time() - t0)

    def reset_fn():
        nonlocal gait_step
        gait_step = 0
        last_actions[:] = 0.
        policy_reset_fn()

    return step_fn, reset_fn
