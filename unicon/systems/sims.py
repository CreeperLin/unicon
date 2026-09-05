def cb_sims_recv_send_close(
    states_q_ctrl,
    states_rpy,
    states_ang_vel,
    states_quat,
    states_q,
    states_qd,
    states_q_tau=None,
    states_q_cur=None,
    states_lin_vel=None,
    states_lin_acc=None,
    states_pos=None,
    dof_names=None,
    states_qd_ctrl=None,
    states_tau_ctrl=None,
    states_x_ft=None,
    copy=False,
    sims_kwds=None,
):
    import numpy as np
    from sims.run import run
    from sims.utils import list2slice
    from unicon.utils import get_ctx, pats2inds

    if states_x_ft is not None:
        ctx = get_ctx()
        robot_def = ctx['robot_def']
        LINK_NAMES = robot_def['LINK_NAMES']
        system_config = sims_kwds['system_config']
        link_ft_names = system_config.get('link_ft_names', [])
        link_ft_keys = list(link_ft_names.keys()) if isinstance(link_ft_names, dict) else link_ft_names
        ft_inds, ft_names, _ = pats2inds(link_ft_keys, LINK_NAMES)
        print('ft_names', ft_names)
        print('ft_inds', ft_inds)
        if link_ft_names:
            system_config['link_ft_names'] = {k: link_ft_names[k] for k in ft_names} if isinstance(link_ft_names, dict) else ft_names

    s = run(**({} if sims_kwds is None else sims_kwds), run_time=False)
    s.cb_init()
    # num_dofs = len(states_q_ctrl) - 1
    num_dofs = len(states_q_ctrl)
    send_msg = {
        'name': dof_names,
        'position': states_q_ctrl,
        # 'position': states_q_ctrl[:num_dofs],
        'velocity': np.zeros(num_dofs) if states_qd_ctrl is None else states_qd_ctrl,
        'effort': np.zeros(num_dofs) if states_tau_ctrl is None else states_tau_ctrl,
    }
    s.cb_recv(send_msg)
    recv_msg = s.cb_send()
    name = recv_msg['name']
    dof_inds = [dof_names.index(n) for n in name if n in dof_names]
    dof_src_inds = [i for i, n in enumerate(name) if n in dof_names]
    dof_inds = list2slice(dof_inds)
    dof_src_inds = list2slice(dof_src_inds)
    print('msg names', len(name), name)
    print('dof names', len(dof_names), dof_names)
    print('dof_inds', dof_inds)
    print('dof_src_inds', dof_src_inds)

    def cb_send():
        if copy:
            send_msg['position'] = states_q_ctrl.copy()
        s.cb_recv(send_msg)

    def cb_recv():
        recv_msg = s.cb_send()
        # print('recv_msg', recv_msg)
        states_q[dof_inds] = recv_msg['position'][dof_src_inds]
        states_qd[dof_inds] = recv_msg['velocity'][dof_src_inds]
        root_states = recv_msg['root_states']
        # states_ang_vel[:] = root_states[10:13]
        imu = recv_msg['imu']
        states_ang_vel[:] = imu[6:9]
        states_quat[:] = root_states[3:7]
        states_rpy[:] = imu[0:3]
        if states_lin_vel is not None:
            # states_lin_vel[:] = root_states[7:10]
            states_lin_vel[:] = imu[9:12]
        if states_pos is not None:
            states_pos[:] = root_states[0:3]
        if states_q_tau is not None:
            states_q_tau[dof_inds] = recv_msg['effort'][dof_src_inds]
        if states_lin_acc is not None:
            # print(imu[3:6])
            states_lin_acc[:] = imu[3:6]
        if states_x_ft is not None and ft_inds:
            states_x_ft[ft_inds] = recv_msg['link_ft']

    def cb_close():
        s.cb_recv(True)

    return cb_recv, cb_send, cb_close
