import numpy as np
try:
    from matplotlib import pyplot as plt
    import matplotlib.animation as animation
except ImportError:
    plt = None


def get_plot_args(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-rp', '--rec_paths', action='append')
    parser.add_argument('-rtp', '--rec_types', action='append')
    parser.add_argument('-d', '--dofs', default='range(12)')
    parser.add_argument('-s', '--subdir', action='store_true')
    parser.add_argument('-n', '--num_steps', type=int, default=None)
    parser.add_argument('-st', '--start', type=int, default=200)
    parser.add_argument('-o', '--offsets', type=int, action='append')
    parser.add_argument('-i', '--info', action='store_true')
    parser.add_argument('-rt', '--robot_type', default=None)
    parser.add_argument('-sp', '--single_plot', action='store_true')
    parser.add_argument('-al', '--auto_lim', action='store_true')
    parser.add_argument('-np', '--no_plot', action='store_true')
    parser.add_argument('-el', '--eval_loss', action='store_true')
    parser.add_argument('-fi', '--fixed_inds0', type=int, default=None)
    parser.add_argument('-simu', '--states_imu', action='store_true')
    parser.add_argument('-sie', '--states_i_extras', action='store_true')
    parser.add_argument('-sqe', '--states_q_extras', action='store_true')
    parser.add_argument('-sxe', '--states_x_extras', action='store_true')
    parser.add_argument('-sxe2', '--states_x_extras2', action='store_true')
    parser.add_argument('-awf', '--acc_world_frame', action='store_true')
    parser.add_argument('-nd', '--no_dof_states', action='store_true')
    parser.add_argument('-g', '--g', type=float, default=-9.85)
    parser.add_argument('-dt', '--dt', type=float, default=None)
    parser.add_argument('-ci', '--check_int', action='store_true')
    parser.add_argument('-nrs', '--no_root_states', action='store_true')
    parser.add_argument('-mo', '--motion_output', default=None)
    parser.add_argument('-l', '--loss_fns', default=None, action='append')
    parser.add_argument('-e', '--ext', default='png')
    parser.add_argument('-ae', '--aniext', default=None)
    parser.add_argument('-pr', '--plot_root', default='plots2')
    parser.add_argument('-tl', '--traj_len', type=int, default=2**8)
    args, _ = parser.parse_known_args(args)
    return args


def plot_states_x_extras2(
    rec_names,
    recs,
    st,
    ed,
    robot_def,
):
    t = list(range(st, ed))
    nplts = 4
    LINK_NAMES = robot_def['LINK_NAMES']
    num_links = len(LINK_NAMES)
    links = list(range(num_links))
    fig, axes = plt.subplots(num_links, nplts, figsize=(10 * nplts, 10 * num_links))
    axes = axes.reshape(-1, nplts)
    print('axes', axes.shape)
    from unicon.utils import mat2rpy_np, mat2pos_np
    for k, (link_idx, axs) in enumerate(zip(links, axes)):
        link_name = LINK_NAMES[link_idx]
        ax1, ax2, ax3 = axs[:3]
        ax4 = axs[3]
        for i, rec in enumerate(recs):
            x = rec['states_x'][st:ed, link_idx]
            xd = rec['states_xd'][st:ed, link_idx]
            xpos = mat2pos_np(x)
            xrpy = mat2rpy_np(x)
            xpos_ref = np.zeros((1, 3))

            for d in range(3):
                ax = axs[d]
                if i == 0:
                    ax.plot(t[0], xpos_ref[0, d])
                ax.plot(t, xpos[:, d])

        # ax1.set_ylim([-x_max, x_max])
        # ax3.set_ylim([-x_max, x_max])
        ax1.legend(['ref'] + rec_names, loc="lower right")
        ax1.set_title(link_name + ' pos')
    fig.tight_layout()
    return fig


def plot(args=None):
    if args is None:
        args = get_plot_args()
    import os
    motion_output = args.motion_output
    states_i_extras = args.states_i_extras
    states_q_extras = args.states_q_extras
    states_x_extras = args.states_x_extras
    states_x_extras2 = args.states_x_extras2
    eval_loss = args.eval_loss
    if eval_loss:
        import unicon.losses
        loss_names = [
            'chamfer_loss',
            'unfolded_mse_loss',
            'mse_loss',
            # 'rmse_loss',
            # 'kldiv_loss',
            # 'rot_loss',
            # 'soft_dtw_loss',
        ]
        if args.loss_fns is not None:
            loss_names = args.loss_fns
        loss_fns = {n: getattr(unicon.losses, n) for n in loss_names}
    no_plot = args.no_plot
    do_plot = not args.no_plot

    from unicon.utils import import_obj, parse_robot_def

    rec_paths = args.rec_paths or []
    rec_types = args.rec_types or []
    rec_types = rec_types + [None] * (len(rec_paths) - len(rec_types))
    rec_names = list(map(os.path.basename, rec_paths))

    def load_rec(rec_path, rec_type=None, load_kwds=None):
        rec_type = None if rec_type == '' else rec_type
        _, ext = os.path.splitext(rec_path)
        rec_type = ext[1:] if rec_type is None else rec_type
        print('rec_type', rec_type)
        mod = import_obj((rec_type, None), default_mod_prefix='unicon.data')
        print('rec_path', rec_path)
        return mod.load(
            rec_path,
            # robot_def=robot_def,
            **(load_kwds or {}),
        )

    recs = [load_rec(p, t) for p, t in zip(rec_paths, rec_types)]
    rec = recs[0]

    robot_type = args.robot_type
    robot_type = rec.get('args', {}).get('robot_type') if robot_type is None else robot_type
    print('robot_type', robot_type)
    robot_def = import_obj(robot_type, default_mod_prefix='unicon.defs', prefer_mod=True)
    robot_def = parse_robot_def(robot_def)
    Q_CTRL_INIT = robot_def.get('Q_CTRL_INIT', None)
    Q_CTRL_MIN = robot_def.get('Q_CTRL_MIN', None)
    Q_CTRL_MAX = robot_def.get('Q_CTRL_MAX', None)
    DOF_NAMES = robot_def.get('DOF_NAMES', None)
    NUM_DOFS = robot_def.get('NUM_DOFS', None)
    TAU_LIMIT = robot_def.get('TAU_LIMIT', None)
    QD_LIMIT = robot_def.get('QD_LIMIT', None)
    KP = robot_def.get('KP', None)
    KD = robot_def.get('KD', None)
    KP_2 = robot_def.get('KP_2', KP)
    KD_2 = robot_def.get('KD_2', KD)
    Q_CTRL_INIT = np.zeros(NUM_DOFS) if Q_CTRL_INIT is None else Q_CTRL_INIT
    # tau_max = max(TAU_LIMIT) * 1.2

    dt = rec.get('dt', args.dt)
    # assert all([r.get('dt', dt) == dt for r in recs])
    num_recs = len(recs)
    print('num_recs', num_recs)
    for n, r in zip(rec_names, recs):
        print(n)
        print('keys', r.keys())
        print('cmd', ' '.join(r.get('argv', [])))
        # print('env_cfg_args', r.get('env_cfg_args'))
        for k, v in r.items():
            if 'states' not in k and k not in ['args', 'argv', 'robot_def', 'ts']:
                print(k, v)
    rec_args = [rec.get('args', {}) for rec in recs]
    if num_recs == 1:
        print('rec_args', rec_args[0])
    if min([len(ra) for ra in rec_args]) > 0:
        rec_arg_keys = [ra.keys() for ra in rec_args if len(ra) == max(map(len, rec_args))][0]
        for k in rec_arg_keys:
            # v = rec[k]
            vs = [r.get(k) for r in rec_args]
            v = vs[0]
            if all([vv == v for vv in vs]):
                continue
            print('diff arg', k, vs)
    print({k: v.shape for k, v in rec.items() if k.startswith('states')})
    offsets = args.offsets
    offsets = [] if offsets is None else offsets
    offsets = offsets + [0] * (len(recs) - len(offsets))
    print('offsets', offsets)
    for rec, ofs in zip(recs, offsets):
        for k in rec:
            if k.startswith('states'):
                rec[k] = rec[k][ofs:]
    states_q = rec['states_q']
    states_q_ctrl = rec['states_q_ctrl']
    num_dofs = states_q.shape[1]
    num_steps_data = [rec['states_q'].shape[0] for rec in recs]
    num_steps = args.num_steps or states_q.shape[0]
    st = args.start
    st = 0 if max(num_steps_data) < st else st
    ed = min(num_steps + st, min(num_steps_data))
    print('num_steps_data', num_steps_data, st, ed)
    num_steps = ed - st
    dofs = args.dofs
    if dofs == 'auto':
        eps = 1e-2
        diff = np.max(states_q_ctrl, axis=0) - np.min(states_q_ctrl, axis=0)
        print(diff)
        cond = diff > eps
        dofs = np.where(cond)[0].tolist()
        print('auto dofs', len(dofs), dofs)
    dofs = eval(dofs) if isinstance(dofs, str) else dofs
    dofs = list(range(num_dofs)) if dofs is None else dofs
    dofs = list(dofs) if isinstance(dofs, range) else dofs
    dofs = list(range(num_dofs))[dofs] if isinstance(dofs, slice) else dofs
    dofs = [dofs] if not isinstance(dofs, (list, tuple)) else dofs
    dofs = dofs[:len(DOF_NAMES)]
    print('dofs', dofs)
    num_dofs = len(dofs)
    print('DOF_NAMES', [DOF_NAMES[i] for i in dofs])
    diff_qc = 0
    if num_recs > 1:
        fixed_inds0 = args.fixed_inds0
        if fixed_inds0 is not None:
            inds1 = list(range(num_recs))
            inds1 = [x for x in inds1 if x != fixed_inds0]
            inds0 = [fixed_inds0 for _ in range(len(inds1))]
        else:
            import itertools
            inds0, inds1 = map(list, zip(*list(itertools.combinations(range(num_recs), 2))))
        print(inds0, inds1)
        q_ctrls = [rec['states_q_ctrl'][st:ed, dofs] for rec in recs]
        qs = [rec['states_q'][st:ed, dofs] for rec in recs]
        q_ctrls = np.stack(q_ctrls, axis=0)
        qs = np.stack(qs, axis=0)
        print(q_ctrls[inds0].shape)
        # q_ctrls = [rec['states_q_ctrl'] for rec in recs]
        # print(q_ctrls[0][:64].tolist())
        # print(q_ctrls[1][:64].tolist())
        diff_qc = np.linalg.norm(q_ctrls[inds0] - q_ctrls[inds1], axis=(-1, -2))
        print('diff q_ctrls', diff_qc)
        diff = np.linalg.norm(qs[inds0] - qs[inds1], axis=(-1, -2))
        print('diff qs', diff)
        if np.all(diff <= 0) and input('continue?'):
            return
        # return
    if motion_output is not None:
        for k in rec:
            if k.startswith('states'):
                rec[k] = rec[k][st:ed]
        save_dir = motion_output
        from unicon.utils import rec2motion
        for n, r in zip(rec_names, recs):
            rec2motion(rec, name=n, dofs=dofs, save_dir=save_dir)
        return
    ext = args.ext
    if args.info:
        q_tau_limit = TAU_LIMIT
        qd_limit = QD_LIMIT
        q_min = Q_CTRL_MIN
        q_max = Q_CTRL_MAX
        q_init = Q_CTRL_INIT
        print('q_tau_limit', q_tau_limit[dofs])
        print('qd_limit', qd_limit[dofs])
        print('q_min', q_min[dofs])
        print('q_max', q_max[dofs])
        print('q_init', q_init[dofs])
        for rec in recs:
            states_q = rec['states_q']
            states_q_ctrl = rec['states_q_ctrl']
            states_qd = rec['states_qd']
            states_q_tau = rec.get('states_q_tau')
            q = states_q[st:ed]
            qc = states_q_ctrl[st:ed]
            qd = states_qd[st:ed]
            if states_q_tau is not None:
                q_tau = states_q_tau[st:ed]
            for k in ['qc', 'q']:
                print(k)
                v = locals().get(k)
                min_v = np.min(v, axis=0)
                max_v = np.max(v, axis=0)
                print('min', np.round(min_v[dofs].astype(np.float64), 2).tolist())
                print('max', np.round(max_v[dofs].astype(np.float64), 2).tolist())
                v_rng_n = (min_v - q_init) / (q_min - q_init)
                v_rng_p = (max_v - q_init) / (q_max - q_init)
                print('rng_n', np.round(v_rng_n[dofs], 2).tolist())
                print('rng_p', np.round(v_rng_p[dofs], 2).tolist())
                v_rng_l = (min_v - q_min) / (q_max - q_min)
                v_rng_r = (max_v - q_min) / (q_max - q_min)
                print('rng_l', np.round(v_rng_l[dofs], 2).tolist())
                print('rng_r', np.round(v_rng_r[dofs], 2).tolist())
                print('rng_lr', np.round((v_rng_r - v_rng_l)[dofs], 2).tolist())
            v = qd
            print('qd')
            min_v = np.min(v, axis=0)
            max_v = np.max(v, axis=0)
            print('min', np.round(min_v[dofs].astype(np.float64), 2).tolist())
            print('max', np.round(max_v[dofs].astype(np.float64), 2).tolist())
            v_rng_l = (min_v + qd_limit) / qd_limit / 2
            v_rng_r = (max_v + qd_limit) / qd_limit / 2
            print('rng_l', np.round(v_rng_l[dofs], 2).tolist())
            print('rng_r', np.round(v_rng_r[dofs], 2).tolist())
            print('rng_lr', np.round((v_rng_r - v_rng_l)[dofs], 2).tolist())
            if states_q_tau is None:
                continue
            v = q_tau
            print('q_tau')
            min_v = np.min(v, axis=0)
            max_v = np.max(v, axis=0)
            print('min', np.round(min_v[dofs].astype(np.float64), 2).tolist())
            print('max', np.round(max_v[dofs].astype(np.float64), 2).tolist())
            v_rng_l = (min_v + q_tau_limit) / q_tau_limit / 2
            v_rng_r = (max_v + q_tau_limit) / q_tau_limit / 2
            print('rng_l', np.round(v_rng_l[dofs], 2).tolist())
            print('rng_r', np.round(v_rng_r[dofs], 2).tolist())
            print('rng_lr', np.round((v_rng_r - v_rng_l)[dofs], 2).tolist())
        for rec in recs:
            for k, v in rec.items():
                if not isinstance(v, np.ndarray):
                    continue
                v = v[st:ed]
                print(k, v.shape)
                std = np.round(np.std(v, axis=0).astype(np.float64), decimals=3).tolist()
                mean = np.round(np.mean(v, axis=0).astype(np.float64), decimals=3).tolist()
                _min = np.round(np.min(v, axis=0).astype(np.float64), decimals=3).tolist()
                _max = np.round(np.max(v, axis=0).astype(np.float64), decimals=3).tolist()
                _med = np.round(np.median(v, axis=0).astype(np.float64), decimals=3).tolist()
                print('std', std)
                print('mean', mean)
                print('min', _min)
                print('max', _max)
                print('med', _med)
        return
    plot_root = args.plot_root
    os.makedirs(plot_root, exist_ok=True)
    if states_i_extras:
        nplts = num_recs
        # nplts = 3
        fig, axes = plt.subplots(1, nplts, figsize=(10 * 3, 10 * nplts))
        if num_recs > 1:
            axes = axes.reshape(-1, nplts)
            d_axes = axes[0]
        else:
            d_axes = [axes]
        # print('axes', axes.shape)
        x_scale = 0.01
        t = np.array(list(range(st, ed)))
        t = t * x_scale
        # ax1, ax2, ax3 = d_axes
        for i, rec in enumerate(recs):
            ax1 = d_axes[i]
            cmd = rec['states_cmd'][st:ed]
            print('cmd', np.max(cmd, axis=0))
            inputs = rec['states_input'][st:ed]
            cmd = inputs if np.max(cmd) == 0 else cmd
            # if i == 0:
            # ax1.plot([st], [0], marker='o')
            for k in range(cmd.shape[1]):
                ax1.plot(t, cmd[:, k])
            # if i == 0:
            # ax2.plot([st], [0], marker='o')
            # ax2.plot(t, cmd[:, 1])
            # if i == 0:
            # ax3.plot([st], [0], marker='o')
            # ax3.plot(t, cmd[:, 2])
            ax1.set_xticks(np.arange(st, ed, 200) * x_scale)
            max_cmd = max(1.5, np.max(np.abs(cmd)) + 0.1)
            ax1.set_ylim([-max_cmd, max_cmd])
            ax1.set_title(rec_names[i])
        ax1.legend([f'cmd_{i}' for i in range(cmd.shape[1])])
        # ax2.set_ylim([-1, 1])
        # ax3.set_ylim([-1, 1])
        fig.tight_layout()
        plot_dir = f'{plot_root}/'
        plot_prefix = plot_dir
        plt.savefig(plot_prefix + f'sie.{ext}')
        plt.close()
        return
    from unicon.utils import quat_rotate_inverse_np, rpy2quat_np, quat2rpy_np3
    from unicon.utils import quat_rotate_np
    states_imu = args.states_imu
    if states_imu:
        nplts = 8
        # fig, axes = plt.subplots(1, nplts, figsize=(10 * nplts, 10 * 1))
        fig, axes = plt.subplots(nplts, 1, figsize=(10 * 1, 10 * nplts))
        axes = axes.reshape(-1, nplts)
        d_axes = axes[0]
        print('axes', axes.shape)
        t = list(range(st, ed))
        max_yaw = 1.0
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = d_axes
        for i, rec in enumerate(recs):
            quat = rec['states_quat'][st:ed]
            rpy = rec['states_rpy'][st:ed]
            ang_vel = rec['states_ang_vel'][st:ed]
            rpy2 = quat2rpy_np3(quat)
            yaw = rpy[:, 2]
            yaw = yaw - yaw[0] - 1.
            # yaw = yaw - yaw[0]
            if i == 0:
                ax1.plot([t[0]], [0], marker='o')
            ax1.plot(t, rpy[:, 0])
            if i == 0:
                ax2.plot([t[0]], [0], marker='o')
            ax2.plot(t, rpy[:, 1])
            if i == 0:
                ax3.plot([0], [0], marker='o')
            ax3.plot(rpy[:, 0], rpy[:, 1])
            if i == 0:
                ax4.plot([0], [0], marker='o')
            ax4.plot(rpy[:, 0], ang_vel[:, 0])
            if i == 0:
                ax5.plot([0], [0], marker='o')
            ax5.plot(rpy[:, 1], ang_vel[:, 1])
            if i == 0:
                ax6.plot([0], [0], marker='o')
            ax6.plot(rpy[:, 1], rpy2[:, 1])
            if i == 0:
                ax7.plot([0], [0], marker='o')
            ax7.plot(yaw, ang_vel[:, 2])
            if i == 0:
                ax8.plot([0], [0], marker='o')
            ax8.plot(rpy[:, 0], yaw)
            max_yaw = max(max_yaw, np.max(np.abs(yaw)) + 0.1)
        max_ang = max(0.2, np.max(np.abs(rpy[:, :2])) + 0.1)
        max_ang_vel = max(1.2, np.max(np.abs(ang_vel[:, :2])) + 0.2)
        ax1.set_ylim([-max_ang, max_ang])
        ax2.set_ylim([-max_ang, max_ang])
        ax3.set_ylim([-max_ang, max_ang])
        ax3.set_xlim([-max_ang, max_ang])
        ax4.set_ylim([-max_ang_vel, max_ang_vel])
        ax4.set_xlim([-max_ang, max_ang])
        ax5.set_ylim([-max_ang_vel, max_ang_vel])
        ax5.set_xlim([-max_ang, max_ang])
        # ax6.set_ylim([-1., 1.])
        # ax6.set_xlim([-1., 1.])
        ax7.set_ylim([-max_ang_vel, max_ang_vel])
        ax7.set_xlim([-max_yaw, max_yaw])
        ax8.set_ylim([-max_yaw, max_yaw])
        ax8.set_xlim([-max_yaw, max_yaw])
        ax1.legend(['ref'] + rec_names, loc="lower right")
        ax1.set_title('r')
        ax2.set_title('p')
        ax3.set_title('rp')
        ax4.set_title('r rv')
        ax5.set_title('p pv')
        ax6.set_title('p quat_p')
        fig.tight_layout()
        plot_dir = f'{plot_root}/'
        plot_prefix = plot_dir
        plt.savefig(plot_prefix + f'simu.{ext}')
        plt.close()
        return
    if states_x_extras:
        for n, rec in zip(rec_names, recs):
            print(n)
            for k in ['states_pos', 'states_lin_vel', 'states_lin_acc']:
                # for k in ['states_pos', 'states_lin_acc']:
                print(k)
                if k not in rec:
                    continue
                v = rec[k][st:ed]
                print(np.round(np.min(v, axis=0).astype(np.float64), decimals=3).tolist())
                print(np.round(np.max(v, axis=0).astype(np.float64), decimals=3).tolist())
                print(np.round(np.max(np.abs(v), axis=0).astype(np.float64), decimals=3).tolist())
                print(np.round(np.mean(v, axis=0).astype(np.float64), decimals=3).tolist())
        nplts = 6
        fig, axes = plt.subplots(1, nplts, figsize=(10 * nplts, 10 * 1))
        axes = axes.reshape(-1, nplts)
        d_axes = axes[0]
        print('axes', axes.shape)
        t = list(range(st, ed))
        ax1, ax2, ax3, ax4, ax5, ax6 = d_axes
        # g = -9.81
        # g = -9.83
        # g = -9.837
        g = -9.84
        g = -9.842
        g = -9.8465
        g = args.g
        print('g', g)
        h0 = 0.9
        # g = -9.9
        # vec_g = np.array([0, 0, g])
        acc_s = np.array([
            1.,
            1.,
            1.,
        ])
        acc_b = np.array([
            0.,
            0.,
            0.,
        ])
        acc_world_frame = args.acc_world_frame
        for i, rec in enumerate(recs):
            quat = rec['states_quat'][st:ed]
            rpy = rec['states_rpy'][st:ed]
            pos = rec['states_pos'][st:ed]
            pos[:, :2] = pos[:, :2] - pos[0:1, :2]
            vel = rec['states_lin_vel'][st:ed]
            acc = rec['states_lin_acc'][st:ed]
            xd_b = (np.pad(pos, ((0, 1), (0, 0)), 'edge')[1:] - np.pad(pos, ((1, 0), (0, 0)), 'edge')[:-1]) / (2 * dt)
            # xd_i = (np.pad(pos, (0,1), 'edge')[1:] - np.pad(pos, (1,0), 'edge')[:-1]) / (2 * dt)
            gv = np.zeros((len(quat), 3))
            gv[:, 2] = 1
            quat2 = rpy2quat_np(rpy)
            # quat2 = rpy2quat_np(rpy[:, [1,0,2]])
            rpy2 = quat2rpy_np3(quat)
            g_w = gv * g
            g_proj = quat_rotate_inverse_np(quat, gv)
            # g_proj = quat_rotate_inverse_np(quat2, gv)
            g_proj *= g
            # g_proj = quat_rotate_inverse_np2(quat, gv)
            if np.max(np.abs(vel)) == 0:
                print('null vel')
                vel = xd_b
            # print(vel.shape)
            # print(xd_b.shape)
            xdd_b = (np.pad(vel, ((0, 1), (0, 0)), 'edge')[1:] - np.pad(vel, ((1, 0), (0, 0)), 'edge')[:-1]) / (2 * dt)
            if np.max(np.abs(acc)) == 0:
                print('null acc')
                acc = xdd_b
                # print(xdd_b.shape)
            print('pos', pos[0].tolist())
            # print('rpy', rpy[0].tolist())
            # print('quat', quat[0].tolist())
            # print('quat2', quat2[0].tolist())
            print('g_proj', g_proj[0].tolist())
            print('acc', acc[0].tolist())
            # print('g_proj', g_proj[10].tolist())
            # print('acc', acc[10].tolist())
            # nacc = acc - g_proj
            acc = acc * acc_s + acc_b
            # print(nacc)
            if acc_world_frame:
                acc_w = acc
            else:
                acc_w = quat_rotate_np(quat, acc)
            nacc_w = acc_w - g_w
            # print(nacc_w)
            # xd_i = np.trapezoid(nacc_w, dx=dt, axis=-1)
            nacc_wa = 0.5 * (nacc_w[:-1] + nacc_w[1:])
            # print(nacc_wa[:10].tolist())
            # print((nacc_wa * dt)[:, 2])
            xd0 = np.zeros((1, 3))
            xd_i = np.cumsum(nacc_wa * dt, axis=0)
            # print(xd_i[:, 2])
            xd_i = np.concatenate([xd0, xd0 + xd_i], axis=0)
            # vel = xd_i
            x0 = np.zeros((1, 3))
            x0[:, 2] = h0
            xd_i_wa = 0.5 * (xd_i[:-1] + xd_i[1:])
            x_i = np.cumsum(xd_i_wa * dt, axis=0)
            x_i = np.concatenate([x0, x0 + x_i], axis=0)
            if i == 0:
                ax1.plot([0], [0], marker='o')
            ax1.plot(pos[:, 0], pos[:, 1])
            ax1.plot(x_i[:, 0], x_i[:, 1])
            if i == 0:
                ax2.plot(t, [h0] * (ed - st), marker='.')
            ax2.plot(t, pos[:, 2], marker='.')
            ax2.plot(t, x_i[:, 2], marker='.')
            if i == 0:
                ax3.plot(t, xd_b[:, 0], marker='.')
            ax3.plot(t, vel[:, 0], marker='.')
            ax3.plot(t, vel[:, 1], marker='.')
            ax3.plot(t, vel[:, 2], marker='.')
            ax3.plot(t, xd_i[:, 0], marker='.')
            ax3.plot(t, xd_i[:, 1], marker='.')
            ax3.plot(t, xd_i[:, 2], marker='.')
            if i == 0:
                ax4.plot(t, [0] * (ed - st), marker='.')
            # ax4.plot(t, acc[:, 0], marker='.')
            # ax4.plot(t, acc[:, 1], marker='.')
            ax4.plot(t, rpy2[:, 0], marker='.')
            ax4.plot(t, rpy2[:, 1], marker='.')
            ax4.plot(t, rpy[:, 0], marker='.')
            ax4.plot(t, rpy[:, 1], marker='.')
            if i == 0:
                ax5.plot(t, [0] * (ed - st), marker='.')
            # ax5.plot(t, g * np.sin(rpy[:, 1]), marker='.')
            # ax5.plot(t, g_proj[:, 0], marker='.')
            # ax5.plot(t, g_proj[:, 1], marker='.')
            # ax5.plot(t, g_proj[:, 2], marker='.')
            # ax5.plot(t, -g * np.sin(rpy[:, 0]), marker='.')
            # ax5.plot(t, -g * np.sin(rpy[:, 1]), marker='.')
            # ax5.plot(t, acc[:, 1]-g * np.sin(rpy[:, 0]), marker='.')
            # ax5.plot(t, acc[:, 0]+g * np.sin(rpy[:, 1]), marker='.')
            ax5.plot(t, nacc_w[:, 0], marker='.')
            ax5.plot(t, nacc_w[:, 1], marker='.')
            ax5.plot(t, nacc_w[:, 2], marker='.')

            ax6.plot(t, acc[:, 0], marker='.')
            ax6.plot(t, acc[:, 1], marker='.')
            ax6.plot(t, acc[:, 2], marker='.')
        max_xy = max(np.max(np.abs(x_i[:, :2])), np.max(np.abs(pos[:, :2]))) + 0.5
        print('max_xy', max_xy)
        # max_xy = min(max_xy, 20)
        ax1.set_xlim([-max_xy, max_xy])
        ax1.set_ylim([-max_xy, max_xy])
        ax2.set_ylim([0, 2])
        ax3.set_ylim([-1, 1])
        ax4.set_ylim([-0.5, 0.5])
        # ax5.set_ylim([-1, 1])
        ax3.legend(['ref', 'vel x', 'vel y', 'vel z', 'xdi x', 'xdi y', 'xdi z'], loc="lower right")
        ax4.legend(['ref', 'qrpy r', 'qrpy p', 'rpy r', 'rpy p'], loc="lower right")
        ax1.legend(['ref', 'pos', 'xi'], loc="lower right")
        ax1.set_title('xy')
        ax2.set_title('z')
        ax3.set_title('xd')
        ax4.set_title('rpy')
        ax5.set_title('nacc_w')
        ax6.set_title('acc')
        fig.tight_layout()
        plot_dir = f'{plot_root}/'
        plot_prefix = plot_dir
        plt.savefig(plot_prefix + f'sxe.{ext}')
        plt.close()
        return
    if states_x_extras2:
        fig = plot_states_x_extras2(rec_names, recs, st, ed, robot_def)
        plot_dir = f'{plot_root}/'
        plot_prefix = plot_dir
        plt.savefig(plot_prefix + f'sxe2.{ext}')
        plt.close()
        return
    if states_q_extras:
        for n, rec in zip(rec_names, recs):
            print(n)
            q_tau = rec['states_q_tau'][st:ed, dofs]
            q_cur = rec['states_q_cur'][st:ed, dofs]
            print('states_q_tau')
            print(np.round(np.min(q_tau, axis=0).astype(np.float64), decimals=1).tolist())
            print(np.round(np.max(q_tau, axis=0).astype(np.float64), decimals=1).tolist())
            print(np.round(np.max(np.abs(q_tau), axis=0).astype(np.float64), decimals=1).tolist())
            print('states_q_cur')
            print(np.round(np.min(q_cur, axis=0).astype(np.float64), decimals=1).tolist())
            print(np.round(np.max(q_cur, axis=0).astype(np.float64), decimals=1).tolist())
            print(np.round(np.max(np.abs(q_cur), axis=0).astype(np.float64), decimals=1).tolist())
            print('ratio')
            # print(np.round(np.mean(np.abs(q_cur)/(np.abs(q_tau) + 1e-6), axis=0).astype(np.float64), decimals=3).tolist())
            # tc_ratio = np.mean(np.abs(q_tau/(q_cur)), axis=0)
            # print(np.round(np.abs(q_tau), decimals=3).flatten())
            # print(np.round(np.abs(q_cur), decimals=3).flatten())
            # print(np.round(np.abs(q_tau)/np.abs(q_cur)).flatten())
            tc_ratio = np.mean((q_tau + 1e-1) / (q_cur + 1e-1), axis=0)
            # tc_ratio = np.max(q_tau, axis=0)/np.max(q_cur, axis=0)
            print(np.round(tc_ratio.astype(np.float64), decimals=3).tolist())
            mmc = rec['args'].get('motor_max_current')
            if mmc is not None:
                mmc = float(mmc)
                print('mmc', mmc)
                print('torque limits')
                print(np.round((mmc * np.abs(tc_ratio)).astype(np.float64)).tolist())
        nplts = 4
        fig, axes = plt.subplots(num_dofs, nplts, figsize=(10 * nplts, 10 * num_dofs))
        axes = axes.reshape(-1, nplts)
        print('axes', axes.shape)
        t = list(range(st, ed))
        for k, (d, d_axes) in enumerate(zip(dofs, axes)):
            ax1, ax2, ax3 = d_axes[:3]
            ax4 = d_axes[3]
            for i, rec in enumerate(recs):
                q = rec['states_q'][st:ed, d]
                qd_b = (np.pad(q, (0, 1), 'edge')[1:] - np.pad(q, (1, 0), 'edge')[:-1]) / (2 * dt)
                qd = rec['states_qd'][st:ed, d]
                q_ctrl = rec['states_q_ctrl'][st:ed, d]
                q_tau = rec['states_q_tau'][st:ed, d]
                q_cur = rec['states_q_cur'][st:ed, d]
                states_q_temp = rec.get('states_q_temp')
                if states_q_temp is not None:
                    q_temp = states_q_temp[st:ed, d]
                # q_rel = q_ctrl - q
                # ax1.plot(t, q_rel)

                # kp = KP[d]
                # kd = KD[d]
                kp = KP_2[d]
                kd = KD_2[d]

                q_ctrl_last = np.concatenate([q_ctrl[0:1], q_ctrl[0:-1]])
                tau_ref = kp * (q_ctrl_last - q) + kd * (-qd)

                # tau_ref = kp * (q_ctrl - q) + kd * (-qd)

                # tau_ref = kp * (q_ctrl - q) + kd * (-qd_b)

                # tau_ref = kp * (q_ctrl[:-1] - q[:-1]) + kd * (-qd[:-1])
                # tau_ref = np.concatenate([[0], tau_ref])

                # print('q', q[0])
                # print('q_ctrl', q_ctrl[0])
                # print('q_tau', q_tau[0])
                # print('tau_ref', tau_ref[0])
                if i == 0:
                    ax1.plot(t, tau_ref)
                ax1.plot(t, q_tau)
                # ax1.plot(t, q_cur)

                # ax2.plot(q_tau, q_cur)
                if i == 0:
                    ax2.plot(q_tau, q_tau)
                    # ax2.plot(t, tau_ref / tc_ratio[k])
                # ax2.plot(t, q_cur)
                # ax2.plot(qd, q_tau)
                # ax2.plot(qd, q_cur)
                ax2.plot(q_tau, q_cur)
                # ax2.plot(qd_b, q_cur)
                # tau_ref = kp * (q_ctrl - q) + kd * (-qd)
                # tau_ref = kp * (q_ctrl - q)
                # tau_limit = mmc * tc_ratio[k]
                # tau_ref = np.clip(tau_ref, -tau_limit, tau_limit)
                if i == 0:
                    ax3.plot(tau_ref, tau_ref)
                # ax3.plot(q_rel, q_cur)
                ax3.plot(tau_ref, q_tau)

                if states_q_temp is not None:
                    if i == 0:
                        ax4.plot(t, [30] * len(t))
                    ax4.plot(t, q_temp)
            name = DOF_NAMES[d]
            tau_max = TAU_LIMIT[k]
            # ax1.set_ylim([-mmc * 4, mmc * 4])
            ax1.set_ylim([-tau_max, tau_max])
            # ax2.set_ylim([-mmc * 2, mmc * 2])
            ax3.set_xlim([-tau_max, tau_max])
            ax3.set_ylim([-tau_max, tau_max])
            # ax1.set_ylim([-80, 80])
            # ax2.set_ylim([-40, 40])
            # ax3.set_ylim([-160, 160])
            # ax3.set_xlim([-160, 160])
            ax4.set_ylim([0, 80])
            ax1.legend(['ref'] + rec_names, loc="lower right")
            ax1.set_title(name + ' sqe')
        fig.tight_layout()
        plot_dir = f'{plot_root}/'
        plot_prefix = plot_dir
        plt.savefig(plot_prefix + f'sqe.{ext}')
        plt.close()
        return
    if num_recs > 1 and eval_loss:
        num_losses = len(loss_fns)
        num_pairs = len(inds0)
        import torch
        trajs = []
        for rec in recs:
            states_qd = rec['states_qd']
            states_q = rec['states_q']
            states_q_ctrl = rec['states_q_ctrl']
            q = states_q[st:ed, dofs]
            qd = states_qd[st:ed, dofs]
            qc = states_q_ctrl[st:ed, dofs]
            # s = np.concatenate([q, qd], axis=-1)
            # traj = q
            # traj = np.concatenate([q, 2 * dt * qd, qc], axis=-1)
            traj = np.concatenate([q, 2 * dt * qd], axis=-1)
            trajs.append(traj)
        trajs = np.stack(trajs, axis=0)
        trajs = torch.from_numpy(trajs)
        print('trajs', trajs.shape)
        traj_len = args.traj_len
        num_segs = trajs.shape[1] // traj_len
        print('traj_len', traj_len)
        print('num_segs', num_segs)
        trajs = trajs[:, :num_segs * traj_len, :].reshape(num_recs, num_segs, traj_len, -1)
        print('trajs', trajs.shape)
        plot_dir = f'{plot_root}/'
        import torch
        unicon.losses.DEBUG = True
        loss_scale = traj_len * trajs[0].shape[-1]
        print('loss_scale', loss_scale, st, ed)
        pair_names = []
        for i1, i2 in zip(inds0, inds1):
            pair_name = f'{rec_names[i1], rec_names[i2]}'
            pair_names.append(pair_name)
        fig, axes = plt.subplots(1, num_losses, figsize=(10 * num_losses, 10 * 1))
        axes = axes.flatten().tolist() if num_losses > 1 else [axes]
        maxs = [0] * num_pairs
        for i1, i2 in zip(inds0, inds1):
            print('traj loss', i1, i2, rec_names[i1], rec_names[i2])
            s0 = trajs[i1]
            s1 = trajs[i2]
            loss_all = []
            for k, loss_fn in loss_fns.items():
                loss = loss_fn(s0, s1)
                loss *= loss_scale
                std, mean = torch.std_mean(loss)
                print(k, std.item(), mean.item())
                loss_all.append(loss)
            # fig, axes = plt.subplots(1, nplts, figsize=(10 * nplts, 10 * 1))
            # axes = axes.flatten().tolist() if nplts > 1 else [axes]
            for i, k in enumerate(loss_fns):
                ax = axes[i]
                loss = loss_all[i]
                # print(loss.shape, loss.tolist())
                t = list(range(len(loss)))
                ax.plot(t, loss, marker='.')
                # ax.set_ylim([-0.01, loss.max() * 1.5 + 0.1])
                ax.set_title(k)
            fig.tight_layout()
            plt.subplots_adjust(top=0.92)
            # plt.suptitle(f'traj loss {rec_names[i1], rec_names[i2]}')
            # plt.savefig(plot_prefix + f'loss_traj_{i1}_{i2}.{ext}')
        ax.legend(pair_names)
        plot_prefix = plot_dir
        plt.savefig(plot_prefix + f'loss_traj.{ext}')
        plt.close()
        unicon.losses.DEBUG = False

        if num_recs == 2:
            unicon.losses.plot_unfolded_mse_loss()
            unicon.losses.plot_chamfer_loss()

        COLORS_5 = [
            # '#000000',
            '#E69F00',
            '#56B4E9',
            '#009E73',
            '#F0E442',
            '#0072B2',
            '#D55E00',
            '#CC79A7',
        ]

        def hex_to_rgb(hexes):
            return [list(int(color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)) for color in hexes]

        colors = hex_to_rgb(COLORS_5)

        figsize = (20 * num_losses, 7 * (num_pairs + 1))
        fig, axes = plt.subplots(1, num_losses, figsize=figsize, layout='constrained')
        axes = axes.flatten().tolist() if num_losses > 1 else [axes]
        dof_names = [DOF_NAMES[i] for i in dofs]
        elm_names = sum([[f'{t} - {n}' for n in dof_names] for t in ['q', 'qd', 'qc']], [])
        multiplier = -(num_pairs // 2)
        maxs = np.zeros((num_pairs, num_losses))
        loss_scale = traj_len
        for m, (i1, i2) in enumerate(zip(inds0, inds1)):
            print('element loss', i1, i2, rec_names[i1], rec_names[i2])
            s0 = trajs[i1]
            s1 = trajs[i2]
            num_elms = s0.shape[-1]
            loss_all = {k: [] for k in loss_fns}
            for i in range(num_elms):
                # print(elm_names[i])
                for k, loss_fn in loss_fns.items():
                    loss = loss_fn(s0[..., i:i + 1], s1[..., i:i + 1])
                    loss *= loss_scale
                    traj_std, mean = [v.item() for v in torch.std_mean(loss)]
                    traj_std = 0 if len(loss) == 1 else traj_std
                    # print(name, traj_std, std, mean)
                    loss_all[k].append([traj_std, mean])
            for name in loss_fns:
                print(name, [x[-1] for x in loss_all[name]])
            x = np.arange(num_elms) * num_pairs * 0.5
            width = 0.3  # the width of the bars
            labels = elm_names[:num_elms]
            for i, k in enumerate(loss_all):
                ax = axes[i]
                loss = loss_all[k]
                loss = np.array(loss)
                traj_std, mean = loss.T
                offset = width * multiplier
                # offset = 0
                # rects = ax.bar(x + offset, mean, width, label=k, yerr=traj_std)
                rects = ax.barh(x + offset,
                                mean,
                                height=width,
                                label=k,
                                xerr=traj_std,
                                tick_label=labels,
                                color=colors[m])
                ax.bar_label(rects, padding=5)
                ax.set_yticks(x)
                ax.set_yticklabels(labels, fontsize='xx-large')
                ax.set_title(k)
                maxs[m, i] = max(maxs[m, i], np.max(mean + traj_std))
                # ax.set_xlim(0, max(0.5, np.max(mean + traj_std) + 0.5))
                ax.set_xlim(0)
            multiplier += 1
            # ax.legend(loc='upper left', ncols=3)
            # fig.tight_layout()
            # plt.suptitle(f'elm loss ')
            # plt.savefig(plot_prefix + f'loss_elm_{i1}_{i2}.{ext}')
        for i, ax in enumerate(axes):
            ax.set_xlim(0, max(np.max(maxs[:, i]) * 1.2, 0.1))
        ax.legend(pair_names)
        plot_prefix = plot_dir
        plt.savefig(plot_prefix + f'loss_elm.{ext}')
        plt.close()
        return
    check_int = args.check_int
    plot_root_states = not args.no_root_states
    plot_root_states = do_plot and plot_root_states
    if plot_root_states:
        nplts = 6 + 2 + 2 + 2
        x_scale = 1 + int(num_steps // 1500)
        fig, axes = plt.subplots(nplts, num_recs, figsize=(10 * x_scale * num_recs, 10 * nplts))
        axes = axes.reshape(axes.shape[0], -1)
        axes = axes.T
        h_num_dofs = (len(dofs) + 1) // 2
        dof_names = [DOF_NAMES[i] for i in dofs]
        max_rp = None
        max_q = None
        for axs, rec, name in zip(axes, recs, rec_names):
            ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12 = axs
            t = list(range(st, ed))
            states_quat = rec.get('states_quat')
            states_rpy = rec.get('states_rpy')
            states_ang_vel = rec.get('states_ang_vel')
            states_q_ctrl = rec['states_q_ctrl']
            states_q = rec['states_q']
            states_qd = rec['states_qd']
            states_q_tau = rec.get('states_q_tau')
            states_q_cur = rec.get('states_q_cur')
            send_ts = rec.get('send_ts')
            recv_ts = rec.get('recv_ts')
            name = name + '   '
            qc = states_q_ctrl[st:ed, dofs]
            q = states_q[st:ed, dofs]
            qd = states_qd[st:ed, dofs]
            if max_q is None:
                max_q = max(np.max(np.abs(qc)) + 0.1, 2)
            if states_quat is not None:
                quat = states_quat[st:ed]
                rpy = states_rpy[st:ed, :2]
                yaw = states_rpy[st:ed, 2]
                q_rpy = quat2rpy_np3(quat)[:, :2]
                ang_vel = states_ang_vel[st:ed]
                ax1.set_title(name + 'rpy')
                ax1.plot(t, rpy, marker='.')
                ax1.plot(t, q_rpy)
                # ax1.plot(t, yaw, marker='.')
                # ax1.set_ylim([-3, 3])
                legends = ['er', 'ep', 'qr', 'qp']
                if max_rp is None:
                    max_rp = max(np.max(np.abs(rpy)) + 0.05, 0.1)
                    max_ang_vel = max(np.max(np.abs(ang_vel)) + 0.2, 1.5)
                ax1.set_ylim([-max_rp, max_rp])
                ax1.legend(legends)
                ax1.set_title(name + 'rpy')
                ax2.plot(t, ang_vel, marker='.')
                ax2.set_ylim([-max_ang_vel, max_ang_vel])
                ax2.legend(['ang_vel_r', 'ang_vel_p', 'ang_vel_y'])
                ax2.set_title(name + 'ang_vel')
            elif recv_ts is not None:
                recv_ts = recv_ts[:, dofs]
                tt = list(range(len(recv_ts)))
                ax1.plot(tt, recv_ts[:, :h_num_dofs], marker='.')
                ax2.plot(tt, recv_ts[:, h_num_dofs:], marker='.')
                ax1.set_title(name + 'recv_ts_1')
                ax2.set_title(name + 'recv_ts_2')
            ax3.plot(t, qc[:, :h_num_dofs], marker='.')
            ax3.legend(dof_names[:h_num_dofs], loc="lower right")
            ax3.set_title(name + 'q_ctrl_1')
            ax3.set_ylim([-max_q, max_q])
            ax4.plot(t, qc[:, h_num_dofs:], marker='.')
            ax4.legend(dof_names[h_num_dofs:], loc="lower right")
            ax4.set_title(name + 'q_ctrl_2')
            ax4.set_ylim([-max_q, max_q])
            ax5.plot(t, q[:, :h_num_dofs], marker='.')
            ax5.legend(dof_names[:h_num_dofs], loc="lower right")
            ax5.set_title(name + 'q_1')
            ax5.set_ylim([-max_q, max_q])
            ax6.plot(t, q[:, h_num_dofs:], marker='.')
            ax6.legend(dof_names[h_num_dofs:], loc="lower right")
            ax6.set_title(name + 'q_2')
            ax6.set_ylim([-max_q, max_q])
            ax7.plot(t, qd[:, :h_num_dofs], marker='.')
            ax7.legend(dof_names[:h_num_dofs], loc="lower right")
            ax7.set_title(name + 'qd_1')
            qd_limit = max(min(10, np.max(QD_LIMIT)), np.max(np.abs(qd)) + 1)
            ax7.set_ylim([-qd_limit, qd_limit])
            ax8.plot(t, qd[:, h_num_dofs:], marker='.')
            ax8.legend(dof_names[h_num_dofs:], loc="lower right")
            ax8.set_title(name + 'qd_2')
            ax8.set_ylim([-qd_limit, qd_limit])

            if check_int:
                _t = t
                eps = 1e-7
                int_span = 1
                int_span = 2
                q1 = qd[:, :h_num_dofs]
                df = (np.pad(q1, ((0, int_span),
                                  (0, 0)), 'edge')[int_span:] - np.pad(q1, ((int_span, 0), (0, 0)), 'edge')[:-int_span])
                cond = (np.abs(df) < eps).astype(np.float32)
                cond = (cond * np.cumsum([1] * cond.shape[1]))
                ax9.plot(_t, cond, marker='.')
                ax9.legend(dof_names[:h_num_dofs], loc="lower right")
                ax9.set_title(name + 'int_1')
                ax9.set_ylim([0, q1.shape[1] + 1])
                q2 = qd[:, h_num_dofs:]
                df = (np.pad(q2, ((0, int_span),
                                  (0, 0)), 'edge')[int_span:] - np.pad(q2, ((int_span, 0), (0, 0)), 'edge')[:-int_span])
                cond = (np.abs(df) < eps).astype(np.float32)
                cond = (cond * np.cumsum([1] * cond.shape[1]))
                ax10.plot(_t, cond, marker='.')
                ax10.legend(dof_names[h_num_dofs:], loc="lower right")
                ax10.set_title(name + 'int_2')
                ax10.set_ylim([0, q2.shape[1] + 1])
                ax10.set_xticks(np.arange(st, ed, 128))
            elif states_rpy is not None:
                roll = states_rpy[st:ed, 0]
                pitch = states_rpy[st:ed, 1]
                ax9.plot(roll, pitch)
                ax9.set_xlim([-max_rp, max_rp])
                ax9.set_ylim([-max_rp, max_rp])
                ax9.set_title(name + 'pitch - roll')
                roll_vel = states_ang_vel[st:ed, 0]
                pitch_vel = states_ang_vel[st:ed, 1]
                ax10.plot(roll_vel, pitch_vel)
                ax10.set_xlim([-max_ang_vel, max_ang_vel])
                ax10.set_ylim([-max_ang_vel, max_ang_vel])
                ax10.set_title(name + 'pitch_vel - roll_vel')

            if states_q_tau is not None and np.sum(np.abs(states_q_tau)) > 1e-3:
                tau_max = np.max(TAU_LIMIT[dofs])
                q_tau = states_q_tau[st:ed, dofs]
                ax11.plot(t, q_tau[:, :h_num_dofs], marker='.')
                ax12.plot(t, q_tau[:, h_num_dofs:], marker='.')
                ax11.legend(dof_names[:h_num_dofs], loc="lower right")
                ax12.legend(dof_names[h_num_dofs:], loc="lower right")
                ax11.set_title(name + 'q_tau_1')
                ax12.set_title(name + 'q_tau_2')
                ax11.set_ylim([-tau_max, tau_max])
                ax12.set_ylim([-tau_max, tau_max])
            elif states_q_cur is not None and np.sum(np.abs(states_q_cur)) > 1e-3:
                q_cur = states_q_cur[st:ed, dofs]
                ax11.plot(t, q_cur[:, :h_num_dofs], marker='.')
                ax11.legend(dof_names[:h_num_dofs], loc="lower right")
                ax11.set_title(name + 'q_cur_1')
                ax11.set_ylim([-50, 50])
                ax12.plot(t, q_cur[:, h_num_dofs:], marker='.')
                ax12.legend(dof_names[h_num_dofs:], loc="lower right")
                ax12.set_title(name + 'q_cur_2')
                ax12.set_ylim([-50, 50])
        fig.tight_layout()
        plot_dir = f'{plot_root}/'
        plot_prefix = plot_dir
        plt.savefig(plot_prefix + f'root_states.{ext}')
        plt.close()
    plot_rel = True
    plot_rel = False
    print('plot_rel', plot_rel)
    single_plot = args.single_plot
    auto_lim = args.auto_lim
    if args.no_dof_states:
        return
    nplts = 4
    if do_plot and single_plot:
        fig, axes = plt.subplots(num_dofs, nplts, figsize=(10 * nplts, 10 * num_dofs))
        axes = axes.reshape(axes.shape[0], -1)
        print('axes', axes.shape)
    qc_ofs = -1
    qc_ofs = 0
    print('qc_ofs', qc_ofs)
    aniext = args.aniext
    save_anim = aniext is not None
    for i, idx in enumerate(dofs):
        print(idx, DOF_NAMES[idx])
        t = list(range(st, ed))
        if no_plot:
            continue
        elif single_plot:
            axs = axes[i]
        else:
            if args.subdir:
                plot_dir = f'{plot_root}/{idx}/'
                plot_prefix = plot_dir
            else:
                plot_dir = f'{plot_root}'
                plot_prefix = f'{plot_dir}/dof_{idx}'
            os.makedirs(plot_dir, exist_ok=True)
            fig, axes = plt.subplots(1, nplts, figsize=(10 * nplts, 10))
            axes = axes.flatten().tolist()
            axs = axes
        ax1, ax2, ax3 = axs[:3]
        ax4 = axs[3]
        first = True
        qrels = []
        qs = []
        qcs = []
        axlines = []
        for rec in recs:
            states_qd = rec['states_qd']
            states_q = rec['states_q']
            states_q_ctrl = rec['states_q_ctrl']
            states_q_tau = rec.get('states_q_tau')
            states_q_cur = rec.get('states_q_cur')
            q = states_q[st:ed, idx]
            qd = states_qd[st:ed, idx]
            qc = states_q_ctrl[st + qc_ofs:ed + qc_ofs, idx]
            qs.append(q)
            qcs.append(qc)
            qrel = q - qc
            qrels.append(qrel)
            if no_plot:
                continue
            print(len(q), len(qc))
            if first:
                # ax1.plot([Q_CTRL_INIT[idx]], [0], marker='o')
                qc_d_b = (np.pad(qc, (0, 1), 'edge')[1:] - np.pad(qc, (1, 0), 'edge')[:-1]) / (2 * dt)
                line, = ax1.plot(qc, qc_d_b, marker='.')
                axlines.append([line, qc, qc_d_b])
            line, = ax1.plot(q, qd, marker='.')
            axlines.append([line, q, qd])
            # fig.savefig(plot_prefix+'qnqd.{ext}')
            # fig.close()
            # fig = plt.figure()
            if first:
                line, = ax2.plot(qc, qc)
                axlines.append([line, qc, qc])
            line, = ax2.plot(qc, q, marker='.')
            axlines.append([line, qc, q])
            # fig.savefig(plot_prefix+'qcnq.{ext}')
            # fig.close()
            # fig = plt.figure(figsize=figsize)
            # qc = qc[:num_steps]
            # q = q[:num_steps]
            # if first or plot_qc_all:
            #     ax3.plot(t, qc, marker='.')
            # ax3.plot(t, q, marker='.')
            if first:
                line, = ax3.plot(t, qc, marker='.')
                axlines.append([line, t, qc])
            if plot_rel:
                line, = ax3.plot(t, qrel, marker='.')
                axlines.append([line, t, qrel])
            else:
                line, = ax3.plot(t, q, marker='.')
                axlines.append([line, t, q])

            # ax3.set_xlim([Q_CTRL_MIN, Q_CTRL_MAX])
            # ax3.set_ylim([Q_CTRL_MIN[idx], Q_CTRL_MAX[idx]])
            # fig.savefig(plot_prefix+'tnqcnq.{ext}')
            kp = KP_2[idx]
            kd = KD_2[idx]
            if first:
                if states_q_cur is not None:
                    cur = states_q_cur[st:ed, idx]
                    line, = ax4.plot(t, cur, marker='.')
                    axlines.append([line, t, cur])
                else:
                    tau_ref = kp * 0.5 * qc + kd * (-np.mean(np.abs(qd)) * 0.5)
                    line, = ax4.plot(t, tau_ref)
                    axlines.append([line, t, tau_ref])
            if states_q_tau is not None:
                tau = states_q_tau[st:ed, idx]
                line, = ax4.plot(t, tau, marker='.')
                axlines.append([line, t, tau])
            else:
                tau = kp * (qc - q) + kd * (-qd)
                line, = ax4.plot(t, tau)
                axlines.append([line, t, tau])
                # ax4.plot(t, kp * qc)
            # fig.close()
            first = False
        if no_plot:
            continue
        q_min = Q_CTRL_MIN[idx]
        q_max = Q_CTRL_MAX[idx]
        max_q = max([np.max(q) for q in qs])
        min_q = min([np.min(q) for q in qs])
        max_qc = max([np.max(qc) for qc in qcs])
        min_qc = min([np.min(qc) for qc in qcs])
        max_taus = [np.max(np.abs(rec['states_q_tau'][:, idx])) for rec in recs if 'states_q_tau' in rec]
        max_tau = max(max_taus + [120])
        max_qd = max([np.max(np.abs(rec['states_qd'][:, idx])) for rec in recs])
        print(min_q, max_q, min_qc, max_qc, max_tau, max_qd)
        min_q = min(min_q, min_qc)
        max_q = max(max_q, max_qc)
        # max_qd = np.max(qd)
        # min_qd = np.min(qd)
        q_range = max_q - min_q
        q_max_range = q_max - q_min
        if auto_lim and q_range < 0.5 * q_max_range:
            # q_min_lim = q_min * 0.8 + q_max * 0.2
            # q_max_lim = q_max * 0.8 + q_min * 0.2
            qrng = max_q - min_q
            q_min_lim = min_q - 0.1 * qrng
            q_max_lim = max_q + 0.1 * qrng
            tau_max = max_tau * 1.1
            qd_limit = max_qd * 1.1
        else:
            qd_scale = 0.8
            q_min_lim = q_min - 0.1
            q_max_lim = q_max + 0.1
            ax2.axhline(y=q_max, color='k', linestyle='--')
            ax2.axhline(y=q_min, color='k', linestyle='--')
            ax2.axvline(x=q_max, color='k', linestyle='--')
            ax2.axvline(x=q_min, color='k', linestyle='--')
            ax3.axhline(y=q_max, color='k', linestyle='--')
            ax3.axhline(y=q_min, color='k', linestyle='--')
            tau_max = TAU_LIMIT[idx]
            qd_limit = QD_LIMIT[idx] * qd_scale
        ax1.set_title('qd - q')
        ax2.set_title('q - q_ctrl')
        ax3.set_title('q - t')
        ax4.set_title('q_tau - t')
        ax1.set_xlim([q_min_lim, q_max_lim])
        ax1.set_ylim([-qd_limit, qd_limit])
        ax2.set_xlim([q_min_lim, q_max_lim])
        ax2.set_ylim([q_min_lim, q_max_lim])
        ax3.set_ylim([q_min_lim, q_max_lim])
        ax4.set_ylim([-tau_max, tau_max])
        fig.tight_layout()
        ax1.legend([
            'ref',
        ] + rec_names, loc="lower right")
        if not single_plot:
            plt.suptitle(DOF_NAMES[idx])
            plt.savefig(plot_prefix + f'.{ext}')
            plt.close()
        if save_anim:
            # frame_step = int(1 // dt)
            frame_step = 1
            intv = frame_step * dt * 1000
            num_frames = (ed - st) // frame_step
            lns = [a[0] for a in axlines]
            blit = True

            # print('num_frames', num_frames)

            def animate(i):
                pt = i * frame_step
                for line, x, y in axlines:
                    line.set_xdata(x[:pt])
                    line.set_ydata(y[:pt])
                return lns

            ani = animation.FuncAnimation(fig, animate, interval=intv, blit=blit, frames=num_frames)
            kwds = {
                'html': dict(writer='html'),
                'apng': dict(writer='pillow'),
                # 'gif': dict(writer='pillow'),
                'gif': dict(writer='ffmpeg'),
            }
            ani.save(plot_prefix + f'.{aniext}', **kwds.get(aniext, {}))

    if do_plot and single_plot:
        plt.title('dofs')
        plt.savefig(plot_prefix + f'dofs.{ext}')
        plt.close()


if __name__ == '__main__':
    plot()
