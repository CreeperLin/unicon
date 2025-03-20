import os
import time
import numpy as np


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-ca', '--cpu_affinity', default=None)
    parser.add_argument('-c', '--close', action='store_true')
    parser.add_argument('-m', '--mode', action='append')
    parser.add_argument('-dt', '--dt', type=float, default=0.02)
    parser.add_argument('-n', '--num_steps', type=int, default=1024)
    parser.add_argument('-s', '--system', default='f')
    parser.add_argument('-ro', '--rec_output', default=None)
    parser.add_argument('-I', '--infer_type', default='gr1')
    parser.add_argument('-i', '--input_type', default='js')
    parser.add_argument('-p', '--policy_type', default='none')
    parser.add_argument('-cmd', '--cmd_type', default='vel')
    parser.add_argument('-ccv', '--cmd_const_v', default=None)
    parser.add_argument('-cir', '--cmd_init_range', default=None)
    parser.add_argument('-d', '--dry', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-nowrp', '--no_wrap', action='store_true')
    parser.add_argument('-w', '--wait', type=float, default=0)
    parser.add_argument('-kpr', '--kp_ratio', type=float, default=1.0)
    parser.add_argument('-kdr', '--kd_ratio', type=float, default=1.0)
    parser.add_argument('-kpd', '--kpd_ratio', type=float, default=1.0)
    parser.add_argument('-spr', '--sample_r', type=str, default=None)
    parser.add_argument('-spd', '--sample_dofs', default=None)
    parser.add_argument('-spl', '--sample_lerp', type=int, default=0)
    parser.add_argument('-spw', '--sample_wait', type=int, default=100)
    parser.add_argument('-spn', '--num_samples', type=int, default=None)
    parser.add_argument('-sprp', '--sample_repeats', type=int, default=None)
    parser.add_argument('-sprg', '--sample_repeat_group', type=int, default=2)
    parser.add_argument('-rps', '--replay_scale', type=float, default=None)
    parser.add_argument('-rpr', '--replay_repeats', type=int, default=None)
    parser.add_argument('-rppt', '--replay_pt', type=int, default=None)
    parser.add_argument('-rplen', '--replay_len', type=int, default=None)
    parser.add_argument('-rpl', '--replay_loop', action='store_true')
    parser.add_argument('-rni', '--replay_no_interp', action='store_true')
    parser.add_argument('-rpsk', '--replay_states_key', default='q_ctrl')
    parser.add_argument('-rpfk', '--replay_frame_key', default='states_q_ctrl')
    parser.add_argument('-qcdm', '--q_ctrl_default_mask', action='store_true')
    parser.add_argument('-qcm', '--q_ctrl_mask', default=None)
    parser.add_argument('-qcc', '--q_ctrl_clr', type=float, default=0)
    parser.add_argument('-iv', '--infer_verify', action='store_true')
    parser.add_argument('-imp', '--infer_model_path', default=None)
    parser.add_argument('-ikwds', '--infer_kwargs', default=None)
    parser.add_argument('-sfuq', '--use_qdd', action='store_true')
    parser.add_argument('-wi', '--wait_input', action='store_true')
    parser.add_argument('-rp', '--rec_path', default=None)
    parser.add_argument('-rap', '--action_path', default='actions_all.pt')
    parser.add_argument('-rop', '--obs_path', default='obs_all.pt')
    parser.add_argument('-ecp', '--env_cfg_path', default=None)
    parser.add_argument('-eco', '--env_cfg_override', default=None)
    parser.add_argument('-ssc', '--sims_config', default='ig.yaml')
    parser.add_argument('-sso', '--sims_override', default=None)
    parser.add_argument('-ssnr', '--sims_no_reset', action='store_true')
    parser.add_argument('-sswc', '--sims_wrapper_config', action='append')
    parser.add_argument('-ssd2', '--sims_dof_names_2', action='store_true')
    parser.add_argument('-ssh', '--sims_headless', action='store_true')
    parser.add_argument('-ssfb', '--sims_fixed_base', action='store_true')
    parser.add_argument('-sspd', '--sims_pd', action='store_true')
    parser.add_argument('-ssuk', '--sims_use_kpd', action='store_true')
    parser.add_argument('-skwds', '--system_kwargs', default=None)
    parser.add_argument('-cqc', '--clip_q_ctrl', action='store_true')
    parser.add_argument('-sfts', '--safety_states', action='store_true')
    parser.add_argument('-sftc', '--safety_ctrl', action='store_true')
    parser.add_argument('-dofs', '--dofs', default=None)
    parser.add_argument('-lrpt', '--lerp_time', type=float, default=5)
    parser.add_argument('-rv', '--run_viz', action='store_true')
    parser.add_argument('-shm', '--shm', action='store_true')
    parser.add_argument('-rt', '--robot_type', default='gr1t2')
    parser.add_argument('-vr', '--verify_recv', action='store_true')
    parser.add_argument('-mkln', '--mkl_n_thread', type=int, default=2)
    parser.add_argument('-ff', '--fast', action='store_true')
    parser.add_argument('-sqm', '--safety_q_margin', type=float, default=0.)
    parser.add_argument('-iis', '--inner_input_stop', action='store_true')
    parser.add_argument('-cms', '--cmd_max_steps', type=int, default=None)
    parser.add_argument('-sd', '--seed', type=int, default=None)
    parser.add_argument('-inp', '--infer_no_profile', action='store_true')
    parser.add_argument('-id', '--infer_device', default='cpu')
    parser.add_argument('-fl', '--fixed_lat', type=float, default=None)
    parser.add_argument('-fw', '--fixed_wait', type=float, default=None)
    parser.add_argument('-lsb', '--loop_sleep_block', action='store_true')
    parser.add_argument('-ldo', '--loop_dt_ofs', type=float, default=None)
    parser.add_argument('-ldt', '--loop_dt', type=float, default=None)
    parser.add_argument('-sqe', '--states_q_extras', action='store_true')
    parser.add_argument('-sxe', '--states_x_extras', action='store_true')
    parser.add_argument('-tc', '--tau_ctrl', action='store_true')
    parser.add_argument('-cct', '--cb_ctrl_tau', default=None)
    parser.add_argument('-cdt', '--ctrl_dt', type=float, default=0.02)
    parser.add_argument('-ilr', '--infer_load_run', default=None)
    parser.add_argument('-rcps', '--rec_post_send', action='store_true')
    parser.add_argument('-ofg', '--output_foxglove', action='store_true')
    parser.add_argument('-o', '--outputs', default=[], action='append')
    parser.add_argument('-itf', '--imu_transform', action='store_true')
    parser.add_argument('-sic', '--safety_integrity_check', action='store_true')
    parser.add_argument('-uspd', '--use_sim_pd', action='store_true')
    parser.add_argument('-dn2', '--dof_names_2', action='store_true')
    parser.add_argument('-dnr', '--dof_names_remap', default=None)
    parser.add_argument('-dns', '--dof_names_sub', default=None)
    parser.add_argument('-dp', '--data_path', default=None)
    parser.add_argument('-dkwds', '--data_kwds', default=None)
    parser.add_argument('-ncmd', '--num_commands', type=int, default=3)
    parser.add_argument('-nice', '--nice', type=int, default=None)
    parser.add_argument('-su', '--sudo', action='store_true')
    parser.add_argument('-qtf', '--q_transform', default=None)
    parser.add_argument('-sie', '--states_infer_extras', action='store_true')
    args, _ = parser.parse_known_args()
    return args


def run(args=None):
    if args is None:
        args = get_args()
    try:
        import isaacgym
        del isaacgym
    except ImportError:
        pass
    import yaml
    from unicon.states import states_init, states_news, states_new, states_get, states_destroy
    from unicon.general import cb_chain, cb_loop, cb_noop, cb_print, cb_prod, cb_zip, \
        cb_timeout, cb_fixed_lat, cb_wait_input, cb_replay
    from unicon.utils import set_nice2, set_cpu_affinity2, sampler_uniform, list2slice, pp_arr, set_seed, \
        import_obj, parse_robot_def
    from unicon.ctrl import cb_ctrl_q_from_target_lerp

    if args.sudo:
        os.system('sudo -v')

    nice = args.nice
    if nice is not None:
        set_nice2(nice)

    affinity = args.cpu_affinity
    if affinity is not None:
        set_cpu_affinity2(affinity)

    wait = args.wait
    if wait:
        print('wait', wait)
        time.sleep(wait)

    argv = list(__import__('sys').argv)
    verbose = args.verbose

    mkl_n_thread = args.mkl_n_thread
    print('mkl_n_thread', mkl_n_thread)
    if mkl_n_thread:
        os.environ['MKL_NUM_THREADS'] = str(mkl_n_thread)

    if args.seed is not None:
        set_seed(args.seed)

    robot_type = args.robot_type
    robot_def = import_obj((robot_type, None), default_mod_prefix='unicon.defs')
    robot_def = parse_robot_def(robot_def)
    KP = robot_def.get('KP', None)
    KD = robot_def.get('KD', None)
    Q_CTRL_MIN = robot_def.get('Q_CTRL_MIN', None)
    Q_CTRL_MAX = robot_def.get('Q_CTRL_MAX', None)
    DOF_NAMES = robot_def.get('DOF_NAMES', None)
    DOF_PRESETS = robot_def.get('DOF_PRESETS', None)
    NUM_DOFS = robot_def.get('NUM_DOFS', None)
    DOF_MAPS = robot_def.get('DOF_MAPS', None)
    TAU_LIMIT = robot_def.get('TAU_LIMIT', None)
    QD_LIMIT = robot_def.get('QD_LIMIT', None)
    Q_BOOT = robot_def.get('Q_BOOT', None)
    Q_RESET = robot_def.get('Q_RESET', None)
    DOF_NAMES_2 = robot_def.get('DOF_NAMES_2', None)

    q_reset = np.zeros(NUM_DOFS) if Q_RESET is None else Q_RESET
    q_boot = q_reset if Q_BOOT is None else Q_BOOT

    if args.dof_names_2:
        DOF_NAMES = DOF_NAMES_2

    saved_info = {}

    tau_ctrl = args.tau_ctrl
    specs = [
        # ('prop', 3 + 3 + 4 + NUM_DOFS * 2),
        ('rpy', 3),
        ('ang_vel', 3),
        ('quat', 4),
        ('q', NUM_DOFS),
        ('qd', NUM_DOFS),
        ('q_ctrl', NUM_DOFS),
    ]
    if tau_ctrl:
        specs.append(('tau_ctrl', NUM_DOFS))
    q_extras = args.states_q_extras
    if q_extras:
        specs.extend([
            ('q_tau', NUM_DOFS),
            ('q_cur', NUM_DOFS),
        ])
    x_extras = args.states_x_extras
    if x_extras:
        specs.extend([
            ('pos', 3),
            ('lin_vel', 3),
            ('lin_acc', 3),
        ])
    states_news(specs)
    states_new('q_target', NUM_DOFS)
    input_keys = import_obj('unicon.inputs:_default_input_keys')
    num_inputs = len(input_keys)
    states_new('input', num_inputs)
    num_commands = args.num_commands
    states_new('cmd', num_commands)
    use_shm = False
    use_shm = args.shm
    reuse = True
    states_init(use_shm=use_shm, reuse=reuse, clear=True)

    proc = None
    if args.run_viz:
        viz_kwds = {
            'robot_type': robot_type,
        }
        from unicon.viz import run_viz
        import multiprocessing
        ctx = multiprocessing.get_context('spawn')
        proc = ctx.Process(target=run_viz, kwargs=viz_kwds, daemon=True)
        proc.start()

    states_q_ctrl = states_get('q_ctrl')
    states_tau_ctrl = states_get('tau_ctrl')
    states_ctrls = {
        'states_q_ctrl': states_q_ctrl,
        'states_tau_ctrl': states_tau_ctrl,
    }
    states_ctrls = {k: v for k, v in states_ctrls.items() if v is not None}
    states_q_target = states_get('q_target')
    states_props = {
        'states_rpy': states_get('rpy'),
        'states_ang_vel': states_get('ang_vel'),
        'states_quat': states_get('quat'),
        'states_q': states_get('q'),
        'states_qd': states_get('qd'),
    }
    states_q = states_props['states_q']
    states_qd = states_props['states_qd']
    states_input = states_get('input')
    states_cmd = states_get('cmd')
    states_q_tau = states_get('q_tau')
    states_q_cur = states_get('q_cur')
    states_pos = states_get('pos')
    states_lin_vel = states_get('lin_vel')
    states_lin_acc = states_get('lin_acc')
    states_extras = {
        'states_cmd': states_cmd,
        'states_q_tau': states_q_tau,
        'states_q_cur': states_q_cur,
        'states_pos': states_pos,
        'states_lin_vel': states_lin_vel,
        'states_lin_acc': states_lin_acc,
    }
    states_extras = {k: v for k, v in states_extras.items() if v is not None}

    dt = args.dt
    ctrl_dt = args.ctrl_dt
    ctrl_dt = dt if ctrl_dt is None else ctrl_dt

    dof_names = None
    default_dof_pos = None

    actions_path = args.action_path
    obs_path = args.obs_path
    env_cfg_path = args.env_cfg_path
    infer_load_run = args.infer_load_run
    infer_model_path = args.infer_model_path
    if infer_load_run is not None:
        _default_infer_root = f'{os.environ["HOME"]}/GitRepo/GR1/logs/'
        _default_infer_root = os.environ.get('UNICON_INFER_ROOT', _default_infer_root)
        root = os.path.join(_default_infer_root, infer_load_run)
        model_file_pat = 'policy_1.pt'
        model_file = None
        for r, _, fs in os.walk(root):
            for f in fs:
                if model_file_pat in f:
                    model_file = os.path.join(r, f)
                    break
        if model_file is None:
            raise ValueError('model_file not found')
        infer_model_path = os.path.join(root, model_file)
        from unicon.utils import md5sum
        infer_model_md5sum = md5sum(infer_model_path)
        saved_info['infer_model_md5sum'] = infer_model_md5sum
        print('infer_load_run', infer_load_run, infer_model_path, infer_model_md5sum)
    if env_cfg_path is None and infer_model_path is not None:
        for ext in ['.json', '.yaml']:
            env_cfg_path = os.path.join(os.path.dirname(infer_model_path), 'env_cfg' + ext)
            if os.path.exists(env_cfg_path):
                break

    env_cfg = None
    env_cfg_args = None
    if env_cfg_path is not None and os.path.exists(env_cfg_path):
        print('env_cfg_path', env_cfg_path)
        yaml.add_multi_constructor('tag:', lambda *_: None, Loader=yaml.SafeLoader)
        with open(env_cfg_path, 'r') as f:
            env_cfg = yaml.safe_load(f)
    if env_cfg is not None:
        env_cfg_override = args.env_cfg_override
        if env_cfg_override is not None:
            env_cfg_override = yaml.safe_load(env_cfg_override)
            from unicon.utils import obj_update
            obj_update(env_cfg, env_cfg_override)
        env_cfg_env = env_cfg['env']
        num_acts = env_cfg_env['num_actions']
        num_obs = env_cfg_env.get('num_partial_obs', env_cfg_env.get('num_observations'))
        init_joint_angles = env_cfg['init_state']['default_joint_angles']
        dof_names = env_cfg.get('dof_names', list(init_joint_angles.keys()))
        default_dof_pos = np.array([init_joint_angles[n] for n in dof_names])
        env_cfg_args = env_cfg.get('cmd_args', env_cfg.get('argv', None))
        print('env_cfg_args', env_cfg_args)

    states_infer_extras = {}
    inf_extras = args.states_infer_extras
    if inf_extras and env_cfg is not None:
        print('num_acts', num_acts, 'num_obs', num_obs)
        states_infer_acts = np.zeros(num_acts)
        states_infer_obs = np.zeros(num_obs)
        states_infer_extras = {
            'states_infer_acts': states_infer_acts,
            'states_infer_obs': states_infer_obs,
        }

    dofs = args.dofs
    if dofs is not None:
        dofs = yaml.safe_load(dofs)
        if isinstance(dofs, str):
            dof_names = DOF_PRESETS[dofs]
        elif isinstance(dofs, (list, tuple)):
            dof_names = dofs
        else:
            raise ValueError(f'invalid dofs {dofs}')

    dof_names = DOF_NAMES if dof_names is None else dof_names

    print('dof_names', len(dof_names), dof_names)

    dof_names_sub = args.dof_names_sub
    if dof_names_sub is not None:
        sub = yaml.safe_load(dof_names_sub)

        def sub_name(x):
            for t1, t2 in sub:
                x = x.replace(t1, t2)
            return x

        dof_names = [sub_name(n) for n in dof_names]

    dof_names_remap = args.dof_names_remap
    if dof_names_remap is not None:
        remap = yaml.safe_load(dof_names_remap)
        dof_names = [remap.get(n, n) for n in dof_names]

    print('DOF_NAMES', len(DOF_NAMES), DOF_NAMES)
    print('dof_names', len(dof_names), dof_names)
    dof_names_extra = [n for n in dof_names if n not in DOF_NAMES]
    dof_states_padded = len(dof_names_extra) > 0
    dof_src_map = [i for i, n in enumerate(dof_names) if n in DOF_NAMES]
    if dof_states_padded:
        dof_map = [DOF_NAMES.index(n) for n in dof_names if n in DOF_NAMES]
        dof_map_padded = [(DOF_NAMES.index(n) if n in DOF_NAMES else -1) for n in dof_names]
        print('dof_states_padded', dof_states_padded)
        print('dof_names_extra', len(dof_names_extra), dof_names_extra)
        print('dof_map_padded', dof_map_padded)
    else:
        dof_map = [DOF_NAMES.index(n) for n in dof_names]
        dof_map_padded = dof_map
    num_dofs = len(dof_map)
    assert num_dofs > 0

    if default_dof_pos is not None:
        q_reset[dof_map] = default_dof_pos[dof_src_map]
    default_dof_pos = q_reset[dof_map] if default_dof_pos is None else default_dof_pos

    _dof_map = dof_map
    dof_map = list2slice(dof_map)
    dof_src_map = list2slice(dof_src_map)
    print('num_dofs', num_dofs)
    print('dof_map', dof_map)
    print('dof_src_map', dof_src_map)
    print('default_dof_pos', default_dof_pos.tolist())
    print('q_reset', q_reset.tolist())

    if dof_states_padded:
        dtype = states_q_ctrl.dtype
        states_q_ctrl_inf = np.zeros(NUM_DOFS + 1, dtype=dtype)
        states_q_inf = np.zeros(NUM_DOFS + 1, dtype=dtype)
        states_qd_inf = np.zeros(NUM_DOFS + 1, dtype=dtype)

        def cb_pad_in():
            states_q_inf[:NUM_DOFS] = states_q
            states_qd_inf[:NUM_DOFS] = states_qd
            # print('states_q_inf', states_q_inf)

        def cb_pad_out():
            states_q_ctrl[:] = states_q_ctrl_inf[:NUM_DOFS]
            # print('states_q_ctrl_inf', states_q_ctrl_inf)

        states_props_inf = states_props.copy()
        states_props_inf['states_q'] = states_q_inf
        states_props_inf['states_qd'] = states_qd_inf
        states_ctrls_inf = states_ctrls.copy()
        states_ctrls_inf['states_q_ctrl'] = states_q_ctrl_inf
    else:
        states_props_inf = states_props
        states_ctrls_inf = states_ctrls
        states_q_ctrl_inf = states_q_ctrl

    replay_loop = args.replay_loop
    rec_path = args.rec_path
    actions_all = None
    rec_q_ctrl = None
    rec_q = None
    loaded_rec = None
    if rec_path is not None:
        print('rec_path', rec_path)
        loaded_rec = np.load(rec_path, allow_pickle=True).item()
    data_path = args.data_path
    if data_path is not None:
        _, ext = os.path.splitext(data_path)
        data_type = ext[1:]
        print('data_type', data_type)
        mod = import_obj((data_type, None), default_mod_prefix='unicon.data')
        print('data_path', data_path)
        kwds = args.data_kwds
        loaded_rec = mod.load(
            data_path,
            robot_def=robot_def,
            **(kwds or {}),
        )
    if loaded_rec is not None:
        rec_type = loaded_rec.get('type')
        if rec_type == 'legged':
            rec_states = loaded_rec['states'][0]
            rec_actions = loaded_rec['actions'][0]
            num_dofs = (rec_states.shape[-1] - 13 - 6) // 2
            rec_q = rec_states[:, 13:13 + num_dofs]
            actions_all = rec_actions
        else:
            rec_q_ctrl = loaded_rec.get('states_q_ctrl')
            rec_q = loaded_rec.get('states_q')
            print('rec_q_ctrl', rec_q_ctrl.shape)
        loaded_rec_args = loaded_rec.get('args', {})
        print('loaded_rec_args', loaded_rec_args)

    seq = []
    mode = args.mode or []
    print('mode', mode)
    for m in mode:
        modes = {k: False for k in ['noop', 'const', 'sample', 'replay', 'infer', 'teleop', 'play']}
        m = {k[0]: k for k in modes.keys()}[m]
        modes[m] = True
        if modes['noop']:
            cb = cb_noop()
        elif modes['const']:
            q_ctrl_const = np.zeros(NUM_DOFS)
            q_ctrl_const[:] = q_reset

            def cb_const():
                states_q_ctrl[:] = q_ctrl_const

            cb = cb_const
        elif modes['sample']:
            states_q_target[:] = q_reset
            smpl_dof_map = args.sample_dofs
            if smpl_dof_map is not None:
                smpl_dof_map = yaml.safe_load(smpl_dof_map)
            if isinstance(smpl_dof_map, int):
                smpl_dof_map = [int(smpl_dof_map)]
            elif isinstance(smpl_dof_map, str):
                smpl_dof_map = DOF_MAPS[smpl_dof_map]
            smpl_r = 0.3 if args.sample_r is None else yaml.safe_load(args.sample_r)
            if isinstance(smpl_r, float):
                print('mean', Q_CTRL_MAX + Q_CTRL_MIN)
                print('range', Q_CTRL_MAX - Q_CTRL_MIN)
                # q_smpl_min = Q_CTRL_MIN + (Q_CTRL_MAX - Q_CTRL_MIN) * (0.5 - smpl_r)
                # q_smpl_max = Q_CTRL_MIN + (Q_CTRL_MAX - Q_CTRL_MIN) * (0.5 + smpl_r)
                q_smpl_min = q_reset + (q_reset - Q_CTRL_MIN) * (-smpl_r)
                q_smpl_max = q_reset + (Q_CTRL_MAX - q_reset) * (+smpl_r)
                q_smpl_min = q_smpl_min[smpl_dof_map]
                q_smpl_max = q_smpl_max[smpl_dof_map]
            else:
                rng_l, rng_r = smpl_r
                print('rng_l', rng_l)
                print('rng_r', rng_r)
                q_min = Q_CTRL_MIN[smpl_dof_map]
                q_max = Q_CTRL_MAX[smpl_dof_map]
                q_rng = q_max - q_min
                q_smpl_min = q_min + q_rng * rng_l
                q_smpl_max = q_min + q_rng * rng_r
            num_samples = args.num_samples
            print('num_samples', num_samples)
            frames = sampler_uniform(low=q_smpl_min, high=q_smpl_max, num_samples=num_samples)
            frames = np.array(list(frames))
            sample_repeats = args.sample_repeats
            if sample_repeats is not None:
                group = args.sample_repeat_group
                print('sample_repeats', group)
                sz = frames.shape[-1]
                frames = np.repeat(frames.reshape(-1, group, sz), sample_repeats, axis=0).reshape(-1, sz)
            print('sample frames', frames.shape)
            # smpl_lerp_steps = lerp_steps
            smpl_lerp_steps = args.sample_lerp
            # smpl_wait_steps = 0
            smpl_wait_steps = args.sample_wait
            print('smpl_r', smpl_r)
            print('smpl_lerp_steps', smpl_lerp_steps)
            print('smpl_wait_steps', smpl_wait_steps)
            print('smpl_dof_map', smpl_dof_map)
            print('q_smpl_min', q_smpl_min)
            print('q_smpl_max', q_smpl_max)
            if smpl_dof_map is not None:
                print('smpl_dof_map', [DOF_NAMES[i] for i in smpl_dof_map])
            if smpl_lerp_steps:
                cb_lerp = cb_ctrl_q_from_target_lerp(
                    **states_ctrls,
                    states_q_target=states_q_target,
                    states_q=states_q,
                    max_steps=smpl_lerp_steps,
                    dof_map=smpl_dof_map,
                )
                cb = cb_replay(
                    states_q_target,
                    frames=frames,
                    inds=smpl_dof_map,
                    loop=replay_loop,
                )
                cb = cb_prod(cb, cb_lerp)
            elif smpl_wait_steps:
                cb = cb_replay(
                    states_q_ctrl,
                    frames=frames,
                    inds=smpl_dof_map,
                    repeats=smpl_wait_steps,
                    loop=replay_loop,
                )
            else:
                cb = cb_replay(
                    states_q_ctrl,
                    frames=frames,
                    inds=smpl_dof_map,
                    loop=replay_loop,
                )
        elif modes['teleop']:
            from unicon.teleop import cb_teleop_q
            cb = cb_teleop_q(
                **states_ctrls,
                states_q=states_q,
                states_input=states_input,
            )
        elif modes['replay']:
            rep_dt = 0.02
            replay_states_key = args.replay_states_key
            states_dest = states_get(replay_states_key)
            if loaded_rec is None:
                import torch
                if actions_all is not None:
                    actions_all = torch.from_numpy(actions_all)
                elif actions_path.endswith('.npy'):
                    r = np.load(actions_path, allow_pickle=True).item()
                    actions_all = r['actions']
                    actions_all = actions_all.squeeze()
                    actions_all = torch.from_numpy(actions_all)
                elif actions_path.endswith('.pt'):
                    actions_all = torch.load(actions_path, map_location='cpu').squeeze()
                print('actions_all', actions_all.shape)
                retained_action_inds = env_cfg.get('retained_action_inds')
                if retained_action_inds is not None:
                    _dof_map = list(range(dof_map.start, dof_map.stop)) if isinstance(dof_map, slice) else dof_map
                    print('retained_action_inds', retained_action_inds)
                    dof_map_action = [_dof_map[i] for i in retained_action_inds]
                    print('dof_map_action', dof_map_action)
                    _actions = torch.zeros(len(actions_all), num_dofs)
                    _actions[:, dof_map_action] = actions_all
                    actions_all = _actions
                assert actions_all.shape[1] == num_dofs
                clip_actions = env_cfg['normalization']['clip_actions']
                action_scale = env_cfg['control']['action_scale']
                print('clip_actions', clip_actions)
                print('action_scale', action_scale)
                actions_all = torch.clip(actions_all, -clip_actions, clip_actions)
                actions_all = actions_all * action_scale + torch.from_numpy(default_dof_pos).to(dtype=torch.float32)
                actions_all = actions_all.numpy()
                frames = actions_all
            else:
                replay_frame_key = args.replay_frame_key
                rec_frames = loaded_rec.get(replay_frame_key)
                frames = rec_frames[:, dof_map]
                rep_dt = loaded_rec_args.get('ctrl_dt', loaded_rec_args.get('dt', ctrl_dt))
            print('frames min', np.min(frames, axis=0))
            print('frames max', np.max(frames, axis=0))
            print('frames rng', np.max(frames, axis=0) - np.min(frames, axis=0))
            scale = args.replay_scale
            if scale is not None:
                frames *= scale
            pt = args.replay_pt
            if pt is not None:
                frames = frames[pt:]
            replay_len = args.replay_len
            if replay_len is not None:
                frames = frames[:replay_len]
            replay_interp = not args.replay_no_interp
            if replay_interp and rep_dt > ctrl_dt:
                num_frames = len(frames) - 1
                ratio = rep_dt / ctrl_dt
                num_frames_new = int(num_frames * ratio)
                inds = num_frames * np.linspace(0, 1, num_frames_new)[:-1]
                inds_i = inds.astype(np.int32)
                inds_f = (inds - inds_i).reshape(-1, 1)
                new_frames = frames[inds_i] * (1 - inds_f) + frames[inds_i + 1] * inds_f
                print('replay_interp', ratio, frames.shape, new_frames.shape)
                frames = new_frames
            repeats = args.replay_repeats
            repeats = max(rep_dt // ctrl_dt, 1) if repeats is None else repeats
            print('repeats', repeats)
            cb = cb_replay(
                states_dest,
                frames=frames,
                inds=dof_map,
                repeats=repeats,
                loop=replay_loop,
            )
        elif modes['infer']:
            infer_type = args.infer_type
            print('infer_type', infer_type)
            cb_infer_cls = import_obj(infer_type, default_name_prefix='cb_infer', default_mod_prefix='unicon.infer')
            import torch
            from unicon.utils.torch import torch_load_jit, torch_no_grad, torch_no_profiling
            torch_no_grad()
            torch_no_profiling()

            infer_device = args.infer_device
            print('infer_model_path', infer_model_path)
            policy = torch_load_jit(infer_model_path, device=infer_device)

            policy_type = args.policy_type
            if policy_type == 'gr1':
                policy_reset_fn = policy.reset_memory
                policy_fn = lambda obs: policy(obs)[0]
            else:
                policy_reset_fn = None
                policy_fn = policy

            infer_kwds = yaml.safe_load(args.infer_kwargs or '') or {}
            cb_infer, reset_fn = cb_infer_cls(
                **states_props_inf,
                # states_q_ctrl=states_q_ctrl,
                states_q_ctrl=states_q_ctrl_inf,
                **states_infer_extras,
                states_cmd=states_cmd,
                policy_fn=policy_fn,
                policy_reset_fn=policy_reset_fn,
                # dof_map=dof_map,
                dof_map=dof_map_padded,
                default_dof_pos=default_dof_pos,
                env_cfg=env_cfg,
                device=infer_device,
                **infer_kwds,
            )

            verify = args.infer_verify
            if verify:
                actions_all = torch.load(actions_path, map_location='cpu').squeeze().numpy()
                print('actions_all', actions_all.shape)
                obs_all = torch.load(obs_path, map_location='cpu').squeeze().numpy()
                print('obs_all', obs_all.shape)
                actions_pred = []
                # policy_reset_fn()
                reset_fn()
                for obs in obs_all:
                    actions = policy(torch.from_numpy(obs).view(1, -1))
                    actions_pred.append(actions)
                actions_pred = torch.stack(actions_pred, dim=1).squeeze()
                print(actions_pred.shape)
                # print(actions_pred[0] - actions_all[0])
                print('min', torch.min(actions_pred, dim=0)[0])
                print('max', torch.max(actions_pred, dim=0)[0])
                err = (actions_pred - actions_all).norm(dim=-1).mean()
                print('err', err)
                # return
            reset_fn()
            infer_profile = not args.infer_no_profile
            if infer_profile:
                import timeit
                # n = 2**11
                n = 2**12
                t = timeit.timeit(cb_infer, number=n)
                print('infer t', t / n)
                ts = []
                for _ in range(n):
                    t0 = time.monotonic()
                    cb_infer()
                    t1 = time.monotonic()
                    ts.append(t1 - t0)
                ts = np.array(ts)
                print('infer min/max/mean/std', np.min(ts), np.max(ts), np.mean(ts), np.std(ts))
                print('q_ctrl', pp_arr(states_q_ctrl))
            reset_fn()
            cb = cb_infer
            # return
        elif modes['play']:
            from unicon.general import cb_replay
            cb = cb_replay(
                **states_ctrls,
                states_input=states_input,
                **states_props,
                **states_extras,
                frames=loaded_rec,
                inds=dof_map,
            )
        seq.append(cb)

    if dof_states_padded:
        seq.insert(0, cb_pad_in)
        seq.insert(-1, cb_pad_out)

    q_ctrl_default_mask = args.q_ctrl_default_mask
    q_ctrl_mask = args.q_ctrl_mask
    if q_ctrl_mask is not None:
        ctrl_mask = np.zeros(len(states_q_ctrl), dtype=bool)
        import yaml
        q_ctrl_mask = yaml.safe_load(q_ctrl_mask)
        q_ctrl_mask = [q_ctrl_mask] if not isinstance(q_ctrl_mask, (tuple, list)) else q_ctrl_mask
        for mask in q_ctrl_mask:
            if isinstance(mask, str):
                # inds = DOF_MAPS[mask]
                idx = DOF_NAMES.index(mask)
            else:
                idx = mask
            ctrl_mask[idx] = 1
        print('q_ctrl_mask', q_ctrl_default_mask, ctrl_mask.astype(int).tolist())
        default_mask = None
        if q_ctrl_default_mask:
            default_mask = ctrl_mask
            ctrl_mask = None
        from unicon.ctrl import cb_ctrl_q_mask
        cb = cb_ctrl_q_mask(
            states_q_ctrl,
            ctrl_mask=ctrl_mask,
            default_mask=default_mask,
            default_q_ctrl=q_reset,
        )
        seq.append(cb)

    q_ctrl_min = Q_CTRL_MIN
    q_ctrl_max = Q_CTRL_MAX
    env_clip_q_ctrl = False
    safety_q_margin = 0.
    if env_cfg is not None:
        env_clip_q_ctrl = env_cfg['control'].get('clip_q_ctrl', env_clip_q_ctrl)
        safety_q_margin = env_cfg['control'].get('safety_q_margin', safety_q_margin)
    safety_q_margin = args.safety_q_margin if args.safety_q_margin else safety_q_margin
    if Q_CTRL_MIN is not None:
        q_ctrl_min = Q_CTRL_MIN * (1 - safety_q_margin) + Q_CTRL_MAX * safety_q_margin
        q_ctrl_max = Q_CTRL_MAX * (1 - safety_q_margin) + Q_CTRL_MIN * safety_q_margin
    clip_q_ctrl = args.clip_q_ctrl or env_clip_q_ctrl
    print('clip_q_ctrl', clip_q_ctrl)
    print('safety_q_margin', safety_q_margin)
    print('q_ctrl_min', q_ctrl_min)
    print('q_ctrl_max', q_ctrl_max)
    if clip_q_ctrl:

        def cb_ctrl_q_clip():
            states_q_ctrl[:] = np.clip(states_q_ctrl, q_ctrl_min, q_ctrl_max)

        seq.append(cb_ctrl_q_clip)

    rec_output = args.rec_output
    rec_post_send = args.rec_post_send
    if rec_output is not None:
        from unicon.general import cb_rec
        rec = {}
        _cb_rec = cb_rec(
            **states_ctrls,
            states_input=states_input,
            **states_props,
            **states_extras,
            **states_infer_extras,
            rec=rec,
        )
        if not rec_post_send:
            seq.append(_cb_rec)

    num_steps = args.num_steps
    if num_steps is not None and num_steps > 0:
        cb = cb_timeout(num_steps)
        seq.append(cb)

    inner_input_stop = args.inner_input_stop
    outer_stop_keys = ['BTN_TL', 'BTN_TR']
    if inner_input_stop:
        seq.append(cb_wait_input(states_input=states_input, keys=outer_stop_keys[:-1]))
        outer_stop_keys = outer_stop_keys[1:]

    print('inner seq', seq)
    cb = seq[0] if len(seq) == 1 else cb_zip(*seq)
    chain = [cb]

    wrapped = not args.no_wrap
    if wrapped:
        # if rec_q_ctrl is not None:
        # q_reset[:] = rec_q_ctrl[0]
        if rec_q is not None:
            q_reset[:] = rec_q[0]
        lerp_steps = int((args.lerp_time + 0.01) // dt)
        cb_lerp = cb_ctrl_q_from_target_lerp(
            **states_ctrls,
            states_q_target=q_reset,
            states_q=states_q,
            max_steps=lerp_steps,
            cycle=False,
            from_q_ctrl=False,
            t_fn='exp',
        )
        cb_lerp2 = cb_ctrl_q_from_target_lerp(
            **states_ctrls,
            states_q_target=q_reset,
            states_q=states_q,
            max_steps=max(1, lerp_steps // 4),
            cycle=False,
        )
        cb_lerp3 = cb_ctrl_q_from_target_lerp(
            **states_ctrls,
            states_q_target=q_boot,
            states_q=states_q,
            max_steps=lerp_steps,
            cycle=False,
        )
        cb_pre_ctrl = [
            # cb_loop(cb_print(), max_steps=1),
            cb_lerp,
            # cb_motor_q_lerp(max_steps=1e2, q_ctrl=q_reset),
        ]
        cb_post_ctrl = [
            cb_lerp2,
            cb_lerp3,
            # cb_motor_q_lerp(max_steps=lerp_steps, q_ctrl=q_boot),
        ]
        chain = cb_pre_ctrl + chain + cb_post_ctrl
    else:
        states_q_ctrl[:NUM_DOFS] = q_reset

    input_type = args.input_type
    input_type = None if input_type == 'none' else input_type
    wait_input = args.wait_input
    if wait_input:
        key = 'BTN_TL'
        cb = cb_wait_input(
            states_input=states_input,
            keys=[key],
            clicks=1,
            prompt=True,
        )
        chain.insert(min(len(chain) - 1, 1), cb)

    print('chain', chain)
    cb = cb_chain(*chain) if len(chain) > 1 else chain[0]

    fixed_lat = args.fixed_lat
    if fixed_lat:
        cb = cb_fixed_lat(cb, fixed_lat=fixed_lat)

    seq = []
    seq.append(cb)

    kp_r = args.kp_ratio
    kd_r = args.kd_ratio
    kp_r *= args.kpd_ratio
    kd_r *= args.kpd_ratio
    print('kp_r', kp_r)
    print('kd_r', kd_r)
    kp = KP
    kd = KD
    sim_kps = None
    sim_kds = None
    kpd_ctrl_only = True
    use_sim_pd = args.use_sim_pd
    if env_cfg is not None and use_sim_pd:
        sim_kps = env_cfg['control']['stiffness']
        sim_kds = env_cfg['control']['damping']
        sim_torque_limits = env_cfg['control'].get('torque_limits', None)
        print('sim_kps', sim_kps)
        print('sim_kds', sim_kds)
        print('sim_torque_limits', sim_torque_limits)
        if kpd_ctrl_only:
            sim_kps = {k: v * (kp_r if any([k in n for n in dof_names]) else 1.) for k, v in sim_kps.items()}
            sim_kds = {k: v * (kd_r if any([k in n for n in dof_names]) else 1.) for k, v in sim_kds.items()}
        else:
            sim_kps = {k: v * kp_r for k, v in sim_kps.items()}
            sim_kds = {k: v * kd_r for k, v in sim_kds.items()}
        kp = [([v for k, v in sim_kps.items() if k in n] or [kp[i]])[0] for i, n in enumerate(DOF_NAMES)]
        kd = [([v for k, v in sim_kds.items() if k in n] or [kd[i]])[0] for i, n in enumerate(DOF_NAMES)]
        assert len(kp) == len(kd) == NUM_DOFS
        kp = np.array(kp)
        kd = np.array(kd)
    elif kp is not None:
        kpd_inds = dof_map if kpd_ctrl_only else None
        kp = np.array(kp)
        kd = np.array(kd)
        kp[kpd_inds] *= kp_r
        kd[kpd_inds] *= kd_r
    if kp is not None:
        print('kp', pp_arr(kp))
        print('kd', pp_arr(kd))

    states_props_sys = states_props
    states_ctrls_sys = states_ctrls

    q_transform = args.q_transform
    if q_transform is not None:
        q_transform = yaml.safe_load(q_transform)
        # q_transform = [1, {'right_hip_roll_joint': -0.15,}]
        print('q_transform', q_transform)
        # sys q to states q
        if isinstance(q_transform, dict):
            w = q_transform.get('w')
            b = q_transform.get('b')
        elif isinstance(q_transform, (list, tuple)):
            w, b = q_transform
        wb = []
        for x, def_y in [(w, 1), (b, 0)]:
            if x is None:
                y = def_y
            elif isinstance(x, (list, tuple)):
                y = np.array(x)
            elif isinstance(x, dict):
                y = np.zeros(NUM_DOFS)
                y[:] = def_y
                for k, v in x.items():
                    y[DOF_NAMES.index(k)] = v
            else:
                y = x
            wb.append(y)

        qtf_w, qtf_b = wb
        qtf_w_inv = 1 / qtf_w
        print('qtf_w', qtf_w)
        print('qtf_b', qtf_b)

        dtype = states_q_ctrl.dtype
        states_q_ctrl_sys = np.zeros(NUM_DOFS, dtype=dtype)
        states_q_sys = np.zeros(NUM_DOFS, dtype=dtype)
        states_props_sys = states_props.copy()
        states_props_sys['states_q'] = states_q_sys
        states_ctrls_sys = states_ctrls.copy()
        states_ctrls_sys['states_q_ctrl'] = states_q_ctrl_sys

        def cb_qtf_recv():
            states_q[:] = states_q_sys * qtf_w + qtf_b
            # states_qd[:] = states_qd

        def cb_qtf_send():
            states_q_ctrl_sys[:] = (states_q_ctrl - qtf_b) * qtf_w_inv

    systems = {
        k: False for k in ['consys', 'mini', 'grx', 'fake', 'FFTAI', 'roslibpy', 'sims', 'a1', 'unitree', 'none']
    }
    system = args.system
    system = {k[0]: k for k in systems.keys()}.get(system, system)
    print('system', system)
    systems[system] = True

    fft_sys = any([systems[s] for s in ['consys', 'grx', 'FFTAI', 'mini']])
    if fft_sys:
        from unicon.utils.fftai import get_battery_level
        bat = get_battery_level()
        print('bat', bat)
        saved_info['battery'] = bat

    kp_c, kd_c = None, None
    if sim_kps is not None and robot_type == 'gr1t2' and fft_sys:
        from unicon.defs.gr1t2 import compute_gains
        kp_c, kd_c = compute_gains(sim_kps, sim_kds)
        assert len(kp_c) == len(kd_c) == NUM_DOFS

    sys_kwds = yaml.safe_load(args.system_kwargs or '') or {}
    sys_clip_q_ctrl = (not clip_q_ctrl)

    cb_recv, cb_send, cb_close = None, None, None
    if systems['none']:
        pass
    elif systems['fake']:
        from unicon.systems.fake import cb_fake_recv_send_close
        use_qdd = args.use_qdd
        cb_recv, cb_send, cb_close = cb_fake_recv_send_close(
            **states_props_sys,
            **states_ctrls_sys,
            q_min=q_ctrl_min,
            q_max=q_ctrl_max,
            kp=kp,
            kd=kd,
            use_qdd=use_qdd,
            inv_mass=1.0,
            **sys_kwds,
        )
    elif systems['mini']:
        from unicon.systems.mini import cb_mini_recv_send_close
        cb_recv, cb_send, cb_close = cb_mini_recv_send_close(
            **states_props_sys,
            **states_ctrls_sys,
            states_q_tau=states_q_tau,
            states_q_cur=states_q_cur,
            states_lin_vel=states_lin_vel,
            states_lin_acc=states_lin_acc,
            states_pos=states_pos,
            kp=kp,
            kd=kd,
            q_ctrl_min=q_ctrl_min,
            q_ctrl_max=q_ctrl_max,
            clip_q_ctrl=sys_clip_q_ctrl,
            robot_def=robot_def,
            **sys_kwds,
        )
    elif systems['FFTAI']:
        from unicon.systems.fftai import cb_fftai_recv_send_close
        cb_recv, cb_send, cb_close = cb_fftai_recv_send_close(
            **states_props_sys,
            **states_ctrls_sys,
            states_q_tau=states_q_tau,
            states_q_cur=states_q_cur,
            kp=kp_c,
            kd=kd_c,
            q_ctrl_min=q_ctrl_min,
            q_ctrl_max=q_ctrl_max,
            clip_q_ctrl=sys_clip_q_ctrl,
            dof_map=dof_map,
            robot_def=robot_def,
            **sys_kwds,
        )
    elif systems['roslibpy']:
        from unicon.systems.roslibpy import cb_roblibpy_recv_send_close
        cb_recv, cb_send, cb_close = cb_roblibpy_recv_send_close(
            **states_props_sys,
            **states_ctrls_sys,
            **sys_kwds,
        )
    elif systems['sims']:
        system_config = yaml.safe_load(open(args.sims_config, 'r'))
        sim_dof_names = DOF_NAMES
        if args.sims_dof_names_2:
            sim_dof_names = robot_def.get('DOF_NAMES_2')
        system_config['realtime'] = False
        fix_base_link = system_config.get('fix_base_link', False) or args.sims_fixed_base
        fix_base_link = fix_base_link or modes['sample']
        system_config['fix_base_link'] = fix_base_link
        default_dof_pos = system_config.get('default_dof_pos', {})
        if rec_q is not None:
            # init_q = rec_q_ctrl[0]
            init_q = rec_q[0]
            def_dof_pos = {k: init_q[[i for i, n in enumerate(sim_dof_names) if k in n][0]] for k in default_dof_pos}
            system_config['default_dof_pos'] = def_dof_pos
            print('sims init_q', init_q)
        if Q_BOOT is not None and wrapped:
            for i, n in enumerate(sim_dof_names):
                default_dof_pos[n] = Q_BOOT[i]
        default_root_states = system_config.get('default_root_states')
        if fix_base_link and default_root_states is not None:
            default_root_states[2] = 1.5
        if args.sims_headless:
            system_config['headless'] = True
        if args.sims_pd:
            system_config['compute_torque'] = False
        # system_config['verbose'] = True
        if sim_kps is not None:
            system_config['Kp'] = system_config.get('Kp', {})
            system_config['Kp'].update(sim_kps)
            system_config['Kd'] = system_config.get('Kd', {})
            system_config['Kd'].update(sim_kds)
            if sim_torque_limits is not None:
                system_config['torque_limits'] = system_config.get('torque_limits', {})
                system_config['torque_limits'].update(sim_torque_limits)
        elif args.sims_use_kpd:
            # system_config['Kp'] = kp
            # system_config['Kd'] = kd
            system_config['Kp'] = {k: v for k, v in zip(sim_dof_names, kp)}
            system_config['Kd'] = {k: v for k, v in zip(sim_dof_names, kd)}
        system_config['dt'] = dt
        sims_override = args.sims_override
        sims_override = yaml.safe_load(sims_override or '') or {}
        from unicon.utils import obj_update
        obj_update(system_config, sims_override)
        # system_config.update(sims_override)
        wrapper_config = []
        if not args.sims_no_reset:
            wrapper_config.append('sims.wrappers.autoreset:AutoResetWrapper')
        sims_wrapper_config = args.sims_wrapper_config
        if sims_wrapper_config is not None:
            wrapper_config.extend([yaml.safe_load(c) for c in sims_wrapper_config])
        from unicon.systems.sims import cb_sims_recv_send_close
        cb_recv, cb_send, cb_close = cb_sims_recv_send_close(
            **states_props_sys,
            **states_ctrls_sys,
            states_pos=states_pos,
            states_lin_vel=states_lin_vel,
            states_lin_acc=states_lin_acc,
            states_q_tau=states_q_tau,
            dof_names=sim_dof_names,
            system_config=system_config,
            wrapper_config=wrapper_config,
            **sys_kwds,
        )
    elif systems['a1']:
        from unicon.systems.a1 import cb_a1_recv_send_close
        cb_recv, cb_send, cb_close = cb_a1_recv_send_close(
            **states_props_sys,
            **states_ctrls_sys,
            states_input=(None if input_type else states_input),
            kp=kp,
            kd=kd,
            q_ctrl_min=q_ctrl_min,
            q_ctrl_max=q_ctrl_max,
            clip_q_ctrl=sys_clip_q_ctrl,
            **sys_kwds,
        )
    elif systems['unitree']:
        from unicon.systems.unitree import cb_unitree_recv_send_close
        cb_recv, cb_send, cb_close = cb_unitree_recv_send_close(
            **states_props_sys,
            **states_ctrls_sys,
            states_q_tau=states_q_tau,
            states_q_cur=states_q_cur,
            states_lin_vel=states_lin_vel,
            states_lin_acc=states_lin_acc,
            states_input=(None if input_type else states_input),
            kp=kp,
            kd=kd,
            q_ctrl_min=q_ctrl_min,
            q_ctrl_max=q_ctrl_max,
            clip_q_ctrl=sys_clip_q_ctrl,
            **sys_kwds,
        )
    else:
        raise ValueError('no system specified')

    dry_run = args.dry
    if dry_run:
        cb_send = cb_noop()

    def close_fn():
        print('closing')
        if rec_output is not None:
            save_path = rec_output
            rec_len = rec.get('len', -1) + 1
            print('rec_len', rec_len, save_path)
            if rec_len:
                data = {
                    'type': 'unicon',
                    'args': vars(args),
                    'env_cfg_args': env_cfg_args,
                    'argv': argv,
                    'dt': dt,
                    'hostname': __import__('platform').node(),
                }
                data.update({k: v[:rec_len] for k, v in rec.items() if isinstance(v, np.ndarray)})
                data.update(saved_info)
                np.save(save_path, data, allow_pickle=True)
        if not reuse:
            states_destroy(force=True)
        if proc is not None:
            proc.terminate()
        if cb_close is not None:
            cb_close()
        exit(0)

    verify_recv = args.verify_recv
    verify_recv = verify_recv and fft_sys

    if verify_recv:
        print('verify_recv')
        states_verify = states_props.copy()
        # states_verify.pop('states_quat')
        num_recvs = 20
        sps = {k: np.zeros((num_recvs, len(v))) for k, v in states_verify.items()}
        for i in range(num_recvs):
            time.sleep(0.2)
            cb_recv()
            for k, v in states_verify.items():
                sps[k][i] = v
        cnd = False
        eps = 1e-8
        recv_quat = sps.pop('states_quat')
        for k, v in sps.items():
            spd = v[1:] - v[:-1]
            maxd = np.max(np.abs(spd), axis=0)
            print(k, np.max(maxd))
            cond = maxd < eps
            if np.any(cond):
                inds = np.where(cond)[0]
                print('verify recv failed static', k, inds)
                print(maxd)
                if 'q' not in k:
                    cnd = True
                    continue
                print([DOF_NAMES[ii] for ii in inds])
                if any([ii in _dof_map for ii in inds]):
                    cnd = True
                if any([ii < 15 for ii in inds]):
                    cnd = True
        min_quat = np.min(np.linalg.norm(recv_quat, axis=-1), axis=0)
        if min_quat < 0.5:
            print('verify recv failed min_quat', min_quat)
            cnd = True
        recv_q = sps['states_q']
        max_q = np.max(np.abs(recv_q), axis=0)
        cond = max_q > np.pi
        if np.any(cond):
            inds = np.where(cond)[0]
            print('verify recv failed max_q', max_q, inds)
            cnd = True
        if cnd:
            close_fn()
            return

    pre_recv = 1
    pre_recv = 0
    if pre_recv and cb_recv is not None:
        print('pre_recv')
        for _ in range(pre_recv):
            cb_recv()

    if ctrl_dt > dt:
        intv = int(ctrl_dt // dt)
        print('ctrl_intv', ctrl_dt, dt, intv)
        cb = seq[0] if len(seq) == 1 else cb_zip(*seq)

        def cb_if(_cb, pred):
            _pt = -1

            def __cb():
                nonlocal _pt
                _pt += 1
                return _cb() if pred(_pt) else None

            return __cb

        # cb = cb_if(cb, intv=int(ctrl_dt // dt))
        cb = cb_if(cb, pred=lambda pt: pt % intv == 0)
        seq = [cb]
    elif ctrl_dt < dt:
        raise NotImplementedError

    cb_ctrl_tau = args.cb_ctrl_tau
    if cb_ctrl_tau is not None:
        print('cb_ctrl_tau', cb_ctrl_tau)
        cb_ctrl_tau_cls = import_obj(cb_ctrl_tau, default_name_prefix='cb_ctrl_tau', default_mod_prefix='unicon.ctrl')
        seq.extend([
            cb_ctrl_tau_cls(
                **states_ctrls,
                states_q=states_q,
                states_qd=states_qd,
                Kp=kp,
                Kd=kd,
            ),
        ])

    if args.safety_integrity_check:
        from unicon.safety import cb_safety_integrity_check
        cb = cb_safety_integrity_check(
            states_q=states_q,
            states_qd=states_qd,
            states_q_ctrl=states_q_ctrl,
            dof_map=dof_map,
        )
        seq.append(cb)

    seq = [cb_recv] + seq + [cb_send]
    seq = [c for c in seq if c is not None]

    if q_transform is not None:
        cb_qtf_recv()
        cb_qtf_send()
        seq.insert(1, cb_qtf_recv)
        seq.insert(-1, cb_qtf_send)

    fixed_wait = args.fixed_wait
    if fixed_wait is not None:
        from unicon.general import cb_timer_set, cb_timer_wait
        cb0 = cb_timer_set()
        cb1 = cb_timer_wait(wait=fixed_wait, stats=(True if fixed_wait == 0 else False))
        seq.insert(0, cb0)
        seq.insert(-1, cb1)

    outputs = args.outputs
    states_out = dict(
        **states_ctrls,
        states_input=states_input,
        **states_props,
        **states_extras,
    )
    for output_type in outputs:
        print('output_type', output_type)
        cb_output_cls = import_obj(output_type, default_name_prefix='cb_send', default_mod_prefix='unicon.io')
        cb = cb_output_cls(**states_out, robot_def=robot_def)
        seq.append(cb)

    if rec_post_send and rec_output is not None:
        seq.append(_cb_rec)

    if verbose:
        seq.append(cb_print())

    if input_type:
        fallback_input_types = ['term']
        for t in [input_type] + fallback_input_types:
            print('input_type', t)
            cb_input_cls = import_obj(t, default_name_prefix='cb_input', default_mod_prefix='unicon.inputs')
            cb_input = cb_input_cls(states_input=states_input)
            if cb_input is not None:
                break
        seq.extend([
            cb_input,
        ])

    seq.append(cb_wait_input(states_input=states_input, keys=outer_stop_keys))

    cmd_type = args.cmd_type
    if cmd_type and cmd_type != 'none':
        print('cmd_type', cmd_type)
        cb_cmd_cls = import_obj(cmd_type, default_name_prefix='cb_cmd', default_mod_prefix='unicon.cmd')
        kwds = {}
        ccv = args.cmd_const_v
        if ccv is not None:
            ccv = yaml.safe_load(ccv)
            ccv = [ccv] if isinstance(ccv, (float, int)) else ccv
            kwds['cmd'] = ccv
        cir = args.cmd_init_range
        if cir is not None:
            cir = int(cir)
            kwds['init_range_pt'] = cir
        if cmd_type == 'vel' and env_cfg is not None:
            ranges = env_cfg['commands']['ranges']
            print('env_cfg cmd ranges', ranges)
            kwds['lin_vel_x'] = ranges['lin_vel_x']
            kwds['lin_vel_y'] = ranges['lin_vel_y']
            kwds['ang_vel_yaw'] = ranges['ang_vel_yaw']
        if cmd_type == 'replay':
            rec_cmd = loaded_rec.get('states_cmd')
            kwds['frames'] = rec_cmd
        if cb_cmd_cls is not None:
            cb_cmd = cb_cmd_cls(states_input=states_input, states_cmd=states_cmd, input_keys=input_keys, **kwds)
            cb_cmd()
            cmd_max_steps = args.cmd_max_steps
            if cmd_max_steps is not None:
                cb_cmd = cb_loop(
                    cb_cmd,
                    max_steps=cmd_max_steps,
                )
                cb_cmd = cb_chain(
                    cb_cmd,
                    lambda: states_cmd.fill(0),
                )
            seq.extend([
                cb_cmd,
            ])

    if args.safety_ctrl:
        from unicon.safety import cb_safety_ctrl
        cb_sft_ctrl = cb_safety_ctrl(
            **states_ctrls,
            states_q=states_props['states_q'],
            states_qd=states_props['states_qd'],
            check_q=False,
            q_min=Q_CTRL_MIN,
            q_max=Q_CTRL_MAX,
            # check_dq=True,
            check_dq=False,
            # check_tau=False,
            # check_tau=True,
            check_tau=2,
            check_tau_power=True,
            qd_limit=QD_LIMIT,
            tau_limit=TAU_LIMIT,
            Kp=kp,
            Kd=kd,
            power_limit=1200,
            power_max_breaks=2**2,
            power_halt=1440,
        )
        seq.extend([
            cb_sft_ctrl,
        ])

    if args.safety_states:
        max_roll = np.pi / 4
        max_pitch = np.pi / 4
        from unicon.safety import cb_safety_states
        cb_sft_states = cb_safety_states(
            states_rpy=states_props['states_rpy'],
            states_qd=states_props['states_qd'],
            check_rpy=True,
            check_qd=False,
            check_power=False,
            qd_limit=QD_LIMIT,
            max_roll=max_roll,
            max_pitch=max_pitch,
        )
        seq.extend([
            cb_sft_states,
        ])

    print('outer seq', seq)
    cb = cb_zip(*seq)

    if args.close:
        close_fn()
        return

    loop_kwds = {}
    loop_sleep_block = args.loop_sleep_block
    if loop_sleep_block:
        loop_kwds['sleep_fn'] = 'sleep_block'
    loop_dt_ofs = args.loop_dt_ofs
    if loop_dt_ofs is not None:
        loop_kwds['dt_ofs'] = loop_dt_ofs

    loop_dt = args.loop_dt
    loop_dt = dt if loop_dt is None else loop_dt
    from unicon.utils import loop_timed
    try:
        if args.fast:
            while True:
                ret = cb()
                if ret is True:
                    break
        else:
            loop_timed(cb, dt=loop_dt, **loop_kwds)
    except KeyboardInterrupt:
        import traceback
        traceback.print_exc()
        print('interrupted')

    close_fn()


if __name__ == '__main__':
    run()
