import os
import time
import numpy as np


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-ca', '--cpu_affinity', default=None)
    parser.add_argument('-c', '--close', action='store_true')
    parser.add_argument('-m', '--mode', action='append')
    parser.add_argument('-dt', '--dt', type=float, default=None)
    parser.add_argument('-n', '--num_steps', type=int, default=0)
    parser.add_argument('-s', '--system', default='none')
    parser.add_argument('-ro', '--rec_output', default=None)
    parser.add_argument('-it', '--infer_type', default='h0')
    parser.add_argument('-di', '--input_dev_type', default='js')
    parser.add_argument('-pt', '--policy_type', default='none')
    parser.add_argument('-mt', '--model_type', default=None)
    parser.add_argument('-cmd', '--cmd', action='append')
    parser.add_argument('-ccv', '--cmd_const_v', default=None)
    parser.add_argument('-iks', '--input_keys', default=None)
    parser.add_argument('-d', '--dry', action='store_true')
    parser.add_argument('-v', '--verbose', default=None)
    parser.add_argument('-nwp', '--no_wrap', action='store_true')
    parser.add_argument('-w', '--wait', type=float, default=0)
    parser.add_argument('-kp', '--kp', default=None)
    parser.add_argument('-kd', '--kd', default=None)
    parser.add_argument('-kpr', '--kp_ratio', type=float, default=1.0)
    parser.add_argument('-kdr', '--kd_ratio', type=float, default=1.0)
    parser.add_argument('-tlr', '--tau_limit_ratio', type=float, default=1.0)
    parser.add_argument('-spt', '--sampler_type', type=str, default='uniform')
    parser.add_argument('-spkwds', '--sampler_kwds', type=str, default=None)
    parser.add_argument('-spr', '--sample_r', type=str, default=None)
    parser.add_argument('-spd', '--sample_dofs', default=None)
    parser.add_argument('-spl', '--sample_lerp', type=int, default=0)
    parser.add_argument('-spw', '--sample_wait', type=int, default=0)
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
    parser.add_argument('-imp', '--infer_model_path', default=None)
    parser.add_argument('-ikwds', '--infer_kwargs', default=None)
    parser.add_argument('-nwi', '--no_wait_input', action='store_true')
    parser.add_argument('-ecp', '--env_cfg_path', default=None)
    parser.add_argument('-eco', '--env_cfg_override', default=None)
    parser.add_argument('-sst', '--sims_type', default='sims.systems.ig')
    parser.add_argument('-ssc', '--sims_config', default=None)
    parser.add_argument('-sso', '--sims_override', default=None)
    parser.add_argument('-ssar', '--sims_auto_reset', action='store_true')
    parser.add_argument('-sswc', '--sims_wrapper_config', action='append')
    parser.add_argument('-ssh', '--sims_headless', action='store_true')
    parser.add_argument('-ssfb', '--sims_fixed_base', action='store_true')
    parser.add_argument('-ssdc', '--sims_decimation', type=int, default=10)
    parser.add_argument('-ssiz', '--sims_init_z', type=float, default=None)
    parser.add_argument('-ssct', '--sims_compute_torque', action='store_true')
    parser.add_argument('-ssuk', '--sims_use_kpd', action='store_true')
    parser.add_argument('-skwds', '--system_kwargs', default=None)
    parser.add_argument('-cqc', '--clip_q_ctrl', action='store_true')
    parser.add_argument('-cqct', '--clip_q_ctrl_tau', action='store_true')
    parser.add_argument('-cqcd', '--clip_q_ctrl_qd', action='store_true')
    parser.add_argument('-sfts', '--safety_states', action='store_true')
    parser.add_argument('-sftc', '--safety_ctrl', action='store_true')
    parser.add_argument('-dofs', '--dofs', default=None)
    parser.add_argument('-lrpt', '--lerp_time', type=float, default=5)
    parser.add_argument('-shm', '--shm', action='store_true')
    parser.add_argument('-shmc', '--shm_clear', action='store_true')
    parser.add_argument('-rt', '--robot_type', default='none')
    parser.add_argument('-vr', '--verify_recv', action='store_true')
    parser.add_argument('-pr', '--pre_recv', type=int, default=0)
    parser.add_argument('-ps', '--pre_send', type=int, default=0)
    parser.add_argument('-mkln', '--mkl_n_thread', type=int, default=2)
    parser.add_argument('-ff', '--fast_fwd', action='store_true')
    parser.add_argument('-sqm', '--safety_q_margin', type=float, default=0.)
    parser.add_argument('-iis', '--inner_input_stop', action='store_true')
    parser.add_argument('-nois', '--no_outer_input_stop', action='store_true')
    parser.add_argument('-sd', '--seed', type=int, default=None)
    parser.add_argument('-ip', '--infer_profile', action='store_true')
    parser.add_argument('-id', '--infer_device', default='cpu')
    parser.add_argument('-fl', '--fixed_lat', type=float, default=None)
    parser.add_argument('-fw', '--fixed_wait', type=float, default=None)
    parser.add_argument('-lsb', '--loop_sleep_block', action='store_true')
    parser.add_argument('-ldo', '--loop_dt_ofs', type=float, default=None)
    parser.add_argument('-ldt', '--loop_dt', type=float, default=None)
    parser.add_argument('-sqe', '--states_q_extras', action='store_true')
    parser.add_argument('-sxe', '--states_x_extras', action='store_true')
    parser.add_argument('-sxe2', '--states_x_extras2', action='store_true')
    parser.add_argument('-scs', '--states_custom_specs', default=None)
    parser.add_argument('-tc', '--tau_ctrl', action='store_true')
    parser.add_argument('-cct', '--cb_ctrl_tau', default=None)
    parser.add_argument('-cdt', '--ctrl_dt', type=float, default=None)
    parser.add_argument('-ilr', '--infer_load_run', default=None)
    parser.add_argument('-rcps', '--rec_post_send', action='store_true')
    parser.add_argument('-i', '--inputs', default=[], action='append')
    parser.add_argument('-o', '--outputs', default=[], action='append')
    parser.add_argument('-io', '--inouts', default=[], action='append')
    parser.add_argument('-uspd', '--use_env_pd', action='store_true')
    parser.add_argument('-dnstd', '--dof_names_std', action='store_true')
    parser.add_argument('-dnr', '--dof_names_remap', default=None)
    parser.add_argument('-dns', '--dof_names_sub', default=None)
    parser.add_argument('-rp', '--rec_path', default=None)
    parser.add_argument('-rtp', '--rec_type', default=None)
    parser.add_argument('-rpt', '--rec_pt', type=int, default=None)
    parser.add_argument('-rkwds', '--rec_kwds', default=None)
    parser.add_argument('-ncmd', '--num_commands', type=int, default=None)
    parser.add_argument('-nice', '--nice', type=int, default=None)
    parser.add_argument('-su', '--sudo', action='store_true')
    parser.add_argument('-qtf', '--q_transform', default=None)
    parser.add_argument('-itf', '--imu_transform', default=None)
    parser.add_argument('-tf', '--transforms', default=[], action='append')
    parser.add_argument('-ptf', '--post_transforms', default=[], action='append')
    parser.add_argument('-pi', '--pre_imports', default=[], action='append')
    parser.add_argument('-sie', '--states_infer_extras', action='store_true')
    parser.add_argument('-nqb', '--no_q_boot', action='store_true')
    parser.add_argument('-rdu', '--robot_def_update', default=None)
    parser.add_argument('-ht', '--hand_type', default='none')
    parser.add_argument('-xcn', '--x_ctrl_eef_names', default=None)
    parser.add_argument('-xcdn', '--x_ctrl_dof_names', default=None)
    parser.add_argument('-simgs', '--states_imgs', default=None)
    args, _ = parser.parse_known_args()
    return args


def run(args=None):
    if args is None:
        args = get_args()
    if not isinstance(args, dict):
        args = vars(args)
    for x in args['pre_imports']:
        try:
            r = __import__(x)
        except ImportError as e:
            r = e
        print('pre_import', x, r)
    try:
        import isaacgym
        del isaacgym
    except ImportError:
        pass
    from unicon.states import states_init, states_news, states_new, states_get, states_destroy, autowired
    from unicon.general import cb_chain, cb_noop, cb_print, cb_prod, cb_zip, cb_if, cb_loop_timed,\
        cb_timeout, cb_fixed_lat, cb_wait_input, cb_replay
    from unicon.utils import set_nice2, set_cpu_affinity2, list2slice, pp_arr, set_seed, \
        import_obj, parse_robot_def, load_obj, match_keys, obj_update, get_ctx, validate_sudo, expect
    from unicon.ctrl import cb_ctrl_q_from_target_lerp

    if args['sudo']:
        validate_sudo(True)

    nice = args['nice']
    if nice is not None:
        set_nice2(nice)

    affinity = args['cpu_affinity']
    if affinity is not None:
        set_cpu_affinity2(affinity)

    wait = args['wait']
    if wait:
        print('wait', wait)
        time.sleep(wait)

    argv = list(__import__('sys').argv)
    verbose = args['verbose']

    mkl_n_thread = args['mkl_n_thread']
    print('mkl_n_thread', mkl_n_thread)
    if mkl_n_thread:
        os.environ['MKL_NUM_THREADS'] = str(mkl_n_thread)

    if args['seed'] is not None:
        set_seed(args['seed'])

    robot_type = args['robot_type']
    if robot_type == 'none':
        robot_def = {}
    else:
        robot_def = import_obj(robot_type, default_mod_prefix='unicon.defs', prefer_mod=True)
        robot_def_update = args['robot_def_update']
        if robot_def_update is not None:
            robot_def_update = load_obj(robot_def_update)
            obj_update(robot_def, robot_def_update)
        robot_def = parse_robot_def(robot_def)

    KP = robot_def.get('KP', None)
    KD = robot_def.get('KD', None)
    Q_CTRL_MIN = robot_def.get('Q_CTRL_MIN', None)
    Q_CTRL_MAX = robot_def.get('Q_CTRL_MAX', None)
    DOF_NAMES = robot_def.get('DOF_NAMES', [])
    NUM_DOFS = robot_def.get('NUM_DOFS', 0)
    DOF_MAPS = robot_def.get('DOF_MAPS', None)
    TAU_LIMIT = robot_def.get('TAU_LIMIT', None)
    QD_LIMIT = robot_def.get('QD_LIMIT', None)
    Q_BOOT = robot_def.get('Q_BOOT', None)
    Q_RESET = robot_def.get('Q_RESET', None)
    DOF_NAMES_STD = robot_def.get('DOF_NAMES_STD', {})
    robot_def['DOF_NAMES_STD'] = DOF_NAMES_STD
    LINK_NAMES = robot_def.get('LINK_NAMES', None)

    if args['no_q_boot']:
        Q_BOOT = None

    q_reset = np.zeros(NUM_DOFS) if Q_RESET is None else Q_RESET

    HAND_NUM_DOFS = robot_def.get('HAND_NUM_DOFS', 0)
    hand_type = args['hand_type']
    if hand_type != 'none':
        hand_def = import_obj(hand_type, default_mod_prefix='unicon.defs', prefer_mod=True)
        hand_def = parse_robot_def(hand_def)
        HAND_NUM_DOFS = hand_def['NUM_DOFS']
        hand_q_min = hand_def.get('Q_CTRL_MIN')
        hand_q_max = hand_def.get('Q_CTRL_MAX')
        hand_q_reset = hand_def.get('Q_RESET')
        if hand_q_reset is None:
            hand_q_reset = np.zeros(HAND_NUM_DOFS, dtype=np.float32)
        hand_q_opened = hand_def.get('Q_OPENED', hand_q_reset)
        hand_q_closed = hand_def.get('Q_CLOSED')
        if hand_q_closed is None and hand_q_min is not None and hand_q_max is not None:
            hand_q_closed = hand_q_max - (hand_q_opened - hand_q_min)
        if hand_q_closed is None and hand_q_reset is not None:
            hand_q_closed = hand_q_reset * 2 - hand_q_opened
        if hand_q_min is not None and hand_q_max is not None:
            hand_q_reset = np.clip(hand_q_reset, hand_q_min, hand_q_max)
            hand_q_opened = np.clip(hand_q_opened, hand_q_min, hand_q_max)
            hand_q_closed = np.clip(hand_q_closed, hand_q_min, hand_q_max)
        hand_def['Q_RESET'] = hand_q_reset
        hand_def['Q_OPENED'] = hand_q_opened
        hand_def['Q_CLOSED'] = hand_q_closed
        robot_def.update({f'HAND_{k}': v for k, v in hand_def.items()})
        robot_def['hand_def'] = hand_def

    dt = args['dt']
    ctrl_dt = args['ctrl_dt']
    ctx = get_ctx()
    ctx.update({
        'args': args,
        'argv': argv,
        'robot_def': robot_def,
        'DOF_NAMES': DOF_NAMES,
        'hostname': __import__('platform').node(),
    })

    dof_names = None

    dofs = args['dofs']
    if dofs is not None:
        dofs = load_obj(dofs)
        if isinstance(dofs, str):
            pass
        elif isinstance(dofs, (list, tuple)):
            dof_names = dofs
        else:
            raise ValueError(f'invalid dofs {dofs}')

    dof_names = DOF_NAMES if dof_names is None else dof_names

    env_cfg_path = args['env_cfg_path']
    infer_load_run = args['infer_load_run']
    infer_model_path = args['infer_model_path']
    infer_type = args['infer_type']
    if infer_load_run is not None:
        infer_load_run = infer_load_run.split(':')
        load_run_path = infer_load_run[0]
        model_file = infer_load_run[1] if len(infer_load_run) > 1 else None
        _default_infer_root = os.environ.get('UNICON_INFER_ROOT')
        if _default_infer_root is None:
            from unicon.utils import find
            root = find(root='..', path=f'*{load_run_path}')
            if root is None:
                root = find(root='~', path=f'*{load_run_path}')
            assert root is not None, 'infer policy dir not found'
            root = root[0]
        else:
            root = os.path.join(_default_infer_root, load_run_path)
        print('infer root', root, model_file)
        model_file_pats = ['policy', 'trace'] if model_file is None else [model_file]
        for r, _, fs in os.walk(root):
            for f in sorted(fs):
                if any([p in f for p in model_file_pats]):
                    model_file = os.path.join(r, f)
                    break
        if model_file is None:
            raise ValueError('model_file not found')
        infer_model_path = model_file
        from unicon.utils import md5sum
        infer_model_md5sum = md5sum(infer_model_path)
        ctx['infer_model_md5sum'] = infer_model_md5sum
        print('infer_load_run', load_run_path, infer_model_path, infer_model_md5sum)
    if env_cfg_path is None and infer_model_path is not None:
        for ext in ['.json', '.yaml']:
            env_cfg_path = os.path.join(os.path.dirname(infer_model_path), 'env_cfg' + ext)
            if os.path.exists(env_cfg_path):
                break
        else:
            print('env_cfg not found', infer_model_path)

    env_cfg = None
    if env_cfg_path is not None and os.path.exists(env_cfg_path):
        print('env_cfg_path', env_cfg_path)
        env_cfg = load_obj(env_cfg_path)
    if env_cfg is not None:
        if infer_type is None:
            infer_type = env_cfg.get('infer_type')
        env_cfg_override = args['env_cfg_override']
        if env_cfg_override is not None:
            env_cfg_override = load_obj(env_cfg_override)
            obj_update(env_cfg, env_cfg_override)
        env_cfg_fn = import_obj(infer_type, default_name_prefix='parse_env_cfg', default_mod_prefix='unicon.infer')
        ret = env_cfg_fn(env_cfg)
        if ret is not None:
            ctx.update(ret)

    dof_names = ctx.get('dof_names', dof_names)
    DOF_NAMES_STD.update(ctx.get('dof_names_std', {}))
    ctrl_dt = ctx.get('ctrl_dt', ctrl_dt)
    env_kps = ctx.get('env_kps', None)
    env_kds = ctx.get('env_kds', None)
    env_torque_limits = ctx.get('env_torque_limits', None)
    default_joint_angles = ctx.get('default_joint_angles', None)
    env_command_keys = ctx.get('env_command_keys', None)
    env_command_ranges = ctx.get('env_command_ranges', None)
    env_command_def_vals = ctx.get('env_command_def_vals', None)
    env_num_commands = 0 if env_command_keys is None else len(env_command_keys)

    if default_joint_angles is not None:
        inds, vals = match_keys(default_joint_angles, DOF_NAMES)
        q_reset[inds] = vals

    fast_fwd = args['fast_fwd']
    ctrl_dt = dt if ctrl_dt is None else ctrl_dt
    dt = ctrl_dt if dt is None else dt
    print('dt', dt, 'ctrl_dt', ctrl_dt)
    expect(fast_fwd or (ctrl_dt is not None and dt is not None))
    ctx.update({
        'dt': dt,
        'ctrl_dt': ctrl_dt,
    })

    tau_ctrl = args['tau_ctrl']
    specs = {}
    if NUM_DOFS:
        assert NUM_DOFS == len(DOF_NAMES)
        specs.update({
            'rpy': 3,
            'ang_vel': 3,
            'quat': 4,
            'q': NUM_DOFS,
            'qd': NUM_DOFS,
            'q_ctrl': NUM_DOFS,
            'q_target': NUM_DOFS,
        })
    if HAND_NUM_DOFS:
        specs.update({
            'hand_q': HAND_NUM_DOFS,
            'hand_q_ctrl': HAND_NUM_DOFS,
        })
    if tau_ctrl:
        specs['tau_ctrl'] = NUM_DOFS
    q_extras = args['states_q_extras']
    if q_extras:
        specs.update({
            'q_tau': NUM_DOFS,
            'q_cur': NUM_DOFS,
            'q_temp': NUM_DOFS,
        })
    x_extras = args['states_x_extras']
    x_extras2 = args['states_x_extras2']
    from unicon.utils import pats2inds
    NUM_LINKS = 0 if LINK_NAMES is None else len(LINK_NAMES)
    x_ctrl_eef_names = load_obj(args['x_ctrl_eef_names'])
    x_ctrl_eef_names = [] if x_ctrl_eef_names is None else x_ctrl_eef_names
    # x_ctrl_eef_names = [n for n in x_ctrl_eef_names if n in LINK_NAMES]
    x_ctrl_eef_inds, x_ctrl_eef_names, _ = pats2inds(x_ctrl_eef_names, LINK_NAMES, None)
    ctx['x_ctrl_eef_names'] = x_ctrl_eef_names
    ctx['x_ctrl_eef_inds'] = x_ctrl_eef_inds
    num_x_ctrl = len(x_ctrl_eef_names)
    print('x_ctrl_eef_names', num_x_ctrl, x_ctrl_eef_names, x_ctrl_eef_inds)

    x_ctrl_dof_names = load_obj(args['x_ctrl_dof_names'])
    # x_ctrl_dof_names = [] if x_ctrl_dof_names is None else x_ctrl_dof_names
    x_ctrl_dof_names = DOF_NAMES[:] if x_ctrl_dof_names is None else x_ctrl_dof_names
    x_ctrl_dof_inds, x_ctrl_dof_names, _ = pats2inds(x_ctrl_dof_names, DOF_NAMES, DOF_NAMES_STD)
    # x_ctrl_dof_names = [n for n in x_ctrl_dof_names if n in DOF_NAMES]
    # x_ctrl_dof_inds = [DOF_NAMES.index(n) for n in x_ctrl_dof_names]
    ctx['x_ctrl_dof_names'] = x_ctrl_dof_names
    ctx['x_ctrl_dof_inds'] = x_ctrl_dof_inds
    print('x_ctrl_dof_names', len(x_ctrl_dof_names), x_ctrl_dof_names, x_ctrl_dof_inds)

    if x_extras or x_extras2:
        specs.update({
            'pos': 3,
            'lin_vel': 3,
            'lin_acc': 3,
        })
    if NUM_LINKS and x_extras2:
        specs.update({
            'x': [(NUM_LINKS, 4, 4)],
            'xd': [(NUM_LINKS, 6)],
            'J': [(NUM_LINKS, 6, NUM_DOFS)],
        })
    if num_x_ctrl:
        specs.update({
            'x_ctrl': [(NUM_LINKS, 4, 4)],
            # 'x_ctrl': [(num_x_ctrl, 4, 4)],
            'x_err': [(NUM_LINKS, 6)],
        })

    imgs = args['states_imgs']
    imgs = load_obj(imgs) if imgs is not None else []
    imgs = imgs if isinstance(imgs, (tuple, list)) else [imgs]
    img_defaults = {
        'dtype': 'B',
    }
    for img in imgs:
        if isinstance(img, str):
            img = img.split(',')
        if isinstance(img, (list, tuple)):
            key = img[0]
            res = img[1]
            dtype = img[2] if len(img) > 2 else img_defaults['dtype']
        elif isinstance(img, dict):
            _img = img_defaults.copy()
            _img.update(img)
            key, res, dtype = [_img.get(k) for k in ['key', 'res', 'dtype']] 
        res = list(map(int, res.split('x'))) if isinstance(res, str) else res
        specs[key] = [res, dtype]

    states_custom_specs = args['states_custom_specs']
    if states_custom_specs is not None:
        states_custom_specs = load_obj(states_custom_specs)
        print('states_custom_specs', states_custom_specs)
        specs.update(states_custom_specs)
    states_news(specs)
    input_keys = args['input_keys']
    input_keys = import_obj('unicon.inputs:DEFAULT_INPUT_KEYS') if input_keys is None else load_obj(input_keys)
    ctx['input_keys'] = input_keys
    num_inputs = len(input_keys)
    if num_inputs > 0:
        states_new('input', num_inputs)
    num_commands = args['num_commands']
    num_commands = env_num_commands if num_commands is None else num_commands
    if env_num_commands is not None and num_commands < 0:
        num_commands = env_num_commands - num_commands
    num_commands = 0 if num_commands is None else num_commands
    if num_commands > 0:
        states_new('cmd', num_commands)
    use_shm = False
    shm_clear = args['shm_clear']
    use_shm = args['shm'] or shm_clear
    load = use_shm and not shm_clear
    save = use_shm
    reuse = True
    states_init(use_shm=use_shm, save=save, load=load, reuse=reuse, clear=shm_clear)

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
    states_q_temp = states_get('q_temp')
    states_pos = states_get('pos')
    states_lin_vel = states_get('lin_vel')
    states_lin_acc = states_get('lin_acc')
    states_x = states_get('x')
    states_x_ctrl = states_get('x_ctrl')
    states_extras = {
        'states_cmd': states_cmd,
        'states_q_temp': states_q_temp,
        'states_q_tau': states_q_tau,
        'states_q_cur': states_q_cur,
        'states_pos': states_pos,
        'states_lin_vel': states_lin_vel,
        'states_lin_acc': states_lin_acc,
        'states_x': states_x,
        'states_xd': states_get('xd'),
        'states_x_ctrl': states_x_ctrl,
    }
    states_extras = {k: v for k, v in states_extras.items() if v is not None}
    if HAND_NUM_DOFS:
        states_hand_q = states_get('hand_q')
        states_hand_q_ctrl = states_get('hand_q_ctrl')

    states_infer_extras = {}
    inf_extras = args['states_infer_extras']
    if inf_extras and env_cfg is not None:
        num_acts = ctx['num_acts']
        num_obs = ctx['num_obs']
        states_infer_acts = np.zeros(num_acts)
        states_infer_obs = np.zeros(num_obs)
        states_infer_extras = {
            'states_infer_acts': states_infer_acts,
            'states_infer_obs': states_infer_obs,
        }

    dof_names_ori = dof_names[:]
    print('dof_names_ori', len(dof_names_ori), dof_names_ori)

    dof_names_sub = args['dof_names_sub']
    if dof_names_sub is not None:
        sub = load_obj(dof_names_sub)

        def sub_name(x):
            for t1, t2 in sub:
                x = x.replace(t1, t2)
            return x

        dof_names = [sub_name(n) for n in dof_names]

    dof_names_remap = args['dof_names_remap']
    if dof_names_remap is not None:
        remap = load_obj(dof_names_remap)
        dof_names = [remap.get(n, n) for n in dof_names]

    if args['dof_names_std']:
        dof_names2std = [DOF_NAMES_STD.get(n, n) for n in DOF_NAMES]
        print('dof_names2std', dof_names2std)
        dof_names_std_rev = {v: k for k, v in DOF_NAMES_STD.items()}
        dof_names = [dof_names_std_rev.get(n, n) for n in dof_names]

    dof_names_map = {k: nk for k, nk in zip(dof_names_ori, dof_names)}
    print('DOF_NAMES', len(DOF_NAMES), DOF_NAMES)
    print('dof_names', len(dof_names), dof_names)
    dof_names_extra = [n for n in dof_names if n not in DOF_NAMES]
    DOF_NAMES_extra = [n for n in DOF_NAMES if n not in dof_names]
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
    assert NUM_DOFS == 0 or num_dofs > 0

    _dof_map = dof_map
    dof_map = list2slice(dof_map)
    dof_src_map = list2slice(dof_src_map)
    print('num_dofs', num_dofs)
    print('dof_map', dof_map)
    print('dof_src_map', dof_src_map)
    print('DOF_NAMES_extra', len(DOF_NAMES_extra), DOF_NAMES_extra)

    q_boot = q_reset if Q_BOOT is None else Q_BOOT

    ctx['Q_RESET'] = q_reset
    print('q_reset', q_reset.tolist())
    print('q_boot', q_boot.tolist())

    if states_x_ctrl is not None or states_x is not None:
        use_algo_fk = False
        robot_pin = robot_def.get('robot_pin')
        if use_algo_fk or robot_pin is None:
            from unicon.algo.fk import cb_fk
            _states_x = np.zeros((NUM_LINKS, 4, 4), dtype=np.float32)
            _cb = cb_fk(states_q=q_reset, states_x=_states_x)
            _cb()
            x_reset = _states_x
        elif robot_pin is not None:
            from unicon.utils.pin import pin_fk
            URDF_DOF_NAMES = robot_def.get('URDF_DOF_NAMES', DOF_NAMES)
            print('URDF_DOF_NAMES', URDF_DOF_NAMES)
            q_reset_urdf = np.zeros(len(URDF_DOF_NAMES), dtype=np.float32)
            urdf_inds, _, dof_inds = pats2inds(DOF_NAMES, URDF_DOF_NAMES)
            q_reset_urdf[urdf_inds] = q_reset[dof_inds]
            print('q_reset_urdf', pp_arr(q_reset_urdf))
            x_reset = pin_fk(q_reset_urdf, robot_pin=robot_pin, dof_names=URDF_DOF_NAMES, link_names=LINK_NAMES)
        else:
            x_reset = np.stack([np.eye(4, dtype=np.float32) for _ in range(len(states_x_ctrl))], axis=0)
        print('x_reset', x_reset.shape)
        if verbose is not None:
            for i in range(NUM_LINKS):
                print(i, LINK_NAMES[i])
                print(np.round(x_reset[i].astype(np.float32), 3))
        for ei in x_ctrl_eef_inds:
            print(ei, LINK_NAMES[ei])
            print(np.round(x_reset[ei].astype(np.float32), 3))
        ctx['x_reset'] = x_reset
    if states_x_ctrl is not None:
        states_x_ctrl[:] = x_reset
    if states_x is not None:
        states_x[:] = x_reset

    if dof_states_padded:
        dtype = states_q_ctrl.dtype
        states_q_ctrl_inf = np.zeros(NUM_DOFS + 1, dtype=dtype)
        states_q_inf = np.zeros(NUM_DOFS + 1, dtype=dtype)
        states_qd_inf = np.zeros(NUM_DOFS + 1, dtype=dtype)

        states_q_ctrl_inf[:NUM_DOFS] = q_reset

        def cb_pad_in():
            states_q_inf[:NUM_DOFS] = states_q
            states_qd_inf[:NUM_DOFS] = states_qd
            states_q_ctrl_inf[:NUM_DOFS] = states_q_ctrl

        def cb_pad_out():
            states_q_ctrl[:] = states_q_ctrl_inf[:NUM_DOFS]

        states_props_inf = states_props.copy()
        states_props_inf['states_q'] = states_q_inf
        states_props_inf['states_qd'] = states_qd_inf
        states_ctrls_inf = states_ctrls.copy()
        states_ctrls_inf['states_q_ctrl'] = states_q_ctrl_inf
    else:
        states_props_inf = states_props
        states_ctrls_inf = states_ctrls
        states_q_ctrl_inf = states_q_ctrl

    replay_loop = args['replay_loop']
    rec_path = args['rec_path']
    rec_type = args['rec_type']
    rec_q_ctrl = None
    rec_q = None
    loaded_rec = None
    if rec_path is not None:
        _, ext = os.path.splitext(rec_path)
        rec_type = ext[1:] if rec_type is None else rec_type
        print('rec_type', rec_type)
        mod = import_obj(rec_type, default_mod_prefix='unicon.data', prefer_mod=True)
        print('rec_path', rec_path)
        kwds = args['rec_kwds']
        loaded_rec = mod.load(
            rec_path,
            **(kwds or {}),
        )
    if loaded_rec is not None:
        rec_q_ctrl = loaded_rec.get('states_q_ctrl')
        rec_q = loaded_rec.get('states_q')
        print('rec_q_ctrl', rec_q_ctrl.shape)
        print('loaded_rec argv', ' '.join(loaded_rec.get('argv', [])))
        loaded_rec_args = loaded_rec.get('args', {})
        print('loaded_rec_args', loaded_rec_args)
        rec_pt = args['rec_pt']
        if rec_pt is not None:
            print('rec_pt', rec_pt)
            loaded_rec = {k: (v[rec_pt:] if isinstance(v, np.ndarray) else v) for k, v in loaded_rec.items()}

    cb_close_seq = []

    num_steps = args['num_steps']
    seq = []
    arg_mode = args['mode'] or []
    default_modes = [
        'noop',
        'const',
        'sample',
        'replay',
        'infer',
        'teleop',
        'play',
        'follow',
    ]
    modes = []
    for m in arg_mode:
        m = {k[0]: k for k in default_modes}.get(m, m)
        modes.append(m)
        if m == 'noop':
            cb = cb_noop()
        elif m == 'const':
            q_ctrl_const = np.zeros(NUM_DOFS)
            q_ctrl_const[:] = q_reset

            def cb_const():
                states_q_ctrl[:] = q_ctrl_const

            cb = cb_const
        elif m == 'sample':
            num_steps = 1024 if num_steps == 0 else num_steps
            states_q_target[:] = q_reset
            smpl_dof_map = args['sample_dofs']
            if smpl_dof_map is not None:
                smpl_dof_map = load_obj(smpl_dof_map)
            if isinstance(smpl_dof_map, int):
                smpl_dof_map = [int(smpl_dof_map)]
            elif isinstance(smpl_dof_map, str):
                smpl_dof_map = DOF_MAPS[smpl_dof_map]
            elif isinstance(smpl_dof_map, list):
                if isinstance(smpl_dof_map[0], str):
                    smpl_dof_map = [DOF_NAMES.index(n) for n in smpl_dof_map]
            smpl_r = 0.3 if args['sample_r'] is None else load_obj(args['sample_r'])
            if isinstance(smpl_r, float):
                print('mean', Q_CTRL_MAX + Q_CTRL_MIN)
                print('range', Q_CTRL_MAX - Q_CTRL_MIN)
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
            num_samples = args['num_samples']
            num_samples = 2**10 if num_samples is None else num_samples
            print('num_samples', num_samples)
            sampler_type = args['sampler_type']
            sampler_kwds = args['sampler_kwds']
            sampler_kwds = {} if sampler_kwds is None else load_obj(sampler_kwds)
            sampler_kwds['dt'] = ctrl_dt
            print('sampler_type', sampler_type)
            sampler_cls = import_obj(sampler_type, default_name_prefix='sampler', default_mod_prefix='unicon.samplers')
            frames = sampler_cls(low=q_smpl_min, high=q_smpl_max, num_samples=num_samples, **sampler_kwds)
            frames = np.array(list(frames))
            sample_repeats = args['sample_repeats']
            if sample_repeats is not None:
                group = args['sample_repeat_group']
                print('sample_repeats', group)
                sz = frames.shape[-1]
                frames = np.repeat(frames.reshape(-1, group, sz), sample_repeats, axis=0).reshape(-1, sz)
            print('sample frames', frames.shape)
            # smpl_lerp_steps = lerp_steps
            smpl_lerp_steps = args['sample_lerp']
            # smpl_wait_steps = 0
            smpl_wait_steps = args['sample_wait']
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
        elif m == 'teleop':
            from unicon.teleop import cb_teleop_q
            cb = cb_teleop_q(
                **states_ctrls,
                states_q=states_q,
                states_input=states_input,
            )
        elif m == 'replay':
            rep_dof_map = dof_map
            replay_states_key = args['replay_states_key']
            states_dest = states_get(replay_states_key)
            if loaded_rec is not None:
                replay_frame_key = args['replay_frame_key']
                rec_frames = loaded_rec.get(replay_frame_key)
                rec_dof_names = loaded_rec.get('DOF_NAMES', DOF_NAMES)
                print('rec_dof_names', rec_dof_names)
                rec_dof_names = [dof_names_map.get(k, k) for k in rec_dof_names]
                rep_dof_map = [DOF_NAMES.index(n) for n in rec_dof_names if n in DOF_NAMES]
                rep_dof_map = list2slice(rep_dof_map)
                rec_dof_src_map = [i for i, n in enumerate(rec_dof_names) if n in DOF_NAMES]
                rep_dt = loaded_rec.get('rec_dt')
                rep_dt = loaded_rec_args.get('ctrl_dt') if rep_dt is None else rep_dt
                rep_dt = loaded_rec_args.get('dt', ctrl_dt) if rep_dt is None else rep_dt
                frames = rec_frames[:, rec_dof_src_map]
                # frames = rec_frames
            print('rep_dt', rep_dt)
            print('rep_dof_map', rep_dof_map)
            print('frames', frames.shape)
            frames_min = np.min(frames, axis=0).astype(np.float64)
            frames_max = np.max(frames, axis=0).astype(np.float64)
            print('frames min', np.round(frames_min, decimals=3).tolist())
            print('frames max', np.round(frames_max, decimals=3).tolist())
            print('frames rng', np.round((frames_max - frames_min), decimals=3).tolist())
            scale = args['replay_scale']
            if scale is not None:
                frames *= scale
            pt = args['replay_pt']
            if pt is not None:
                frames = frames[pt:]
            replay_len = args['replay_len']
            if replay_len is not None:
                frames = frames[:replay_len]
            replay_interp = not args['replay_no_interp']
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
            repeats = args['replay_repeats']
            repeats = max(rep_dt // ctrl_dt, 1) if repeats is None else repeats
            print('repeats', repeats)
            cb = cb_replay(
                states_dest,
                frames=frames,
                inds=rep_dof_map,
                repeats=repeats,
                loop=replay_loop,
            )
        elif m == 'infer':
            print('infer_type', infer_type)
            cb_infer_cls = import_obj(infer_type, default_name_prefix='cb_infer', default_mod_prefix='unicon.infer')

            policy_fn = None
            policy_reset_fn = None
            infer_device = args['infer_device']
            if infer_model_path is not None:
                print('infer_model_path', infer_model_path)
                from unicon.models import load_model
                model_type = args['model_type']
                model = load_model(infer_model_path, model_type=model_type, device=infer_device)
                policy_type = args['policy_type']
                if policy_type == 'none':
                    policy_fn = model
                elif policy_type == 'debug':

                    def _policy_fn(obs):
                        print('policy obs', obs.shape)
                        ret = model(obs)
                        print('policy ret', ret.shape)
                        return ret

                    policy_fn = _policy_fn
                elif policy_type == 'mem':
                    policy_reset_fn = model.reset_memory
                    policy_fn = model

            infer_kwds = load_obj(args['infer_kwargs'] or '') or {}
            infer_states = {}
            infer_states.update(states_props_inf)
            infer_states.update(states_infer_extras)
            infer_states.update(states_ctrls_inf)
            cb_infer, reset_fn = autowired(cb_infer_cls, states=infer_states)(
                policy_fn=policy_fn,
                policy_reset_fn=policy_reset_fn,
                dof_map=dof_map_padded,
                dof_names=dof_names,
                env_cfg=env_cfg,
                device=infer_device,
                **infer_kwds,
            )

            reset_fn()
            infer_profile = args['infer_profile']
            if infer_profile:
                import timeit
                n = 2**11
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
            else:
                for _ in range(32):
                    cb_infer()
            reset_fn()
            if dof_states_padded:
                cb = [cb_pad_in, cb_infer, cb_pad_out]
            else:
                cb = cb_infer

        elif m == 'play':
            from unicon.general import cb_replay
            cb = cb_replay(
                **states_ctrls,
                states_input=states_input,
                **states_props,
                **states_extras,
                frames=loaded_rec,
                inds=dof_map,
                use_tqdm=True,
            )
        elif m == 'follow':

            pats = ['shoulder', 'elbow', 'wrist']
            follow_dof_names = [n for n in DOF_NAMES if any(p in n for p in pats)]
            # follow_dof_names = DOF_NAMES
            follow_inds = [DOF_NAMES.index(n) for n in follow_dof_names]
            follow_inds = list2slice(follow_inds)
            print('follow_dof_names', follow_dof_names)
            print('follow_inds', follow_inds)

            def cb_follow():
                states_q_ctrl[follow_inds] = states_q[follow_inds]

            cb = cb_follow
        elif m == 'hand_sample':
            hand_pt = 0
            hand_q_reset = hand_def.get('Q_RESET')
            HAND_DOF_NAMES = hand_def.get('DOF_NAMES')

            def cb_hand_sample():
                nonlocal hand_pt
                hand_pt += 1
                idx = (hand_pt // 500) % len(states_hand_q_ctrl)
                states_hand_q_ctrl[:] = hand_q_reset
                states_hand_q_ctrl[idx] = 3.14 * (np.sin(hand_pt * 0.02) + 1) * 0.5
                print('hand_pt', hand_pt, idx, states_hand_q_ctrl[idx], states_hand_q[idx])
                # print('states_hand_q_ctrl', states_hand_q_ctrl)
                # print('states_hand_q', states_hand_q)

            cb = cb_hand_sample

        cbs = cb if isinstance(cb, (list, tuple)) else [cb]
        print('mode', m, cbs)
        seq.extend(cbs)

    q_ctrl_default_mask = args['q_ctrl_default_mask']
    q_ctrl_mask = args['q_ctrl_mask']
    if q_ctrl_mask is not None:
        ctrl_mask = np.zeros(len(states_q_ctrl), dtype=bool)
        q_ctrl_mask = load_obj(q_ctrl_mask)
        q_ctrl_mask = [q_ctrl_mask] if not isinstance(q_ctrl_mask, (tuple, list)) else q_ctrl_mask
        inds = match_keys(q_ctrl_mask, DOF_NAMES)
        ctrl_mask[inds] = 1
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
    safety_q_margin = args['safety_q_margin'] if args['safety_q_margin'] else safety_q_margin
    if Q_CTRL_MIN is not None:
        q_ctrl_min = Q_CTRL_MIN * (1 - safety_q_margin) + Q_CTRL_MAX * safety_q_margin
        q_ctrl_max = Q_CTRL_MAX * (1 - safety_q_margin) + Q_CTRL_MIN * safety_q_margin
    ctx['q_ctrl_min'] = q_ctrl_min
    ctx['q_ctrl_max'] = q_ctrl_max
    clip_q_ctrl = args['clip_q_ctrl'] or env_clip_q_ctrl
    print('clip_q_ctrl', clip_q_ctrl)
    print('safety_q_margin', safety_q_margin)
    print('q_ctrl_min', q_ctrl_min)
    print('q_ctrl_max', q_ctrl_max)
    clip_q_ctrl_tau = args['clip_q_ctrl_tau']
    clip_q_ctrl_qd = args['clip_q_ctrl_qd']

    rec_output = args['rec_output']
    rec_post_send = args['rec_post_send']
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

    if num_steps is not None and num_steps > 0:
        cb = cb_timeout(num_steps)
        seq.append(cb)

    inner_input_stop = args['inner_input_stop']
    outer_stop_keys = ['BTN_TL', 'BTN_TR']
    if inner_input_stop and num_inputs > 0:
        seq.append(cb_wait_input(states_input=states_input, keys=outer_stop_keys[:-1]))
        outer_stop_keys = outer_stop_keys[1:]

    print('inner seq', seq)
    cb = None
    if len(seq):
        cb = seq[0] if len(seq) == 1 else cb_zip(*seq)
    chain = [cb]

    if args['system'] == 's':
        args['no_wrap'] = True
        args['no_wait_input'] = True
    wrapped = not args['no_wrap']
    if not NUM_DOFS:
        pass
    elif wrapped:
        # if rec_q_ctrl is not None:
        # q_reset[:] = rec_q_ctrl[0]
        q_lerp1 = q_reset.copy()
        if rec_q is not None:
            # q_lerp1[rep_dof_map] = frames[0]
            q_lerp1[rep_dof_map] = rec_q[0]
        lerp_steps = int((args['lerp_time'] + 0.01) // ctrl_dt)
        cb_lerp0 = None
        if Q_BOOT is not None:
            cb_lerp0 = cb_ctrl_q_from_target_lerp(
                **states_ctrls,
                states_q_target=q_boot,
                states_q=states_q,
                max_steps=lerp_steps,
                cycle=False,
                from_q_ctrl=False,
            )
        cb_lerp1 = cb_ctrl_q_from_target_lerp(
            **states_ctrls,
            states_q_target=q_lerp1,
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
            cb_lerp0,
            cb_lerp1,
            # cb_motor_q_lerp(max_steps=1e2, q_ctrl=q_reset),
        ]
        cb_post_ctrl = [
            cb_lerp2,
            cb_lerp3,
            # cb_motor_q_lerp(max_steps=lerp_steps, q_ctrl=q_boot),
        ]
        chain = cb_pre_ctrl + chain + cb_post_ctrl
        chain = [x for x in chain if x is not None]
    else:
        states_q_ctrl[:NUM_DOFS] = q_reset

    input_dev_type = args['input_dev_type']
    input_dev_type = None if input_dev_type == 'none' else input_dev_type
    wait_input = num_inputs > 0 and not args['no_wait_input']
    if wait_input:
        # key = 'BTN_TL'
        start_keys = ['BTN_SELECT', 'BTN_START']
        cb_idx = chain.index(cb)
        cb_wi = cb_wait_input(
            states_input=states_input,
            keys=start_keys,
            clicks=1,
            prompt=True,
        )
        chain.insert(cb_idx, cb_wi)

    print('chain', chain)
    cb = cb_chain(*chain) if len(chain) > 1 else chain[0]

    fixed_lat = args['fixed_lat']
    if fixed_lat:
        cb = cb_fixed_lat(cb, fixed_lat=fixed_lat)

    seq = []
    seq.append(cb)

    kp_r = args['kp_ratio']
    kd_r = args['kd_ratio']
    tau_limit_r = args['tau_limit_ratio']
    print('kp_r', kp_r)
    print('kd_r', kd_r)
    kp = np.zeros(NUM_DOFS) if KP is None else KP.copy()
    kd = np.zeros(NUM_DOFS) if KD is None else KD.copy()
    tau_limit = np.zeros(NUM_DOFS) if TAU_LIMIT is None else TAU_LIMIT.copy()
    use_env_pd = args['use_env_pd']
    if env_cfg is not None and use_env_pd:
        nxs = []
        for x in [env_kps, env_kds, env_torque_limits]:
            if x is None:
                nx = None
            else:
                nx = {dof_names_map.get(k, k): v for k, v in x.items()}
            nxs.append(nx)
        env_kps, env_kds, env_torque_limits = nxs
        print('env_kps', env_kps)
        print('env_kds', env_kds)
        print('env_torque_limits', env_torque_limits)
        kp = [([v for k, v in env_kps.items() if k in n] or [kp[i]])[0] for i, n in enumerate(DOF_NAMES)]
        kd = [([v for k, v in env_kds.items() if k in n] or [kd[i]])[0] for i, n in enumerate(DOF_NAMES)]
        tau_limit = [
            ([v for k, v in env_torque_limits.items() if k in n] or [tau_limit[i]])[0] for i, n in enumerate(DOF_NAMES)
        ]
        assert len(kp) == len(kd) == NUM_DOFS
        kp = np.array(kp, dtype=np.float64)
        kd = np.array(kd, dtype=np.float64)
        tau_limit = np.array(tau_limit, dtype=np.float64)
    args_kp = args['kp']
    args_kd = args['kd']
    for x, ax in [[kp, args_kp], [kd, args_kd]]:
        if ax is not None:
            ax = load_obj(ax)
        if isinstance(ax, dict):
            for k, v in ax.items():
                inds = match_keys(k, DOF_NAMES)
                x[inds] = v
        elif isinstance(ax, list):
            x[:] = ax
    if kp is not None and len(kp):
        kp = kp if isinstance(kp, np.ndarray) else np.array(kp, dtype=np.float64)
        kp[:] *= kp_r
    if kd is not None and len(kd):
        kd = kd if isinstance(kd, np.ndarray) else np.array(kd, dtype=np.float64)
        kd[:] *= kd_r
    if tau_limit is not None and len(tau_limit):
        tau_limit = tau_limit if isinstance(tau_limit, np.ndarray) else np.array(tau_limit, dtype=np.float64)
        tau_limit[:] *= tau_limit_r
    print('kp', kp if kp is None else pp_arr(kp))
    print('kd', kd if kd is None else pp_arr(kd))
    print('tau_limit', tau_limit if tau_limit is None else pp_arr(tau_limit))
    ctx['KP'] = kp
    ctx['KD'] = kd
    ctx['TAU_LIMIT'] = tau_limit

    states_props_sys = states_props
    states_ctrls_sys = states_ctrls

    q_transform = args['q_transform']
    if q_transform is not None:
        q_transform = load_obj(q_transform)
        qtf_recv = False
        print('q_transform', q_transform)
        # states q to sys q
        if isinstance(q_transform, dict):
            w = q_transform.get('w')
            b = q_transform.get('b')
            qtf_recv = q_transform.get('qtf_recv', qtf_recv)
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
                    y[match_keys(k, DOF_NAMES)] = v
            else:
                y = x
            wb.append(y)

        qtf_w, qtf_b = wb
        qtf_w_inv = 1 / qtf_w
        print('qtf_w', qtf_w)
        print('qtf_b', qtf_b)

        dtype = states_q_ctrl.dtype
        # states_q_ctrl_sys = np.zeros(NUM_DOFS, dtype=dtype)
        # states_q_sys = np.zeros(NUM_DOFS, dtype=dtype)
        states_q_ctrl_sys = states_q_ctrl
        states_q_sys = states_q
        states_props_sys = states_props.copy()
        states_props_sys['states_q'] = states_q_sys
        states_ctrls_sys = states_ctrls.copy()
        states_ctrls_sys['states_q_ctrl'] = states_q_ctrl_sys

        def cb_qtf_recv():
            states_q[:] = (states_q_sys - qtf_b) * qtf_w_inv
            states_qd[:] = states_qd * qtf_w_inv

        def cb_qtf_send():
            states_q_ctrl_sys[:] = states_q_ctrl * qtf_w + qtf_b

    system_type = args['system']
    system_type = {k[0]: k for k in ['sims', 'none']}.get(system_type, system_type)
    print('system', system_type)

    sys_kwds = {
        'kp': kp,
        'kd': kd,
    }
    sys_kwds.update(load_obj(args['system_kwargs'] or '') or {})

    states_extras_sys = dict(
        states_q_tau=states_q_tau,
        states_q_cur=states_q_cur,
        states_q_temp=states_q_temp,
        states_lin_vel=states_lin_vel,
        states_lin_acc=states_lin_acc,
        states_pos=states_pos,
        states_input=states_input,
    )
    states_sys = {}
    states_sys.update(states_props_sys)
    states_sys.update(states_ctrls_sys)
    states_sys.update(states_extras_sys)
    cb_recv, cb_send, cb_close = None, None, None
    if system_type == 'sims':
        sims_config = args['sims_config']
        if sims_config is None:
            sims_type = args['sims_type']
            init_z = robot_def.get('INIT_Z', 1.)
            init_z = init_z if args['sims_init_z'] is None else args['sims_init_z']
            urdf_path = robot_def.get('URDF')
            system_config = {
                'type': sims_type,
                'urdf_path': urdf_path,
                'default_root_states': [0., 0., init_z, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                'decimation': args['sims_decimation'],
                # 'sim_device': 'cuda',
            }
            if sims_type == 'sims.systems.ig':
                asset_options = robot_def.get('ASSET_OPTIONS')
                if asset_options is not None:
                    system_config.update({'asset_options': asset_options})
            print('system_config', system_config)
        else:
            system_config = load_obj(sims_config)
        convert_mjcf = True
        if sims_type in ['sims.systems.mujoco', 'sims.systems.brax']:
            xml_path = robot_def.get('MJCF')
            if convert_mjcf and system_config.get('xml_path') is None:
                import tempfile
                from unicon.utils.urdf2mjcf import urdf2mjcf
                xml_file = tempfile.NamedTemporaryFile(mode='w', delete=True)
                xml_path = xml_file.name
                no_collision_mesh = False
                use_sensor = False
                urdf2mjcf(
                    urdf_path=urdf_path,
                    mjcf_path=xml_path,
                    no_collision_mesh=no_collision_mesh,
                    copy_meshes=True,
                    no_frc_limit=True,
                    use_sensor=use_sensor,
                )
            system_config['xml_path'] = xml_path
        sim_dof_names = DOF_NAMES
        system_config['realtime'] = False
        fix_base_link = system_config.get('fix_base_link', False) or args['sims_fixed_base']
        fix_base_link = fix_base_link or 'sample' in modes
        system_config['fix_base_link'] = fix_base_link
        sims_def_dof_pos = system_config.get('default_dof_pos', {})
        system_config['default_dof_pos'] = sims_def_dof_pos
        if rec_q is not None and args['sims_fixed_base']:
            # init_q = rec_q_ctrl[0]
            init_q = rec_q[0]
            def_dof_pos = {n: init_q[i] for i, n in enumerate(sim_dof_names)}
            system_config['default_dof_pos'] = def_dof_pos
            print('sims init_q', def_dof_pos)
        if Q_BOOT is not None and wrapped:
            for i, n in enumerate(sim_dof_names):
                sims_def_dof_pos[n] = Q_BOOT[i]
        elif not wrapped:
            for i, n in enumerate(sim_dof_names):
                sims_def_dof_pos[n] = q_reset[i]
        default_root_states = system_config.get('default_root_states')
        if fix_base_link and default_root_states is not None:
            default_root_states[2] += 0.5
        if args['sims_headless']:
            system_config['headless'] = True
        if args['sims_compute_torque']:
            system_config['compute_torque'] = True
        system_config['compute_torque'] = system_config.get('compute_torque', False)
        if use_env_pd:
            system_config['Kp'] = env_kps
            system_config['Kd'] = env_kds
            if env_torque_limits is not None:
                system_config['torque_limits'] = env_torque_limits
        system_config['torque_limits'] = env_torque_limits
        print('env_torque_limits', env_torque_limits)
        sims_use_kpkd = ('Kp' not in system_config) or args['sims_use_kpd']
        if sims_use_kpkd:
            system_config['Kp'] = {k: v for k, v in zip(sim_dof_names, kp)}
            system_config['Kd'] = {k: v for k, v in zip(sim_dof_names, kd)}
            system_config['torque_limits'] = {k: v for k, v in zip(sim_dof_names, tau_limit)}
        system_config['dt'] = dt
        sims_override = args['sims_override']
        sims_override = load_obj(sims_override or '') or {}
        obj_update(system_config, sims_override)
        print('default_root_states', system_config.get('default_root_states'))
        print('default_dof_pos', system_config.get('default_dof_pos'))
        print('torque_limits', system_config.get('torque_limits'))
        # system_config.update(sims_override)
        wrapper_config = []
        if args['sims_auto_reset']:
            wrapper_config.append('sims.wrappers.autoreset:AutoResetWrapper')
        sims_wrapper_config = args['sims_wrapper_config']
        if sims_wrapper_config is not None:
            wrapper_config.extend([load_obj(c) for c in sims_wrapper_config])
        sims_kwds = dict(
            system_config=system_config,
            wrapper_config=wrapper_config,
        )
        sys_kwds['sims_kwds'] = sims_kwds
        sys_kwds['dof_names'] = sim_dof_names
        sys_kwds.pop('kp', None)
        sys_kwds.pop('kd', None)
    if system_type == 'none':
        pass
    else:
        print('importing system', system_type)
        sys_cls = import_obj(f'{system_type}:cb_{system_type}_recv_send_close', default_mod_prefix='unicon.systems')
        cb_recv, cb_send, cb_close = autowired(sys_cls, states=states_sys)(**sys_kwds,)
        cb_close_seq.append(cb_close)

        dry_run = args['dry']
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
                    'rec_dt': dt if rec_post_send else ctrl_dt,
                }
                data.update(ctx)
                data.update({k: v[:rec_len] for k, v in rec.items() if isinstance(v, np.ndarray)})
                _, ext = os.path.splitext(save_path)
                save_type = ext[1:]
                print('save_type', save_type)
                print('save_path', save_path)
                mod = import_obj(save_type, default_mod_prefix='unicon.data', prefer_mod=True)
                mod.save(save_path, data)
        if not reuse:
            states_destroy(force=True)
        for cb in cb_close_seq:
            if cb is not None:
                cb()
        exit(0)

    verify_recv = args['verify_recv']

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
            return close_fn

    pre_recv = args['pre_recv']
    if pre_recv and cb_recv is not None:
        print('pre_recv', pre_recv)
        for _ in range(pre_recv):
            cb_recv()
    pre_send = args['pre_send']
    if pre_send and cb_send is not None:
        print('pre_send', pre_send)
        for _ in range(pre_send):
            cb_send()

    if ctrl_dt > dt:
        intv = int(ctrl_dt // dt)
        print('ctrl_intv', ctrl_dt, dt, intv)
        cb = seq[0] if len(seq) == 1 else cb_zip(*seq)
        cb = cb_if(cb, pred=lambda pt: pt % intv == 0)
        seq = [cb]
    elif ctrl_dt < dt:
        raise ValueError('ctrl_dt < dt', ctrl_dt, dt)

    cb_ctrl_tau = args['cb_ctrl_tau']
    if cb_ctrl_tau is not None:
        print('cb_ctrl_tau', cb_ctrl_tau)
        cb_ctrl_tau_cls = import_obj(cb_ctrl_tau, default_name_prefix='cb_ctrl_tau', default_mod_prefix='unicon.ctrl')
        seq.extend([
            cb_ctrl_tau_cls(
                **states_ctrls,
                states_q=states_q,
                states_qd=states_qd,
            ),
        ])

    transforms = args['transforms']
    for i, tf in enumerate(transforms):
        tf_kwds = load_obj(tf)
        tf_type = tf_kwds.pop('type')
        tf_cls = import_obj(tf_type, default_name_prefix='cb_tf', default_mod_prefix='unicon.transforms')
        cb = autowired(tf_cls)(**tf_kwds)
        seq.insert(i, cb)

    imu_transform = args['imu_transform']
    if imu_transform is not None:
        states_quat = states_props['states_quat']
        states_rpy = states_props['states_rpy']
        states_ang_vel = states_props['states_ang_vel']
        print('imu_transform', imu_transform)
        imu_transform = load_obj(imu_transform)
        if isinstance(imu_transform, dict):
            tfs = [imu_transform.get(k) for k in ['rpy', 'ang_vel', 'quat']]
        elif isinstance(imu_transform, (list, tuple)):
            tfs = [imu_transform[i] if len(imu_transform) > i else None for i in range(3)]
        else:
            raise ValueError('imu_transform')
        rpy_tf, ang_vel_tf, quat_tf = tfs

        print('rpy_tf', rpy_tf)
        print('ang_vel_tf', ang_vel_tf)
        print('quat_tf', quat_tf)
        rpy_w, rpy_b = rpy_tf if rpy_tf is not None else (1, 0)
        ang_vel_w, ang_vel_b = ang_vel_tf if ang_vel_tf is not None else (1, 0)
        wb = rpy_w, rpy_b, ang_vel_w, ang_vel_b
        wb = [(x if isinstance(x, (float, int)) else np.array(x)) for x in wb]
        rpy_w, rpy_b, ang_vel_w, ang_vel_b = wb
        if quat_tf is not None:
            from unicon.utils import quat_mul_np, quat2rpy_np3, quat2mat_np
            qt = np.array(quat_tf)
            mat_t = quat2mat_np(qt).T
            print('mat_t', mat_t)

        def cb_itf_recv():
            if rpy_tf is not None:
                states_rpy[:] = states_rpy * rpy_w + rpy_b
            if ang_vel_tf is not None:
                states_ang_vel[:] = states_ang_vel * ang_vel_w + ang_vel_b
            if quat_tf is not None:
                states_quat[:] = quat_mul_np(states_quat, qt)
                states_rpy[:] = quat2rpy_np3(states_quat)
                # states_rpy[:] = np.matmul(mat_t, states_rpy)
                states_ang_vel[:] = np.matmul(mat_t, states_ang_vel)

        seq.insert(0, cb_itf_recv)

    post_transforms = args['post_transforms']
    for i, tf in enumerate(post_transforms):
        tf_kwds = load_obj(tf)
        tf_type = tf_kwds.pop('type')
        tf_cls = import_obj(tf_type, default_name_prefix='cb_tf', default_mod_prefix='unicon.transforms')
        cb = autowired(tf_cls)(**tf_kwds)
        seq.append(cb)

    if q_transform is not None:
        if qtf_recv:
            cb_qtf_recv()
            seq.insert(0, cb_qtf_recv)
        cb_qtf_send()
        seq.append(cb_qtf_send)

    if clip_q_ctrl or clip_q_ctrl_tau or clip_q_ctrl_qd:
        from unicon.ctrl import cb_ctrl_q_clip
        cb = autowired(cb_ctrl_q_clip)(
            clip_q_limit=clip_q_ctrl,
            clip_qd_limit=clip_q_ctrl_qd,
            clip_tau_limit=clip_q_ctrl_tau,
        )
        seq.append(cb)

        if HAND_NUM_DOFS:
            cb = cb_ctrl_q_clip(
                states_q_ctrl=states_hand_q_ctrl,
                states_q=states_hand_q,
                clip_q_limit=True,
                clip_qd_limit=False,
                clip_tau_limit=False,
                q_ctrl_min=hand_def['Q_CTRL_MIN'],
                q_ctrl_max=hand_def['Q_CTRL_MAX'],
            )
            seq.append(cb)

    clear_input = True
    if clear_input and states_input is not None:

        if states_cmd is None:

            def cb_input_clear():
                states_input[:] = 0.
        else:

            def cb_input_clear():
                states_input[:] = 0.
                states_cmd[:] = 0.

        seq.extend([
            cb_input_clear,
        ])

    seq = [cb_recv] + seq + [cb_send]
    seq = [c for c in seq if c is not None]

    fixed_wait = args['fixed_wait']
    if fixed_wait is not None:
        from unicon.general import cb_timer_set, cb_timer_wait
        cb0 = cb_timer_set()
        cb1 = cb_timer_wait(wait=max(0, fixed_wait), stats=(True if fixed_wait <= 0 else False))
        seq.insert(0, cb0)
        pos = -1 if fixed_wait >= 0 else len(seq)
        seq.insert(pos, cb1)

    if num_inputs and input_dev_type is not None:
        fallback_input_types = ['term']
        cb_input_dev = None
        for t in [input_dev_type] + fallback_input_types:
            input_dev_kwds = load_obj(t)
            input_dev_kwds = {'type': input_dev_kwds} if isinstance(input_dev_kwds, str) else input_dev_kwds
            input_dev_type = input_dev_kwds.pop('type')
            print('input_dev_type', input_dev_type)
            cb_input_dev_cls = import_obj(
                input_dev_type,
                default_name_prefix='cb_input',
                default_mod_prefix='unicon.inputs',
            )
            try:
                cb_input_dev = autowired(cb_input_dev_cls)(**input_dev_kwds)
            except Exception as e:
                print(e)
                continue
            if cb_input_dev is not None:
                break
        if cb_input_dev is not None:
            seq.extend([
                cb_input_dev,
            ])

    cmd_states = dict()
    cmds = args['cmd']
    cmds = [] if cmds is None else cmds
    for cmd in cmds:
        if cmd == 'none':
            continue
        cmd_kwds = load_obj(cmd)
        cmd_kwds = {'type': cmd_kwds} if isinstance(cmd_kwds, str) else cmd_kwds
        cmd_type = cmd_kwds.pop('type')
        print('cmd_type', cmd_type)
        cb_cmd_cls = import_obj(cmd_type, default_name_prefix='cb_cmd', default_mod_prefix='unicon.cmd')
        kwds = {}
        ccv = args['cmd_const_v']
        if ccv is not None:
            ccv = load_obj(ccv)
            ccv = [ccv] if isinstance(ccv, (float, int)) else ccv
            kwds['cmd'] = ccv
        if cmd_type == 'replay':
            rec_cmd = loaded_rec.get('states_cmd')
            kwds['frames'] = rec_cmd
        kwds.update(cmd_kwds)
        if cb_cmd_cls is not None:
            cb_cmd = autowired(cb_cmd_cls, states=cmd_states)(input_keys=input_keys, **kwds)
            cb_cmd()
            seq.append(cb_cmd)

    print('seq before io', seq)
    inputs = args['inputs']
    outputs = args['outputs']
    inouts = args['inouts']
    n_cmd_cbs = len(cmds)
    def_insert_pos = -n_cmd_cbs if n_cmd_cbs else None
    def_insert_pos = None
    for output_kwds in outputs:
        output_kwds = load_obj(output_kwds)
        output_kwds = {'type': output_kwds} if isinstance(output_kwds, str) else output_kwds
        output_type = output_kwds.pop('type')
        insert_pos = output_kwds.pop('_pos', def_insert_pos)
        insert_pos = len(seq) if insert_pos is None else insert_pos
        print('output_type', output_type, insert_pos)
        cb_output_cls = import_obj(output_type, default_name_prefix='cb_send', default_mod_prefix='unicon.io')
        cb = autowired(cb_output_cls)(**output_kwds)
        seq.insert(insert_pos, cb)
    for input_kwds in inputs:
        input_kwds = load_obj(input_kwds)
        input_kwds = {'type': input_kwds} if isinstance(input_kwds, str) else input_kwds
        input_type = input_kwds.pop('type')
        insert_pos = input_kwds.pop('_pos', def_insert_pos)
        insert_pos = len(seq) if insert_pos is None else insert_pos
        print('input_type', input_type, insert_pos)
        cb_input_cls = import_obj(input_type, default_name_prefix='cb_recv', default_mod_prefix='unicon.io')
        cb = autowired(cb_input_cls)(**input_kwds)
        seq.insert(insert_pos, cb)
    for specs in inouts:
        specs = load_obj(specs)
        specs = {'type': specs} if isinstance(specs, str) else specs
        cb_type = specs.pop('type')
        insert_pos = specs.pop('_pos', def_insert_pos)
        insert_pos = len(seq) if insert_pos is None else insert_pos
        print('io_type', cb_type, insert_pos)
        cb_io_cls = import_obj(cb_type)
        cb = autowired(cb_io_cls)(**specs)
        cb_recv, cb_send, cb_close = cb
        seq.insert(insert_pos, cb_send)
        seq.insert(insert_pos, cb_recv)
        cb_close_seq.append(cb_close)

    if rec_post_send and rec_output is not None:
        seq.append(_cb_rec)

    if verbose is not None:
        seq.append(cb_print(intvs=load_obj(verbose)))

    outer_input_stop = not args['no_outer_input_stop']
    if outer_input_stop and num_inputs > 0:
        seq.append(cb_wait_input(states_input=states_input, keys=outer_stop_keys))

    if args['safety_ctrl']:
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
            tau_limit=tau_limit,
            kp=kp,
            kd=kd,
            power_limit=1200,
            power_max_breaks=2**2,
            power_halt=1440,
        )
        seq.extend([
            cb_sft_ctrl,
        ])

    if args['safety_states']:
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

    if args['close']:
        return close_fn

    states_quat = states_get('quat')
    if states_quat is not None:
        states_quat[:] = [0, 0, 0, 1.]
    if HAND_NUM_DOFS:
        hand_q_ctrl_init = hand_def['Q_RESET']
        print('hand_q_ctrl_init', hand_q_ctrl_init)
        if hand_q_ctrl_init is not None:
            states_hand_q_ctrl[:] = hand_q_ctrl_init

    loop_kwds = {}
    loop_sleep_block = args['loop_sleep_block']
    if loop_sleep_block:
        loop_kwds['sleep_fn'] = 'sleep_block'
    loop_dt_ofs = args['loop_dt_ofs']
    if loop_dt_ofs is not None:
        loop_kwds['dt_ofs'] = loop_dt_ofs

    sync_time_phase = True
    if sync_time_phase:
        from unicon.general import cb_wait_datetime
        cb_phase = cb_wait_datetime(date='.000001', delta='@1')
        while not cb_phase(): pass
        # cb = cb_chain(cb_timephase, cb)

    loop_dt = args['loop_dt']
    loop_dt = dt if loop_dt is None else loop_dt
    if not fast_fwd:
        cb = cb_loop_timed(cb, dt=loop_dt, **loop_kwds)
    try:
        while True:
            ret = cb()
            if ret is True:
                break
    except KeyboardInterrupt:
        import traceback
        traceback.print_exc()
        print('interrupted')

    return close_fn


if __name__ == '__main__':
    close_fn = run()
    close_fn()
