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
    parser.add_argument('-s', '--system', default='fake')
    parser.add_argument('-ro', '--rec_output', default=None)
    parser.add_argument('-it', '--infer_type', default='gr1')
    parser.add_argument('-di', '--input_dev_type', default='js')
    parser.add_argument('-pt', '--policy_type', default='none')
    parser.add_argument('-mt', '--model_type', default=None)
    parser.add_argument('-cmd', '--cmd', default='none')
    parser.add_argument('-ccv', '--cmd_const_v', default=None)
    parser.add_argument('-iks', '--input_keys', default=None)
    parser.add_argument('-d', '--dry', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-nowrp', '--no_wrap', action='store_true')
    parser.add_argument('-w', '--wait', type=float, default=0)
    parser.add_argument('-kp', '--kp', default=None)
    parser.add_argument('-kd', '--kd', default=None)
    parser.add_argument('-kpr', '--kp_ratio', type=float, default=1.0)
    parser.add_argument('-kdr', '--kd_ratio', type=float, default=1.0)
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
    parser.add_argument('-wi', '--wait_input', action='store_true')
    parser.add_argument('-ecp', '--env_cfg_path', default=None)
    parser.add_argument('-eco', '--env_cfg_override', default=None)
    parser.add_argument('-sst', '--sims_type', default='sims.systems.ig')
    parser.add_argument('-ssc', '--sims_config', default=None)
    parser.add_argument('-sso', '--sims_override', default=None)
    parser.add_argument('-ssar', '--sims_auto_reset', action='store_true')
    parser.add_argument('-sswc', '--sims_wrapper_config', action='append')
    parser.add_argument('-ssd2', '--sims_dof_names_2', action='store_true')
    parser.add_argument('-ssh', '--sims_headless', action='store_true')
    parser.add_argument('-ssfb', '--sims_fixed_base', action='store_true')
    parser.add_argument('-ssdc', '--sims_decimation', type=int, default=10)
    parser.add_argument('-ssiz', '--sims_init_z', type=float, default=None)
    parser.add_argument('-ssct', '--sims_compute_torque', action='store_true')
    parser.add_argument('-ssuk', '--sims_use_kpd', action='store_true')
    parser.add_argument('-skwds', '--system_kwargs', default=None)
    parser.add_argument('-cqc', '--clip_q_ctrl', action='store_true')
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
    parser.add_argument('-ff', '--fast', action='store_true')
    parser.add_argument('-sqm', '--safety_q_margin', type=float, default=0.)
    parser.add_argument('-iis', '--inner_input_stop', action='store_true')
    parser.add_argument('-nois', '--no_outer_input_stop', action='store_true')
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
    parser.add_argument('-sxe2', '--states_x_extras2', action='store_true')
    parser.add_argument('-scs', '--states_custom_specs', default=None)
    parser.add_argument('-tc', '--tau_ctrl', action='store_true')
    parser.add_argument('-cct', '--cb_ctrl_tau', default=None)
    parser.add_argument('-cdt', '--ctrl_dt', type=float, default=None)
    parser.add_argument('-ilr', '--infer_load_run', default=None)
    parser.add_argument('-rcps', '--rec_post_send', action='store_true')
    parser.add_argument('-i', '--inputs', default=[], action='append')
    parser.add_argument('-o', '--outputs', default=[], action='append')
    parser.add_argument('-uspd', '--use_sim_pd', action='store_true')
    parser.add_argument('-dn2', '--dof_names_2', action='store_true')
    parser.add_argument('-dnstd', '--dof_names_std', action='store_true')
    parser.add_argument('-dnr', '--dof_names_remap', default=None)
    parser.add_argument('-dns', '--dof_names_sub', default=None)
    parser.add_argument('-rp', '--rec_path', default=None)
    parser.add_argument('-rtp', '--rec_type', default=None)
    parser.add_argument('-rpt', '--rec_pt', type=int, default=None)
    parser.add_argument('-rkwds', '--rec_kwds', default=None)
    parser.add_argument('-ncmd', '--num_commands', type=int, default=3)
    parser.add_argument('-nice', '--nice', type=int, default=None)
    parser.add_argument('-su', '--sudo', action='store_true')
    parser.add_argument('-qtf', '--q_transform', default=None)
    parser.add_argument('-itf', '--imu_transform', default=None)
    parser.add_argument('-tf', '--transforms', default=[], action='append')
    parser.add_argument('-ptf', '--post_transforms', default=[], action='append')
    parser.add_argument('-sie', '--states_infer_extras', action='store_true')
    parser.add_argument('-nqb', '--no_q_boot', action='store_true')
    parser.add_argument('-qru', '--q_reset_update', default=None)
    parser.add_argument('-uss', '--use_secondary_sensor', action='store_true')
    args, _ = parser.parse_known_args()
    return args


def run(args=None):
    if args is None:
        args = get_args()
        args = vars(args)
    try:
        import isaacgym
        del isaacgym
    except ImportError:
        pass
    from unicon.states import states_init, states_news, states_new, states_get, states_destroy, autowired
    from unicon.general import cb_chain, cb_noop, cb_print, cb_prod, cb_zip, cb_if, \
        cb_timeout, cb_fixed_lat, cb_wait_input, cb_replay
    from unicon.utils import set_nice2, set_cpu_affinity2, list2slice, pp_arr, set_seed, \
        import_obj, parse_robot_def, load_obj, match_keys, obj_update, get_ctx
    from unicon.ctrl import cb_ctrl_q_from_target_lerp

    if args['sudo']:
        os.system('sudo -v')

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
        robot_def = parse_robot_def(robot_def)
    KP = robot_def.get('KP', [])
    KD = robot_def.get('KD', [])
    Q_CTRL_MIN = robot_def.get('Q_CTRL_MIN', None)
    Q_CTRL_MAX = robot_def.get('Q_CTRL_MAX', None)
    DOF_NAMES = robot_def.get('DOF_NAMES', [])
    NUM_DOFS = robot_def.get('NUM_DOFS', 0)
    DOF_MAPS = robot_def.get('DOF_MAPS', None)
    TAU_LIMIT = robot_def.get('TAU_LIMIT', None)
    QD_LIMIT = robot_def.get('QD_LIMIT', None)
    Q_BOOT = robot_def.get('Q_BOOT', None)
    Q_RESET = robot_def.get('Q_RESET', None)
    DOF_NAMES_2 = robot_def.get('DOF_NAMES_2', None)
    DOF_NAMES_STD = robot_def.get('DOF_NAMES_STD', None)
    LINK_NAMES = robot_def.get('LINK_NAMES', None)

    if args['no_q_boot']:
        Q_BOOT = None

    q_reset = np.zeros(NUM_DOFS) if Q_RESET is None else Q_RESET

    if args['dof_names_2']:
        DOF_NAMES = DOF_NAMES_2

    dt = args['dt']
    ctrl_dt = args['ctrl_dt']
    ctx = get_ctx()
    ctx.update({
        'args': args,
        'argv': argv,
        'robot_def': robot_def,
        'dof_names': DOF_NAMES,
        'hostname': __import__('platform').node(),
    })

    dof_names = None
    default_dof_pos = None

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
    if infer_load_run is not None:
        _default_infer_root = os.environ.get('UNICON_INFER_ROOT')
        print('infer_load_run', infer_load_run)
        print('_default_infer_root', _default_infer_root)
        if _default_infer_root is None:
            from unicon.utils import find
            root = find(root='..', wholename=f'*{infer_load_run}')
            if root is None:
                root = find(root='~', wholename=f'*{infer_load_run}')
            assert root is not None
            root = root[0]
        else:
            root = os.path.join(_default_infer_root, infer_load_run)

        if os.path.isfile(root):
            root = os.path.dirname(root)
        
        print('infer root', root)
        model_file_pats = ['policy', 'trace']
        model_file = None
        for r, _, fs in os.walk(root):
            for f in fs:
                if any([p in f for p in model_file_pats]):
                    model_file = os.path.join(r, f)
                    break
        print('model_file', model_file)
        if model_file is None:
            raise ValueError('model_file not found')
        infer_model_path = model_file
        from unicon.utils import md5sum
        infer_model_md5sum = md5sum(infer_model_path)
        ctx['infer_model_md5sum'] = infer_model_md5sum
        print('infer_load_run', infer_load_run, infer_model_path, infer_model_md5sum)
    if env_cfg_path is None and infer_model_path is not None:
        for ext in ['.json', '.yaml']:
            env_cfg_path = os.path.join(os.path.dirname(infer_model_path), 'env_cfg' + ext)
            if os.path.exists(env_cfg_path):
                break

    sim_kps = None
    sim_kds = None
    sim_torque_limits = None
    init_joint_angles = None
    env_cfg = None
    env_cfg_args = None
    env_num_commands = None
    env_command_keys = None
    if env_cfg_path is not None and os.path.exists(env_cfg_path):
        print('env_cfg_path', env_cfg_path)
        env_cfg = load_obj(env_cfg_path)
    if env_cfg is not None:
        env_cfg_override = args['env_cfg_override']
        if env_cfg_override is not None:
            env_cfg_override = load_obj(env_cfg_override)
            obj_update(env_cfg, env_cfg_override)
        env_cfg_env = env_cfg['env']
        num_acts = env_cfg_env['num_actions']
        num_obs = env_cfg_env.get('num_partial_obs', env_cfg_env.get('num_observations'))
        env_dof_names = env_cfg.get('dof_names')
        print('env dof_names', env_dof_names)
        if env_dof_names is not None:
            dof_names = env_dof_names
        init_joint_angles = env_cfg['init_state']['default_joint_angles']
        sim_kps = env_cfg['control']['stiffness']
        sim_kds = env_cfg['control']['damping']
        sim_torque_limits = env_cfg['control'].get('torque_limits', None)
        if any(['mix' in x for x in env_cfg.get('argv', [])]):
            NAME = robot_def['NAME']
            params = env_cfg['robot_params'].get(NAME, {})
            init_joint_angles = params.get('default_joint_angles', None)
            print('mix env', NAME, init_joint_angles)
            sim_kps = params.get('stiffness', None)
            sim_kds = params.get('damping', None)
            sim_torque_limits = params.get('torque_limits', None)
        ctrl_dt = env_cfg['sim']['dt'] * env_cfg['control']['decimation']
        env_cfg_args = env_cfg.get('cmd_args', env_cfg.get('argv', None))
        print('num_acts', num_acts, 'num_obs', num_obs, 'num_dofs', len(dof_names))
        print('env_cfg_args', env_cfg_args)

        env_num_commands = env_cfg['commands']['num_commands']
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
        env_command_keys = [x for x in env_command_keys if x is not None]
        print('env_num_commands', env_num_commands)
        print('env_command_keys', env_command_keys)
        assert len(env_command_keys) == env_num_commands
        ranges = env_cfg['commands']['ranges']
        print('env_cfg ranges', ranges)
        def_ranges = {
            'gait_frequency': [env_cfg['commands'].get('gait_frequency')] * 2,
            'foot_swing_height': [env_cfg['commands'].get('foot_swing_height')] * 2,
            'phase': [0.5] * 2,
            'duration': [env_cfg['commands'].get('duty_cycle')] * 2,
        }
        env_command_ranges = [ranges.get(k, def_ranges.get(k)) for k in env_command_keys]
        print('env_command_ranges', env_command_ranges)
        ctx.update({
            'env_command_keys': env_command_keys,
            'env_command_ranges': env_command_ranges,
        })

    ctrl_dt = dt if ctrl_dt is None else ctrl_dt
    dt = ctrl_dt if dt is None else dt
    print('dt', dt, 'ctrl_dt', ctrl_dt)
    assert ctrl_dt is not None and dt is not None
    ctx.update({
        'dt': dt,
        'ctrl_dt': ctrl_dt,
    })

    tau_ctrl = args['tau_ctrl']
    specs = []
    if NUM_DOFS:
        assert NUM_DOFS == len(DOF_NAMES)
        specs.extend([
            ('rpy', 3),
            ('ang_vel', 3),
            ('quat', 4),
            ('q', NUM_DOFS),
            ('qd', NUM_DOFS),
            ('q_ctrl', NUM_DOFS),
            ('q_target', NUM_DOFS),
        ])
    if tau_ctrl:
        specs.append(('tau_ctrl', NUM_DOFS))
    q_extras = args['states_q_extras']
    if q_extras:
        specs.extend([
            ('q_tau', NUM_DOFS),
            ('q_cur', NUM_DOFS),
            ('q_temp', NUM_DOFS),
        ])
    x_extras = args['states_x_extras']
    x_extras2 = args['states_x_extras2']
    num_links = 0 if LINK_NAMES is None else len(LINK_NAMES)
    if x_extras or x_extras2:
        specs.extend([
            ('pos', 3),
            ('lin_vel', 3),
            ('lin_acc', 3),
        ])
    if num_links and x_extras2:
        specs.extend([
            ('x', (num_links, 4, 4)),
            ('xd', (num_links, 6)),
        ])
    states_custom_specs = args['states_custom_specs']
    if states_custom_specs is not None:
        states_custom_specs = load_obj(states_custom_specs)
        print('states_custom_specs', states_custom_specs)
        specs.extend(states_custom_specs)
    states_news(specs)
    input_keys = args['input_keys']
    input_keys = import_obj('unicon.inputs:_default_input_keys') if input_keys is None else input_keys
    ctx['input_keys'] = input_keys
    num_inputs = len(input_keys)
    states_new('input', num_inputs)

    print('input_keys', input_keys)

    num_commands = args['num_commands']
    if env_num_commands is not None and num_commands <= 0:
        num_commands = env_num_commands - num_commands
    if num_commands > 0:
        states_new('cmd', num_commands)
    use_shm = False
    shm_clear = args['shm_clear']
    use_shm = args['shm'] or shm_clear
    load = use_shm and not shm_clear
    save = use_shm
    reuse = True
    states_init(use_shm=use_shm, save=save, load=load, reuse=reuse, clear=shm_clear)

    proc = None

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
    states_custom_extra_obs = {
        f'states_{k}': states_get(k) for k in robot_def.get('CUSTOM_EXTRA_OBS', [])
    }

    states_extras = {
        'states_cmd': states_cmd,
        'states_q_temp': states_q_temp,
        'states_q_tau': states_q_tau,
        'states_q_cur': states_q_cur,
        'states_pos': states_pos,
        'states_lin_vel': states_lin_vel,
        'states_lin_acc': states_lin_acc,
        'states_x': states_get('x'),
        'states_xd': states_get('xd'),
        'states_rpy2': states_get('rpy2'),
        'states_ang_vel2': states_get('ang_vel2'),
        'states_quat2': states_get('quat2'),
        'states_left_target': states_get('left_target'),
        'states_right_target': states_get('right_target'),
        'states_left_target_real_time': states_get('left_target_real_time'),
        'states_right_target_real_time': states_get('right_target_real_time'),
    }
    states_extras = {k: v for k, v in states_extras.items() if v is not None}

    states_infer_extras = {}
    inf_extras = args['states_infer_extras']
    if inf_extras and env_cfg is not None:
        print('num_acts', num_acts, 'num_obs', num_obs)
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

    if init_joint_angles is not None:
        default_dof_pos = np.array([init_joint_angles.get(n, 0.) for n in dof_names])
    if default_dof_pos is not None:
        q_reset[dof_map] = default_dof_pos[dof_src_map]

    q_reset_update = args['q_reset_update']
    if q_reset_update is not None:
        q_reset_update = load_obj(q_reset_update)
        if isinstance(q_reset_update, dict) and isinstance(list(q_reset_update.keys())[0], str):
            q_reset_update = {DOF_NAMES.index(k): v for k, v in q_reset_update.items()}
            obj_update(q_reset, q_reset_update)

    default_dof_pos = q_reset[dof_map] if default_dof_pos is None else default_dof_pos
    q_boot = q_reset if Q_BOOT is None else Q_BOOT

    print('default_dof_pos', default_dof_pos.tolist())
    print('q_reset', q_reset.tolist())
    print('q_boot', q_boot.tolist())

    if dof_states_padded:
        dtype = states_q_ctrl.dtype
        states_q_ctrl_inf = np.zeros(NUM_DOFS + 1, dtype=dtype)
        states_q_inf = np.zeros(NUM_DOFS + 1, dtype=dtype)
        states_qd_inf = np.zeros(NUM_DOFS + 1, dtype=dtype)

        def cb_pad_in():
            states_q_inf[:NUM_DOFS] = states_q
            states_qd_inf[:NUM_DOFS] = states_qd

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

    cb_close_list = []

    num_steps = args['num_steps']
    seq = []
    mode = args['mode'] or []
    print('mode', mode)
    all_modes = [
        'noop',
        'const',
        'sample',
        'replay',
        'infer',
        'teleop',
        'play',
        'follow',
    ]
    for m in mode:
        modes = {k: False for k in all_modes}
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
        elif modes['teleop']:
            from unicon.teleop import cb_teleop_q
            cb = cb_teleop_q(
                **states_ctrls,
                states_q=states_q,
                states_input=states_input,
            )
        elif modes['replay']:
            rep_dof_map = dof_map
            replay_states_key = args['replay_states_key']
            states_dest = states_get(replay_states_key)
            if loaded_rec is not None:
                replay_frame_key = args['replay_frame_key']
                rec_frames = loaded_rec.get(replay_frame_key)
                rec_dof_names = loaded_rec.get('dof_names', dof_names)
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
        elif modes['infer']:
            infer_type = args['infer_type']
            print('infer_type', infer_type)
            cb_infer_cls = import_obj(infer_type, default_name_prefix='cb_infer', default_mod_prefix='unicon.infer')

            policy_fn = None
            policy_reset_fn = None
            infer_device = args['infer_device']
            if infer_model_path is not None:
                print('infer_model_path', infer_model_path)
                from unicon.utils import load_model
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
            infer_states.update(states_custom_extra_obs)
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
            infer_profile = not args['infer_no_profile']
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
        elif modes['play']:
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
        elif modes['follow']:

            def cb_follow():
                states_q_ctrl[:] = states_q

            cb = cb_follow
        seq.append(cb)

    if dof_states_padded:
        seq.insert(0, cb_pad_in)
        seq.insert(-1, cb_pad_out)

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
    clip_q_ctrl = args['clip_q_ctrl'] or env_clip_q_ctrl
    print('clip_q_ctrl', clip_q_ctrl)
    print('safety_q_margin', safety_q_margin)
    print('q_ctrl_min', q_ctrl_min)
    print('q_ctrl_max', q_ctrl_max)

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
    if inner_input_stop:
        seq.append(cb_wait_input(states_input=states_input, keys=outer_stop_keys[:-1]))
        outer_stop_keys = outer_stop_keys[1:]

    print('inner seq', seq)
    cb = seq[0] if len(seq) == 1 else cb_zip(*seq)
    chain = [cb]

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
    wait_input = args['wait_input']
    if wait_input:
        key = 'BTN_TL'
        cb_idx = chain.index(cb)
        cb_wi = cb_wait_input(
            states_input=states_input,
            keys=[key],
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
    print('kp_r', kp_r)
    print('kd_r', kd_r)
    kp = KP.copy()
    kd = KD.copy()
    use_sim_pd = args['use_sim_pd']
    if env_cfg is not None and use_sim_pd:
        nxs = []
        for x in [sim_kps, sim_kds, sim_torque_limits]:
            if x is None:
                nx = None
            else:
                nx = {dof_names_map.get(k, k): v for k, v in x.items()}
            nxs.append(nx)
        sim_kps, sim_kds, sim_torque_limits = nxs
        print('sim_kps', sim_kps)
        print('sim_kds', sim_kds)
        print('sim_torque_limits', sim_torque_limits)
        kp = [([v for k, v in sim_kps.items() if k in n] or [kp[i]])[0] for i, n in enumerate(DOF_NAMES)]
        kd = [([v for k, v in sim_kds.items() if k in n] or [kd[i]])[0] for i, n in enumerate(DOF_NAMES)]
        assert len(kp) == len(kd) == NUM_DOFS
        kp = np.array(kp, dtype=np.float64)
        kd = np.array(kd, dtype=np.float64)
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
    if kp is not None:
        kp = kp if isinstance(kp, np.ndarray) else np.array(kp, dtype=np.float64)
        kd = kd if isinstance(kd, np.ndarray) else np.array(kd, dtype=np.float64)
        kp[:] *= kp_r
        kd[:] *= kd_r
    print('kp', kp if kp is None else pp_arr(kp))
    print('kd', kd if kd is None else pp_arr(kd))
    ctx['kp'] = kp
    ctx['kd'] = kd

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

    systems = {k: False for k in ['sims', 'none']}
    system_type = args['system']
    system_type = {k[0]: k for k in systems.keys()}.get(system_type, system_type)
    print('system', system_type)
    systems[system_type] = True

    sys_kwds = load_obj(args['system_kwargs'] or '') or {}
    sys_clip_q_ctrl = (not clip_q_ctrl)

    states_extras_sys = dict(
        states_q_tau=states_q_tau,
        states_q_cur=states_q_cur,
        states_q_temp=states_q_temp,
        states_lin_vel=states_lin_vel,
        states_lin_acc=states_lin_acc,
        states_pos=states_pos,
        states_input=states_input,
        states_left_target=states_extras['states_left_target'],
        states_right_target=states_extras['states_right_target'],
        states_left_target_real_time=states_extras['states_left_target_real_time'],
        states_right_target_real_time=states_extras['states_right_target_real_time'],
    )
    for k in states_custom_extra_obs:
        states_extras_sys[k] = states_custom_extra_obs[k]
    
    states_sys = {}
    states_sys.update(states_props_sys)
    states_sys.update(states_ctrls_sys)
    states_sys.update(states_extras_sys)
    cb_recv, cb_send, cb_close = None, None, None
    if systems['none']:
        pass
    elif systems['sims']:
        sims_config = args['sims_config']
        if sims_config is None:
            sims_type = args['sims_type']
            init_z = robot_def.get('INIT_Z', 1.)
            init_z = init_z if args['sims_init_z'] is None else args['sims_init_z']
            system_config = {
                'type': sims_type,
                'urdf_path': robot_def.get('URDF', robot_def.get('MJCF')),
                'default_root_states': [0., 0., init_z, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                'decimation': args['sims_decimation'],
                # 'sim_device': 'cuda',
            }
            if sims_type == 'sims.systems.ig':
                asset_options = robot_def.get('ASSET_OPTIONS')
                if asset_options is not None:
                    system_config.update({'asset_options': asset_options})
                system_config['use_secondary_imu_link'] = args['use_secondary_sensor']
                if args['use_secondary_sensor'] and isinstance(robot_def.get('USE_SENSOR'), list):
                    system_config['imu_name'] = robot_def.get('USE_SENSOR')[0]
                    system_config['use_imu_link'] = True
                    system_config['secondary_imu_name'] = robot_def.get('USE_SENSOR')[1]
                    system_config['use_secondary_imu_link'] = True
            if sims_type == 'sims.systems.mujoco':
                system_config['xml_path'] = robot_def.get('MJCF')
                system_config['use_sensor'] = robot_def.get('USE_SENSOR', False)
                system_config['use_secondary_sensor'] = args['use_secondary_sensor']
            print('system_config', system_config)
        else:
            system_config = load_obj(sims_config)
        sim_dof_names = DOF_NAMES
        if args['sims_dof_names_2']:
            sim_dof_names = robot_def.get('DOF_NAMES_2')
        system_config['realtime'] = False
        fix_base_link = system_config.get('fix_base_link', False) or args['sims_fixed_base']
        fix_base_link = fix_base_link or modes['sample']
        system_config['fix_base_link'] = fix_base_link
        default_dof_pos = system_config.get('default_dof_pos', {})
        system_config['default_dof_pos'] = default_dof_pos
        if rec_q is not None and args['sims_fixed_base']:
            # init_q = rec_q_ctrl[0]
            init_q = rec_q[0]
            def_dof_pos = {n: init_q[i] for i, n in enumerate(sim_dof_names)}
            system_config['default_dof_pos'] = def_dof_pos
            print('sims init_q', def_dof_pos)
        if Q_BOOT is not None and wrapped:
            for i, n in enumerate(sim_dof_names):
                default_dof_pos[n] = Q_BOOT[i]
        elif not wrapped:
            for i, n in enumerate(sim_dof_names):
                default_dof_pos[n] = q_reset[i]
        default_root_states = system_config.get('default_root_states')
        if fix_base_link and default_root_states is not None:
            default_root_states[2] += 0.5
        if args['sims_headless']:
            system_config['headless'] = True
        if args['sims_compute_torque']:
            system_config['compute_torque'] = True
        system_config['compute_torque'] = system_config.get('compute_torque', False)
        # system_config['verbose'] = True
        if use_sim_pd:
            # system_config['Kp'] = system_config.get('Kp', {})
            # system_config['Kp'].update(sim_kps)
            # system_config['Kd'] = system_config.get('Kd', {})
            # system_config['Kd'].update(sim_kds)
            system_config['Kp'] = sim_kps
            system_config['Kd'] = sim_kds
            if sim_torque_limits is not None:
                # system_config['torque_limits'] = system_config.get('torque_limits', {})
                # system_config['torque_limits'].update(sim_torque_limits)
                system_config['torque_limits'] = sim_torque_limits
        system_config['torque_limits'] = sim_torque_limits
        print('sim_torque_limits', sim_torque_limits)
        sims_use_kpkd = ('Kp' not in system_config) or args['sims_use_kpd']
        if sims_use_kpkd:
            tau_limits = TAU_LIMIT.copy()
            # system_config['Kp'] = kp
            # system_config['Kd'] = kd
            system_config['Kp'] = {k: v for k, v in zip(sim_dof_names, kp)}
            system_config['Kd'] = {k: v for k, v in zip(sim_dof_names, kd)}
            system_config['torque_limits'] = {k: v for k, v in zip(sim_dof_names, tau_limits)}
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
        from unicon.systems.sims import cb_sims_recv_send_close
        cb_recv, cb_send, cb_close = cb_sims_recv_send_close(
            **states_sys,
            dof_names=sim_dof_names,
            sims_kwds=sims_kwds,
            **sys_kwds,
        )
    else:
        system_type = args['system']
        print('importing system', system_type)
        sys_cls = import_obj(f'{system_type}:cb_{system_type}_recv_send_close', default_mod_prefix='unicon.systems')
        cb_recv, cb_send, cb_close = autowired(sys_cls, states=states_sys)(
            kp=kp,
            kd=kd,
            q_ctrl_min=q_ctrl_min,
            q_ctrl_max=q_ctrl_max,
            clip_q_ctrl=sys_clip_q_ctrl,
            **sys_kwds,
        )
    cb_close_list.append(cb_close)

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
                    'env_cfg_args': env_cfg_args,
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
        if proc is not None:
            proc.terminate()
        for cb in cb_close_list:
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
                Kp=kp,
                Kd=kd,
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

    if clip_q_ctrl:

        def cb_ctrl_q_clip():
            states_q_ctrl[:] = np.clip(states_q_ctrl, q_ctrl_min, q_ctrl_max)

        seq.append(cb_ctrl_q_clip)

    clear_input = True
    if clear_input:

        def cb_input_clear():
            states_input[:] = 0.
            states_cmd[:] = 0.

        seq.extend([
            cb_input_clear,
        ])

    seq = [cb_recv] + seq + [cb_send]
    seq = [c for c in seq if c is not None]

    # def cb_cmd_clear():
    #     states_cmd[:] = 0.

    # seq.append(cb_cmd_clear)

    fixed_wait = args['fixed_wait']
    if fixed_wait is not None:
        from unicon.general import cb_timer_set, cb_timer_wait
        cb0 = cb_timer_set()
        cb1 = cb_timer_wait(wait=fixed_wait, stats=(True if fixed_wait == 0 else False))
        seq.insert(0, cb0)
        seq.insert(-1, cb1)

    if input_dev_type is not None:
        fallback_input_types = ['term']
        for t in [input_dev_type] + fallback_input_types:
            input_dev_kwds = load_obj(t)
            input_dev_kwds = {'type': input_dev_kwds} if isinstance(input_dev_kwds, str) else input_dev_kwds
            input_dev_type = input_dev_kwds.pop('type')
            print('input_dev_type', input_dev_type)
            cb_input_dev_cls = import_obj(input_dev_type,
                                          default_name_prefix='cb_input',
                                          default_mod_prefix='unicon.inputs')
            cb_input_dev = autowired(cb_input_dev_cls)(**input_dev_kwds)
            if cb_input_dev is not None:
                break
        seq.extend([
            cb_input_dev,
        ])

    cmd_states = dict(
        # states_cmd=np.zeros_like(states_cmd),
    )
    cmd = args['cmd']
    if cmd and cmd != 'none':
        cmd_kwds = load_obj(cmd)
        cmd_kwds = {'type': cmd_kwds} if isinstance(cmd_kwds, str) else cmd_kwds
        cmd_type = cmd_kwds.pop('type')
        print('cmd_type', cmd_type)
        cb_cmd_cls = import_obj(cmd_type, default_name_prefix='cb_cmd', default_mod_prefix='unicon.command')
        kwds = {}
        ccv = args['cmd_const_v']
        if ccv is not None:
            ccv = load_obj(ccv)
            ccv = [ccv] if isinstance(ccv, (float, int)) else ccv
            kwds['cmd'] = ccv
        if env_cfg is not None:
            if cmd_type == 'vel':
                kwds['env_cfg'] = env_cfg
            elif cmd_type == 'wb':
                kwds['cmd_keys'] = env_command_keys
                kwds['cmd_ranges'] = env_command_ranges
        if cmd_type == 'replay':
            rec_cmd = loaded_rec.get('states_cmd')
            kwds['frames'] = rec_cmd
        kwds.update(cmd_kwds)
        if cb_cmd_cls is not None:
            cb_cmd = autowired(cb_cmd_cls, states=cmd_states)(input_keys=input_keys, **kwds)
            cb_cmd()
            seq.append(cb_cmd)
        if len(cmd_states):

            def cb_cmd_merge():
                # print('states_cmd', states_cmd)
                states_cmd[:] = states_cmd + cmd_states['states_cmd']
                # print('states_cmd2', states_cmd)

            # seq.append(cb_cmd_merge)
            seq[-1] = cb_zip(cb_cmd, cb_cmd_merge)

    print('seq before io', seq)
    inputs = args['inputs']
    outputs = args['outputs']
    def_insert_pos = -1
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

    if x_extras2:
        from unicon.kinematics import cb_kinematics_fwd
        cb = autowired(cb_kinematics_fwd)()
        seq.append(cb)

    if rec_post_send and rec_output is not None:
        seq.append(_cb_rec)

    if verbose:
        seq.append(cb_print())

    outer_input_stop = not args['no_outer_input_stop']
    if outer_input_stop:
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

    loop_kwds = {}
    loop_sleep_block = args['loop_sleep_block']
    if loop_sleep_block:
        loop_kwds['sleep_fn'] = 'sleep_block'
    loop_dt_ofs = args['loop_dt_ofs']
    if loop_dt_ofs is not None:
        loop_kwds['dt_ofs'] = loop_dt_ofs

    loop_dt = args['loop_dt']
    loop_dt = dt if loop_dt is None else loop_dt
    from unicon.utils import loop_timed
    try:
        if args['fast']:
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

    return close_fn


if __name__ == '__main__':
    close_fn = run()
    close_fn()
