import os
import time
import types
import numpy as np
import re
import threading

_default_time_fn = time.perf_counter
_ctx = {}
_edge_memo = {}
_registry = {}


def is_rising_edge(x, idx=None, key=None):
    if idx is not None:
        x = float(x[idx])
    key = idx if key is None else key
    last_x = _edge_memo.get(key, 0)
    r = last_x == 0 and x > 0
    # print(_edge_memo, idx, key, r, last_x, x)
    _edge_memo[key] = x
    return r


def set_ctx(ctx):
    _ctx.clear()
    _ctx.update(ctx)


def get_ctx():
    return _ctx


def get_host_ip1():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    addr = s.getsockname()[0]
    s.close()
    return addr


def get_host_ip2():
    import socket
    hostname = socket.gethostname()
    addr = socket.gethostbyname(hostname)
    return addr


get_host_ip = get_host_ip1


class watchdog:

    def __init__(self, timeout=20):
        self.timeout = timeout

    def __enter__(self):
        timeout = self.timeout
        self.running = True

        def _th():
            time.sleep(timeout)
            if self.running:
                print('watchdog barked', timeout)
                force_quit()

        th = threading.Thread(target=_th, daemon=True)
        th.start()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.running = False


def find(root='.', name=None, wholename=None, follow_links=True):
    if root == '~':
        root = os.environ["HOME"]
    import subprocess
    PIPE = subprocess.PIPE
    args = [
        'find',
    ]
    if follow_links:
        args.append('-L')
    args.append(root)
    if name is not None:
        args.extend(['-name', name])
    if wholename is not None:
        args.extend(['-wholename', wholename])
    print(' '.join(args))
    res = subprocess.run(args, stdout=PIPE)
    out = res.stdout.decode()
    if not len(out):
        return None
    out = list(map(str.strip, out.split('\n')))[:-1]
    return out


def load_model_torch(model_path, device=None):
    from unicon.utils.torch import torch_load_jit, torch_no_grad, torch_no_profiling
    torch_no_grad()
    torch_no_profiling()
    model = torch_load_jit(model_path, device=device)
    return model


def load_model_onnx2pytorch(model_path, device=None):
    import onnx
    from onnx2pytorch import ConvertModel
    onnx_model = onnx.load(model_path)
    model = ConvertModel(onnx_model)
    print(model)
    from unicon.utils.torch import torch_no_grad, torch_no_profiling
    torch_no_grad()
    torch_no_profiling()
    return model


def load_model_ort(model_path, device=None):
    import onnxruntime as ort
    options = ort.SessionOptions()
    # options.enable_profiling = True
    ort_sess = ort.InferenceSession(model_path, sess_options=options)
    input0 = ort_sess.get_inputs()[0]
    print('input0', input0.name, input0.type, input0.shape)
    in_k = input0.name
    output0 = ort_sess.get_outputs()[0]
    print('output0', output0.name, output0.type, output0.shape)
    output_names = [output0.name]

    def model(obs):
        # print(obs.shape, type(obs))
        obs = obs.numpy()
        outputs = ort_sess.run(output_names=output_names, input_feed={in_k: obs})
        result = outputs[0]
        return result

    return model


_model_type_ext = {
    '.pt': 'torch',
    '.onnx': 'ort',
    # '.onnx': 'onnx2pytorch',
}


def load_model(model_path, model_type=None, **kwds):
    name, ext = os.path.splitext(model_path)
    if model_type is None:
        model_type = _model_type_ext[ext]
    return globals().get(f'load_model_{model_type}')(model_path, **kwds)


def obj2dict(obj, memo=None):
    if type(obj).__module__ in ['builtins', 'numpy']:
        return obj
    dct = {}
    memo = set() if memo is None else memo
    for key in dir(obj):
        # if key.startswith('_'):
        #     continue
        if key.startswith('__'):
            continue
        attr = getattr(obj, key)
        if callable(attr):
            continue
        if attr is obj:
            continue
        # print(obj, key)
        if key in memo:
            dct[key] = attr
            continue
        memo.add(key)
        dct[key] = obj2dict(attr, memo)
    return dct


def dump_policy_cfg(env=None, env_cfg=None, path=None, ppo_runner=None, train_cfg=None):
    import os
    import sys
    import yaml
    if path is None:
        path = '.' if ppo_runner is None else ppo_runner.path
    print('dump_policy_cfg', path)
    os.makedirs(path, exist_ok=True)
    # os.system('git diff > {}'.format(os.path.join(path, 'diff.patch')))
    if env is not None:
        env_cfg.dof_names = env.dof_names
    if env_cfg is not None:
        env_cfg.argv = sys.argv
        with open(os.path.join(path, 'env_cfg.yaml'), 'w') as f:
            f.write(yaml.dump(obj2dict(env_cfg), sort_keys=False))
    if train_cfg is not None:
        with open(os.path.join(path, 'train_cfg.yaml'), 'w') as f:
            f.write(yaml.dump(obj2dict(train_cfg), sort_keys=False))


def match_keys(pats, keys, substr=True, regex=False):
    if not isinstance(pats, (tuple, list)):
        pats = [pats]
    inds = []
    for i, k in enumerate(keys):
        for pat in pats:
            assert isinstance(pat, str)
            if pat == k:
                inds.append(i)
                break
            if substr:
                if pat in k:
                    inds.append(i)
                    break
            if regex:
                match = re.match(pat, k)
                if match is not None:
                    inds.append(i)
                    break
    return inds


def load_rec(rec_path, rec_type=None, load_kwds=None, robot_def=None):
    rec_type = None if rec_type == '' else rec_type
    _, ext = os.path.splitext(rec_path)
    rec_type = ext[1:] if rec_type is None else rec_type
    print('rec_type', rec_type)
    mod = import_obj((rec_type, None), default_mod_prefix='unicon.data')
    print('rec_path', rec_path)
    return mod.load(
        rec_path,
        robot_def=robot_def,
        **(load_kwds or {}),
    )


def load_obj(obj):
    import yaml
    yaml.add_multi_constructor('tag:', lambda *_: None, Loader=yaml.SafeLoader)
    if isinstance(obj, str):
        if os.path.exists(obj):
            with open(obj, 'r') as f:
                obj = f.read()
    return yaml.safe_load(obj)
    # return yaml.full_load(obj)


def states_ts2dt(states_and_ts=None, states=None, tss=None, dt=0.02):
    if states_and_ts is not None:
        states = {k: v for k, v in states_and_ts.items() if k + '.ts' in states_and_ts}
        tss = {k[:-3]: v for k, v in states_and_ts.items() if '.ts' in k}
    t0 = min([np.min(ts) for ts in tss.values()])
    t1 = max([np.max(ts) for ts in tss.values()])
    dura = t1 - t0
    num_frames = int(dura / dt)
    print('mean intv', {k: (tss[k][-1] - tss[k][0]) / len(states[k]) for k in states.keys()})
    print('t0', t0, 't1', t1)
    print('dura', dura)
    print('shapes', {k: v.shape for k, v in states.items()})
    print('num_frames', num_frames)

    tss = {k: v - t0 for k, v in tss.items()}
    t0 = 0
    import torch

    states_new = {}
    for k, states_ori in states.items():
        ts = torch.from_numpy(tss[k])
        ft = torch.arange(num_frames) * dt + t0
        inds0 = torch.searchsorted(ts, ft, side='left').squeeze().numpy()
        inds0[inds0 >= states_ori.shape[0]] = 0
        s0 = states_ori[inds0]
        inds1 = inds0 + 1
        inds1[inds0 >= states_ori.shape[0]] = 0
        s1 = states_ori[inds1]
        ts0 = ts[inds0]
        r = (ts0 - ft).numpy() / dt
        assert np.all(r >= 0)
        r = r.reshape(-1, 1)
        s = s0 * (1 - r) + s1 * r
        states_new[k] = s
    return states_new


def get_min_z(topo, base_link_name):

    def dfs(link_name, cur_z: float = 0):
        min_z = cur_z
        joints = topo[link_name]
        for joint in joints:
            origin = joint.origin
            joint_z = 0 if origin is None else origin[2, 3]
            # print(link_name, joint.name, cur_z, joint_z, min_z)
            min_z = min(min_z, dfs(joint.child, cur_z + joint_z))
        return min_z

    return dfs(base_link_name)


def parse_urdf(
    urdf_path,
    default_kp=43.2,
    default_kd=4.2,
    default_q_min=-np.pi,
    default_q_max=np.pi,
    default_tau_limit=1000.,
    default_qd_limit=30.,
):
    from unicon.utils.yourdfpy import URDF
    print('parse_urdf', urdf_path)
    if hasattr(URDF, 'load'):
        urdf = URDF.load(urdf_path, load_meshes=False, build_scene_graph=False)
    else:
        urdf = URDF.from_xml_file(urdf_path)
    links = urdf.robot.links
    joints = urdf.robot.joints
    from collections import defaultdict
    topo = defaultdict(list)
    for j in joints:
        topo[j.parent].append(j)
    base_link_name = links[0].name
    min_z = get_min_z(topo, base_link_name)
    min_z = -1.23 if min_z >= 0 else min_z
    print('min_z', min_z)
    joints = [j for j in joints if j.type != 'fixed']
    link_names = [k.name for k in links]
    dof_names = [j.name for j in joints]
    joint_limits = [j.limit for j in joints]
    joint_dynamics = [j.dynamics for j in joints]
    q_min = [(default_q_min if m is None else m.lower) for m in joint_limits]
    q_max = [(default_q_max if m is None else m.upper) for m in joint_limits]
    q_min = [(-3.14 if q is None else q) for q in q_min]
    q_max = [(3.14 if q is None else q) for q in q_max]
    tau_limit = [(default_tau_limit if m is None else m.effort) for m in joint_limits]
    qd_limit = [(default_qd_limit if m is None else m.velocity) for m in joint_limits]
    damping = [(0 if (d is None or d.damping is None) else float(d.damping)) for d in joint_dynamics]
    friction = [(0 if (d is None or d.friction is None) else float(d.friction)) for d in joint_dynamics]
    armature = [(0 if (d is None or d.armature is None) else float(d.armature)) for d in joint_dynamics]
    num_dofs = len(dof_names)
    print('dof_names', num_dofs, dof_names)
    print('link_names', len(link_names), link_names)
    print('q_min', np.round(np.array(q_min), 2).tolist())
    print('q_max', np.round(np.array(q_max), 2).tolist())
    print('tau_limit', np.round(np.array(tau_limit), 2).tolist())
    print('qd_limit', np.round(np.array(qd_limit), 2).tolist())
    print('damping', np.round(np.array(damping), 2).tolist())
    print('friction', np.round(np.array(friction), 2).tolist())
    print('armature', np.round(np.array(armature), 2).tolist())
    q_min = {k: v for k, v in zip(dof_names, q_min)}
    q_max = {k: v for k, v in zip(dof_names, q_max)}
    tau_limit = {k: v for k, v in zip(dof_names, tau_limit)}
    qd_limit = {k: v for k, v in zip(dof_names, qd_limit)}
    damping = {k: v for k, v in zip(dof_names, damping)}
    friction = {k: v for k, v in zip(dof_names, friction)}
    armature = {k: v for k, v in zip(dof_names, armature)}
    kps = {'*': default_kp}
    kds = {'*': default_kd}
    robot_def = {
        'NUM_DOFS': num_dofs,
        'DOF_NAMES': dof_names,
        'LINK_NAMES': link_names,
        'Q_CTRL_MIN': q_min,
        'Q_CTRL_MAX': q_max,
        'TAU_LIMIT': tau_limit,
        'QD_LIMIT': qd_limit,
        'KP': kps,
        'KD': kds,
        'Q_DAMPING': damping,
        'Q_FRICTION': friction,
        'Q_ARMATURE': armature,
        'INIT_Z': -min_z * (1.2),
        '_URDF': urdf,
    }
    return robot_def


def parse_robot_def(robot_def):
    if not isinstance(robot_def, dict):
        robot_def = {k: v for k, v in vars(robot_def).items() if k.isupper()}
    print('robot_def', robot_def.keys())
    urdf_path = robot_def.get('URDF')
    mjcf_path = robot_def.get('MJCF')
    _default_asset_dir = f'{os.environ["HOME"]}/GitRepo/GR1/resources/robots/'
    _default_asset_dir = os.environ.get('UNICON_ASSET_DIR', _default_asset_dir)
    if urdf_path is not None:
        urdf_path = urdf_path if urdf_path.startswith('/') else os.path.join(_default_asset_dir, urdf_path)
        robot_def['URDF'] = urdf_path
        try:
            urdf_def = parse_urdf(urdf_path)
            # urdf_def.update(robot_def)
            obj_update(urdf_def, robot_def, verbose=False)
            robot_def = urdf_def
        except Exception:
            import traceback
            traceback.print_exc()
    elif mjcf_path is not None:
        mjcf_path = mjcf_path if mjcf_path.startswith('/') else os.path.join(_default_asset_dir, mjcf_path)
        robot_def['MJCF'] = mjcf_path
        try:
            from unicon.utils.mjcf2urdf import convert
            import tempfile
            with tempfile.TemporaryDirectory() as urdf_tmp_dir:
                urdf_path = os.path.join(urdf_tmp_dir, os.path.basename(mjcf_path.replace('.xml', '.urdf')))
                convert(mjcf_path, urdf_path)
                # urdf_path = convert_mjcf_to_urdf(mjcf_path, urdf_tmp_dir)[-1]
                print('urdf_path', urdf_path)
                os.system(f'cp {urdf_path} tmp.urdf')
                # robot_def['URDF'] = urdf_path
                urdf_def = parse_urdf(urdf_path)
                urdf_def.update(robot_def)
                robot_def = urdf_def
        except Exception:
            import traceback
            traceback.print_exc()
    robot_def['NUM_DOFS'] = len(robot_def['DOF_NAMES'])
    print('robot_def', robot_def.keys())
    num_dofs = robot_def['NUM_DOFS']
    dof_names = robot_def['DOF_NAMES']
    dof_attr_keys = ['QD_LIMIT', 'TAU_LIMIT', 'KP', 'KD', 'Q_CTRL_MIN', 'Q_CTRL_MAX', 'Q_RESET', 'Q_BOOT']
    for k in dof_attr_keys:
        v = robot_def.get(k)
        if v is None:
            continue
        if isinstance(v, (float, int)):
            v = [v] * num_dofs
        if isinstance(v, dict):
            v = [v.get(n, ([vv for kk, vv in v.items() if kk in n] or [v.get('*', 0.)])[0]) for n in dof_names]
        robot_def[k] = v

    def is_num_list(x):
        return isinstance(x, list) and isinstance(x[0] if len(x) else None, (float, int))

    robot_def = {k: np.array(v, dtype=np.float64) if is_num_list(v) else v for k, v in robot_def.items()}
    for k, v in robot_def.items():
        if isinstance(v, np.ndarray):
            v[np.isnan(v)] = 0
    return robot_def


def force_quit():
    import os
    import signal
    os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)


def md5sum(path):
    import subprocess
    PIPE = subprocess.PIPE
    return subprocess.run(['md5sum', path], stdout=PIPE).stdout.split()[0].decode()


def register_obj(
    obj,
    spec=None,
    default_name_prefix=None,
    default_mod_prefix=None,
):
    mod = name = None
    if spec is not None:
        if isinstance(spec, str):
            spec = spec.split(':')
        if len(spec) == 1:
            spec = spec[0]
            mod = spec
            name = spec
        elif len(spec) == 2:
            mod, name = spec
        else:
            raise ValueError
    default_mod_prefix = None if (mod is not None and '.' in mod) else default_mod_prefix
    path = [x for x in [default_mod_prefix, mod] if x is not None]
    names = [x for x in [default_name_prefix, name] if x is not None]
    full_name = '_'.join(names)
    obj_name = getattr(obj, '__name__', None)
    full_name = full_name if len(full_name) else obj_name
    _registry[full_name] = obj
    if not any(map(len, path)):
        print('register_obj registry', mod, full_name, obj)
        return
    mod_path = '.'.join(path)
    if isinstance(obj, types.ModuleType):
        import sys
        sys.modules[mod_path] = obj
        print('register_obj mod', mod_path, obj)
        return
    try:
        mod = __import__(mod_path, fromlist=[''])
    except ImportError:
        if default_mod_prefix is not None and name is not None:
            print('import failed', mod_path)
            mod = __import__(default_mod_prefix, fromlist=[''])
        else:
            raise
    setattr(mod, full_name, obj)
    print('register_obj attr', mod, full_name, obj)


def import_file(path, name=None):
    """Import modules from file."""
    import sys
    import importlib
    name = '_default' if name is None else name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if name:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def import_obj(
    spec=None,
    default_name_prefix=None,
    default_mod_prefix=None,
    prefer_mod=False,
):
    mod = name = None
    if spec is not None:
        if isinstance(spec, str):
            spec = spec.split(':')
        if len(spec) == 1:
            spec = spec[0]
            mod = spec
            name = None if prefer_mod else spec
        elif len(spec) == 2:
            mod, name = spec
        else:
            raise ValueError(f'invalid spec {spec}')
    if isinstance(name, str) and name in _registry:
        return _registry.get(name)
    if isinstance(mod, str) and mod[-3:] in ['.py', '.so']:
        mod = import_file(mod)
    else:
        default_mod_prefix = None if (isinstance(mod, str) and '.' in mod) else default_mod_prefix
        path = [x for x in [default_mod_prefix, mod] if x is not None]
        mod_path = '.'.join(path)
        try:
            mod = __import__(mod_path, fromlist=[''])
        except ImportError:
            if default_mod_prefix is not None and name is not None:
                print('import failed', mod_path)
                mod = __import__(default_mod_prefix, fromlist=[''])
            else:
                raise
    names = [x for x in [default_name_prefix, name] if x is not None]
    full_name = '_'.join(names)
    if not len(full_name):
        return mod
    if hasattr(mod, full_name):
        return getattr(mod, full_name)
    return getattr(mod, name)


def dict2mod(dct, name=None):
    import types
    mod = types.ModuleType(name)
    mod.__dict__.update(dct)
    return mod


def list2slice(lst):
    if len(lst) < 2:
        return lst
    st, ed = lst[0], lst[-1]
    if len(lst) == (ed - st + 1) and tuple(sorted(lst)) == tuple(lst):
        lst = slice(st, ed + 1)
    return lst


def quat_mul_np(q1, q2, w_first=False):
    if w_first:
        inds = [1, 2, 3, 0]
    else:
        inds = [0, 1, 2, 3]
    x1, y1, z1, w1 = [q1[..., i] for i in inds]
    x2, y2, z2, w2 = [q2[..., i] for i in inds]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    out = [w, x, y, z] if w_first else [x, y, z, w]
    quat = np.stack(out, axis=-1)
    return quat


def quat_from_axis_angle_np(axis, angle):
    theta = (angle / 2).reshape(*axis.shape[:-1], 1)
    xyz = (axis / np.linalg.norm(axis, axis=-1)) * np.sin(theta)
    w = np.cos(theta)
    q = np.concatenate([xyz, w], axis=-1)
    q = q / np.linalg.norm(q, axis=-1)
    return q


def quat_conjugate_np(q, w_first=False):
    q_w, q_vec = (q[..., :1], q[..., 1:]) if w_first else (q[..., -1:], q[..., :3])
    return np.concatenate((q_w, -q_vec) if w_first else (-q_vec, q_w), axis=-1)


def quat_rotate_np(q, v, w_first=False):
    q_w, q_vec = (q[..., 0], q[..., 1:]) if w_first else (q[..., -1], q[..., :3])
    a = v * (2.0 * q_w**2 - 1.0)[..., np.newaxis]
    b = np.cross(q_vec, v) * q_w[..., np.newaxis] * 2.0
    c = q_vec * np.sum(q_vec[..., np.newaxis, :] * v[..., np.newaxis, :], axis=-1) * 2.0
    return a + b + c


def quat_rotate_inverse_np(q, v, w_first=False):
    q_w, q_vec = (q[..., 0], q[..., 1:]) if w_first else (q[..., -1], q[..., :3])
    a = v * (2.0 * q_w**2 - 1.0)[..., np.newaxis]
    b = np.cross(q_vec, v) * q_w[..., np.newaxis] * 2.0
    c = q_vec * np.sum(q_vec[..., np.newaxis, :] * v[..., np.newaxis, :], axis=-1) * 2.0
    return a - b + c


def quat_rotate_inverse_np2(q, v, w_first=False):
    q_w, q_vec = (q[..., 0], q[..., 1:]) if w_first else (q[..., -1], q[..., :3])
    a = v * (2.0 * q_w**2 - 1.0)[..., np.newaxis]
    # b = np.cross(q_vec, v) * q_w * 2.0
    # directly compute the cross is faster than np.cross
    qx, qy, qz = (q_vec[..., 0], q_vec[..., 1], q_vec[..., 2])
    vx, vy, vz = (v[..., 0], v[..., 1], v[..., 2])
    # b = np.array([
    b = np.stack(
        [
            # q_vec[1]*v[2] - v[1] * q_vec[2], q_vec[2]*v[0] - v[2] * q_vec[0], q_vec[0]*v[1] - v[0] * q_vec[1]
            qy * vz - vy * qz,
            qz * vx - vz * qx,
            qx * vy - vx * qy
        ],
        axis=-1) * q_w[..., np.newaxis] * 2.0
    # c = q_vec * np.dot(q_vec, v) * 2.0
    c = q_vec * np.sum(q_vec[..., np.newaxis, :] * v[..., np.newaxis, :], axis=-1) * 2.0
    return a - b + c


# isaacgym/python/isaacgym/torch_utils.py
def get_axis_params(value, axis_idx, x_value=0., dtype=np.float32, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))


def set_seed(seed):
    import random
    import torch
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def try_conn(address, port):
    import socket
    s = socket.socket()
    fail = False
    try:
        s.connect((address, port))
    except Exception as e:
        print(address, port, e)
        fail = True
    finally:
        s.close()
    return fail


def rpy_reorder(rpy, src='rpy', dest='rpy'):
    inds = [src.index(c) for c in dest]
    return rpy[..., inds]


def quat2rpy_np3(q, w_first=False):
    if w_first:
        inds = [1, 2, 3, 0]
    else:
        inds = [0, 1, 2, 3]
    x, y, z, w = [q[..., i] for i in inds]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    # roll_x = np.atan2(t0, t1)
    roll_x = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1, 1)
    # pitch_y = np.asin(t2)
    pitch_y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    # yaw_z = np.atan2(t3, t4)
    yaw_z = np.arctan2(t3, t4)
    return np.stack([roll_x, pitch_y, yaw_z], axis=-1)


def quat2rpy_np2(q, w_first=False):
    if w_first:
        inds = [1, 2, 3, 0]
    else:
        inds = [0, 1, 2, 3]
    qx, qy, qz, qw = [q[..., i] for i in inds]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    r = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        p = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        p = np.arcsin(sinp)
    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    y = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack((r, p, y), axis=-1)


def quat2rpy_np1(q, w_first=False):
    if w_first:
        inds = [1, 2, 3, 0]
    else:
        inds = [0, 1, 2, 3]
    qx, qy, qz, qw = [q[..., i] for i in inds]
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = qw * qw - qx * \
                qx - qy * qy + qz * qz
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2, sinp), np.asin(sinp))
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = qw * qw + qx * \
                qx - qy * qy - qz * qz
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack((roll, pitch, yaw), axis=-1)


quat2rpy_np = quat2rpy_np3


def quat2mat_np2(quat, w_first=False):
    if w_first:
        q0, q1, q2, q3 = quat
    else:
        q1, q2, q3, q0 = quat
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    # 3x3 rotation matrix
    mat = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    return mat


def quat2mat_np1(quat, w_first=False):
    shape = quat.shape[:-1]
    quat = quat.reshape(-1, 4)
    if w_first:
        w, x, y, z = quat.T
    else:
        x, y, z, w = quat.T
    norm = np.linalg.norm(quat, axis=-1)
    norm[norm < 1e-4] = 1.
    s = 2.0 / norm
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    mat = np.array([[1.0 - (yY + zZ), xY - wZ, xZ + wY], [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
                    [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])
    return np.transpose(mat, (2, 0, 1)).reshape(*shape, 3, 3)


quat2mat_np = quat2mat_np1


def mat2quat_np(m):
    m00 = m[..., 0, 0]
    m11 = m[..., 1, 1]
    m22 = m[..., 2, 2]
    m21 = m[..., 2, 1]
    m12 = m[..., 1, 2]
    m02 = m[..., 0, 2]
    m20 = m[..., 2, 0]
    m01 = m[..., 0, 1]
    m10 = m[..., 1, 0]
    w = np.sqrt(np.maximum(1 + m00 + m11 + m22, 0))
    x = np.sqrt(np.maximum(1 + m00 - m11 - m22, 0))
    y = np.sqrt(np.maximum(1 - m00 + m11 - m22, 0))
    z = np.sqrt(np.maximum(1 - m00 - m11 + m22, 0))
    x = np.copysign(x, m21 - m12)
    y = np.copysign(y, m02 - m20)
    z = np.copysign(z, m10 - m01)
    return 0.5 * np.stack([x, y, z, w], axis=-1)


def mat2rpy_np(m):
    m00 = m[..., 0, 0]
    # m11 = m[..., 1, 1]
    m22 = m[..., 2, 2]
    m21 = m[..., 2, 1]
    # m12 = m[..., 1, 2]
    # m02 = m[..., 0, 2]
    m20 = m[..., 2, 0]
    # m01 = m[..., 0, 1]
    m10 = m[..., 1, 0]
    cy = np.sqrt(m00 * m00 + m10 * m10)
    # _EPS = np.finfo(float).eps * 4.0
    # if cy > _EPS:
    ax = np.arctan2(m21, m22)
    ay = np.arctan2(-m20, cy)
    az = np.arctan2(m10, m00)
    # else:
    # ax = np.atan2(-m12, m11)
    # ay = np.atan2(-m20, cy)
    # az = 0.0
    return np.stack([ax, ay, az], axis=-1)


def mat2pos_np(m):
    return m[..., :3, 3]


def rpy2mat_np(rpy):
    shape = []
    if len(rpy.shape) > 1:
        shape = rpy.shape[:-1]
        rpy = rpy.reshape(-1, 3).T
    cos = np.cos(rpy)
    sin = np.sin(rpy)
    cr, cp, cy = cos
    sr, sp, sy = sin
    mat = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    if len(rpy.shape) > 1:
        return np.transpose(mat, (2, 0, 1)).reshape(*shape, 3, 3)
    return mat


def rpy2quat_np(rpy):
    rpy = 0.5 * rpy
    # c = np.cos(a._x / 2)
    # d = np.cos(a._y / 2)
    # e = np.cos(a._z / 2)
    cos = np.cos(rpy)
    sin = np.sin(rpy)
    c, d, e = cos[..., 0], cos[..., 1], cos[..., 2]
    f, g, h = sin[..., 0], sin[..., 1], sin[..., 2]
    # f = np.sin(a._x / 2)
    # g = np.sin(a._y / 2)
    # h = np.sin(a._z / 2)
    x = f * d * e + c * g * h
    y = c * g * e - f * d * h
    z = c * d * h + f * g * e
    w = c * d * e - f * g * h
    return np.stack([x, y, z, w], axis=-1)


def compose_mat_np(xyz=None, rpy=None):
    mat = np.eye(4)
    if rpy is not None:
        mat[:3, :3] = rpy2mat_np(rpy)
    if xyz is not None:
        mat[:3, 3] = xyz
    return mat


def pp_arr(arr):
    return np.round(arr.astype(float), decimals=3).tolist()


def set_nice(nice):
    niceness = os.nice(nice)
    print('niceness set to', niceness)


def set_cpu_affinity(cpu_affinity):
    import psutil
    p = psutil.Process()
    p.cpu_affinity(cpu_affinity)
    print('cpu_affinity set to', p.cpu_affinity())


def set_nice2(nice):
    os.system(f'sudo renice -n {nice} -p {os.getpid()}')


def set_cpu_affinity2(cpu_affinity):
    os.system(f'sudo taskset -cp {cpu_affinity} {os.getpid()}')


def fn_lat(func, num_runs=2**16, time_fn=_default_time_fn):
    lat = 0
    v_lat = 0

    def void():
        pass

    for _ in range(num_runs):
        t0 = time_fn()
        void()
        t1 = time_fn()
        v_lat += (t1 - t0)
    # print('void', v_lat / num_runs)
    for _ in range(num_runs):
        t0 = time_fn()
        func()
        t1 = time_fn()
        lat += (t1 - t0)
    return (lat - v_lat) / num_runs


def sleep_spin(t0, s, spin_t=0.0004, time_fn=_default_time_fn):
    # t_ns = time.perf_counter_ns
    # tmp_ns = int((s + t0) * 1e9)
    tmp = s + t0
    t_s = (tmp - time_fn()) - spin_t
    if t_s > 0:
        time.sleep(t_s)
    while time_fn() < tmp:
        pass
    # while t_ns() < tmp_ns:
    # pass


def get_sleep_cffi(t_spin=0.0004):
    from cffi import FFI
    ffibuilder = FFI()
    ffibuilder.cdef("void _sleep(double t0, double s);")
    ffibuilder.set_source(
        "_utils", r"""
        #include <time.h>
        #include <unistd.h>
        const double T_SPIN = <T_SPIN>;

        static void _sleep(double t0, double s)
        {
            struct timespec tc;
            struct timespec t1;
            clock_gettime(CLOCK_MONOTONIC, &tc);
            double tt0 = t0;
            double ttc = tc.tv_sec + tc.tv_nsec / 1e9;
            double tt2 = tt0 + s;
            double tts = tt2 - ttc - T_SPIN;
            if (tts > 0) usleep(tts * 1e6);
            if (T_SPIN == 0) return;
            double tt1;
            do {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                tt1 = t1.tv_sec + t1.tv_nsec / 1e9;
            } while (tt1 < tt2);
        }
    """.replace('<T_SPIN>', str(t_spin)))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tmpdir = d
        # print('tmpdir', tmpdir)
        ffibuilder.compile(
            tmpdir=tmpdir,
            verbose=True,
        )
        import sys
        sys.path.append(tmpdir)
        import _utils
        # print(dir(_utils.lib))
        return _utils.lib._sleep


_default_sleep_fn = sleep_spin

use_cffi_sleep = True
use_cffi_sleep = False
if use_cffi_sleep:
    sleep_cffi = None
    try:
        sleep_cffi = get_sleep_cffi()
        _default_sleep_fn = sleep_cffi
    except Exception:
        import traceback
        traceback.print_exc()


def sleep_block(t0, s, time_fn=_default_time_fn):
    ts = t0 + s - time_fn()
    if ts > 0:
        time.sleep(ts)


_ev = threading.Event()


def sleep_wait(t0, s, time_fn=_default_time_fn):
    ts = t0 + s - time_fn()
    _ev.wait(ts)


def time_res(time_fn=_default_time_fn):
    ts = np.array([time_fn() for _ in range(2**8)])
    dt = ts[1:] - ts[:1]
    mean_dt = np.mean(dt)
    print('dt', mean_dt, np.min(dt), np.max(dt))
    return mean_dt


def loop_timed(
    cb,
    num_steps=None,
    dt=0.02,
    dt_ofs=0,
    cb_dt=False,
    time_fn=_default_time_fn,
    sleep_fn=_default_sleep_fn,
    stats=True,
):
    sleep_fn = globals().get(sleep_fn) if isinstance(sleep_fn, str) else sleep_fn
    if int(dt_ofs) == 1:

        def test_fn():
            t0 = time_fn()
            sleep_fn(t0, dt)
            time_fn()

        lat = fn_lat(test_fn, num_runs=int(3 // dt)) - dt
        print('sleep_fn lat', lat)
        dt_ofs = -lat
    print('dt_ofs', dt_ofs)
    frameno = -1
    timeouts = 0
    t_idles = 0
    if stats:
        t_ss = 0
        tis = []
    t_start = t0 = time_fn()
    while True:
        frameno += 1
        if num_steps is not None and frameno >= num_steps:
            break
        t_s = t0 + dt - time_fn()
        if t_s > 0:
            sleep_fn(t0, dt + dt_ofs)
        else:
            timeouts += 1
            if timeouts < 64:
                print('### loop timeout:', frameno, t_s)
        if stats:
            t1 = time_fn()
        ret = cb()
        if stats:
            t_ss += (t1 - t0)
            tis.append(t_s)
        if ret is True:
            break
        t0 += dt
        t_idles += t_s
        if cb_dt:
            t0 = time_fn()
    t_stop = time_fn()
    dura = t_stop - t_start
    print(f'{frameno} steps in {dura}s')
    if frameno > 0:
        avg = dura / frameno
        print(f'avg/timeout/idle: {avg}, {timeouts}, {t_idles / frameno}')
        if stats:
            tis = np.array(tis)[1:]
            print('t_idle min/max/avg/std', np.min(tis), np.max(tis), np.mean(tis), np.std(tis))
            t_ss_avg = t_ss / frameno
            print('t_ss avg/err', t_ss_avg, t_ss_avg - dt)
    return frameno


UPDATE_PREFIX = '__update__'
UPDATE_RET_DEL = '__update_ret_del__'


def obj_update(obj, updates, verbose=True):
    if not isinstance(updates, dict):
        return updates
    obj_indexed = isinstance(obj, (list, tuple)) or hasattr(obj, 'tolist')
    obj_keyed = isinstance(obj, dict)
    upd_indexed = all(map(lambda x: isinstance(x, int), updates))
    if upd_indexed and not obj_indexed:
        return updates
    for key, val in updates.items():
        if isinstance(key, str) and key.startswith(UPDATE_PREFIX):
            fn = key.split(':')[1]
            if verbose:
                print('update', type(obj), fn)
            getattr(obj, fn)(*val)
            continue
        if obj_keyed:
            src = obj.get(key, None)
        elif obj_indexed:
            src = obj[key]
        else:
            src = getattr(obj, key, None)
        if isinstance(val, dict) and src is not None:
            ret = obj_update(src, val, verbose=verbose)
        else:
            if verbose:
                print('set', key, 'src', type(src), 'val', val)
            ret = val
        if isinstance(ret, str) and ret == UPDATE_RET_DEL:
            if isinstance(obj, (dict, list, tuple)):
                del obj[key]
            else:
                delattr(obj, key)
        else:
            if obj_keyed or obj_indexed:
                obj[key] = ret
            else:
                setattr(obj, key, ret)
    return obj


def rec2motion(rec, name=None, dofs=None, save_dir='datasets'):
    dofs = slice(dofs) if dofs is None else dofs
    states_quat = rec.get('states_quat')
    states_ang_vel = rec.get('states_ang_vel')
    states_q = rec['states_q']
    states_qd = rec['states_qd']
    states_lin_vel = rec.get('states_lin_vel')
    states_pos = rec.get('states_pos')
    num_frames = len(states_q)
    zeros3 = np.zeros((num_frames, 3))
    pos = zeros3 if states_pos is None else states_pos
    rot = states_quat
    joint_pos = states_q[:, dofs]
    foot_pos = np.zeros((num_frames, states_q.shape[1]))
    base_lin_vel = zeros3 if states_lin_vel is None else states_lin_vel
    base_ang_vel = states_ang_vel
    joint_vel = states_qd[:, dofs]
    foot_vel = np.zeros((num_frames, states_q.shape[1]))
    frames = np.concatenate([
        pos,
        rot,
        joint_pos,
        foot_pos,
        base_lin_vel,
        base_ang_vel,
        joint_vel,
        foot_vel,
    ], axis=-1)
    print('frames', frames.shape)
    dt = 0.02
    data = {
        "LoopMode": "Wrap",
        "FrameDuration": dt,
        "MotionWeight": 1.0,
        "Frames": frames.tolist(),
    }

    import json
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, name + '.txt'), 'w') as f:
        json.dump(data, f)
