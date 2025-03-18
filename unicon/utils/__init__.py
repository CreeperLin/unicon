import os
import time
import numpy as np
from threading import Event

_default_time_fn = time.perf_counter


def force_quit():
    import os
    import signal
    os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)


def md5sum(path):
    import subprocess
    return subprocess.run(['md5sum', path], capture_output=True).stdout.split()[0].decode()


def import_obj(spec, default_name_prefix=None, default_mod_prefix=None):
    spec = spec.split(':')
    if len(spec) == 1:
        mod = None
        name = spec[0]
    elif len(spec) == 2:
        mod, name = spec
    else:
        raise ValueError
    default_mod_prefix = None if '.' in mod else default_mod_prefix
    path = [x for x in [default_mod_prefix, mod] if x is not None]
    mod = '.'.join(path)
    mod = __import__(mod, fromlist=[''])
    name = ('' if default_name_prefix is None else default_name_prefix) + name
    return getattr(mod, name)


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


def sampler_uniform(low, high, rng=None, num_samples=100, seed=1):
    rng = np.random.RandomState(seed) if rng is None else rng
    low = np.array(low)
    high = np.array(high)
    i = 0
    while True:
        if num_samples is not None and i >= num_samples:
            return
        yield rng.uniform(low, high)
        i += 1


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


def quat2rpy_np(q, w_first=False):
    if w_first:
        inds = [1, 2, 3, 0]
    else:
        inds = [0, 1, 2, 3]
    qx, qy, qz, qw = [q[..., i] for i in inds]
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = qw * qw - qx * \
                qx - qy * qy + qz * qz
    roll = np.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2, sinp), np.asin(sinp))
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = qw * qw + qx * \
                qx - qy * qy - qz * qz
    yaw = np.atan2(siny_cosp, cosy_cosp)
    return np.stack((roll, pitch, yaw), axis=-1)


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


def quat2mat_np(quat, w_first=False):
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
    ax = np.atan2(m21, m22)
    ay = np.atan2(-m20, cy)
    az = np.atan2(m10, m00)
    # else:
    # ax = np.atan2(-m12, m11)
    # ay = np.atan2(-m20, cy)
    # az = 0.0
    return np.stack([ax, ay, az], axis=-1)


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
            // double ts = s - T_SPIN;
            double tts = tt2 - ttc - T_SPIN;
            // printf("%lf %lf %lf\n", ttc, t0, tts);
            if (tts > 0) usleep(tts * 1e6);
            // if (tts > 0) {
            //     struct timespec ts;
            //     ts.tv_nsec = tts * 1e9;
            //     nanosleep(&ts, NULL);
            // }
            if (T_SPIN == 0) return;
            double tt1;
            do {
                clock_gettime(CLOCK_MONOTONIC, &t1);
                tt1 = t1.tv_sec + t1.tv_nsec / 1e9;
                // printf("%lf %lf\n", tt1, tt2);
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


_ev = Event()


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


def obj_update(obj, updates):
    if not isinstance(updates, dict):
        return updates
    for key, val in updates.items():
        if isinstance(key, str) and key.startswith(UPDATE_PREFIX):
            fn = key.split(':')[1]
            print('update', type(obj), fn)
            getattr(obj, fn)(*val)
            continue
        if isinstance(obj, dict):
            src = obj.get(key, None)
        elif isinstance(obj, (list, tuple)):
            src = obj[key]
        else:
            src = getattr(src, key, None)
        if isinstance(val, dict) and src is not None:
            ret = obj_update(src, val)
        else:
            print('set', key, 'src', type(src), 'val', val)
            ret = val
        if ret == UPDATE_RET_DEL:
            if isinstance(obj, (dict, list, tuple)):
                del obj[key]
            else:
                delattr(obj, key)
        else:
            if isinstance(obj, (dict, list, tuple)):
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
