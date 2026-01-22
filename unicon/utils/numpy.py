import numpy as np


def rpy_reorder(rpy, src='rpy', dest='rpy'):
    inds = [src.index(c) for c in dest]
    return rpy[..., inds]


def quat2rpy3(q, w_first=False):
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


def quat2rpy2(q, w_first=False):
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


def quat2rpy1(q, w_first=False):
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


quat2rpy = quat2rpy3


def quat2mat2(quat, w_first=False):
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


def quat2mat1(quat, w_first=False):
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
    mat = np.array([
        [1.0 - (yY + zZ), xY - wZ, xZ + wY],
        [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
        [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
    ])
    return np.transpose(mat, (2, 0, 1)).reshape(*shape, 3, 3)


quat2mat = quat2mat1


def mat2quat(m):
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


_EPS = np.finfo(float).eps * 4.0


def mat2rpy(m):
    m00 = m[..., 0, 0]
    m22 = m[..., 2, 2]
    m21 = m[..., 2, 1]
    # m02 = m[..., 0, 2]
    m20 = m[..., 2, 0]
    # m01 = m[..., 0, 1]
    m10 = m[..., 1, 0]
    cy = np.sqrt(m00 * m00 + m10 * m10)
    if cy > _EPS:
        ax = np.arctan2(m21, m22)
        ay = np.arctan2(-m20, cy)
        az = np.arctan2(m10, m00)
    else:
        m11 = m[..., 1, 1]
        m12 = m[..., 1, 2]
        ax = np.atan2(-m12, m11)
        ay = np.atan2(-m20, cy)
        az = 0.0
    return np.stack([ax, ay, az], axis=-1)


def mat2pos(m):
    return m[..., :3, 3]


def rpy2mat(rpy):
    shape = []
    if len(rpy.shape) > 1:
        shape = rpy.shape[:-1]
        rpy = rpy.reshape(-1, 3).T
    cos = np.cos(rpy)
    sin = np.sin(rpy)
    cr, cp, cy = cos
    sr, sp, sy = sin
    mat = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    if len(rpy.shape) > 1:
        return np.transpose(mat, (2, 0, 1)).reshape(*shape, 3, 3)
    return mat


def rpy2quat(rpy):
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


def axang2mat(axis, angle):
    sina = np.sin(angle)
    cosa = np.cos(angle)

    axis = axis / np.linalg.norm(axis)

    mat = np.diag([cosa, cosa, cosa])
    mat[:] += np.outer(axis, axis) * (1.0 - cosa)

    axis = axis * sina
    x, y, z = axis
    mat[:] += np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])

    # if point is specified, rotation is not around origin
    # if point is not None:
    #     point = np.asarray(point[:3], dtype=np.float64)
    #     M[:] = point - np.dot(M[:], point)
    return mat


def mat2rotvec(m):
    theta = np.arccos((np.trace(m) - 1) / 2.0)
    if abs(theta) < 1e-8:
        return np.zeros(3)
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) / (2 * np.sin(theta))
    return theta * axis


def xyzrpy2mat(xyz=None, rpy=None):
    mat = np.eye(4)
    if rpy is not None:
        mat[:3, :3] = rpy2mat(rpy)
    if xyz is not None:
        mat[:3, 3] = xyz
    return mat


def mat4_inv(mat):
    ret = np.eye(4, dtype=np.float32)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


def mat4_eye():
    return np.eye(4, dtype=np.float32)


def mat3_eye():
    return np.eye(3, dtype=np.float32)


def vec3_zeros():
    return np.zeros(3, dtype=np.float32)


def pp_arr(arr):
    if arr is None:
        return None
    return np.round(arr.astype(float), decimals=3).tolist()


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def quat_slerp(q0, q1, t):
    # from AMP
    qx, qy, qz, qw = 0, 1, 2, 3

    cos_half_theta = (
        q0[..., qw] * q1[..., qw] +
        q0[..., qx] * q1[..., qx] +
        q0[..., qy] * q1[..., qy] +
        q0[..., qz] * q1[..., qz]
    )

    neg_mask = cos_half_theta < 0
    q1 = np.where(neg_mask[..., None], -q1, q1)

    cos_half_theta = np.abs(cos_half_theta)
    cos_half_theta = np.expand_dims(cos_half_theta, axis=-1)

    half_theta = np.arccos(np.clip(cos_half_theta, -1.0, 1.0))
    sin_half_theta = np.maximum(np.sqrt(np.maximum(1.0 - cos_half_theta**2, 0.0)), 1e-6)

    ratioA = np.sin((1.0 - t) * half_theta) / sin_half_theta
    ratioB = np.sin(t * half_theta) / sin_half_theta

    new_q_x = ratioA * q0[..., qx:qx + 1] + ratioB * q1[..., qx:qx + 1]
    new_q_y = ratioA * q0[..., qy:qy + 1] + ratioB * q1[..., qy:qy + 1]
    new_q_z = ratioA * q0[..., qz:qz + 1] + ratioB * q1[..., qz:qz + 1]
    new_q_w = ratioA * q0[..., qw:qw + 1] + ratioB * q1[..., qw:qw + 1]

    new_q = np.concatenate([new_q_x, new_q_y, new_q_z, new_q_w], axis=-1)

    new_q = np.where(np.abs(sin_half_theta) < 1e-3, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = np.where(np.abs(cos_half_theta) >= 1.0, q0, new_q)

    return new_q


def mat_slerp(mat1, mat2, t):
    q1 = mat2quat(mat1)
    q2 = mat2quat(mat2)
    return quat2mat(quat_slerp(q1, q2, t))


def mat4_slerp(mat1, mat2, t, inplace=True):
    assert inplace is True
    if t <= 0:
        return mat1
    if t >= 1:
        mat1[:] = mat2
        return mat1
    t1 = mat1[:3, 3]
    r1 = mat1[:3, :3]
    t2 = mat2[:3, 3]
    r2 = mat2[:3, :3]
    mat1[:3, 3] = t1 * (1 - t) + t2 * t
    mat1[:3, :3] = mat_slerp(r1, r2, t)
    return mat1
