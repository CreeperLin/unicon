import numpy as np
from unicon.utils.numpy import mat2rotvec


def x_err(x_ctrl, x, x_inds):
    x_ctrl = x_ctrl[x_inds]
    x = x[x_inds]
    pos_err = x_ctrl[:3, 3] - x[:3, 3]
    rot_err = mat2rotvec(x_ctrl[:3, :3] @ x[:3, :3].T)
    err = np.concatenate([pos_err, rot_err])
    return err
