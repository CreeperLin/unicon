from scipy.spatial.transform import Rotation as R
import numpy as np


def cb_ik_dls(
    states_q_ctrl,
    states_q,
    states_x=None,
    states_x_ctrl=None,
    states_x_err=None,
    states_J=None,
    # damping=0.01,
    damping=0.0001,
    dt=0.02,
    dof_inds=None,
    eef_inds=None,
    pos_mask=True,
    rot_mask=True,
    included_dofs=None,
    excluded_dofs=None,
    use_x_err=False,
):
    from unicon.utils import get_ctx, coalesce
    ctx = get_ctx()

    dof_inds = coalesce(dof_inds, ctx.get('x_ctrl_dof_inds'))
    eef_inds = coalesce(eef_inds, ctx.get('x_ctrl_eef_inds'))

    # num_dofs = len(dof_inds)
    jac_inds = [max(0, x - 1) for x in eef_inds]
    num_eefs = len(eef_inds)
    # lmbda = np.eye(6) * (damping ** 2)
    # lmbda = (damping ** 2) * np.eye(JJT.shape[0])
    # lmbda = (damping ** 2) * np.eye(6 * num_eefs)
    lmbda = np.eye(6 * num_eefs)
    lmbda[np.diag_indices_from(lmbda)] = damping**2
    # np.diag_indices_from(arr)
    use_x_err = use_x_err and states_x_err is not None

    dt = coalesce(dt, ctx.get('dt'))

    def cb():
        errs = []
        Js = []
        for i in range(num_eefs):
            ei = eef_inds[i]
            ji = jac_inds[i]
            if use_x_err:
                err = states_x_err[ei]
            else:
                x_ctrl = states_x_ctrl[ei]
                x = states_x[ei]
                pos_err = x_ctrl[:3, 3] - x[:3, 3]
                rot_err = R.from_matrix(x_ctrl[:3, :3] @ x[:3, :3].T).as_rotvec()
                err = np.concatenate([pos_err, rot_err])
            J = states_J[ji, :, :][:, dof_inds]
            # j_eef_T = j_eef.T
            errs.append(err)
            Js.append(J)
        # inv_term = np.linalg.inv(j_eef @ j_eef_T + lmbda)
        # u = j_eef_T @ inv_term @ dpose
        # u = np.linalg.solve(j_eef_T @ j_eef + lmbda, j_eef_T @ dpose)
        J = np.vstack(Js)
        JT = J.T
        err = np.concatenate(errs)
        dq = JT @ np.linalg.solve(J @ JT + lmbda, err)
        # dq = np.linalg.solve(JT @ J + lmbda, JT @ err)
        # print('err', err.tolist())
        # print('dq', dq.tolist())
        # print('states_x', states_x[eef_inds].tolist())
        # print('states_x_ctrl', states_x_ctrl[eef_inds].tolist())
        # print('states_x_err', states_x_err[eef_inds].tolist())
        # states_q_ctrl[dof_inds] = states_q[dof_inds] + dq[dof_inds] * ik_dt
        states_q_ctrl[dof_inds] = states_q[dof_inds] + dq * dt

    return cb
