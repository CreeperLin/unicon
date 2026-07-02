import numpy as np


def cb_ik_gn(
    states_q_ctrl,
    states_q,
    states_x_ctrl,
    dt=None,
    x_ctrl_dof_inds=None,
    x_ctrl_eef_inds=None,
    pos_mask=True,
    rot_mask=True,
    included_dofs=None,
    excluded_dofs=None,
    reduce=True,
    q_lerp=0.5,
    weight_translation: float = 50.0,
    weight_rotation: float = 1.0,
    weight_regularization: float = 0.02,
    weight_smoothness: float = 0.1,
    qd_max=5.,
    lam=1e-3,
):
    import casadi
    from pinocchio import casadi as cpin

    from unicon.utils import get_ctx, coalesce, pats2inds
    from unicon.utils.pin import load_robot_pin, dedup_frames

    ctx = get_ctx()
    robot_def = ctx.get('robot_def')
    q_reset = ctx.get('Q_RESET')
    q_min = robot_def['Q_CTRL_MIN']
    q_max = robot_def['Q_CTRL_MAX']
    LINK_NAMES = robot_def.get('LINK_NAMES')
    DOF_NAMES = robot_def.get('DOF_NAMES')
    PIN_DOF_NAMES = robot_def.get('PIN_DOF_NAMES', DOF_NAMES)
    num_links = len(LINK_NAMES)
    robot = robot_def.get('robot_pin')
    if robot is None:
        urdf_path = robot_def['URDF']
        robot = load_robot_pin(urdf_path)
    model, data = robot.model, robot.data
    model = dedup_frames(model)
    robot.model = model
    num_frames = len(model.frames)
    frame_names = [f.name for f in model.frames]
    njoints = model.njoints

    print('njoints', njoints, num_frames, num_links)
    print('frame_names', frame_names)

    x_ctrl_dof_inds = coalesce(x_ctrl_dof_inds, ctx.get('x_ctrl_dof_inds'))
    x_ctrl_dof_names = [DOF_NAMES[i] for i in x_ctrl_dof_inds]
    x_ctrl_eef_inds = coalesce(x_ctrl_eef_inds, ctx.get('x_ctrl_eef_inds'))
    x_ctrl_eef_names = [LINK_NAMES[i] for i in x_ctrl_eef_inds]

    num_dofs_ik = len(x_ctrl_dof_inds)
    num_eefs_ik = len(x_ctrl_eef_inds)
    print('x_ctrl_dof_inds', num_dofs_ik, x_ctrl_dof_inds)
    print('x_ctrl_eef_inds', num_eefs_ik, x_ctrl_eef_inds)

    pos_mask = [1., 1., 1.] if pos_mask is True else pos_mask
    pos_mask = [0., 0., 0.] if pos_mask is False else pos_mask
    pos_mask = np.asarray(pos_mask, dtype=np.float32)
    rot_mask = [1., 1., 1.] if rot_mask is True else pos_mask
    rot_mask = [0., 0., 0.] if rot_mask is False else pos_mask
    rot_mask = np.asarray(rot_mask, dtype=np.float32)

    dt = coalesce(dt, ctx.get('dt'))
    dq_max = dt * qd_max

    pin_dof_names = list(PIN_DOF_NAMES)
    if reduce and len(x_ctrl_dof_inds) != njoints:
        fixed_dof_names = [n for n in pin_dof_names if n not in x_ctrl_dof_names]
        print('fixed_dof_names', fixed_dof_names)
        reduced_robot = robot.buildReducedRobot(
            list_of_joints_to_lock=fixed_dof_names,
        )
        robot = reduced_robot
        pin_dof_names = [n for n in pin_dof_names if n not in fixed_dof_names]
    print('pin_dof_names', pin_dof_names)

    _, _, dof_inds_in = pats2inds(DOF_NAMES, pin_dof_names)
    _, _, dof_inds_out = pats2inds(pin_dof_names, x_ctrl_dof_names)
    print('dof_inds_in', len(dof_inds_in), dof_inds_in)
    print('dof_inds_out', len(dof_inds_out), dof_inds_out)

    q_in_min = q_min[dof_inds_in]
    q_in_max = q_max[dof_inds_in]
    q_out_min = q_min[dof_inds_out]
    q_out_max = q_max[dof_inds_out]

    init_data = np.zeros(len(pin_dof_names), dtype=np.float32)
    if q_reset is not None:
        init_data[:] = q_reset[dof_inds_in]

    x_ctrl_idx_left, x_ctrl_idx_right = x_ctrl_eef_inds

    L_hand_id = robot.model.getFrameId(LINK_NAMES[x_ctrl_idx_left])
    R_hand_id = robot.model.getFrameId(LINK_NAMES[x_ctrl_idx_right])

    cmodel = cpin.Model(robot.model)
    cdata = cmodel.createData()
    nq = robot.model.nq

    # Symbolic variables
    cq = casadi.SX.sym("q", nq)
    cTf_l = casadi.SX.sym("tf_l", 4, 4)
    cTf_r = casadi.SX.sym("tf_r", 4, 4)
    c_q_last = casadi.SX.sym("q_last", nq)
    cpin.framesForwardKinematics(cmodel, cdata, cq)

    trans_err = casadi.vertcat(
            cdata.oMf[L_hand_id].translation - cTf_l[:3, 3],
            cdata.oMf[R_hand_id].translation - cTf_r[:3, 3]
        )

    rot_err = casadi.vertcat(
            cpin.log3(cdata.oMf[L_hand_id].rotation @ cTf_l[:3, :3].T),
            cpin.log3(cdata.oMf[R_hand_id].rotation @ cTf_r[:3, :3].T)
        )

    weight_translation = np.sqrt(weight_translation)
    weight_rotation = np.sqrt(weight_rotation)
    weight_regularization = np.sqrt(weight_regularization)
    weight_smoothness = np.sqrt(weight_smoothness)

    geom_err = casadi.vertcat(
        weight_translation * trans_err,
        weight_rotation * rot_err,
    )

    reg_err = weight_regularization * cq
    smooth_err = weight_smoothness * (cq - c_q_last)

    err_expr = casadi.vertcat(
        geom_err, reg_err, smooth_err,
    )

    err_fun = casadi.Function(
        "err",
        [cq, cTf_l, cTf_r, c_q_last],
        [err_expr],
    )

    J_expr = casadi.jacobian(err_expr, cq)
    J_expr = casadi.reshape(J_expr, err_expr.size1(), nq)

    J_fun = casadi.Function(
        "J_err",
        [cq, cTf_l, cTf_r, c_q_last],
        [J_expr],
    )

    dt = coalesce(dt, ctx.get('dt'), 0.02)

    def newton_multistep(
        q,
        Tf_l,
        Tf_r,
        q_last,
        err_fun,
        J_fun,
        q_min,
        q_max,
        dq_max,
        steps=3,
        lam=1e-3,
    ):
        q = q.astype(np.float32)
        Tf_l = Tf_l.astype(np.float32)
        Tf_r = Tf_r.astype(np.float32)
        q_last = q_last.astype(np.float32)

        for _ in range(steps):
            e = np.array(err_fun(q, Tf_l, Tf_r, q_last)).astype(np.float32).ravel()
            J = np.array(J_fun(q, Tf_l, Tf_r, q_last)).astype(np.float32)

            JTJ = J.T @ J
            JTe = J.T @ e

            A = JTJ + lam * np.eye(JTJ.shape[0], dtype=np.float32)

            dq = -np.linalg.solve(A, JTe)

            return dq * dt

            q = q + dq * dt
            q = np.clip(q, q_min, q_max)

        return q.astype(np.float32)

    def cb():
        if np.any(states_x_ctrl[x_ctrl_eef_inds, 3, 3] == 0):
            return
        # reduced_q_init = states_q[dof_inds_in]

        q = states_q[dof_inds_in]
        
        Tf_l = states_x_ctrl[x_ctrl_eef_inds[0]]
        Tf_r = states_x_ctrl[x_ctrl_eef_inds[1]]

        steps = 1

        q_sol = newton_multistep(
            q=q,
            Tf_l=Tf_l,
            Tf_r=Tf_r,
            q_last=init_data,
            err_fun=err_fun,
            J_fun=J_fun,
            q_min=q_in_min,
            q_max=q_in_max,
            dq_max=dq_max,
            steps=steps,
            lam=lam,
        )

        # print('sol_q', sol_q)
        # states_q_ctrl[x_ctrl_dof_inds] = states_q_ctrl[x_ctrl_dof_inds] * q_lerp + sol_q[dof_inds_out] * (1 - q_lerp)
        q_ctrl_prev = states_q_ctrl[x_ctrl_dof_inds]

        # dq = (q_sol[dof_inds_out] - q_ctrl_prev) * (1 - q_lerp)
        dq = q_sol[dof_inds_out]
        dq = np.clip(dq, -dq_max, dq_max)

        # init_data[:] = q_sol
        init_data[:] = q + q_sol

        q_ctrl = q_ctrl_prev + dq
        q_ctrl = np.clip(q_ctrl, q_out_min, q_out_max)
        states_q_ctrl[x_ctrl_dof_inds] = q_ctrl

    return cb
