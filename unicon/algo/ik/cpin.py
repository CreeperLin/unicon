import numpy as np


def cb_ik_cpin(
    states_q_ctrl,
    states_q,
    states_x_ctrl,
    dt=None,
    x_ctrl_dof_inds=None,
    x_ctrl_eef_inds=None,
    reduce=True,
    q_lerp=0.5,
    weight_translation=50.0,
    weight_rotation=1.0,
    weight_regularization=0.02,
    weight_smoothness=0.1,
    qd_max=5.,
    solver_opts=None,
    solver_preset='normal',
):
    # adapted from: https://github.com/unitreerobotics/xr_teleoperate
    import casadi
    from pinocchio import casadi as cpin

    from unicon.utils import get_ctx, coalesce, pats2inds
    from unicon.utils.numpy import mat4_eye
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

    q_min_in = q_min[dof_inds_in]
    q_max_in = q_max[dof_inds_in]
    q_min_x_ctrl = q_min[x_ctrl_dof_inds]
    q_max_x_ctrl = q_max[x_ctrl_dof_inds]

    init_data = np.zeros(len(pin_dof_names), dtype=np.float32)
    if q_reset is not None:
        init_data[:] = q_reset[dof_inds_in]

    x_ctrl_idx_left, x_ctrl_idx_right = x_ctrl_eef_inds

    L_hand_id = robot.model.getFrameId(LINK_NAMES[x_ctrl_idx_left])
    R_hand_id = robot.model.getFrameId(LINK_NAMES[x_ctrl_idx_right])

    cmodel = cpin.Model(robot.model)
    cdata = cmodel.createData()

    # Symbolic variables
    nq = robot.model.nq
    cq = casadi.SX.sym("q", nq)
    cTf_l = casadi.SX.sym("tf_l", 4, 4)
    cTf_r = casadi.SX.sym("tf_r", 4, 4)
    # c_q_last = casadi.SX.sym("q_last", robot.model.nq)
    cpin.framesForwardKinematics(cmodel, cdata, cq)

    # Define error functions
    translational_error = casadi.Function(
        "translational_error",
        [cq, cTf_l, cTf_r],
        [casadi.vertcat(
            cdata.oMf[L_hand_id].translation - cTf_l[:3, 3],
            cdata.oMf[R_hand_id].translation - cTf_r[:3, 3]
        )],
    )

    rotational_error = casadi.Function(
        "rotational_error",
        [cq, cTf_l, cTf_r],
        [casadi.vertcat(
            cpin.log3(cdata.oMf[L_hand_id].rotation @ cTf_l[:3, :3].T),
            cpin.log3(cdata.oMf[R_hand_id].rotation @ cTf_r[:3, :3].T)
        )],
    )

    # Setup optimization problem
    opti = casadi.Opti()
    var_q = opti.variable(nq)
    param_q_last = opti.parameter(nq)
    param_tf_l = opti.parameter(4, 4)
    param_tf_r = opti.parameter(4, 4)

    # Cost function
    translational_cost = casadi.sumsqr(
        translational_error(var_q, param_tf_l, param_tf_r)
    )
    rotation_cost = casadi.sumsqr(
        rotational_error(var_q, param_tf_l, param_tf_r)
    )
    regularization_cost = casadi.sumsqr(var_q)
    smooth_cost = casadi.sumsqr(var_q - param_q_last)

    # Constraints
    opti.subject_to(opti.bounded(
        robot.model.lowerPositionLimit,
        var_q,
        robot.model.upperPositionLimit,
    ))

    # Objective
    opti.minimize(
        weight_translation * translational_cost +
        weight_rotation * rotation_cost +
        weight_regularization * regularization_cost +
        weight_smoothness * smooth_cost
    )

    opts_normal = {
        'expand': True,
        'detect_simple_bounds': True,
        'calc_lam_p': False,
        'print_time': False,
        'ipopt.sb': 'yes',
        'ipopt.print_level': 0,
        'ipopt.max_iter': 30,
        'ipopt.tol': 1e-4,
        'ipopt.acceptable_tol': 5e-4,
        'ipopt.acceptable_iter': 5,
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.derivative_test': 'none',
        'ipopt.jacobian_approximation': 'exact',
    }
    opts_fast = {
        'expand': False,
        'detect_simple_bounds': True,
        'calc_lam_p': False,
        'print_time': False,
        'ipopt.sb': 'yes',
        'ipopt.print_level': 0,
        'ipopt.max_iter': 3,
        'ipopt.tol': 1e9,
        'ipopt.acceptable_tol': 1e9,
        'ipopt.acceptable_iter': 0,

        'ipopt.warm_start_init_point': 'yes',
        'ipopt.warm_start_bound_push': 1e-6,
        'ipopt.warm_start_mult_bound_push': 1e-6,
        'ipopt.derivative_test': 'none',
        'ipopt.jacobian_approximation': 'exact',
        'ipopt.linear_solver': 'mumps',
    }
    presets = {
        'normal': opts_normal,
        'fast': opts_fast,
    }
    opts = presets[solver_preset]
    solver_opts = {} if solver_opts is None else solver_opts
    opts.update(solver_opts)
    opti.solver("ipopt", opts)

    reduced_q_init = states_q[dof_inds_in]
    opti.set_initial(var_q, reduced_q_init)
    opti.set_value(param_tf_l, mat4_eye())
    opti.set_value(param_tf_r, mat4_eye())
    opti.set_value(param_q_last, init_data)

    opti.solve_limited()

    def cb():
        if np.any(states_x_ctrl[x_ctrl_eef_inds, 3, 3] == 0):
            return

        reduced_q_init = states_q[dof_inds_in]
        opti.set_initial(var_q, reduced_q_init)
        opti.set_value(param_tf_l, states_x_ctrl[x_ctrl_idx_left])
        opti.set_value(param_tf_r, states_x_ctrl[x_ctrl_idx_right])
        opti.set_value(param_q_last, init_data)

        try:
            sol = opti.solve_limited()
        except RuntimeError as e:
            print(f"[CasadiIK] Solver failed: {e}")
            return
        status = opti.stats()['return_status']
        if status != 'Solve_Succeeded':
            sol_q = opti.debug.value(var_q)
        else:
            sol_q = opti.value(var_q)
        init_data[:] = sol_q
        q_ctrl_prev = states_q_ctrl[x_ctrl_dof_inds]
        dq = (sol_q[dof_inds_out] - q_ctrl_prev) * (1 - q_lerp)
        dq = np.clip(dq, -dq_max, dq_max)
        q_ctrl = q_ctrl_prev + dq
        q_ctrl = np.clip(q_ctrl, q_min_x_ctrl, q_max_x_ctrl)
        states_q_ctrl[x_ctrl_dof_inds] = q_ctrl

    return cb
