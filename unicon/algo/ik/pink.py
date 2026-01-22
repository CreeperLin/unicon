import numpy as np


def cb_ik_pink(
    states_q_ctrl,
    states_q,
    states_x_ctrl,
    # damping=1e-8,
    damping=1e-3,
    dt=None,
    x_ctrl_dof_inds=None,
    x_ctrl_eef_inds=None,
    pos_cost=True,
    rot_cost=True,
    reduce=True,
    # reduce=False,
    # dv_lerp=0.0,
    dv_lerp=0.5,
    dv_scale=0.9,
    qd_max=5.,
):
    import pink
    from pink.tasks import FrameTask

    from unicon.utils import get_ctx, coalesce, pats2inds
    from unicon.utils.pin import load_robot_pin, mat2se3, dedup_frames

    ctx = get_ctx()
    robot_def = ctx.get('robot_def')
    q_min = robot_def['Q_CTRL_MIN']
    q_max = robot_def['Q_CTRL_MAX']
    LINK_NAMES = robot_def.get('LINK_NAMES')
    DOF_NAMES = robot_def.get('DOF_NAMES')
    PIN_DOF_NAMES = robot_def.get('PIN_DOF_NAMES', DOF_NAMES)
    num_links = len(LINK_NAMES)
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

    num_dofs_ik = len(x_ctrl_dof_inds)
    num_eefs_ik = len(x_ctrl_eef_inds)
    print('x_ctrl_dof_inds', num_dofs_ik, x_ctrl_dof_inds)
    print('x_ctrl_eef_inds', num_eefs_ik, x_ctrl_eef_inds)

    pos_cost = ([float(pos_cost)] * 3) if isinstance(pos_cost, (float, int, bool)) else pos_cost
    pos_cost = np.asarray(pos_cost, dtype=np.float32)
    rot_cost = ([float(rot_cost)] * 3) if isinstance(rot_cost, (float, int, bool)) else rot_cost
    rot_cost = np.asarray(rot_cost, dtype=np.float32)

    dt = coalesce(dt, ctx.get('dt'))
    dq_max = dt * qd_max

    tasks = []
    for ei in x_ctrl_eef_inds:
        task = FrameTask(LINK_NAMES[ei], pos_cost.tolist(), rot_cost.tolist())
        tasks.append(task)

    pin_dof_names = list(PIN_DOF_NAMES)
    if reduce and len(x_ctrl_dof_inds) != njoints:
        fixed_dof_names = [n for n in pin_dof_names if n not in x_ctrl_dof_names]
        print('fixed_dof_names', fixed_dof_names)
        reduced_robot = robot.buildReducedRobot(list_of_joints_to_lock=fixed_dof_names,)
        robot = reduced_robot
        pin_dof_names = [n for n in pin_dof_names if n not in fixed_dof_names]
    print('pin_dof_names', pin_dof_names)

    _, _, dof_inds_in = pats2inds(DOF_NAMES, pin_dof_names)
    _, _, dof_inds_out = pats2inds(pin_dof_names, x_ctrl_dof_names)
    print('dof_inds_in', len(dof_inds_in), dof_inds_in)
    print('dof_inds_out', len(dof_inds_out), dof_inds_out)

    q_min_x_ctrl = q_min[x_ctrl_dof_inds]
    q_max_x_ctrl = q_max[x_ctrl_dof_inds]

    cur_dv = np.zeros(len(pin_dof_names), dtype=np.float32)

    def cb():
        if np.any(states_x_ctrl[x_ctrl_eef_inds, 3, 3] == 0):
            return
        configuration = pink.Configuration(robot.model, robot.data, states_q[dof_inds_in])
        for t, ei in zip(tasks, x_ctrl_eef_inds):
            t.set_target(mat2se3(states_x_ctrl[ei]))
        dv = pink.solve_ik(
            configuration,
            tasks=tasks,
            dt=dt,
            damping=damping,
            solver="quadprog",
            safety_break=False,
        )
        cur_dv[:] = cur_dv * dv_lerp + dv * (1 - dv_lerp)
        dq = cur_dv[dof_inds_out] * (dt * dv_scale)
        dq = np.clip(dq, -dq_max, dq_max)
        q_ctrl_prev = states_q_ctrl[x_ctrl_dof_inds]
        q_ctrl = q_ctrl_prev + dq
        q_ctrl = np.clip(q_ctrl, q_min_x_ctrl, q_max_x_ctrl)
        states_q_ctrl[x_ctrl_dof_inds] = q_ctrl

    return cb
