import numpy as np


def cb_fk(
    states_q,
    states_x,
    states_qd=None,
    states_xd=None,
    states_pos=None,
    states_rpy=None,
    states_quat=None,
    states_J=None,
    # fix_base_link=False,
    fix_base_link=True,
    x_inds=True,
    jac_inds=True,
):
    from unicon.utils import get_ctx, topological_sort
    robot_def = get_ctx()['robot_def']
    from collections import defaultdict
    import trimesh.transformations as tra
    robot = robot_def['_URDF'].robot
    dof_names = robot_def['DOF_NAMES']
    joints = robot.joints
    links = robot.links
    num_links = len(links)
    link_names = [k.name for k in links]
    print('link_names', [(i, n) for i, n in enumerate(link_names)])
    joint_q_inds = {}
    joint_mats = {}
    link_x = {}
    link_p_inds = {}
    link_j_inds = {}
    link_topo_inds = []
    link_graph = defaultdict(list)
    for j_idx, j in enumerate(joints):
        p = j.parent
        c = j.child
        p_idx = link_names.index(p)
        c_idx = link_names.index(c)
        link_p_inds[c_idx] = p_idx
        link_j_inds[c_idx] = j_idx
        j.origin = np.eye(4) if j.origin is None else j.origin
        joint_mats[j_idx] = j.origin
        j_name = j.name
        joint_q_inds[j_idx] = dof_names.index(j_name) if j_name in dof_names else None
        link_graph[p_idx].append(c_idx)

    link_topo_inds = topological_sort(link_graph)
    link_base_idx = link_topo_inds[0]
    link_x[link_base_idx] = np.eye(4)
    has_root_states = any([x is not None for x in [states_pos, states_rpy, states_quat]])

    x_inds = False if states_x is None else x_inds
    x_inds = list(range(num_links)) if x_inds is True else x_inds
    x_inds = [] if x_inds in [False, None] else x_inds

    jac_inds = False if states_J is None else jac_inds
    jac_inds = x_inds if jac_inds is True else jac_inds
    jac_inds = [] if jac_inds in [False, None] else jac_inds

    print('cb_fk', has_root_states, fix_base_link, link_base_idx)
    print('link_graph', link_graph)
    print('link_topo_inds', link_topo_inds)

    nq = len(dof_names)
    J = np.zeros((6, nq))

    def cb():
        for j_idx, j in enumerate(joints):
            j_type = j.type
            q_idx = joint_q_inds[j_idx]
            q = 0 if q_idx is None else states_q[q_idx]
            origin = j.origin
            if j_type == 'prismatic':
                mat = origin @ tra.translation_matrix(q * j.axis)
            elif j_type == 'revolute' or j_type == 'continuous':
                mat = origin @ tra.rotation_matrix(q, j.axis)
            else:
                mat = origin
            joint_mats[j_idx] = mat

        if (not fix_base_link) and has_root_states:
            pos = states_pos if states_pos is not None else np.zeros(3)
            rpy = states_rpy if states_rpy is not None else np.zeros(3)
            kx = tra.compose_matrix(
                translate=pos,
                angles=rpy,
            )
            link_x[link_base_idx] = kx

        # for k_idx, k in enumerate(links):
        for k_idx in link_topo_inds:
            p_idx = link_p_inds.get(k_idx)
            if p_idx is None:
                continue
            j_idx = link_j_inds[k_idx]
            mat = joint_mats[j_idx]
            px = link_x[p_idx]
            kx = px @ mat
            link_x[k_idx] = kx
            states_x[k_idx] = kx
        states_x[link_base_idx] = link_x[link_base_idx]

        if not len(jac_inds):
            return

        for k_idx in jac_inds:
            J.fill(0)
            # position of link in world
            T_link = link_x[k_idx]
            p_link = T_link[:3, 3]
            # walk back through kinematic chain
            cur_idx = k_idx
            while cur_idx != link_base_idx:
                j_idx = link_j_inds[cur_idx]
                q_idx = joint_q_inds[j_idx]
                # print(k_idx, cur_idx, j_idx, q_idx, link_base_idx)
                p_idx = link_p_inds[cur_idx]
                cur_idx = p_idx
                if q_idx is None:
                    continue
                T_joint = link_x[p_idx] @ joint_mats[j_idx]
                p_joint = T_joint[:3, 3]
                R_joint = T_joint[:3, :3]
                axis = R_joint @ joints[j_idx].axis
                if joints[j_idx].type in ['revolute', 'continuous']:
                    J[:3, q_idx] = axis
                    J[3:, q_idx] = np.cross(axis, p_link - p_joint)
                elif joints[j_idx].type == 'prismatic':
                    J[:3, q_idx] = np.zeros(3)
                    J[3:, q_idx] = axis
            states_J[k_idx] = J

    return cb


def cb_fk_pin(
    states_q,
    states_x,
    states_qd=None,
    states_xd=None,
    states_J=None,
    x_inds=True,
    jac_inds=True,
):
    import pinocchio as pin

    from unicon.utils import get_ctx
    from unicon.utils.pin import load_robot_pin

    ctx = get_ctx()
    robot_def = ctx.get('robot_def')
    LINK_NAMES = robot_def.get('LINK_NAMES')
    num_links = len(LINK_NAMES)
    urdf_path = robot_def['URDF']
    robot = load_robot_pin(urdf_path)
    model, data = robot.model, robot.data
    num_frames = len(model.frames)
    frame_names = [f.name for f in model.frames]
    njoints = model.njoints

    print('njoints', njoints, num_frames, num_links)
    print('frame_names', frame_names)
    pin.forwardKinematics(model, data, states_q)
    pin.updateFramePlacements(model, data)

    x_inds = False if states_x is None else x_inds
    x_inds = ctx.get('x_ctrl_eef_inds') if x_inds is True else x_inds
    x_inds = list(range(num_links)) if x_inds is None else x_inds
    x_inds = [] if x_inds is False else x_inds

    x_inds_pin = [frame_names.index(LINK_NAMES[i]) for i in x_inds]
    print('x_inds', x_inds)
    print('x_inds_pin', x_inds_pin)

    jac_inds = False if states_J is None else jac_inds
    jac_inds = ctx.get('x_ctrl_eef_inds') if jac_inds is True else jac_inds
    jac_inds = x_inds if jac_inds is None else jac_inds
    jac_inds = [] if jac_inds is False else jac_inds

    jac_inds_pin = [frame_names.index(LINK_NAMES[i]) for i in jac_inds]
    print('jac_inds', jac_inds)
    print('jac_inds_pin', jac_inds_pin)

    ref_frame = pin.ReferenceFrame.LOCAL
    # ref_frame = pin.ReferenceFrame.WORLD

    def cb():
        if len(x_inds):
            pin.forwardKinematics(model, data, states_q)
            pin.updateFramePlacements(model, data)
            states_x[x_inds] = [data.oMf[i].homogeneous for i in x_inds_pin]
        if not len(jac_inds):
            return
        pin.computeJointJacobians(model, data, states_q)
        js = [
            pin.getFrameJacobian(model, data, i, ref_frame) for i in jac_inds_pin
        ]
        states_J[jac_inds] = js

    return cb
