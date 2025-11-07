



# 全局参数
COLOR_LEFT_ON = [0, 0, 1, 1]
COLOR_LEFT_OFF = [0, 0, 0.3, 0.5]
COLOR_RIGHT_ON = [0, 1, 0, 1]
COLOR_RIGHT_OFF = [0, 0.3, 0, 0.5]
ARROW_RADIUS = 0.01
ARROW_LENGTH = 0.15
SPHERE_RADIUS = 0.05


def cb_send_swv(
    states_q,
    states_quat=None,
    states_rpy=None,
    states_pos=None,
    host='localhost',
    port=6000,
    start_server=True,
    init_z=1.0,
    use_rpy=False,
    rot=True,
    verbose=False,
    **states,
):
    import numpy as np
    import time
    import subprocess
    from sim_web_visualizer import MeshCatVisualizerBase
    from sim_web_visualizer.parser.yourdfpy import URDF
    import meshcat.geometry as g
    import meshcat.transformations as tf
    from unicon.utils import get_ctx, quat2mat_np, rpy2mat_np, try_conn
    from scipy.spatial.transform import Rotation as R

    robot_def = get_ctx()['robot_def']
    dof_names = robot_def.get('DOF_NAMES')
    urdf_path = robot_def.get('URDF')

    if use_rpy or states_quat is None:
        print('swv rpy')
        states_quat = None
    else:
        states_rpy = None
        print('swv quat')

    if not rot:
        states_quat = None
        states_rpy = None

    port = int(port)
    proc = None
    if start_server and try_conn(host, port):
        print('starting meshcat-server')
        args = ['python3', '-m', 'meshcat.servers.zmqserver']
        proc = subprocess.Popen(args)
        time.sleep(2)
        ret = proc.poll()
        if ret is not None:
            proc.kill()
            raise RuntimeError('meshcat-server failed')

    viz = MeshCatVisualizerBase(port=port, host=host)

    num_states_dofs = len(states_q)
    robot = URDF.load(urdf_path, build_tree=True)
    num_dofs = robot.num_dofs
    urdf_dof_names = robot.actuated_joint_names
    print('urdf_dof_names', len(urdf_dof_names), urdf_dof_names)
    print('dof_names', len(dof_names), dof_names)
    print(num_states_dofs, num_dofs)

    dof_map = None
    if dof_names is not None:
        dof_map = [dof_names.index(n) for n in urdf_dof_names]
    print('viz dof_map', dof_map)

    viz.viz['/URDF'].delete()
    asset_resource = viz.dry_load_asset(urdf_path, collapse_fixed_joints=False)
    viz.load_asset_resources(asset_resource, '/URDF', scale=1.0)
    urdf_viz = viz.viz['/URDF']
    base_tf = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., init_z], [0., 0., 0., 1.]])
    urdf_viz.set_transform(base_tf)
    qpos = np.zeros(num_dofs)
    last_h = None
    pt = 0


    def get_arrow_transform(start, end):
        dir_vec = end - start
        length = np.linalg.norm(dir_vec)
        if length < 1e-6:
            return tf.translation_matrix(start)
        z = dir_vec / length
        # 默认cylinder沿z轴，需旋转到z方向
        # 找到旋转四元数
        axis = np.cross([0,0,1], z)
        angle = np.arccos(np.clip(np.dot([0,0,1], z), -1.0, 1.0))
        if np.linalg.norm(axis) < 1e-6:
            rotmat = np.eye(4)
        else:
            rotmat = tf.rotation_matrix(angle, axis)
        T = tf.translation_matrix(start)
        return tf.concatenate_matrices(T, rotmat)
    

    def viz_target(viz, idx, pos, quat, color):
        # idx: 0=left, 1=right
        # 目录统一为/targets/left, /targets/left_x, /targets/right, /targets/right_x
        base_path = '/targets'
        name = f'{base_path}/left' if idx == 0 else f'{base_path}/right'
        name_x = f'{base_path}/left_x' if idx == 0 else f'{base_path}/right_x'
        # 机器人base的绝对位置和姿态
        # 取当前base_tf的平移和旋转
        # base_tf为4x4齐次矩阵
        base_pos = base_tf[:3, 3]
        base_rot = base_tf[:3, :3]
        # 目标点的绝对位置 = base_pos + base_rot @ pos
        abs_pos = base_pos + base_rot @ pos
        # 目标点的绝对姿态 = base_rot @ quat
        # 四元数变换：q_abs = q_base * q_target
        from scipy.spatial.transform import Rotation as R
        base_quat = R.from_matrix(base_rot).as_quat()
        q_abs = (R.from_quat(base_quat) * R.from_quat(quat)).as_quat()
        # 球体
        viz.viz[name].set_object(g.Sphere(SPHERE_RADIUS))
        viz.viz[name].set_property('color', color)
        T = np.dot(tf.translation_matrix(abs_pos), tf.quaternion_matrix(q_abs))
        viz.viz[name].set_transform(T)
        # x方向箭头
        rot = R.from_quat(q_abs)
        x_dir = rot.apply([1, 0, 0]) * ARROW_LENGTH
        arrow_start = abs_pos
        arrow_end = abs_pos + x_dir
        viz.viz[name_x].set_object(g.Cylinder(np.linalg.norm(x_dir), ARROW_RADIUS))
        viz.viz[name_x].set_property('color', color)
        viz.viz[name_x].set_transform(get_arrow_transform(arrow_start, arrow_end))

    class cb_cls:
        def __call__(self):
            nonlocal last_h, pt, states_quat
            h = (states_q + 100).sum()
            if last_h == h:
                return
            last_h = h
            if verbose and pt % 50 == 0:
                print('states_q', pt)
            pt += 1

            qpos[:] = states_q[dof_map]
            robot.update_kinematics(qpos)
            for link_name, link in robot.link_map.items():
                pose = robot.get_link_global_transform(link_name)
                urdf_viz[link_name].set_transform(pose)
            if states_quat is not None:
                if np.sum(np.abs(states_quat)) < 1e-3:
                    states_quat = None
                else:
                    base_tf[:3, :3] = quat2mat_np(states_quat)
            if states_rpy is not None:
                base_tf[:3, :3] = rpy2mat_np(states_rpy)
            if states_pos is not None:
                base_tf[:3, 3] = states_pos
            if states_pos is not None or states_quat is not None or states_rpy is not None:
                urdf_viz.set_transform(base_tf)

            # 读取共享内存中的target和mask
            try:
                from unicon.states import states_get
                left_target = states_get('left_target_real_time')
                right_target = states_get('right_target_real_time')
                reach_mask = states_get('states_reach_mask')
            except Exception as e:
                print('states_get error:', e)
                left_target = None
                right_target = None
                reach_mask = None

            # 可视化target，只保留/targets目录下对象
            # reach_mask: [0]控制左右, [1]控制右手
            mask_left = (reach_mask is not None and reach_mask[0])
            mask_right = (reach_mask is not None and reach_mask[0] and reach_mask[1])

            # 清理独立的/left_target和/right_target等元素（只执行一次即可）
            for old_name in ['/left_target', '/right_target', '/left_target_x', '/right_target_x']:
                # if old_name in viz.viz:
                try:
                    viz.viz[old_name].delete()
                except Exception:
                    pass

            # target位置为相对机器人中心的相对位置，统一目录/targets
            if left_target is not None and len(left_target) == 7:
                color_left = COLOR_LEFT_ON if mask_left else COLOR_LEFT_OFF
                pos_left = left_target[:3]
                quat_left = left_target[3:]
                if np.sum(np.abs(quat_left)) < 1e-6:
                    quat_left = [0,0,0,1]
                viz_target(viz, 0, pos_left, quat_left, color_left)

            if right_target is not None and len(right_target) == 7:
                color_right = COLOR_RIGHT_ON if mask_right else COLOR_RIGHT_OFF
                pos_right = right_target[:3]
                quat_right = right_target[3:]
                if np.sum(np.abs(quat_right)) < 1e-6:
                    quat_right = [0,0,0,1]
                viz_target(viz, 1, pos_right, quat_right, color_right)

        def __del__(self):
            if proc is not None:
                print('killing meshcat-server')
                proc.kill()

    return cb_cls()
