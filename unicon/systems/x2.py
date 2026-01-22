# yapf: disable
def cb_x2_recv_send_close(
    states_q_ctrl,
    states_rpy,
    states_ang_vel,
    states_quat,
    states_q,
    states_qd,
    states_q_tau=None,
    states_q_temp=None,
    kp=None,
    kd=None,
    node_name='default',
    # use_th=True,
    use_th=False,
    restart_ros=True,
    sub_qos=None,
    pub_qos=None,
    sub_excludes=('head',),
    pub_excludes=('head',),
    compute_rpy=True,
    kill_apps=True,
):
    # yapf: enable
    import os
    import time
    from unicon.utils import get_ctx, find, list2slice, expect, ssh
    from unicon.utils import quat2rpy_np3

    try:
        import rclpy
        import aimdk_msgs
    except ImportError:
        env_dirs = [
            '/opt/ros/humble',
            '/agibot/software/housekeeper/bin/aimdk_msgs/share/aimdk_msgs/',
        ]
        print(f'need to source envs in {env_dirs}')
        raise

    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from rclpy.qos import qos_profile_sensor_data
    from aimdk_msgs.msg import JointCommandArray, JointStateArray, JointCommand
    from sensor_msgs.msg import Imu

    if restart_ros:
        print('restart_ros')
        profiles_path = '/agibot/software/entry/cfg/privileged_ros_dds_configuration.xml'
        os.environ['FASTRTPS_DEFAULT_PROFILES_FILE'] = profiles_path
        os.system('ros2 daemon stop && ros2 daemon start')

    robot_def = get_ctx()['robot_def']
    DOF_NAMES = robot_def['DOF_NAMES']

    dof_matches = [
        ['leg', ['hip', 'knee', 'ankle']],
        ['arm', ['shoulder', 'elbow', 'wrist']],
        ['waist', ['waist']],
        ['head', ['head']],
    ]
    dof_inds_map = {}
    dof_names_map = {}
    for key, pats in dof_matches:
        print(key)
        dof_names_key = [n for n in DOF_NAMES if any(x in n for x in pats)]
        num_dofs_key = len(dof_names_key)
        dof_inds_key = [DOF_NAMES.index(n) for n in dof_names_key]
        dof_inds_sl_key = list2slice(dof_inds_key)
        print('num_dofs_key', num_dofs_key)
        print('dof_inds_sl_key', dof_inds_sl_key)
        dof_inds_map[key] = dof_inds_key, dof_inds_sl_key
        dof_names_map[key] = dof_names_key

    dof_cmd_msgs = {}
    for key, (inds, _) in dof_inds_map.items():
        cmd = JointCommandArray()
        for i in inds:
            c = JointCommand()
            c.name = DOF_NAMES[i]
            c.position = 0.0
            c.velocity = 0.0
            c.effort = 0.0
            if kp is not None:
                c.stiffness = kp[i]
            if kd is not None:
                c.damping = kd[i]
            cmd.joints.append(c)
        dof_cmd_msgs[key] = cmd

    soc0_kill_list = [
        'app_proxy',
        'atop',
        'cloud_proxy',
        'drp',
        'housekeeper',
        'mqtt_broker',
        'ota',
        'teleop_bridge',
        'hds_master',
        'hds_slave',
        'aima-rc-app',
        'mc_app_main',
        'mosquitto',
        'ptp4l',
    ]
    pats = '|'.join(soc0_kill_list)
    cmd = f'sudo pkill -ef "{pats}"'
    if kill_apps:
        expect(not ssh('run@10.0.1.40', cmd, password='1'))

    rclpy.init()

    default_subscriber_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )

    default_publisher_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )

    node = rclpy.create_node(node_name)

    sub_on_recv = lambda _: None
    # def sub_on_recv(msg):
    #     print('recv', msg)
    sub_qos = default_subscriber_qos if sub_qos is None else sub_qos
    joint_sub_keys = [
        'waist',
        'arm',
        'head',
        'leg',
    ]
    if sub_excludes is not None:
        sub_excludes = sub_excludes if isinstance(sub_excludes, (list, tuple)) else sub_excludes
        joint_sub_keys = list(filter(lambda x: all(y != x for y in sub_excludes), joint_sub_keys))
    print('joint_sub_keys', joint_sub_keys)
    sub_topics = {k: f'/aima/hal/joint/{k}/state' for k in joint_sub_keys}
    subs = {k: node.create_subscription(JointStateArray, sub_topics[k], sub_on_recv, sub_qos) for k in joint_sub_keys}

    topic_imu_torso_state = '/aima/hal/imu/torso/state'
    sub_qos = qos_profile_sensor_data
    sub_imu_torso = node.create_subscription(Imu, topic_imu_torso_state, sub_on_recv, sub_qos)

    pub_qos = default_publisher_qos if pub_qos is None else pub_qos
    joint_pub_keys = [
        'leg',
        'waist',
        'arm',
        'head',
    ]
    if pub_excludes is not None:
        pub_excludes = pub_excludes if isinstance(pub_excludes, (list, tuple)) else pub_excludes
        joint_pub_keys = list(filter(lambda x: all(y != x for y in pub_excludes), joint_pub_keys))
    print('joint_pub_keys', joint_pub_keys)
    pub_topics = {k: f'/aima/hal/joint/{k}/command' for k in joint_pub_keys}
    pubs = {k: node.create_publisher(JointCommandArray, pub_topics[k], pub_qos) for k in joint_pub_keys}

    joint_sub_specs = [(subs[k], *dof_inds_map[k]) for k in joint_sub_keys]

    joint_pub_specs = [(pubs[k], dof_cmd_msgs[k], dof_inds_map[k][0]) for k in joint_pub_keys]

    # def on_shutdown(*args):
    #     print('on_shutdown', args)
    #     cb_close()

    # node.context.on_shutdown(on_shutdown)

    if use_th:

        def worker():
            rclpy.spin_until_future_complete(node, future=future)

        import threading
        import asyncio
        future = asyncio.Future()
        th = threading.Thread(target=worker, daemon=True)
        th.start()

    def take_one(sub):
        for i in range(10):
            # if not use_th:
            rclpy.spin_once(node, timeout_sec=0)
            msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
            if msg_info is not None:
                return msg_info[0]
            print('take one', i)
            time.sleep(0.5)
        return None

    for k in joint_sub_keys:
        sub = subs[k]
        dof_names_key = dof_names_map[k]
        msg = take_one(sub)
        joints = msg.joints
        joint_names = [m.name for m in joints]
        print('joint_sub_key', k)
        print('joint_names', joint_names)
        print('dof_names_key', dof_names_key)
        expect(joint_names == dof_names_key)

    def cb_recv():
        try:
            if not use_th:
                rclpy.spin_once(node, timeout_sec=0)
            for sub, dof_inds, dof_inds_sl in joint_sub_specs:
                # sub.handle.take_message(sub.msg_type, sub.raw)
                msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
                if msg_info is None:
                    continue
                msg = msg_info[0]
                joints = msg.joints
                q = [m.position for m in joints]
                dq = [m.velocity for m in joints]
                states_q[dof_inds_sl] = q
                states_qd[dof_inds_sl] = dq
                if states_q_tau is not None:
                    tau_est = [m.effort for m in joints]
                    states_q_tau[dof_inds_sl] = tau_est
                if states_q_temp is not None:
                    motor_temp = [m.motor_temp for m in joints]
                    states_q_temp[dof_inds_sl] = motor_temp

            sub = sub_imu_torso
            msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
            if msg_info is None:
                return
            msg = msg_info[0]
            quat = msg.orientation
            w = msg.angular_velocity
            # a = msg.linear_acceleration
            # oc = msg.orientation_covariance
            # wc = msg.angular_velocity_covariance
            # ac = msg.linear_acceleration_covariance
            states_quat[:] = [quat.x, quat.y, quat.z, quat.w]
            # states_rpy[:] = state.imu.rpy
            states_ang_vel[:] = [w.x, w.y, w.z]
        except RuntimeError:
            import traceback
            traceback.print_exc()
            return True
        if compute_rpy:
            states_rpy[:] = quat2rpy_np3(states_quat)

    def cb_send():
        q_ctrl = states_q_ctrl.tolist()
        try:
            for pub, msg, dof_inds in joint_pub_specs:
                for c, idx in zip(msg.joints, dof_inds):
                    c.position = q_ctrl[idx]
                t0 = time.monotonic()
                pub.publish(msg)
                if (time.monotonic() - t0 > 0.1):
                    print('pub timeout', time.monotonic() - t0, msg.joints[0].name)
        except RuntimeError:
            import traceback
            traceback.print_exc()
            return True
        # if not use_th:
        #     rclpy.spin_once(node, timeout_sec=0)

    def cb_close():
        nonlocal joint_pub_specs, node
        for pub, msg, _ in joint_pub_specs:
            for c in msg.joints:
                c.position = 0.
                c.velocity = 0.
                c.effort = 0.
                c.stiffness = 0.
                c.damping = 0.
        pub, msg, _ = joint_pub_specs[0]
        new_ctx = False
        try:
            pub.publish(msg)
        except Exception as e:
            print('ctx', e)
            new_ctx = True
        if new_ctx:
            print('new_node')
            node.destroy_node()
            new_context = rclpy.Context()
            rclpy.init(context=new_context)
            new_node = rclpy.create_node("test", context=new_context)
            new_pubs = [new_node.create_publisher(JointCommandArray, pub_topics[k], pub_qos) for k in joint_pub_keys]
            joint_pub_specs = [[pub, msg, inds] for pub, (_, msg, inds) in zip(new_pubs, joint_pub_specs)]
            node = new_node
        for _ in range(2):
            print('disabling motors')
            time.sleep(1)
            for pub, msg, _ in joint_pub_specs:
                print(len(msg.joints))
                pub.publish(msg)

        if use_th:
            future.set_result(1)
            th.join()

        node.destroy_node()
        rclpy.try_shutdown()

    return cb_recv, cb_send, cb_close
