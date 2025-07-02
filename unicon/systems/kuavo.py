_launch_str = '''<?xml version="1.0" ?>

<launch>
    <arg name="robot_version"         default="$(optenv ROBOT_VERSION 40)"/>
    <arg name="use_joystick"          default="true"/>
    <arg name="cali"                  default="false" />
    <arg name="joystick_type"       default="bt2"/>
    <arg name="start_way"           default="manual"/>
    <arg name="cali_arm"              default="false" />
    <arg name="build_cppad_state"     default="2" />

    <include file="$(find humanoid_controllers)/launch/robot_version_manager.launch">
      <arg name="robot_version" value="$(arg robot_version)"/>
    </include>

    <param name="robot_version"     value="$(arg robot_version)"/>
    <param name="joystick_type"               value="$(arg joystick_type)"/>
    
    <param name="build_cppad_state"    value="$(arg build_cppad_state)" />
    <param name="cali_arm"             value="$(arg cali_arm)" />
    <param name="cali"                 value="$(arg cali)" />
    <param name="start_way"            value="$(arg start_way)"/>

    <rosparam param="initial_state">{initial_state}</rosparam>
    <rosparam param="squat_initial_state">{squat_initial_state}</rosparam>
    <!-- nodelet manager -->
   <node pkg="nodelet" type="nodelet" name="nodelet_manager" args="manager" respawn="false" output="screen" required="true">
        <param name="num_worker_threads" type="int" value="8" />
    </node>

    <node pkg="nodelet" type="nodelet" name="nodelet_hardware" args="load HardwareNodelet nodelet_manager" respawn="false" output="screen" required="true"  />
    <include file="$(find humanoid_interface)/launch/rosbag_nodelet.launch"/>

    <group if="$(arg use_joystick)">
      <group if="$(eval arg('joystick_type') == 'h12')">
        <include file="$(find humanoid_controllers)/launch/joy/joy_control_h12.launch">
          <arg name="start_way" value="$(arg start_way)"/>
        </include>
      </group>
      <group unless="$(eval arg('joystick_type') == 'h12')">
        <include file="$(find humanoid_controllers)/launch/joy/joy_control_bt.launch">
          <arg name="joystick_type" value="$(arg joystick_type)"/>
        </include>
      </group>
    </group>
</launch>

'''


def cb_kuavo_recv_send_close(
    states_q_ctrl,
    states_rpy,
    states_ang_vel,
    states_quat,
    states_q,
    states_qd,
    states_q_tau=None,
    states_input=None,
    kp=None,
    kd=None,
    q_ctrl_min=None,
    q_ctrl_max=None,
    clip_q_ctrl=True,
    input_keys=None,
    callback_min_delay=0.01,
    control_mode=2,
    robot_def=None,
    launch=True,
    stop=None,
    use_input=False,
    # use_h12=True,
    use_h12=False,
    deploy_path=None,
    **states,
):
    import time
    import numpy as np

    import rospy
    from std_msgs.msg import Bool
    from sensor_msgs.msg import Joy
    from kuavo_msgs.msg import sensorsData, jointCmd
    from h12pro_controller_node.msg import h12proRemoteControllerChannel

    from unicon.utils import quat2rpy_np

    tau_limit = robot_def.get('TAU_LIMIT').copy()
    NAME = robot_def.get('NAME')
    robot_version = int(NAME[1:])
    print('robot_version', robot_version)
    num_dofs = len(states_q)

    proc = None
    stop = launch if stop is None else stop
    if launch:
        import os
        import subprocess
        if deploy_path is None:
            from unicon.utils import find
            deploy_path = find(root='~', name='kuavo-robot-deploy')[0]
        os.system('sudo pkill -ef roslaunch')
        time.sleep(2)
        _init_state = [0.0] * (3 * 4 + num_dofs)
        # print('_init_state', _init_state)
        launch_str = _launch_str.format(initial_state=_init_state, squat_initial_state=_init_state)
        launch_path = '/tmp/tmp.launch'
        launch_params = ''
        if use_h12:
            launch_params = 'start_way:=manual joystick_type:=h12'
        with open(launch_path, 'w') as f:
            f.write(launch_str)
        script_str = f'''#!/bin/bash
        source {deploy_path}/devel/setup.bash
        export ROBOT_VERSION={robot_version}
        echo o | roslaunch {launch_path} {launch_params}
        '''
        script_path = '/tmp/tmp.bash'
        with open(script_path, 'w') as f:
            f.write(script_str)
        args = [
            'sudo',
            'bash',
            script_path,
        ]
        proc = subprocess.Popen(args)
        for i in range(27):
            print('waiting for ros nodes', i)
            time.sleep(1)
            ret = proc.poll()
            if ret is not None:
                proc.kill()
                raise RuntimeError('roslaunch failed')

    rospy.init_node('unicon_sys_kuavo', anonymous=True)
    cmd_pub = rospy.Publisher('/joint_cmd', jointCmd, queue_size=10)
    stop_pub = rospy.Publisher('/stop_robot', Bool, queue_size=10)

    base_ang_vel = np.zeros(3, dtype=np.float32)
    # base_rpy = np.zeros(3, dtype=np.float32)
    base_quat = np.zeros(4, dtype=np.float32)
    dof_pos = np.zeros(num_dofs, dtype=np.float32)
    dof_vel = np.zeros(num_dofs, dtype=np.float32)

    print('kp', kp)
    print('kd', kd)
    print('tau_limit', tau_limit)

    last_recv_ts = 0

    def sensor_callback(msg):
        if time.monotonic() - last_recv_ts < callback_min_delay:
            return
        joint_data = msg.joint_data
        dof_pos[:] = joint_data.joint_q
        dof_vel[:] = joint_data.joint_v
        imu_data = msg.imu_data
        base_ang_vel[:] = imu_data.gyro.x, imu_data.gyro.y, imu_data.gyro.z
        base_quat[:] = imu_data.quat.x, imu_data.quat.y, imu_data.quat.z, imu_data.quat.w

    _ = rospy.Subscriber('/sensors_data_raw', sensorsData, sensor_callback)

    def joy_callback(msg):
        if time.monotonic() - last_recv_ts < callback_min_delay:
            return
        axes = msg.axes
        buttons = msg.buttons

    h12_min = 282
    h12_max = 1722
    h12_rng = h12_max - h12_min
    h12_mean = (h12_max + h12_min) // 2
    h12_mapping = [
        'ABS_RX',
        'ABS_RY',
        'ABS_Y',
        'ABS_X',
        # 'E', 'F',
        'BTN_TL',
        'BTN_TR',
        'BTN_A',
        'BTN_B',
        'BTN_X',
        'BTN_Y',
        'G',
        'H',
    ]
    h12_channels = np.zeros(len(h12_mapping))

    def h12_callback(msg):
        if time.monotonic() - last_recv_ts < callback_min_delay:
            return
        h12_channels[:] = msg.channels

    use_joy = False
    if states_input is not None and use_input:
        from unicon.inputs import _default_input_keys
        input_keys = _default_input_keys
        if use_h12:
            h12_inds = [h12_mapping.index(n) for n in input_keys if n in h12_mapping]
            input_inds = [i for i, n in enumerate(input_keys) if n in h12_mapping]
            print('h12_inds', h12_inds)
            print('input_inds', input_inds)
            _ = rospy.Subscriber('/h12pro_channel', h12proRemoteControllerChannel, h12_callback)
        else:
            _ = rospy.Subscriber('/joy', Joy, joy_callback)
            use_joy = True
    else:
        use_h12 = False

    while cmd_pub.get_num_connections() == 0:
        rospy.loginfo("Waiting for subscribers to connect...")
        rospy.sleep(1)

    ros_joint_cmd = jointCmd()

    def cb_recv():
        nonlocal last_recv_ts
        base_rpy = quat2rpy_np(base_quat)
        states_rpy[:] = base_rpy
        states_quat[:] = base_quat
        states_ang_vel[:] = base_ang_vel
        states_q[:] = dof_pos
        states_qd[:] = dof_vel
        last_recv_ts = time.monotonic()

    ros_joint_cmd.control_modes = [0.] * num_dofs
    ros_joint_cmd.joint_q = [0.] * num_dofs
    ros_joint_cmd.joint_v = [0.] * num_dofs
    ros_joint_cmd.tau = [0.] * num_dofs
    ros_joint_cmd.tau_max = [0.] * num_dofs
    ros_joint_cmd.tau_ratio = [0.] * num_dofs
    ros_joint_cmd.joint_kp = [0.] * num_dofs
    ros_joint_cmd.joint_kd = [0.] * num_dofs
    for i in range(num_dofs):
        ros_joint_cmd.control_modes[i] = control_mode
        ros_joint_cmd.joint_v[i] = 0.
        ros_joint_cmd.tau[i] = 0.
        ros_joint_cmd.tau_ratio[i] = 1.
        if kp is not None:
            ros_joint_cmd.joint_kp[i] = kp[i]
        if kd is not None:
            ros_joint_cmd.joint_kd[i] = kd[i]
        ros_joint_cmd.tau_max[i] = tau_limit[i]
    print('control_modes', ros_joint_cmd.control_modes)

    def cb_send():
        q_ctrl = states_q_ctrl
        for i in range(num_dofs):
            ros_joint_cmd.joint_q[i] = q_ctrl[i]
            ros_joint_cmd.joint_v[i] = 0.
            ros_joint_cmd.tau[i] = 0.
        cmd_pub.publish(ros_joint_cmd)
        if use_h12:
            channels = h12_channels.copy()
            channels[:4] = (channels[:4] - h12_mean) / h12_rng * 2
            channels[4:] = (channels[4:] - h12_min) / h12_rng
            states_input[input_inds] = channels[h12_inds]
        if use_joy:
            pass
        if launch:
            ret = proc.poll()
            if ret is not None:
                # raise RuntimeError('roslaunch crashed')
                print('roslaunch crashed')
                return True

    def cb_close():
        if stop:
            stop_msg = Bool()
            stop_msg.data = True
            stop_pub.publish(stop_msg)
        if launch:
            proc.wait()
            os.system('sudo pkill -ef roslaunch')
        rospy.signal_shutdown("Node shutting down")

    return cb_recv, cb_send, cb_close
