def run_rosbridge_server(host, port):
    # apt install ros-humble-rosbridge-suite
    # start the rosbridge server first:
    # ros2 launch rosbridge_server rosbridge_websocket_launch.xml
    import time
    import subprocess

    def try_conn():
        import socket
        s = socket.socket()
        fail = False
        try:
            s.connect((host, port))
        except Exception:
            fail = True
        finally:
            s.close()
        return fail

    if not try_conn():
        print('rosbridge server already running')
        return None
    args = ['ros2', 'launch', 'rosbridge_server', 'rosbridge_websocket_launch.xml', f'address:={host}', f'port:={port}']
    proc = subprocess.Popen(args)
    time.sleep(2)
    ret = proc.poll()
    if ret is not None:
        proc.kill()
        raise RuntimeError('rosbridge server failed to start')
    time.sleep(2)
    if try_conn():
        proc.kill()
        raise RuntimeError('failed connecting to rosbridge server')
    return proc


def cb_roblibpy_recv_send_close(
    states_q_ctrl,
    states_rpy,
    states_ang_vel,
    states_quat,
    states_q,
    states_qd,
    states_qd_ctrl=None,
    states_q_tau=None,
    states_q_tau_ctrl=None,
    states_q_cur=None,
    states_lin_vel=None,
    states_lin_acc=None,
    states_pos=None,
    dof_names=None,
    lin_vel_as_rpy=False,
    pos_as_rpy=True,
    sub_joint_states_topic='/sim_joint_states',
    sub_pose_states_topic='/sim_pose_states',
    sub_twist_states_topic='/sim_twist_states',
    pub_joint_commands_topic='/sim_joint_commands',
    verbose=False,
    run_server=True,
    host='127.0.0.1',
    port=9090,
    **kwds,
):
    import threading
    import time
    from roslibpy import Message, Ros, Time, Topic
    from roslibpy.ros2 import Header
    from roslibpy.core import RosTimeoutError
    if run_server:
        proc = run_rosbridge_server(host, port)

    context = {}
    context['stop'] = False

    ros = Ros(host, port)

    try:
        ros.run()
    except RosTimeoutError as e:
        import traceback
        traceback.print_exc()
        ros.close()
        raise e

    sub_ps = Topic(ros, sub_pose_states_topic, 'geometry_msgs/Pose')
    sub_ts = Topic(ros, sub_twist_states_topic, 'geometry_msgs/Twist')
    sub_js = Topic(ros, sub_joint_states_topic, 'sensor_msgs/JointState')
    pub_jcmd = Topic(ros, pub_joint_commands_topic, 'sensor_msgs/JointState')

    dof_map = {k: i for i, k in enumerate(dof_names)}

    def on_recv_sub_js(msg):
        if verbose:
            print('on_recv_sub_js', msg['header']['stamp'])
        name = msg['name']
        position = msg.get('position', [])
        velocity = msg.get('velocity', [])
        effort = msg.get('effort', [])
        f = 1
        for i, k in enumerate(name):
            idx = dof_map.get(k)
            if idx is None:
                continue
            states_q[idx] = position[i]
            states_qd[idx] = velocity[i]
            if states_q_tau is not None:
                states_q_tau[idx] = effort[i]
            f = 0
        if f:
            print('no dof set', dof_map, name)

    def on_recv_sub_ps(msg):
        if verbose:
            print('on_recv_sub_ps', msg['header']['stamp'])
        pos = msg.get('position', None)
        rot = msg.get('orientation', None)
        if rot is not None:
            states_quat[:] = rot
        if pos_as_rpy and pos is not None:
            states_rpy[:] = pos

    def on_recv_sub_ts(msg):
        if verbose:
            print('on_recv_sub_ts', msg['header']['stamp'])
        lin_vel = msg.get('linear', None)
        ang_vel = msg.get('angular', None)
        if ang_vel is not None:
            states_ang_vel[:] = ang_vel
        if lin_vel is not None and lin_vel_as_rpy:
            states_rpy[:] = lin_vel

    def th_recv_sub_js():
        sub_js.subscribe(on_recv_sub_js)
        sub_ps.subscribe(on_recv_sub_ps)
        sub_ts.subscribe(on_recv_sub_ts)

    t1 = threading.Thread(target=th_recv_sub_js)
    t1.start()

    def cb_send():
        send_name = dof_names
        send_position = states_q_ctrl
        send_velocity = states_qd_ctrl
        send_effort = states_q_tau_ctrl
        msg = dict(header=Header(stamp=Time.now(), frame_id=''))
        # print('send_position', send_position)
        msg['name'] = send_name
        msg['position'] = send_position
        msg['velocity'] = send_velocity
        msg['effort'] = send_effort
        msg = Message(msg)
        pub_jcmd.publish(msg)
        if verbose:
            print('send', msg['header']['stamp'])

    def cb_recv():
        pass

    def cb_close():
        pub_jcmd.unadvertise()
        sub_js.unsubscribe()
        context['stop'] = True
        ros.close()
        t1.join()
        # t2.join()
        if run_server and proc is not None:
            proc.terminate()
            time.sleep(1)
            proc.kill()
            proc.wait()

    return cb_recv, cb_send, cb_close
