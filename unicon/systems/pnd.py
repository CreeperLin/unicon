_ip2name_act = {
    '10.10.10.70': 'hipPitch_Left',
    '10.10.10.71': 'hipRoll_Left',
    '10.10.10.72': 'hipYaw_Left',
    '10.10.10.73': 'kneePitch_Left',
    '10.10.10.74': 'anklePitch_Left',
    '10.10.10.75': 'ankleRoll_Left',
    '10.10.10.50': 'hipPitch_Right',
    '10.10.10.51': 'hipRoll_Right',
    '10.10.10.52': 'hipYaw_Right',
    '10.10.10.53': 'kneePitch_Right',
    '10.10.10.54': 'anklePitch_Right',
    '10.10.10.55': 'ankleRoll_Right',
    '10.10.10.90': 'waistRoll',
    '10.10.10.91': 'waistPitch',
    '10.10.10.92': 'waistYaw',
    '10.10.10.10': 'shoulderPitch_Left',
    '10.10.10.11': 'shoulderRoll_Left',
    '10.10.10.12': 'shoulderYaw_Left',
    '10.10.10.13': 'elbow_Left',
    '10.10.10.14': 'wristYaw_Left',
    '10.10.10.15': 'wristPitch_Left',
    '10.10.10.16': 'wristRoll_Left',
    '10.10.10.17': 'gripper_Left',
    '10.10.10.30': 'shoulderPitch_Right',
    '10.10.10.31': 'shoulderRoll_Right',
    '10.10.10.32': 'shoulderYaw_Right',
    '10.10.10.33': 'elbow_Right',
    '10.10.10.34': 'wristYaw_Right',
    '10.10.10.35': 'wristPitch_Right',
    '10.10.10.36': 'wristRoll_Right',
    '10.10.10.37': 'gripper_Right',
}
_ip2name_abs = {f'{k[:8]}.{int(k[-2:])+10}': 'ABS_' + v for k, v in _ip2name_act.items()}


def cb_pnd_recv_send_close(
    states_q_ctrl,
    states_rpy,
    states_ang_vel,
    states_quat,
    states_q,
    states_qd,
    states_q_tau=None,
    kp=None,
    kd=None,
    q_ctrl_min=None,
    q_ctrl_max=None,
    clip_q_ctrl=True,
    input_keys=None,
    # joint_pd_config=None,
    joint_pd_config=True,
    # err_exit=False,
    err_exit=True,
    compute_quat=False,
    zero_roll_kp=True,
    reboot=True,
    abort_timeout=0.1,
):
    import os
    import time
    import numpy as np
    import requests
    import json
    from unicon.utils import cmd

    print('_ip2name_act', _ip2name_act)
    print('_ip2name_abs', _ip2name_abs)

    cmd('sudo chmod 555 -R /root/.adam')
    cmd('sudo chmod 555 /root')
    cmd('sudo chmod 777 /dev/ttyUSB0')
    # cmd('sudo rm -rf /tmp/log')
    cmd('sudo chmod 777 -R /tmp/log')

    joint_abs_config = '/root/.adam/joint_abs_config.json'
    with open(joint_abs_config, 'r') as f:
        joint_abs_config = json.load(f)

    import pnd_py

    power_info = requests.get('http://localhost:8086/table_refresh', timeout=5).json()
    print('power_info', power_info)
    assert power_info['bt_capacity'] > 20

    from unicon.utils import get_host_ip
    ip = get_host_ip()
    print('host_ip', ip)
    if reboot:
        print('servo off')
        print(requests.post(f'http://{ip}:8626/robot_command/actuator_power', json={'on': 'false'}, timeout=5).text)
        time.sleep(3)

    res = requests.get(f'http://{ip}:8626/robot_command/actuator_status', json={'on': 'true'}, timeout=5).json()
    if not res['data']['actuator_status']:
        print('servo on')
        print(requests.post(f'http://{ip}:8626/robot_command/actuator_power', json={'on': 'true'}, timeout=5).text)
        time.sleep(11)

    from unicon.utils import rpy2quat_np
    from unicon.utils import get_ctx
    robot_def = get_ctx()['robot_def']

    abs_json_path = 'abs.json'
    lib_path = pnd_py.__file__
    build_dir = os.path.dirname(lib_path)
    proj_dir = os.path.dirname(build_dir)
    script_path = os.path.join(proj_dir, 'python_scripts')
    print('script_path', script_path)

    for i in range(3):
        cmd('rm -rf source', [abs_json_path])
        assert not os.path.exists(abs_json_path)
        os.mkdir('source')
        cmd('python3', [f'{script_path}/read_abs.py'], capture_output=None)
        time.sleep(1)
        cmd('python3', [f'{script_path}/read_abs.py'])
        res = cmd('python3', [f'{script_path}/check_abs.py'], capture_output=True)
        res = res.stdout.strip()
        if res == 'True':
            break
        print(i, res)
        time.sleep(5)
    else:
        raise RuntimeError('get abs failed')
    cmd('cp source/abs.json', [abs_json_path])

    assert os.path.exists(abs_json_path)
    with open(abs_json_path, 'r') as f:
        abs_json = json.load(f)
    print('abs_json', {_ip2name_abs[k]: v.get('radian') for k, v in abs_json.items()})

    # return

    so_path = 'libpnd.so.1.5.1'
    if not os.path.exists(so_path):
        _so_path = 'src/pnd-cpp-sdk/pnd/lib/linux_x86_64/libpnd.so.1.5.1'
        so_path_orig = os.path.join(proj_dir, _so_path)
        cmd('ln -s', [so_path_orig, so_path])

    pnd_py.global_init()
    pcfg = pnd_py.PConfig.getInst()
    joint_names = pcfg.jointNames()
    print('joint_names', len(joint_names), joint_names)
    num_joints = len(joint_names)
    num_dofs = len(states_q)
    assert num_dofs == num_joints, f'{num_dofs}, {num_joints}'

    d = pnd_py.RobotData()
    print('kRobotDof', pnd_py.kRobotDof)
    print('kRobotDataSize', pnd_py.kRobotDataSize)
    sz = pnd_py.kRobotDataSize
    ofs = sz - pnd_py.kRobotDof
    data_inds = slice(ofs, ofs + num_dofs)
    assert num_dofs == pnd_py.kRobotDof
    zeros_sz = np.zeros(sz)
    d.q_d_ = zeros_sz
    d.q_dot_d_ = zeros_sz
    d.tau_d_ = zeros_sz
    d.clip_qd_ = False
    d.pos_mode_ = True
    d.error_state_ = False

    r = pnd_py.RealRobot()

    pnd_py.PndStateEstimateInit()
    cmd('rm -f', [so_path])

    _kp = None
    _kd = None
    if joint_pd_config is True:
        _pd_path = 'robot_interface/config/joint_pd_config.json'
        joint_pd_config = os.path.join(proj_dir, _pd_path)
    if joint_pd_config is not None and os.path.exists(joint_pd_config):
        print('using joint_pd_config', joint_pd_config)
        _kp = np.zeros(len(joint_names))
        _kd = np.zeros(len(joint_names))
        with open(joint_pd_config, 'r') as f:
            joint_pd_config = json.load(f)
        for i, n in enumerate(joint_names):
            _kp[i] = joint_pd_config[n]['control_config']['Kp']
            _kd[i] = joint_pd_config[n]['control_config']['Kd']
    if _kp is None and kp is not None:
        _kp = kp.copy()
        _kd = kd.copy()

    if zero_roll_kp:
        _kp[joint_names.index('ankleRoll_Right')] = 0.
        _kp[joint_names.index('ankleRoll_Left')] = 0.
    print('kp', _kp)
    print('kd', _kd)
    if _kp is not None:
        r.joint_Kp_ = _kp
        r.joint_Kd_ = _kd

    import socket
    import json
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.5)
    motor_port = 2334
    enc_port = 2334
    enc_port_new = 2561

    def motors_get(msg, ips):
        msg = json.dumps(msg).encode()
        reps = []
        for ip in ips:
            s.sendto(msg, (ip, motor_port))
            try:
                for i in range(10):
                    data, addr = s.recvfrom(1024)
                    if addr[0] == ip:
                        break
                rep = json.loads(data.decode())
            except socket.timeout:
                rep = None
            print(f"ip: {ip} rep: {rep}")
            reps.append(rep)
        return reps

    def encs_get(msg, ips):
        msg = json.dumps(msg).encode()
        reps = []
        for ip in ips:
            try:
                sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sk.settimeout(0.03)
                sk.connect((ip, enc_port))
                sk.send(msg)
                data = sk.recvfrom(1024)[0]
                rep = json.loads(data.decode())
                sk.close()
            except Exception as e:
                print(e)
                rep = None
            print(f"ip: {ip} rep: {rep}")
            reps.append(rep)
        return reps

    def encs_get_new(msg, ips):
        msg = json.dumps(msg).encode()
        reps = []
        for ip in ips:
            s.sendto(msg, (ip, enc_port_new))
            try:
                for i in range(10):
                    data, addr = s.recvfrom(1024)
                    if addr[0] == ip:
                        break
                rep = json.loads(data.decode())
            except socket.timeout:
                rep = None
            print(f"ip: {ip} rep: {rep}")
            reps.append(rep)
        return reps

    MOTOR_IPS = robot_def.get('MOTOR_IPS', [])
    motor_ips = MOTOR_IPS
    msg = {
        "method": "GET",
        "reqTarget": "/",
    }
    reps = motors_get(msg, motor_ips)
    assert (all([r is not None and r['status'] == 'OK' and r['motor_drive_ready'] for r in reps]))
    msg = {
        "method": "GET",
        "reqTarget": "/m1/encoder/is_ready",
    }
    reps = motors_get(msg, motor_ips)
    assert (all([r is not None and r['status'] == 'OK' and r['property'] for r in reps]))

    msg = {
        "id": 1,
        "method": "Encoder.Angle",
        "params": "",
    }
    enc_ips = [f'{k[:8]}.{int(k[-2:])+10}' for k in MOTOR_IPS]
    # enc_ips = motor_ips
    reps = encs_get(msg, enc_ips)

    if not any(reps):
        msg_new = {"id": 0, "method": "encoder.angle"}
        reps = encs_get_new(msg_new, enc_ips)

    print('joint_abs_config', {k: v['absolute_pos_zero'] for k, v in joint_abs_config.items()})

    r.init()

    _q_ctrl = np.zeros(sz)

    def cb_recv():
        if abort_timeout is not None:
            t0 = time.monotonic()
            r.getState(0, d)
            if time.monotonic() - t0 > abort_timeout:
                print('abort_timeout', abort_timeout)
                return True
        else:
            r.getState(0, d)
        # print(d.imu_data_)
        states_rpy[:] = d.imu_data_[[2, 1, 0]]
        # states_quat[:] = 0
        if compute_quat:
            states_quat[:] = rpy2quat_np(states_rpy)
        states_ang_vel[:] = d.imu_data_[[3, 4, 5]]
        states_q[:] = d.q_a_[data_inds]
        states_qd[:] = d.q_dot_a_[data_inds]
        if states_q_tau is not None:
            states_q_tau[:] = d.tau_a_[data_inds]
        if d.error_state_ and err_exit:
            print('error_state_')
            return True

    def cb_send():
        _q_ctrl[data_inds] = states_q_ctrl
        d.q_d_ = _q_ctrl
        d.q_dot_d_ = zeros_sz
        d.tau_d_ = zeros_sz
        # print('d.q_d_', d.q_d_)
        # print('d.q_dot_d_', d.q_dot_d_)
        # print('d.tau_d_', d.tau_d_)
        # r.setCommand(d)
        if abort_timeout is not None:
            t0 = time.monotonic()
            r.setCommand(d)
            if time.monotonic() - t0 > abort_timeout:
                print('abort_timeout', abort_timeout)
                return True
        else:
            r.setCommand(d)

    def cb_close():
        # cmd('rm -rf source', [abs_json_path])
        r.disableAllJoints()
        time.sleep(1)
        r.disableAllJoints()

    return cb_recv, cb_send, cb_close
