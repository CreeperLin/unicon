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
    robot_def=None,
    zero_roll_kp=True,
    **states,
):
    import os
    import time
    import numpy as np
    import pnd_py
    import subprocess
    import requests

    power_info = requests.get('http://localhost:8086/table_refresh').json()
    print('power_info', power_info)
    assert power_info['bt_capacity'] > 20

    from unicon.utils import rpy2quat_np

    abs_json_path = 'abs.json'
    lib_path = pnd_py.__file__
    build_dir = os.path.dirname(lib_path)
    proj_dir = os.path.dirname(build_dir)
    script_path = os.path.join(proj_dir, 'python_scripts')
    print('script_path', script_path)
    os.system('sudo chmod 777 /dev/ttyUSB0')
    os.system('sudo rm -rf /tmp/log')

    for i in range(3):
        os.system(f'rm -rf source {abs_json_path}')
        assert not os.path.exists(abs_json_path)
        os.mkdir('source')
        os.system(f'python3 {script_path}/read_abs.py')
        time.sleep(1)
        os.system(f'python3 {script_path}/read_abs.py')
        res = subprocess.run(
            f'python3 {script_path}/check_abs.py',
            shell=True,
            stdout=subprocess.PIPE,
        )
        res = res.stdout.decode().strip()
        if res == 'True':
            break
        print('get abs failed', i)
        time.sleep(5)
    assert res == 'True', res
    os.system(f'cp source/abs.json {abs_json_path}')

    assert os.path.exists(abs_json_path)

    so_path = 'libpnd.so.1.5.1'
    if not os.path.exists(so_path):
        _so_path = 'src/pnd-cpp-sdk/pnd/lib/linux_x86_64/libpnd.so.1.5.1'
        so_path_orig = os.path.join(proj_dir, _so_path)
        os.system(f'ln -s {so_path_orig} {so_path}')

    pnd_py.global_init()
    pcfg = pnd_py.PConfig.getInst()
    joint_names = pcfg.jointNames()
    print(joint_names)
    num_joints = len(joint_names)
    num_dofs = len(states_q)
    assert num_dofs == num_joints

    d = pnd_py.RobotData()
    print('kRobotDof', pnd_py.kRobotDof)
    print('kRobotDataSize', pnd_py.kRobotDataSize)
    sz = pnd_py.kRobotDataSize
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
    os.system(f'rm -f {so_path}')

    _kp = None
    _kd = None
    if joint_pd_config is True:
        _pd_path = 'robot_interface/config/joint_pd_config.json'
        joint_pd_config = os.path.join(proj_dir, _pd_path)
    if joint_pd_config is not None and os.path.exists(joint_pd_config):
        print('using joint_pd_config', joint_pd_config)
        _kp = np.zeros(len(joint_names))
        _kd = np.zeros(len(joint_names))
        import json
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
    port = 2334

    def motors_get(msg, ips):
        msg = json.dumps(msg).encode()
        reps = []
        for ip in ips:
            s.sendto(msg, (ip, port))
            for i in range(10):
                data, addr = s.recvfrom(1024)
                if addr[0] == ip:
                    break
            rep = json.loads(data.decode())
            print(f"ip: {ip} rep: {rep}")
            reps.append(rep)
        return reps

    MOTOR_IPS = robot_def.get('MOTOR_IPS', [])
    ips = MOTOR_IPS
    msg = {
        "method": "GET",
        "reqTarget": "/",
    }
    reps = motors_get(msg, ips)
    assert (all([r['status'] == 'OK' and r['motor_drive_ready'] == True for r in reps]))
    msg = {
        "method": "GET",
        "reqTarget": "/m1/encoder/is_ready",
    }
    reps = motors_get(msg, ips)
    assert (all([r['status'] == 'OK' and r['property'] == True for r in reps]))

    r.init()

    _q_ctrl = np.zeros(sz)

    def cb_recv():
        r.getState(0, d)
        # print(d.imu_data_)
        states_rpy[:] = d.imu_data_[[2, 1, 0]]
        # states_quat[:] = 0
        if compute_quat:
            states_quat[:] = rpy2quat_np(states_rpy)
        states_ang_vel[:] = d.imu_data_[[3, 4, 5]]
        states_q[:] = d.q_a_[-num_dofs:]
        states_qd[:] = d.q_dot_a_[-num_dofs:]
        if states_q_tau is not None:
            states_q_tau[:] = d.tau_a_[-num_dofs:]
        if d.error_state_ and err_exit:
            print('error_state_')
            return True

    def cb_send():
        _q_ctrl[-num_dofs:] = states_q_ctrl
        d.q_d_ = _q_ctrl
        d.q_dot_d_ = zeros_sz
        d.tau_d_ = zeros_sz
        # print('d.q_d_', d.q_d_)
        # print('d.q_dot_d_', d.q_dot_d_)
        # print('d.tau_d_', d.tau_d_)
        r.setCommand(d)

    def cb_close():
        r.disableAllJoints()
        time.sleep(1)
        r.disableAllJoints()

    return cb_recv, cb_send, cb_close
