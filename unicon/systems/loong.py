def cb_loong_recv_send_close(
    states_q_ctrl,
    states_rpy,
    states_ang_vel,
    states_quat,
    states_q,
    states_qd,
    states_q_tau=None,
    sdk_path=None,
    kp=None,
    kd=None,
    # launch=False,
    launch=True,
    bin_path=None,
    finger_dof=0,
    kill_on_close=False,
    filtRate=0.05,
    # filtRate=1.0,
    # finger_dof=6,
):
    import os
    import sys
    import time
    import socket
    import threading
    import numpy as np
    from unicon.utils import find, watchdog, cmd

    from unicon.utils import get_ctx
    robot_def = get_ctx()['robot_def']
    driver_out_path = '/tmp/driver.out'

    def kill_fn():
        cmd('sudo pkill -9 -ef "/[l]oong_"')
        cmd('tmux kill-session')

    if launch and cmd('pgrep -l loong_'):
        print('launching sdk')
        kill_fn()
        time.sleep(5)
        import subprocess
        if bin_path is None:
            bin_path = os.path.abspath(os.path.dirname(find(root='~', name='loong_driver')[-1]))
        print('bin_path', bin_path)
        driver_path = os.path.join(bin_path, 'loong_driver')
        interface_path = os.path.join(bin_path, 'loong_interface')
        loco_path = os.path.join(bin_path, 'loong_locomotion')
        assert os.path.exists(driver_path)
        assert os.path.exists(interface_path)
        assert os.path.exists(loco_path)
        script_str = f'''#!/bin/bash
        rm -f {driver_out_path}
        tmux kill-session -t loong
        tmux new-session -d -s loong -c '{bin_path}' '{driver_path} >{driver_out_path} 2>&1'
        tmux new-window -c '{bin_path}' '{interface_path}'
        tmux new-window -c '{bin_path}' '{loco_path}'
        tmux new-window -c '{bin_path}' 'tali -f {driver_out_path}'
        '''
        script_path = '/tmp/tmp.bash'
        with open(script_path, 'w') as f:
            f.write(script_str)
        args = [
            'sudo',
            'bash',
            script_path,
        ]
        subprocess.run(args)
        for i in range(5):
            print('waiting driver', i)
            os.system(f'grep -E "\\b561\\b" {driver_out_path} | wc -l')
            os.system(f'grep -E "\\b536\\b" {driver_out_path} | wc -l')
            time.sleep(10)
    else:
        launch = False

    def is_sdk_failed():
        if not os.path.exists(driver_out_path):
            return False
        if cmd('grep -q -E "\\b561\\b"', driver_out_path):
            print('no 561')
            return True
        for x in ['536', '592', '545']:
            if cmd(f'grep -q -E "\\b{x}\\b"', driver_out_path) == 0:
                print(f'found {x}')
                return True
        return False

    if is_sdk_failed():
        kill_fn()
        raise RuntimeError('launch failed')

    def get_cmd():
        # yapf: disable
        return bytearray([
            0x81, 0x00, 0x00, 0x00, 0x60, 0x00,
            0x00,
            *([0x00, 0x00, 0x00, 0x00,] * 11),
            0x29, 0x5c, 0xf, 0x3f,
            *([0x00, 0x00, 0x00, 0x00,] * 6),
            0x9a, 0x99, 0x19, 0x3e,
            0x00, 13, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ])
        # yapf: enable

    # tips=['en','dis','idle','damp','rc','rl','jntSdk']
    # keys=[1,13,3,12,2, 4,23]
    CMD_ENABLE = 1
    CMD_DISABLE = 13
    CMD_IDLE = 3
    CMD_DAMP = 12
    CMD_RC = 2
    CMD_RL = 4
    CMD_SDK = 23
    CMD_CLEAR = 230

    ip = '0.0.0.0'
    cmd_port = 8000
    sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd = get_cmd()

    def skLoop():
        while True:
            sk.sendto(cmd, (ip, cmd_port))
            time.sleep(0.5)

    th = threading.Thread(target=skLoop, daemon=True)
    th.start()

    def set_cmd(key):
        cmd[84] = key

    sdk_path = find(root='~', name='loong_jnt_sdk', follow_links=False)[0] if sdk_path is None else sdk_path
    sdk_path = os.path.abspath(os.path.join(sdk_path, os.path.pardir))
    print('sdk_path', sdk_path)
    sys.path.append(sdk_path)
    from loong_jnt_sdk.loong_jnt_sdk_datas import jntSdkCtrlDataClass
    jntNum = 31
    fingerDofLeft = finger_dof
    fingerDofRight = finger_dof
    use_udp = True
    if use_udp:
        from loong_jnt_sdk.loong_jnt_sdk_udp import jntSdkClass
        sdk = jntSdkClass(ip, 8006, jntNum, fingerDofLeft, fingerDofRight)
    else:
        from loong_jnt_sdk.loong_jnt_sdk_shm import jntSdkClass
        sdk = jntSdkClass(jntNum, fingerDofLeft, fingerDofRight)

    with watchdog(5):
        sdk.waitSens()

    print('rebooting motor')
    set_cmd(CMD_CLEAR)
    time.sleep(2)
    set_cmd(CMD_DAMP)
    time.sleep(5)
    set_cmd(CMD_DISABLE)
    time.sleep(2)
    set_cmd(CMD_ENABLE)
    time.sleep(2)
    set_cmd(CMD_RC)
    time.sleep(5)
    set_cmd(CMD_SDK)

    # ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_jntSdkSensDataClass__fmtSizes', '_jntSdkSensDataClass__fmts', 'acc', 'actFingerLeft', 'actFingerRight', 'actJ', 'actT', 'actW', 'drvErr', 'drvState', 'drvTemp', 'gyr', 'joy', 'key', 'planName', 'print', 'rpy', 'size', 'state', 'tgtFingerLeft', 'tgtFingerRight', 'tgtJ', 'tgtT', 'tgtW', 'timestamp', 'unpackData']
    # ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'checker', 'filtRate', 'fingerLeft', 'fingerRight', 'getStdJnt', 'j', 'kd', 'kp', 'packData', 'reset', 'size', 'state', 't', 'torLimitRate', 'w']
    ctrl = jntSdkCtrlDataClass(jntNum, fingerDofLeft, fingerDofRight)
    ctrl.reset()
    ctrl.state = 5
    ctrl.torLimitRate = 1
    ctrl.filtRate = filtRate
    # yapf: disable
    _default_kp = np.array([
        10,10,10,10,10,10,10,
        10,10,10,10,10,10,10,
        10,10,10,10,10,
        400, 200, 400, 400, 120, 120,
        400, 200, 400, 400, 120, 120,
    ], np.float32)
    _default_kd = np.array([
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1,
        2, 2, 2, 4, 0.5, 0.5,
        2, 2, 2, 4, 0.5, 0.5,
    ], np.float32)
    # yapf: enable
    kp = None
    kd = None
    kp = _default_kp if kp is None else kp
    kd = _default_kd if kd is None else kd
    ctrl.kp = kp
    ctrl.kd = kd
    ctrl.w[:] = 0.
    ctrl.t[:] = 0.
    print('ctrl.kp', ctrl.kp)
    print('ctrl.kd', ctrl.kd)

    DOF_NAMES = robot_def.get('DOF_NAMES')
    num_dofs = len(states_q)
    q_w = np.ones(num_dofs)
    q_w[DOF_NAMES.index('J_arm_l_02')] = -1.
    # q_w[DOF_NAMES.index('J_arm_l_07')] = -1.
    # q_w[DOF_NAMES.index('J_arm_r_07')] = -1.
    q_w[DOF_NAMES.index('J_head_pitch')] = -1.
    print('q_w', q_w)

    def cb_recv():
        sens = sdk.recv()
        states_q[:] = sens.actJ * q_w
        states_qd[:] = sens.actW * q_w
        states_ang_vel[:] = sens.gyr
        states_rpy[:] = sens.rpy
        if states_q_tau is not None:
            states_q_tau[:] = sens.actT

    def cb_send():
        ctrl.j[:] = states_q_ctrl * q_w
        sdk.send(ctrl)
        if is_sdk_failed():
            print('launch failed')
            return True

    def cb_close():
        set_cmd(CMD_DAMP)
        time.sleep(5)
        set_cmd(CMD_CLEAR)
        time.sleep(1)
        if launch and kill_on_close:
            kill_fn()

    return cb_recv, cb_send, cb_close
