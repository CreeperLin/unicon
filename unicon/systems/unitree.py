import numpy as np


def cb_unitree_recv_send_close(
    states_q_ctrl,
    states_rpy,
    states_ang_vel,
    states_quat,
    states_q,
    states_qd,
    states_input=None,
    states_q_tau=None,
    states_q_cur=None,
    states_lin_vel=None,
    states_lin_acc=None,
    states_pos=None,
    kp=None,
    kd=None,
    q_ctrl_min=None,
    q_ctrl_max=None,
    clip_q_ctrl=True,
    input_keys=None,
    network_interface='eth0',
    lowcmd_topic='rt/lowcmd',
    lowstate_topic='rt/lowstate',
    msg_type='hg',
    mode_machine=None,
    mode_pr=0,
    robot_def=None,
    **states,
):
    NAME = robot_def.get('NAME')
    num_dofs = len(states_q)
    motor_inds = range(num_dofs)
    if NAME == 'h1':
        msg_type = 'go'
        motor_inds = list(range(num_dofs + 1))
        motor_inds.pop(9)
    assert kp is not None and kd is not None
    from unitree_sdk2py.core.channel import (
        ChannelPublisher,
        ChannelSubscriber,
        ChannelFactoryInitialize,
    )
    # from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
    if msg_type == 'go':
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
        from unitree_sdk2py.utils.crc import CRC
        low_cmd_cls = LowCmdGo
        low_state_cls = LowStateGo
        low_cmd_def_cls = unitree_go_msg_dds__LowCmd_
    elif msg_type == 'hg':
        # g1 and h1_2 use the hg msg type
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
        from unitree_sdk2py.utils.crc import CRC
        low_cmd_cls = LowCmdHG
        low_state_cls = LowStateHG
        low_cmd_def_cls = unitree_hg_msg_dds__LowCmd_

    import struct
    if input_keys is None:
        from unicon.inputs import _default_input_keys
        input_keys = _default_input_keys
    key_mapping = {
        'ABS_X': 'lx',
        'ABS_Y': 'ly',
        'ABS_RX': 'rx',
        'ABS_RY': 'ry',
        'BTN_A': 'A',
        'BTN_B': 'B',
        'BTN_X': 'X',
        'BTN_Y': 'Y',
        'BTN_TL': 'L1',
        'BTN_TR': 'R1',
    }
    mapped_keys = [key_mapping.get(k) for k in input_keys]
    mapped = [k is not None for k in mapped_keys]
    mapped_keys = [k for k in mapped_keys if k is not None]

    ChannelFactoryInitialize(networkInterface=network_interface,)
    pub = ChannelPublisher(lowcmd_topic, low_cmd_cls)
    pub.Init()
    sub = ChannelSubscriber(lowstate_topic, low_state_cls)
    sub.Init(None, 10)
    cmd = low_cmd_def_cls()
    state = None
    crc = CRC()
    if msg_type == 'go':
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0
        PosStopF = 2.146e9
        VelStopF = 16000.0
        motor_cmd = cmd.motor_cmd
        if NAME == 'h1':
            for c in motor_cmd[:9]:
                c.mode = 0x0A
                c.q = PosStopF
                c.qd = VelStopF
                c.kp = 0
                c.kd = 0
                c.tau = 0
            for c in motor_cmd[10:20]:
                c.mode = 0x01
                c.q = PosStopF
                c.qd = VelStopF
                c.kp = 0
                c.kd = 0
                c.tau = 0
        else:
            for i, c in enumerate(motor_cmd):
                c.mode = 0x0A
                c.q = PosStopF
                c.qd = VelStopF
                c.kp = 0
                c.kd = 0
                c.tau = 0
    elif msg_type == 'hg':
        cmd.mode_machine = mode_machine
        cmd.mode_pr = mode_pr
        motor_cmd = cmd.motor_cmd
        for i, c in enumerate(motor_cmd):
            c.mode = 1
            c.q = 0
            c.qd = 0
            c.kp = 0
            c.kd = 0
            c.tau = 0
    state_rem_keys = [
        'R1',
        'L1',
        'START',
        'SELECT',
        'R2',
        'L2',
        'F1',
        'F2',
        'A',
        'B',
        'X',
        'Y',
        'up',
        'right',
        'down',
        'left',
    ]

    import time
    while True:
        state = sub.Read()
        print(state.tick)
        if state.tick > 0:
            break
        time.sleep(1)

    if msg_type == 'hg' and mode_machine is None:
        mode_machine = state.mode_machine
        cmd.mode_machine = mode_machine
        print('mode_machine', mode_machine)

    def input_fn():
        rem = bytearray(state.wireless_remote)
        unpacked_data = struct.unpack('<2B H 5f 16B', rem)

        lx = unpacked_data[3]
        ly = unpacked_data[7]
        rx = unpacked_data[4]
        ry = unpacked_data[5]

        value = unpacked_data[2]
        inputs = dict(lx=lx, rx=rx, ry=-ry, ly=-ly)
        for i, k in enumerate(state_rem_keys):
            inputs[k] = (value & (1 << i)) >> i
        # print(inputs)
        vals = list(map(inputs.get, mapped_keys))
        states_input[mapped] = vals

    def cb_recv():
        nonlocal state
        state = sub.Read()
        if state is None:
            return True
        motor_state = state.motor_state
        for i, mi in enumerate(motor_inds):
            s = motor_state[mi]
            states_q[i] = s.q
            states_qd[i] = s.dq
        if states_q_tau is not None:
            for i, mi in enumerate(motor_inds):
                s = motor_state[mi]
                states_q_tau[i] = s.tau_est

        quat = state.imu_state.quaternion
        states_quat[3] = quat[0]
        states_quat[:3] = quat[1:]
        states_rpy[:] = state.imu_state.rpy
        states_ang_vel[:] = state.imu_state.gyroscope
        # if states_input is not None:
        # input_fn()

    def cb_send():
        # udp.InitCmdData(cmd)
        q_ctrl_clip = np.clip(states_q_ctrl, q_ctrl_min, q_ctrl_max) if clip_q_ctrl else states_q_ctrl
        motor_cmd = cmd.motor_cmd
        for i, mi in enumerate(motor_inds):
            c = motor_cmd[mi]
            c.kp = kp[i]
            c.kd = kd[i]
            c.q = q_ctrl_clip[i]
            c.qd = 0
            c.tau = 0
        cmd.crc = crc.Crc(cmd)
        # print(cmd)
        pub.Write(cmd)
        if states_input is not None:
            input_fn()

    def cb_close():
        motor_cmd = cmd.motor_cmd
        for c in motor_cmd:
            c.kp = 0
            c.kd = 0
            c.mode = 0x00
        pub.Write(cmd)
        pub.Close()

    return cb_recv, cb_send, cb_close
