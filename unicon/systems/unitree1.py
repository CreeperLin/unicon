import numpy as np


def cb_unitree1_recv_send_close(
    states_q_ctrl,
    states_rpy,
    states_ang_vel,
    states_quat,
    states_q,
    states_qd,
    states_q_tau=None,
    states_q_cur=None,
    states_q_temp=None,
    states_lin_vel=None,
    states_lin_acc=None,
    states_pos=None,
    states_input=None,
    highlevel=False,
    kp=None,
    kd=None,
    q_ctrl_min=None,
    q_ctrl_max=None,
    clip_q_ctrl=True,
    input_keys=None,
    send_init_cmd=True,
    dry_run=False,
    safety=False,
    power_limit=8,
    legged_type=None,
    position_limit=True,
    position_protect=False,
    sdk_path=None,
    # sdk_name='robot_interface',
    sdk_name='unitree_legged_sdk',
    **kwds,
):
    import sys
    import os
    from unicon.utils import get_ctx, find
    try:
        sdk = __import__(sdk_name)
    except ImportError:
        import sysconfig
        EXT_SUFFIX = sysconfig.get_config_var('EXT_SUFFIX')
        so_name = sdk_name + EXT_SUFFIX
        print('finding sdk_path', so_name)
        if sdk_path is None:
            sdk_path = os.path.dirname(find('~', name=so_name)[0])
        print('sdk_path', sdk_path)
        sys.path.append(sdk_path)
        sdk = __import__(sdk_name)

    robot_def = get_ctx()['robot_def']
    NAME = robot_def['NAME']
    if legged_type is None:
        if NAME == 'a1':
            legged_type = 'A1'
        elif NAME == 'aliengo':
            legged_type = 'Aliengo'
    legged_type = getattr(sdk.LeggedType, legged_type)

    num_dofs = len(states_q)
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
        'ABS_HAT0Y-': 'up',
        'ABS_HAT0X+': 'right',
        'ABS_HAT0Y+': 'down',
        'ABS_HAT0X-': 'left',
        'ABS_BRAKE': 'L2',
        'ABS_GAS': 'R2',
    }
    mapped_keys = [key_mapping.get(k) for k in input_keys]
    # print(input_keys, mapped_keys)
    mapped = [k is not None for k in mapped_keys]
    mapped_keys = [k for k in mapped_keys if k is not None]

    # constexpr int HIGHLEVEL = 0x00;
    # constexpr int LOWLEVEL = 0xff;
    # constexpr double PosStopF = (2.146E+9f);
    # constexpr double VelStopF = (16000.0f);
    # HIGHLEVEL = 0x00
    LOWLEVEL = getattr(sdk, 'LOWLEVEL', 0xff)

    sdk.InitEnvironment()
    if highlevel:
        cli_port = 8090
        cli_port = 8070
        # udp = sdk.UDP(sdk.HIGHLEVEL, sdk.HighLevelType.Sport)
        udp = sdk.UDP(cli_port, "192.168.123.161", 8082, sdk.HIGH_CMD_LENGTH, sdk.HIGH_STATE_LENGTH, -1)
        cmd_cls = sdk.HighCmd
        state_cls = sdk.HighState
        print('cmd', sdk.HIGH_CMD_LENGTH, 'state', sdk.HIGH_STATE_LENGTH)
    else:
        LOCAL_PORT = 8082
        TARGET_PORT = 8007
        TARGET_IP = "192.168.123.10"
        udp = sdk.UDP(LOWLEVEL, sdk.HighLevelType.Basic)
        # udp = sdk.UDP(sdk.LOWLEVEL, sdk.HighLevelType.Sport)
        cmd_cls = sdk.LowCmd
        state_cls = sdk.LowState
        print('cmd', sdk.LOW_CMD_LENGTH, 'state', sdk.LOW_STATE_LENGTH)
    cmd = cmd_cls()
    state = state_cls()
    if safety:
        safety = sdk.Safety(legged_type)
    udp.InitCmdData(cmd)
    if send_init_cmd:
        udp.SetSend(cmd)
        udp.Send()

    def input_fn():
        rem = bytearray(state.wirelessRemote)
        # print('rem', list(rem))
        bumper = rem[2]
        btns = rem[3]
        R1, L1, START, SELECT, R2, L2 = [(bumper >> i) & 0x1 for i in range(6)]
        A, B, X, Y = [(btns >> i) & 0x1 for i in range(4)]
        lx, rx, ry, _, ly = struct.unpack('fffff', rem[4:24])
        # print(lx, rx, ry, ly)
        inputs = dict(
            lx=lx,
            rx=rx,
            ry=-ry,
            ly=-ly,
            A=A,
            B=B,
            X=X,
            Y=Y,
            R1=R1,
            L1=L1,
            START=START,
            SELECT=SELECT,
            R2=R2,
            L2=L2,
        )
        vals = list(map(inputs.get, mapped_keys))
        states_input[mapped] = vals

    def cb_recv():
        udp.Recv()
        udp.GetRecv(state)
        motorState = state.motorState[:num_dofs]
        states_q[:] = motorState['q']
        states_qd[:] = motorState['dq']
        if states_q_tau is not None:
            states_q_tau[:] = motorState['tauEst']
        if states_q_temp is not None:
            states_q_temp[:] = motorState['temperature']
        quat = state.imu.quaternion
        states_quat[3] = quat[0]
        states_quat[:3] = quat[1:]
        states_rpy[:] = state.imu.rpy
        states_ang_vel[:] = state.imu.gyroscope

    def cb_send():
        udp.InitCmdData(cmd)
        q_ctrl_clip = np.clip(states_q_ctrl, q_ctrl_min, q_ctrl_max) if clip_q_ctrl else states_q_ctrl
        motorCmd = cmd.motorCmd[:num_dofs]
        motorCmd['dq'].fill(0)
        motorCmd['q'][:] = q_ctrl_clip
        motorCmd['Kp'][:] = kp
        motorCmd['Kd'][:] = kd
        if safety:
            if power_limit:
                safety.PowerProtect(cmd, state, power_limit)
            if position_limit:
                safety.PositionLimit(cmd)
            if position_protect:
                safety.PositionProtect(cmd, state)
        if not dry_run:
            udp.SetSend(cmd)
            udp.Send()
        if states_input is not None:
            input_fn()

    return cb_recv, cb_send, None
