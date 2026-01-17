def cb_unitree_recv_send_close(
    states_q_ctrl,
    states_rpy,
    states_ang_vel,
    states_quat,
    states_q,
    states_qd,
    states_input=None,
    states_q_tau=None,
    states_q_temp=None,
    kp=None,
    kd=None,
    input_keys=None,
    iface='eth0',
    lowcmd_topic='rt/lowcmd',
    lowstate_topic='rt/lowstate',
    msg_type='hg',
    mode_machine=None,
    mode_pr=0,
    sim=False,
    disable_services=True,
):
    import time
    from unicon.utils import get_ctx
    ctx = get_ctx()
    robot_def = ctx['robot_def']
    NAME = robot_def.get('NAME')
    num_dofs = len(states_q)
    motor_inds = list(range(num_dofs))
    if NAME == 'h1':
        msg_type = 'go'
    if NAME == 'go2':
        msg_type = 'go'
    assert kp is not None and kd is not None
    from unitree_sdk2py.core.channel import (
        ChannelPublisher,
        ChannelSubscriber,
        ChannelFactoryInitialize,
    )
    from unitree_sdk2py.utils.crc import CRC
    if msg_type == 'go':
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
        low_cmd_cls = LowCmdGo
        low_state_cls = LowStateGo
        low_cmd_def_cls = unitree_go_msg_dds__LowCmd_

        crc = CRC()
        crc_fn = lambda x: crc._crc_ctypes(crc._CRC__PackLowCmd(x))

        from unicon.utils.unitree import detect_state_err_go
        detect_err_fn = detect_state_err_go

    elif msg_type == 'hg':
        # g1 and h1_2 use the hg msg type
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
        low_cmd_cls = LowCmdHG
        low_state_cls = LowStateHG
        low_cmd_def_cls = unitree_hg_msg_dds__LowCmd_

        crc = CRC()
        crc_fn = lambda x: crc._crc_ctypes(crc._CRC__PackHGLowCmd(x))

        from unicon.utils.unitree import detect_state_err_hg
        detect_err_fn = detect_state_err_hg

    input_keys = ctx.get('input_keys') if input_keys is None else input_keys
    from unicon.utils.unitree import get_key_mapping, unpack_wireless_remote
    key_mapping = get_key_mapping()
    mapped_keys = [key_mapping.get(k) for k in input_keys]
    mapped = [k is not None for k in mapped_keys]
    mapped_keys = [k for k in mapped_keys if k is not None]
    print('input_keys', input_keys, mapped_keys)

    domain_id = 0
    if sim:
        iface = 'lo'
        domain_id = 1

    ChannelFactoryInitialize(
        domain_id,
        networkInterface=iface,
    )
    pub = ChannelPublisher(lowcmd_topic, low_cmd_cls)
    pub.Init()
    sub = ChannelSubscriber(lowstate_topic, low_state_cls)
    sub.Init(None, 0)
    cmd = low_cmd_def_cls()
    crc_fn(cmd)
    state = None
    go_motor_modes = {
        'h1': [0x0A] * 9 + [0x01] * 11,
        'go2': 0x01,
        # 'h1_2': [0x0A, 0x0A, 0x0A, 0x0A, 0x01, 0x01] * 2 + [0x0A] + [0x01] * 14,
    }
    if msg_type == 'go':
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        PosStopF = 2.146e9
        VelStopF = 16000.0
        motor_cmd = cmd.motor_cmd
        modes = go_motor_modes.get(NAME, 0x0A)
        modes = ([modes] * num_dofs) if isinstance(modes, int) else modes
        for i, c in enumerate(motor_cmd):
            c.mode = modes[i]
            c.q = PosStopF
            c.qd = VelStopF
            c.kp = 0.
            c.kd = 0.
            c.tau = 0.
    elif msg_type == 'hg':
        cmd.mode_machine = mode_machine
        cmd.mode_pr = mode_pr
        motor_cmd = cmd.motor_cmd
        for i, c in enumerate(motor_cmd):
            c.mode = 0x01
            c.q = 0.
            c.qd = 0.
            c.kp = 0.
            c.kd = 0.
            c.tau = 0.

    if disable_services:
        from unicon.utils.unitree import disable_motion, disable_lidar
        print('disabling motion')
        disable_motion()
        print('disabling lidar')
        disable_lidar()

        from unicon.utils.unitree import pkill_services
        pkill_services()

    for i in range(10):
        print('waiting for sub', i)
        state = sub.Read()
        print(state.tick)
        if state.tick > 0:
            break
        time.sleep(1)
    else:
        raise RuntimeError('sub read timeout')

    if msg_type == 'hg':
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MainBoardState_
        mbs_sub = ChannelSubscriber('rt/lf/mainboardstate', MainBoardState_)
        mbs_sub.Init(None, 0)
        mbs = mbs_sub.Read()
        print('mainboardstate', mbs)
        errs = detect_err_fn(low_state=state, mainboard_state=mbs)
    else:
        errs = detect_err_fn(low_state=state)
    if errs:
        raise RuntimeError('state err', errs)

    if msg_type == 'hg' and mode_machine is None:
        mode_machine = state.mode_machine
        cmd.mode_machine = mode_machine
        print('mode_machine', mode_machine)

    motor_cmd = cmd.motor_cmd
    for i, mi in enumerate(motor_inds):
        c = motor_cmd[mi]
        c.kp = kp[i]
        c.kd = kd[i]
        c.qd = 0.
        c.tau = 0.

    motor_cmd_ctrl = [motor_cmd[mi] for mi in motor_inds]

    def input_fn():
        inputs = unpack_wireless_remote(state.wireless_remote)
        vals = list(map(inputs.get, mapped_keys))
        states_input[mapped] = vals
        for i, k in enumerate(input_keys):
            if k.startswith('ABS_HAT'):
                neg = inputs.get(key_mapping[k + '-'], 0)
                pos = inputs.get(key_mapping[k + '+'], 0)
                states_input[i] = -neg + pos

    def cb_recv():
        nonlocal state
        state = sub.Read()
        if state is None:
            return True
        motor_state = state.motor_state
        motor_states_ctrl = [motor_state[mi] for mi in motor_inds]
        for i, s in enumerate(motor_states_ctrl):
            states_q[i] = s.q
            states_qd[i] = s.dq
        if states_q_tau is not None:
            for i, s in enumerate(motor_states_ctrl):
                states_q_tau[i] = s.tau_est

        quat = state.imu_state.quaternion
        states_quat[3] = quat[0]
        states_quat[:3] = quat[1:]
        states_rpy[:] = state.imu_state.rpy
        states_ang_vel[:] = state.imu_state.gyroscope
        # if states_input is not None:
        # input_fn()

    def cb_send():
        q_ctrl = states_q_ctrl
        for i, c in enumerate(motor_cmd_ctrl):
            # c = motor_cmd[mi]
            # c.kp = kp[i]
            # c.kd = kd[i]
            c.q = q_ctrl[i]
            # c.qd = 0
            # c.tau = 0
        cmd.crc = crc_fn(cmd)
        # print(cmd)
        pub.Write(cmd)

        if states_input is not None:
            input_fn()
        motor_state = state.motor_state
        motor_states_ctrl = [motor_state[mi] for mi in motor_inds]
        if states_q_temp is not None:
            for i, s in enumerate(motor_states_ctrl):
                temp = s.temperature
                temp = temp[1] if isinstance(temp, list) else temp
                states_q_temp[i] = temp
        errs = detect_err_fn(low_state=state)
        if len(errs):
            print('errs', errs)
            return True

    def cb_close():
        motor_cmd = cmd.motor_cmd
        for c in motor_cmd:
            c.kp = 0.
            c.kd = 0.
            c.mode = 0x00
        pub.Write(cmd)
        pub.Close()

    return cb_recv, cb_send, cb_close
