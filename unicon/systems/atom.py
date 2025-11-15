def cb_atom_recv_send_close(
    states_q_ctrl,
    states_rpy,
    states_ang_vel,
    states_quat,
    states_q,
    states_qd,
    states_q_tau=None,
    states_q_temp=None,
    states_hand_q=None,
    states_hand_q_ctrl=None,
    kp=None,
    kd=None,
    sdk_path=None,
    sdk_name='robot_control_dds',
    recv_upper=True,
    new_fsm_id=2,
    reboot=True,
    close=True,
    rpc_ip='192.168.8.234',
    rpc_port=51234,
    # switch_upper_limb_control=True,
    switch_upper_limb_control=False,
    init_q_th=1.0,
    init_bat_th=2000,
    use_schunk_hands=False,
    use_hands=None,
    kill_algs=False,
):
    import numpy as np
    import time
    import sys
    import os
    os.environ['CYCLONEDDS_URI'] = '/dobot/userdata/project/dds/cyclonedds.xml'
    os.environ['CYCLONEDDS_HOME'] = '/home/dobot/third_party/cyclonedds-0.10.5'
    os.environ['LD_LIBRARY_PATH'] = '/home/dobot/third_party/cyclonedds-0.10.5/lib'
    import json
    import socket
    from unicon.utils import get_ctx, find, list2slice, expect

    robot_def = get_ctx()['robot_def']
    NAME = robot_def['NAME']
    DOF_NAMES = robot_def['DOF_NAMES']
    num_dofs = len(states_q)

    dof_names_lower = [n for n in DOF_NAMES if any(x in n for x in ['hip', 'knee', 'ankle'])]
    dof_names_upper = [n for n in DOF_NAMES if any(x in n for x in ['waist', 'shoulder', 'elbow', 'wrist', 'head'])]
    num_dofs_lower = len(dof_names_lower)
    num_dofs_upper = len(dof_names_upper)
    print('num_dofs_lower', num_dofs_lower)
    print('num_dofs_upper', num_dofs_upper)
    dof_inds_lower = [DOF_NAMES.index(n) for n in dof_names_lower]
    dof_inds_upper = [DOF_NAMES.index(n) for n in dof_names_upper]
    dof_inds_lower_sl = list2slice(dof_inds_lower)
    dof_inds_upper_sl = list2slice(dof_inds_upper)
    print('dof_inds_lower_sl', dof_inds_lower_sl)
    print('dof_inds_upper_sl', dof_inds_upper_sl)

    try:
        sdk = __import__(sdk_name)
    except ImportError:
        sdk = None

    if sdk is None:
        import sys
        if sdk_path is None:
            sdk_path = os.path.dirname(find('~', name=sdk_name)[0])
        print('sdk_path', sdk_path)
        sys.path.append(sdk_path)
        sys.path.append(os.path.join(sdk_path, 'robot_control_dds'))

    from robot_control_dds.atom.msg import dds_

    from dataclasses import dataclass

    import cyclonedds.idl as idl
    import cyclonedds.idl.annotations as annotate
    import cyclonedds.idl.types as types

    from cyclonedds.domain import DomainParticipant, Domain
    from cyclonedds.topic import Topic
    from cyclonedds.sub import DataReader
    from cyclonedds.pub import DataWriter
    from cyclonedds.util import duration
    from cyclonedds.idl import IdlStruct

    timeout = duration(seconds=5)

    @dataclass
    @annotate.final
    @annotate.autoid('sequential')
    class EnableMotors_(idl.IdlStruct, typename='dobot_atom.msg.dds_.EnableMotors_'):
        flag: types.array[bool, num_dofs]

    @dataclass
    @annotate.final
    @annotate.autoid('sequential')
    class PowerState_(idl.IdlStruct, typename='dobot_atom.msg.dds_.PowerState_'):
        power_supply_state: types.uint16
        strong_power_current: types.uint16
        weak_power_current: types.uint16
        weak_power_voltage: types.uint16
        replenishment_port_voltage: types.uint16
        hardware_version: types.uint16
        software_version: types.uint16
        error_codes: types.uint16
        heartbeat: types.uint16
        strong_power_voltage: types.uint16
        brk_2v5_voltage: types.uint16
        brk_state: types.uint16

    def get_motor_cmd():
        return dds_.MotorCmd_(
            mode=1,
            q=0.,
            dq=0.,
            tau=0.,
            kp=0,
            kd=0,
        )

    participant = DomainParticipant()

    lower_state_topic = Topic(participant, 'rt/lower/state', dds_.LowerState_)
    lower_cmd_topic = Topic(participant, 'rt/lower/cmd', dds_.LowerCmd_)
    lower_state_reader = DataReader(participant, lower_state_topic)
    lower_cmd_writer = DataWriter(participant, lower_cmd_topic)

    upper_state_topic = Topic(participant, 'rt/upper/state', dds_.UpperState_)
    upper_cmd_topic = Topic(participant, 'rt/upper/cmd', dds_.UpperCmd_)
    upper_state_reader = DataReader(participant, upper_state_topic)
    upper_cmd_writer = DataWriter(participant, upper_cmd_topic)

    cmd_msg_lower = dds_.LowerCmd_(motor_cmd=[get_motor_cmd() for _ in range(num_dofs_lower)],)
    cmd_msg_upper = dds_.UpperCmd_(motor_cmd=[get_motor_cmd() for _ in range(num_dofs_upper)],)

    def clear_cmd_msg(msg):
        for c in msg.motor_cmd:
            c.mode = 0
            c.q = 0.
            c.dq = 0.
            c.tau = 0.
            c.kp = 0.
            c.kd = 0.

    # enable_motor_topic = Topic(participant, 'rt/enable/motors', EnableMotors_)
    # enable_motor_cmd_writer = DataWriter(participant, enable_motor_topic)

    os.system(f'/dobot/debug/bin/clearErrors')

    def toggle_motors(enabled=True):
        print('toggle_motors', enabled)
        # clear_cmd_msg(cmd_msg_lower)
        # lower_cmd_writer.write(cmd_msg_lower)
        # clear_cmd_msg(cmd_msg_upper)
        # upper_cmd_writer.write(cmd_msg_upper)
        os.system(f'/dobot/debug/bin/clearErrors')
        os.system(f'/dobot/debug/bin/clearUpperCmds')
        os.system(f'/dobot/debug/bin/clearLowerCmds')
        os.system(f'/dobot/debug/bin/enableMotors {1 if enabled else 0}')
        print('')
        # enable_motor_cmd_writer.write(EnableMotors_([enabled] * num_dofs))

    main_state_topic = Topic(participant, 'rt/main/nodes/state', dds_.MainNodesState_)
    main_state_reader = DataReader(participant, main_state_topic)

    power_state_topic = Topic(participant, 'rt/power/state', PowerState_)
    power_state_reader = DataReader(participant, power_state_topic)

    axis_keys = [
        'left_leg',
        'right_leg',
        'left_arm',
        'right_arm',
        'head',
        'waist',
    ]

    def check_main_msg_errs(msg):
        attr_keys = ['ecat2can'] + axis_keys
        attrs = [getattr(msg, x) for x in attr_keys]
        attrs[-1] = [attrs[-1]]
        for k in ['error_code', 'pos_err_code', 'vel_err_code', 'torque_err_code', 'warn_code']:
            errs = [[getattr(xx, k, False) for xx in x] for x in attrs]
            if any(any(x) for x in errs):
                print('errs', k, errs)
                print('attr_keys', attr_keys)
                if k == 'warn_code':
                    continue
                return True

    def check_main_msg_enabled(msg):
        attr_keys = axis_keys
        attrs = [getattr(msg, x) for x in attr_keys]
        attrs[-1] = [attrs[-1]]
        sts = [[xx.servo_state for xx in x] for x in attrs]
        if any(any(map(lambda y: y != 3, x)) for x in sts):
            print('sts', sts)
            print('attr_keys', attr_keys)
            return True

    def recv_all(sock, buffer_size=1024):
        data = b''
        while True:
            part = sock.recv(buffer_size)
            if not part:
                break
            data += part
            if len(part) < buffer_size:
                break
        return data

    def rpc(method, **params):
        request = {
            'jsonrpc': '2.0',
            'method': method,
            'params': params,
            'id': 1,
        }
        request_str = json.dumps(request)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((rpc_ip, rpc_port))
                s.sendall(request_str.encode())
                response_bytes = recv_all(s)
                response_str = response_bytes.decode()
                response = json.loads(response_str)
                print('rpc', method, params, response)
                if 'result' in response:
                    return False
                if 'error' in response:
                    return True
                print('rpc unexpected response format')
                return True
        except Exception as e:
            print('rpc exception', e)
            return True

    if reboot:
        toggle_motors(False)
        time.sleep(3)

    msg = main_state_reader.take_one(timeout=timeout)
    if check_main_msg_enabled(msg):
        toggle_motors(True)
        time.sleep(3)

    for c, idx in zip(cmd_msg_upper.motor_cmd, dof_inds_upper):
        c.mode = 1
        c.q = 0.
        c.dq = 0.
        c.tau = 0.
        if kp is not None:
            c.kp = kp[idx]
        if kd is not None:
            c.kd = kd[idx]

    for c, idx in zip(cmd_msg_lower.motor_cmd, dof_inds_lower):
        c.mode = 1
        c.q = 0.
        c.dq = 0.
        c.tau = 0.
        if kp is not None:
            c.kp = kp[idx]
        if kd is not None:
            c.kd = kd[idx]

    if use_hands is None:
        use_hands = states_hand_q is not None
    if use_hands:
        if use_schunk_hands:
            hand_state_topic = Topic(participant, 'rt/hands/schunk/state', dds_.HandsSchunkState_)
            hand_cmd_topic = Topic(participant, 'rt/hands/schunk/cmd', dds_.HandsSchunkCmd_)
            hand_state_reader = DataReader(participant, hand_state_topic)
            hand_cmd_writer = DataWriter(participant, hand_cmd_topic)
            cmd_msg_hand = dds_.HandsSchunkCmd_(hands=[get_motor_cmd() for _ in range(18)],)
        else:
            hand_state_topic = Topic(participant, 'rt/hands/state', dds_.HandsState_)
            hand_cmd_topic = Topic(participant, 'rt/hands/cmd', dds_.HandsCmd_)
            hand_state_reader = DataReader(participant, hand_state_topic)
            hand_cmd_writer = DataWriter(participant, hand_cmd_topic)
            cmd_msg_hand = dds_.HandsCmd_(hands=[get_motor_cmd() for _ in range(12)],)
        hand_kp = 0.0
        hand_kd = 0.0
        for idx, c in enumerate(cmd_msg_hand.hands):
            c.mode = 11
            c.q = 0.
            c.dq = 0.
            c.tau = 0.
            c.kp = hand_kp
            c.kd = hand_kd
        num_dofs_hand = len(states_hand_q)
        hand_msg = hand_state_reader.take_one()
        num_dofs_hand_msg = len(hand_msg.hands)
        print('num_dofs_hand_msg', num_dofs_hand_msg)
        expect(num_dofs_hand == num_dofs_hand_msg)

    print('cmd_msg_upper', cmd_msg_upper)
    print('cmd_msg_lower', cmd_msg_lower)

    upper_msg = upper_state_reader.take_one()
    num_dofs_upper_msg = len(upper_msg.motor_state)
    print('num_dofs_upper_msg', num_dofs_upper_msg)
    expect(num_dofs_upper == num_dofs_upper_msg)

    lower_msg = lower_state_reader.take_one()
    num_dofs_lower_msg = len(lower_msg.motor_state)
    print('num_dofs_lower_msg', num_dofs_lower_msg)
    expect(num_dofs_lower == num_dofs_lower_msg)

    print('lower_msg.imu_state', lower_msg.imu_state)
    print('lower_msg.bms_state', lower_msg.bms_state)
    bat_level = lower_msg.bms_state.battery_level
    print('bat_level', bat_level)
    expect(bat_level > init_bat_th, 'battery low')
    motor_state = lower_msg.motor_state
    q = np.array([m.q for m in motor_state])
    q_raw = np.array([m.q_raw for m in motor_state])
    print('lower_msg.motor_state.q', q)
    print('lower_msg.motor_state.q_raw', q_raw)
    cond = np.abs(q) > init_q_th
    expect(not np.any(cond), ('joint init q', np.where(cond)))

    msg = power_state_reader.take_one(timeout=timeout)
    print('power_state', msg)
    expect(msg.error_codes == 0)

    msg = main_state_reader.take_one(timeout=timeout)
    if msg is None:
        return None
    if check_main_msg_errs(msg):
        return None
    if check_main_msg_enabled(msg):
        return None

    if new_fsm_id is not None:
        fsm_topic = Topic(participant, 'rt/set/fsm/id', dds_.SetFsmId_)
        # fsm_cmd_writer = DataWriter(participant, fsm_topic)
        fsm_reader = DataReader(participant, fsm_topic)
        fsm_msg = fsm_reader.take_one(timeout=timeout)
        print('fsm_msg', fsm_msg)
        # cmd_fsm = type(fsm_msg)(new_fsm_id)
        # cmd_fsm.id = new_fsm_id
        # fsm_msg.id = new_fsm_id
        # print('cmd_fsm', type(cmd_fsm), cmd_fsm)
        # fsm_cmd_writer.write(cmd_fsm)
        # fsm_cmd_writer.write(fsm_msg)
        if fsm_msg.id != new_fsm_id:
            rpc('SetFsmId', fsm_id=new_fsm_id)
        for _ in range(100):
            time.sleep(1)
            fsm_msg = fsm_reader.take_one(timeout=timeout)
            print('fsm_msg', fsm_msg)
            if fsm_msg.id == new_fsm_id:
                break
            print('fsm state != 2: press LB + A then LT + RT')
        expect(fsm_msg.id == new_fsm_id)

    if switch_upper_limb_control is not None:
        expect(not rpc('SwitchUpperLimbControl', is_on=False))

    if kill_algs:
        os.system('sudo pkill -ef "humanoid"')

    def cb_recv():
        if recv_upper:
            msg = upper_state_reader.take_next()
            if msg is not None:
                motor_state = msg.motor_state
                q = [m.q for m in motor_state]
                dq = [m.dq for m in motor_state]
                states_q[dof_inds_upper_sl] = q
                states_qd[dof_inds_upper_sl] = dq
                if states_q_tau is not None:
                    tau_est = [m.tau_est for m in motor_state]
                    states_q_tau[dof_inds_upper_sl] = tau_est
                if states_q_temp is not None:
                    motor_temp = [m.motor_temp for m in motor_state]
                    states_q_temp[dof_inds_upper_sl] = motor_temp
        msg = lower_state_reader.take_next()
        if msg is not None:
            motor_state = msg.motor_state
            q = [m.q for m in motor_state]
            dq = [m.dq for m in motor_state]
            states_q[dof_inds_lower_sl] = q
            states_qd[dof_inds_lower_sl] = dq
            if states_q_tau is not None:
                tau_est = [m.tau_est for m in motor_state]
                states_q_tau[dof_inds_lower_sl] = tau_est
            if states_q_temp is not None:
                motor_temp = [m.motor_temp for m in motor_state]
                states_q_temp[dof_inds_lower_sl] = motor_temp
            imu = msg.imu_state
            quat = imu.quaternion
            states_quat[3] = quat[0]
            states_quat[:3] = quat[1:]
            states_rpy[:] = imu.rpy
            states_ang_vel[:] = imu.gyroscope

    def cb_send():
        q_ctrl = states_q_ctrl

        for c, idx in zip(cmd_msg_lower.motor_cmd, dof_inds_lower):
            c.q = q_ctrl[idx]
        lower_cmd_writer.write(cmd_msg_lower)

        for c, idx in zip(cmd_msg_upper.motor_cmd, dof_inds_upper):
            c.q = q_ctrl[idx]
        upper_cmd_writer.write(cmd_msg_upper)

        msg = main_state_reader.take_next()
        if msg is None:
            return
        if check_main_msg_errs(msg):
            return True

        if not use_hands:
            return
        q_ctrl = states_hand_q_ctrl
        for idx, c in enumerate(cmd_msg_hand.hands):
            c.q = q_ctrl[idx]
        hand_cmd_writer.write(cmd_msg_hand)

        msg = hand_state_reader.take_next()
        if msg is not None:
            motor_state = msg.hands
            q = [m.q for m in motor_state]
            # dq = [m.dq for m in motor_state]
            states_hand_q[:] = q
            # states_hand_qd[:] = dq

    def cb_close():
        if close:
            toggle_motors(False)

    return cb_recv, cb_send, cb_close
