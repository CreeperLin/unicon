
def cb_system_xhand(
    states_hand_q=None,
    states_hand_q_ctrl=None,
    states_hand_tactile=None,
    protocol='RS485',
    port_right='/dev/ttyUSB0',
    port_left='/dev/ttyUSB1',
    baud_rate=3000000,
    iface='enp8s0',
    kp=255,
    kd=0,
    tor_max=300,
    check_errs=False,
    ignore_cli_err=True,
):
    import os
    import ctypes
    from unicon.utils import get_ctx, list2slice, expect, find, find_import_ext

    # Set up LD_LIBRARY_PATH for xhand_controller library
    sdk_dir = find(root='~', name='xhand_control_sdk')[0]
    print('sdk_dir', sdk_dir)
    lib_dir = os.path.join(sdk_dir, 'lib')
    LD_LIBRARY_PATH = os.environ.get('LD_LIBRARY_PATH', '')
    LD_LIBRARY_PATH = lib_dir + os.pathsep + LD_LIBRARY_PATH
    print('LD_LIBRARY_PATH', LD_LIBRARY_PATH)
    os.environ['LD_LIBRARY_PATH'] = LD_LIBRARY_PATH
    ctypes.CDLL(None)

    # import sys
    # pybind_dir = find(root='~', name='xhand_controller')[0]
    # pybind_dir = os.path.join(pybind_dir, 'xhand_controller')
    # print('pybind_dir', pybind_dir)
    # sys.path.append(pybind_dir)
    # import xhand_controller
    # from xhand_controller import xhand_control
    xhand_control = find_import_ext('xhand_control')

    ctx = get_ctx()
    hand_def = ctx['robot_def']['hand_def']
    DOF_NAMES = hand_def['DOF_NAMES']
    Q_RESET = hand_def['Q_RESET']

    sides = ['left', 'right']
    ports = {'left': port_left, 'right': port_right}
    device_specs = []

    for side in sides:
        dof_names_side = [n for n in DOF_NAMES if f'{side}_hand' in n]
        num_dofs_side = len(dof_names_side)
        dof_inds_side = [DOF_NAMES.index(n) for n in dof_names_side]
        dof_inds_sl_side = list2slice(dof_inds_side)
        expect(num_dofs_side == 12)
        print(f'num_dofs_side {side}:', num_dofs_side)
        print(f'dof_inds_sl_side {side}:', dof_inds_sl_side)

        # _error = xhand_control.ErrorStruct()
        device = xhand_control.XHandControl()

        if protocol == 'RS485':
            ret = device.open_serial(ports[side], baud_rate)
        elif protocol == 'EtherCAT':
            ret = device.open_ethercat(iface)
        else:
            raise ValueError(f'Unsupported protocol: {protocol}')

        if not ret:
            print(f'xhand {side} connect error {ports[side]}')
            expect(ignore_cli_err, 'xhand device failed')
            continue

        hand_ids = device.list_hands_id()
        if not hand_ids:
            print(f'No xhand {side} device found')
            expect(ignore_cli_err, 'no xhand devices')
            continue

        device_id = hand_ids[0]
        print(f'xhand {side} device_id: {device_id}')

        err, info = device.read_device_info(device_id)
        if err.error_code == 0:
            sn = ''.join(info.serial_number[0:16])
            print(f'xhand {side} sn: {sn}')

        device_specs.append((device, device_id, dof_inds_sl_side))

    if not device_specs:
        print('No xhand devices initialized')
        expect(ignore_cli_err, 'no xhand devices initialized')
        return None, None, None

    hand_commands = []
    for device, device_id, _ in device_specs:
        hand_command = xhand_control.HandCommand_t()
        for i in range(num_dofs_side):
            hand_command.finger_command[i].id = i
            hand_command.finger_command[i].kp = 0
            hand_command.finger_command[i].kd = 0
            hand_command.finger_command[i].position = 0
            hand_command.finger_command[i].tor_max = 0
            hand_command.finger_command[i].mode = 0
        hand_commands.append(hand_command)

    def cb_recv():
        for (device, device_id, dof_inds_sl), hand_command in zip(device_specs, hand_commands):
            err, state = device.read_state(device_id, False)
            if err.error_code != 0:
                if check_errs:
                    print(f'xhand {device_id} read_state error: {err.error_message}')
                continue

            q_side = states_hand_q[dof_inds_sl]
            for finger_state in state.finger_state:
                i = finger_state.id
                q_side[i] = finger_state.position

    def cb_send():
        q_ctrl = states_hand_q_ctrl
        for (device, device_id, dof_inds_sl), hand_command in zip(device_specs, hand_commands):
            q_ctrl_side = q_ctrl[dof_inds_sl]
            for i in range(num_dofs_side):
                hand_command.finger_command[i].id = i
                hand_command.finger_command[i].kp = kp
                hand_command.finger_command[i].kd = kd
                hand_command.finger_command[i].position = q_ctrl_side[i]
                hand_command.finger_command[i].tor_max = tor_max
                hand_command.finger_command[i].mode = 3

            device.send_command(device_id, hand_command)

    def cb_close():
        for (device, device_id, dof_inds_sl), hand_command in zip(device_specs, hand_commands):
            q_reset_side = Q_RESET[dof_inds_sl]
            for i in range(num_dofs_side):
                hand_command.finger_command[i].id = i
                hand_command.finger_command[i].kp = 0
                hand_command.finger_command[i].kd = 0
                hand_command.finger_command[i].position = q_reset_side[i]
                hand_command.finger_command[i].tor_max = 0
                hand_command.finger_command[i].mode = 0

            device.send_command(device_id, hand_command)
            device.close_device()

    return cb_recv, cb_send, cb_close
