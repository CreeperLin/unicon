import numpy as np

# _default_mmc = 100
_default_mmc = 28
_default_mma = 60000
_default_mms = 3000


def cb_fftai_recv_send_close(
    # states_prop,
    states_rpy,
    states_ang_vel,
    states_quat,
    states_q,
    states_qd,
    states_q_ctrl,
    states_q_tau=None,
    states_q_cur=None,
    states_lin_vel=None,
    states_lin_acc=None,
    states_pos=None,
    kp=None,
    kd=None,
    # radian
    q_ctrl_min=None,
    q_ctrl_max=None,
    q_ctrl_init=True,
    clip_q_ctrl=True,
    dtype=np.float64,
    verbose=False,
    dof_map=None,
    lazy_send=False,
    # lazy_send=True,
    unique_send=False,
    # lazy_recv=False,
    lazy_recv=True,
    # check_recv=False,
    check_recv=True,
    rcs_config=None,
    sensor_offset='/home/gr1/sensor_offset.json',
    update_offset=False,
    set_control_params=False,
    motor_max_current=16,
    motor_max_acceleration=60000,
    motor_max_speed=3000,
    enable_motors=True,
    # enable_motors=False,
    enable_imu=True,
    # enable_imu=False,
    enable_bias=True,
    strict=False,
    imu_device='/dev/ttyUSB0',
    wait=1,
    # default_pd=False,
    default_pd=True,
    zero_pd=False,
    fsa_time_out=0.01,
    init_send_zeros=False,
    use_pd_control=False,
    reboot=True,
    **kwds,
):
    import time
    import socket
    import json
    import sys
    import struct
    import os
    from unicon.utils import find
    fsa_path = find(root='~', name='Wiki-FSA')[0]
    fsa_path = os.path.join(fsa_path, 'sdk-python/v3')
    print('fsa_path', fsa_path)
    sys.path.append(fsa_path)
    import fi_fsa
    sys.argv = sys.argv[:1]
    if rcs_config is not None:
        sys.argv.extend(['--rcs_config', rcs_config])
    s = fi_fsa.s
    fsa_port_fast = fi_fsa.fsa_port_fast
    fsa_port_ctrl = fi_fsa.fsa_port_ctrl
    fsa_network = fi_fsa.fsa_network
    fsa_port_comm = fi_fsa.fsa_port_comm
    json_buf_size = 1024
    # buf_size = 64
    buf_size = 32

    from unicon.utils import get_ctx
    robot_def = get_ctx()['robot_def']
    fsa_sign = robot_def.FSA_SIGN
    fsa_ips = robot_def.FSA_IPS
    fse_ips = robot_def.FSE_IPS
    fse2fsa = robot_def.get('FSE2FSA', None)
    default_position_kp = robot_def.get('FSA_POSITION_KP', None)
    default_velocity_kp = robot_def.get('FSA_VELOCITY_KP', None)
    # actuator_types = robot_def.get('FSA_ACTUATOR_TYPES', None)

    if reboot:
        from unicon.utils.fftai import reboot_fsa, fsa_broadcast
        # fsa_ips = fsa_broadcast(fi_fsa, 'Actuator')
        num_ips = len(fsa_ips)
        print('rebooting fsa', num_ips)
        for i, ip in enumerate(fsa_ips):
            reboot_fsa(fi_fsa, ip)
        __import__('time').sleep(13)
        _fsa_ips = fsa_broadcast(fi_fsa, 'Actuator', max_ips=num_ips)
        assert num_ips == len(_fsa_ips), f'{num_ips} != {len(_fsa_ips)}'

    if isinstance(sensor_offset, str) and sensor_offset.endswith('.json'):
        sensor_offset_path = sensor_offset
        if os.path.exists(sensor_offset_path):
            with open(sensor_offset, 'r') as f:
                sensor_offset = json.load(f)
        else:
            sensor_offset = None

    def broadcast_func_with_filter(filter_type=None):
        found_server = False
        address_list = []
        s.sendto("Is any fourier smart server here?".encode("utf-8"), (fsa_network, fsa_port_comm))
        while True:
            try:
                data, address = s.recvfrom(json_buf_size)
                if filter_type is None:
                    address_list.append(address[0])
                    found_server = True
                    continue
                else:
                    pass
                json_obj = json.loads(data.decode("utf-8"))
                if "type" in json_obj:
                    if json_obj["type"] == filter_type:
                        address_list.append(address[0])
                        found_server = True
            except socket.timeout:
                return address_list if found_server else False

    def get_abs_encoder_angle(server_ip):
        data = {"method": "GET", "reqTarget": "/measured", "property": ""}
        json_str = json.dumps(data)
        s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
        try:
            while True:
                data, address = s.recvfrom(json_buf_size)
                if address[0] != server_ip:
                    continue
                json_obj = json.loads(data.decode("utf-8"))
                assert json_obj.get("status") == "OK"
                return json_obj.get("angle")
        except Exception:
            return None

    def set_control_param_imm(server_ip, dct):
        data = {
            "method": "SET",
            "reqTarget": "/control_param_imm",
            "property": "",
            "motor_max_speed_imm": dct["motor_max_speed_imm"],
            "motor_max_acceleration_imm": dct["motor_max_acceleration_imm"],
            "motor_max_current_imm": dct["motor_max_current_imm"],
        }
        json_str = json.dumps(data)
        s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    def set_pid_param_imm(server_ip, dict):
        data = {
            "method": "SET",
            "reqTarget": "/pid_param_imm",
            "property": "",
            "control_position_kp_imm": dict["control_position_kp_imm"],
            "control_velocity_kp_imm": dict["control_velocity_kp_imm"],
            "control_velocity_ki_imm": dict["control_velocity_ki_imm"],
            "control_current_kp_imm": dict["control_current_kp_imm"],
            "control_current_ki_imm": dict["control_current_ki_imm"],
        }

        json_str = json.dumps(data)
        s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))

    def get_pid_param_imm(server_ip):
        data = {"method": "GET", "reqTarget": "/pid_param_imm", "property": ""}

        json_str = json.dumps(data)

        s.sendto(str.encode(json_str), (server_ip, fsa_port_ctrl))
        try:
            data, address = s.recvfrom(json_buf_size)
            json_obj = json.loads(data.decode("utf-8"))
            return json_obj
        except Exception:
            return None

    def fast_set_disable(server_ip):
        tx_messages = struct.pack('>B', 0x02)
        s.sendto(tx_messages, (server_ip, fsa_port_fast))

    def fast_set_enable(server_ip):
        tx_messages = struct.pack('>B', 0x01)
        s.sendto(tx_messages, (server_ip, fsa_port_fast))

    def fast_set_position_mode(server_ip):
        tx_messages = struct.pack('>B', 0x04)
        s.sendto(tx_messages, (server_ip, fsa_port_fast))

    def fast_set_pd_mode(server_ip):
        tx_messages = struct.pack('>B', 0x09)
        s.sendto(tx_messages, (server_ip, fsa_port_fast))

    def fast_set_position_control_2(server_ip, position):
        msg = struct.pack('>Bfxxxxxxxx', 0x0A, position)
        s.sendto(msg, (server_ip, fsa_port_fast))

    def fast_set_pd_control_2(server_ip, position):
        msg = struct.pack('>Bf', 0x0E, position)
        s.sendto(msg, (server_ip, fsa_port_fast))

    get_pvc_msg = struct.pack('>B', 0x1a)
    get_pvct_msg = struct.pack('>B', 0x1d)

    def fast_get_pvc(server_ip):
        s.sendto(get_pvc_msg, (server_ip, fsa_port_fast))
        try:
            data, address = s.recvfrom(buf_size)
            assert address[0] == server_ip
            feedback, position, velocity, current = struct.unpack('>Bfff', data[0:1 + 4 + 4 + 4])
            return position, velocity, current
        except Exception:
            return 0, 0, 0

    def fast_get_pvct(server_ip):
        s.sendto(get_pvct_msg, (server_ip, fsa_port_fast))
        try:
            data, address = s.recvfrom(buf_size)
            assert address[0] == server_ip
            feedback, position, velocity, current, torque = struct.unpack_from('>Bffff', data)
            return position, velocity, current, torque
        except Exception:
            return 0, 0, 0, 0

    positions = np.zeros(len(states_q))
    velocities = np.zeros(len(states_qd))

    def fast_get_pvc_batch(server_ips, cb=None):
        # t0 = time.monotonic()
        for ip in server_ips:
            try:
                s.sendto(get_pvc_msg, (ip, fsa_port_fast))
            except socket.timeout:
                print('pvc send timeout', ip)
                continue
        # print('send', time.monotonic() - t0)
        if cb is not None:
            cb()
        if check_recv:
            inds = set()
        # print('cb', time.monotonic() - t0)
        for _ in range(len(server_ips)):
            try:
                data, address = s.recvfrom(buf_size)
                # print(address)
                feedback, position, velocity, current = struct.unpack('>Bfff', data[0:13])
                # return position, velocity, current
                # print(address)
                ip = address[0]
                idx = fsa_ip2inds[ip]
                positions[idx] = position
                velocities[idx] = velocity
                if states_q_cur is not None:
                    states_q_cur[idx] = current
                if check_recv:
                    inds.add(ip)
            except socket.timeout:
                print('pvc recv timeout', ip)
                break
            except Exception as e:
                print(_, e, type(e))
        if check_recv and len(inds) != len(server_ips):
            print('check recv failed', len(inds), len(server_ips), set(server_ips) - inds)
            return True
        # print('recv', time.monotonic() - t0)
        # print('positions', positions.tolist())
        # print('velocities', velocities.tolist())
        states_q[:] = positions * a2q_scale + a2q_bias
        states_qd[:] = velocities * a2q_scale

    fi_fsa.fsa_debug = False
    fsa_found_ips = broadcast_func_with_filter(filter_type="Actuator")
    print('fsa_found_ips', fsa_found_ips)
    assert fsa_found_ips is not False
    num_fsa_ips = len(fsa_found_ips)
    print('num_fsa_ips', num_fsa_ips)
    if num_fsa_ips != len(fsa_ips):
        print('num_fsa_ips not matched', num_fsa_ips, len(fsa_ips))
        if strict:
            raise RuntimeError('fftai check failed')
    num_dofs = len(states_q_ctrl)
    fsa_ip2inds = {k: i for i, k in enumerate(fsa_ips)}

    dof_map = list(range(num_dofs))[dof_map] if isinstance(dof_map, slice) else dof_map
    send_inds = dof_map if lazy_send else range(num_dofs)
    recv_inds = dof_map if lazy_recv else range(num_dofs)
    recv_ips = [fsa_ips[i] for i in recv_inds]
    print('dof_map', dof_map)
    print('send_inds', send_inds)
    print('recv_inds', recv_inds)

    fse_found_ips = broadcast_func_with_filter(filter_type="AbsEncoder")
    print('fse_found_ips', fse_found_ips)
    num_fse_ips = len(fse_found_ips) if fse_found_ips else 0
    print('num_fse_ips', num_fse_ips)
    if num_fse_ips != len(fse_ips):
        print('num_fse_ips not matched', num_fse_ips, len(fse_ips))
        if strict:
            raise RuntimeError('fftai check failed')
    if sensor_offset is not None:
        fse_zero_angles = [sensor_offset[k] for k in fse_ips]
    else:
        fse_zero_angles = [0] * len(fse_ips)
    fse_zero_angles = np.array(fse_zero_angles)
    check_recv = check_recv and strict

    rad2deg = (180. / np.pi)
    deg2rad = (np.pi / 180.)

    num_fses = len(fse_ips)
    num_fse_samples = 10
    fse_init_angles = []
    if num_fse_ips:
        for ip in fse_ips:
            a = get_abs_encoder_angle(ip)
            if a is None:
                if sensor_offset is not None:
                    ang = sensor_offset[ip]
                else:
                    ang = 0
                print('fallback', ip, ang)
            else:
                ang_s = 0
                for _ in range(num_fse_samples):
                    a = get_abs_encoder_angle(ip)
                    ang_s += a
                    time.sleep(0.05)
                ang = ang_s / num_fse_samples
                print(ip, ang)
            fse_init_angles.append(ang)
    if update_offset:
        offset = {ip: ang for ip, ang in zip(fse_ips, fse_init_angles)}
        with open(sensor_offset_path, 'w') as f:
            json.dump(offset, f, indent=4)
        exit(0)
    fse_init_angles = np.array(fse_init_angles)
    fsa_init_angles = np.zeros(len(fse_init_angles))
    for i, ip in enumerate(fse_ips):
        fsa_ip = fse2fsa[ip]
        p, v, c = fast_get_pvc(fsa_ip)
        fsa_init_angles[i] = p
    print('fsa_init_angles', np.round(fsa_init_angles).tolist())
    print('fse_init_angles', np.round(fse_init_angles).tolist())
    print('fse_zero_angles', np.round(fse_zero_angles).tolist())
    fsa_sign = np.array(fsa_sign)
    a2q_scale = fsa_sign * deg2rad
    q2a_scale = fsa_sign * rad2deg
    fsa_zeros = np.zeros(num_dofs)
    if enable_bias and num_fse_ips:
        fse_sign = np.array(robot_def.FSE_SIGN)
        fse_ratio = np.array(robot_def.FSE_RATIO)
        e2a_scale = fse_sign * fse_ratio
        print('e2a_scale', e2a_scale)
        e2a_scale = np.array(e2a_scale)
        a2q_bias = np.zeros(num_dofs)
        q2a_bias = np.zeros(num_dofs)
        fse_offset = (fse_init_angles - fse_zero_angles)
        a2q_bias[:num_fses] = -a2q_scale[:num_fses] * fsa_init_angles + e2a_scale * a2q_scale[:num_fses] * fse_offset
        q2a_bias[:num_fses] = fsa_init_angles - e2a_scale * fse_offset
        fsa_zeros[:num_fses] = q2a_bias[:num_fses]
        print('a2q_bias', np.round(a2q_bias[:num_fses], decimals=1).tolist())
        print('q2a_bias', np.round(q2a_bias[:num_fses], decimals=1).tolist())
    else:
        a2q_bias = 0
        q2a_bias = 0

    proc = None

    q_ctrl_min = np.array(q_ctrl_min, dtype=dtype)
    q_ctrl_max = np.array(q_ctrl_max, dtype=dtype)

    if set_control_params:
        import yaml
        if isinstance(motor_max_current, str):
            motor_max_current = yaml.safe_load(motor_max_current)
        if isinstance(motor_max_acceleration, str):
            motor_max_acceleration = yaml.safe_load(motor_max_acceleration)
        if isinstance(motor_max_speed, str):
            motor_max_speed = yaml.safe_load(motor_max_speed)
        num_ips = len(fsa_ips)
        mms = _default_mms if motor_max_speed is None else motor_max_speed
        mma = _default_mma if motor_max_acceleration is None else motor_max_acceleration
        mmc = _default_mmc if motor_max_current is None else motor_max_current
        mms = mms if isinstance(motor_max_speed, (list, tuple)) else [mms] * num_ips
        mma = mma if isinstance(motor_max_acceleration, (list, tuple)) else [mma] * num_ips
        mmc = mmc if isinstance(motor_max_current, (list, tuple)) else [mmc] * num_ips
        print('mms', mms)
        print('mma', mma)
        print('mmc', mmc)
        for i, ip in enumerate(fsa_ips):
            dct = {
                'motor_max_speed_imm': mms[i],
                'motor_max_acceleration_imm': mma[i],
                'motor_max_current_imm': mmc[i],
            }
            set_control_param_imm(ip, dct)

    fast_get_pvc_batch(fsa_ips)

    q_ctrl_fn = fast_set_pd_control_2 if use_pd_control else fast_set_position_control_2

    if enable_motors:
        print('disabling control')
        for ip in fsa_ips:
            fast_set_disable(ip)
        time.sleep(2)
        if init_send_zeros:
            q_send = fsa_zeros
        else:
            q_send = states_q * q2a_scale + q2a_bias
        print('q_send', q_send.tolist())
        print('enabling control')
        try:
            for i, ip in enumerate(fsa_ips):
                q_ctrl_fn(ip, q_send[i])
        except TimeoutError:
            print(i, ip)
            return
        if default_pd:
            kp = default_position_kp
            kd = default_velocity_kp
        if zero_pd:
            kp = 0.
            kd = 0.
        if kp is not None and kd is not None:
            kps = ([kp] * num_dofs) if isinstance(kp, float) else kp
            kds = ([kd] * num_dofs) if isinstance(kd, float) else kd
            kps = np.array(kps, dtype=dtype)
            kds = np.array(kds, dtype=dtype)
            print('fftai kps', kps)
            print('fftai kds', kds)
            for i, ip in enumerate(fsa_ips):
                dct = {  # 36
                    'control_position_kp_imm': kps[i],
                    'control_velocity_kp_imm': kds[i],
                    'control_velocity_ki_imm': 0.0,
                    'control_current_kp_imm': 0.0,  # not work for now
                    'control_current_ki_imm': 0.0,  # not work for now
                }
                set_pid_param_imm(ip, dct)
        for ip in fsa_ips:
            print(ip, get_pid_param_imm(ip))
        for ip in fsa_ips:
            fast_set_enable(ip)
        for ip in fsa_ips:
            fast_set_position_mode(ip)
        from unicon.utils.fftai import get_root_infos
        infos = get_root_infos(fi_fsa, fsa_ips)
        print('get_root_infos', infos)
        if fsa_time_out is not None:
            s.settimeout(fsa_time_out)
        time.sleep(wait)
        fast_get_pvc_batch(fsa_ips)

    def imu_read():
        w = ser.inWaiting()
        z = ser.read(w)[-82 * 2:]
        try:
            i = z.index(sig)
        except ValueError:
            return
        if i > 81:
            print('imu failed sig', i)
            return
        seg = z[i:i + 82]
        if len(seg) < 82:
            return
        d = struct.unpack(fmt, seg)
        v2 = d[-1]
        if v2 != 5023066:
            print('imu failed', v2)
            return
        if states_lin_acc is not None:
            acc = d[1:4]
            states_lin_acc[:] = acc
        ang_vel = d[4:7]
        rpy = [d[8], d[7], d[9]]
        states_rpy[:] = [x * deg2rad for x in rpy]
        states_ang_vel[:] = [x * deg2rad for x in ang_vel]
        states_quat[3] = d[10]
        states_quat[0:3] = d[11:14]

    if enable_imu:
        import os
        os.system(f'sudo chmod 666 {imu_device}')
        import serial
        ser = serial.Serial(imu_device, 921600)
        import struct
        # fmt = '<xxxxi'+'fff'+'fff'+'iii'+'fff'+'ffff'+'ixxxxxx'
        # fmt = '<xxxxi'+'fff'+'fff'+'fff'+'fff'+'ffff'+'ixxxxxx'
        fmt = '<xxxxi' + 'fff' + 'fff' + 'x' * 12 + 'fff' + 'ffff' + 'ixxxxxx'
        w = ser.inWaiting()
        if w:
            ser.read(w)
        z = ser.read(82 * 2)
        sig = [0, 0, 0, 0]
        sig = bytes(sig)
        i = z.index(sig)
        pvc_cb = imu_read
    else:
        pvc_cb = None

    def cb_recv():
        fast_get_pvc_batch(recv_ips, pvc_cb)

    _last_q_ctrl = None

    def cb_send():
        nonlocal _last_q_ctrl
        q_ctrl_clip = np.clip(states_q_ctrl, q_ctrl_min, q_ctrl_max) if clip_q_ctrl else states_q_ctrl
        q_ctrl_send = q_ctrl_clip * q2a_scale + q2a_bias
        # print('q_ctrl_send', q_ctrl_send.tolist())
        if unique_send and _last_q_ctrl is not None:
            qcd = np.abs(q_ctrl_clip - _last_q_ctrl) > 1e-3
            _send_inds = np.where(qcd)[0]
        else:
            _send_inds = send_inds
        try:
            for i in _send_inds:
                q_ctrl_fn(fsa_ips[i], q_ctrl_send[i])
        except TimeoutError:
            print('send timeout', i, fsa_ips[i])
        if not unique_send:
            return
        if _last_q_ctrl is None:
            _last_q_ctrl = q_ctrl_clip.copy()
        else:
            _last_q_ctrl[:] = q_ctrl_clip

    def cb_close():
        if enable_motors:
            for ip in fsa_ips:
                fast_set_disable(ip)
        if proc is not None:
            proc.kill()
        from unicon.utils import force_quit
        force_quit()

    return cb_recv, cb_send, cb_close
