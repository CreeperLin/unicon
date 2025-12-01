
_default_mmc = [100] * 15
_default_mma = [60000] * 15
_default_mms = [3000] * 15


def cb_n1_recv_send_close(
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
    consys=None,
    kp=None,
    kd=None,
    # radian
    enable_motors=True,
    servo_on_all=False,
    set_control_params=False,
    motor_max_current=None,
    motor_max_acceleration=None,
    motor_max_speed=None,
    config=None,
    use_r=False,
    # control_mode=4,
    control_mode=6,
    reboot=False,
    use_fi_fsa=True,
    init_servo_on=True,
    use_deg2rad=None,
):
    import sys
    import time
    import numpy as np
    from unicon.utils import watchdog

    rad2deg = 180. / np.pi
    deg2rad = np.pi / 180.

    def get_consys(servo_on=False, config=None):
        with watchdog(20):
            t0 = time.time()
            # print(sys.argv)
            sys.argv = sys.argv[:1]
            if config is not None:
                sys.argv.extend(['--config', config])
            mod_path = 'fourier_grx.sdk.developer'
            consys_cls = 'ControlSystem'
            sdk = __import__(mod_path, fromlist=[''])
            mod_path = 'fourier_grx.sdk.user'
            user_sdk = __import__(mod_path, fromlist=[''])
            ControlSystem = getattr(sdk, consys_cls)
            consys = ControlSystem()
            consys.developer_mode(servo_on=servo_on)
            print(consys.get_info())
            print('consys init', time.time() - t0)
        sdk.TaskCommand = user_sdk.TaskCommand
        return consys, sdk


    def servo_off(consys):
        consys.robot_control_set_task_command(task_command=_sdk.TaskCommand.TASK_SERVO_OFF)
        time.sleep(1)


    def servo_on(consys, control_mode=None, q_send=None):
        ips = r.actuator_group.ips
        # print(len(ips), num_dofs)
        num_dofs = len(ips)
        if control_mode == 4:
            # JointControlMode.POSITION
            params = {
                'control_mode': [4] * num_dofs,
                'kp': [0.] * num_dofs,
                'kd': [0.] * num_dofs,
                'position': q_send,
            }
            consys.robot_control_loop_set_control(params)
        if control_mode == 6:
            # JointControlMode.PD
            params = {
                'control_mode': [6] * num_dofs,
                'pd_control_kp': [0] * num_dofs,
                'pd_control_kd': [0] * num_dofs,
                'position': q_send,
            }
            consys.robot_control_loop_set_control(params)
        servo_off(consys)
        consys.robot_control_set_task_command(task_command=_sdk.TaskCommand.TASK_SERVO_ON)
        time.sleep(1)


    from unicon.utils import get_ctx
    robot_def = get_ctx()['robot_def']
    fi_fsa = None
    if use_fi_fsa:
        import socket
        fsa_port_ctrl = 2333
        fsa_port_comm = 2334
        fsa_port_fast = 2335
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        fi_fsa = type('fi_fsa', (), {})()
        fi_fsa.s = s
        fi_fsa.fsa_port_ctrl = fsa_port_ctrl
        fi_fsa.fsa_port_comm = fsa_port_comm
        fi_fsa.fsa_port_fast = fsa_port_fast
        fi_fsa.fsa_network = "192.168.137.255"

    if fi_fsa is None:
        print('fi_fsa not found')
        set_control_params = False
        reboot = False

    if reboot:
        from unicon.utils.fftai import reboot_fsa, fsa_broadcast
        fsa_ips_def = robot_def['FSA_IPS']
        # fsa_ips = fsa_broadcast(fi_fsa, 'Actuator')
        num_ips = len(fsa_ips_def)
        print('rebooting fsa', num_ips)
        for i, ip in enumerate(fsa_ips_def):
            reboot_fsa(fi_fsa, ip)
        __import__('time').sleep(15)
        fsa_ips = fsa_broadcast(fi_fsa, 'Actuator', max_ips=num_ips)
        assert num_ips == len(fsa_ips), f'{num_ips} != {len(fsa_ips)} {set(fsa_ips_def) - set(fsa_ips)}'

    num_dofs = len(states_q_ctrl)
    if config is None:
        import os
        default_config_path = os.path.join(
            os.environ['HOME'], 'fourier-grx/config/grmini1/config_GRMini1_T2_debug.yaml'
        )
        config = default_config_path
    if consys is None:
        consys, _sdk = get_consys(servo_on=init_servo_on, config=config)
    r = consys.robot_interface
    import fourier_grx
    grx_version = fourier_grx.__version__
    print('grx_version', grx_version)

    if use_deg2rad is None:
        use_deg2rad = int(grx_version.split('.')[0]) < 4
    deg2rad_st = None
    rad2deg_st = None
    if use_deg2rad:
        deg2rad_st = deg2rad
        rad2deg_st = rad2deg
    print('use_deg2rad', use_deg2rad)

    fsa_ips = r.actuator_group.ips

    if enable_motors:
        state_dict = consys.robot_control_loop_get_state()
        from unicon.utils.fftai import \
            get_root_infos, set_control_param_imm, get_control_param_imm, servo_on_fsa, servo_off_fsa
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
            mms = mms if isinstance(mms, (list, tuple)) else [mms] * num_ips
            mma = mma if isinstance(mma, (list, tuple)) else [mma] * num_ips
            mmc = mmc if isinstance(mmc, (list, tuple)) else [mmc] * num_ips
            print('mms', mms)
            print('mma', mma)
            print('mmc', mmc)
            for i, ip in enumerate(fsa_ips):
                if i >= len(mmc) or i >= len(mma) or i >= len(mms):
                    break
                dct = {
                    'motor_max_speed': mms[i],
                    'motor_max_acceleration': mma[i],
                    'motor_max_current': mmc[i],
                }
                print('set_control_param_imm', ip, mms[i], mma[i], mmc[i])
                set_control_param_imm(fi_fsa, ip, dct)
                # set_control_param(fi_fsa, ip, dct)
            for i, ip in enumerate(fsa_ips):
                params = get_control_param_imm(fi_fsa, ip)
                print('get_control_param_imm', ip, params)
        if not init_servo_on:
            servo_off(consys)
            init_send_zeros = False
            if init_send_zeros:
                q_send = [0] * num_dofs
            else:
                q_send = state_dict['joint_position']
            print('q_send', q_send.tolist())
            servo_on(consys, control_mode=control_mode, q_send=q_send)
        infos = get_root_infos(fi_fsa, fsa_ips)
        print('root_infos', infos)
        from unicon.utils.fftai import get_comm_infos
        infos = get_comm_infos(fi_fsa, fsa_ips)
        print('comm_infos', infos)
        if servo_on_all:
            servo_on_fsa(fi_fsa, fsa_ips)

    # from sdk
    print('_sdk.JointControlMode.POSITION', _sdk.JointControlMode.POSITION)
    print('_sdk.JointControlMode.PD', _sdk.JointControlMode.PD)
    if control_mode == 4:
        # assert kp is None
        kp = r.joint_position_control_kp
        kd = r.joint_velocity_control_kp
    elif control_mode == 6:
        assert kp is not None
    kps = ([kp] * num_dofs) if isinstance(kp, float) else kp
    kds = ([kd] * num_dofs) if isinstance(kd, float) else kd
    modes = ([control_mode] * num_dofs) if isinstance(control_mode, int) else control_mode
    modes = np.array(modes, dtype=np.int32)
    kps = np.array(kps, dtype=np.float64)
    kds = np.array(kds, dtype=np.float64)
    params = {
        'control_mode': modes.tolist(),
    }
    if control_mode == 4:
        params.update({
            'kp': kps.tolist(),
            'kd': kds.tolist(),
        })
    elif control_mode == 6:
        params.update({
            'pd_control_kp': kps.tolist(),
            'pd_control_kd': kds.tolist(),
        })
    print('modes', len(modes))
    print('kps', len(kps))
    print('kds', len(kds))
    print('params', params)
    print('q_ctrl', len(states_q_ctrl))

    _states_map = [
        (states_rpy, deg2rad_st, "imu_euler_angle"),
        (states_ang_vel, deg2rad_st, "imu_angular_velocity"),
        (states_quat, None, "imu_quat"),
        (states_q, deg2rad_st, "joint_position"),
        (states_qd, deg2rad_st, "joint_velocity"),
        # (states_pos, None, "base_estimate_xyz"),
        # (states_lin_vel, None, "base_estimate_xyz_vel"),
        (states_q_tau, None, "joint_effort"),
        (states_lin_acc, None, "imu_acceleration"),
    ]
    if use_r:
        imu = r.sensor_usb_imus[0]
        _states_map = [
            # (states_rpy, deg2rad, r, "sensor_usb_imu_group_measured_angle"),
            # (states_ang_vel, deg2rad, r, "sensor_usb_imu_group_measured_angular_velocity"),
            # (states_quat, None, r, "sensor_usb_imu_group_measured_quat"),
            # (states_rpy, deg2rad, imu, "measured_angle"),
            # (states_ang_vel, deg2rad, imu, "measured_angular_velocity"),
            # (states_quat, None, imu, "measured_quat"),
            (states_rpy, deg2rad_st, r, "share_sensor_usb_imu_group_measured_angle"),
            (states_ang_vel, deg2rad_st, r, "share_sensor_usb_imu_group_measured_angular_velocity"),
            (states_quat, None, r, "share_sensor_usb_imu_group_measured_quat"),
            (states_q, deg2rad_st, r, "joint_urdf_group_measured_position"),
            (states_qd, deg2rad_st, r, "joint_urdf_group_measured_velocity"),
            (states_pos, None, r, "base_xyz"),
            (states_lin_vel, None, r, "base_xyz_vel"),
            (states_q_tau, None, r, "joint_urdf_group_measured_kinetic"),
            (states_q_cur, None, r, "actuator_group_measured_current"),
            (states_lin_acc, None, imu, "measured_acceleration"),
        ]
    _states_map = list(filter(lambda x: x[0] is not None, _states_map))
    state_dict = consys.robot_control_loop_get_state()
    print(state_dict.keys())

    def cb_recv_c():
        # t0 = time.time()
        state_dict = consys.robot_control_loop_get_state()
        for s, c, k in _states_map:
            v = state_dict[k]
            s[:] = (v if c is None else v * c)
        if states_q_cur is not None:
            states_q_cur[:] = r.actuator_group_measured_current

    def cb_recv_r():
        for s, c, o, k in _states_map:
            v = getattr(o, k)
            s[:] = v
            if c is not None:
                s *= c
                # s[:] = s * c

    cb_recv = cb_recv_r if use_r else cb_recv_c

    def cb_send_sync():
        # t0 = time.time()
        q_ctrl = states_q_ctrl
        if rad2deg_st is not None:
            q_ctrl = q_ctrl * rad2deg_st
        params['position'] = q_ctrl.tolist()
        consys.robot_control_loop_set_control(params)
        # print('send', time.time() - t0)

    cb_send = cb_send_sync

    def cb_close():
        if enable_motors:
            servo_off(consys)
            if servo_on_all:
                servo_off_fsa(fi_fsa, fsa_ips)
        from unicon.utils import force_quit
        force_quit()

    return cb_recv, cb_send, cb_close
