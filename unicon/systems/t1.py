_default_cfg = {
    "common": {
        "dt": 0.005,
        "stiffness": [
            20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 200, 200, 200, 200, 200, 50, 50, 200, 200, 200, 200, 50, 50
        ],
        "damping": [0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5, 5, 5, 5, 5, 3, 3, 5, 5, 5, 5, 3, 3],
        "default_qpos": [
            0, 0, 0.2, -1.3, 0, -1.6, 0.2, 1.3, 0, 1.6, 0, -0.2, 0, 0, 0.4, -0.25, 0, -0.2, 0, 0, 0.4, -0.25, 0
        ],
        "torque_limit": [7, 7, 10, 10, 10, 10, 10, 10, 10, 10, 30, 60, 25, 30, 60, 24, 15, 60, 25, 30, 60, 24, 15]
    },
    "mech": {
        "parallel_mech_indexes": [15, 16, 21, 22]
    },
}


def _proc_cmd_send(q_buf, q_ctrl_buf, enable_conv, send_dt, kp, kd, torque_limit, parallel_mech_indexes, running):
    import time
    import numpy as np
    print('_proc_cmd_send init')
    from booster_robotics_sdk_python import (
        B1LowCmdPublisher,
        ChannelFactory,
        LowCmd,
        LowCmdType,
        MotorCmd,
        B1JointCnt,
    )
    low_cmd = LowCmd()
    low_cmd.cmd_type = LowCmdType.SERIAL
    motorCmds = [MotorCmd() for _ in range(B1JointCnt)]
    low_cmd.motor_cmd = motorCmds
    for i in range(B1JointCnt):
        low_cmd.motor_cmd[i].q = 0.0
        low_cmd.motor_cmd[i].dq = 0.0
        low_cmd.motor_cmd[i].tau = 0.0
        low_cmd.motor_cmd[i].kp = kp[i]
        low_cmd.motor_cmd[i].kd = kd[i]
        # weight is not effective in custom mode
        low_cmd.motor_cmd[i].weight = 0.0
        low_cmd.motor_cmd[i].q = 0
        low_cmd.motor_cmd[i].tau = 0
    if enable_conv:
        for i in parallel_mech_indexes:
            low_cmd.motor_cmd[i].kp = 0.0
    net = '127.0.0.1'
    ChannelFactory.Instance().Init(0, net)
    low_cmd_publisher = B1LowCmdPublisher()
    low_cmd_publisher.InitChannel()
    print('_proc_cmd_send wait', send_dt)
    while not running.value:
        time.sleep(send_dt)
    print('_proc_cmd_send start')
    while running.value:
        for i, mc in enumerate(low_cmd.motor_cmd):
            mc.q = q_ctrl_buf[i]
            mc.tau = 0
            mc.dq = 0
        if enable_conv:
            # Use series-parallel conversion for torque to avoid non-linearity
            for i in parallel_mech_indexes:
                low_cmd.motor_cmd[i].tau = np.clip(
                    (q_ctrl_buf[i] - q_buf[i]) * kp[i],
                    -torque_limit[i],
                    torque_limit[i],
                )
        low_cmd_publisher.Write(low_cmd)
        time.sleep(send_dt)
    low_cmd_publisher.CloseChannel()
    print('_proc_cmd_send done')


def cb_t1_recv_send_close(
    states_q_ctrl,
    states_rpy,
    states_ang_vel,
    states_quat,
    states_q,
    states_qd,
    states_q_tau=None,
    states_q_temp=None,
    kp=None,
    kd=None,
    q_ctrl_min=None,
    q_ctrl_max=None,
    clip_q_ctrl=True,
    input_keys=None,
    net='127.0.0.1',
    cfg_file='config_t1.yaml',
    # enable_conv=False,
    enable_conv=True,
    restart=True,
    # restart=False,
    # send_mode=None,
    send_mode='th',
    # send_mode='mp',
    compute_quat=False,
    send_dt=0.001,
):
    import os
    import time
    import yaml
    import numpy as np
    from booster_robotics_sdk_python import (
        ChannelFactory,
        B1LocoClient,
        B1LowCmdPublisher,
        B1LowStateSubscriber,
        LowCmd,
        LowState,
        LowCmdType,
        MotorCmd,
        B1JointCnt,
        RobotMode,
    )

    print('B1JointCnt', B1JointCnt)
    num_dofs = len(states_q)
    assert num_dofs == B1JointCnt

    th_send = False
    recv_send = False
    mp_send = False
    if send_mode == 'th':
        th_send = True
    elif send_mode == 'mp':
        mp_send = True
    elif send_mode == 'recv':
        recv_send = True
    print('send_mode', send_mode)

    def init_Cmd_T1(low_cmd: LowCmd):
        low_cmd.cmd_type = LowCmdType.SERIAL
        motorCmds = [MotorCmd() for _ in range(B1JointCnt)]
        low_cmd.motor_cmd = motorCmds

        for i in range(B1JointCnt):
            low_cmd.motor_cmd[i].q = 0.0
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
            low_cmd.motor_cmd[i].kp = 0.0
            low_cmd.motor_cmd[i].kd = 0.0
            # weight is not effective in custom mode
            low_cmd.motor_cmd[i].weight = 0.0

    def create_first_frame_rl_cmd(low_cmd: LowCmd, cfg):
        init_Cmd_T1(low_cmd)
        for i in range(B1JointCnt):
            low_cmd.motor_cmd[i].kp = kp[i]
            low_cmd.motor_cmd[i].kd = kd[i]
            low_cmd.motor_cmd[i].q = 0
            low_cmd.motor_cmd[i].tau = 0
        return low_cmd

    def _send_cmd(cmd: LowCmd):
        low_cmd_publisher.Write(cmd)

    def _low_state_handler(_low_state_msg: LowState):
        low_state_msg = _low_state_msg
        base_rpy[:] = low_state_msg.imu_state.rpy
        base_ang_vel[:] = low_state_msg.imu_state.gyro
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            dof_pos_latest[i] = motor.q
            dof_vel[i] = motor.dq
        if states_q_temp is not None:
            for i, motor in enumerate(low_state_msg.motor_state_serial):
                states_q_temp[i] = motor.temperature
        if states_q_tau is not None:
            for i, motor in enumerate(low_state_msg.motor_state_serial):
                states_q_tau[i] = motor.tau_est
        if mp_send:
            shared_q[:] = dof_pos_latest
        if recv_send and running:
            _write_cmd()

    def _write_cmd():
        for i in range(B1JointCnt):
            low_cmd.motor_cmd[i].q = filtered_dof_target[i]
            low_cmd.motor_cmd[i].tau = 0
            low_cmd.motor_cmd[i].dq = 0

        if enable_conv:
            # Use series-parallel conversion for torque to avoid non-linearity
            for i in parallel_mech_indexes:
                low_cmd.motor_cmd[i].tau = np.clip(
                    (filtered_dof_target[i] - dof_pos_latest[i]) * kp[i],
                    -torque_limit[i],
                    torque_limit[i],
                )

            # start_time = time.perf_counter()
        low_cmd_publisher.Write(low_cmd)

    def _th_cmd_send():
        while not running:
            time.sleep(send_dt)
        while running:
            _write_cmd()
            time.sleep(send_dt)

    running = False
    if th_send:
        import threading
        thd_send = threading.Thread(target=_th_cmd_send, daemon=True)
        thd_send.start()

    if os.path.exists(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f.read())
    else:
        cfg = _default_cfg

    ChannelFactory.Instance().Init(0, net)
    base_ang_vel = np.zeros(3, dtype=np.float32)
    base_rpy = np.zeros(3, dtype=np.float32)
    # dof_pos = np.zeros(B1JointCnt, dtype=np.float32)
    dof_vel = np.zeros(B1JointCnt, dtype=np.float32)
    # dof_target = np.zeros(B1JointCnt, dtype=np.float32)
    filtered_dof_target = np.zeros(B1JointCnt, dtype=np.float32)
    dof_pos_latest = np.zeros(B1JointCnt, dtype=np.float32)

    if kp is None:
        kp = cfg["common"]["stiffness"]
    if kd is None:
        kd = cfg["common"]["damping"]
    print('kp', kp)
    print('kd', kd)
    torque_limit = cfg['common']['torque_limit']
    parallel_mech_indexes = cfg["mech"]["parallel_mech_indexes"]

    if mp_send:
        import ctypes
        import multiprocessing
        ctx = multiprocessing.get_context('spawn')
        _running = ctx.Value('i', 0)
        q_buf = ctx.Array(
            ctypes.c_float,
            num_dofs,
            lock=False,
        )
        q_ctrl_buf = ctx.Array(
            ctypes.c_float,
            num_dofs,
            lock=False,
        )
        shared_q = np.frombuffer(q_buf, dtype=ctypes.c_float)
        shared_q_ctrl = np.frombuffer(q_ctrl_buf, dtype=ctypes.c_float)
        for i in range(num_dofs):
            shared_q[i] = 0
            shared_q_ctrl[i] = 0
        proc = ctx.Process(
            target=_proc_cmd_send,
            daemon=True,
            args=(
                q_buf,
                q_ctrl_buf,
                enable_conv,
                send_dt,
                kp,
                kd,
                torque_limit,
                parallel_mech_indexes,
                _running,
            ),
        )
        proc.start()
        time.sleep(1)
        print(proc.pid, proc.is_alive(), proc.exitcode)

    if restart:
        os.system('timeout 9 booster-cli launch -c restart')
    try:
        low_cmd = LowCmd()
        low_state_subscriber = B1LowStateSubscriber(_low_state_handler)
        low_state_subscriber.InitChannel()
        low_cmd_publisher = B1LowCmdPublisher()
        low_cmd_publisher.InitChannel()
        client = B1LocoClient()
        client.Init()
    except Exception as e:
        print(f"Failed to initialize communication: {e}")
        raise

    def servo_off():
        init_Cmd_T1(low_cmd)
        for i in range(B1JointCnt):
            low_cmd.motor_cmd[i].kp = 0
            low_cmd.motor_cmd[i].kd = 0
        _send_cmd(low_cmd)

    servo_off()
    time.sleep(5)
    client.ChangeMode(RobotMode.kCustom)
    create_first_frame_rl_cmd(low_cmd, cfg)

    if enable_conv:
        for i in parallel_mech_indexes:
            low_cmd.motor_cmd[i].kp = 0.0

    from unicon.utils import rpy2quat_np

    def cb_recv():
        states_rpy[:] = base_rpy
        if compute_quat:
            states_quat[:] = rpy2quat_np(base_rpy)
        states_ang_vel[:] = base_ang_vel
        states_q[:] = dof_pos_latest
        states_qd[:] = dof_vel

    def cb_send():
        nonlocal running
        filtered_dof_target[:] = states_q_ctrl
        if send_mode is None:
            _write_cmd()
        if not running:
            running = True
            if mp_send:
                _running.value = 1
            return
        if mp_send:
            shared_q_ctrl[:] = states_q_ctrl
            if not proc.is_alive():
                return True

    def cb_close():
        nonlocal running
        running = False
        if mp_send:
            _running.value = 0
            time.sleep(1)
            print(proc.is_alive(), proc.exitcode)
            proc.kill()
        if th_send:
            thd_send.join()
        low_state_subscriber.CloseChannel()
        servo_off()

        low_cmd_publisher.CloseChannel()
        from unicon.utils import force_quit
        force_quit()

    return cb_recv, cb_send, cb_close
