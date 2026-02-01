# Address | Meaning | Abbrev | Length | R/W Permission

# 1000  Dexterous hand ID                         HAND_ID              1 byte      W/R
# 1002  Baud rate setting                         REDU_RATIO           1 byte      W/R
# 1004  Clear errors                              CLEAR_ERROR          1 byte      W/R
# 1005  Save data to Flash                        SAVE                 1 byte      W/R
# 1006  Restore factory settings                  RESET_PARA           1 byte      W/R
# 1009  Force sensor calibration                  GESTURE_FORCE_CLB    1 byte      W/R

# 1032  Default startup speed for each DOF        DEFAULT_SPEED_SET    6 short     W/R
# 1044  Default startup force threshold per DOF   DEFAULT_FORCE_SET    6 short     W/R

# 1474  Driver position set values per DOF        POS_SET              6 short     W/R
# 1486  Angle set values per DOF                  ANGLE_SET            6 short     W/R
# 1498  Force threshold set values per DOF        FORCE_SET            6 short     W/R
# 1522  Speed set values per DOF                  SPEED_SET            6 short     W/R

# 1534  Actual driver position per DOF            POS_ACT              6 short     R
# 1546  Actual angle per DOF                      ANGLE_ACT            6 short     R
# 1582  Actual finger force                       FORCE_ACT            6 short     R
# 1594  Driver current per DOF                    CURRENT              6 short     R
# 1606  Driver fault info per DOF                 ERROR                6 byte      R
# 1612  Status info per DOF                       STATUS               6 byte      R
# 1618  Driver temperature per DOF                TEMP                 6 byte      R

# 1700  IP address field 1 (default 192)          IP_PART1             1 byte      W/R
#       Range 0–255, takes effect after reboot
# 1701  IP address field 2 (default 168)          IP_PART2             1 byte      W/R
#       Range 0–255, takes effect after reboot
# 1702  IP address field 3 (default 11)           IP_PART3             1 byte      W/R
#       Range 0–255, takes effect after reboot
# 1703  IP address field 4 (default 210)          IP_PART4             1 byte      W/R
#       Range 0–255, takes effect after reboot

# 3000  Pinky tactile sensor                      FINGERONE_TOUCH      370 byte    R
# 3370  Ring finger tactile sensor                FINGERTWO_TOUCH      370 byte    R
# 3740  Middle finger tactile sensor              FINGERTHE_TOUCH      370 byte    R
# 4110  Index finger tactile sensor               FINGERFOR_TOUCH      370 byte    R
# 4480  Thumb tactile sensor                      FINGERFIV_TOUCH      420 byte    R
# 4900  Palm tactile sensor                       FINGERPALM_TOUCH     224 byte    R

# ----- Basic Settings -----
REG_HAND_ID = 1000
REG_REDU_RATIO = 1002
REG_CLEAR_ERROR = 1004
REG_SAVE = 1005
REG_RESET_PARA = 1006
REG_GESTURE_FORCE_CLB = 1009

# ----- Default DOF Settings -----
REG_DEFAULT_SPEED_SET = 1032  # 6 short
REG_DEFAULT_FORCE_SET = 1044  # 6 short

# ----- DOF Set Values -----
REG_POS_SET = 1474  # 6 short
REG_ANGLE_SET = 1486  # 6 short
REG_FORCE_SET = 1498  # 6 short
REG_SPEED_SET = 1522  # 6 short

# ----- DOF Actual Values -----
REG_POS_ACT = 1534  # 6 short
REG_ANGLE_ACT = 1546  # 6 short
REG_FORCE_ACT = 1582  # 6 short
REG_CURRENT = 1594  # 6 short
REG_ERROR = 1606  # 6 byte
REG_STATUS = 1612  # 6 byte
REG_TEMP = 1618  # 6 byte

# ----- Network Settings -----
REG_IP_PART1 = 1700
REG_IP_PART2 = 1701
REG_IP_PART3 = 1702
REG_IP_PART4 = 1703

# ----- Tactile Sensors -----
REG_FINGERONE_TOUCH = 3000  # 370 bytes
REG_FINGERTWO_TOUCH = 3370  # 370 bytes
REG_FINGERTHE_TOUCH = 3740  # 370 bytes
REG_FINGERFOR_TOUCH = 4110  # 370 bytes
REG_FINGERFIV_TOUCH = 4480  # 420 bytes
REG_FINGERPALM_TOUCH = 4900  # 224 bytes

ERR_MOTOR_STALL = 1 << 0  # Bit0  stall / blocked motor
ERR_OVERHEAT = 1 << 1  # Bit1  over‑temperature fault
ERR_OVERCURRENT = 1 << 2  # Bit2  over‑current fault
ERR_MOTOR_ERROR = 1 << 3  # Bit3  motor abnormal
ERR_COMM_ERROR = 1 << 4  # Bit4  communication fault
ERR_BITS = {
    'ERR_MOTOR_STALL': ERR_MOTOR_STALL,
    'ERR_OVERHEAT': ERR_OVERHEAT,
    'ERR_OVERCURRENT': ERR_OVERCURRENT,
    'ERR_MOTOR_ERROR': ERR_MOTOR_ERROR,
    'ERR_COMM_ERROR': ERR_COMM_ERROR,
}

STATUS_RELEASING = 0  # Releasing
STATUS_GRASPING = 1  # Grasping
STATUS_POSITION_STOP = 2  # Stopped: position reached
STATUS_FORCE_STOP = 3  # Stopped: force reached
STATUS_CURRENT_PROTECT_STOP = 5  # Stopped: over‑current protection
STATUS_STALL_STOP = 6  # Stopped: stall
STATUS_CYLINDER_FAULT_STOP = 7  # Stopped: actuator fault
STATUS_ERROR = 255  # Error


def cb_system_inspire(
    states_hand_q=None,
    states_hand_q_ctrl=None,
    states_hand_tactile=None,
    async_recv=False,
    tcp_left='192.168.11.211:6000',
    tcp_right='192.168.11.210:6000',
    port_left='/dev/ttyUSB1',
    port_right='/dev/ttyUSB0',
    baudrate=115200,
    timeout=1,
    client_type='tcp',
    speed_limit=1000,
    force_limit=3000,
    to_radian=False,
    check_errs=False,
    ignore_cli_err=True,
):
    from pymodbus.client import ModbusTcpClient, ModbusSerialClient
    from unicon.utils import get_ctx, list2slice, expect
    ctx = get_ctx()
    hand_def = ctx['robot_def']['hand_def']
    DOF_NAMES = hand_def['DOF_NAMES']
    Q_RESET = hand_def['Q_RESET']
    # num_dofs = len(states_hand_q)

    use_tcp = client_type == 'tcp'
    use_serial = client_type == 'serial'

    sides = ['left', 'right']
    addrs = {'left': tcp_left, 'right': tcp_right}
    ser_ports = {'left': port_left, 'right': port_right}
    cli_specs = []
    for side in sides:
        if use_tcp:
            addr = addrs[side]
            ip, port = addr.split(':')
            client = ModbusTcpClient(ip, port)
        elif use_serial:
            port = ser_ports[side]
            client = ModbusSerialClient(port=port, framer='rtu', baudrate=baudrate, timeout=timeout)
        if not client.connect():
            print('ModbusTcpClient connect error', side, addr)
            expect(ignore_cli_err, 'modbus client failed')
        dof_names_side = [n for n in DOF_NAMES if side in n]
        num_dofs_side = len(dof_names_side)
        dof_inds_side = [DOF_NAMES.index(n) for n in dof_names_side]
        dof_inds_sl_side = list2slice(dof_inds_side)
        print('num_dofs_side', num_dofs_side)
        print('dof_inds_sl_side', dof_inds_sl_side)
        cli_specs.append((client, dof_inds_sl_side))

    expect(num_dofs_side == 6)
    num_dofs_side_h = num_dofs_side // 2

    def read_ushorts(cli, adr, n):
        res = cli.read_holding_registers(adr, n)
        if res.isError():
            return None
        return res.registers

    def read_shorts(cli, adr, n):
        res = cli.read_holding_registers(adr, n)
        if res.isError():
            return None
        regs = res.registers
        return [(r - 0x10000 if r & 0x8000 else r) for r in regs]

    def read_bytes(cli, adr, n):
        res = cli.read_holding_registers(adr, n)
        if res.isError():
            return None
        regs = res.registers
        return [b for r in regs for b in ((r >> 8) & 0xFF, r & 0xFF)]

    def write_shorts(cli, adr, vals):
        vals = [int(v) & 0xFFFF for v in vals]
        cli.write_registers(adr, vals, no_response_expected=True)

    def write_bytes(cli, adr, vals):
        vals = [int(v) & 0xFFFF for v in vals]
        cli.write_registers(adr, vals, no_response_expected=True)

    speed_limit = ([speed_limit] * num_dofs_side) if isinstance(speed_limit, (int, float)) else speed_limit
    force_limit = ([force_limit] * num_dofs_side) if isinstance(force_limit, (int, float)) else force_limit
    for cli, _ in cli_specs:
        errs = read_bytes(cli, REG_ERROR, num_dofs_side_h)
        print('errs', errs)
        status = read_bytes(cli, REG_STATUS, num_dofs_side_h)
        print('status', status)
        temps = read_bytes(cli, REG_TEMP, num_dofs_side_h)
        print('temps', temps)
        write_bytes(cli, REG_CLEAR_ERROR, [1])
        if speed_limit is not None:
            write_shorts(cli, REG_SPEED_SET, speed_limit)
        if force_limit is not None:
            write_shorts(cli, REG_FORCE_SET, force_limit)

    def cb_recv():
        for cli, dof_inds_sl in cli_specs:
            angs = read_shorts(cli, REG_ANGLE_ACT, num_dofs_side)
            if angs is None:
                continue
            states_hand_q[dof_inds_sl] = angs
            if not check_errs:
                continue
            errs = read_bytes(cli, REG_ERROR, num_dofs_side_h)
            if not any(errs):
                continue
            details = [[i, e, [k for k, v in ERR_BITS.items() if (e & v)]] for i, e in enumerate(errs) if e]
            print('inspire err', errs, details)
            return True

    def cb_send():
        for cli, dof_inds_sl in cli_specs:
            q_ctrl = states_hand_q_ctrl[dof_inds_sl]
            write_shorts(cli, REG_ANGLE_SET, q_ctrl)

    def cb_close():
        for cli, dof_inds_sl in cli_specs:
            write_shorts(cli, REG_ANGLE_ACT, Q_RESET[dof_inds_sl])
            cli.close()

    return cb_recv, cb_send, cb_close
