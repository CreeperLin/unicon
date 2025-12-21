import struct
import time

_state_rem_keys = [
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

_key_mapping = {
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
    'BTN_SELECT': 'SELECT',
    'BTN_START': 'START',
    'ABS_HAT0Y-': 'up',
    'ABS_HAT0X+': 'right',
    'ABS_HAT0Y+': 'down',
    'ABS_HAT0X-': 'left',
    'ABS_BRAKE': 'L2',
    'ABS_GAS': 'R2',
}


def get_key_mapping():
    return _key_mapping.copy()


def unpack_wireless_remote(wireless_remote):
    rem = bytearray(wireless_remote)
    unpacked_data = struct.unpack('<2B H 5f 16B', rem)
    lx = unpacked_data[3]
    ly = unpacked_data[7]
    rx = unpacked_data[4]
    ry = unpacked_data[5]
    value = unpacked_data[2]
    states = dict(lx=lx, rx=rx, ry=-ry, ly=-ly)
    for i, k in enumerate(_state_rem_keys):
        states[k] = (value & (1 << i)) >> i
    return states


def init_channel(*args, participant=None, domain=None, **kwds):
    from unitree_sdk2py.core.channel import ChannelFactory, ChannelFactoryInitialize
    factory = ChannelFactory()
    if participant is not None:
        factory._ChannelFactory__participant = participant
    if domain is not None:
        factory._ChannelFactory__domain = domain
    if factory._ChannelFactory__participant is not None:
        print('init_channel skipped')
        return
    ChannelFactoryInitialize(*args, **kwds)


def disable_motion():
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    init_channel()
    print('sport init')
    sc = SportClient()
    sc.SetTimeout(1.0)
    sc.Init()
    sc.StandDown()
    print('msc init')
    msc = MotionSwitcherClient()
    msc.SetTimeout(1.0)
    msc.Init()
    status, result = msc.CheckMode()
    for _ in range(1):
        print('msc', status, result)
        msc.ReleaseMode()
        status, result = msc.CheckMode()
        if result is None or not result['name']:
            break
        time.sleep(1)


def disable_lidar():
    from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    from unitree_sdk2py.core.channel import ChannelPublisher
    init_channel()
    lidar_pub = ChannelPublisher("rt/utlidar/switch", String_)
    lidar_pub.Init()
    lidar_cmd = std_msgs_msg_dds__String_()
    lidar_cmd.data = "OFF"
    lidar_pub.Write(lidar_cmd)


def pkill_services():
    from unicon.utils import pkill
    pkill(
        'master_service',
        'social',
        'humanoid',
        'motion_switcher',
        'WebRTC',
        'nginx',
        'bashrunner',
        'net_switcher',
        'multicast_responder',
        'signal_server',
        'ota_box',
        'beanstalkd',
        'robot_state',
    )


# Mainboard communication firmware errors
hg_mainboard_errors = {
    # 0x00001: "Upper-level control command timeout",
    0x00002: "Lower-level feedback data timeout",
    0x00004: "IMU feedback data timeout",
    0x00008: "Motor feedback data timeout",
    0x00080: "Soft start error",
    0x00100: "Motor state error",
    0x00200: "Motor overcurrent protection, triggered low limit protection",
    0x00400: "Motor undervoltage protection, triggered high limit protection",
    0x00800: "Motor overcurrent protection, triggered high limit protection",
    0x01000: "Lower soft emergency stop switch pressed",
    0x02000: "SN error",
    0x04000: "Upper-level model error",
    0x08000: "Lower-level model error",
    0x10000: "USB device error",
    0x20000: "Joint limit exceeded error",
}

# Motor errors grouped by reserve[2] state
hg_motor_errors = {
    0: {  # reserve[2] == 0
        0x01: "Overcurrent",
        0x02: "Overvoltage",
        0x04: "Driver overheating",
        0x08: "Bus undervoltage",
        0x10: "Winding overheating",
        0x20: "Encoder abnormal",
        0x40000000: "Motor-side disconnection timeout",
        0x80000000: "PC-side disconnection timeout",
    },
    2: {  # reserve[2] == 2
        0x01: "Overcurrent",
        0x02: "Phase leakage",
        0x04: "MOS overheating",
        0x08: "Bus undervoltage",
        0x10: "Winding overheating",
        0x20: "Encoder communication error",
        0x40: "Encoder abnormal",
        0x0100: "Warning: Humidity sensor abnormal",
        0x0200: "Warning: Output encoder magnetic field too low",
        0x0400: "Warning: MOS temperature high",
        0x0800: "Warning: Output encoder magnetic field too high",
        0x1000: "Warning: Winding temperature high",
        # Note: 0x0400 reused for disconnection timeout, ambiguous in source
        0x0400: "Motor-side/PC-side disconnection timeout",
    },
    4: {  # reserve[2] == 4
        0x00001: "Overcurrent",
        0x00002: "Transient overvoltage",
        0x00004: "Continuous overvoltage",
        0x00008: "Transient undervoltage",
        0x00010: "Chip overheating",
        0x00020: "MOS overheat/overcool",
        0x00040: "MOS temperature abnormal",
        0x00080: "Housing overheat/overcool",
        0x00100: "Housing temperature abnormal",
        0x00200: "Winding overheating",
        0x00400: "Rotor encoder 1 error",
        0x00800: "Rotor encoder 2 error",
        0x01000: "Output encoder error",
        0x02000: "Calibration/BOOT data error",
        0x04000: "Abnormal reset",
        0x08000: "Motor locked, master control authentication error",
        0x10000: "Chip verification error",
        0x20000: "Calibration mode warning",
        0x40000000: "Motor-side disconnection timeout",
        0x80000000: "PC-side disconnection timeout",
    }
}

go_bit_flag_errors = {
    0x20: "Startup joint abnormal",
    0x01: "Motion control command timeout",
    0x40: "MCU communication abnormal",
    0x80: "Motor communication abnormal",
    # 0x10: "Battery communication abnormal",
    0x02: "Motor overcurrent",
}

# Motor errors (MotorState[motorID].reserve[0])
go_motor_errors = {
    0x0A: {
        0x01: "Overcurrent (MotorState[motorID].reserve[1] indicates current communication frequency)",
        0x02: "Phase leakage",
        0x04: "MOS overheating",
        0x08: "Bus undervoltage",
        0x10: "Winding overheating",
        0x20: "Encoder communication error",
        0x40: "Encoder abnormal",
        0x100: "Motor communication interrupted",
        0x1000: "Command abnormal",
        0x10000: "State abnormal",
    },
    0x01: {
        0x01: "Overcurrent (specific to motorID 10 or 19)",
        0x02: "Overvoltage",
        0x04: "Driver overheating",
        0x08: "Bus undervoltage",
        0x10: "Winding overheating",
        0x20: "Encoder abnormal",
        0x40000000: "Disconnection timeout type 1",
        0x80000000: "Disconnection timeout type 2",
    }
}


def detect_state_err_hg(low_state, mainboard_state=None):
    errors = []

    # --- Mainboard errors ---
    if mainboard_state is not None:
        state_val = mainboard_state.state[0]
        for code, desc in hg_mainboard_errors.items():
            if state_val & code:
                errors.append(("mainboard", hex(code), desc))

    # --- Motor errors ---
    motor_states = low_state.motor_state
    for motor_id, motor in enumerate(motor_states):
        if motor.mode == 0:
            continue
        code_val = motor.motorstate
        reserve = motor.reserve
        reserve2 = reserve[2] if len(reserve) > 2 else 0

        # choose dictionary based on reserve[2]
        err_dict = hg_motor_errors.get(reserve2)
        if err_dict is None:
            continue
        for code, desc in err_dict.items():
            if code_val & code:
                errors.append((f"motor {motor_id}", hex(code), desc))

    return errors


def detect_state_err_go(low_state):
    errors = []

    # --- Mainboard errors ---
    bit_flag = getattr(low_state, "bit_flag", 0)
    for code, desc in go_bit_flag_errors.items():
        if bit_flag & code:
            errors.append(("mainboard", hex(code), desc))

    # --- Motor errors ---
    motor_states = getattr(low_state, "motor_state", [])
    for motor_id, motor in enumerate(motor_states):
        if motor.mode == 0:
            continue
        code_val = motor.reserve[0]

        err_dict = go_motor_errors.get(motor.mode)
        if err_dict is None:
            continue

        for code, desc in err_dict.items():
            if code_val & code:
                errors.append((f"motor {motor_id}", hex(code), desc))

    return errors
