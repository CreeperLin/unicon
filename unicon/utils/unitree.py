import struct

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


def init_channel(*args, **kwds):
    from unitree_sdk2py.core.channel import ChannelFactory, ChannelFactoryInitialize
    factory = ChannelFactory()
    if factory._ChannelFactory__participant is not None:
        # print('init_channel skipped')
        return
    ChannelFactoryInitialize(*args, **kwds)


def disable_motion():
    import time
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
    msc.SetTimeout(5.0)
    msc.Init()
    status, result = msc.CheckMode()
    for _ in range(10):
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
