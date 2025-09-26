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
