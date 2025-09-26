# _default_input_keys = ['ABS_X', 'ABS_Y', 'ABS_Z', 'ABS_RZ', 'BTN_A', 'BTN_B', 'BTN_X', 'BTN_Y', 'BTN_TL', 'BTN_TR']
_default_input_keys = [
    'ABS_X',
    'ABS_Y',
    'ABS_RX',
    'ABS_RY',
    'ABS_HAT0X',
    'ABS_HAT0Y',
    'ABS_BRAKE',
    'ABS_GAS',
    'BTN_A',
    'BTN_B',
    'BTN_X',
    'BTN_Y',
    'BTN_SELECT',
    'BTN_START',
    'BTN_TL',
    'BTN_TR',
]

_key_descs = {
    'ABS_X-': 'left joystick left',
    'ABS_X+': 'left joystick right',
    'ABS_Y-': 'left joystick up',
    'ABS_Y+': 'left joystick down',
    'ABS_RX-': 'right joystick left',
    'ABS_RX+': 'right joystick right',
    'ABS_RY-': 'right joystick up',
    'ABS_RY+': 'right joystick down',
    'ABS_HAT0X-': 'Dpad left',
    'ABS_HAT0X+': 'Dpad right',
    'ABS_HAT0Y-': 'Dpad up',
    'ABS_HAT0Y+': 'Dpad down',
    'ABS_BRAKE+': 'left trigger pushed',
    'ABS_GAS+': 'right trigger pushed',
    'ABS_BRAKE.': 'left trigger rest',
    'ABS_GAS.': 'right trigger rest',
    'BTN_A': 'button south',
    'BTN_B': 'button east',
    'BTN_X': 'button west',
    'BTN_Y': 'button north',
    'BTN_SELECT': 'button option left',
    'BTN_START': 'button option right',
    'BTN_TL': 'left bumper',
    'BTN_TR': 'right bumper',
}


def test_cb_input(cb_input_cls, cb=None, states_input=None, input_keys=None):
    import numpy as np
    import time
    dt = 0.02
    th = 0.1
    input_keys = _default_input_keys if input_keys is None else input_keys
    states_input = np.zeros(len(input_keys)) if states_input is None else states_input
    cb = cb_input_cls(states_input) if cb is None else cb
    assert cb is not None
    num_steps = int(3 // dt)

    def wait_for_input(idx, tgt):
        for _ in range(num_steps):
            cb()
            v = states_input[idx]
            if abs(v - tgt) < th:
                print('passed')
                return
            time.sleep(dt)
        if abs(v - tgt) > 1:
            print('wrong dir', tgt, v)
        else:
            print('failed', idx, tgt, v)
        print('states_input', np.round(states_input, 2).tolist())

    # for _ in range(2**10):
    #     cb()
    #     time.sleep(dt)
    #     print('states_input', np.round(states_input, 2).tolist())

    for _ in range(num_steps):
        cb()
        time.sleep(dt)

    if np.any(np.abs(states_input) > 0.1):
        print('zero test failed', np.round(states_input, 2).tolist())

    d2k = ['-', '.', '+']
    for i, key in enumerate(input_keys):
        if 'ABS' not in key:
            continue
        for d in [-1, 0, +1]:
            k = d2k[int(d + 1)]
            desc = _key_descs.get(f"{key}{k}", None)
            if desc is None:
                continue
            print(f'push key {key} with dir {d} ({desc})')
            wait_for_input(i, d)

    for i, key in enumerate(input_keys):
        if 'BTN' not in key:
            continue
        desc = _key_descs.get(key, key)
        print(f'push key {key} ({desc})')
        d = 1
        wait_for_input(i, d)
