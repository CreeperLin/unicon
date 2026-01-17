def cb_input_ev(
    states_input,
    verbose=False,
    # verbose=True,
    retry_sudo=False,
    device_path=None,
    input_keys=None,
    blocking=False,
    remap_trigger=True,
    abs_type=0,
    dev_path='/dev/input',
    z2r=False,
    z2t=True,
    use_nesw=True,
    ecodes_abs_updates=None,
    ecodes_btn_updates=None,
    input_key_updates=None,
    includes=(
        'joystick', 'controller', 'wireless',
    ),
    excludes=(
        'touchpad', 'motion sensor', 'hdmi', 'pch', 'button', 'video bus', 'hotkey', 'hid events',
    ),
    wait_dev=True,
):
    from unicon.utils import cmd, coalesce, get_ctx, import_obj, expect, validate_sudo
    import os
    import fcntl
    input_keys = coalesce(get_ctx().get('input_keys'), input_keys, import_obj('unicon.inputs:DEFAULT_INPUT_KEYS'))

    input_states = {}
    states = {}

    if not os.path.exists(dev_path):
        print(f'{dev_path} not exist')
        return None

    excluded_devs = set()

    def find_device(_sudo):
        devices = []
        has_sudo = not validate_sudo(interactive=_sudo)
        for dev in os.listdir(dev_path):
            if not dev.startswith('event'):
                continue
            if dev in excluded_devs:
                continue
            path = os.path.join(dev_path, dev)
            if not os.access(path, os.R_OK) and has_sudo:
                cmd('sudo chmod +r', [path], verbose=True)
            try:
                with open(path, 'rb') as f:
                    # EVIOCGNAME ioctl to get device name
                    buf = bytearray(256)
                    fcntl.ioctl(f, 0x82004506, buf)  # EVIOCGNAME(len)
                    name = buf.split(b'\0', 1)[0].decode()
            except Exception as e:
                print(path, e)
                continue
            print(path, name)
            name = name.lower()
            if any(p in name for p in excludes) or all(p not in name for p in includes):
                excluded_devs.add(dev)
                continue
            devices.append(path)
        if not len(devices):
            return None
        print('evdev find_device', devices)
        return devices[0]

    device_path = find_device(_sudo=True) if device_path is None else device_path
    expect(device_path is not None or wait_dev, 'no evdev found')

    import evdev
    import evdev.ecodes as ecodes
    ecodes_btn = {}
    ecodes_btn.update(ecodes.BTN)
    ecodes_btn.update(ecodes.KEY)

    input_key_updates = {} if input_key_updates is None else input_key_updates
    if use_nesw:
        input_key_updates.update({
            'BTN_A': 'BTN_SOUTH',
            'BTN_B': 'BTN_EAST',
            'BTN_X': 'BTN_WEST',
            'BTN_Y': 'BTN_NORTH',
        })
    input_keys = [input_key_updates.get(k, k) for k in input_keys]

    ecodes_abs = ecodes.ABS.copy()
    if z2r:
        ecodes_abs[ecodes.ABS_Z] = ['ABS_Z', 'ABS_RX']
        ecodes_abs[ecodes.ABS_RZ] = ['ABS_RZ', 'ABS_RY']
    elif z2t:
        ecodes_abs[ecodes.ABS_Z] = ['ABS_Z', 'ABS_BRAKE']
        ecodes_abs[ecodes.ABS_RZ] = ['ABS_RZ', 'ABS_GAS']

    for codes, updates in [[ecodes_abs, ecodes_abs_updates], [ecodes_btn, ecodes_btn_updates]]:
        if updates is None:
            continue
        updates = {(getattr(ecodes, k) if isinstance(k, str) else k): v for k, v in updates.items()}
        codes.update(updates)

    ecodes_btn = {k: ([v] if isinstance(v, str) else v) for k, v in ecodes_btn.items()}
    ecodes_abs = {k: ([v] if isinstance(v, str) else v) for k, v in ecodes_abs.items()}

    code_maps = {
        ecodes.EV_KEY: ecodes_btn,
        ecodes.EV_ABS: ecodes_abs,
        ecodes.EV_REL: ecodes.REL,
        ecodes.EV_MSC: ecodes.MSC,
    }

    print('cb_input_ev', use_nesw, z2r)
    print('input_keys', input_keys)
    print('ecodes_btn', [ecodes_btn.get(getattr(ecodes, k)) for k in input_keys if k.startswith('BTN')])
    print('ecodes_abs', [ecodes_abs.get(getattr(ecodes, k)) for k in input_keys if k.startswith('ABS')])

    dev = None
    if device_path is not None:
        dev = evdev.InputDevice(device_path)

    def read_nonblocking():
        nonlocal dev
        while True:
            try:
                ev = dev.read_one()
            except Exception as e:
                print('evdev read error', e)
                input_states.clear()
                states.clear()
                dev = None
                return
            if ev is None:
                return
            yield ev

    n_fails = 0
    retry_intv = 100
    retry_pt = 0

    def cb():
        nonlocal abs_type, device_path, dev, retry_pt, n_fails
        for i, k in enumerate(input_keys):
            states_input[i] = states.get(k, 0)
        if device_path is None:
            if retry_pt:
                retry_pt -= 1
                return
            device_path = find_device(_sudo=retry_sudo)
            retry_pt = 0
        if device_path is None:
            print('evdev no device')
            retry_pt = retry_intv
            return
        if dev is None:
            if retry_pt:
                retry_pt -= 1
                return
            try:
                print('opening event device', device_path)
                if not os.access(device_path, os.R_OK) and not validate_sudo(retry_sudo):
                    cmd('sudo chmod +r', [device_path], verbose=True)
                dev = evdev.InputDevice(device_path)
                print('evdev reopened', device_path)
                n_fails = 0
            except Exception as e:
                n_fails += 1
                retry_pt = retry_intv
                print('evdev reopen error', n_fails, e, device_path)
                return
        gen = dev.read_loop if blocking else read_nonblocking
        for event in gen():
            # print(event)
            ecode = event.code
            evalue = event.value
            etype = event.type
            codes = code_maps.get(etype)
            if codes is None:
                if verbose:
                    print('ignored event', ecodes.EV[etype], ecode, evalue)
                    continue
            else:
                cds = codes[ecode]
                for c in cds:
                    input_states[c] = evalue
                if verbose:
                    print(ecodes.EV[etype], cds, evalue)
            # print('input_states', input_states)
            for i, k in enumerate(input_keys):
                v = input_states.get(k)
                if k.startswith('ABS_HAT'):
                    v = 0 if v is None else v
                elif k.startswith('ABS'):
                    if v is not None and v > 256:
                        abs_type = 1
                    if abs_type == 1:
                        v = 0 if v is None else v / 32767.0
                    else:
                        v = 0 if v is None else (v - 128) / 128
                else:
                    v = 0 if v is None else v
                v = min(v, 1, max(v, -1))
                if remap_trigger and k in ['ABS_BRAKE', 'ABS_GAS'] and k in input_states:
                    v = (v + 1) * 0.5
                states[k] = v
        # if verbose:
        # print('states_input', states_input.tolist())

    return cb


if __name__ == '__main__':
    from unicon.inputs import test_cb_input
    test_cb_input(cb_input_ev)
