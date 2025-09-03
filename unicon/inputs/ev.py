def cb_input_ev(
    states_input,
    verbose=False,
    # verbose=True,
    device=None,
    input_keys=None,
    blocking=False,
    remap_pedals=True,
    abs_type=0,
    dev_path='/dev/input',
    z2r=True,
    use_nesw=True,
):
    from unicon.utils import cmd
    import os
    input_keys = __import__('unicon.inputs').inputs._default_input_keys if input_keys is None else input_keys

    input_states = {}
    states = {}

    if not os.path.exists(dev_path):
        print(f'{dev_path} not exist')
        return None
    if device is None:
        dev_root = os.path.join(dev_path, 'by-path')
        devices = sorted(filter(lambda x: 'event' in x and 'joystick' in x, os.listdir(dev_root)))
        if len(devices):
            device = os.path.join(dev_root, devices[0])
    if device is None:
        dev_root = dev_path
        devs = filter(lambda x: 'event' in x, os.listdir(dev_root))
        device = os.path.join(dev_root, sorted(devs, key=lambda x: int(x[5:]))[-1])

    import evdev
    ecodes_kb = {}
    ecodes_kb.update(evdev.ecodes.BTN)
    ecodes_kb.update(evdev.ecodes.KEY)

    if use_nesw:
        abxy2nesw = {
            'BTN_A': 'BTN_SOUTH',
            'BTN_B': 'BTN_EAST',
            'BTN_X': 'BTN_WEST',
            'BTN_Y': 'BTN_NORTH',
        }
        input_keys = [abxy2nesw.get(k, k) for k in input_keys]

    ecodes_abs = evdev.ecodes.ABS.copy()
    if z2r:
        ecodes_abs[evdev.ecodes.ABS_Z] = ['ABS_Z', 'ABS_RX']
        ecodes_abs[evdev.ecodes.ABS_RZ] = ['ABS_RZ', 'ABS_RY']

    code_maps = {
        evdev.ecodes.EV_KEY: ecodes_kb,
        evdev.ecodes.EV_ABS: ecodes_abs,
        evdev.ecodes.EV_REL: evdev.ecodes.REL,
        evdev.ecodes.EV_MSC: evdev.ecodes.MSC,
    }

    print('cb_input_ev', use_nesw, z2r)
    print('ecodes_kb', [ecodes_kb.get(getattr(evdev.ecodes, k)) for k in input_keys if k.startswith('BTN')])
    print('ecodes_abs', [ecodes_abs.get(getattr(evdev.ecodes, k)) for k in input_keys if k.startswith('ABS')])

    print('opening event device', device)

    if cmd('test -r', [device]):
        raise RuntimeError(f'no read permission on {device}')

    dev = evdev.InputDevice(device)

    def read_nonblocking():
        nonlocal dev
        while True:
            try:
                ev = dev.read_one()
            except Exception as e:
                print('evdev read error', e)
                dev = None
                return
            if ev is None:
                return
            yield ev

    n_fails = 0
    retry_intv = 100
    retry_pt = 0

    def cb():
        nonlocal abs_type, dev, retry_pt, n_fails
        for i, k in enumerate(input_keys):
            states_input[i] = states.get(k, 0)
        if dev is None:
            if retry_pt:
                retry_pt -= 1
                return
            try:
                dev = evdev.InputDevice(device)
                print('evdev reopened', device)
                n_fails = 0
            except Exception as e:
                n_fails += 1
                retry_pt = retry_intv
                print('evdev reopen error', n_fails, e, device)
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
                    print('ignored event', evdev.ecodes.EV[etype], ecode, evalue)
                    continue
            else:
                cds = codes[ecode]
                cds = [cds] if isinstance(cds, str) else cds
                for c in cds:
                    input_states[c] = evalue
                if verbose:
                    print(evdev.ecodes.EV[etype], cds, evalue)
            # print('input_states', input_states)
            for i, k in enumerate(input_keys):
                v = input_states.get(k)
                if remap_pedals and k in ['ABS_GAS', 'ABS_BRAKE']:
                    if v is not None and v > 256:
                        abs_type = 1
                    if abs_type == 1:
                        v = 0 if v is None else v / 32767.0
                    else:
                        v = 0 if v is None else v / 255
                elif k.startswith('ABS_HAT'):
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
                states[k] = v
        if verbose:
            print('states_input', states_input.tolist())

    return cb


if __name__ == '__main__':
    from unicon.inputs import test_cb_input
    test_cb_input(cb_input_ev)
