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
):
    import os
    input_keys = __import__('unicon.inputs').inputs._default_input_keys if input_keys is None else input_keys

    input_states = {}

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
        device = os.path.join(dev_root, sorted(filter(lambda x: 'event' in x, os.listdir(dev_root)))[-1])

    import evdev
    kbs = {}
    kbs.update(evdev.ecodes.BTN)
    kbs.update(evdev.ecodes.KEY)

    code_maps = {
        evdev.ecodes.EV_KEY: kbs,
        evdev.ecodes.EV_ABS: evdev.ecodes.ABS,
        evdev.ecodes.EV_REL: evdev.ecodes.REL,
        evdev.ecodes.EV_MSC: evdev.ecodes.MSC,
    }

    print('opening event device', device)
    os.system(f'sudo chmod 666 {device}')
    device = evdev.InputDevice(device)

    def read_nonblocking():
        while True:
            try:
                ev = device.read_one()
            except Exception as e:
                print('evdev read error', e)
                return
            if ev is None:
                return
            yield ev

    gen = device.read_loop if blocking else read_nonblocking

    def cb():
        nonlocal abs_type
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
            for i, k in enumerate(input_keys):
                v = input_states.get(k)
                if remap_pedals and k in ['ABS_GAS', 'ABS_BRAKE']:
                    if v is not None and v > 256:
                        abs_type = 1
                    if abs_type == 1:
                        v = 0 if v is None else v / 32767.0
                    else:
                        v = 0 if v is None else v / 255
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
                states_input[i] = v
            # print('states_input', states_input.tolist())
            if verbose:
                print('states_input', states_input.tolist())

    return cb


if __name__ == '__main__':
    import numpy as np
    states_input = np.zeros(7)
    cb = cb_input_ev(states_input, verbose=True)
    while True:
        cb()
