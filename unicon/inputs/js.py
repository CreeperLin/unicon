def cb_input_js(
    states_input,
    verbose=False,
    # verbose=True,
    device=None,
    input_keys=None,
    min_num_axes=2,
    min_num_buttons=2,
    blocking=False,
    dev_path='/dev/input',
    mode=None,
    z2r=True,
    z2t=False,
    remap_trigger=True,
    wait_no_dev=False,
    try_chmod=True,
):
    from unicon.utils import cmd
    input_keys = __import__('unicon.inputs').inputs._default_input_keys if input_keys is None else input_keys
    # Released by rdb under the Unlicense (unlicense.org)
    # Based on information from:
    # https://www.kernel.org/doc/Documentation/input/joystick-api.txt
    import os
    import struct
    import array
    from fcntl import ioctl
    try:
        import evdev
        button_names = evdev.ecodes.BTN
        axis_names = evdev.ecodes.ABS
    except ImportError:
        axis_names = {
            10: 'ABS_BRAKE',
            64: 'ABS_CNT',
            25: 'ABS_DISTANCE',
            9: 'ABS_GAS',
            16: 'ABS_HAT0X',
            17: 'ABS_HAT0Y',
            18: 'ABS_HAT1X',
            19: 'ABS_HAT1Y',
            20: 'ABS_HAT2X',
            21: 'ABS_HAT2Y',
            22: 'ABS_HAT3X',
            23: 'ABS_HAT3Y',
            63: 'ABS_MAX',
            40: 'ABS_MISC',
            56: 'ABS_MT_BLOB_ID',
            59: 'ABS_MT_DISTANCE',
            52: 'ABS_MT_ORIENTATION',
            53: 'ABS_MT_POSITION_X',
            54: 'ABS_MT_POSITION_Y',
            58: 'ABS_MT_PRESSURE',
            47: 'ABS_MT_SLOT',
            55: 'ABS_MT_TOOL_TYPE',
            60: 'ABS_MT_TOOL_X',
            61: 'ABS_MT_TOOL_Y',
            48: 'ABS_MT_TOUCH_MAJOR',
            49: 'ABS_MT_TOUCH_MINOR',
            57: 'ABS_MT_TRACKING_ID',
            50: 'ABS_MT_WIDTH_MAJOR',
            51: 'ABS_MT_WIDTH_MINOR',
            24: 'ABS_PRESSURE',
            33: 'ABS_PROFILE',
            46: 'ABS_RESERVED',
            7: 'ABS_RUDDER',
            3: 'ABS_RX',
            4: 'ABS_RY',
            5: 'ABS_RZ',
            6: 'ABS_THROTTLE',
            26: 'ABS_TILT_X',
            27: 'ABS_TILT_Y',
            28: 'ABS_TOOL_WIDTH',
            32: 'ABS_VOLUME',
            8: 'ABS_WHEEL',
            0: 'ABS_X',
            1: 'ABS_Y',
            2: 'ABS_Z'
        }
        button_names = {
            256: ['BTN_0', 'BTN_MISC'],
            257: 'BTN_1',
            258: 'BTN_2',
            259: 'BTN_3',
            260: 'BTN_4',
            261: 'BTN_5',
            262: 'BTN_6',
            263: 'BTN_7',
            264: 'BTN_8',
            265: 'BTN_9',
            304: ['BTN_A', 'BTN_GAMEPAD', 'BTN_SOUTH'],
            305: ['BTN_B', 'BTN_EAST'],
            278: 'BTN_BACK',
            294: 'BTN_BASE',
            295: 'BTN_BASE2',
            296: 'BTN_BASE3',
            297: 'BTN_BASE4',
            298: 'BTN_BASE5',
            299: 'BTN_BASE6',
            306: 'BTN_C',
            303: 'BTN_DEAD',
            320: ['BTN_DIGI', 'BTN_TOOL_PEN'],
            545: 'BTN_DPAD_DOWN',
            546: 'BTN_DPAD_LEFT',
            547: 'BTN_DPAD_RIGHT',
            544: 'BTN_DPAD_UP',
            276: 'BTN_EXTRA',
            277: 'BTN_FORWARD',
            336: ['BTN_GEAR_DOWN', 'BTN_WHEEL'],
            337: 'BTN_GEAR_UP',
            288: ['BTN_JOYSTICK', 'BTN_TRIGGER'],
            272: ['BTN_LEFT', 'BTN_MOUSE'],
            274: 'BTN_MIDDLE',
            316: 'BTN_MODE',
            307: ['BTN_NORTH', 'BTN_X'],
            293: 'BTN_PINKIE',
            273: 'BTN_RIGHT',
            314: 'BTN_SELECT',
            275: 'BTN_SIDE',
            315: 'BTN_START',
            331: 'BTN_STYLUS',
            332: 'BTN_STYLUS2',
            329: 'BTN_STYLUS3',
            279: 'BTN_TASK',
            289: 'BTN_THUMB',
            290: 'BTN_THUMB2',
            317: 'BTN_THUMBL',
            318: 'BTN_THUMBR',
            310: 'BTN_TL',
            312: 'BTN_TL2',
            324: 'BTN_TOOL_AIRBRUSH',
            322: 'BTN_TOOL_BRUSH',
            333: 'BTN_TOOL_DOUBLETAP',
            325: 'BTN_TOOL_FINGER',
            327: 'BTN_TOOL_LENS',
            326: 'BTN_TOOL_MOUSE',
            323: 'BTN_TOOL_PENCIL',
            335: 'BTN_TOOL_QUADTAP',
            328: 'BTN_TOOL_QUINTTAP',
            321: 'BTN_TOOL_RUBBER',
            334: 'BTN_TOOL_TRIPLETAP',
            291: 'BTN_TOP',
            292: 'BTN_TOP2',
            330: 'BTN_TOUCH',
            311: 'BTN_TR',
            313: 'BTN_TR2',
            308: ['BTN_WEST', 'BTN_Y'],
            309: 'BTN_Z'
        }
    if mode is not None:
        mode = int(mode)
        z2r = (mode) & 1 > 0
        z2t = (mode >> 1) & 1 > 0
    if z2r:
        axis_names.update({
            2: 'ABS_RX',
            5: 'ABS_RY',
        })
    elif z2t:
        axis_names.update({
            2: 'ABS_BRAKE',
            5: 'ABS_GAS',
        })
    print('cb_input_js', 'z2t', z2t, 'z2r', z2r)

    states = {}
    axis_map = []
    button_map = []

    if not os.path.exists(dev_path):
        print(f'{dev_path} not exist')
        return None

    def init_js(path):
        if cmd('test -r', [path]):
            if try_chmod:
                cmd('sudo chmod 666', [path])
            else:
                print(f'no read permission on {path}')
                return None
        print('Opening %s...' % path)
        jsdev = open(path, 'rb')
        # Get the device name.
        buf = array.array('B', [0] * 64)
        ioctl(jsdev, 0x80006a13 + (0x10000 * len(buf)), buf)  # JSIOCGNAME(len)
        js_name = buf.tobytes().rstrip(b'\x00').decode('utf-8')
        print('Device name: %s' % js_name)
        # Get number of axes and buttons.
        buf = array.array('B', [0])
        ioctl(jsdev, 0x80016a11, buf)  # JSIOCGAXES
        num_axes = buf[0]
        buf = array.array('B', [0])
        ioctl(jsdev, 0x80016a12, buf)  # JSIOCGBUTTONS
        num_buttons = buf[0]
        # Get the axis map.
        buf = array.array('B', [0] * 0x40)
        ioctl(jsdev, 0x80406a32, buf)  # JSIOCGAXMAP
        for axis in buf[:num_axes]:
            axis_name = axis_names.get(axis, 'unknown(0x%02x)' % axis)
            axis_map.append(axis_name)
            states[axis_name] = 0.0
        # Get the button map.
        buf = array.array('H', [0] * 200)
        ioctl(jsdev, 0x80406a34, buf)  # JSIOCGBTNMAP
        for btn in buf[:num_buttons]:
            btn_names = button_names.get(btn, 'unknown(0x%03x)' % btn)
            btn_names = [btn_names] if isinstance(btn_names, str) else btn_names
            button_map.append(btn_names)
            for n in btn_names:
                states[n] = 0
        print('%d axes found: %s' % (num_axes, ', '.join(axis_map)))
        print('%d buttons found: %s' % (num_buttons, ', '.join(map(str, button_map))))
        if num_axes < min_num_axes or num_buttons < min_num_buttons:
            print('below minimum', num_axes, min_num_axes, num_buttons, min_num_buttons)
            return None
        if not blocking:
            os.set_blocking(jsdev.fileno(), False)
        return jsdev

    def try_jsdevs():
        if device is None:
            paths = list(filter(lambda x: 'js' in x, map(lambda x: os.path.join(dev_path, x), os.listdir(dev_path))))
        else:
            paths = [device]
        for path in paths:
            if not os.path.exists(path):
                continue
            jsdev = init_js(path)
            if jsdev is not None:
                return jsdev

    jsdev = try_jsdevs()
    if jsdev is None:
        print('no js device')
        if not wait_no_dev:
            return None

    n_fails = 0
    retry_intv = 100
    retry_pt = 0

    def cb():
        nonlocal jsdev, n_fails, retry_pt
        for i, k in enumerate(input_keys):
            states_input[i] = states.get(k, 0)
        if jsdev is None:
            if retry_pt:
                retry_pt -= 1
                return
            e2 = None
            try:
                jsdev = try_jsdevs()
            except Exception as e2:
                jsdev = None
            if jsdev is None:
                print('jsdev reopen error', n_fails, e2)
                n_fails += 1
                retry_pt = retry_intv
                return
            else:
                print('jsdev reconnected', n_fails)
        while True:
            try:
                evbuf = jsdev.read(8)
            except Exception as e:
                n_fails += 1
                if n_fails < 50:
                    print('jsdev error', n_fails, e)
                else:
                    jsdev = None
                return
            n_fails = 0
            if evbuf is None or not evbuf:
                return
            time, value, typ, number = struct.unpack('IhBB', evbuf)
            if typ & 0x80:
                if verbose:
                    print(time, "(initial)", value, number)
            elif typ & 0x01:
                btn_names = button_map[number]
                for n in btn_names:
                    states[n] = value
                if verbose:
                    print(time, "button: %d %s %s" % (number, btn_names, 'pressed' if value else 'released'))
            elif typ & 0x02:
                axis = axis_map[number]
                fvalue = value / 32767.0
                if remap_trigger and axis in ['ABS_BRAKE', 'ABS_GAS']:
                    fvalue = (fvalue + 1) * 0.5
                states[axis] = fvalue
                if verbose:
                    print(time, 'axis', number, axis, value, fvalue)
            elif verbose:
                print('unknown js ev', typ)
            if verbose:
                print('states_input', states_input.tolist())

    return cb


from functools import partial

cb_input_js0 = partial(cb_input_js, mode=0)
cb_input_js1 = partial(cb_input_js, mode=1)
cb_input_js2 = partial(cb_input_js, mode=2)

if __name__ == '__main__':
    from unicon.inputs import test_cb_input
    test_cb_input(cb_input_js)
