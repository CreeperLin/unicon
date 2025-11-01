axis_mapping_dinput = {
    'ABS_X': 0,
    'ABS_Y': 1,
    'ABS_RX': 3,
    'ABS_RY': 4,
    'ABS_BRAKE': 2,
    'ABS_GAS': 5,
}
btn_mapping_dinput = {
    'BTN_A': 0,
    'BTN_B': 1,
    'BTN_X': 3,
    'BTN_Y': 4,
    'BTN_TL': 6,
    'BTN_TR': 7,
    'BTN_SELECT': 8,
    'BTN_START': 9,
}
hat_mapping_dinput = {
    'ABS_HAT0': (0, 1, -1),
}

axis_mapping_xbox = {
    'ABS_X': 0,
    'ABS_Y': 1,
    'ABS_RX': 2,
    'ABS_RY': 3,
    'ABS_BRAKE': 4,
    'ABS_GAS': 5,
}
btn_mapping_xbox = {
    'BTN_A': 0,
    'BTN_B': 1,
    'BTN_X': 2,
    'BTN_Y': 3,
    'BTN_TL': 4,
    'BTN_TR': 5,
    'BTN_SELECT': 6,
    'BTN_START': 7,
}
hat_mapping_xbox = {
    'ABS_HAT0': (0, 1, -1),
}

axis_mapping_ps4 = {
    'ABS_X': 0,
    'ABS_Y': 1,
    'ABS_RX': 2,
    'ABS_RY': 3,
    'ABS_BRAKE': 4,
    'ABS_GAS': 5,
}
btn_mapping_ps4 = {
    'BTN_A': 0,
    'BTN_B': 1,
    'BTN_X': 2,
    'BTN_Y': 3,
    'BTN_TL': 9,
    'BTN_TR': 10,
    'BTN_SELECT': 4,
    'BTN_START': 6,
    'ABS_HAT0Y-': 11,
    'ABS_HAT0Y+': 12,
    'ABS_HAT0X-': 13,
    'ABS_HAT0X+': 14,
}

axis_mapping_ps5 = {
    'ABS_X': 0,
    'ABS_Y': 1,
    'ABS_RX': 2,
    'ABS_RY': 3,
    'ABS_BRAKE': 4,
    'ABS_GAS': 5,
}
btn_mapping_ps5 = {
    'BTN_A': 0,
    'BTN_B': 1,
    'BTN_X': 2,
    'BTN_Y': 3,
    'BTN_TL': 9,
    'BTN_TR': 10,
    'BTN_SELECT': 4,
    'BTN_START': 6,
    'ABS_HAT0Y-': 11,
    'ABS_HAT0Y+': 12,
    'ABS_HAT0X-': 13,
    'ABS_HAT0X+': 14,
}

axis_mapping_ps5_alt = {
    'ABS_X': 0,
    'ABS_Y': 1,
    'ABS_RX': 2,
    'ABS_BRAKE': 3,
    'ABS_GAS': 4,
    'ABS_RY': 5,
}
btn_mapping_ps5_alt = {
    'BTN_X': 0,
    'BTN_A': 1,
    'BTN_B': 2,
    'BTN_Y': 3,
    'BTN_TL': 4,
    'BTN_TR': 5,
    'BTN_SELECT': 8,
    'BTN_START': 9,
}
hat_mapping_ps5_alt = {
    'ABS_HAT0': (0, 1, -1),
}

axis_mapping_switchpro = {
    'ABS_X': 0,
    'ABS_Y': 1,
    'ABS_RX': 2,
    'ABS_RY': 3,
    'ABS_BRAKE': 4,
    'ABS_GAS': 5,
}
btn_mapping_switchpro = {
    'BTN_A': 0,
    'BTN_B': 1,
    'BTN_X': 2,
    'BTN_Y': 3,
    'BTN_TL': 9,
    'BTN_TR': 10,
    'BTN_SELECT': 4,
    'BTN_START': 6,
    'ABS_HAT0Y-': 11,
    'ABS_HAT0Y+': 12,
    'ABS_HAT0X-': 13,
    'ABS_HAT0X+': 14,
}


def cb_input_pygame(
    states_input,
    device_index=0,
    verbose=False,
    input_keys=None,
    joystick_type=None,
    remap_trigger=True,
    alt=False,
):
    from unicon.utils import printv
    input_keys = __import__('unicon.inputs').inputs._default_input_keys if input_keys is None else input_keys
    import pygame
    pygame.init()
    pygame.joystick.init()
    assert pygame.joystick.get_init()

    js = None
    num_buttons = None
    num_axes = None
    num_hats = None
    name = None
    instance_id = None
    axis_mapping, btn_mapping, hat_mapping = None, None, None
    states = {}

    def init_js(dev_idx):
        nonlocal js
        nonlocal num_buttons, num_axes, num_hats, name, instance_id
        nonlocal axis_mapping, btn_mapping, hat_mapping
        js = pygame.joystick.Joystick(dev_idx)
        js.init()
        if not js.get_init():
            js.quit()
            js = None
            return 1
        num_buttons = js.get_numbuttons()
        num_axes = js.get_numaxes()
        num_hats = js.get_numhats()
        name = js.get_name()
        instance_id = js.get_instance_id()
        print('get_id', js.get_id())
        print('get_instance_id', js.get_instance_id())
        print('get_guid', js.get_guid())
        print('get_power_level', js.get_power_level())
        print('get_name', js.get_name())
        print('get_numaxes', js.get_numaxes())
        print('get_numballs', js.get_numballs())
        print('get_numbuttons', js.get_numbuttons())
        print('get_numhats', js.get_numhats())

        js_type = joystick_type
        if js_type is None:
            if 'PS4' in name:
                js_type = 'ps4'
            elif 'DualSense' in name:
                js_type = 'ps5'
            elif 'Xbox' in name:
                js_type = 'xbox'
            elif 'Switch Pro' in name:
                js_type = 'switchpro'
            elif 'Wireless' in name:
                js_type = 'dinput'
        if js_type is None:
            return 1
        js_type = f'{js_type}_alt' if alt else js_type
        print('joystick_type', js_type)
        axis_mapping = globals().get(f'axis_mapping_{js_type}', {})
        btn_mapping = globals().get(f'btn_mapping_{js_type}', {})
        hat_mapping = globals().get(f'hat_mapping_{js_type}', {})

    def cb():
        nonlocal js
        for i, k in enumerate(input_keys):
            if k.startswith('ABS_HAT'):
                neg = states.get(k + '-', 0)
                pos = states.get(k + '+', 0)
                val = states.get(k, 0)
                states_input[i] = val - neg + pos
                continue
            states_input[i] = states.get(k, 0)
        pygame.event.pump()
        try:
            evs = list(pygame.event.get())
        except SystemError:
            if js is not None:
                js.quit()
            js = None
            evs = []
        for event in evs:
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.JOYDEVICEADDED:
                if js is None and event.device_index == device_index:
                    if init_js(device_index):
                        print(f"Joystick init failed")
                print(f"Joystick {event.device_index} connencted {js}")

            if event.type == pygame.JOYDEVICEREMOVED:
                if event.instance_id == instance_id:
                    if js is not None:
                        js.quit()
                    js = None
                    states_input[:] = 0.
                print(f"Joystick {event.instance_id} disconnected {js}")

        if js is None:
            if pygame.joystick.get_count() == 0:
                printv('pygame no joystick')
            return
        axes = [js.get_axis(i) for i in range(num_axes)]
        btns = [js.get_button(i) for i in range(num_buttons)]
        hats = [js.get_hat(i) for i in range(num_hats)]
        if verbose:
            print(axes, btns, hats)
        for k, v in axis_mapping.items():
            if v >= len(axes):
                continue
            states[k] = axes[v]
        for k, v in btn_mapping.items():
            if v >= len(btns):
                continue
            states[k] = btns[v]
        for k, v in hat_mapping.items():
            v = [v, 1, 1] if isinstance(v, int) else v
            idx, xd, yd = v
            if idx >= len(hats):
                continue
            hat = hats[idx]
            states[k + 'X'] = hat[0] * xd
            states[k + 'Y'] = hat[1] * yd
        if remap_trigger:
            for k in ['ABS_BRAKE', 'ABS_GAS']:
                if k not in states:
                    continue
                states[k] = (states[k] + 1) * 0.5
        if verbose:
            print('states_input', states_input.tolist())

    def cb_close():
        pygame.joystick.quit()

    return cb


from functools import partial

cb_input_alt = partial(cb_input_pygame, alt=True)

if __name__ == '__main__':
    from unicon.inputs import test_cb_input
    test_cb_input(cb_input_pygame)
