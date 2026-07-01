XR_HAND_JOINTS = [
    'wrist',
    'thumb-metacarpal',
    'thumb-phalanx-proximal',
    'thumb-phalanx-distal',
    'thumb-tip',
    'index-finger-metacarpal',
    'index-finger-phalanx-proximal',
    'index-finger-phalanx-intermediate',
    'index-finger-phalanx-distal',
    'index-finger-tip',
    'middle-finger-metacarpal',
    'middle-finger-phalanx-proximal',
    'middle-finger-phalanx-intermediate',
    'middle-finger-phalanx-distal',
    'middle-finger-tip',
    'ring-finger-metacarpal',
    'ring-finger-phalanx-proximal',
    'ring-finger-phalanx-intermediate',
    'ring-finger-phalanx-distal',
    'ring-finger-tip',
    'pinky-finger-metacarpal',
    'pinky-finger-phalanx-proximal',
    'pinky-finger-phalanx-intermediate',
    'pinky-finger-phalanx-distal',
    'pinky-finger-tip',
]

XR_BODY_JOINTS = [
    'hips',
    'spine-lower',
    'spine-middle',
    'spine-upper',
    'chest',
    'neck',
    'head',
    'left-shoulder',
    'left-scapula',
    'left-arm-upper',
    'left-arm-lower',
    'left-hand-wrist-twist',
    'right-shoulder',
    'right-scapula',
    'right-arm-upper',
    'right-arm-lower',
    'right-hand-wrist-twist',
    'left-hand-palm',
    'left-hand-wrist',
    'left-hand-thumb-metacarpal',
    'left-hand-thumb-phalanx-proximal',
    'left-hand-thumb-phalanx-distal',
    'left-hand-thumb-tip',
    'left-hand-index-metacarpal',
    'left-hand-index-phalanx-proximal',
    'left-hand-index-phalanx-intermediate',
    'left-hand-index-phalanx-distal',
    'left-hand-index-tip',
    'left-hand-middle-phalanx-metacarpal',
    'left-hand-middle-phalanx-proximal',
    'left-hand-middle-phalanx-intermediate',
    'left-hand-middle-phalanx-distal',
    'left-hand-middle-tip',
    'left-hand-ring-metacarpal',
    'left-hand-ring-phalanx-proximal',
    'left-hand-ring-phalanx-intermediate',
    'left-hand-ring-phalanx-distal',
    'left-hand-ring-tip',
    'left-hand-little-metacarpal',
    'left-hand-little-phalanx-proximal',
    'left-hand-little-phalanx-intermediate',
    'left-hand-little-phalanx-distal',
    'left-hand-little-tip',
    'right-hand-palm',
    'right-hand-wrist',
    'right-hand-thumb-metacarpal',
    'right-hand-thumb-phalanx-proximal',
    'right-hand-thumb-phalanx-distal',
    'right-hand-thumb-tip',
    'right-hand-index-metacarpal',
    'right-hand-index-phalanx-proximal',
    'right-hand-index-phalanx-intermediate',
    'right-hand-index-phalanx-distal',
    'right-hand-index-tip',
    'right-hand-middle-metacarpal',
    'right-hand-middle-phalanx-proximal',
    'right-hand-middle-phalanx-intermediate',
    'right-hand-middle-phalanx-distal',
    'right-hand-middle-tip',
    'right-hand-ring-metacarpal',
    'right-hand-ring-phalanx-proximal',
    'right-hand-ring-phalanx-intermediate',
    'right-hand-ring-phalanx-distal',
    'right-hand-ring-tip',
    'right-hand-little-metacarpal',
    'right-hand-little-phalanx-proximal',
    'right-hand-little-phalanx-intermediate',
    'right-hand-little-phalanx-distal',
    'right-hand-little-tip',
    'left-upper-leg',
    'left-lower-leg',
    'left-foot-ankle-twist',
    'left-foot-ankle',
    'left-foot-subtalar',
    'left-foot-transverse',
    'left-foot-ball',
    'right-upper-leg',
    'right-lower-leg',
    'right-foot-ankle-twist',
    'right-foot-ankle',
    'right-foot-subtalar',
    'right-foot-transverse',
    'right-foot-ball',
]

DEFAULT_XR_GAMEPAD_KEYS = [
    'LEFT_AXIS_TOUCHPAD_X',
    'LEFT_AXIS_TOUCHPAD_Y',
    'LEFT_AXIS_THUMBSTICK_X',
    'LEFT_AXIS_THUMBSTICK_Y',
    'LEFT_AXIS_TRIGGER',
    'LEFT_AXIS_SQUEEZE',
    'LEFT_BTN_TOUCHPAD',
    'LEFT_BTN_THUMBSTICK',
    'LEFT_BTN_A',
    'LEFT_BTN_B',
    'RIGHT_AXIS_TOUCHPAD_X',
    'RIGHT_AXIS_TOUCHPAD_Y',
    'RIGHT_AXIS_THUMBSTICK_X',
    'RIGHT_AXIS_THUMBSTICK_Y',
    'RIGHT_AXIS_TRIGGER',
    'RIGHT_AXIS_SQUEEZE',
    'RIGHT_BTN_TOUCHPAD',
    'RIGHT_BTN_THUMBSTICK',
    'RIGHT_BTN_A',
    'RIGHT_BTN_B',
]

DEFAULT_XR_GAMEPAD_KEYS_REDUCED = [
    'LEFT_AXIS_X',
    'LEFT_AXIS_Y',
    'LEFT_AXIS_TRIGGER',
    'LEFT_AXIS_SQUEEZE',
    'LEFT_BTN_AXIS',
    'LEFT_BTN_A',
    'LEFT_BTN_B',
    'RIGHT_AXIS_X',
    'RIGHT_AXIS_Y',
    'RIGHT_AXIS_TRIGGER',
    'RIGHT_AXIS_SQUEEZE',
    'RIGHT_BTN_AXIS',
    'RIGHT_BTN_A',
    'RIGHT_BTN_B',
]

DEFAULT_DESC_XR_GAMEPAD = {
    'name': 'states_xr_gamepad',
    'shape': (24,),
    'layout': DEFAULT_XR_GAMEPAD_KEYS,
}


def cb_xr_gamepad_to_input(
    states_xr_gamepad,
    states_input,
    input_keys=None,
    xr_gamepad_keys=None,
    use_touchpads=False,
    key_mappings=None,
    map_axes=True,
    map_buttons=False,
    map_triggers=False,
    map_bumpers=False,
    add=True,
):
    import numpy as np
    from unicon.utils import coalesce, get_ctx, import_obj, map2inds
    ctx = get_ctx()
    input_keys = coalesce(ctx.get('input_keys'), input_keys, import_obj('unicon.inputs:DEFAULT_INPUT_KEYS'))
    xr_gamepad_keys = coalesce(
        ctx.get('xr_gamepad_keys'),
        xr_gamepad_keys,
        import_obj('unicon.xr:DEFAULT_XR_GAMEPAD_KEYS'),
    )
    default_mapping = {
        'ABS_X': 'LEFT_AXIS_THUMBSTICK_X',
        'ABS_Y': 'LEFT_AXIS_THUMBSTICK_Y',
        'ABS_RX': 'RIGHT_AXIS_THUMBSTICK_X',
        'ABS_RY': 'RIGHT_AXIS_THUMBSTICK_Y',
        'ABS_BRAKE': 'LEFT_AXIS_TRIGGER',
        'ABS_GAS': 'RIGHT_AXIS_TRIGGER',
        'BTN_A': 'RIGHT_BTN_A',
        'BTN_B': 'RIGHT_BTN_B',
        'BTN_X': 'LEFT_BTN_A',
        'BTN_Y': 'LEFT_BTN_B',
        'BTN_THUMBL': 'LEFT_BTN_THUMBSTICK',
        'BTN_THUMBR': 'RIGHT_BTN_THUMBSTICK',
        'BTN_TL': 'LEFT_AXIS_SQUEEZE',
        'BTN_TR': 'RIGHT_AXIS_SQUEEZE',
    }
    key_mappings = coalesce(key_mappings, {})
    maps = default_mapping.copy()
    if use_touchpads:
        maps.update({k: v.replace('THUMBSTICK', 'TOUCHPAD') for k, v in maps.items()})
    if not map_bumpers:
        maps.update({'BTN_TL': None, 'BTN_TR': None})
    if not map_triggers:
        maps.update({'ABS_BRAKE': None, 'ABS_GAS': None})
    if not map_axes:
        maps.update({k: None for k in [
            'ABS_X',
            'ABS_Y',
            'ABS_RX',
            'ABS_RX',
        ]})
    if not map_buttons:
        maps.update({k: None for k in [
            'BTN_A',
            'BTN_B',
            'BTN_X',
            'BTN_Y',
            'BTN_THUMBL',
            'BTN_THUMBR',
        ]})
    maps.update(key_mappings)
    key_mappings = maps
    key_mappings = {k: v for k, v in key_mappings.items() if v is not None}
    inp_inds, xrg_inds = map2inds(input_keys, xr_gamepad_keys, map_a2b=key_mappings)
    print('inp_inds, xrg_inds', inp_inds, xrg_inds)

    def cb():
        if add:
            states_input[inp_inds] = np.clip(states_input[inp_inds] + states_xr_gamepad[xrg_inds], -1., 1.)
        else:
            states_input[inp_inds] = states_xr_gamepad[xrg_inds]

    return cb
