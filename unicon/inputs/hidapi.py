def cb_input_hidapi(
    states_input,
    verbose=False,
    device=None,
    input_keys=None,
    vendor_id=None,
    product_id=None,
    serial_number=None,
    remap_trigger=True,
    blocking=False,
):
    """
    HID API input callback for gamepad/joystick devices.

    Args:
        states_input: numpy array to store input states
        verbose: print debug information
        device: HID device path (optional)
        input_keys: list of input keys to map
        vendor_id: USB vendor ID to filter devices
        product_id: USB product ID to filter devices
        serial_number: device serial number to filter
        remap_trigger: remap trigger axes from [-1,1] to [0,1]
        blocking: use blocking reads (not recommended)
    """
    from unicon.utils import coalesce, get_ctx, import_obj
    input_keys = coalesce(get_ctx().get('input_keys'), input_keys, import_obj('unicon.inputs:DEFAULT_INPUT_KEYS'))

    try:
        import hid
    except ImportError:
        print('hidapi not available, install with: pip install hidapi')
        return None

    input_states = {}
    states = {}
    dev = None
    n_fails = 0
    retry_intv = 100
    retry_pt = 0

    # Common gamepad HID report layouts
    # These are approximate and may need adjustment for specific devices
    gamepad_parsers = {
        # Xbox controllers (typical layout)
        'xbox': {
            'axes': {
                'ABS_X': (1, 2, 'int16'),
                'ABS_Y': (3, 2, 'int16'),
                'ABS_RX': (5, 2, 'int16'),
                'ABS_RY': (7, 2, 'int16'),
                'ABS_BRAKE': (9, 1, 'uint8'),
                'ABS_GAS': (10, 1, 'uint8'),
            },
            'buttons': {
                'BTN_A': (11, 0x01),
                'BTN_B': (11, 0x02),
                'BTN_X': (11, 0x08),
                'BTN_Y': (11, 0x10),
                'BTN_TL': (11, 0x40),
                'BTN_TR': (11, 0x80),
                'BTN_SELECT': (12, 0x04),
                'BTN_START': (12, 0x08),
            },
            'hat': {
                'offset': 12,
                'mask': 0xF0,
            }
        },
        # PS4/PS5 controllers - Short report format (Bluetooth)
        # Report format: [01, left_x, left_y, right_x, right_y, face_buttons, shoulder_buttons, dpad, L2_analog, R2_analog]
        'playstation_short': {
            'axes': {
                'ABS_X': (1, 1, 'uint8'),  # Left stick X
                'ABS_Y': (2, 1, 'uint8'),  # Left stick Y
                'ABS_RX': (3, 1, 'uint8'),  # Right stick X
                'ABS_RY': (4, 1, 'uint8'),  # Right stick Y
                'ABS_BRAKE': (8, 1, 'uint8'),  # L2 analog trigger
                'ABS_GAS': (9, 1, 'uint8'),  # R2 analog trigger
            },
            'buttons': {
                # Byte 6 (index 5): Face buttons in upper nibble
                'BTN_X': (5, 0x10),  # Square (bit 4)
                'BTN_A': (5, 0x20),  # Cross (bit 5)
                'BTN_B': (5, 0x40),  # Circle (bit 6)
                'BTN_Y': (5, 0x80),  # Triangle (bit 7)
                # Byte 7 (index 6): Shoulder buttons and options
                'BTN_TL': (6, 0x01),  # L1 (bit 0)
                'BTN_TR': (6, 0x02),  # R1 (bit 1)
                'BTN_SELECT': (6, 0x10),  # Share/Create (bit 4)
                'BTN_START': (6, 0x20),  # Options (bit 5)
            },
            'hat': {
                'offset': 5,  # D-pad in lower nibble of byte 6 (0x0F), shares with face buttons
                'mask': 0x0F,
            },
            'inverted_buttons': False  # Normal logic: 1=pressed, 0=not pressed
        },
        # PS4/PS5 controllers - Long report format (USB enhanced mode)
        # Report format: [31, a1, left_x, left_y, right_x, right_y, left_trigger, right_trigger, seq, dpad+buttons, select+bumpers, ...]
        'playstation_long': {
            'axes': {
                'ABS_X': (2, 1, 'uint8'),  # Left stick X (byte 3)
                'ABS_Y': (3, 1, 'uint8'),  # Left stick Y (byte 4)
                'ABS_RX': (4, 1, 'uint8'),  # Right stick X (byte 5)
                'ABS_RY': (5, 1, 'uint8'),  # Right stick Y (byte 6)
                'ABS_BRAKE': (6, 1, 'uint8'),  # Left trigger (byte 7)
                'ABS_GAS': (7, 1, 'uint8'),  # Right trigger (byte 8)
            },
            'buttons': {
                # Byte 10 (index 9): dpad and face buttons
                'BTN_X': (9, 0x10),  # Square
                'BTN_A': (9, 0x20),  # Cross
                'BTN_B': (9, 0x40),  # Circle
                'BTN_Y': (9, 0x80),  # Triangle
                # Byte 11 (index 10): selection and bumpers
                'BTN_TL': (10, 0x01),  # L1
                'BTN_TR': (10, 0x02),  # R1
                'BTN_SELECT': (10, 0x10),  # Share/Create
                'BTN_START': (10, 0x20),  # Options
            },
            'hat': {
                'offset': 9,  # D-pad in lower nibble of byte 10
                'mask': 0x0F,
            },
            'inverted_buttons': False  # Normal logic: 1=pressed, 0=not pressed
        },
        # Generic fallback
        'generic': {
            'axes': {
                'ABS_X': (1, 1, 'uint8'),
                'ABS_Y': (2, 1, 'uint8'),
                'ABS_RX': (3, 1, 'uint8'),
                'ABS_RY': (4, 1, 'uint8'),
                'ABS_BRAKE': (5, 1, 'uint8'),
                'ABS_GAS': (6, 1, 'uint8'),
            },
            'buttons': {
                'BTN_A': (7, 0x01),
                'BTN_B': (7, 0x02),
                'BTN_X': (7, 0x04),
                'BTN_Y': (7, 0x08),
                'BTN_TL': (7, 0x10),
                'BTN_TR': (7, 0x20),
                'BTN_SELECT': (7, 0x40),
                'BTN_START': (7, 0x80),
            },
            'hat': {
                'offset': 8,
                'mask': 0x0F,
            }
        }
    }

    def parse_int16(data, offset):
        """Parse little-endian signed 16-bit integer"""
        value = data[offset] | (data[offset + 1] << 8)
        if value >= 32768:
            value -= 65536
        return value / 32767.0

    def parse_uint8(data, offset):
        """Parse unsigned 8-bit integer to normalized float"""
        return (data[offset] - 128) / 128.0

    def parse_uint8_trigger(data, offset):
        """Parse unsigned 8-bit trigger value"""
        return data[offset] / 255.0

    def parse_hat(data, offset, mask):
        """Parse D-pad/hat values"""
        hat_value = (data[offset] & mask) if offset < len(data) else 0
        # Standard HID hat values: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW, 8-15=center
        hat_to_xy = {
            0: (0, -1),
            1: (1, -1),
            2: (1, 0),
            3: (1, 1),
            4: (0, 1),
            5: (-1, 1),
            6: (-1, 0),
            7: (-1, -1),
        }
        # Values 8-15 all mean centered/not pressed
        return hat_to_xy.get(hat_value, (0, 0))

    def detect_parser(device_info, report_data=None):
        """Detect appropriate parser based on device info and report format"""
        vendor = device_info.get('vendor_id', 0)
        product = device_info.get('product_id', 0)

        # Microsoft Xbox controllers
        if vendor == 0x045e:
            return 'xbox'
        # Sony PlayStation controllers
        elif vendor == 0x054c:
            # Detect PS report format based on first byte or length
            if report_data is not None:
                if len(report_data) <= 10 and report_data[0] == 0x01:
                    return 'playstation_short'
                elif len(report_data) > 10 and report_data[0] == 0x31:
                    return 'playstation_long'
            # Default to short format for initial detection
            return 'playstation_short'
        else:
            return 'generic'

    def find_gamepad_device():
        """Find a suitable HID gamepad device"""
        devices = hid.enumerate()

        for dev_info in devices:
            # Filter by vendor/product ID if specified
            if vendor_id is not None and dev_info['vendor_id'] != vendor_id:
                continue
            if product_id is not None and dev_info['product_id'] != product_id:
                continue
            if serial_number is not None and dev_info['serial_number'] != serial_number:
                continue

            # Look for gamepad-like devices
            # Common gamepad usage pages: 0x01 (Generic Desktop), 0x05 (Game Controls)
            if dev_info.get('usage_page') in [0x01, 0x05]:
                return dev_info

        # If no specific match, return first device
        if devices:
            return devices[0]
        return None

    def init_device():
        """Initialize HID device"""
        nonlocal dev

        if device is not None:
            # Use specified device path
            dev_info = {'path': device}
            dev = hid.device()
            try:
                dev.open_path(device.encode() if isinstance(device, str) else device)
            except Exception as e:
                print(f'Failed to open HID device {device}: {e}')
                return None
        else:
            # Auto-detect device
            dev_info = find_gamepad_device()
            if dev_info is None:
                print('No HID gamepad device found')
                return None

            dev = hid.device()
            try:
                dev.open(dev_info['vendor_id'], dev_info['product_id'])
            except Exception as e:
                print(f'Failed to open HID device: {e}')
                return None

        # Set non-blocking mode unless blocking is requested
        if not blocking:
            dev.set_nonblocking(1)

        # Get device info
        try:
            manufacturer = dev.get_manufacturer_string()
            product = dev.get_product_string()
            serial = dev.get_serial_number_string()
            print(f'Opened HID device: {manufacturer} {product} (Serial: {serial})')
        except:
            print('Opened HID device (info unavailable)')

        parser_type = detect_parser(dev_info)
        print(f'Using parser: {parser_type}')

        return gamepad_parsers[parser_type]

    parser = init_device()
    if parser is None and dev is None:
        print('HID device initialization failed')
        return None

    def cb():
        nonlocal dev, parser, retry_pt, n_fails

        # Handle device reconnection
        if dev is None:
            if retry_pt:
                retry_pt -= 1
                # Still update output with last known states
                for i, k in enumerate(input_keys):
                    states_input[i] = states.get(k, 0)
                return
            try:
                parser = init_device()
                if dev is not None:
                    print('HID device reconnected')
                    n_fails = 0
            except Exception as e:
                n_fails += 1
                retry_pt = retry_intv
                print(f'HID device reconnect error (attempt {n_fails}): {e}')
                for i, k in enumerate(input_keys):
                    states_input[i] = states.get(k, 0)
                return

        if dev is None or parser is None:
            for i, k in enumerate(input_keys):
                states_input[i] = states.get(k, 0)
            return

        # Read all available HID reports to get the latest state (reduces latency)
        last_data = None
        num_reads = 0
        max_reads = 10  # Prevent infinite loop

        try:
            while num_reads < max_reads:
                data = dev.read(64)
                if not data:
                    break

                last_data = data
                num_reads += 1

                # In blocking mode, only read once
                if blocking:
                    break

            if last_data is None:
                # No new data, update output with current states
                for i, k in enumerate(input_keys):
                    states_input[i] = states.get(k, 0)
                return

            data = last_data
            n_fails = 0

            # Auto-detect parser format if PS controller (check parser name)
            parser_name = None
            for name, p in gamepad_parsers.items():
                if p == parser:
                    parser_name = name
                    break

            if parser_name and 'playstation' in parser_name:
                dev_info = {'vendor_id': 0x054c, 'product_id': 0}
                new_parser_type = detect_parser(dev_info, data)
                new_parser = gamepad_parsers.get(new_parser_type)
                if new_parser and new_parser != parser:
                    parser = new_parser
                    if verbose:
                        print(f'Switched to parser: {new_parser_type}')

            if verbose:
                if num_reads > 1:
                    print(f'Flushed {num_reads-1} old reports, processing latest')
                print(f'HID report: {" ".join(f"{b:02x}" for b in data)}')

            # Parse axes
            for axis_name, axis_spec in parser['axes'].items():
                if axis_name not in input_keys:
                    continue

                offset, size, dtype = axis_spec
                if offset >= len(data):
                    continue

                if dtype == 'int16' and offset + 1 < len(data):
                    value = parse_int16(data, offset)
                elif dtype == 'uint8':
                    if axis_name in ['ABS_BRAKE', 'ABS_GAS']:
                        value = parse_uint8_trigger(data, offset)
                    else:
                        value = parse_uint8(data, offset)
                else:
                    continue

                input_states[axis_name] = value

            # Parse buttons
            inverted_buttons = parser.get('inverted_buttons', False)
            for btn_name, btn_spec in parser['buttons'].items():
                if btn_name not in input_keys:
                    continue

                offset, mask = btn_spec
                if offset >= len(data):
                    continue

                if inverted_buttons:
                    # Inverted: 0 = pressed, 1 = not pressed
                    value = 0 if (data[offset] & mask) else 1
                else:
                    # Normal: 1 = pressed, 0 = not pressed
                    value = 1 if (data[offset] & mask) else 0
                input_states[btn_name] = value

            # Parse D-pad/hat (two modes: hat value or individual bits)
            if 'hat_bits' in parser:
                # D-pad as individual bits (PS5 short format)
                for key, (offset, mask) in parser['hat_bits'].items():
                    if offset < len(data):
                        value = 1 if (data[offset] & mask) else 0
                        input_states[key] = value
                # Combine bits into hat axes
                hat_up = input_states.get('ABS_HAT0Y-', 0)
                hat_down = input_states.get('ABS_HAT0Y+', 0)
                hat_left = input_states.get('ABS_HAT0X-', 0)
                hat_right = input_states.get('ABS_HAT0X+', 0)
                input_states['ABS_HAT0X'] = hat_right - hat_left
                input_states['ABS_HAT0Y'] = hat_down - hat_up
            elif 'hat' in parser:
                # D-pad as hat value (standard HID)
                hat_offset = parser['hat']['offset']
                hat_mask = parser['hat']['mask']
                if hat_offset < len(data):
                    hat_x, hat_y = parse_hat(data, hat_offset, hat_mask)
                    input_states['ABS_HAT0X'] = hat_x
                    input_states['ABS_HAT0Y'] = hat_y

            # Update states from input_states
            for k in input_keys:
                v = input_states.get(k, 0)

                # Clamp values
                v = max(-1, min(1, v))

                # Remap triggers if requested
                if remap_trigger and k in ['ABS_BRAKE', 'ABS_GAS']:
                    # Already in [0,1] range from parse_uint8_trigger
                    pass

                states[k] = v

            # Update output array after all parsing is complete
            for i, k in enumerate(input_keys):
                states_input[i] = states.get(k, 0)

            if verbose:
                print('states:', {k: f'{v:.2f}' for k, v in states.items() if abs(v) > 0.01})
                print('states_input:', states_input.tolist())

        except Exception as e:
            n_fails += 1
            if n_fails < 10:
                print(f'HID read error: {e}')
            if n_fails > 50:
                if dev:
                    dev.close()
                dev = None
            return

    return cb


if __name__ == '__main__':
    from unicon.inputs import test_cb_input
    test_cb_input(cb_input_hidapi)
